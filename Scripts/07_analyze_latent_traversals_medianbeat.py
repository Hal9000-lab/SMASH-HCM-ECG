#!/usr/bin/env python
"""
07_analyze_latent_traversals_medianbeat.py

SMASH-HCM -- Phase I
Physiological traversal analysis for the frozen normal-anchored
median-beat beta-VAE.

For each latent coordinate z_k:
- estimate empirical mean and SD using normal_train only
- start from the empirical normal_train latent centroid
- vary z_k through -3,-2,-1,0,+1,+2,+3 SD
- keep all other latent coordinates fixed
- decode the synthetic 12-lead median beat
- convert decoder output back to mV
- use physical time axis from -350 ms to +650 ms
- quantify global and lead-specific waveform excursions
- save all decoded waveforms, figures, and metrics

No HCM ECG contributes to the traversal coordinate system.

Expected inputs
---------------
Results/06_Phase1_MedianBeat_Beta_VAE/best_model.pt
Results/06_Phase1_MedianBeat_Beta_VAE/phase1_medianbeat_latents.csv

Expected preprocessing
----------------------
Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5

Outputs
-------
Results/07_Phase1_MedianBeat_LatentTraversal/
    latent_empirical_stats.csv
    latent_traversal_metrics.csv
    latent_traversal_excursion_ranking.csv
    latent_traversals_medianbeat.npz
    traversal_run_summary.json
    figures/z_00_traversal.png
    ...
    figures/z_15_traversal.png

Requirements
------------
pip install torch numpy pandas matplotlib h5py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SEED = 20261004
FS_HZ = 500.0
N_SAMPLES = 500
PRE_MS = 350.0
POST_MS = 650.0
LATENT_DIM_DEFAULT = 16
LEVELS_SD_DEFAULT = [-3, -2, -1, 0, 1, 2, 3]

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def infer_project_root() -> Path:
    root = Path(__file__).resolve().parent.parent

    if (root / "Data").exists() and (root / "Results").exists():
        return root

    cwd = Path.cwd().resolve()

    if (cwd / "Data").exists() and (cwd / "Results").exists():
        return cwd

    return root


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                c_in,
                c_out,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=min(8, c_out),
                num_channels=c_out,
            ),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        output_padding: int = 0,
        activate: bool = True,
    ):
        super().__init__()

        layers = [
            nn.ConvTranspose1d(
                c_in,
                c_out,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=output_padding,
            )
        ]

        if activate:
            layers += [
                nn.GroupNorm(
                    num_groups=min(8, c_out),
                    num_channels=c_out,
                ),
                nn.SiLU(),
            ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MedianBeatBetaVAE(nn.Module):
    """
    Exact architecture used by 06_train_phase1_medianbeat_beta_vae.py

    Encoder:
        [B,12,500]
        -> [B,32,250]
        -> [B,64,125]
        -> [B,96,62]
        -> [B,128,31]
        -> FC96
        -> mu/logvar

    Decoder:
        31 -> 62 -> 125 -> 250 -> 500
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()

        self.latent_dim = int(latent_dim)

        self.encoder = nn.Sequential(
            ConvBlock(12, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 96),
            ConvBlock(96, 128),
        )

        self.enc_channels = 128
        self.enc_length = 31
        self.enc_flat_dim = self.enc_channels * self.enc_length

        self.enc_fc = nn.Sequential(
            nn.Linear(
                self.enc_flat_dim,
                96,
            ),
            nn.SiLU(),
        )

        self.fc_mu = nn.Linear(
            96,
            self.latent_dim,
        )

        self.fc_logvar = nn.Linear(
            96,
            self.latent_dim,
        )

        self.dec_fc = nn.Sequential(
            nn.Linear(
                self.latent_dim,
                96,
            ),
            nn.SiLU(),
            nn.Linear(
                96,
                self.enc_flat_dim,
            ),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            # 31 -> 62
            DeconvBlock(
                128,
                96,
                output_padding=0,
                activate=True,
            ),
            # 62 -> 125
            DeconvBlock(
                96,
                64,
                output_padding=1,
                activate=True,
            ),
            # 125 -> 250
            DeconvBlock(
                64,
                32,
                output_padding=0,
                activate=True,
            ),
            # 250 -> 500
            DeconvBlock(
                32,
                12,
                output_padding=0,
                activate=False,
            ),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        h = self.enc_fc(h)

        return (
            self.fc_mu(h),
            self.fc_logvar(h),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(std)

        return (
            mu
            + eps * std
        )

    def decode(self, z):
        h = self.dec_fc(z)

        h = h.view(
            -1,
            self.enc_channels,
            self.enc_length,
        )

        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return (
            recon,
            mu,
            logvar,
        )


def load_scaling(
    root: Path,
    explicit_h5: Path | None,
    explicit_mean: float | None,
    explicit_std: float | None,
):
    if (
        explicit_mean is None
    ) ^ (
        explicit_std is None
    ):
        raise ValueError(
            "Provide both --scale-mean-mv and --scale-std-mv, or neither."
        )

    if (
        explicit_mean is not None
        and explicit_std is not None
    ):
        if explicit_std <= 0:
            raise ValueError(
                "--scale-std-mv must be > 0."
            )

        return (
            float(explicit_mean),
            float(explicit_std),
            "explicit CLI values",
        )

    candidates = []

    if explicit_h5 is not None:
        candidates.append(
            explicit_h5
        )

    candidates += [
        root
        / "Results"
        / "05_Phase1_MedianBeat_VAE"
        / "Phase1_MedianBeat_NormalAnchored_preprocessed.h5"
    ]

    for p in candidates:
        if not p.exists():
            continue

        try:
            with h5py.File(
                p,
                "r",
            ) as h5:

                if (
                    "preprocessing/global_mean_mv"
                    in h5
                    and
                    "preprocessing/global_std_mv"
                    in h5
                ):
                    mean_mv = float(
                        h5[
                            "preprocessing/global_mean_mv"
                        ][()]
                    )

                    std_mv = float(
                        h5[
                            "preprocessing/global_std_mv"
                        ][()]
                    )

                    if std_mv > 0:
                        return (
                            mean_mv,
                            std_mv,
                            f"preprocessed HDF5: {p}",
                        )

        except Exception:
            pass

    raise FileNotFoundError(
        "Could not determine median-beat global scaling. "
        "Provide --preprocessed-h5 or explicit --scale-mean-mv/--scale-std-mv."
    )


def compute_latent_stats(
    latent_df: pd.DataFrame,
    latent_dim: int,
):
    zcols = [
        f"z_{k:02d}"
        for k in range(
            latent_dim
        )
    ]

    missing = [
        c
        for c in zcols
        if c not in latent_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing latent columns: "
            + ", ".join(
                missing
            )
        )

    if "split_name" not in latent_df.columns:
        raise ValueError(
            "Latent CSV does not contain split_name."
        )

    normal_train = latent_df[
        latent_df[
            "split_name"
        ].astype(
            str
        )
        == "normal_train"
    ].copy()

    if len(normal_train) == 0:
        raise ValueError(
            "No normal_train records found."
        )

    Z = normal_train[
        zcols
    ].to_numpy(
        dtype=np.float64
    )

    mu = np.mean(
        Z,
        axis=0,
    )

    sd = np.std(
        Z,
        axis=0,
        ddof=1,
    )

    if (
        np.any(
            ~np.isfinite(mu)
        )
        or np.any(
            ~np.isfinite(sd)
        )
    ):
        raise ValueError(
            "Non-finite latent statistics."
        )

    if np.any(
        sd <= 0
    ):
        raise ValueError(
            "At least one latent dimension has non-positive SD."
        )

    return (
        normal_train,
        zcols,
        mu,
        sd,
    )


@torch.no_grad()
def decode_traversals(
    model,
    mu,
    sd,
    levels_sd,
    device,
):
    latent_dim = len(mu)
    n_levels = len(levels_sd)

    latent_coordinates = np.empty(
        (
            latent_dim,
            n_levels,
            latent_dim,
        ),
        dtype=np.float32,
    )

    decoded_scaled = np.empty(
        (
            latent_dim,
            n_levels,
            12,
            N_SAMPLES,
        ),
        dtype=np.float32,
    )

    model.eval()

    for k in range(
        latent_dim
    ):
        Z = np.tile(
            mu[None, :],
            (
                n_levels,
                1,
            ),
        ).astype(
            np.float32
        )

        for j, level in enumerate(
            levels_sd
        ):
            Z[
                j,
                k,
            ] = (
                mu[k]
                + float(level)
                * sd[k]
            )

        z_tensor = torch.from_numpy(
            Z
        ).float().to(
            device
        )

        x = model.decode(
            z_tensor
        )

        if tuple(
            x.shape[1:]
        ) != (
            12,
            N_SAMPLES,
        ):
            raise RuntimeError(
                f"Unexpected decoder output shape: {tuple(x.shape)}"
            )

        latent_coordinates[
            k
        ] = Z

        decoded_scaled[
            k
        ] = (
            x.detach()
            .cpu()
            .numpy()
        )

    return (
        latent_coordinates,
        decoded_scaled,
    )


def make_time_axis_ms():
    dt_ms = (
        1000.0
        / FS_HZ
    )

    return (
        np.arange(
            N_SAMPLES,
            dtype=np.float64,
        )
        * dt_ms
        - PRE_MS
    )


def make_metrics(
    decoded_mv,
    latent_coordinates,
    mu,
    sd,
    levels_sd,
):
    levels = np.asarray(
        levels_sd,
        dtype=float,
    )

    zero_idx = np.where(
        np.isclose(
            levels,
            0.0,
        )
    )[0]

    if len(zero_idx) != 1:
        raise ValueError(
            "Traversal levels must contain exactly one 0."
        )

    i0 = int(
        zero_idx[0]
    )

    rows = []

    # Broad waveform regions for morphology interpretation.
    # These are descriptive windows, not clinical interval measurements.
    regions = {
        "pre_qrs": (-300.0, -60.0),
        "qrs": (-60.0, 80.0),
        "st_t": (80.0, 500.0),
    }

    t_ms = make_time_axis_ms()

    for k in range(
        decoded_mv.shape[0]
    ):
        center = decoded_mv[
            k,
            i0,
        ]

        for j, level in enumerate(
            levels
        ):
            x = decoded_mv[
                k,
                j,
            ]

            delta = (
                x
                - center
            )

            rms_delta_by_lead = np.sqrt(
                np.mean(
                    delta ** 2,
                    axis=1,
                )
            )

            p2p_by_lead = np.ptp(
                x,
                axis=1,
            )

            sensitive_idx = int(
                np.argmax(
                    rms_delta_by_lead
                )
            )

            row = {
                "latent":
                    f"z_{k:02d}",
                "latent_index":
                    k,
                "level_sd":
                    float(level),
                "latent_value":
                    float(
                        latent_coordinates[
                            k,
                            j,
                            k,
                        ]
                    ),
                "normal_train_latent_mean":
                    float(
                        mu[k]
                    ),
                "normal_train_latent_sd":
                    float(
                        sd[k]
                    ),
                "global_rms_change_vs_center_mv":
                    float(
                        np.sqrt(
                            np.mean(
                                delta ** 2
                            )
                        )
                    ),
                "global_maxabs_change_vs_center_mv":
                    float(
                        np.max(
                            np.abs(
                                delta
                            )
                        )
                    ),
                "most_sensitive_lead_by_rms_change":
                    LEAD_NAMES[
                        sensitive_idx
                    ],
                "most_sensitive_lead_rms_change_mv":
                    float(
                        rms_delta_by_lead[
                            sensitive_idx
                        ]
                    ),
            }

            for region_name, (
                t0,
                t1,
            ) in regions.items():

                mask = (
                    (t_ms >= t0)
                    & (t_ms < t1)
                )

                if np.any(mask):
                    region_delta = delta[
                        :,
                        mask,
                    ]

                    row[
                        f"{region_name}_global_rms_change_mv"
                    ] = float(
                        np.sqrt(
                            np.mean(
                                region_delta ** 2
                            )
                        )
                    )

            for lead_i, lead in enumerate(
                LEAD_NAMES
            ):
                key = (
                    lead.replace(
                        "a",
                        "A",
                    )
                )

                row[
                    f"{key}_peak_to_peak_mv"
                ] = float(
                    p2p_by_lead[
                        lead_i
                    ]
                )

                row[
                    f"{key}_rms_change_vs_center_mv"
                ] = float(
                    rms_delta_by_lead[
                        lead_i
                    ]
                )

            rows.append(row)

    return pd.DataFrame(
        rows
    )


def make_empirical_stats_table(
    latent_df,
    normal_train,
    zcols,
    mu,
    sd,
):
    rows = []

    for k, col in enumerate(
        zcols
    ):
        row = {
            "latent":
                col,
            "latent_index":
                k,
            "normal_train_n":
                int(
                    len(
                        normal_train
                    )
                ),
            "posterior_mean_empirical_mean":
                float(
                    mu[k]
                ),
            "posterior_mean_empirical_sd":
                float(
                    sd[k]
                ),
            "posterior_mean_empirical_variance":
                float(
                    sd[k] ** 2
                ),
        }

        logvar_col = (
            f"logvar_{k:02d}"
        )

        if (
            logvar_col
            in latent_df.columns
        ):
            lv = normal_train[
                logvar_col
            ].to_numpy(
                dtype=np.float64
            )

            pv = np.exp(
                lv
            )

            z = normal_train[
                col
            ].to_numpy(
                dtype=np.float64
            )

            kl = 0.5 * (
                z ** 2
                + pv
                - lv
                - 1.0
            )

            row[
                "mean_logvar"
            ] = float(
                np.mean(lv)
            )

            row[
                "mean_posterior_variance"
            ] = float(
                np.mean(pv)
            )

            row[
                "mean_KL_per_dimension"
            ] = float(
                np.mean(kl)
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def plot_latent_traversal(
    decoded_mv_k,
    levels_sd,
    latent_name,
    mu_k,
    sd_k,
    out_path,
):
    t_ms = make_time_axis_ms()

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(15, 11),
        sharex=True,
    )

    axes = axes.ravel()

    for lead_idx, ax in enumerate(
        axes
    ):
        for j, level in enumerate(
            levels_sd
        ):
            lw = (
                1.8
                if np.isclose(
                    level,
                    0.0,
                )
                else 1.0
            )

            alpha = (
                1.0
                if np.isclose(
                    level,
                    0.0,
                )
                else 0.85
            )

            ax.plot(
                t_ms,
                decoded_mv_k[
                    j,
                    lead_idx,
                ],
                linewidth=lw,
                alpha=alpha,
                label=f"{level:+g} SD",
            )

        ax.axvline(
            0.0,
            linewidth=0.7,
            alpha=0.45,
        )

        ax.axhline(
            0.0,
            linewidth=0.6,
            alpha=0.45,
        )

        ax.set_title(
            LEAD_NAMES[
                lead_idx
            ]
        )

        ax.set_ylabel(
            "mV"
        )

        ax.grid(
            alpha=0.15
        )

    for ax in axes[-3:]:
        ax.set_xlabel(
            "Time from R/QRS fiducial (ms)"
        )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(
            levels_sd
        ),
        frameon=False,
    )

    fig.suptitle(
        f"{latent_name} traversal around empirical Normal-train centroid\n"
        f"mean={mu_k:.3f}, SD={sd_k:.3f}",
        y=0.995,
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.95,
        )
    )

    fig.savefig(
        out_path,
        dpi=180,
    )

    plt.close(
        fig
    )


def main():
    root = infer_project_root()

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            root
            / "Results"
            / "06_Phase1_MedianBeat_Beta_VAE"
            / "best_model.pt"
        ),
    )

    ap.add_argument(
        "--latents",
        type=Path,
        default=(
            root
            / "Results"
            / "06_Phase1_MedianBeat_Beta_VAE"
            / "phase1_medianbeat_latents.csv"
        ),
    )

    ap.add_argument(
        "--results-dir",
        type=Path,
        default=(
            root
            / "Results"
            / "07_Phase1_MedianBeat_LatentTraversal"
        ),
    )

    ap.add_argument(
        "--preprocessed-h5",
        type=Path,
        default=(
            root
            / "Results"
            / "05_Phase1_MedianBeat_VAE"
            / "Phase1_MedianBeat_NormalAnchored_preprocessed.h5"
        ),
    )

    ap.add_argument(
        "--scale-mean-mv",
        type=float,
        default=None,
    )

    ap.add_argument(
        "--scale-std-mv",
        type=float,
        default=None,
    )

    ap.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=LEVELS_SD_DEFAULT,
    )

    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if not args.latents.exists():
        raise FileNotFoundError(
            f"Latent CSV not found: {args.latents}"
        )

    levels_sd = [
        float(x)
        for x in args.levels
    ]

    if len(
        [
            x
            for x in levels_sd
            if np.isclose(
                x,
                0.0,
            )
        ]
    ) != 1:
        raise ValueError(
            "Traversal levels must contain exactly one zero."
        )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig_dir = (
        args.results_dir
        / "figures"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Project root : {root}"
    )

    print(
        f"Device       : {device}"
    )

    print(
        f"Checkpoint   : {args.checkpoint}"
    )

    print(
        f"Latent CSV   : {args.latents}"
    )

    latent_df = pd.read_csv(
        args.latents
    )

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    latent_dim = int(
        ckpt.get(
            "latent_dim",
            LATENT_DIM_DEFAULT,
        )
    )

    (
        normal_train,
        zcols,
        mu,
        sd,
    ) = compute_latent_stats(
        latent_df,
        latent_dim,
    )

    print(
        f"Normal-train records: {len(normal_train):,}"
    )

    print(
        f"Latent dimensions   : {latent_dim}"
    )

    model = MedianBeatBetaVAE(
        latent_dim=latent_dim
    ).to(
        device
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ]
    )

    model.eval()

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Model parameters    : {n_params:,}"
    )

    (
        scale_mean_mv,
        scale_std_mv,
        scale_source,
    ) = load_scaling(
        root,
        args.preprocessed_h5,
        args.scale_mean_mv,
        args.scale_std_mv,
    )

    print(
        f"Scaling source      : {scale_source}"
    )

    print(
        f"Global mean         : {scale_mean_mv:.12g} mV"
    )

    print(
        f"Global std          : {scale_std_mv:.12g} mV"
    )

    (
        latent_coordinates,
        decoded_scaled,
    ) = decode_traversals(
        model,
        mu,
        sd,
        levels_sd,
        device,
    )

    decoded_mv = (
        decoded_scaled
        * scale_std_mv
        + scale_mean_mv
    ).astype(
        np.float32
    )

    empirical_stats = (
        make_empirical_stats_table(
            latent_df,
            normal_train,
            zcols,
            mu,
            sd,
        )
    )

    empirical_stats_path = (
        args.results_dir
        / "latent_empirical_stats.csv"
    )

    empirical_stats.to_csv(
        empirical_stats_path,
        index=False,
    )

    metrics = make_metrics(
        decoded_mv,
        latent_coordinates,
        mu,
        sd,
        levels_sd,
    )

    metrics_path = (
        args.results_dir
        / "latent_traversal_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    npz_path = (
        args.results_dir
        / "latent_traversals_medianbeat.npz"
    )

    np.savez_compressed(
        npz_path,
        decoded_scaled=decoded_scaled,
        decoded_mv=decoded_mv,
        latent_coordinates=latent_coordinates,
        latent_normal_train_mean=mu.astype(
            np.float32
        ),
        latent_normal_train_sd=sd.astype(
            np.float32
        ),
        levels_sd=np.asarray(
            levels_sd,
            dtype=np.float32,
        ),
        lead_names=np.asarray(
            LEAD_NAMES
        ),
        sampling_rate_hz=np.asarray(
            [FS_HZ],
            dtype=np.float32,
        ),
        pre_ms=np.asarray(
            [PRE_MS],
            dtype=np.float32,
        ),
        post_ms=np.asarray(
            [POST_MS],
            dtype=np.float32,
        ),
        time_axis_ms=make_time_axis_ms().astype(
            np.float32
        ),
        scale_mean_mv=np.asarray(
            [scale_mean_mv],
            dtype=np.float64,
        ),
        scale_std_mv=np.asarray(
            [scale_std_mv],
            dtype=np.float64,
        ),
    )

    print(
        "Generating traversal figures..."
    )

    for k in range(
        latent_dim
    ):
        out = (
            fig_dir
            / f"z_{k:02d}_traversal.png"
        )

        plot_latent_traversal(
            decoded_mv[
                k
            ],
            levels_sd,
            f"z_{k:02d}",
            float(
                mu[k]
            ),
            float(
                sd[k]
            ),
            out,
        )

        print(
            f"  saved {out.name}"
        )

    levels_arr = np.asarray(
        levels_sd,
        dtype=float,
    )

    minus3 = np.where(
        np.isclose(
            levels_arr,
            -3.0,
        )
    )[0]

    plus3 = np.where(
        np.isclose(
            levels_arr,
            3.0,
        )
    )[0]

    excursion_path = None

    if (
        len(minus3) == 1
        and len(plus3) == 1
    ):
        im = int(
            minus3[0]
        )

        ip = int(
            plus3[0]
        )

        rows = []

        t_ms = make_time_axis_ms()

        regions = {
            "pre_qrs": (-300.0, -60.0),
            "qrs": (-60.0, 80.0),
            "st_t": (80.0, 500.0),
        }

        for k in range(
            latent_dim
        ):
            delta = (
                decoded_mv[
                    k,
                    ip,
                ]
                - decoded_mv[
                    k,
                    im,
                ]
            )

            lead_rms = np.sqrt(
                np.mean(
                    delta ** 2,
                    axis=1,
                )
            )

            idx = int(
                np.argmax(
                    lead_rms
                )
            )

            row = {
                "latent":
                    f"z_{k:02d}",
                "global_rms_minus3_to_plus3_mv":
                    float(
                        np.sqrt(
                            np.mean(
                                delta ** 2
                            )
                        )
                    ),
                "global_maxabs_minus3_to_plus3_mv":
                    float(
                        np.max(
                            np.abs(
                                delta
                            )
                        )
                    ),
                "most_sensitive_lead":
                    LEAD_NAMES[
                        idx
                    ],
                "most_sensitive_lead_rms_minus3_to_plus3_mv":
                    float(
                        lead_rms[
                            idx
                        ]
                    ),
            }

            for region_name, (
                t0,
                t1,
            ) in regions.items():

                mask = (
                    (t_ms >= t0)
                    & (t_ms < t1)
                )

                region_delta = delta[
                    :,
                    mask,
                ]

                row[
                    f"{region_name}_rms_minus3_to_plus3_mv"
                ] = float(
                    np.sqrt(
                        np.mean(
                            region_delta ** 2
                        )
                    )
                )

            rows.append(
                row
            )

        excursion_df = pd.DataFrame(
            rows
        ).sort_values(
            "global_rms_minus3_to_plus3_mv",
            ascending=False,
        )

        excursion_path = (
            args.results_dir
            / "latent_traversal_excursion_ranking.csv"
        )

        excursion_df.to_csv(
            excursion_path,
            index=False,
        )

    summary = {
        "design":
            "normal-anchored median-beat latent traversal",
        "checkpoint":
            str(
                args.checkpoint
            ),
        "latent_csv":
            str(
                args.latents
            ),
        "normal_train_n_used_for_latent_statistics":
            int(
                len(
                    normal_train
                )
            ),
        "latent_dim":
            int(
                latent_dim
            ),
        "levels_sd":
            levels_sd,
        "generated_median_beats":
            int(
                latent_dim
                * len(
                    levels_sd
                )
            ),
        "sampling_rate_hz":
            FS_HZ,
        "samples_per_lead":
            N_SAMPLES,
        "window_seconds":
            float(
                N_SAMPLES
                / FS_HZ
            ),
        "window_ms":
            {
                "pre":
                    PRE_MS,
                "post":
                    POST_MS,
            },
        "lead_names":
            LEAD_NAMES,
        "reference_latent_vector":
            (
                "empirical posterior-mean centroid "
                "of normal_train"
            ),
        "traversal_scale":
            (
                "empirical normal_train posterior-mean "
                "SD per latent dimension"
            ),
        "hcm_used_for_latent_statistics":
            False,
        "hcm_used_for_traversal":
            False,
        "decoder_output_unscaled_to_mv":
            True,
        "global_scaling_mean_mv":
            float(
                scale_mean_mv
            ),
        "global_scaling_std_mv":
            float(
                scale_std_mv
            ),
        "scaling_source":
            scale_source,
        "model_parameters":
            int(
                n_params
            ),
        "outputs":
            {
                "empirical_stats_csv":
                    str(
                        empirical_stats_path
                    ),
                "metrics_csv":
                    str(
                        metrics_path
                    ),
                "npz":
                    str(
                        npz_path
                    ),
                "figures_dir":
                    str(
                        fig_dir
                    ),
                "excursion_ranking_csv":
                    (
                        str(
                            excursion_path
                        )
                        if excursion_path
                        else None
                    ),
            },
    }

    summary_path = (
        args.results_dir
        / "traversal_run_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Median-beat latent traversal analysis complete."
    )

    print(
        f"Empirical latent statistics : "
        f"{empirical_stats_path}"
    )

    print(
        f"Traversal metrics           : "
        f"{metrics_path}"
    )

    if excursion_path is not None:
        print(
            f"Excursion ranking           : "
            f"{excursion_path}"
        )

    print(
        f"All decoded traversals      : "
        f"{npz_path}"
    )

    print(
        f"Figures                     : "
        f"{fig_dir}"
    )

    print(
        f"Run summary                 : "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()
