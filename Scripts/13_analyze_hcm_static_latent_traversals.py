#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
13_analyze_hcm_static_latent_traversals.py

SMASH-HCM -- Model 2 / Step 13
Physiological inspection of the frozen HCM-static median-beat beta-VAE.

SCIENTIFIC QUESTION
-------------------
Model 1 defined deviations from normal physiology.
Model 2 defines the principal continuous axes of morphology WITHIN HCM.

This script interprets the HCM-specific latent coordinates by:
    1. estimating latent mean and SD from HCM-train ONLY;
    2. starting from the empirical HCM-train latent centroid;
    3. varying one HCM latent coordinate at a time through:
           -3, -2, -1, 0, +1, +2, +3 SD
    4. keeping all other HCM latent coordinates fixed at the HCM-train centroid;
    5. decoding synthetic 12-lead median beats;
    6. converting the decoder output back to physical mV;
    7. quantifying global, lead-specific, and temporal-region changes.

No Normal ECG contributes to the traversal coordinate system.
No HCM validation/test ECG contributes to the traversal coordinate system.

Primary outputs
---------------
Results/13_HCM_Static_LatentTraversal/
├── latent_empirical_stats.csv
├── latent_traversal_metrics.csv
├── latent_traversal_excursion_ranking.csv
├── latent_traversals_hcm_static.npz
├── traversal_run_summary.json
├── hcm_train_centroid_beat.csv
└── figures/
    ├── z_00_traversal.png
    ├── ...
    ├── z_07_traversal.png
    ├── latent_excursion_global_ranking.png
    └── latent_excursion_region_ranking.png

Run
---
python Scripts\\13_analyze_hcm_static_latent_traversals.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


SEED = 20261004
FS_HZ = 500.0
N_SAMPLES = 500
LATENT_DIM_DEFAULT = 8
LEVELS_SD_DEFAULT = [-3, -2, -1, 0, 1, 2, 3]

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Same timing convention used throughout the median-beat pipeline.
BEAT_START_MS = -350.0

FIXED_REGIONS_MS = {
    "preQRS": (-300.0, -60.0),
    "QRS": (-60.0, 80.0),
    "STT": (80.0, 500.0),
}


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze HCM-static beta-VAE latent traversals."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("Results/12_HCM_Static_Beta_VAE/best_model.pt"),
    )
    p.add_argument(
        "--latents",
        type=Path,
        default=Path("Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv"),
    )
    p.add_argument(
        "--h5",
        type=Path,
        default=Path("Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5"),
    )
    p.add_argument(
        "--model-summary",
        type=Path,
        default=Path("Results/12_HCM_Static_Beta_VAE/model_summary.json"),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Results/13_HCM_Static_LatentTraversal"),
    )
    p.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=LEVELS_SD_DEFAULT,
        help="Traversal levels in empirical HCM-train SD units.",
    )
    p.add_argument("--fs", type=float, default=FS_HZ)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


# =====================================================================
# Exact Model-2 architecture from script 12
# =====================================================================

class HCMStaticBetaVAE(nn.Module):
    """
    Exact HCM-static architecture from Step 12.

    Encoder:
        [12,500]
        12 -> 16 -> 32 -> 48 -> 64
        500 -> 250 -> 125 -> 62 -> 31
        flatten(64*31) -> FC48 -> mu/logvar

    Decoder:
        z -> FC48 -> 64*31
        64 -> 48 -> 32 -> 16 -> 12
        31 -> 62 -> 125 -> 250 -> 500
    """

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(12, 16, 4, 2, 1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),

            nn.Conv1d(16, 32, 4, 2, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.Conv1d(32, 48, 4, 2, 1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),

            nn.Conv1d(48, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 31, 48),
            nn.SiLU(),
        )

        self.fc_mu = nn.Linear(48, latent_dim)
        self.fc_logvar = nn.Linear(48, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 64 * 31),
            nn.SiLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(
                64, 48, 4, 2, 1, output_padding=0
            ),  # 31 -> 62
            nn.GroupNorm(8, 48),
            nn.SiLU(),

            nn.ConvTranspose1d(
                48, 32, 4, 2, 1, output_padding=1
            ),  # 62 -> 125
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.ConvTranspose1d(
                32, 16, 4, 2, 1, output_padding=0
            ),  # 125 -> 250
            nn.GroupNorm(4, 16),
            nn.SiLU(),

            nn.ConvTranspose1d(
                16, 12, 4, 2, 1, output_padding=0
            ),  # 250 -> 500
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(z.shape[0], 64, 31)
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(mu), mu, logvar


# =====================================================================
# Input loading / validation
# =====================================================================

def latent_columns(df: pd.DataFrame, latent_dim: int) -> List[str]:
    expected = [f"z_{k:02d}" for k in range(latent_dim)]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing latent columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return expected


def identify_hcm_train(df: pd.DataFrame) -> np.ndarray:
    if "split_code" in df.columns:
        split = pd.to_numeric(df["split_code"], errors="coerce").to_numpy()
        mask = split == 0
        if np.sum(mask):
            return mask

    if "split_name" in df.columns:
        txt = df["split_name"].astype(str).str.lower()
        mask = txt.eq("hcm_train").to_numpy()
        if np.sum(mask):
            return mask

    raise RuntimeError(
        "Could not identify HCM-train records in latent CSV. "
        "Expected split_code==0 or split_name=='hcm_train'."
    )


def load_scaling(
    h5_path: Path,
    model_summary_path: Path,
) -> Tuple[float, float, str]:
    # Preferred source: prepared Model-2 HDF5 attributes.
    if h5_path.exists():
        with h5py.File(h5_path, "r") as h5:
            mean = h5.attrs.get("hcm_train_scaling_mean_mv", np.nan)
            std = h5.attrs.get("hcm_train_scaling_std_mv", np.nan)

        try:
            mean = float(mean)
            std = float(std)
            if np.isfinite(mean) and np.isfinite(std) and std > 0:
                return mean, std, f"HDF5 attributes: {h5_path}"
        except Exception:
            pass

    # Fallback: model summary JSON.
    if model_summary_path.exists():
        obj = json.loads(model_summary_path.read_text(encoding="utf-8"))
        mean = float(obj["hcm_train_scaling_mean_mv"])
        std = float(obj["hcm_train_scaling_std_mv"])
        if np.isfinite(mean) and np.isfinite(std) and std > 0:
            return mean, std, f"model_summary.json: {model_summary_path}"

    raise RuntimeError(
        "Could not recover HCM-train global scaling mean/std."
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[HCMStaticBetaVAE, Dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(ckpt, dict):
        raise TypeError("Expected checkpoint dictionary from Step 12.")

    latent_dim = int(ckpt.get("latent_dim", LATENT_DIM_DEFAULT))
    architecture = str(
        ckpt.get("architecture", "hcm_static_compact_v1")
    )

    if architecture != "hcm_static_compact_v1":
        raise RuntimeError(
            f"Unexpected architecture '{architecture}'. "
            "This Step-13 script expects hcm_static_compact_v1."
        )

    model = HCMStaticBetaVAE(latent_dim=latent_dim).to(device)
    nparams = sum(p.numel() for p in model.parameters())

    if latent_dim == 8 and nparams != 236988:
        raise RuntimeError(
            f"Internal architecture mismatch: {nparams:,} parameters "
            "instead of expected 236,988."
        )

    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint has no model_state_dict.")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, ckpt


# =====================================================================
# Latent statistics and traversals
# =====================================================================

def compute_hcm_train_latent_stats(
    latent_df: pd.DataFrame,
    latent_dim: int,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    train_mask = identify_hcm_train(latent_df)
    zcols = latent_columns(latent_df, latent_dim)

    Z = latent_df.loc[train_mask, zcols].to_numpy(dtype=np.float64)

    if len(Z) == 0:
        raise RuntimeError("No HCM-train latent rows found.")

    mu = np.mean(Z, axis=0)
    sd = np.std(Z, axis=0, ddof=1)

    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sd)):
        raise ValueError("Non-finite HCM-train latent statistics.")
    if np.any(sd <= 0):
        raise ValueError("At least one HCM latent has non-positive SD.")

    return train_mask, zcols, mu, sd


@torch.inference_mode()
def decode_traversals(
    model: HCMStaticBetaVAE,
    mu: np.ndarray,
    sd: np.ndarray,
    levels_sd: Sequence[float],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    latent_dim = len(mu)
    levels = np.asarray(levels_sd, dtype=float)

    latent_coordinates = np.empty(
        (latent_dim, len(levels), latent_dim),
        dtype=np.float32,
    )
    decoded_scaled = np.empty(
        (latent_dim, len(levels), 12, N_SAMPLES),
        dtype=np.float32,
    )

    for k in range(latent_dim):
        Z = np.tile(mu[None, :], (len(levels), 1)).astype(np.float32)

        for j, level in enumerate(levels):
            Z[j, k] = float(mu[k] + level * sd[k])

        z_tensor = torch.from_numpy(Z).to(device)
        x = model.decode(z_tensor)

        if tuple(x.shape[1:]) != (12, N_SAMPLES):
            raise RuntimeError(
                f"Unexpected decoder output shape {tuple(x.shape)}."
            )

        latent_coordinates[k] = Z
        decoded_scaled[k] = x.detach().cpu().numpy()

    return latent_coordinates, decoded_scaled


# =====================================================================
# Traversal metrics
# =====================================================================

def time_axis_ms(fs: float) -> np.ndarray:
    return BEAT_START_MS + np.arange(N_SAMPLES) * 1000.0 / fs


def region_indices(t_ms: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for name, (lo, hi) in FIXED_REGIONS_MS.items():
        out[name] = np.where((t_ms >= lo) & (t_ms < hi))[0]
    return out


def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def make_empirical_stats_table(
    latent_df: pd.DataFrame,
    train_mask: np.ndarray,
    zcols: Sequence[str],
    mu: np.ndarray,
    sd: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for k, col in enumerate(zcols):
        row = {
            "latent": col,
            "latent_index": k,
            "hcm_train_n": int(np.sum(train_mask)),
            "posterior_mean_empirical_mean": float(mu[k]),
            "posterior_mean_empirical_sd": float(sd[k]),
            "posterior_mean_empirical_variance": float(sd[k] ** 2),
        }

        lv_col = f"logvar_{k:02d}"
        if lv_col in latent_df.columns:
            lv = pd.to_numeric(
                latent_df.loc[train_mask, lv_col],
                errors="coerce",
            ).to_numpy(dtype=float)
            post_var = np.exp(lv)

            row["mean_logvar"] = float(np.nanmean(lv))
            row["mean_posterior_variance"] = float(
                np.nanmean(post_var)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def make_metrics(
    decoded_mv: np.ndarray,
    latent_coordinates: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    levels_sd: Sequence[float],
    fs: float,
) -> pd.DataFrame:
    levels = np.asarray(levels_sd, dtype=float)
    zero_idx = np.where(np.isclose(levels, 0.0))[0]

    if len(zero_idx) != 1:
        raise ValueError(
            "Traversal levels must contain exactly one 0-SD level."
        )

    i0 = int(zero_idx[0])
    t_ms = time_axis_ms(fs)
    reg = region_indices(t_ms)

    rows = []

    for k in range(decoded_mv.shape[0]):
        center = decoded_mv[k, i0]

        for j, level in enumerate(levels):
            x = decoded_mv[k, j]
            delta = x - center

            lead_rms = rms(delta, axis=1)
            p2p = np.ptp(x, axis=1)
            lead_idx = int(np.argmax(lead_rms))

            row = {
                "latent": f"z_{k:02d}",
                "latent_index": k,
                "level_sd": float(level),
                "latent_value": float(latent_coordinates[k, j, k]),
                "hcm_train_latent_mean": float(mu[k]),
                "hcm_train_latent_sd": float(sd[k]),
                "global_rms_change_vs_center_mv": float(rms(delta)),
                "global_maxabs_change_vs_center_mv": float(
                    np.max(np.abs(delta))
                ),
                "most_sensitive_lead": LEAD_NAMES[lead_idx],
                "most_sensitive_lead_rms_change_mv": float(
                    lead_rms[lead_idx]
                ),
            }

            # Fixed temporal regions.
            region_values = {}
            for region, idx in reg.items():
                val = float(rms(delta[:, idx]))
                region_values[region] = val
                row[f"{region}_rms_change_vs_center_mv"] = val

            dominant_region = max(
                region_values,
                key=region_values.get,
            )
            row["dominant_temporal_region"] = dominant_region

            # Lead-specific morphology.
            for li, lead in enumerate(LEAD_NAMES):
                row[f"{lead}_peak_to_peak_mv"] = float(p2p[li])
                row[f"{lead}_rms_change_vs_center_mv"] = float(
                    lead_rms[li]
                )

                for region, idx in reg.items():
                    row[
                        f"{region}_{lead}_rms_change_vs_center_mv"
                    ] = float(rms(delta[li, idx]))

            rows.append(row)

    return pd.DataFrame(rows)


def make_excursion_ranking(
    decoded_mv: np.ndarray,
    levels_sd: Sequence[float],
    fs: float,
) -> pd.DataFrame:
    levels = np.asarray(levels_sd, dtype=float)

    minus3 = np.where(np.isclose(levels, -3.0))[0]
    plus3 = np.where(np.isclose(levels, 3.0))[0]

    if len(minus3) != 1 or len(plus3) != 1:
        raise ValueError(
            "Excursion ranking requires exactly one -3 and one +3 SD level."
        )

    im = int(minus3[0])
    ip = int(plus3[0])

    t_ms = time_axis_ms(fs)
    reg = region_indices(t_ms)

    rows = []

    for k in range(decoded_mv.shape[0]):
        delta = decoded_mv[k, ip] - decoded_mv[k, im]
        lead_rms = rms(delta, axis=1)
        lead_idx = int(np.argmax(lead_rms))

        row = {
            "latent": f"z_{k:02d}",
            "latent_index": k,
            "global_rms_minus3_to_plus3_mv": float(rms(delta)),
            "global_maxabs_minus3_to_plus3_mv": float(
                np.max(np.abs(delta))
            ),
            "most_sensitive_lead": LEAD_NAMES[lead_idx],
            "most_sensitive_lead_rms_minus3_to_plus3_mv": float(
                lead_rms[lead_idx]
            ),
        }

        region_vals = {}
        for region, idx in reg.items():
            v = float(rms(delta[:, idx]))
            region_vals[region] = v
            row[f"{region}_rms_minus3_to_plus3_mv"] = v

        row["dominant_temporal_region"] = max(
            region_vals,
            key=region_vals.get,
        )

        # Region fractions of squared excursion energy.
        energies = {
            region: float(np.mean(delta[:, idx] ** 2))
            for region, idx in reg.items()
        }
        etot = sum(energies.values()) + 1e-12
        for region, e in energies.items():
            row[f"{region}_energy_fraction"] = e / etot

        for li, lead in enumerate(LEAD_NAMES):
            row[f"{lead}_rms_minus3_to_plus3_mv"] = float(
                lead_rms[li]
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            "global_rms_minus3_to_plus3_mv",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =====================================================================
# Figures
# =====================================================================

def plot_latent_traversal(
    decoded_mv_k: np.ndarray,
    levels_sd: Sequence[float],
    latent_name: str,
    mu_k: float,
    sd_k: float,
    out_path: Path,
    fs: float,
):
    t_ms = time_axis_ms(fs)
    levels = np.asarray(levels_sd, dtype=float)

    fig, axes = plt.subplots(
        4, 3,
        figsize=(15, 11),
        sharex=True,
    )
    axes = axes.ravel()

    for li, ax in enumerate(axes):
        for j, level in enumerate(levels):
            linewidth = 2.0 if np.isclose(level, 0.0) else 1.0
            alpha = 1.0 if np.isclose(level, 0.0) else 0.85

            ax.plot(
                t_ms,
                decoded_mv_k[j, li],
                linewidth=linewidth,
                alpha=alpha,
                label=f"{level:+g} SD",
            )

        ax.axvline(0, linewidth=0.8, linestyle=":")
        ax.axvline(-60, linewidth=0.7, linestyle="--")
        ax.axvline(80, linewidth=0.7, linestyle="--")
        ax.set_title(LEAD_NAMES[li])
        ax.grid(alpha=0.15)

    for ax in axes[-3:]:
        ax.set_xlabel("Time from R reference [ms]")

    for row in range(4):
        axes[row * 3].set_ylabel("Amplitude [mV]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(levels),
        frameon=False,
    )

    fig.suptitle(
        f"HCM-static traversal: {latent_name} | "
        f"HCM-train mean={mu_k:.3f}, SD={sd_k:.3f}",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_global_ranking(
    ranking: pd.DataFrame,
    out_path: Path,
):
    r = ranking.sort_values(
        "global_rms_minus3_to_plus3_mv",
        ascending=False,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        r["latent"],
        r["global_rms_minus3_to_plus3_mv"],
    )
    ax.set_xlabel("HCM latent coordinate")
    ax.set_ylabel("RMS change, -3 to +3 SD [mV]")
    ax.set_title("HCM-static latent morphology excursion")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_region_ranking(
    ranking: pd.DataFrame,
    out_path: Path,
):
    r = ranking.sort_values(
        "global_rms_minus3_to_plus3_mv",
        ascending=False,
    )

    x = np.arange(len(r))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - width,
        r["preQRS_rms_minus3_to_plus3_mv"],
        width,
        label="pre-QRS",
    )
    ax.bar(
        x,
        r["QRS_rms_minus3_to_plus3_mv"],
        width,
        label="QRS",
    )
    ax.bar(
        x + width,
        r["STT_rms_minus3_to_plus3_mv"],
        width,
        label="ST-T",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(r["latent"])
    ax.set_xlabel("HCM latent coordinate")
    ax.set_ylabel("RMS change, -3 to +3 SD [mV]")
    ax.set_title("Temporal localization of HCM latent excursions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_centroid_csv(
    centroid_mv: np.ndarray,
    fs: float,
    out_path: Path,
):
    t_ms = time_axis_ms(fs)
    d = {"time_ms": t_ms}
    for li, lead in enumerate(LEAD_NAMES):
        d[lead] = centroid_mv[li]
    pd.DataFrame(d).to_csv(out_path, index=False)


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    levels_sd = [float(x) for x in args.levels]

    if len([x for x in levels_sd if np.isclose(x, 0.0)]) != 1:
        raise ValueError(
            "Traversal levels must contain exactly one zero."
        )

    if not (
        any(np.isclose(levels_sd, -3.0))
        and any(np.isclose(levels_sd, 3.0))
    ):
        raise ValueError(
            "Default excursion ranking requires -3 and +3 SD levels."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 78)
    print("13 - Analyze HCM-static latent traversals")
    print("=" * 78)
    print("Device       :", device)
    print("Checkpoint   :", args.checkpoint)
    print("Latent CSV   :", args.latents)
    print("Prepared HDF5:", args.h5)
    print("Results      :", args.results_dir)

    latent_df = pd.read_csv(args.latents)
    model, ckpt = load_model(args.checkpoint, device)

    latent_dim = int(ckpt.get("latent_dim", LATENT_DIM_DEFAULT))
    nparams = sum(p.numel() for p in model.parameters())

    train_mask, zcols, mu, sd = compute_hcm_train_latent_stats(
        latent_df,
        latent_dim,
    )

    n_hcm_train = int(np.sum(train_mask))

    print(f"HCM-train records : {n_hcm_train}")
    print(f"Latent dimensions : {latent_dim}")
    print(f"Model parameters  : {nparams:,}")

    print("[HCM-train latent statistics]")
    for k in range(latent_dim):
        print(
            f"  {zcols[k]}: mean={mu[k]:+.6f}, "
            f"SD={sd[k]:.6f}"
        )

    scale_mean_mv, scale_std_mv, scale_source = load_scaling(
        args.h5,
        args.model_summary,
    )

    print(f"Scaling mean      : {scale_mean_mv:.12g} mV")
    print(f"Scaling std       : {scale_std_mv:.12g} mV")
    print(f"Scaling source    : {scale_source}")

    latent_coordinates, decoded_scaled = decode_traversals(
        model=model,
        mu=mu,
        sd=sd,
        levels_sd=levels_sd,
        device=device,
    )

    decoded_mv = (
        decoded_scaled * scale_std_mv + scale_mean_mv
    ).astype(np.float32)

    empirical_stats = make_empirical_stats_table(
        latent_df=latent_df,
        train_mask=train_mask,
        zcols=zcols,
        mu=mu,
        sd=sd,
    )
    empirical_stats.to_csv(
        args.results_dir / "latent_empirical_stats.csv",
        index=False,
    )

    metrics = make_metrics(
        decoded_mv=decoded_mv,
        latent_coordinates=latent_coordinates,
        mu=mu,
        sd=sd,
        levels_sd=levels_sd,
        fs=args.fs,
    )
    metrics.to_csv(
        args.results_dir / "latent_traversal_metrics.csv",
        index=False,
    )

    ranking = make_excursion_ranking(
        decoded_mv=decoded_mv,
        levels_sd=levels_sd,
        fs=args.fs,
    )
    ranking.to_csv(
        args.results_dir / "latent_traversal_excursion_ranking.csv",
        index=False,
    )

    t_ms = time_axis_ms(args.fs)

    np.savez_compressed(
        args.results_dir / "latent_traversals_hcm_static.npz",
        decoded_scaled=decoded_scaled,
        decoded_mv=decoded_mv,
        latent_coordinates=latent_coordinates,
        latent_hcm_train_mean=mu.astype(np.float32),
        latent_hcm_train_sd=sd.astype(np.float32),
        levels_sd=np.asarray(levels_sd, dtype=np.float32),
        lead_names=np.asarray(LEAD_NAMES),
        sampling_rate_hz=np.asarray([args.fs], dtype=np.float32),
        beat_start_ms=np.asarray(
            [BEAT_START_MS],
            dtype=np.float32,
        ),
        time_axis_ms=t_ms.astype(np.float32),
        scale_mean_mv=np.asarray(
            [scale_mean_mv],
            dtype=np.float64,
        ),
        scale_std_mv=np.asarray(
            [scale_std_mv],
            dtype=np.float64,
        ),
    )

    # The 0-SD decoded waveform is identical across latent traversals.
    zero_idx = int(
        np.where(np.isclose(np.asarray(levels_sd), 0.0))[0][0]
    )
    centroid_mv = decoded_mv[0, zero_idx]
    save_centroid_csv(
        centroid_mv,
        args.fs,
        args.results_dir / "hcm_train_centroid_beat.csv",
    )

    print("Generating traversal figures...")
    for k in range(latent_dim):
        out_path = fig_dir / f"z_{k:02d}_traversal.png"
        plot_latent_traversal(
            decoded_mv_k=decoded_mv[k],
            levels_sd=levels_sd,
            latent_name=f"z_{k:02d}",
            mu_k=float(mu[k]),
            sd_k=float(sd[k]),
            out_path=out_path,
            fs=args.fs,
        )
        print("  saved", out_path.name)

    plot_global_ranking(
        ranking,
        fig_dir / "latent_excursion_global_ranking.png",
    )
    plot_region_ranking(
        ranking,
        fig_dir / "latent_excursion_region_ranking.png",
    )

    summary = {
        "script": "13_analyze_hcm_static_latent_traversals.py",
        "version": "1.0",
        "design": "HCM-static latent traversal",
        "scientific_question": (
            "physiological interpretation of continuous electrical "
            "heterogeneity within HCM"
        ),
        "checkpoint": str(args.checkpoint),
        "latent_csv": str(args.latents),
        "prepared_h5": str(args.h5),
        "hcm_train_n_used_for_latent_statistics": n_hcm_train,
        "latent_dim": latent_dim,
        "levels_sd": levels_sd,
        "generated_median_beats": int(
            latent_dim * len(levels_sd)
        ),
        "sampling_rate_hz": float(args.fs),
        "samples_per_lead": N_SAMPLES,
        "window_ms": {
            "start": BEAT_START_MS,
            "end_nominal": (
                BEAT_START_MS
                + N_SAMPLES * 1000.0 / args.fs
            ),
        },
        "fixed_regions_ms": {
            k: [float(v[0]), float(v[1])]
            for k, v in FIXED_REGIONS_MS.items()
        },
        "reference_latent_vector": (
            "empirical posterior-mean centroid of HCM-train"
        ),
        "traversal_scale": (
            "empirical HCM-train posterior-mean SD per latent dimension"
        ),
        "normal_used_for_latent_statistics": False,
        "hcm_val_used_for_latent_statistics": False,
        "hcm_test_used_for_latent_statistics": False,
        "decoder_output_unscaled_to_mv": True,
        "global_scaling_mean_mv": scale_mean_mv,
        "global_scaling_std_mv": scale_std_mv,
        "scaling_source": scale_source,
        "model_parameters": int(nparams),
        "top_global_excursion_latents": (
            ranking[
                [
                    "latent",
                    "global_rms_minus3_to_plus3_mv",
                    "dominant_temporal_region",
                    "most_sensitive_lead",
                ]
            ]
            .head(8)
            .to_dict(orient="records")
        ),
        "interpretive_caution": (
            "Traversal morphology represents directions learned from "
            "HCM-train. Latent axes are not clinical labels until linked "
            "to conventional ECG descriptors, imaging, genotype, or outcomes."
        ),
    }

    (
        args.results_dir / "traversal_run_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("[excursion ranking]")
    print(
        ranking[
            [
                "latent",
                "global_rms_minus3_to_plus3_mv",
                "QRS_rms_minus3_to_plus3_mv",
                "STT_rms_minus3_to_plus3_mv",
                "dominant_temporal_region",
                "most_sensitive_lead",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 78)
    print("DONE")
    print("Results:", args.results_dir)
    print("=" * 78)


if __name__ == "__main__":
    main()
