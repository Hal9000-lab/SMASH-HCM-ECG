#!/usr/bin/env python
"""
06_train_phase1_medianbeat_beta_vae.py

SMASH-HCM -- Phase I
Normal-anchored beta-VAE for 12-lead median beats.

Expected input
--------------
Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5

Input signal
------------
/signals/ecg
    [N, 12, 500]
    median_all waveform, globally z-scaled using retained normal_train only.

Strict normal-anchored design
-----------------------------
- train ONLY on normal_train
- model selection ONLY on normal_val
- normal_test is held out for physiological reference evaluation
- HCM is evaluation only and is NEVER used for optimization or model selection

Architecture
------------
Default compact encoder:

    [B, 12, 500]
      -> [B, 32, 250]
      -> [B, 64, 125]
      -> [B, 96, 62]
      -> [B, 128, 31]
      -> FC 96
      -> mu/logvar

Default latent dimension = 16.

Decoder mirrors the encoder exactly:

    31 -> 62 -> 125 -> 250 -> 500

Training
--------
- beta-VAE
- reconstruction loss = global mean squared error
- KL loss = mean KL across batch and latent dimensions
- beta warm-up reaches beta_target at epoch KL_WARMUP_EPOCHS
- model selection starts AT that epoch, when beta first equals beta_target
- AdamW + ReduceLROnPlateau + early stopping
- deterministic latent export uses posterior mean mu

Outputs
-------
Results/06_Phase1_MedianBeat_Beta_VAE/
    best_model.pt
    training_history.csv
    training_curves.png
    phase1_medianbeat_latents.csv
    phase1_medianbeat_latents.npz
    model_summary.json

The exported latent CSV also carries selected rhythm and morphology-QC
descriptors when present in the input HDF5. These auxiliary descriptors
are NOT model inputs.

Requirements
------------
pip install torch numpy pandas h5py matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEED = 20261004

SPLIT_NORMAL_TRAIN = 0
SPLIT_NORMAL_VAL = 1
SPLIT_NORMAL_TEST = 2
SPLIT_HCM_EVAL = 3

SPLIT_NAME_MAP = {
    SPLIT_NORMAL_TRAIN: "normal_train",
    SPLIT_NORMAL_VAL: "normal_val",
    SPLIT_NORMAL_TEST: "normal_test",
    SPLIT_HCM_EVAL: "hcm_eval",
}

EXPECTED_LEADS = 12
EXPECTED_SAMPLES = 500


def infer_project_root() -> Path:
    p = Path(__file__).resolve().parent.parent

    if (p / "Data").exists():
        return p

    cwd = Path.cwd().resolve()

    if (cwd / "Data").exists():
        return cwd

    return p


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_str_array(arr) -> np.ndarray:
    out = []

    for x in np.asarray(arr):
        if isinstance(x, bytes):
            out.append(
                x.decode(
                    "utf-8",
                    errors="replace",
                )
            )
        else:
            out.append(str(x))

    return np.asarray(
        out,
        dtype=object,
    )


def read_optional_1d(
    h5: h5py.File,
    path: str,
    n: int,
    default=np.nan,
):
    if path not in h5:
        return np.full(
            n,
            default,
        )

    x = np.asarray(
        h5[path][:]
    )

    if x.ndim != 1 or len(x) != n:
        raise ValueError(
            f"Optional dataset {path} has shape {x.shape}; "
            f"expected [{n}]."
        )

    return x


class ECGDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        indices: np.ndarray,
    ):
        self.signals = signals
        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self):
        return len(
            self.indices
        )

    def __getitem__(self, i):
        idx = int(
            self.indices[i]
        )

        x = torch.from_numpy(
            self.signals[idx]
        ).float()

        return x, idx


class ConvBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
    ):
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
                num_groups=min(
                    8,
                    c_out,
                ),
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
                    num_groups=min(
                        8,
                        c_out,
                    ),
                    num_channels=c_out,
                ),
                nn.SiLU(),
            ]

        self.block = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.block(x)


class MedianBeatBetaVAE(nn.Module):
    """
    Compact beta-VAE for [B, 12, 500].

    Encoder lengths:
        500 -> 250 -> 125 -> 62 -> 31

    Decoder lengths:
        31 -> 62 -> 125 -> 250 -> 500

    The 62 -> 125 transposed-convolution step uses output_padding=1.
    """

    def __init__(
        self,
        latent_dim: int = 16,
    ):
        super().__init__()

        self.latent_dim = int(
            latent_dim
        )

        self.encoder = nn.Sequential(
            ConvBlock(12, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 96),
            ConvBlock(96, 128),
        )

        self.enc_channels = 128
        self.enc_length = 31
        self.enc_flat_dim = (
            self.enc_channels
            * self.enc_length
        )

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

        if (
            h.shape[1] != self.enc_channels
            or h.shape[2] != self.enc_length
        ):
            raise RuntimeError(
                f"Unexpected encoder feature map: {tuple(h.shape)}"
            )

        h = h.flatten(
            start_dim=1
        )

        h = self.enc_fc(h)

        return (
            self.fc_mu(h),
            self.fc_logvar(h),
        )

    @staticmethod
    def reparameterize(
        mu,
        logvar,
    ):
        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(
            std
        )

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

        z = self.reparameterize(
            mu,
            logvar,
        )

        recon = self.decode(z)

        return (
            recon,
            mu,
            logvar,
        )


def vae_loss(
    x,
    recon,
    mu,
    logvar,
    beta: float,
):
    recon_loss = torch.mean(
        (recon - x) ** 2
    )

    kl_per_latent = -0.5 * (
        1.0
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    # IMPORTANT:
    # mean across BOTH batch and latent dimensions.
    kl_loss = torch.mean(
        kl_per_latent
    )

    total = (
        recon_loss
        + beta * kl_loss
    )

    return (
        total,
        recon_loss,
        kl_loss,
    )


@dataclass
class EpochMetrics:
    loss: float
    recon: float
    kl: float


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    beta: float,
    train: bool,
):
    if train:
        model.train()
    else:
        model.eval()

    sum_loss = 0.0
    sum_recon = 0.0
    sum_kl = 0.0
    n = 0

    context = (
        torch.enable_grad()
        if train
        else torch.no_grad()
    )

    with context:
        for x, _ in loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            if train:
                optimizer.zero_grad(
                    set_to_none=True
                )

            recon, mu, logvar = model(x)

            loss, rec, kl = vae_loss(
                x,
                recon,
                mu,
                logvar,
                beta,
            )

            if train:
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            bs = int(
                x.shape[0]
            )

            sum_loss += (
                float(loss.item())
                * bs
            )

            sum_recon += (
                float(rec.item())
                * bs
            )

            sum_kl += (
                float(kl.item())
                * bs
            )

            n += bs

    if n == 0:
        raise RuntimeError(
            "Empty DataLoader encountered."
        )

    return EpochMetrics(
        loss=sum_loss / n,
        recon=sum_recon / n,
        kl=sum_kl / n,
    )


@torch.no_grad()
def evaluate_indices(
    model,
    signals,
    indices,
    batch_size,
    device,
    beta,
):
    ds = ECGDataset(
        signals,
        indices,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    return run_epoch(
        model,
        loader,
        optimizer=None,
        device=device,
        beta=beta,
        train=False,
    )


@torch.no_grad()
def export_latents(
    model,
    signals,
    batch_size,
    device,
):
    model.eval()

    n = int(
        signals.shape[0]
    )

    latent_dim = int(
        model.latent_dim
    )

    z_mu = np.empty(
        (n, latent_dim),
        dtype=np.float32,
    )

    z_logvar = np.empty(
        (n, latent_dim),
        dtype=np.float32,
    )

    recon_mse = np.empty(
        n,
        dtype=np.float32,
    )

    ds = ECGDataset(
        signals,
        np.arange(
            n,
            dtype=np.int64,
        ),
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    for x, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        mu, logvar = model.encode(x)

        # Deterministic reconstruction from posterior mean.
        recon = model.decode(mu)

        mse = torch.mean(
            (recon - x) ** 2,
            dim=(1, 2),
        )

        idx_np = idx.numpy()

        z_mu[idx_np] = (
            mu.detach()
            .cpu()
            .numpy()
        )

        z_logvar[idx_np] = (
            logvar.detach()
            .cpu()
            .numpy()
        )

        recon_mse[idx_np] = (
            mse.detach()
            .cpu()
            .numpy()
        )

    return (
        z_mu,
        z_logvar,
        recon_mse,
    )


def plot_history(
    hist: pd.DataFrame,
    out_path: Path,
    selection_start_epoch: int,
):
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        hist["epoch"],
        hist["train_recon"],
        label="Train reconstruction",
    )

    ax.plot(
        hist["epoch"],
        hist["val_recon"],
        label="Validation reconstruction",
    )

    ax.axvline(
        selection_start_epoch,
        linestyle="--",
        linewidth=1.2,
        label="Fixed beta / selection start",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Reconstruction MSE"
    )

    ax.set_title(
        "Median-beat normal-only beta-VAE"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=170,
    )

    plt.close(fig)


def main():
    root = infer_project_root()

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        type=Path,
        default=(
            root
            / "Results"
            / "05_Phase1_MedianBeat_VAE"
            / "Phase1_MedianBeat_NormalAnchored_preprocessed.h5"
        ),
    )

    ap.add_argument(
        "--results-dir",
        type=Path,
        default=(
            root
            / "Results"
            / "06_Phase1_MedianBeat_Beta_VAE"
        ),
    )

    ap.add_argument(
        "--latent-dim",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=180,
    )

    ap.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    ap.add_argument(
        "--beta",
        type=float,
        default=0.01,
    )

    ap.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = ap.parse_args()

    if args.latent_dim <= 0:
        raise ValueError(
            "--latent-dim must be > 0."
        )

    if args.epochs < args.kl_warmup_epochs:
        raise ValueError(
            "--epochs must be >= --kl-warmup-epochs."
        )

    if args.kl_warmup_epochs <= 0:
        raise ValueError(
            "--kl-warmup-epochs must be > 0."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input not found: {args.input}"
        )

    set_seed(
        args.seed
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Input : {args.input}"
    )

    print(
        f"Results: {args.results_dir}"
    )

    with h5py.File(
        args.input,
        "r",
    ) as h5:

        signals = h5[
            "signals/ecg"
        ][:].astype(
            np.float32
        )

        labels = h5[
            "labels/hcm"
        ][:].astype(
            np.uint8
        )

        split = h5[
            "labels/split_code"
        ][:].astype(
            np.uint8
        )

        subject_key = decode_str_array(
            h5[
                "metadata/subject_key"
            ][:]
        )

        ecg_id = decode_str_array(
            h5[
                "metadata/ecg_id"
            ][:]
        )

        patient_id = decode_str_array(
            h5[
                "metadata/patient_id"
            ][:]
        )

        source = decode_str_array(
            h5[
                "metadata/source"
            ][:]
        )

        n = len(
            labels
        )

        original_record_index = read_optional_1d(
            h5,
            "metadata/original_record_index",
            n,
            default=-1,
        ).astype(
            np.int64
        )

        n_usable_beats = read_optional_1d(
            h5,
            "quality/n_usable_beats",
            n,
        )

        morph_similarity = read_optional_1d(
            h5,
            "quality/morph_pairwise_similarity_median",
            n,
        )

        morph_rms_mv = read_optional_1d(
            h5,
            "quality/morph_rms_to_all_median_mv",
            n,
        )

        morph_relative = read_optional_1d(
            h5,
            "quality/morph_rms_relative_to_template",
            n,
        )

        dominant_fraction = read_optional_1d(
            h5,
            "quality/dominant_cluster_fraction",
            n,
        )

        morph_components = read_optional_1d(
            h5,
            "quality/morph_components_at_threshold",
            n,
        )

        rr_median_ms = read_optional_1d(
            h5,
            "rhythm/rr_median_ms",
            n,
        )

        rr_sd_ms = read_optional_1d(
            h5,
            "rhythm/rr_sd_ms",
            n,
        )

        rr_mad_ms = read_optional_1d(
            h5,
            "rhythm/rr_mad_ms",
            n,
        )

        rr_cv = read_optional_1d(
            h5,
            "rhythm/rr_cv",
            n,
        )

        rmssd_ms = read_optional_1d(
            h5,
            "rhythm/rmssd_ms",
            n,
        )

        heart_rate_bpm = read_optional_1d(
            h5,
            "rhythm/heart_rate_from_median_rr_bpm",
            n,
        )

        scale_mean_mv = (
            float(
                h5[
                    "preprocessing/global_mean_mv"
                ][()]
            )
            if "preprocessing/global_mean_mv" in h5
            else np.nan
        )

        scale_std_mv = (
            float(
                h5[
                    "preprocessing/global_std_mv"
                ][()]
            )
            if "preprocessing/global_std_mv" in h5
            else np.nan
        )

    if signals.shape[1:] != (
        EXPECTED_LEADS,
        EXPECTED_SAMPLES,
    ):
        raise ValueError(
            f"Expected [N,{EXPECTED_LEADS},{EXPECTED_SAMPLES}], "
            f"found {signals.shape}."
        )

    if not np.isfinite(
        signals
    ).all():
        raise ValueError(
            "Input contains NaN/Inf."
        )

    if not (
        len(signals)
        == len(labels)
        == len(split)
        == len(subject_key)
        == len(ecg_id)
        == len(patient_id)
        == len(source)
    ):
        raise ValueError(
            "Record-level arrays have inconsistent lengths."
        )

    train_idx = np.where(
        split
        == SPLIT_NORMAL_TRAIN
    )[0]

    val_idx = np.where(
        split
        == SPLIT_NORMAL_VAL
    )[0]

    test_idx = np.where(
        split
        == SPLIT_NORMAL_TEST
    )[0]

    hcm_idx = np.where(
        split
        == SPLIT_HCM_EVAL
    )[0]

    assert np.all(
        labels[train_idx] == 0
    )

    assert np.all(
        labels[val_idx] == 0
    )

    assert np.all(
        labels[test_idx] == 0
    )

    assert np.all(
        labels[hcm_idx] == 1
    )

    # Subject leakage assertions across Normal splits.
    train_subjects = set(
        subject_key[
            train_idx
        ].tolist()
    )

    val_subjects = set(
        subject_key[
            val_idx
        ].tolist()
    )

    test_subjects = set(
        subject_key[
            test_idx
        ].tolist()
    )

    assert train_subjects.isdisjoint(
        val_subjects
    )

    assert train_subjects.isdisjoint(
        test_subjects
    )

    assert val_subjects.isdisjoint(
        test_subjects
    )

    print()
    print(
        f"Normal train: {len(train_idx):,}"
    )

    print(
        f"Normal val  : {len(val_idx):,}"
    )

    print(
        f"Normal test : {len(test_idx):,}"
    )

    print(
        f"HCM eval    : {len(hcm_idx):,}"
    )

    pin = (
        device.type
        == "cuda"
    )

    train_loader = DataLoader(
        ECGDataset(
            signals,
            train_idx,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        ECGDataset(
            signals,
            val_idx,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    model = MedianBeatBetaVAE(
        latent_dim=args.latent_dim
    ).to(
        device
    )

    # Shape sanity check.
    with torch.no_grad():
        test_x = torch.zeros(
            2,
            EXPECTED_LEADS,
            EXPECTED_SAMPLES,
            device=device,
        )

        test_recon, _, _ = model(
            test_x
        )

        if (
            test_recon.shape
            != test_x.shape
        ):
            raise RuntimeError(
                f"Model reconstruction shape "
                f"{tuple(test_recon.shape)} != input "
                f"{tuple(test_x.shape)}"
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
        )
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Latent dimensions   : {args.latent_dim}"
    )

    print(
        f"Trainable parameters: {n_params:,}"
    )

    print(
        f"Beta target         : {args.beta}"
    )

    print(
        f"Warm-up epochs      : 1-{args.kl_warmup_epochs}"
    )

    print(
        "Model selection starts at epoch "
        f"{args.kl_warmup_epochs}, when beta first equals target."
    )

    print()

    best_val = math.inf
    best_epoch = -1
    wait = 0
    history = []

    checkpoint_path = (
        args.results_dir
        / "best_model.pt"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        if (
            epoch
            <= args.kl_warmup_epochs
        ):
            beta_epoch = (
                args.beta
                * epoch
                / args.kl_warmup_epochs
            )
        else:
            beta_epoch = (
                args.beta
            )

        selection_enabled = (
            epoch
            >= args.kl_warmup_epochs
        )

        train_m = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta_epoch,
            train=True,
        )

        val_m = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            beta=beta_epoch,
            train=False,
        )

        lr_now = (
            optimizer.param_groups[0]["lr"]
        )

        phase_name = (
            "selection"
            if selection_enabled
            else "warmup"
        )

        history.append({
            "epoch":
                epoch,
            "phase":
                phase_name,
            "selection_enabled":
                selection_enabled,
            "beta":
                beta_epoch,
            "lr":
                lr_now,
            "train_loss":
                train_m.loss,
            "train_recon":
                train_m.recon,
            "train_kl":
                train_m.kl,
            "val_loss":
                val_m.loss,
            "val_recon":
                val_m.recon,
            "val_kl":
                val_m.kl,
        })

        tag = (
            "SELECT"
            if selection_enabled
            else "WARMUP"
        )

        print(
            f"Epoch {epoch:03d} "
            f"[{tag}] | "
            f"beta={beta_epoch:.5f} | "
            f"lr={lr_now:.2e} | "
            f"train={train_m.loss:.6f} "
            f"(rec={train_m.recon:.6f}, "
            f"kl={train_m.kl:.6f}) | "
            f"val={val_m.loss:.6f} "
            f"(rec={val_m.recon:.6f}, "
            f"kl={val_m.kl:.6f})"
        )

        if not selection_enabled:
            continue

        # Losses are now comparable because beta is fixed.
        scheduler.step(
            val_m.loss
        )

        if (
            val_m.loss
            < best_val - 1e-6
        ):
            best_val = (
                val_m.loss
            )

            best_epoch = (
                epoch
            )

            wait = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "architecture":
                        "medianbeat_compact_v1",
                    "input_shape":
                        [
                            EXPECTED_LEADS,
                            EXPECTED_SAMPLES,
                        ],
                    "latent_dim":
                        args.latent_dim,
                    "beta":
                        args.beta,
                    "best_epoch":
                        best_epoch,
                    "best_val_loss":
                        best_val,
                    "warmup_epochs":
                        args.kl_warmup_epochs,
                    "seed":
                        args.seed,
                    "normal_train_scaling_mean_mv":
                        scale_mean_mv,
                    "normal_train_scaling_std_mv":
                        scale_std_mv,
                },
                checkpoint_path,
            )
        else:
            wait += 1

        if (
            wait
            >= args.patience
        ):
            print()
            print(
                "Early stopping at epoch "
                f"{epoch}; best fixed-beta epoch = {best_epoch}."
            )

            break

    hist_df = pd.DataFrame(
        history
    )

    history_path = (
        args.results_dir
        / "training_history.csv"
    )

    hist_df.to_csv(
        history_path,
        index=False,
    )

    plot_history(
        hist_df,
        args.results_dir
        / "training_curves.png",
        args.kl_warmup_epochs,
    )

    if (
        best_epoch < 0
        or not checkpoint_path.exists()
    ):
        raise RuntimeError(
            "No valid fixed-beta checkpoint selected."
        )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ]
    )

    model.eval()

    print()
    print(
        "Evaluating frozen best model..."
    )

    normal_test_m = evaluate_indices(
        model,
        signals,
        test_idx,
        args.batch_size,
        device,
        args.beta,
    )

    hcm_eval_m = evaluate_indices(
        model,
        signals,
        hcm_idx,
        args.batch_size,
        device,
        args.beta,
    )

    print(
        f"Normal test | "
        f"loss={normal_test_m.loss:.6f}, "
        f"rec={normal_test_m.recon:.6f}, "
        f"kl={normal_test_m.kl:.6f}"
    )

    print(
        f"HCM eval    | "
        f"loss={hcm_eval_m.loss:.6f}, "
        f"rec={hcm_eval_m.recon:.6f}, "
        f"kl={hcm_eval_m.kl:.6f}"
    )

    print()
    print(
        "Exporting frozen-encoder latent representations..."
    )

    (
        z_mu,
        z_logvar,
        recon_mse,
    ) = export_latents(
        model,
        signals,
        batch_size=args.batch_size,
        device=device,
    )

    latent_df = pd.DataFrame({
        "record_index":
            np.arange(
                len(labels)
            ),
        "original_record_index":
            original_record_index,
        "ecg_id":
            ecg_id,
        "patient_id":
            patient_id,
        "subject_key":
            subject_key,
        "source":
            source,
        "hcm":
            labels,
        "split_code":
            split,
        "split_name":
            [
                SPLIT_NAME_MAP[
                    int(v)
                ]
                for v in split
            ],
        "n_usable_beats":
            n_usable_beats,
        "reconstruction_mse":
            recon_mse,
        "morph_pairwise_similarity_median":
            morph_similarity,
        "morph_rms_to_all_median_mv":
            morph_rms_mv,
        "morph_rms_relative_to_template":
            morph_relative,
        "dominant_cluster_fraction":
            dominant_fraction,
        "morph_components_at_threshold":
            morph_components,
        "rr_median_ms":
            rr_median_ms,
        "rr_sd_ms":
            rr_sd_ms,
        "rr_mad_ms":
            rr_mad_ms,
        "rr_cv":
            rr_cv,
        "rmssd_ms":
            rmssd_ms,
        "heart_rate_bpm":
            heart_rate_bpm,
    })

    for j in range(
        args.latent_dim
    ):
        latent_df[
            f"z_{j:02d}"
        ] = z_mu[
            :,
            j,
        ]

        latent_df[
            f"logvar_{j:02d}"
        ] = z_logvar[
            :,
            j,
        ]

    latent_csv = (
        args.results_dir
        / "phase1_medianbeat_latents.csv"
    )

    latent_df.to_csv(
        latent_csv,
        index=False,
    )

    npz_path = (
        args.results_dir
        / "phase1_medianbeat_latents.npz"
    )

    np.savez_compressed(
        npz_path,
        z_mu=z_mu,
        z_logvar=z_logvar,
        reconstruction_mse=recon_mse,
        labels=labels,
        split=split,
        ecg_id=ecg_id,
        original_record_index=original_record_index,
    )

    best_hist_row = (
        hist_df[
            hist_df[
                "epoch"
            ]
            == best_epoch
        ]
        .iloc[0]
    )

    summary = {
        "design":
            "normal-anchored median-beat beta-VAE",
        "architecture":
            "medianbeat_compact_v1",
        "input_representation":
            "median_all",
        "input_shape":
            [
                EXPECTED_LEADS,
                EXPECTED_SAMPLES,
            ],
        "sampling_rate_hz":
            500,
        "window_seconds":
            1.0,
        "latent_dim":
            int(
                args.latent_dim
            ),
        "beta_target":
            float(
                args.beta
            ),
        "kl_definition":
            (
                "mean KL across batch and latent dimensions"
            ),
        "kl_warmup_epochs":
            int(
                args.kl_warmup_epochs
            ),
        "selection_starts_at_epoch":
            int(
                args.kl_warmup_epochs
            ),
        "normal_train_n":
            int(
                len(train_idx)
            ),
        "normal_val_n":
            int(
                len(val_idx)
            ),
        "normal_test_n":
            int(
                len(test_idx)
            ),
        "hcm_eval_n":
            int(
                len(hcm_idx)
            ),
        "hcm_used_for_training":
            False,
        "hcm_used_for_model_selection":
            False,
        "best_epoch":
            int(
                best_epoch
            ),
        "best_normal_val_loss_fixed_beta":
            float(
                best_val
            ),
        "best_epoch_train_loss":
            float(
                best_hist_row[
                    "train_loss"
                ]
            ),
        "best_epoch_train_recon":
            float(
                best_hist_row[
                    "train_recon"
                ]
            ),
        "best_epoch_train_kl":
            float(
                best_hist_row[
                    "train_kl"
                ]
            ),
        "best_epoch_val_recon":
            float(
                best_hist_row[
                    "val_recon"
                ]
            ),
        "best_epoch_val_kl":
            float(
                best_hist_row[
                    "val_kl"
                ]
            ),
        "normal_test_loss":
            float(
                normal_test_m.loss
            ),
        "normal_test_recon":
            float(
                normal_test_m.recon
            ),
        "normal_test_kl":
            float(
                normal_test_m.kl
            ),
        "hcm_eval_loss":
            float(
                hcm_eval_m.loss
            ),
        "hcm_eval_recon":
            float(
                hcm_eval_m.recon
            ),
        "hcm_eval_kl":
            float(
                hcm_eval_m.kl
            ),
        "trainable_parameters":
            int(
                n_params
            ),
        "device":
            str(
                device
            ),
        "normal_train_scaling_mean_mv":
            (
                float(
                    scale_mean_mv
                )
                if np.isfinite(
                    scale_mean_mv
                )
                else None
            ),
        "normal_train_scaling_std_mv":
            (
                float(
                    scale_std_mv
                )
                if np.isfinite(
                    scale_std_mv
                )
                else None
            ),
        "checkpoint":
            str(
                checkpoint_path
            ),
        "training_history_csv":
            str(
                history_path
            ),
        "latent_csv":
            str(
                latent_csv
            ),
        "latent_npz":
            str(
                npz_path
            ),
    }

    summary_path = (
        args.results_dir
        / "model_summary.json"
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
        "Training complete."
    )

    print(
        f"Best fixed-beta epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best normal val loss : "
        f"{best_val:.8f}"
    )

    print(
        f"Trainable parameters : "
        f"{n_params:,}"
    )

    print(
        f"Checkpoint           : "
        f"{checkpoint_path}"
    )

    print(
        f"Latent coordinates   : "
        f"{latent_csv}"
    )

    print(
        f"Summary              : "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()
