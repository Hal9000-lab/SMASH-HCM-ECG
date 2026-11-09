#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
33_project_mimic_frozen_dual_manifold.py

SMASH-HCM -- Step 33
External MIMIC-IV HCM projection through the two frozen SMASH-HCM manifolds.

Scientific role
---------------
Project the conservatively deduplicated MIMIC-IV HCM median beats through the
already-trained, frozen SMASH-HCM encoders:

    Normal-anchored VAE  -> 16-dimensional z^N
    HCM-oriented VAE     ->  8-dimensional z^HCM

No model is retrained, fine-tuned, recalibrated, or normalized on MIMIC.
The exact original SMASH-HCM training scalings are applied separately for the
two frozen models.

For each MIMIC record eligible for projection, the script exports:
- deterministic posterior means and log-variances for both VAEs;
- Mahalanobis distance d_N from the Normal-training latent distribution;
- Mahalanobis distance d_HCM from the HCM-training latent distribution;
- physical reconstruction errors for each frozen decoder;
- cross-manifold reconstruction-error contrasts.

Reference-distance covariance estimators are fit ONLY to the original SMASH-HCM
training posterior means (Normal-train and HCM-train respectively), never to
MIMIC data.

Canonical inputs
----------------
Results/31_MIMIC_ConservativeDedup/mimic_smash_median_beats_deduplicated.h5
Results/06_Phase1_MedianBeat_Beta_VAE/best_model.pt
Results/06_Phase1_MedianBeat_Beta_VAE/phase1_medianbeat_latents.csv
Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5
Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5
Results/12_HCM_Static_Beta_VAE/best_model.pt
Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv

Outputs
-------
Results/33_MIMIC_FrozenDualManifoldProjection/
    mimic_frozen_dual_manifold.csv
    mimic_frozen_dual_manifold_valid.csv
    mimic_projection_subject_summary.csv
    mimic_frozen_dual_manifold.h5
    projection_distribution_summary.csv
    run_summary.json

Run
---
python Scripts/33_project_mimic_frozen_dual_manifold.py

Dependencies
------------
numpy pandas h5py torch scikit-learn
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import torch
import torch.nn as nn


STEP = 33
SEED = 20261004
FS_HZ = 500.0
BEAT_SAMPLES = 500
NORMAL_LATENT_DIM = 16
HCM_LATENT_DIM = 8
EPS = 1e-12

EXPECTED_NORMAL_PARAMS = 939_116
EXPECTED_HCM_PARAMS = 236_988

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


# =============================================================================
# Paths / CLI
# =============================================================================

def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "Data").exists() and (p / "Results").exists():
            return p

    cwd = Path.cwd().resolve()
    if (cwd / "Data").exists() and (cwd / "Results").exists():
        return cwd

    return here.parent


def first_existing(paths: Sequence[Path], description: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    tried = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(f"Could not find {description}. Tried:\n{tried}")


def resolve_default_paths(root: Path) -> Dict[str, Path]:
    return {
        "mimic_h5": first_existing(
            [
                root / "Results" / "31_MIMIC_ConservativeDedup" / "mimic_smash_median_beats_deduplicated.h5",
            ],
            "Step-31 deduplicated MIMIC HDF5",
        ),
        "normal_checkpoint": first_existing(
            [
                root / "Results" / "06_Phase1_MedianBeat_Beta_VAE" / "best_model.pt",
                root / "Results" / "Phase1_MedianBeat_VAE" / "best_model.pt",
            ],
            "Normal-anchored VAE checkpoint",
        ),
        "normal_latents": first_existing(
            [
                root / "Results" / "06_Phase1_MedianBeat_Beta_VAE" / "phase1_medianbeat_latents.csv",
                root / "Results" / "Phase1_MedianBeat_VAE" / "phase1_medianbeat_latents.csv",
            ],
            "Normal-anchored latent CSV",
        ),
        "normal_prepared_h5": first_existing(
            [
                root / "Results" / "05_Phase1_MedianBeat_VAE" / "Phase1_MedianBeat_NormalAnchored_preprocessed.h5",
                root / "Results" / "05_Phase1_MedianBeat_VAE" / "phase1_medianbeat_normal_anchored_preprocessed.h5",
                root / "Data" / "Harmonized" / "Phase1_MedianBeat_NormalAnchored_preprocessed.h5",
            ],
            "Normal-anchored prepared HDF5",
        ),
        "hcm_prepared_h5": first_existing(
            [
                root / "Results" / "11_HCM_Static_VAE_Preparation" / "Phase2_HCMStatic_preprocessed.h5",
                root / "Data" / "Harmonized" / "Phase2_HCMStatic_preprocessed.h5",
            ],
            "HCM-static prepared HDF5",
        ),
        "hcm_checkpoint": first_existing(
            [
                root / "Results" / "12_HCM_Static_Beta_VAE" / "best_model.pt",
                root / "Results" / "Phase2_HCMStatic_VAE" / "best_model.pt",
            ],
            "HCM-oriented VAE checkpoint",
        ),
        "hcm_latents": first_existing(
            [
                root / "Results" / "12_HCM_Static_Beta_VAE" / "hcm_static_latents.csv",
                root / "Results" / "Phase2_HCMStatic_VAE" / "hcm_static_latents.csv",
            ],
            "HCM-oriented latent CSV",
        ),
    }


def parse_args() -> argparse.Namespace:
    root = infer_project_root()
    defaults = resolve_default_paths(root)

    p = argparse.ArgumentParser(
        description="Project deduplicated MIMIC HCM median beats through frozen SMASH-HCM VAEs."
    )
    p.add_argument("--mimic-h5", type=Path, default=defaults["mimic_h5"])
    p.add_argument("--normal-checkpoint", type=Path, default=defaults["normal_checkpoint"])
    p.add_argument("--normal-latents", type=Path, default=defaults["normal_latents"])
    p.add_argument("--normal-prepared-h5", type=Path, default=defaults["normal_prepared_h5"])
    p.add_argument("--hcm-prepared-h5", type=Path, default=defaults["hcm_prepared_h5"])
    p.add_argument("--hcm-checkpoint", type=Path, default=defaults["hcm_checkpoint"])
    p.add_argument("--hcm-latents", type=Path, default=defaults["hcm_latents"])
    p.add_argument(
        "--results-dir",
        type=Path,
        default=root / "Results" / "33_MIMIC_FrozenDualManifoldProjection",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


# =============================================================================
# Exact frozen architectures
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, output_padding: int, activate: bool):
        super().__init__()
        layers: List[nn.Module] = [
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
                nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
                nn.SiLU(),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MedianBeatBetaVAE(nn.Module):
    """Exact Step-06 Normal-anchored VAE architecture."""

    def __init__(self, latent_dim: int = NORMAL_LATENT_DIM):
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
            nn.Linear(self.enc_flat_dim, 96),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(96, self.latent_dim)
        self.fc_logvar = nn.Linear(96, self.latent_dim)
        self.dec_fc = nn.Sequential(
            nn.Linear(self.latent_dim, 96),
            nn.SiLU(),
            nn.Linear(96, self.enc_flat_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(128, 96, 0, True),
            DeconvBlock(96, 64, 1, True),
            DeconvBlock(64, 32, 0, True),
            DeconvBlock(32, 12, 0, False),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        h = self.enc_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, self.enc_channels, self.enc_length)
        return self.decoder(h)


class HCMStaticBetaVAE(nn.Module):
    """Exact Step-12 HCM-oriented VAE architecture."""

    def __init__(self, latent_dim: int = HCM_LATENT_DIM):
        super().__init__()
        self.latent_dim = int(latent_dim)
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
        self.fc_mu = nn.Linear(48, self.latent_dim)
        self.fc_logvar = nn.Linear(48, self.latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 64 * 31),
            nn.SiLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 48, 4, 2, 1, output_padding=0),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.ConvTranspose1d(48, 32, 4, 2, 1, output_padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose1d(32, 16, 4, 2, 1, output_padding=0),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.ConvTranspose1d(16, 12, 4, 2, 1, output_padding=0),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(z.shape[0], 64, 31)
        return self.decoder_conv(h)


# =============================================================================
# Generic I/O helpers
# =============================================================================

def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def decode_array(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind in ("S", "O", "U"):
        out = []
        for x in a.ravel():
            if isinstance(x, bytes):
                out.append(x.decode("utf-8", errors="replace"))
            else:
                out.append(str(x))
        return np.asarray(out, dtype=object).reshape(a.shape)
    return a


def write_string_dataset(group: h5py.Group, name: str, values: Iterable) -> None:
    vals = ["" if pd.isna(v) else str(v) for v in values]
    group.create_dataset(name, data=np.asarray(vals, dtype=h5py.string_dtype("utf-8")))


def load_mimic_h5(path: Path) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, object]]:
    with h5py.File(path, "r") as h5:
        if "signals/median_all_mv" not in h5:
            raise KeyError("Input HDF5 lacks signals/median_all_mv.")
        waves = np.asarray(h5["signals/median_all_mv"], dtype=np.float32)
        if waves.ndim != 3 or waves.shape[1:] != (12, BEAT_SAMPLES):
            raise ValueError(f"Unexpected MIMIC median-beat shape: {waves.shape}")

        n = len(waves)
        metadata: Dict[str, np.ndarray] = {}
        if "metadata" in h5:
            for key, ds in h5["metadata"].items():
                arr = np.asarray(ds)
                if arr.ndim == 1 and len(arr) == n:
                    metadata[key] = decode_array(arr)

        attrs = {str(k): h5.attrs[k] for k in h5.attrs.keys()}

    if not metadata:
        raise RuntimeError("No per-record metadata found in Step-31 HDF5.")

    df = pd.DataFrame(metadata)
    df.insert(0, "projection_row_index", np.arange(n, dtype=np.int64))

    if "valid_for_projection" not in df.columns:
        raise KeyError("Step-31 HDF5 metadata lacks valid_for_projection.")

    return waves, df, attrs


def get_h5_attr_float(path: Path, name: str) -> float:
    with h5py.File(path, "r") as h5:
        if name not in h5.attrs:
            raise KeyError(f"Missing HDF5 attribute {name!r} in {path}")
        value = float(h5.attrs[name])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite scaling attribute {name!r} in {path}")
    return value


# =============================================================================
# Frozen model loading and fixed scaling
# =============================================================================

def load_frozen_models(args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    nckpt = torch_load_checkpoint(args.normal_checkpoint, device)
    normal_dim = int(nckpt.get("latent_dim", NORMAL_LATENT_DIM))
    if normal_dim != NORMAL_LATENT_DIM:
        raise RuntimeError(f"Unexpected Normal latent dimension: {normal_dim}")

    normal_model = MedianBeatBetaVAE(normal_dim).to(device)
    normal_model.load_state_dict(nckpt["model_state_dict"])
    normal_model.eval()
    normal_params = sum(p.numel() for p in normal_model.parameters())
    if normal_params != EXPECTED_NORMAL_PARAMS:
        raise RuntimeError(
            f"Normal-VAE architecture mismatch: {normal_params:,} parameters; "
            f"expected {EXPECTED_NORMAL_PARAMS:,}."
        )

    if "normal_train_scaling_mean_mv" in nckpt and "normal_train_scaling_std_mv" in nckpt:
        normal_mean = float(nckpt["normal_train_scaling_mean_mv"])
        normal_std = float(nckpt["normal_train_scaling_std_mv"])
        normal_scaling_source = "Step-06 checkpoint"
    else:
        normal_mean = get_h5_attr_float(args.normal_prepared_h5, "normal_train_scaling_mean_mv")
        normal_std = get_h5_attr_float(args.normal_prepared_h5, "normal_train_scaling_std_mv")
        normal_scaling_source = "Step-05 prepared HDF5"

    hckpt = torch_load_checkpoint(args.hcm_checkpoint, device)
    hcm_dim = int(hckpt.get("latent_dim", HCM_LATENT_DIM))
    if hcm_dim != HCM_LATENT_DIM:
        raise RuntimeError(f"Unexpected HCM latent dimension: {hcm_dim}")

    hcm_model = HCMStaticBetaVAE(hcm_dim).to(device)
    hcm_model.load_state_dict(hckpt["model_state_dict"])
    hcm_model.eval()
    hcm_params = sum(p.numel() for p in hcm_model.parameters())
    if hcm_params != EXPECTED_HCM_PARAMS:
        raise RuntimeError(
            f"HCM-VAE architecture mismatch: {hcm_params:,} parameters; "
            f"expected {EXPECTED_HCM_PARAMS:,}."
        )

    hcm_mean = get_h5_attr_float(args.hcm_prepared_h5, "hcm_train_scaling_mean_mv")
    hcm_std = get_h5_attr_float(args.hcm_prepared_h5, "hcm_train_scaling_std_mv")

    for label, value in [("normal std", normal_std), ("HCM std", hcm_std)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid frozen {label}: {value}")

    return {
        "normal_model": normal_model,
        "hcm_model": hcm_model,
        "normal_mean_mv": normal_mean,
        "normal_std_mv": normal_std,
        "hcm_mean_mv": hcm_mean,
        "hcm_std_mv": hcm_std,
        "normal_params": normal_params,
        "hcm_params": hcm_params,
        "normal_scaling_source": normal_scaling_source,
        "hcm_scaling_source": "Step-11 prepared HDF5",
    }


# =============================================================================
# Projection and reconstruction
# =============================================================================

@torch.inference_mode()
def encode_reconstruct(
    model: nn.Module,
    x_mv: np.ndarray,
    scale_mean: float,
    scale_std: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mv = np.asarray(x_mv, dtype=np.float32)
    x_scaled = ((x_mv - scale_mean) / scale_std).astype(np.float32)

    latent_dim = int(model.latent_dim)
    mu_all = np.empty((len(x_mv), latent_dim), dtype=np.float32)
    logvar_all = np.empty_like(mu_all)
    recon_mv = np.empty_like(x_mv, dtype=np.float32)

    for start in range(0, len(x_mv), batch_size):
        stop = min(start + batch_size, len(x_mv))
        xb = torch.from_numpy(x_scaled[start:stop]).to(device)
        mu, logvar = model.encode(xb)
        recon_scaled = model.decode(mu)

        mu_all[start:stop] = mu.detach().cpu().numpy().astype(np.float32)
        logvar_all[start:stop] = logvar.detach().cpu().numpy().astype(np.float32)
        recon_mv[start:stop] = (
            recon_scaled.detach().cpu().numpy() * scale_std + scale_mean
        ).astype(np.float32)

    return mu_all, logvar_all, recon_mv


def time_axis_ms() -> np.ndarray:
    return -350.0 + (1000.0 / FS_HZ) * np.arange(BEAT_SAMPLES, dtype=float)


def compute_reconstruction_metrics(x_mv: np.ndarray, recon_mv: np.ndarray) -> Dict[str, np.ndarray]:
    residual = x_mv.astype(np.float64) - recon_mv.astype(np.float64)
    sq = residual ** 2
    t = time_axis_ms()

    masks = {
        "preQRS": (t >= -300.0) & (t < -60.0),
        "QRS": (t >= -60.0) & (t <= 80.0),
        "STT": (t > 80.0) & (t <= 500.0),
    }

    out: Dict[str, np.ndarray] = {
        "mse": np.mean(sq, axis=(1, 2)),
        "rmse": np.sqrt(np.mean(sq, axis=(1, 2))),
    }
    for region, mask in masks.items():
        out[f"mse_{region}"] = np.mean(sq[:, :, mask], axis=(1, 2))
    out["STT_to_QRS_mse_ratio"] = out["mse_STT"] / (out["mse_QRS"] + EPS)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


# =============================================================================
# Frozen latent reference distributions
# =============================================================================

def get_latent_columns(df: pd.DataFrame, expected: int) -> List[str]:
    cols = [c for c in df.columns if c.startswith("z_") and c[2:].isdigit()]
    cols = sorted(cols, key=lambda c: int(c.split("_")[1]))
    if len(cols) != expected:
        raise RuntimeError(f"Expected {expected} latent columns; found {len(cols)}: {cols}")
    return cols


def fit_reference_gaussians(
    normal_latents_path: Path,
    hcm_latents_path: Path,
) -> Tuple[LedoitWolf, LedoitWolf, Dict[str, object]]:
    normal = pd.read_csv(normal_latents_path)
    hcm = pd.read_csv(hcm_latents_path)

    ncols = get_latent_columns(normal, NORMAL_LATENT_DIM)
    hcols = get_latent_columns(hcm, HCM_LATENT_DIM)

    if "split_name" not in normal.columns or "split_name" not in hcm.columns:
        raise KeyError("Training latent CSVs must contain split_name.")

    nref = normal.loc[normal["split_name"].astype(str).eq("normal_train"), ncols].to_numpy(dtype=float)
    href = hcm.loc[hcm["split_name"].astype(str).eq("hcm_train"), hcols].to_numpy(dtype=float)

    nref = nref[np.isfinite(nref).all(axis=1)]
    href = href[np.isfinite(href).all(axis=1)]

    if len(nref) <= NORMAL_LATENT_DIM or len(href) <= HCM_LATENT_DIM:
        raise RuntimeError("Insufficient finite training latent rows to fit reference Gaussians.")

    lw_n = LedoitWolf().fit(nref)
    lw_h = LedoitWolf().fit(href)

    info = {
        "normal_reference_n": int(len(nref)),
        "hcm_reference_n": int(len(href)),
        "normal_latent_columns": ncols,
        "hcm_latent_columns": hcols,
    }
    return lw_n, lw_h, info


def mahalanobis_distance(x: np.ndarray, model: LedoitWolf) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    delta = x - model.location_[None, :]
    d2 = np.einsum("ni,ij,nj->n", delta, model.precision_, delta)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2).astype(np.float32), d2.astype(np.float32)


# =============================================================================
# Output tables / HDF5
# =============================================================================

def add_projection_columns(
    df: pd.DataFrame,
    valid_idx: np.ndarray,
    n_mu: np.ndarray,
    n_logvar: np.ndarray,
    h_mu: np.ndarray,
    h_logvar: np.ndarray,
    d_n: np.ndarray,
    d_n2: np.ndarray,
    d_h: np.ndarray,
    d_h2: np.ndarray,
    n_metrics: Dict[str, np.ndarray],
    h_metrics: Dict[str, np.ndarray],
) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    out["projected_frozen_dual_manifold"] = False
    out.loc[valid_idx, "projected_frozen_dual_manifold"] = True

    for k in range(NORMAL_LATENT_DIM):
        vals = np.full(n, np.nan, dtype=np.float32)
        vals[valid_idx] = n_mu[:, k]
        out[f"N_z_{k:02d}"] = vals
        vals_lv = np.full(n, np.nan, dtype=np.float32)
        vals_lv[valid_idx] = n_logvar[:, k]
        out[f"N_logvar_{k:02d}"] = vals_lv

    for k in range(HCM_LATENT_DIM):
        vals = np.full(n, np.nan, dtype=np.float32)
        vals[valid_idx] = h_mu[:, k]
        out[f"H_z_{k:02d}"] = vals
        vals_lv = np.full(n, np.nan, dtype=np.float32)
        vals_lv[valid_idx] = h_logvar[:, k]
        out[f"H_logvar_{k:02d}"] = vals_lv

    scalar = {
        "d_N": d_n,
        "d_N2": d_n2,
        "d_HCM": d_h,
        "d_HCM2": d_h2,
    }
    for prefix, metrics in [("RE_N", n_metrics), ("RE_H", h_metrics)]:
        for name, arr in metrics.items():
            scalar[f"{prefix}_{name}"] = arr

    for name, arr in scalar.items():
        vals = np.full(n, np.nan, dtype=np.float32)
        vals[valid_idx] = np.asarray(arr, dtype=np.float32)
        out[name] = vals

    out["Delta_RE_mse"] = out["RE_N_mse"] - out["RE_H_mse"]
    out["log_RE_N_over_RE_H"] = np.log(
        (out["RE_N_mse"].to_numpy(dtype=float) + EPS)
        / (out["RE_H_mse"].to_numpy(dtype=float) + EPS)
    )
    return out


def build_subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "subject_id" not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d["subject_id"] = d["subject_id"].astype(str)
    if "acquisition_datetime" in d.columns:
        d["acquisition_datetime_parsed"] = pd.to_datetime(d["acquisition_datetime"], errors="coerce")
    else:
        d["acquisition_datetime_parsed"] = pd.NaT

    rows = []
    for subject, g in d.groupby("subject_id", dropna=False, sort=True):
        gv = g[g["projected_frozen_dual_manifold"].astype(bool)].copy()
        gv = gv.sort_values(["acquisition_datetime_parsed", "projection_row_index"], kind="mergesort")
        dates = gv["acquisition_datetime_parsed"].dropna()
        followup_years = np.nan
        if len(dates) >= 2:
            followup_years = (dates.max() - dates.min()).total_seconds() / (86400.0 * 365.2425)

        row = {
            "subject_id": subject,
            "n_records_after_dedup": int(len(g)),
            "n_projected_records": int(len(gv)),
            "n_dated_projected_records": int(gv["acquisition_datetime_parsed"].notna().sum()),
            "first_projected_datetime": "" if len(dates) == 0 else dates.min().isoformat(),
            "last_projected_datetime": "" if len(dates) == 0 else dates.max().isoformat(),
            "projected_followup_years": followup_years,
        }

        if len(gv):
            first = gv.iloc[0]
            for col in ["d_N", "d_HCM", "RE_N_mse", "RE_H_mse", "Delta_RE_mse", "H_z_06"]:
                if col in first.index:
                    row[f"baseline_{col}"] = first[col]

        rows.append(row)

    return pd.DataFrame(rows)


def distribution_summary(valid_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c for c in valid_df.columns
        if c.startswith("N_z_")
        or c.startswith("H_z_")
        or c in [
            "d_N", "d_HCM", "RE_N_mse", "RE_H_mse",
            "Delta_RE_mse", "log_RE_N_over_RE_H",
        ]
    ]
    rows = []
    for c in cols:
        x = pd.to_numeric(valid_df[c], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        rows.append({
            "variable": c,
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "median": float(np.median(x)),
            "q025": float(np.quantile(x, 0.025)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
            "q975": float(np.quantile(x, 0.975)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        })
    return pd.DataFrame(rows)


def save_projection_h5(
    path: Path,
    table: pd.DataFrame,
    normal_mean: float,
    normal_std: float,
    hcm_mean: float,
    hcm_std: float,
    source_h5: Path,
    normal_checkpoint: Path,
    hcm_checkpoint: Path,
) -> None:
    n = len(table)
    valid = table["projected_frozen_dual_manifold"].astype(bool).to_numpy()

    with h5py.File(path, "w") as h5:
        h5.attrs["description"] = "MIMIC-IV HCM frozen SMASH-HCM dual-manifold projection"
        h5.attrs["script"] = Path(__file__).name
        h5.attrs["step"] = STEP
        h5.attrs["source_h5"] = str(source_h5.resolve())
        h5.attrs["normal_checkpoint"] = str(normal_checkpoint.resolve())
        h5.attrs["hcm_checkpoint"] = str(hcm_checkpoint.resolve())
        h5.attrs["models_frozen"] = True
        h5.attrs["mimic_used_for_model_fitting"] = False
        h5.attrs["mimic_used_for_scaling"] = False
        h5.attrs["normal_train_scaling_mean_mv"] = normal_mean
        h5.attrs["normal_train_scaling_std_mv"] = normal_std
        h5.attrs["hcm_train_scaling_mean_mv"] = hcm_mean
        h5.attrs["hcm_train_scaling_std_mv"] = hcm_std
        h5.attrs["n_records"] = n
        h5.attrs["n_projected_records"] = int(valid.sum())

        mg = h5.create_group("metadata")
        for c in [
            "projection_row_index", "subject_id", "record_id", "study_id_folder",
            "acquisition_datetime", "valid_for_projection", "qc_label",
            "n_detected_qrs", "n_usable_beats", "valid_visit_number",
            "n_valid_records_subject", "time_from_first_valid_ecg_days",
            "time_from_first_valid_ecg_years", "delta_from_previous_valid_ecg_days",
            "projected_frozen_dual_manifold",
        ]:
            if c not in table.columns:
                continue
            s = table[c]
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                write_string_dataset(mg, c, s)
            elif pd.api.types.is_bool_dtype(s):
                mg.create_dataset(c, data=s.fillna(False).astype(bool).to_numpy(), compression="gzip")
            else:
                mg.create_dataset(c, data=pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64), compression="gzip")

        ng = h5.create_group("normal_manifold")
        ng.create_dataset("z_mu", data=table[[f"N_z_{k:02d}" for k in range(NORMAL_LATENT_DIM)]].to_numpy(dtype=np.float32), compression="gzip")
        ng.create_dataset("logvar", data=table[[f"N_logvar_{k:02d}" for k in range(NORMAL_LATENT_DIM)]].to_numpy(dtype=np.float32), compression="gzip")
        for c in ["d_N", "d_N2", "RE_N_mse", "RE_N_rmse", "RE_N_mse_preQRS", "RE_N_mse_QRS", "RE_N_mse_STT", "RE_N_STT_to_QRS_mse_ratio"]:
            ng.create_dataset(c.replace("RE_N_", ""), data=table[c].to_numpy(dtype=np.float32), compression="gzip")

        hg = h5.create_group("hcm_manifold")
        hg.create_dataset("z_mu", data=table[[f"H_z_{k:02d}" for k in range(HCM_LATENT_DIM)]].to_numpy(dtype=np.float32), compression="gzip")
        hg.create_dataset("logvar", data=table[[f"H_logvar_{k:02d}" for k in range(HCM_LATENT_DIM)]].to_numpy(dtype=np.float32), compression="gzip")
        for c in ["d_HCM", "d_HCM2", "RE_H_mse", "RE_H_rmse", "RE_H_mse_preQRS", "RE_H_mse_QRS", "RE_H_mse_STT", "RE_H_STT_to_QRS_mse_ratio"]:
            hg.create_dataset(c.replace("RE_H_", ""), data=table[c].to_numpy(dtype=np.float32), compression="gzip")

        cg = h5.create_group("cross_manifold")
        cg.create_dataset("Delta_RE_mse", data=table["Delta_RE_mse"].to_numpy(dtype=np.float32), compression="gzip")
        cg.create_dataset("log_RE_N_over_RE_H", data=table["log_RE_N_over_RE_H"].to_numpy(dtype=np.float32), compression="gzip")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = choose_device(args.device)

    print("=" * 88)
    print("33 - MIMIC frozen dual-manifold projection")
    print("=" * 88)
    print(f"Input MIMIC H5 : {args.mimic_h5}")
    print(f"Output dir     : {args.results_dir}")
    print(f"Device         : {device}")
    print()

    waves, meta, input_attrs = load_mimic_h5(args.mimic_h5)
    n_records = len(waves)

    valid_prespecified = meta["valid_for_projection"].astype(bool).to_numpy()
    finite_waveform = np.isfinite(waves).all(axis=(1, 2))
    projection_mask = valid_prespecified & finite_waveform
    valid_idx = np.flatnonzero(projection_mask)

    n_valid_nonfinite = int(np.sum(valid_prespecified & ~finite_waveform))
    n_invalid_but_finite = int(np.sum(~valid_prespecified & finite_waveform))
    if n_valid_nonfinite:
        print(f"WARNING: {n_valid_nonfinite} prespecified-valid records contain non-finite waveform values and will not be projected.")

    models = load_frozen_models(args, device)
    lw_n, lw_h, ref_info = fit_reference_gaussians(args.normal_latents, args.hcm_latents)

    x_valid = waves[valid_idx]
    if len(x_valid) == 0:
        raise RuntimeError("No MIMIC records are eligible for frozen projection.")

    print(f"Records after Step 31 : {n_records:,}")
    print(f"Eligible for projection: {len(valid_idx):,}")
    print(f"Subjects               : {meta['subject_id'].astype(str).nunique() if 'subject_id' in meta.columns else 'NA'}")
    print()
    print("Projecting through frozen Normal-anchored VAE...")
    n_mu, n_logvar, n_recon = encode_reconstruct(
        models["normal_model"], x_valid,
        float(models["normal_mean_mv"]), float(models["normal_std_mv"]),
        args.batch_size, device,
    )
    n_metrics = compute_reconstruction_metrics(x_valid, n_recon)
    d_n, d_n2 = mahalanobis_distance(n_mu, lw_n)

    print("Projecting through frozen HCM-oriented VAE...")
    h_mu, h_logvar, h_recon = encode_reconstruct(
        models["hcm_model"], x_valid,
        float(models["hcm_mean_mv"]), float(models["hcm_std_mv"]),
        args.batch_size, device,
    )
    h_metrics = compute_reconstruction_metrics(x_valid, h_recon)
    d_h, d_h2 = mahalanobis_distance(h_mu, lw_h)

    table = add_projection_columns(
        meta, valid_idx,
        n_mu, n_logvar, h_mu, h_logvar,
        d_n, d_n2, d_h, d_h2,
        n_metrics, h_metrics,
    )

    valid_table = table[table["projected_frozen_dual_manifold"].astype(bool)].copy()
    subject_summary = build_subject_summary(table)
    dist_summary = distribution_summary(valid_table)

    out_all = args.results_dir / "mimic_frozen_dual_manifold.csv"
    out_valid = args.results_dir / "mimic_frozen_dual_manifold_valid.csv"
    out_subject = args.results_dir / "mimic_projection_subject_summary.csv"
    out_dist = args.results_dir / "projection_distribution_summary.csv"
    out_h5 = args.results_dir / "mimic_frozen_dual_manifold.h5"
    out_json = args.results_dir / "run_summary.json"

    table.to_csv(out_all, index=False)
    valid_table.to_csv(out_valid, index=False)
    subject_summary.to_csv(out_subject, index=False)
    dist_summary.to_csv(out_dist, index=False)

    save_projection_h5(
        out_h5, table,
        float(models["normal_mean_mv"]), float(models["normal_std_mv"]),
        float(models["hcm_mean_mv"]), float(models["hcm_std_mv"]),
        args.mimic_h5, args.normal_checkpoint, args.hcm_checkpoint,
    )

    n_subjects = int(table["subject_id"].astype(str).nunique()) if "subject_id" in table.columns else None
    n_subjects_projected = int(valid_table["subject_id"].astype(str).nunique()) if "subject_id" in valid_table.columns else None
    n_2plus = None
    n_3plus = None
    if len(subject_summary):
        n_2plus = int((subject_summary["n_projected_records"] >= 2).sum())
        n_3plus = int((subject_summary["n_projected_records"] >= 3).sum())

    summary = {
        "script": Path(__file__).name,
        "step": STEP,
        "scientific_role": "External MIMIC-IV HCM frozen dual-manifold projection",
        "input_mimic_h5": str(args.mimic_h5.resolve()),
        "results_dir": str(args.results_dir.resolve()),
        "device": str(device),
        "n_records_after_dedup": int(n_records),
        "n_prespecified_valid_for_projection": int(valid_prespecified.sum()),
        "n_projected_records": int(len(valid_idx)),
        "n_prespecified_valid_but_nonfinite": n_valid_nonfinite,
        "n_invalid_but_finite_not_projected": n_invalid_but_finite,
        "n_unique_subjects": n_subjects,
        "n_unique_subjects_with_projection": n_subjects_projected,
        "n_subjects_with_2plus_projected_records": n_2plus,
        "n_subjects_with_3plus_projected_records": n_3plus,
        "external_evaluation_constraints": {
            "models_frozen": True,
            "retraining": False,
            "fine_tuning": False,
            "mimic_used_for_scaling": False,
            "mimic_used_for_reference_covariance": False,
            "posterior_representation": "deterministic posterior mean mu",
        },
        "normal_model": {
            "checkpoint": str(args.normal_checkpoint.resolve()),
            "latent_dim": NORMAL_LATENT_DIM,
            "parameters": int(models["normal_params"]),
            "scaling_mean_mv": float(models["normal_mean_mv"]),
            "scaling_std_mv": float(models["normal_std_mv"]),
            "scaling_source": models["normal_scaling_source"],
            "reference_distribution": "Normal-train posterior means; Ledoit-Wolf covariance",
            "reference_n": int(ref_info["normal_reference_n"]),
        },
        "hcm_model": {
            "checkpoint": str(args.hcm_checkpoint.resolve()),
            "latent_dim": HCM_LATENT_DIM,
            "parameters": int(models["hcm_params"]),
            "scaling_mean_mv": float(models["hcm_mean_mv"]),
            "scaling_std_mv": float(models["hcm_std_mv"]),
            "scaling_source": models["hcm_scaling_source"],
            "reference_distribution": "HCM-train posterior means; Ledoit-Wolf covariance",
            "reference_n": int(ref_info["hcm_reference_n"]),
        },
        "input_h5_attributes": {
            k: (v.item() if isinstance(v, np.generic) else str(v) if isinstance(v, bytes) else v)
            for k, v in input_attrs.items()
            if isinstance(v, (str, int, float, bool, np.generic, bytes))
        },
        "outputs": {
            "all_records_csv": str(out_all.resolve()),
            "valid_records_csv": str(out_valid.resolve()),
            "subject_summary_csv": str(out_subject.resolve()),
            "distribution_summary_csv": str(out_dist.resolve()),
            "projection_h5": str(out_h5.resolve()),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print("DONE")
    print(f"Projected records : {len(valid_idx):,} / {n_records:,}")
    if n_subjects_projected is not None:
        print(f"Projected subjects: {n_subjects_projected:,}")
    if n_2plus is not None and n_3plus is not None:
        print(f"Subjects >=2 projected records: {n_2plus:,}")
        print(f"Subjects >=3 projected records: {n_3plus:,}")
    print(f"CSV all   : {out_all}")
    print(f"CSV valid : {out_valid}")
    print(f"HDF5      : {out_h5}")
    print(f"Summary   : {out_json}")


if __name__ == "__main__":
    main()
