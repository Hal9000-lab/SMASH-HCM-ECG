#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_analyze_normal_manifold_residuals.py

SMASH-HCM / Phase 1
Spatial and temporal analysis of reconstruction residuals from the frozen
Normal-anchored median-beat beta-VAE.

SCIENTIFIC QUESTION
-------------------
The Normal-anchored beta-VAE defines a physiological ECG manifold. For an HCM ECG:

    X_HCM = X_normal-manifold + residual_off-manifold

This script asks:

    WHERE is the morphology that the Normal manifold cannot explain?

The reconstruction residual is decomposed by:
    * temporal region:
        pre-QRS  : -300 to -60 ms
        QRS      :  -60 to +80 ms
        ST-T     :  +80 to +500 ms
    * lead:
        I, II, III, aVR, aVL, aVF, V1-V6
    * lead x temporal region

Primary analyses:
    1. deterministic reconstruction using decoder(mu)
    2. physical residuals in mV
    3. Normal-test vs HCM comparison
    4. residual localization by lead and temporal region
    5. association between latent coordinates and residual compartments
    6. focused test of the z_06 "normal QRS-T coupling" hypothesis

IMPORTANT
---------
This is NOT a diagnostic classifier analysis. AUROC values are included only as
descriptive measures of group separation and must not be interpreted as evidence of
clinical diagnostic performance because Normal and HCM cohorts originate from different
source datasets.

Expected project layout
-----------------------
_SMASH-HCM/
├── Results/
│   ├── 05_Phase1_MedianBeat_VAE/
│   │   └── Phase1_MedianBeat_NormalAnchored_preprocessed.h5
│   ├── 06_Phase1_MedianBeat_Beta_VAE/
│   │   ├── best_model.pt
│   │   └── phase1_medianbeat_latents.csv
│   └── 09_Phase1_MedianBeat_LatentECGFeatures/
│       └── ecg_classical_features.csv        [optional but recommended]
└── Scripts/
    └── 10_analyze_normal_manifold_residuals_v1_3.py

Default outputs
---------------
Results/10_Phase1_MedianBeat_NormalManifoldResiduals/
├── residual_record_summary.csv
├── residual_group_summary.csv
├── residual_group_effects_hcm_vs_normal_test.csv
├── residual_hotspot_ranking.csv
├── latent_residual_associations_normal_train.csv
├── latent_residual_associations_hcm.csv
├── latent_residual_associations_normal_test.csv
├── z06_qrst_uncoupling_analysis.csv
├── mean_residual_maps.npz
├── run_summary.json
└── figures/
    ├── residual_region_distribution.png
    ├── residual_region_ratio_distribution.png
    ├── normal_test_mean_abs_residual_heatmap.png
    ├── hcm_mean_abs_residual_heatmap.png
    ├── hcm_minus_normal_test_abs_residual_heatmap.png
    ├── lead_region_effect_heatmap.png
    ├── residual_time_profile.png
    ├── latent_residual_heatmap_hcm.png
    └── z06_stt_residual.png

Run
---
python Scripts\\10_analyze_normal_manifold_residuals.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

FIXED_REGIONS_MS = {
    "preQRS": (-300.0, -60.0),
    "QRS": (-60.0, 80.0),
    "STT": (80.0, 500.0),
}

EPS = 1e-12


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze Normal-manifold reconstruction residuals in time and lead space."
    )
    p.add_argument(
        "--h5",
        type=Path,
        default=Path("Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5"),
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path("Results/06_Phase1_MedianBeat_Beta_VAE/best_model.pt"),
    )
    p.add_argument(
        "--latents",
        type=Path,
        default=Path("Results/06_Phase1_MedianBeat_Beta_VAE/phase1_medianbeat_latents.csv"),
    )
    p.add_argument(
        "--features09",
        type=Path,
        default=Path("Results/09_Phase1_MedianBeat_LatentECGFeatures/ecg_classical_features.csv"),
        help="Optional script-09 output; used for per-record QRS-boundary supplemental metrics.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("Results/10_Phase1_MedianBeat_NormalManifoldResiduals"),
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--fs", type=float, default=500.0)
    p.add_argument("--pre-ms", type=float, default=-350.0)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Bootstrap repetitions for descriptive AUROC confidence intervals.",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Exact median-beat VAE architecture from script 06
# =====================================================================

class MedianBeatBetaVAE(nn.Module):
    """
    Architecture:
        input [12,500]
        Conv 12->32 : 500 -> 250
        Conv 32->64 : 250 -> 125
        Conv 64->96 : 125 -> 62
        Conv 96->128: 62 -> 31
        flatten 128*31 -> FC 96 -> mu/logvar latent
        decoder FC 96 -> 128*31
        ConvTranspose:
            128->96 : 31 -> 62
             96->64 : 62 -> 125, output_padding=1
             64->32 : 125 -> 250
             32->12 : 250 -> 500
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            nn.Conv1d(64, 96, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 96),
            nn.SiLU(),

            nn.Conv1d(96, 128, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 31, 96),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(96, latent_dim)
        self.fc_logvar = nn.Linear(96, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.SiLU(),
            nn.Linear(96, 128 * 31),
            nn.SiLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(
                128, 96, kernel_size=6, stride=2, padding=2, output_padding=0
            ),  # 31 -> 62
            nn.GroupNorm(8, 96),
            nn.SiLU(),

            nn.ConvTranspose1d(
                96, 64, kernel_size=5, stride=2, padding=2, output_padding=0
            ),  # 62 -> 123? corrected below in robust architecture fallback
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            nn.ConvTranspose1d(
                64, 32, kernel_size=6, stride=2, padding=2, output_padding=0
            ),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.ConvTranspose1d(
                32, 12, kernel_size=6, stride=2, padding=2, output_padding=0
            ),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.shape[0], 128, 31)
        y = self.decoder_conv(h)
        # Architecture-independent final safety: exact input length.
        if y.shape[-1] != 500:
            y = torch.nn.functional.interpolate(
                y, size=500, mode="linear", align_corners=False
            )
        return y

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(mu), mu, logvar


class MedianBeatBetaVAEExact(nn.Module):
    """
    Exact shape-preserving implementation matching the documented script-06 dimensions.
    ConvTranspose parameters are chosen to reproduce:
       31 -> 62 -> 125 -> 250 -> 500.
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc1 = nn.Conv1d(12, 32, 7, 2, 3)
        self.gn1 = nn.GroupNorm(8, 32)
        self.enc2 = nn.Conv1d(32, 64, 7, 2, 3)
        self.gn2 = nn.GroupNorm(8, 64)
        self.enc3 = nn.Conv1d(64, 96, 7, 2, 3)
        self.gn3 = nn.GroupNorm(8, 96)
        self.enc4 = nn.Conv1d(96, 128, 7, 2, 3)
        self.gn4 = nn.GroupNorm(8, 128)

        self.fc_enc = nn.Linear(128 * 31, 96)
        self.fc_mu = nn.Linear(96, latent_dim)
        self.fc_logvar = nn.Linear(96, latent_dim)

        self.fc_dec1 = nn.Linear(latent_dim, 96)
        self.fc_dec2 = nn.Linear(96, 128 * 31)

        # output length = (Lin-1)*s - 2p + k + output_padding
        self.dec1 = nn.ConvTranspose1d(128, 96, 6, 2, 2, output_padding=0)  # 31->62
        self.dgn1 = nn.GroupNorm(8, 96)

        self.dec2 = nn.ConvTranspose1d(96, 64, 7, 2, 3, output_padding=0)   # 62->123?
        self.dgn2 = nn.GroupNorm(8, 64)

        self.dec3 = nn.ConvTranspose1d(64, 32, 6, 2, 2, output_padding=0)
        self.dgn3 = nn.GroupNorm(8, 32)

        self.dec4 = nn.ConvTranspose1d(32, 12, 6, 2, 2, output_padding=0)

        self.act = nn.SiLU()

    def encode(self, x):
        h = self.act(self.gn1(self.enc1(x)))
        h = self.act(self.gn2(self.enc2(h)))
        h = self.act(self.gn3(self.enc3(h)))
        h = self.act(self.gn4(self.enc4(h)))
        h = h.reshape(h.shape[0], -1)
        h = self.act(self.fc_enc(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.act(self.fc_dec1(z))
        h = self.act(self.fc_dec2(h))
        h = h.view(z.shape[0], 128, 31)

        h = self.act(self.dgn1(self.dec1(h)))
        h = self.act(self.dgn2(self.dec2(h)))
        h = self.act(self.dgn3(self.dec3(h)))
        y = self.dec4(h)

        if y.shape[-1] != 500:
            y = torch.nn.functional.interpolate(
                y, size=500, mode="linear", align_corners=False
            )
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(mu), mu, logvar


# =====================================================================
# Checkpoint loading
# =====================================================================

def extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in [
            "model_state_dict", "state_dict", "model", "net",
            "vae_state_dict", "best_model_state_dict",
        ]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]

        # Raw state dict.
        if len(obj) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj

    raise ValueError("Could not identify model state_dict in checkpoint.")


def strip_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    prefixes = ["module.", "model.", "vae."]
    for k, v in state.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        out[nk] = v
    return out


def infer_model_from_checkpoint(
    checkpoint_path: Path,
    latent_dim: int,
    device: torch.device,
) -> nn.Module:
    """
    Load the exact 939,116-parameter model.

    Strategy:
      1. use a serialized nn.Module directly, if present;
      2. try exact state_dict key matching;
      3. if names differ but architecture is identical, remap tensors by
         state_dict order only after verifying every tensor shape.

    This tolerates harmless naming differences while refusing incompatible models.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
        model.eval()
        nparams = sum(p.numel() for p in model.parameters())
        print(f"[model] serialized module parameters: {nparams:,}")
        return model

    source = strip_prefixes(extract_state_dict(ckpt))
    model = CompatibleMedianBeatVAE(latent_dim=latent_dim)

    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] instantiated parameters: {nparams:,}")

    if latent_dim == 16 and nparams != 939116:
        raise RuntimeError(
            f"Internal architecture mismatch: expected 939,116 parameters, got {nparams:,}."
        )

    try:
        model.load_state_dict(source, strict=True)
        print("[model] checkpoint loaded with exact state_dict keys")
    except Exception as exact_error:
        target = model.state_dict()
        src_items = list(source.items())
        tgt_items = list(target.items())

        if len(src_items) != len(tgt_items):
            print("[model] exact-key load error:", exact_error)
            print(f"[model] source tensors={len(src_items)}, target tensors={len(tgt_items)}")
            print("[model] source key preview:")
            for k, v in src_items[:30]:
                print(f"    {k}: {tuple(v.shape)}")
            raise RuntimeError(
                "Checkpoint tensor count does not match the documented median-beat VAE."
            )

        src_shapes = [tuple(v.shape) for _, v in src_items]
        tgt_shapes = [tuple(v.shape) for _, v in tgt_items]

        if src_shapes != tgt_shapes:
            print("[model] exact-key load error:", exact_error)
            print("[model] first tensor-shape mismatches:")
            shown = 0
            for (sk, sv), (tk, tv) in zip(src_items, tgt_items):
                if tuple(sv.shape) != tuple(tv.shape):
                    print(
                        f"    source {sk} {tuple(sv.shape)} "
                        f"!= target {tk} {tuple(tv.shape)}"
                    )
                    shown += 1
                    if shown >= 12:
                        break
            raise RuntimeError(
                "Checkpoint tensor shapes do not match the documented "
                "939,116-parameter median-beat VAE."
            )

        remapped = {
            tk: sv
            for (sk, sv), (tk, tv) in zip(src_items, tgt_items)
        }
        model.load_state_dict(remapped, strict=True)
        print(
            "[model] checkpoint names differed; loaded by verified "
            "tensor-order/shape mapping"
        )

    model = model.to(device)
    model.eval()
    return model


class CompatibleMedianBeatVAE(nn.Module):
    """
    Exact architecture used by script 06.

    With latent_dim=16 the parameter count is exactly 939,116.

    Encoder:
        Conv1d(kernel=4, stride=2, padding=1)
        500 -> 250 -> 125 -> 62 -> 31

    Decoder:
        ConvTranspose1d(kernel=4, stride=2, padding=1)
        31 -> 62   (output_padding=0)
        62 -> 125  (output_padding=1)
        125 -> 250 (output_padding=0)
        250 -> 500 (output_padding=0)
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            nn.Conv1d(64, 96, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),

            nn.Conv1d(96, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 31, 96),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(96, latent_dim)
        self.fc_logvar = nn.Linear(96, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.SiLU(),
            nn.Linear(96, 128 * 31),
            nn.SiLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(
                128, 96, kernel_size=4, stride=2, padding=1,
                output_padding=0
            ),
            nn.GroupNorm(8, 96),
            nn.SiLU(),

            nn.ConvTranspose1d(
                96, 64, kernel_size=4, stride=2, padding=1,
                output_padding=1
            ),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            nn.ConvTranspose1d(
                64, 32, kernel_size=4, stride=2, padding=1,
                output_padding=0
            ),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.ConvTranspose1d(
                32, 12, kernel_size=4, stride=2, padding=1,
                output_padding=0
            ),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.shape[0], 128, 31)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        return self.decode(mu), mu, logvar


def build_compatible_model_from_state_dict(
    state: Dict[str, torch.Tensor],
    latent_dim: int,
) -> nn.Module:
    # The documented script-06 architecture most naturally maps to these names.
    return CompatibleMedianBeatVAE(latent_dim=latent_dim)


# =====================================================================
# HDF5 data loading
# =====================================================================

def decode_array(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind in ("S", "O"):
        vals = []
        for x in a.ravel():
            if isinstance(x, bytes):
                vals.append(x.decode("utf-8", errors="replace"))
            else:
                vals.append(str(x))
        return np.asarray(vals, dtype=object).reshape(a.shape)
    return a


def h5_inventory(h5: h5py.File) -> Dict[str, h5py.Dataset]:
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj

    h5.visititems(visit)
    return out


def find_dataset(
    inv: Dict[str, h5py.Dataset],
    candidates: Sequence[str],
) -> Optional[h5py.Dataset]:
    for c in candidates:
        if c in inv:
            return inv[c]

    base = {Path(k).name.lower(): v for k, v in inv.items()}
    for c in candidates:
        cc = Path(c).name.lower()
        if cc in base:
            return base[cc]

    for c in candidates:
        cl = c.lower()
        for name, ds in inv.items():
            if cl in name.lower():
                return ds
    return None


def load_phase1_h5(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    """
    Returns:
        x_scaled [N,12,500]
        x_mv     [N,12,500]
        metadata
        attrs
    """
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as h5:
        inv = h5_inventory(h5)
        attrs = {}
        for k, v in h5.attrs.items():
            try:
                if isinstance(v, bytes):
                    attrs[k] = v.decode("utf-8")
                elif np.isscalar(v):
                    attrs[k] = v.item() if hasattr(v, "item") else v
                else:
                    attrs[k] = np.asarray(v).tolist()
            except Exception:
                attrs[k] = str(v)

        mv_ds = find_dataset(inv, [
            "median_all_mv", "median_beat_all_mv", "median_beats_all_mv", "median_all"
        ])
        scaled_ds = find_dataset(inv, [
            "median_all_scaled", "median_all_z", "x_scaled",
            "vae_input_scaled", "medianbeat_scaled", "median_scaled"
        ])

        if mv_ds is None and scaled_ds is None:
            raise KeyError(
                "Could not find physical or scaled median-beat waveforms in HDF5.\n"
                + "\n".join(inv.keys())
            )

        def standardize(a):
            a = np.asarray(a, dtype=np.float32)
            if a.ndim != 3:
                raise ValueError(f"Expected 3D waveform array, got {a.shape}")
            if a.shape[1] == 12:
                return a
            if a.shape[2] == 12:
                return np.transpose(a, (0, 2, 1))
            raise ValueError(f"Cannot identify lead axis in {a.shape}")

        x_mv = standardize(mv_ds) if mv_ds is not None else None
        x_scaled = standardize(scaled_ds) if scaled_ds is not None else None

        mean_mv = float(attrs.get("normal_train_scaling_mean_mv", np.nan))
        std_mv = float(attrs.get("normal_train_scaling_std_mv", np.nan))

        if x_scaled is None:
            if x_mv is None or not np.isfinite(mean_mv) or not np.isfinite(std_mv):
                raise ValueError(
                    "Scaled waveform missing and scaling attributes unavailable."
                )
            x_scaled = (x_mv - mean_mv) / std_mv

        if x_mv is None:
            if not np.isfinite(mean_mv) or not np.isfinite(std_mv):
                raise ValueError(
                    "Physical waveform missing and scaling attributes unavailable."
                )
            x_mv = x_scaled * std_mv + mean_mv

        n = x_scaled.shape[0]

        metadata = {}
        # Retain all per-record one-dimensional datasets.
        for name, ds in inv.items():
            try:
                arr = decode_array(np.asarray(ds))
            except Exception:
                continue
            if arr.ndim == 1 and len(arr) == n:
                metadata[Path(name).name] = arr

        # Recover membership from shorter index arrays.
        membership_tokens = {
            "normal_train": ("normal_train", "train_normal"),
            "normal_val": ("normal_val", "normal_validation", "val_normal"),
            "normal_test": ("normal_test", "test_normal"),
            "hcm": ("hcm_eval", "hcm_test", "hcm_indices", "hcm_index"),
        }

        for group_name, tokens in membership_tokens.items():
            if f"__membership_{group_name}" in metadata:
                continue

            for name, ds in inv.items():
                lname = name.lower()
                if not any(tok in lname for tok in tokens):
                    continue

                try:
                    arr = np.asarray(ds)
                except Exception:
                    continue

                if arr.ndim != 1:
                    continue

                if len(arr) == n:
                    try:
                        vals = np.asarray(arr, dtype=float)
                        finite = vals[np.isfinite(vals)]
                        if set(np.unique(finite).tolist()).issubset({0.0, 1.0}):
                            metadata[f"__membership_{group_name}"] = vals.astype(bool)
                            break
                    except Exception:
                        pass

                if np.issubdtype(arr.dtype, np.integer) and 0 < len(arr) < n:
                    idx = np.asarray(arr, dtype=int)
                    if len(idx) and np.min(idx) >= 0 and np.max(idx) < n:
                        mask = np.zeros(n, dtype=bool)
                        mask[idx] = True
                        metadata[f"__membership_{group_name}"] = mask
                        break

    return (
        np.asarray(x_scaled, dtype=np.float32),
        np.asarray(x_mv, dtype=np.float32),
        metadata,
        attrs,
    )


# =====================================================================
# Group membership
# =====================================================================

def metadata_frame(meta: Dict[str, np.ndarray], n: int) -> pd.DataFrame:
    d = {}
    for k, v in meta.items():
        a = np.asarray(v)
        if a.ndim == 1 and len(a) == n:
            d[k] = a
    return pd.DataFrame(d)


def binary_mask(s: pd.Series) -> Optional[np.ndarray]:
    if s is None:
        return None

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).to_numpy(dtype=bool)

    num = pd.to_numeric(s, errors="coerce")
    finite = num[np.isfinite(num)]
    if len(finite):
        u = set(np.unique(finite).tolist())
        if u.issubset({0.0, 1.0}):
            return num.fillna(0).to_numpy(dtype=float).astype(bool)

    return None


def first_mask(df: pd.DataFrame, names: Sequence[str]) -> Optional[np.ndarray]:
    for c in names:
        if c in df.columns:
            m = binary_mask(df[c])
            if m is not None:
                return m
    return None


def derive_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    n = len(df)

    nt = first_mask(df, [
        "__membership_normal_train",
        "normal_train_mask", "normal_train", "is_normal_train"
    ])
    nv = first_mask(df, [
        "__membership_normal_val",
        "normal_val_mask", "normal_val", "is_normal_val"
    ])
    nte = first_mask(df, [
        "__membership_normal_test",
        "normal_test_mask", "normal_test", "is_normal_test"
    ])
    hcm = first_mask(df, [
        "__membership_hcm",
        "hcm_eval_mask", "hcm_mask", "is_hcm", "hcm_eval"
    ])

    text_fields = [
        c for c in [
            "split", "set", "partition", "subset", "group",
            "dataset_split", "split_name", "sample_split",
            "cohort", "source", "dataset", "class", "class_name"
        ]
        if c in df.columns
    ]

    txt = pd.Series([""] * n, index=df.index, dtype=object)
    for c in text_fields:
        txt = txt + " " + df[c].astype(str).str.lower().fillna("")

    if hcm is None:
        hcm = txt.str.contains(r"\bhcm\b|hcm_eval|hcm_test", regex=True).to_numpy()

    if (hcm is None or not hcm.any()) and "label" in df.columns:
        lab = pd.to_numeric(df["label"], errors="coerce")
        if np.isfinite(lab).any():
            hcm = lab.to_numpy(dtype=float) == 1.0

    if hcm is None:
        hcm = np.zeros(n, dtype=bool)

    if nt is None:
        nt = txt.str.contains(
            r"normal[_\-\s]*train|train[_\-\s]*normal",
            regex=True
        ).to_numpy()
    if nv is None:
        nv = txt.str.contains(
            r"normal[_\-\s]*(?:val|validation)|(?:val|validation)[_\-\s]*normal",
            regex=True
        ).to_numpy()
    if nte is None:
        nte = txt.str.contains(
            r"normal[_\-\s]*test|test[_\-\s]*normal",
            regex=True
        ).to_numpy()

    # Generic fallback.
    if not nt.any():
        nt = txt.str.contains(r"\btrain\b", regex=True).to_numpy() & ~hcm
    if not nv.any():
        nv = txt.str.contains(r"\bval\b|\bvalidation\b", regex=True).to_numpy() & ~hcm
    if not nte.any():
        nte = txt.str.contains(r"\btest\b", regex=True).to_numpy() & ~hcm

    normal = ~hcm

    return {
        "normal_train": np.asarray(nt, dtype=bool),
        "normal_val": np.asarray(nv, dtype=bool),
        "normal_test": np.asarray(nte, dtype=bool),
        "normal_all": np.asarray(normal, dtype=bool),
        "hcm": np.asarray(hcm, dtype=bool),
        "all": np.ones(n, dtype=bool),
    }


# =====================================================================
# Reconstruction
# =====================================================================

def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def reconstruct_dataset(
    model: nn.Module,
    x_scaled: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic posterior-mean reconstruction:
        mu = encoder(x)
        xhat = decoder(mu)

    Returns:
        recon_scaled [N,12,T]
        mu           [N,K]
        logvar       [N,K]
    """
    n = len(x_scaled)
    recon = np.empty_like(x_scaled, dtype=np.float32)
    mus = []
    logvars = []

    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        xb = torch.from_numpy(x_scaled[start:stop]).to(device=device, dtype=torch.float32)

        if hasattr(model, "encode") and hasattr(model, "decode"):
            mu, logvar = model.encode(xb)
            yb = model.decode(mu)
        else:
            out = model(xb)
            if isinstance(out, (tuple, list)) and len(out) >= 3:
                yb, mu, logvar = out[:3]
            else:
                raise RuntimeError(
                    "Loaded model does not expose encode/decode and forward output "
                    "is not (recon, mu, logvar)."
                )

        recon[start:stop] = yb.detach().cpu().numpy().astype(np.float32)
        mus.append(mu.detach().cpu().numpy())
        logvars.append(logvar.detach().cpu().numpy())

        if start == 0 or stop == n or (start // batch_size) % 10 == 0:
            print(f"[reconstruction] {stop}/{n}")

    return (
        recon,
        np.concatenate(mus, axis=0),
        np.concatenate(logvars, axis=0),
    )


# =====================================================================
# Residual metrics
# =====================================================================

def time_axis_ms(n_samples: int, fs: float, pre_ms: float) -> np.ndarray:
    return pre_ms + np.arange(n_samples) * 1000.0 / fs


def region_indices(t_ms: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for name, (lo, hi) in FIXED_REGIONS_MS.items():
        # Left-inclusive; right-exclusive except final region effectively harmless.
        out[name] = np.where((t_ms >= lo) & (t_ms < hi))[0]
    return out


def mse(a: np.ndarray, axis=None) -> np.ndarray:
    return np.mean(np.square(a), axis=axis)


def mae(a: np.ndarray, axis=None) -> np.ndarray:
    return np.mean(np.abs(a), axis=axis)


def rmse(a: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(mse(a, axis=axis))


def load_script09_boundaries(
    path: Path,
    n: int,
) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if len(df) != n:
        warnings.warn(
            f"Script-09 feature file has {len(df)} rows but HDF5 has {n}; "
            "individualized QRS residual metrics will be skipped."
        )
        return None

    required = {"qrs_on_ms", "qrs_off_ms"}
    if not required.issubset(df.columns):
        return None

    return df


def compute_record_metrics(
    residual_mv: np.ndarray,
    x_mv: np.ndarray,
    recon_mv: np.ndarray,
    metadata: pd.DataFrame,
    masks: Dict[str, np.ndarray],
    t_ms: np.ndarray,
    features09: Optional[pd.DataFrame],
) -> pd.DataFrame:
    n, nlead, nsamp = residual_mv.shape
    regions = region_indices(t_ms)

    rows = []

    for i in range(n):
        r = residual_mv[i]
        row = {
            "row_index": i,
            "residual_mse_mv2": float(mse(r)),
            "residual_rmse_mv": float(rmse(r)),
            "residual_mae_mv": float(mae(r)),
            "signal_rms_mv": float(np.sqrt(np.mean(np.square(x_mv[i])))),
            "reconstruction_rms_mv": float(np.sqrt(np.mean(np.square(recon_mv[i])))),
        }

        # Copy compact identity/group metadata.
        for c in [
            "ecg_id", "subject_id", "patient_id", "source", "cohort",
            "split", "set", "partition", "label", "is_hcm",
            "age", "sex", "gender", "median_rr_ms", "rr_ms"
        ]:
            if c in metadata.columns:
                row[c] = metadata.iloc[i][c]

        # Explicit analysis group.
        if masks["hcm"][i]:
            row["analysis_group"] = "hcm"
        elif masks["normal_test"][i]:
            row["analysis_group"] = "normal_test"
        elif masks["normal_val"][i]:
            row["analysis_group"] = "normal_val"
        elif masks["normal_train"][i]:
            row["analysis_group"] = "normal_train"
        else:
            row["analysis_group"] = "normal_other"

        # Per lead global.
        for li, lead in enumerate(LEADS):
            row[f"residual_mse_mv2_{lead}"] = float(mse(r[li]))
            row[f"residual_rmse_mv_{lead}"] = float(rmse(r[li]))
            row[f"residual_mae_mv_{lead}"] = float(mae(r[li]))

        # Fixed regions, globally and by lead.
        region_mses = {}
        for region, idx in regions.items():
            rr = r[:, idx]
            region_mses[region] = float(mse(rr))

            row[f"residual_mse_mv2_{region}"] = region_mses[region]
            row[f"residual_rmse_mv_{region}"] = float(rmse(rr))
            row[f"residual_mae_mv_{region}"] = float(mae(rr))

            for li, lead in enumerate(LEADS):
                rrl = r[li, idx]
                row[f"residual_mse_mv2_{region}_{lead}"] = float(mse(rrl))
                row[f"residual_rmse_mv_{region}_{lead}"] = float(rmse(rrl))
                row[f"residual_mae_mv_{region}_{lead}"] = float(mae(rrl))

        # Ratios with epsilon regularization.
        row["STT_to_QRS_mse_ratio"] = (
            region_mses["STT"] / (region_mses["QRS"] + EPS)
        )
        row["preQRS_to_QRS_mse_ratio"] = (
            region_mses["preQRS"] / (region_mses["QRS"] + EPS)
        )
        denom = (
            region_mses["preQRS"]
            + region_mses["QRS"]
            + region_mses["STT"]
            + EPS
        )
        row["residual_fraction_preQRS"] = region_mses["preQRS"] / denom
        row["residual_fraction_QRS"] = region_mses["QRS"] / denom
        row["residual_fraction_STT"] = region_mses["STT"] / denom

        # Individualized regions using QRS boundaries from script 09.
        if features09 is not None:
            q_on = pd.to_numeric(
                pd.Series([features09.iloc[i]["qrs_on_ms"]]),
                errors="coerce"
            ).iloc[0]
            q_off = pd.to_numeric(
                pd.Series([features09.iloc[i]["qrs_off_ms"]]),
                errors="coerce"
            ).iloc[0]

            if np.isfinite(q_on) and np.isfinite(q_off) and q_off > q_on:
                idx_q = np.where((t_ms >= q_on) & (t_ms <= q_off))[0]
                idx_pre = np.where((t_ms >= -300.0) & (t_ms < q_on))[0]
                idx_stt = np.where((t_ms > q_off) & (t_ms <= 500.0))[0]

                if len(idx_q):
                    row["residual_mse_mv2_QRS_individual"] = float(mse(r[:, idx_q]))
                if len(idx_pre):
                    row["residual_mse_mv2_preQRS_individual"] = float(mse(r[:, idx_pre]))
                if len(idx_stt):
                    row["residual_mse_mv2_STT_individual"] = float(mse(r[:, idx_stt]))

                if len(idx_q) and len(idx_stt):
                    row["STT_to_QRS_mse_ratio_individual"] = (
                        row["residual_mse_mv2_STT_individual"]
                        / (row["residual_mse_mv2_QRS_individual"] + EPS)
                    )

        rows.append(row)

        if i == 0 or (i + 1) % 500 == 0 or i == n - 1:
            print(f"[metrics] {i + 1}/{n}")

    return pd.DataFrame(rows)


# =====================================================================
# Group statistics
# =====================================================================

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Efficient Cliff's delta:
      delta = P(x > y) - P(x < y)
    Here x is HCM and y is Normal-test.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    y_sorted = np.sort(y)
    gt = 0
    lt = 0

    for val in x:
        less = np.searchsorted(y_sorted, val, side="left")
        greater = len(y_sorted) - np.searchsorted(y_sorted, val, side="right")
        gt += less
        lt += greater

    return float((gt - lt) / (len(x) * len(y)))


def descriptive_auc(
    normal_values: np.ndarray,
    hcm_values: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    if roc_auc_score is None:
        return np.nan, np.nan, np.nan

    nval = np.asarray(normal_values, dtype=float)
    hval = np.asarray(hcm_values, dtype=float)
    nval = nval[np.isfinite(nval)]
    hval = hval[np.isfinite(hval)]

    if len(nval) < 5 or len(hval) < 5:
        return np.nan, np.nan, np.nan

    y = np.r_[np.zeros(len(nval), dtype=int), np.ones(len(hval), dtype=int)]
    s = np.r_[nval, hval]
    auc = float(roc_auc_score(y, s))

    if n_boot <= 0:
        return auc, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        ns = rng.choice(nval, size=len(nval), replace=True)
        hs = rng.choice(hval, size=len(hval), replace=True)
        yy = np.r_[np.zeros(len(ns), dtype=int), np.ones(len(hs), dtype=int)]
        ss = np.r_[ns, hs]
        boots[b] = roc_auc_score(yy, ss)

    lo, hi = np.quantile(boots, [0.025, 0.975])
    return auc, float(lo), float(hi)


def bh_fdr(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)

    if not np.any(ok):
        return q

    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    temp = np.empty_like(adj)
    temp[order] = adj
    q[ok] = temp
    return q


def make_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        c for c in df.columns
        if c.startswith("residual_")
        or c in [
            "STT_to_QRS_mse_ratio",
            "preQRS_to_QRS_mse_ratio",
            "STT_to_QRS_mse_ratio_individual",
        ]
    ]

    rows = []
    for group, gdf in df.groupby("analysis_group"):
        for metric in metrics:
            vals = pd.to_numeric(gdf[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            rows.append({
                "group": group,
                "metric": metric,
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "median": float(np.median(vals)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
                "q95": float(np.quantile(vals, 0.95)),
            })

    return pd.DataFrame(rows)


def hcm_vs_normal_test_effects(
    df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    ndf = df[df["analysis_group"] == "normal_test"]
    hdf = df[df["analysis_group"] == "hcm"]

    metrics = [
        c for c in df.columns
        if (
            c.startswith("residual_mse_mv2")
            or c.startswith("residual_rmse_mv")
            or c.startswith("residual_fraction_")
            or c in [
                "STT_to_QRS_mse_ratio",
                "preQRS_to_QRS_mse_ratio",
                "STT_to_QRS_mse_ratio_individual",
            ]
        )
    ]

    rows = []
    for j, metric in enumerate(metrics):
        nv = pd.to_numeric(ndf[metric], errors="coerce").to_numpy(dtype=float)
        hv = pd.to_numeric(hdf[metric], errors="coerce").to_numpy(dtype=float)
        nv = nv[np.isfinite(nv)]
        hv = hv[np.isfinite(hv)]

        if len(nv) < 5 or len(hv) < 5:
            continue

        u, p = stats.mannwhitneyu(hv, nv, alternative="two-sided")
        delta = cliffs_delta(hv, nv)

        auc, auc_lo, auc_hi = descriptive_auc(
            normal_values=nv,
            hcm_values=hv,
            n_boot=n_boot,
            seed=seed + j,
        )

        med_n = float(np.median(nv))
        med_h = float(np.median(hv))

        rows.append({
            "metric": metric,
            "n_normal_test": len(nv),
            "n_hcm": len(hv),
            "normal_test_median": med_n,
            "hcm_median": med_h,
            "median_ratio_hcm_over_normal_test": med_h / (med_n + EPS),
            "mannwhitney_u": float(u),
            "p_value": float(p),
            "cliffs_delta_hcm_vs_normal_test": delta,
            "descriptive_auc_hcm_vs_normal_test": auc,
            "auc_ci95_low": auc_lo,
            "auc_ci95_high": auc_hi,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["q_fdr"] = bh_fdr(out["p_value"].to_numpy())
        out["abs_cliffs_delta"] = np.abs(out["cliffs_delta_hcm_vs_normal_test"])
        out = out.sort_values("abs_cliffs_delta", ascending=False).reset_index(drop=True)
    return out


# =====================================================================
# Residual maps / hotspot ranking
# =====================================================================

def mean_maps(
    residual_mv: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    out = {}
    for group in ["normal_train", "normal_test", "hcm"]:
        m = masks[group]
        if not m.any():
            continue
        r = residual_mv[m]
        out[f"{group}_mean_signed_residual_mv"] = np.mean(r, axis=0)
        out[f"{group}_mean_abs_residual_mv"] = np.mean(np.abs(r), axis=0)
        out[f"{group}_rmse_residual_mv"] = np.sqrt(np.mean(r ** 2, axis=0))
    return out


def hotspot_ranking(
    record_df: pd.DataFrame,
) -> pd.DataFrame:
    eff = hcm_vs_normal_test_effects(record_df, n_boot=0, seed=1)
    if eff.empty:
        return eff

    # Keep only physically localizable residual MSE/RMSE metrics.
    loc = eff[
        eff["metric"].str.startswith("residual_mse_mv2_")
        | eff["metric"].str.startswith("residual_rmse_mv_")
    ].copy()

    # Exclude completely global metric.
    loc = loc[
        ~loc["metric"].isin(["residual_mse_mv2", "residual_rmse_mv"])
    ]

    return loc.sort_values(
        ["abs_cliffs_delta", "median_ratio_hcm_over_normal_test"],
        ascending=[False, False],
    ).reset_index(drop=True)


# =====================================================================
# Latent-residual associations
# =====================================================================

def normalize_z_name(c: str) -> str:
    m = re.fullmatch(r"z_?(\d+)", str(c), flags=re.I)
    if not m:
        return c
    return f"z_{int(m.group(1)):02d}"


def load_latents(path: Path, n: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        nc = normalize_z_name(c)
        if nc != c and re.fullmatch(r"z_\d{2}", nc):
            rename[c] = nc
    df = df.rename(columns=rename)

    if len(df) != n:
        raise ValueError(
            f"Latent CSV has {len(df)} rows, HDF5 has {n}. "
            "No row-order assumption beyond exact one-to-one length is accepted."
        )
    return df


def latent_cols(df: pd.DataFrame) -> List[str]:
    out = [
        c for c in df.columns
        if re.fullmatch(r"z_\d+", str(c))
    ]
    return sorted(out, key=lambda z: int(re.findall(r"\d+", z)[0]))


def spearman_safe(x, y) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())

    if n < 10 or np.nanstd(x[ok]) < EPS or np.nanstd(y[ok]) < EPS:
        return np.nan, np.nan, n

    rho, p = stats.spearmanr(x[ok], y[ok])
    return float(rho), float(p), n


def latent_residual_associations(
    lat: pd.DataFrame,
    rec: pd.DataFrame,
    mask: np.ndarray,
    group: str,
) -> pd.DataFrame:
    zcols = latent_cols(lat)

    residual_metrics = [
        "residual_mse_mv2",
        "residual_mse_mv2_preQRS",
        "residual_mse_mv2_QRS",
        "residual_mse_mv2_STT",
        "STT_to_QRS_mse_ratio",
        "residual_fraction_preQRS",
        "residual_fraction_QRS",
        "residual_fraction_STT",
    ]

    # Add lead-region MSE metrics.
    residual_metrics += [
        c for c in rec.columns
        if re.fullmatch(r"residual_mse_mv2_(preQRS|QRS|STT)_.+", c)
    ]

    rows = []
    for z in zcols:
        zv = pd.to_numeric(lat.loc[mask, z], errors="coerce").to_numpy(dtype=float)

        for metric in residual_metrics:
            if metric not in rec.columns:
                continue
            rv = pd.to_numeric(
                rec.loc[mask, metric],
                errors="coerce"
            ).to_numpy(dtype=float)

            rho, p, n = spearman_safe(zv, rv)
            rows.append({
                "group": group,
                "latent": z,
                "residual_metric": metric,
                "n": n,
                "spearman_rho": rho,
                "p_value": p,
            })

    out = pd.DataFrame(rows)
    if len(out):
        out["q_fdr"] = bh_fdr(out["p_value"].to_numpy())
        out["abs_spearman_rho"] = np.abs(out["spearman_rho"])
        out = out.sort_values(
            ["latent", "abs_spearman_rho"],
            ascending=[True, False],
        ).reset_index(drop=True)
    return out


# =====================================================================
# Focused z06 QRS-T uncoupling analysis
# =====================================================================

def z06_analysis(
    lat: pd.DataFrame,
    rec: pd.DataFrame,
    masks: Dict[str, np.ndarray],
) -> pd.DataFrame:
    if "z_06" not in lat.columns:
        return pd.DataFrame()

    metrics = [
        "residual_mse_mv2_QRS",
        "residual_mse_mv2_STT",
        "STT_to_QRS_mse_ratio",
        "residual_fraction_STT",
    ]

    # Include key precordial residuals expected from z06 traversal.
    for lead in ["V2", "V3", "V4", "V5", "V6"]:
        for region in ["QRS", "STT"]:
            c = f"residual_mse_mv2_{region}_{lead}"
            if c in rec.columns:
                metrics.append(c)

    rows = []
    for group in ["normal_train", "normal_test", "hcm"]:
        mask = masks[group]
        if not mask.any():
            continue

        z = pd.to_numeric(lat.loc[mask, "z_06"], errors="coerce").to_numpy(dtype=float)

        for metric in metrics:
            y = pd.to_numeric(rec.loc[mask, metric], errors="coerce").to_numpy(dtype=float)
            rho, p, n = spearman_safe(z, y)

            rows.append({
                "analysis": "z06_vs_residual",
                "group": group,
                "metric": metric,
                "n": n,
                "spearman_rho": rho,
                "p_value": p,
            })

    out = pd.DataFrame(rows)

    # Add high-vs-low z06 descriptive comparison within HCM.
    mh = masks["hcm"]
    z_h = pd.to_numeric(lat.loc[mh, "z_06"], errors="coerce").to_numpy(dtype=float)

    if len(z_h) >= 20:
        q25, q75 = np.nanquantile(z_h, [0.25, 0.75])
        h_idx = np.where(mh)[0]
        low_idx = h_idx[z_h <= q25]
        high_idx = h_idx[z_h >= q75]

        for metric in metrics:
            low = pd.to_numeric(rec.loc[low_idx, metric], errors="coerce").to_numpy(dtype=float)
            high = pd.to_numeric(rec.loc[high_idx, metric], errors="coerce").to_numpy(dtype=float)
            low = low[np.isfinite(low)]
            high = high[np.isfinite(high)]

            if len(low) < 5 or len(high) < 5:
                continue

            u, p = stats.mannwhitneyu(high, low, alternative="two-sided")

            out = pd.concat([
                out,
                pd.DataFrame([{
                    "analysis": "hcm_z06_top_vs_bottom_quartile",
                    "group": "hcm",
                    "metric": metric,
                    "n": len(low) + len(high),
                    "spearman_rho": np.nan,
                    "p_value": float(p),
                    "z06_low_cutoff_q25": float(q25),
                    "z06_high_cutoff_q75": float(q75),
                    "low_z06_median": float(np.median(low)),
                    "high_z06_median": float(np.median(high)),
                    "high_over_low_median_ratio": float(
                        np.median(high) / (np.median(low) + EPS)
                    ),
                    "cliffs_delta_high_vs_low": cliffs_delta(high, low),
                }])
            ], ignore_index=True)

    if len(out):
        out["q_fdr"] = bh_fdr(out["p_value"].to_numpy())

    return out


# =====================================================================
# Figures
# =====================================================================

def plot_region_distribution(df: pd.DataFrame, out: Path) -> None:
    groups = ["normal_test", "hcm"]
    metrics = [
        ("preQRS", "residual_rmse_mv_preQRS"),
        ("QRS", "residual_rmse_mv_QRS"),
        ("ST-T", "residual_rmse_mv_STT"),
    ]

    data = []
    labels = []

    for region, metric in metrics:
        for group in groups:
            vals = pd.to_numeric(
                df.loc[df["analysis_group"] == group, metric],
                errors="coerce"
            ).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            data.append(vals)
            labels.append(f"{region}\n{group}")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Residual RMSE [mV]")
    ax.set_title("Normal-manifold reconstruction residual by temporal region")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_distribution(df: pd.DataFrame, out: Path) -> None:
    data = []
    labels = []

    for group in ["normal_test", "hcm"]:
        vals = pd.to_numeric(
            df.loc[df["analysis_group"] == group, "STT_to_QRS_mse_ratio"],
            errors="coerce"
        ).to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        # Visualization only: cap extreme tail at 99.5th percentile.
        if len(vals):
            cap = np.quantile(vals, 0.995)
            vals = np.minimum(vals, cap)

        data.append(vals)
        labels.append(group)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("ST-T / QRS residual MSE ratio")
    ax.set_title("Relative localization of off-manifold residual")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mean_abs_heatmap(
    arr: np.ndarray,
    t_ms: np.ndarray,
    out: Path,
    title: str,
    symmetric: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    if symmetric:
        vmax = np.nanpercentile(np.abs(arr), 99)
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="upper",
            extent=[t_ms[0], t_ms[-1], 11.5, -0.5],
            vmin=-vmax,
            vmax=vmax,
        )
    else:
        vmax = np.nanpercentile(arr, 99.5)
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="upper",
            extent=[t_ms[0], t_ms[-1], 11.5, -0.5],
            vmin=0,
            vmax=vmax,
        )

    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(LEADS)
    ax.set_xlabel("Time from R reference [ms]")
    ax.set_ylabel("Lead")
    ax.set_title(title)

    for x in [-60, 80]:
        ax.axvline(x, linestyle="--", linewidth=1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Residual [mV]")

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_residual_time_profile(
    residual_mv: np.ndarray,
    masks: Dict[str, np.ndarray],
    t_ms: np.ndarray,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    for group in ["normal_test", "hcm"]:
        m = masks[group]
        if not m.any():
            continue

        # Global mean absolute residual per time sample.
        prof = np.mean(np.abs(residual_mv[m]), axis=(0, 1))
        ax.plot(t_ms, prof, label=group)

    for x in [-60, 80]:
        ax.axvline(x, linestyle="--", linewidth=1)

    ax.set_xlabel("Time from R reference [ms]")
    ax.set_ylabel("Mean absolute residual [mV]")
    ax.set_title("Temporal profile of Normal-manifold reconstruction residual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lead_region_effect_heatmap(
    effects: pd.DataFrame,
    out: Path,
) -> None:
    mat = np.full((12, 3), np.nan)

    for li, lead in enumerate(LEADS):
        for ri, region in enumerate(["preQRS", "QRS", "STT"]):
            metric = f"residual_mse_mv2_{region}_{lead}"
            rr = effects[effects["metric"] == metric]
            if len(rr):
                mat[li, ri] = rr.iloc[0]["cliffs_delta_hcm_vs_normal_test"]

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["pre-QRS", "QRS", "ST-T"])
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(LEADS)
    ax.set_xlabel("Temporal region")
    ax.set_ylabel("Lead")
    ax.set_title("HCM vs Normal-test residual localization")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cliff's delta (positive = larger residual in HCM)")

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_latent_residual_heatmap(
    assoc: pd.DataFrame,
    out: Path,
) -> None:
    if assoc.empty:
        return

    metrics = [
        "residual_mse_mv2_preQRS",
        "residual_mse_mv2_QRS",
        "residual_mse_mv2_STT",
        "STT_to_QRS_mse_ratio",
        "residual_fraction_STT",
    ]
    metrics = [m for m in metrics if m in set(assoc["residual_metric"])]

    zcols = sorted(
        assoc["latent"].unique(),
        key=lambda z: int(re.findall(r"\d+", z)[0])
    )

    mat = np.full((len(metrics), len(zcols)), np.nan)

    for i, metric in enumerate(metrics):
        for j, z in enumerate(zcols):
            rr = assoc[
                (assoc["residual_metric"] == metric)
                & (assoc["latent"] == z)
            ]
            if len(rr):
                mat[i, j] = rr.iloc[0]["spearman_rho"]

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(zcols)))
    ax.set_xticklabels(zcols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Latent coordinate")
    ax.set_title("HCM: association between latent position and off-manifold residual")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spearman rho")

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_z06_stt(
    lat: pd.DataFrame,
    rec: pd.DataFrame,
    masks: Dict[str, np.ndarray],
    out: Path,
) -> None:
    if "z_06" not in lat.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for group in ["normal_train", "hcm"]:
        m = masks[group]
        z = pd.to_numeric(lat.loc[m, "z_06"], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(
            rec.loc[m, "residual_mse_mv2_STT"],
            errors="coerce"
        ).to_numpy(dtype=float)

        ok = np.isfinite(z) & np.isfinite(r)
        if ok.sum() == 0:
            continue

        # Downsample display only for visual clarity.
        idx = np.where(ok)[0]
        if len(idx) > 1500:
            rng = np.random.default_rng(20261004)
            idx = rng.choice(idx, size=1500, replace=False)

        ax.scatter(z[idx], r[idx], s=9, alpha=0.35, label=group)

    ax.set_xlabel("$z_{06}$")
    ax.set_ylabel("ST-T residual MSE [mV$^2$]")
    ax.set_title("$z_{06}$ and deviation from the normal ST-T manifold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)
    figdir = args.outdir / "figures"
    ensure_dir(figdir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)

    print("=" * 80)
    print("10 - Analyze Normal-manifold reconstruction residuals")
    print("=" * 80)
    print(f"HDF5      : {args.h5}")
    print(f"Model     : {args.model}")
    print(f"Latents   : {args.latents}")
    print(f"Features09: {args.features09}")
    print(f"Output    : {args.outdir}")
    print(f"Device    : {device}")
    print()

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------
    x_scaled, x_mv, meta, attrs = load_phase1_h5(args.h5)
    n, nlead, nsamp = x_scaled.shape

    print(f"[data] scaled shape: {x_scaled.shape}")
    print(f"[data] physical shape: {x_mv.shape}")
    print(f"[data] finite scaled: {np.isfinite(x_scaled).mean():.6f}")
    print(f"[data] finite mV    : {np.isfinite(x_mv).mean():.6f}")

    if (nlead, nsamp) != (12, 500):
        raise ValueError(f"Expected [N,12,500], got {x_scaled.shape}")

    mean_mv = float(attrs.get("normal_train_scaling_mean_mv", np.nan))
    std_mv = float(attrs.get("normal_train_scaling_std_mv", np.nan))

    if not np.isfinite(mean_mv) or not np.isfinite(std_mv):
        raise ValueError("Normal-train scaling attributes missing from HDF5.")

    meta_df = metadata_frame(meta, n)
    masks = derive_masks(meta_df)

    print("[groups]")
    for k, m in masks.items():
        print(f"  {k:12s}: {int(m.sum())}")

    expected = {
        "normal_train": 3635,
        "normal_val": 779,
        "normal_test": 780,
        "hcm": 649,
    }
    for k, count in expected.items():
        if masks[k].sum() != count:
            warnings.warn(
                f"Expected {count} records in {k}, found {masks[k].sum()}."
            )

    # ---------------------------------------------------------------
    # Model / deterministic reconstruction
    # ---------------------------------------------------------------
    model = infer_model_from_checkpoint(
        args.model,
        latent_dim=args.latent_dim,
        device=device,
    )

    recon_scaled, mu, logvar = reconstruct_dataset(
        model=model,
        x_scaled=x_scaled,
        batch_size=args.batch_size,
        device=device,
    )

    if recon_scaled.shape != x_scaled.shape:
        raise ValueError(
            f"Reconstruction shape mismatch: {recon_scaled.shape} vs {x_scaled.shape}"
        )

    recon_mv = recon_scaled * std_mv + mean_mv
    residual_mv = x_mv - recon_mv

    print(f"[residual] global physical MSE: {np.mean(residual_mv**2):.6f} mV^2")

    # Cross-check posterior means against saved latent CSV.
    lat = load_latents(args.latents, n)
    zcols = latent_cols(lat)

    if len(zcols) == mu.shape[1]:
        z_saved = lat[zcols].to_numpy(dtype=float)
        dif = np.abs(z_saved - mu)
        print(
            f"[latent cross-check] median |saved mu - reencoded mu| = "
            f"{np.nanmedian(dif):.6g}"
        )
        med_mu_diff = float(np.nanmedian(dif))
        max_mu_diff = float(np.nanmax(dif))
        print(
            f"[latent cross-check] max |saved mu - reencoded mu|    = "
            f"{max_mu_diff:.6g}"
        )

        # Numerical consistency check.
        #
        # Saved posterior means can differ slightly from a fresh forward pass
        # because of serialization / CSV precision / library-level floating-point
        # differences.  Latent SDs are of order 1, therefore discrepancies at
        # 1e-4--1e-3 absolute scale are negligible for the present analysis.
        #
        # Policy:
        #   excellent agreement : median <= 1e-3 and max <= 1e-2
        #   caution             : median <= 5e-3 and max <= 5e-2
        #   incompatible        : larger than the caution bounds
        if med_mu_diff <= 1e-3 and max_mu_diff <= 1e-2:
            print(
                "[latent cross-check] PASS: numerical agreement is excellent "
                "relative to latent SD ~ 1."
            )
        elif med_mu_diff <= 5e-3 and max_mu_diff <= 5e-2:
            warnings.warn(
                "Saved and re-encoded posterior means show small numerical "
                f"differences (median={med_mu_diff:.3g}, max={max_mu_diff:.3g}). "
                "Residual analysis will continue, but this should be documented."
            )
        else:
            raise RuntimeError(
                "Re-encoded posterior means are materially inconsistent with "
                f"the saved latent coordinates (median diff={med_mu_diff:.3g}, "
                f"max diff={max_mu_diff:.3g}). Stopping to prevent residual "
                "analysis with an incompatible model."
            )
    else:
        warnings.warn(
            "Saved latent dimension differs from model output; cross-check skipped."
        )

    # ---------------------------------------------------------------
    # Record-level residual decomposition
    # ---------------------------------------------------------------
    t_ms = time_axis_ms(nsamp, args.fs, args.pre_ms)
    features09 = load_script09_boundaries(args.features09, n)

    rec = compute_record_metrics(
        residual_mv=residual_mv,
        x_mv=x_mv,
        recon_mv=recon_mv,
        metadata=meta_df,
        masks=masks,
        t_ms=t_ms,
        features09=features09,
    )
    rec.to_csv(args.outdir / "residual_record_summary.csv", index=False)

    group_summary = make_group_summary(rec)
    group_summary.to_csv(args.outdir / "residual_group_summary.csv", index=False)

    effects = hcm_vs_normal_test_effects(
        rec,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    effects.to_csv(
        args.outdir / "residual_group_effects_hcm_vs_normal_test.csv",
        index=False,
    )

    hotspots = hotspot_ranking(rec)
    hotspots.to_csv(
        args.outdir / "residual_hotspot_ranking.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Residual maps
    # ---------------------------------------------------------------
    maps = mean_maps(residual_mv, masks)
    np.savez_compressed(
        args.outdir / "mean_residual_maps.npz",
        time_axis_ms=t_ms,
        lead_names=np.asarray(LEADS, dtype=object),
        **maps,
    )

    # ---------------------------------------------------------------
    # Latent-residual associations
    # ---------------------------------------------------------------
    assoc = {}
    for group in ["normal_train", "normal_test", "hcm"]:
        tab = latent_residual_associations(
            lat=lat,
            rec=rec,
            mask=masks[group],
            group=group,
        )
        assoc[group] = tab
        tab.to_csv(
            args.outdir / f"latent_residual_associations_{group}.csv",
            index=False,
        )

    z06 = z06_analysis(lat, rec, masks)
    z06.to_csv(
        args.outdir / "z06_qrst_uncoupling_analysis.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------
    plot_region_distribution(
        rec,
        figdir / "residual_region_distribution.png",
    )
    plot_ratio_distribution(
        rec,
        figdir / "residual_region_ratio_distribution.png",
    )

    if "normal_test_mean_abs_residual_mv" in maps:
        plot_mean_abs_heatmap(
            maps["normal_test_mean_abs_residual_mv"],
            t_ms,
            figdir / "normal_test_mean_abs_residual_heatmap.png",
            "Normal-test mean absolute reconstruction residual",
        )

    if "hcm_mean_abs_residual_mv" in maps:
        plot_mean_abs_heatmap(
            maps["hcm_mean_abs_residual_mv"],
            t_ms,
            figdir / "hcm_mean_abs_residual_heatmap.png",
            "HCM mean absolute reconstruction residual",
        )

    if (
        "normal_test_mean_abs_residual_mv" in maps
        and "hcm_mean_abs_residual_mv" in maps
    ):
        diff = (
            maps["hcm_mean_abs_residual_mv"]
            - maps["normal_test_mean_abs_residual_mv"]
        )
        plot_mean_abs_heatmap(
            diff,
            t_ms,
            figdir / "hcm_minus_normal_test_abs_residual_heatmap.png",
            "Excess HCM absolute residual relative to Normal-test",
            symmetric=True,
        )

    plot_lead_region_effect_heatmap(
        effects,
        figdir / "lead_region_effect_heatmap.png",
    )
    plot_residual_time_profile(
        residual_mv,
        masks,
        t_ms,
        figdir / "residual_time_profile.png",
    )

    if not assoc["hcm"].empty:
        plot_latent_residual_heatmap(
            assoc["hcm"],
            figdir / "latent_residual_heatmap_hcm.png",
        )

    plot_z06_stt(
        lat,
        rec,
        masks,
        figdir / "z06_stt_residual.png",
    )

    # ---------------------------------------------------------------
    # Compact headline results for JSON
    # ---------------------------------------------------------------
    def get_effect(metric: str) -> Optional[Dict[str, float]]:
        rr = effects[effects["metric"] == metric]
        if rr.empty:
            return None
        r = rr.iloc[0]
        return {
            "normal_test_median": float(r["normal_test_median"]),
            "hcm_median": float(r["hcm_median"]),
            "median_ratio_hcm_over_normal_test": float(
                r["median_ratio_hcm_over_normal_test"]
            ),
            "cliffs_delta": float(r["cliffs_delta_hcm_vs_normal_test"]),
            "descriptive_auc": float(r["descriptive_auc_hcm_vs_normal_test"])
            if np.isfinite(r["descriptive_auc_hcm_vs_normal_test"]) else None,
            "auc_ci95_low": float(r["auc_ci95_low"])
            if np.isfinite(r["auc_ci95_low"]) else None,
            "auc_ci95_high": float(r["auc_ci95_high"])
            if np.isfinite(r["auc_ci95_high"]) else None,
            "q_fdr": float(r["q_fdr"]) if np.isfinite(r["q_fdr"]) else None,
        }

    headline = {
        "overall_mse": get_effect("residual_mse_mv2"),
        "preQRS_mse": get_effect("residual_mse_mv2_preQRS"),
        "QRS_mse": get_effect("residual_mse_mv2_QRS"),
        "STT_mse": get_effect("residual_mse_mv2_STT"),
        "STT_to_QRS_ratio": get_effect("STT_to_QRS_mse_ratio"),
    }

    # z06 headline correlations
    z06_headline = {}
    if len(z06):
        zz = z06[
            (z06["analysis"] == "z06_vs_residual")
            & (z06["metric"].isin([
                "residual_mse_mv2_QRS",
                "residual_mse_mv2_STT",
                "STT_to_QRS_mse_ratio",
            ]))
        ]
        for _, r in zz.iterrows():
            z06_headline[f'{r["group"]}:{r["metric"]}'] = {
                "spearman_rho": None if not np.isfinite(r["spearman_rho"])
                else float(r["spearman_rho"]),
                "q_fdr": None if not np.isfinite(r["q_fdr"])
                else float(r["q_fdr"]),
            }

    # Top hotspots
    top_hotspots = []
    for _, r in hotspots.head(15).iterrows():
        top_hotspots.append({
            "metric": str(r["metric"]),
            "normal_test_median": float(r["normal_test_median"]),
            "hcm_median": float(r["hcm_median"]),
            "median_ratio": float(r["median_ratio_hcm_over_normal_test"]),
            "cliffs_delta": float(r["cliffs_delta_hcm_vs_normal_test"]),
        })

    run_summary = {
        "script": "10_analyze_normal_manifold_residuals.py",
        "version": "1.3",
        "scientific_role": "Localization of morphology not explained by the frozen Normal ECG manifold",
        "input_h5": str(args.h5),
        "input_model": str(args.model),
        "input_latents": str(args.latents),
        "input_features09": str(args.features09),
        "n_records": int(n),
        "groups": {k: int(v.sum()) for k, v in masks.items()},
        "input_shape": list(x_scaled.shape),
        "sampling_rate_hz": float(args.fs),
        "beat_start_ms": float(args.pre_ms),
        "normal_train_scaling_mean_mv": mean_mv,
        "normal_train_scaling_std_mv": std_mv,
        "reconstruction": "deterministic decoder(posterior mean mu)",
        "fixed_regions_ms": {
            k: [float(v[0]), float(v[1])]
            for k, v in FIXED_REGIONS_MS.items()
        },
        "individual_qrs_regions_available": bool(features09 is not None),
        "headline_hcm_vs_normal_test": headline,
        "z06_qrst_uncoupling_headline": z06_headline,
        "top_residual_hotspots": top_hotspots,
        "cautions": [
            "HCM and Normal ECGs originate from different source datasets; residual group separation may contain acquisition/domain effects.",
            "AUROC is reported only as a descriptive effect-size-like separation measure and not as validated diagnostic performance.",
            "Fixed temporal regions are used for primary cross-record comparability.",
            "Script-09 per-record QRS boundaries are used only for supplemental individualized residual metrics when available.",
            "Residual localization does not by itself identify the biological substrate; associations with imaging, genotype and clinical variables are required."
        ],
    }

    with open(args.outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("DONE")
    print(f"Results: {args.outdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
