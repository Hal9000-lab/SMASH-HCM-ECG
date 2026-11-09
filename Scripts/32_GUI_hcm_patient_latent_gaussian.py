#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
32_hcm_patient_latent_gaussian_gui.py

SMASH-HCM utility GUI: visualize one baseline HCM median beat in either
(1) the frozen Normal-anchored VAE latent space (16 dimensions) or
(2) the frozen HCM-oriented static VAE latent space (8 dimensions).

For every latent coordinate the GUI shows the Gaussian fitted to the empirical
training posterior means for the corresponding model and marks the selected HCM
patient's deterministic posterior mean. This is a marginal-coordinate view, not
a statement that the joint latent distribution is diagonal Gaussian.

Patient waveform source
-----------------------
The patient selector uses the HCM baseline records contained in the Step-11
HCM-static prepared HDF5 (one baseline ECG per retained HCM subject). The
physical median beat in mV is rescaled with the training-only global scaling of
the model selected in the GUI and projected through the corresponding frozen
encoder.

Default canonical inputs
------------------------
Normal model:
    Results/06_Phase1_MedianBeat_Beta_VAE/best_model.pt
    Results/06_Phase1_MedianBeat_Beta_VAE/phase1_medianbeat_latents.csv
    Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5

HCM model:
    Results/12_HCM_Static_Beta_VAE/best_model.pt
    Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv
    Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5

Legacy path fallbacks are supported automatically.

Requirements
------------
pip install numpy pandas h5py torch matplotlib PyQt5

Run
---
python Scripts/32_hcm_patient_latent_gaussian_gui.py

Notes
-----
- Normal VAE latent dimensionality: 16.
- HCM-oriented VAE latent dimensionality: 8.
- The point shown for the patient is the posterior mean mu_phi(X).
- The optional patient posterior interval uses exp(0.5*logvar).
- Gaussian reference curves are fitted independently to each coordinate from
  normal_train (Normal VAE) or hcm_train (HCM VAE) posterior means.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# =============================================================================
# Configuration
# =============================================================================

NORMAL_DIM = 16
HCM_DIM = 8
EPS = 1e-12

MODEL_NORMAL = "Normal-anchored VAE (16D)"
MODEL_HCM = "HCM-oriented VAE (8D)"


# =============================================================================
# Paths and generic utilities
# =============================================================================


def infer_project_root() -> Path:
    try:
        here = Path(__file__).resolve()
        candidates = [here.parent, *here.parents]
    except NameError:
        candidates = []

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if (p / "Data").exists() and (p / "Results").exists():
            return p

    raise RuntimeError(
        "Cannot locate SMASH-HCM project root; expected Data/ and Results/."
    )


def first_existing(
    root: Path,
    relative_candidates: Sequence[str],
    glob_patterns: Sequence[str] = (),
) -> Path:
    tried = []
    for rel in relative_candidates:
        p = root / rel
        tried.append(p)
        if p.exists():
            return p

    for pattern in glob_patterns:
        hits = sorted(root.glob(pattern))
        for p in hits:
            if p.is_file():
                return p

    msg = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(f"Required file not found. Tried:\n{msg}")


def decode_array(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind not in ("S", "O", "U"):
        return a

    out = []
    for x in a.ravel():
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object).reshape(a.shape)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def latent_columns(df: pd.DataFrame, expected_dim: int) -> list[str]:
    cols = [
        c for c in df.columns
        if c.startswith("z_") and c[2:].isdigit()
    ]
    cols = sorted(cols, key=lambda c: int(c.split("_")[1]))
    if len(cols) < expected_dim:
        raise ValueError(
            f"Expected at least {expected_dim} latent columns, found {len(cols)}."
        )
    return cols[:expected_dim]


def gaussian_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    sd = max(float(sd), EPS)
    z = (x - float(mu)) / sd
    return np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


# =============================================================================
# Exact Normal VAE architecture (Step 06)
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
                nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
                nn.SiLU(),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MedianBeatBetaVAE(nn.Module):
    """Exact Normal-anchored beta-VAE architecture from Step 06."""

    def __init__(self, latent_dim: int = NORMAL_DIM):
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


# =============================================================================
# Exact HCM-static VAE architecture (Step 12)
# =============================================================================


class HCMStaticBetaVAE(nn.Module):
    """Exact HCM-oriented static beta-VAE architecture from Step 12."""

    def __init__(self, latent_dim: int = HCM_DIM):
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
# Data containers
# =============================================================================


@dataclass
class PatientRecord:
    h5_index: int
    subject_id: str
    ecg_id: str
    split_name: str
    label: str


@dataclass
class ModelBundle:
    label: str
    model: nn.Module
    latent_dim: int
    reference_name: str
    latent_mean: np.ndarray
    latent_sd: np.ndarray
    n_reference: int
    scale_mean_mv: float
    scale_std_mv: float
    checkpoint: Path
    latent_csv: Path


# =============================================================================
# Project assets
# =============================================================================


class ProjectAssets:
    def __init__(self, root: Path, device: torch.device):
        self.root = root
        self.device = device

        self.hcm_h5 = first_existing(
            root,
            [
                "Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5",
                "Results/11_HCM_Static_VAE_Preparation/phase2_hcmstatic_preprocessed.h5",
                "Data/Harmonized/Phase2_HCMStatic_preprocessed.h5",
            ],
            glob_patterns=(
                "Results/11_HCM_Static_VAE_Preparation/*HCMStatic*.h5",
                "Results/11_HCM_Static_VAE_Preparation/*.h5",
            ),
        )

        self.normal_h5 = first_existing(
            root,
            [
                "Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5",
                "Results/05_Phase1_MedianBeat_VAE/phase1_medianbeat_normalanchored_preprocessed.h5",
                "Data/Harmonized/Phase1_MedianBeat_NormalAnchored_preprocessed.h5",
            ],
            glob_patterns=(
                "Results/05_Phase1_MedianBeat_VAE/*preprocessed*.h5",
                "Results/05_Phase1_MedianBeat_VAE/*.h5",
            ),
        )

        self.normal_ckpt = first_existing(
            root,
            [
                "Results/06_Phase1_MedianBeat_Beta_VAE/best_model.pt",
                "Results/Phase1_MedianBeat_VAE/best_model.pt",
            ],
            glob_patterns=("Results/06_Phase1_MedianBeat_Beta_VAE/*model*.pt",),
        )
        self.normal_latents = first_existing(
            root,
            [
                "Results/06_Phase1_MedianBeat_Beta_VAE/phase1_medianbeat_latents.csv",
                "Results/Phase1_MedianBeat_VAE/phase1_medianbeat_latents.csv",
            ],
            glob_patterns=("Results/06_Phase1_MedianBeat_Beta_VAE/*latent*.csv",),
        )

        self.hcm_ckpt = first_existing(
            root,
            [
                "Results/12_HCM_Static_Beta_VAE/best_model.pt",
                "Results/Phase2_HCMStatic_VAE/best_model.pt",
            ],
            glob_patterns=("Results/12_HCM_Static_Beta_VAE/*model*.pt",),
        )
        self.hcm_latents = first_existing(
            root,
            [
                "Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv",
                "Results/Phase2_HCMStatic_VAE/hcm_static_latents.csv",
            ],
            glob_patterns=("Results/12_HCM_Static_Beta_VAE/*latent*.csv",),
        )

        self.patient_records: list[PatientRecord] = []
        self.patient_beats_mv: np.ndarray
        self._load_patients()

        self.bundles: Dict[str, ModelBundle] = {}

    @staticmethod
    def _find_1d_dataset(h5: h5py.File, names: Iterable[str], n: int) -> Optional[np.ndarray]:
        for name in names:
            if name in h5 and isinstance(h5[name], h5py.Dataset):
                a = decode_array(h5[name][:])
                if a.ndim == 1 and len(a) == n:
                    return a
            path = f"metadata/{name}"
            if path in h5 and isinstance(h5[path], h5py.Dataset):
                a = decode_array(h5[path][:])
                if a.ndim == 1 and len(a) == n:
                    return a
        return None

    def _load_patients(self):
        with h5py.File(self.hcm_h5, "r") as h5:
            if "median_all_mv" in h5:
                beats = np.asarray(h5["median_all_mv"][:], dtype=np.float32)
            elif "signals/median_all_mv" in h5:
                beats = np.asarray(h5["signals/median_all_mv"][:], dtype=np.float32)
            else:
                raise KeyError(
                    f"No median_all_mv dataset found in {self.hcm_h5}."
                )

            if beats.ndim != 3 or tuple(beats.shape[1:]) != (12, 500):
                raise ValueError(
                    f"Expected HCM median beats [N,12,500], found {beats.shape}."
                )
            n = len(beats)

            if "is_hcm" in h5:
                is_hcm = np.asarray(h5["is_hcm"][:]).astype(bool)
            else:
                split = self._find_1d_dataset(h5, ["split_name"], n)
                if split is None:
                    is_hcm = np.ones(n, dtype=bool)
                else:
                    is_hcm = np.asarray(
                        [str(x).lower() != "normal_eval" for x in split],
                        dtype=bool,
                    )

            subject = self._find_1d_dataset(
                h5,
                ["subject_id", "subject_key", "analysis_subject_key", "patient_id"],
                n,
            )
            ecg = self._find_1d_dataset(h5, ["ecg_id", "record_id"], n)
            split = self._find_1d_dataset(h5, ["split_name"], n)

            if subject is None:
                subject = np.asarray([f"HCM_{i:04d}" for i in range(n)], dtype=object)
            if ecg is None:
                ecg = np.asarray([str(i) for i in range(n)], dtype=object)
            if split is None:
                split = np.asarray(["HCM"] * n, dtype=object)

            idx = np.where(is_hcm)[0]
            self.patient_beats_mv = beats[idx].copy()
            self.patient_records = []

            for local_i, h5_i in enumerate(idx):
                sid = str(subject[h5_i]).strip()
                eid = str(ecg[h5_i]).strip()
                sp = str(split[h5_i]).strip()
                display = f"{sid} | ECG {eid} | {sp}"
                self.patient_records.append(
                    PatientRecord(
                        h5_index=int(h5_i),
                        subject_id=sid,
                        ecg_id=eid,
                        split_name=sp,
                        label=display,
                    )
                )

        # Deterministic ordering by subject then ECG, preserving beat alignment.
        order = sorted(
            range(len(self.patient_records)),
            key=lambda i: (
                self.patient_records[i].subject_id,
                self.patient_records[i].ecg_id,
            ),
        )
        self.patient_records = [self.patient_records[i] for i in order]
        self.patient_beats_mv = self.patient_beats_mv[np.asarray(order, dtype=int)]

    def get_bundle(self, model_label: str) -> ModelBundle:
        if model_label in self.bundles:
            return self.bundles[model_label]

        if model_label == MODEL_NORMAL:
            bundle = self._load_normal_bundle()
        elif model_label == MODEL_HCM:
            bundle = self._load_hcm_bundle()
        else:
            raise ValueError(f"Unknown model: {model_label}")

        self.bundles[model_label] = bundle
        return bundle

    def _load_normal_bundle(self) -> ModelBundle:
        ckpt = torch_load(self.normal_ckpt, self.device)
        latent_dim = int(ckpt.get("latent_dim", NORMAL_DIM))
        if latent_dim != NORMAL_DIM:
            raise ValueError(
                f"Normal checkpoint latent_dim={latent_dim}, expected {NORMAL_DIM}."
            )

        model = MedianBeatBetaVAE(latent_dim=latent_dim).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        df = pd.read_csv(self.normal_latents)
        zcols = latent_columns(df, latent_dim)
        if "split_name" not in df.columns:
            raise ValueError("Normal latent CSV has no split_name column.")
        ref = df[df["split_name"].astype(str).eq("normal_train")]
        if len(ref) == 0:
            raise ValueError("No normal_train records in Normal latent CSV.")
        z = ref[zcols].to_numpy(dtype=float)

        with h5py.File(self.normal_h5, "r") as h5:
            mean = float(h5.attrs["normal_train_scaling_mean_mv"])
            sd = float(h5.attrs["normal_train_scaling_std_mv"])

        return ModelBundle(
            label=MODEL_NORMAL,
            model=model,
            latent_dim=latent_dim,
            reference_name="Normal-train",
            latent_mean=np.mean(z, axis=0),
            latent_sd=np.std(z, axis=0, ddof=1),
            n_reference=len(z),
            scale_mean_mv=mean,
            scale_std_mv=sd,
            checkpoint=self.normal_ckpt,
            latent_csv=self.normal_latents,
        )

    def _load_hcm_bundle(self) -> ModelBundle:
        ckpt = torch_load(self.hcm_ckpt, self.device)
        latent_dim = int(ckpt.get("latent_dim", HCM_DIM))
        if latent_dim != HCM_DIM:
            raise ValueError(
                f"HCM checkpoint latent_dim={latent_dim}, expected {HCM_DIM}."
            )

        model = HCMStaticBetaVAE(latent_dim=latent_dim).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        df = pd.read_csv(self.hcm_latents)
        zcols = latent_columns(df, latent_dim)
        if "split_name" not in df.columns:
            raise ValueError("HCM latent CSV has no split_name column.")
        ref = df[df["split_name"].astype(str).eq("hcm_train")]
        if len(ref) == 0:
            raise ValueError("No hcm_train records in HCM latent CSV.")
        z = ref[zcols].to_numpy(dtype=float)

        with h5py.File(self.hcm_h5, "r") as h5:
            mean = float(h5.attrs["hcm_train_scaling_mean_mv"])
            sd = float(h5.attrs["hcm_train_scaling_std_mv"])

        return ModelBundle(
            label=MODEL_HCM,
            model=model,
            latent_dim=latent_dim,
            reference_name="HCM-train",
            latent_mean=np.mean(z, axis=0),
            latent_sd=np.std(z, axis=0, ddof=1),
            n_reference=len(z),
            scale_mean_mv=mean,
            scale_std_mv=sd,
            checkpoint=self.hcm_ckpt,
            latent_csv=self.hcm_latents,
        )

    def encode_patient(
        self,
        patient_local_index: int,
        bundle: ModelBundle,
    ) -> Tuple[np.ndarray, np.ndarray]:
        beat_mv = self.patient_beats_mv[int(patient_local_index)].astype(np.float32)
        x = (beat_mv - bundle.scale_mean_mv) / bundle.scale_std_mv
        if not np.isfinite(x).all():
            raise ValueError("Selected patient median beat contains NaN/Inf after scaling.")

        xt = torch.from_numpy(x[None, :, :]).float().to(self.device)
        with torch.no_grad():
            mu, logvar = bundle.model.encode(xt)
        return (
            mu.detach().cpu().numpy()[0].astype(float),
            logvar.detach().cpu().numpy()[0].astype(float),
        )


# =============================================================================
# Matplotlib canvas
# =============================================================================


class LatentGaussianCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(14.0, 8.8), constrained_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def draw_latents(
        self,
        bundle: ModelBundle,
        patient_mu: np.ndarray,
        patient_logvar: np.ndarray,
        standardized_axis: bool,
        show_posterior_interval: bool,
        patient_label: str,
    ) -> None:
        self.fig.clear()

        dim = bundle.latent_dim
        ncols = 4
        nrows = int(math.ceil(dim / ncols))
        axes = self.fig.subplots(nrows, ncols, squeeze=False)
        flat_axes = axes.ravel()

        ref_mu = bundle.latent_mean
        ref_sd = np.maximum(bundle.latent_sd, EPS)
        patient_z = (patient_mu - ref_mu) / ref_sd
        patient_post_sd = np.exp(0.5 * patient_logvar)

        for k in range(dim):
            ax = flat_axes[k]
            zscore = float(patient_z[k])

            if standardized_axis:
                p = zscore
                p_sigma = float(patient_post_sd[k] / ref_sd[k])
                lo = min(-4.0, p - 0.75)
                hi = max(+4.0, p + 0.75)
                x = np.linspace(lo, hi, 500)
                y = gaussian_pdf(x, 0.0, 1.0)
                xlabel = "SD from training mean"
                ref_text = f"raw mean={ref_mu[k]:+.3f}, SD={ref_sd[k]:.3f}"
            else:
                p = float(patient_mu[k])
                p_sigma = float(patient_post_sd[k])
                lo = min(ref_mu[k] - 4.0 * ref_sd[k], p - 0.75 * ref_sd[k])
                hi = max(ref_mu[k] + 4.0 * ref_sd[k], p + 0.75 * ref_sd[k])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = p - 1.0, p + 1.0
                x = np.linspace(lo, hi, 500)
                y = gaussian_pdf(x, ref_mu[k], ref_sd[k])
                xlabel = "Latent coordinate"
                ref_text = f"reference mean={ref_mu[k]:+.3f}, SD={ref_sd[k]:.3f}"

            ax.plot(x, y, linewidth=1.8, label=f"{bundle.reference_name} Gaussian")
            ax.fill_between(x, 0.0, y, alpha=0.12)
            ax.axvline(p, linewidth=1.7, linestyle="--", label="Patient posterior mean")

            y_patient = float(np.interp(p, x, y)) if lo <= p <= hi else 0.0
            y_marker = max(y_patient, float(np.max(y)) * 0.035)
            ax.scatter([p], [y_marker], s=38, zorder=5)

            if show_posterior_interval and np.isfinite(p_sigma) and p_sigma > 0:
                ybar = float(np.max(y)) * 0.10
                ax.hlines(ybar, p - p_sigma, p + p_sigma, linewidth=3.0, alpha=0.75)
                ax.vlines(
                    [p - p_sigma, p + p_sigma],
                    ybar * 0.86,
                    ybar * 1.14,
                    linewidth=1.2,
                    alpha=0.75,
                )

            ax.set_title(
                f"z_{k:02d}   patient {zscore:+.2f} SD",
                fontsize=10,
                fontweight="bold" if abs(zscore) >= 2.0 else "normal",
            )
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.18)
            ax.text(
                0.02,
                0.96,
                ref_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                alpha=0.75,
            )

        for k in range(dim, len(flat_axes)):
            flat_axes[k].set_visible(False)

        max_abs = float(np.max(np.abs(patient_z)))
        n_gt2 = int(np.sum(np.abs(patient_z) >= 2.0))
        self.fig.suptitle(
            f"{bundle.label} | {patient_label}\n"
            f"Reference: {bundle.reference_name} posterior means (n={bundle.n_reference:,}); "
            f"patient point = posterior mean | max |z|={max_abs:.2f} SD; |z|>=2 in {n_gt2}/{dim} dimensions",
            fontsize=12,
        )
        self.draw_idle()


# =============================================================================
# Main GUI
# =============================================================================


class PatientLatentExplorer(QMainWindow):
    def __init__(self, assets: ProjectAssets):
        super().__init__()
        self.assets = assets
        self.patient_mu: Optional[np.ndarray] = None
        self.patient_logvar: Optional[np.ndarray] = None
        self.current_bundle: Optional[ModelBundle] = None

        self.setWindowTitle("SMASH-HCM | HCM Patient Latent Gaussian GUI")
        self.resize(1500, 980)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        controls = QGroupBox("Selection")
        grid = QGridLayout(controls)

        grid.addWidget(QLabel("Latent model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([MODEL_NORMAL, MODEL_HCM])
        grid.addWidget(self.model_combo, 0, 1)

        grid.addWidget(QLabel("HCM patient / baseline ECG:"), 0, 2)
        self.patient_combo = QComboBox()
        self.patient_combo.setEditable(True)
        self.patient_combo.setInsertPolicy(QComboBox.NoInsert)
        self.patient_combo.setMinimumWidth(430)
        for i, rec in enumerate(self.assets.patient_records):
            self.patient_combo.addItem(rec.label, i)
        completer = self.patient_combo.completer()
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        grid.addWidget(self.patient_combo, 0, 3, 1, 3)

        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.save_btn = QPushButton("Save figure")
        grid.addWidget(self.prev_btn, 1, 0)
        grid.addWidget(self.next_btn, 1, 1)
        grid.addWidget(self.save_btn, 1, 2)

        self.standardized_check = QCheckBox("Show x-axis in training SD units")
        self.standardized_check.setChecked(True)
        grid.addWidget(self.standardized_check, 1, 3)

        self.posterior_check = QCheckBox("Show patient posterior ±1σ")
        self.posterior_check.setChecked(False)
        grid.addWidget(self.posterior_check, 1, 4)

        self.reload_btn = QPushButton("Re-project")
        grid.addWidget(self.reload_btn, 1, 5)

        layout.addWidget(controls)

        info = QGroupBox("Current projection")
        info_layout = QGridLayout(info)
        self.info_model = QLabel("-")
        self.info_patient = QLabel("-")
        self.info_reference = QLabel("-")
        self.info_summary = QLabel("-")
        self.info_paths = QLabel("-")
        self.info_paths.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_paths.setWordWrap(True)

        info_layout.addWidget(QLabel("Model:"), 0, 0)
        info_layout.addWidget(self.info_model, 0, 1)
        info_layout.addWidget(QLabel("Patient:"), 0, 2)
        info_layout.addWidget(self.info_patient, 0, 3)
        info_layout.addWidget(QLabel("Reference:"), 1, 0)
        info_layout.addWidget(self.info_reference, 1, 1)
        info_layout.addWidget(QLabel("Latent summary:"), 1, 2)
        info_layout.addWidget(self.info_summary, 1, 3)
        info_layout.addWidget(QLabel("Assets:"), 2, 0)
        info_layout.addWidget(self.info_paths, 2, 1, 1, 3)
        layout.addWidget(info)

        self.canvas = LatentGaussianCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        note = QLabel(
            "Each panel is a marginal Gaussian fitted to the training posterior means of one latent coordinate. "
            "The patient marker is the deterministic posterior mean μφ(X); the panels do not imply independence of the joint latent coordinates."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        layout.addWidget(note)

        self.model_combo.currentIndexChanged.connect(self.refresh_projection)
        self.patient_combo.currentIndexChanged.connect(self.refresh_projection)
        self.standardized_check.stateChanged.connect(self.redraw_only)
        self.posterior_check.stateChanged.connect(self.redraw_only)
        self.prev_btn.clicked.connect(self.previous_patient)
        self.next_btn.clicked.connect(self.next_patient)
        self.reload_btn.clicked.connect(self.refresh_projection)
        self.save_btn.clicked.connect(self.save_figure)

        self.refresh_projection()

    def _selected_patient_local_index(self) -> Optional[int]:
        idx = self.patient_combo.currentIndex()
        if idx < 0:
            return None
        data = self.patient_combo.itemData(idx)
        if data is None:
            return None
        return int(data)

    def refresh_projection(self):
        patient_idx = self._selected_patient_local_index()
        if patient_idx is None:
            return

        model_label = self.model_combo.currentText()
        try:
            bundle = self.assets.get_bundle(model_label)
            mu, logvar = self.assets.encode_patient(patient_idx, bundle)
        except Exception as exc:
            QMessageBox.critical(self, "Projection error", str(exc))
            return

        self.current_bundle = bundle
        self.patient_mu = mu
        self.patient_logvar = logvar

        rec = self.assets.patient_records[patient_idx]
        zscore = (mu - bundle.latent_mean) / np.maximum(bundle.latent_sd, EPS)
        self.info_model.setText(f"{bundle.label} | latent dim={bundle.latent_dim}")
        self.info_patient.setText(
            f"subject={rec.subject_id} | ECG={rec.ecg_id} | split={rec.split_name}"
        )
        self.info_reference.setText(
            f"{bundle.reference_name}, n={bundle.n_reference:,}, Gaussian fitted per latent coordinate"
        )
        self.info_summary.setText(
            f"max |z-score|={np.max(np.abs(zscore)):.2f}; "
            f"mean |z-score|={np.mean(np.abs(zscore)):.2f}; "
            f"dimensions |z|>=2: {np.sum(np.abs(zscore) >= 2)}/{bundle.latent_dim}"
        )
        self.info_paths.setText(
            f"checkpoint: {bundle.checkpoint} | latent reference: {bundle.latent_csv} | patient H5: {self.assets.hcm_h5}"
        )
        self.redraw_only()

    def redraw_only(self):
        patient_idx = self._selected_patient_local_index()
        if (
            patient_idx is None
            or self.current_bundle is None
            or self.patient_mu is None
            or self.patient_logvar is None
        ):
            return
        rec = self.assets.patient_records[patient_idx]
        patient_label = f"subject {rec.subject_id}, ECG {rec.ecg_id}"
        self.canvas.draw_latents(
            bundle=self.current_bundle,
            patient_mu=self.patient_mu,
            patient_logvar=self.patient_logvar,
            standardized_axis=self.standardized_check.isChecked(),
            show_posterior_interval=self.posterior_check.isChecked(),
            patient_label=patient_label,
        )

    def previous_patient(self):
        idx = self.patient_combo.currentIndex()
        if idx > 0:
            self.patient_combo.setCurrentIndex(idx - 1)

    def next_patient(self):
        idx = self.patient_combo.currentIndex()
        if 0 <= idx < self.patient_combo.count() - 1:
            self.patient_combo.setCurrentIndex(idx + 1)

    def save_figure(self):
        patient_idx = self._selected_patient_local_index()
        if patient_idx is None or self.current_bundle is None:
            return
        rec = self.assets.patient_records[patient_idx]
        model_tag = "normal16" if self.current_bundle.label == MODEL_NORMAL else "hcm8"
        suggested = f"latent_gaussians_{model_tag}_subject_{rec.subject_id}_ecg_{rec.ecg_id}.png"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save latent Gaussian figure",
            suggested,
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not filename:
            return
        try:
            self.canvas.fig.savefig(filename, dpi=220, bbox_inches="tight")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))


# =============================================================================
# Entry point
# =============================================================================


def main():
    root = infer_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("SMASH-HCM patient latent Gaussian explorer")
    print("=" * 80)
    print("Project root:", root)
    print("Device      :", device)

    assets = ProjectAssets(root=root, device=device)
    print("HCM patient H5:", assets.hcm_h5)
    print("HCM baseline patients:", len(assets.patient_records))
    print("Normal checkpoint:", assets.normal_ckpt)
    print("HCM checkpoint   :", assets.hcm_ckpt)

    app = QApplication(sys.argv)
    window = PatientLatentExplorer(assets)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
