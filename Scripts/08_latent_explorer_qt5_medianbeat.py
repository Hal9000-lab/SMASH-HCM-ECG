#!/usr/bin/env python
"""
08_latent_explorer_qt5_medianbeat.py

SMASH-HCM -- Interactive latent-space explorer for the
normal-anchored median-beat beta-VAE.

Features
--------
- Select latent dimension z_k from a dropdown.
- Move z_k continuously in Normal-train SD units.
- Optionally activate a second latent dimension and vary it independently.
- Keep all remaining latent dimensions fixed at the empirical
  Normal-train posterior-mean centroid.
- Decode the synthetic 12-lead median beat with the frozen decoder.
- Display all 12 leads or one lead enlarged.
- Physical time axis: -350 ms to +650 ms relative to the R/QRS fiducial.
- Compare the current decoded beat with the 0-SD Normal centroid reference.
- Show current latent values and global waveform RMS change.
- Save the current figure as PNG.

Methodological constraint
-------------------------
Latent means and SDs are computed ONLY from normal_train records.
No HCM ECG contributes to the latent reference coordinate system.

Default inputs
--------------
Results/06_Phase1_MedianBeat_Beta_VAE/best_model.pt
Results/06_Phase1_MedianBeat_Beta_VAE/phase1_medianbeat_latents.csv
Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5

Default figure output
---------------------
Figures saved from the GUI default to:
    Results/08_Phase1_MedianBeat_LatentExplorer/

Requirements
------------
pip install torch numpy pandas h5py matplotlib PyQt5

Run
---
python Scripts/08_latent_explorer_qt5_medianbeat.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


SEED = 20261004
FS_HZ = 500.0
N_SAMPLES = 500
PRE_MS = 350.0
POST_MS = 650.0
LATENT_DIM_DEFAULT = 16

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
    """Exact architecture used by 06_train_phase1_medianbeat_beta_vae.py."""

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
            DeconvBlock(
                128,
                96,
                output_padding=0,
                activate=True,
            ),
            DeconvBlock(
                96,
                64,
                output_padding=1,
                activate=True,
            ),
            DeconvBlock(
                64,
                32,
                output_padding=0,
                activate=True,
            ),
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

        std = torch.exp(
            0.5 * logvar
        )

        z = (
            mu
            + torch.randn_like(std)
            * std
        )

        return (
            self.decode(z),
            mu,
            logvar,
        )


def load_normal_latent_stats(
    csv_path: Path,
    latent_dim: int,
):
    df = pd.read_csv(
        csv_path
    )

    if "split_name" not in df.columns:
        raise ValueError(
            "Latent CSV has no split_name column."
        )

    normal_train = df[
        df[
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

    zcols = [
        f"z_{k:02d}"
        for k in range(
            latent_dim
        )
    ]

    missing = [
        c
        for c in zcols
        if c not in normal_train.columns
    ]

    if missing:
        raise ValueError(
            "Missing latent columns: "
            + ", ".join(missing)
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
            "Invalid latent statistics."
        )

    if np.any(
        sd <= 0
    ):
        raise ValueError(
            "At least one latent dimension has SD <= 0."
        )

    return (
        mu,
        sd,
        len(normal_train),
    )


def load_scaling(
    h5_path: Path,
):
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Preprocessed HDF5 not found: {h5_path}"
        )

    with h5py.File(
        h5_path,
        "r",
    ) as h5:

        mean_path = (
            "preprocessing/global_mean_mv"
        )

        std_path = (
            "preprocessing/global_std_mv"
        )

        if (
            mean_path not in h5
            or std_path not in h5
        ):
            raise KeyError(
                "Median-beat global scaling not found in HDF5."
            )

        mean_mv = float(
            h5[
                mean_path
            ][()]
        )

        std_mv = float(
            h5[
                std_path
            ][()]
        )

    if std_mv <= 0:
        raise ValueError(
            "Invalid global scaling std."
        )

    return (
        mean_mv,
        std_mv,
    )


def make_time_axis_ms():
    return (
        np.arange(
            N_SAMPLES,
            dtype=np.float64,
        )
        * (
            1000.0
            / FS_HZ
        )
        - PRE_MS
    )


class ECGCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(
            figsize=(13, 9),
            tight_layout=False,
        )

        super().__init__(
            self.fig
        )

    def plot_all(
        self,
        current_mv,
        reference_mv,
        latent_description,
    ):
        self.fig.clear()

        axes = self.fig.subplots(
            4,
            3,
            sharex=True,
        ).ravel()

        t_ms = make_time_axis_ms()

        for i, ax in enumerate(
            axes
        ):
            ax.plot(
                t_ms,
                reference_mv[i],
                linewidth=1.0,
                alpha=0.65,
                label="Normal centroid reference",
            )

            ax.plot(
                t_ms,
                current_mv[i],
                linewidth=1.45,
                label="Current decoded beat",
            )

            ax.axvline(
                0.0,
                linewidth=0.7,
                alpha=0.45,
            )

            ax.axhline(
                0.0,
                linewidth=0.6,
                alpha=0.4,
            )

            ax.set_title(
                LEAD_NAMES[i]
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

        self.fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
        )

        self.fig.suptitle(
            latent_description,
            y=0.995,
        )

        self.fig.subplots_adjust(
            left=0.06,
            right=0.99,
            bottom=0.07,
            top=0.92,
            hspace=0.34,
            wspace=0.23,
        )

        self.draw_idle()

    def plot_single(
        self,
        current_mv,
        reference_mv,
        lead_index,
        latent_description,
    ):
        self.fig.clear()

        ax = self.fig.add_subplot(
            111
        )

        t_ms = make_time_axis_ms()

        ax.plot(
            t_ms,
            reference_mv[
                lead_index
            ],
            linewidth=1.2,
            alpha=0.65,
            label="Normal centroid reference",
        )

        ax.plot(
            t_ms,
            current_mv[
                lead_index
            ],
            linewidth=1.7,
            label="Current decoded beat",
        )

        ax.axvline(
            0.0,
            linewidth=0.8,
            alpha=0.45,
        )

        ax.axhline(
            0.0,
            linewidth=0.7,
            alpha=0.4,
        )

        ax.grid(
            alpha=0.15
        )

        ax.set_xlabel(
            "Time from R/QRS fiducial (ms)"
        )

        ax.set_ylabel(
            "mV"
        )

        ax.set_title(
            f"{latent_description} | Lead {LEAD_NAMES[lead_index]}"
        )

        ax.legend(
            frameon=False
        )

        self.fig.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.10,
            top=0.92,
        )

        self.draw_idle()


class LatentExplorer(
    QMainWindow
):
    SLIDER_SCALE = 10
    SIGMA_MIN = -4.0
    SIGMA_MAX = 4.0

    def __init__(
        self,
        model,
        device,
        latent_mu,
        latent_sd,
        scale_mean_mv,
        scale_std_mv,
        n_normal_train,
        checkpoint_path,
        results_dir,
    ):
        super().__init__()

        self.model = model
        self.device = device

        self.latent_mu = (
            latent_mu.astype(
                np.float32
            )
        )

        self.latent_sd = (
            latent_sd.astype(
                np.float32
            )
        )

        self.scale_mean_mv = float(
            scale_mean_mv
        )

        self.scale_std_mv = float(
            scale_std_mv
        )

        self.n_normal_train = int(
            n_normal_train
        )

        self.checkpoint_path = (
            checkpoint_path
        )

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.latent_dim = len(
            self.latent_mu
        )

        self.current_latent_1 = 0
        self.current_sigma_1 = 0.0

        self.current_latent_2 = (
            1
            if self.latent_dim > 1
            else 0
        )

        self.current_sigma_2 = 0.0
        self.use_second_latent = False

        self.reference_mv = (
            self.decode_current(
                force_zero=True
            )
        )

        self.current_mv = (
            self.reference_mv.copy()
        )

        self.setWindowTitle(
            "SMASH-HCM | Median-Beat Latent ECG Explorer"
        )

        self.resize(
            1500,
            1000,
        )

        self.build_ui()
        self.update_view()

    def build_ui(self):
        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QVBoxLayout(
            central
        )

        controls = QGroupBox(
            "Latent controls"
        )

        grid = QGridLayout(
            controls
        )

        # Latent 1.
        grid.addWidget(
            QLabel("Latent 1"),
            0,
            0,
        )

        self.latent_combo_1 = (
            QComboBox()
        )

        for k in range(
            self.latent_dim
        ):
            self.latent_combo_1.addItem(
                f"z_{k:02d}",
                k,
            )

        self.latent_combo_1.currentIndexChanged.connect(
            self.on_latent_1_changed
        )

        grid.addWidget(
            self.latent_combo_1,
            0,
            1,
        )

        self.sigma_spin_1 = (
            QDoubleSpinBox()
        )

        self.sigma_spin_1.setRange(
            self.SIGMA_MIN,
            self.SIGMA_MAX,
        )

        self.sigma_spin_1.setDecimals(
            1
        )

        self.sigma_spin_1.setSingleStep(
            0.1
        )

        self.sigma_spin_1.setSuffix(
            " SD"
        )

        self.sigma_spin_1.valueChanged.connect(
            self.on_sigma_1_spin
        )

        grid.addWidget(
            self.sigma_spin_1,
            0,
            2,
        )

        self.sigma_slider_1 = (
            QSlider(
                Qt.Horizontal
            )
        )

        self.sigma_slider_1.setMinimum(
            int(
                self.SIGMA_MIN
                * self.SLIDER_SCALE
            )
        )

        self.sigma_slider_1.setMaximum(
            int(
                self.SIGMA_MAX
                * self.SLIDER_SCALE
            )
        )

        self.sigma_slider_1.setValue(
            0
        )

        self.sigma_slider_1.setTickInterval(
            5
        )

        self.sigma_slider_1.setTickPosition(
            QSlider.TicksBelow
        )

        self.sigma_slider_1.valueChanged.connect(
            self.on_sigma_1_slider
        )

        grid.addWidget(
            self.sigma_slider_1,
            0,
            3,
            1,
            3,
        )

        # Second latent.
        self.second_checkbox = (
            QCheckBox(
                "Activate second latent"
            )
        )

        self.second_checkbox.stateChanged.connect(
            self.on_second_toggled
        )

        grid.addWidget(
            self.second_checkbox,
            1,
            0,
        )

        self.latent_combo_2 = (
            QComboBox()
        )

        for k in range(
            self.latent_dim
        ):
            self.latent_combo_2.addItem(
                f"z_{k:02d}",
                k,
            )

        if self.latent_dim > 1:
            self.latent_combo_2.setCurrentIndex(
                1
            )

        self.latent_combo_2.currentIndexChanged.connect(
            self.on_latent_2_changed
        )

        grid.addWidget(
            self.latent_combo_2,
            1,
            1,
        )

        self.sigma_spin_2 = (
            QDoubleSpinBox()
        )

        self.sigma_spin_2.setRange(
            self.SIGMA_MIN,
            self.SIGMA_MAX,
        )

        self.sigma_spin_2.setDecimals(
            1
        )

        self.sigma_spin_2.setSingleStep(
            0.1
        )

        self.sigma_spin_2.setSuffix(
            " SD"
        )

        self.sigma_spin_2.valueChanged.connect(
            self.on_sigma_2_spin
        )

        grid.addWidget(
            self.sigma_spin_2,
            1,
            2,
        )

        self.sigma_slider_2 = (
            QSlider(
                Qt.Horizontal
            )
        )

        self.sigma_slider_2.setMinimum(
            int(
                self.SIGMA_MIN
                * self.SLIDER_SCALE
            )
        )

        self.sigma_slider_2.setMaximum(
            int(
                self.SIGMA_MAX
                * self.SLIDER_SCALE
            )
        )

        self.sigma_slider_2.setValue(
            0
        )

        self.sigma_slider_2.setTickInterval(
            5
        )

        self.sigma_slider_2.setTickPosition(
            QSlider.TicksBelow
        )

        self.sigma_slider_2.valueChanged.connect(
            self.on_sigma_2_slider
        )

        grid.addWidget(
            self.sigma_slider_2,
            1,
            3,
            1,
            3,
        )

        # Display selector.
        grid.addWidget(
            QLabel("Display"),
            2,
            0,
        )

        self.lead_combo = (
            QComboBox()
        )

        self.lead_combo.addItem(
            "All 12 leads",
            -1,
        )

        for i, lead in enumerate(
            LEAD_NAMES
        ):
            self.lead_combo.addItem(
                f"Lead {lead}",
                i,
            )

        self.lead_combo.currentIndexChanged.connect(
            self.update_plot
        )

        grid.addWidget(
            self.lead_combo,
            2,
            1,
        )

        button_row = (
            QHBoxLayout()
        )

        self.reset_button = (
            QPushButton(
                "Reset to centroid"
            )
        )

        self.reset_button.clicked.connect(
            self.reset_all
        )

        self.save_button = (
            QPushButton(
                "Save figure"
            )
        )

        self.save_button.clicked.connect(
            self.save_figure
        )

        button_row.addWidget(
            self.reset_button
        )

        button_row.addWidget(
            self.save_button
        )

        button_row.addStretch(
            1
        )

        grid.addLayout(
            button_row,
            2,
            2,
            1,
            4,
        )

        main_layout.addWidget(
            controls
        )

        # Info panel.
        info_box = QGroupBox(
            "Current latent coordinates"
        )

        info_grid = QGridLayout(
            info_box
        )

        self.info_z1 = QLabel()
        self.info_z2 = QLabel()
        self.info_rms = QLabel()
        self.info_maxabs = QLabel()

        info_grid.addWidget(
            QLabel("Latent 1"),
            0,
            0,
        )

        info_grid.addWidget(
            self.info_z1,
            0,
            1,
        )

        info_grid.addWidget(
            QLabel("Latent 2"),
            0,
            2,
        )

        info_grid.addWidget(
            self.info_z2,
            0,
            3,
        )

        info_grid.addWidget(
            QLabel("Global RMS change"),
            1,
            0,
        )

        info_grid.addWidget(
            self.info_rms,
            1,
            1,
        )

        info_grid.addWidget(
            QLabel("Global max |change|"),
            1,
            2,
        )

        info_grid.addWidget(
            self.info_maxabs,
            1,
            3,
        )

        main_layout.addWidget(
            info_box
        )

        self.canvas = ECGCanvas()

        main_layout.addWidget(
            self.canvas,
            stretch=1,
        )

        footer = QLabel(
            f"Reference: empirical centroid of "
            f"{self.n_normal_train:,} Normal-train median beats | "
            f"Device: {self.device} | "
            f"Checkpoint: {self.checkpoint_path.name}"
        )

        footer.setWordWrap(
            True
        )

        main_layout.addWidget(
            footer
        )

        self.update_second_enabled_state()

    @torch.no_grad()
    def decode_current(
        self,
        force_zero: bool = False,
    ):
        z = (
            self.latent_mu.copy()
        )

        if not force_zero:
            z[
                self.current_latent_1
            ] = (
                self.latent_mu[
                    self.current_latent_1
                ]
                + self.current_sigma_1
                * self.latent_sd[
                    self.current_latent_1
                ]
            )

            if (
                self.use_second_latent
            ):
                z[
                    self.current_latent_2
                ] = (
                    self.latent_mu[
                        self.current_latent_2
                    ]
                    + self.current_sigma_2
                    * self.latent_sd[
                        self.current_latent_2
                    ]
                )

        z_tensor = (
            torch.from_numpy(
                z[
                    None,
                    :
                ]
            )
            .float()
            .to(
                self.device
            )
        )

        x_scaled = (
            self.model.decode(
                z_tensor
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        x_mv = (
            x_scaled
            * self.scale_std_mv
            + self.scale_mean_mv
        )

        return x_mv.astype(
            np.float32
        )

    def update_second_enabled_state(
        self
    ):
        enabled = (
            self.use_second_latent
        )

        self.latent_combo_2.setEnabled(
            enabled
        )

        self.sigma_spin_2.setEnabled(
            enabled
        )

        self.sigma_slider_2.setEnabled(
            enabled
        )

    def on_latent_1_changed(
        self,
        index,
    ):
        self.current_latent_1 = int(
            self.latent_combo_1.itemData(
                index
            )
        )

        self.update_view()

    def on_latent_2_changed(
        self,
        index,
    ):
        self.current_latent_2 = int(
            self.latent_combo_2.itemData(
                index
            )
        )

        self.update_view()

    def on_sigma_1_slider(
        self,
        raw,
    ):
        sigma = (
            raw
            / self.SLIDER_SCALE
        )

        self.sigma_spin_1.blockSignals(
            True
        )

        self.sigma_spin_1.setValue(
            sigma
        )

        self.sigma_spin_1.blockSignals(
            False
        )

        self.current_sigma_1 = (
            float(
                sigma
            )
        )

        self.update_view()

    def on_sigma_1_spin(
        self,
        sigma,
    ):
        sigma = float(
            sigma
        )

        raw = int(
            round(
                sigma
                * self.SLIDER_SCALE
            )
        )

        self.sigma_slider_1.blockSignals(
            True
        )

        self.sigma_slider_1.setValue(
            raw
        )

        self.sigma_slider_1.blockSignals(
            False
        )

        self.current_sigma_1 = (
            sigma
        )

        self.update_view()

    def on_sigma_2_slider(
        self,
        raw,
    ):
        sigma = (
            raw
            / self.SLIDER_SCALE
        )

        self.sigma_spin_2.blockSignals(
            True
        )

        self.sigma_spin_2.setValue(
            sigma
        )

        self.sigma_spin_2.blockSignals(
            False
        )

        self.current_sigma_2 = (
            float(
                sigma
            )
        )

        self.update_view()

    def on_sigma_2_spin(
        self,
        sigma,
    ):
        sigma = float(
            sigma
        )

        raw = int(
            round(
                sigma
                * self.SLIDER_SCALE
            )
        )

        self.sigma_slider_2.blockSignals(
            True
        )

        self.sigma_slider_2.setValue(
            raw
        )

        self.sigma_slider_2.blockSignals(
            False
        )

        self.current_sigma_2 = (
            sigma
        )

        self.update_view()

    def on_second_toggled(
        self,
        state,
    ):
        self.use_second_latent = (
            state
            == Qt.Checked
        )

        self.update_second_enabled_state()
        self.update_view()

    def reset_all(self):
        self.sigma_spin_1.setValue(
            0.0
        )

        self.sigma_spin_2.setValue(
            0.0
        )

        self.second_checkbox.setChecked(
            False
        )

    def current_description(
        self
    ):
        z1 = (
            f"z_{self.current_latent_1:02d}"
            f" = {self.current_sigma_1:+.1f} SD"
        )

        if (
            self.use_second_latent
        ):
            z2 = (
                f"z_{self.current_latent_2:02d}"
                f" = {self.current_sigma_2:+.1f} SD"
            )

            return (
                f"{z1} | {z2}"
            )

        return z1

    def update_view(self):
        self.current_mv = (
            self.decode_current(
                force_zero=False
            )
        )

        self.update_info()
        self.update_plot()

    def update_info(self):
        k1 = (
            self.current_latent_1
        )

        z1_value = (
            self.latent_mu[k1]
            + self.current_sigma_1
            * self.latent_sd[k1]
        )

        self.info_z1.setText(
            f"z_{k1:02d}: "
            f"{self.current_sigma_1:+.1f} SD "
            f"(value={z1_value:.4f}, "
            f"mean={self.latent_mu[k1]:.4f}, "
            f"SD={self.latent_sd[k1]:.4f})"
        )

        if (
            self.use_second_latent
        ):
            k2 = (
                self.current_latent_2
            )

            z2_value = (
                self.latent_mu[k2]
                + self.current_sigma_2
                * self.latent_sd[k2]
            )

            self.info_z2.setText(
                f"z_{k2:02d}: "
                f"{self.current_sigma_2:+.1f} SD "
                f"(value={z2_value:.4f}, "
                f"mean={self.latent_mu[k2]:.4f}, "
                f"SD={self.latent_sd[k2]:.4f})"
            )
        else:
            self.info_z2.setText(
                "inactive"
            )

        delta = (
            self.current_mv
            - self.reference_mv
        )

        rms = float(
            np.sqrt(
                np.mean(
                    delta ** 2
                )
            )
        )

        maxabs = float(
            np.max(
                np.abs(
                    delta
                )
            )
        )

        self.info_rms.setText(
            f"{rms:.5f} mV"
        )

        self.info_maxabs.setText(
            f"{maxabs:.5f} mV"
        )

    def update_plot(self):
        lead_idx = int(
            self.lead_combo.currentData()
        )

        description = (
            self.current_description()
        )

        if lead_idx < 0:
            self.canvas.plot_all(
                current_mv=self.current_mv,
                reference_mv=self.reference_mv,
                latent_description=description,
            )
        else:
            self.canvas.plot_single(
                current_mv=self.current_mv,
                reference_mv=self.reference_mv,
                lead_index=lead_idx,
                latent_description=description,
            )

    def save_figure(self):
        description = (
            self.current_description()
            .replace(" ", "")
            .replace("|", "_")
            .replace("=", "")
            .replace("+", "plus")
            .replace("-", "minus")
        )

        suggested = (
            self.results_dir
            / f"medianbeat_{description}.png"
        )

        filename, _ = (
            QFileDialog.getSaveFileName(
                self,
                "Save current latent figure",
                str(suggested),
                "PNG image (*.png)",
            )
        )

        if filename:
            try:
                self.canvas.fig.savefig(
                    filename,
                    dpi=200,
                    bbox_inches="tight",
                )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Save error",
                    str(exc),
                )


def parse_args():
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
        "--results-dir",
        type=Path,
        default=(
            root
            / "Results"
            / "08_Phase1_MedianBeat_LatentExplorer"
        ),
    )

    return ap.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if not args.latents.exists():
        raise FileNotFoundError(
            f"Latent CSV not found: {args.latents}"
        )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    (
        latent_mu,
        latent_sd,
        n_normal_train,
    ) = load_normal_latent_stats(
        args.latents,
        latent_dim,
    )

    (
        scale_mean_mv,
        scale_std_mv,
    ) = load_scaling(
        args.preprocessed_h5
    )

    print(
        f"Device              : {device}"
    )

    print(
        f"Checkpoint          : {args.checkpoint}"
    )

    print(
        f"Latent CSV          : {args.latents}"
    )

    print(
        f"Normal-train beats  : {n_normal_train:,}"
    )

    print(
        f"Latent dimensions   : {latent_dim}"
    )

    print(
        f"Scaling mean [mV]   : {scale_mean_mv}"
    )

    print(
        f"Scaling std [mV]    : {scale_std_mv}"
    )

    print(
        f"Default save dir    : {args.results_dir}"
    )

    app = QApplication(
        sys.argv
    )

    window = LatentExplorer(
        model=model,
        device=device,
        latent_mu=latent_mu,
        latent_sd=latent_sd,
        scale_mean_mv=scale_mean_mv,
        scale_std_mv=scale_std_mv,
        n_normal_train=n_normal_train,
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
    )

    window.show()

    sys.exit(
        app.exec_()
    )


if __name__ == "__main__":
    main()
