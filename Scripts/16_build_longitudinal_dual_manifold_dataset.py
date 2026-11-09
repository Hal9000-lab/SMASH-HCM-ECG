#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16_build_longitudinal_dual_manifold_dataset.py

SMASH-HCM -- Longitudinal Phase / Step 16
Build the frozen dual-manifold longitudinal ECG representation.

SCIENTIFIC DESIGN
-----------------
No new VAE is trained in this step.

All available longitudinal HCM ECG visits are processed with the same morphology
pipeline used to train the cross-sectional models, and are then projected through
the two FROZEN encoders:

    Normal-anchored VAE  -> z^N(t)
    HCM-static VAE       -> z^HCM(t)

For every valid visit the script also computes:

    d_N(t)        distance from the Normal-train latent distribution
    d_HCM(t)      distance from the HCM-train latent distribution
    RE_N(t)       physical reconstruction MSE of the Normal decoder
    RE_H(t)       physical reconstruction MSE of the HCM decoder
    Delta_RE(t)   RE_N(t) - RE_H(t)

as well as rhythm and beat-to-beat morphology descriptors.

IMPORTANT
---------
- The raw longitudinal 4-s HCM records are first filtered with the SAME
  4th-order zero-phase Butterworth 0.5--40 Hz filter used in Phase I.
- Median beats use the SAME common multilead QRS detector and the SAME
  [-350,+650] ms extraction window used in the baseline analysis.
- No beat is removed because its morphology differs from the dominant beat.
- A visit is considered valid for latent projection when >=2 complete finite
  beats are available.
- No longitudinal visit is used to update either VAE.
- patient_id is NOT used as the subject identity key.
- subject_key is preferred; subject_index is retained as a second coherent
  identity variable.
- Time is measured from the earliest recorded ECG for each coherent subject.

DEFAULT INPUTS
--------------
Results/02_SMASH_HCM_Harmonized_4s/Phase2_HCM_longitudinal_4s500Hz.h5
Results/06_Phase1_MedianBeat_Beta_VAE/best_model.pt
Results/06_Phase1_MedianBeat_Beta_VAE/phase1_medianbeat_latents.csv
Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5
Results/12_HCM_Static_Beta_VAE/best_model.pt
Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv

DEFAULT OUTPUTS
---------------
Results/16_Longitudinal_DualManifold/Phase3_HCM_Longitudinal_DualManifold.h5

Results/16_Longitudinal_DualManifold/
    longitudinal_dual_manifold.csv
    longitudinal_subject_summary.csv
    baseline_projection_consistency.csv
    longitudinal_qc.csv
    run_summary.json
    visits_per_subject.png
    followup_duration_years.png

RUN
---
python Scripts\\16_build_longitudinal_dual_manifold_dataset.py

DEPENDENCIES
------------
numpy pandas scipy h5py torch scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.covariance import LedoitWolf

import torch
import torch.nn as nn


SEED = 20261004

FS_HZ = 500.0
LOW_HZ = 0.5
HIGH_HZ = 40.0
FILTER_ORDER = 4

PRE_MS = 350.0
POST_MS = 650.0
BEAT_SAMPLES = 500
MORPH_SIMILARITY_THRESHOLD = 0.90
MIN_BEATS_FOR_LATENT = 2

NORMAL_LATENT_DIM = 16
HCM_LATENT_DIM = 8

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

EPS = 1e-12


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "Results/02_SMASH_HCM_Harmonized_4s/Phase2_HCM_longitudinal_4s500Hz.h5"
        ),
    )
    p.add_argument(
        "--normal-checkpoint",
        type=Path,
        default=Path(
            "Results/06_Phase1_MedianBeat_Beta_VAE/best_model.pt"
        ),
    )
    p.add_argument(
        "--normal-latents",
        type=Path,
        default=Path(
            "Results/06_Phase1_MedianBeat_Beta_VAE/"
            "phase1_medianbeat_latents.csv"
        ),
    )
    p.add_argument(
        "--hcm-prepared-h5",
        type=Path,
        default=Path(
            "Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5"
        ),
    )
    p.add_argument(
        "--hcm-checkpoint",
        type=Path,
        default=Path(
            "Results/12_HCM_Static_Beta_VAE/best_model.pt"
        ),
    )
    p.add_argument(
        "--hcm-latents",
        type=Path,
        default=Path(
            "Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv"
        ),
    )
    p.add_argument(
        "--output-h5",
        type=Path,
        default=Path(
            "Results/16_Longitudinal_DualManifold/Phase3_HCM_Longitudinal_DualManifold.h5"
        ),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "Results/16_Longitudinal_DualManifold"
        ),
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--min-beats",
        type=int,
        default=MIN_BEATS_FOR_LATENT,
    )
    p.add_argument(
        "--morph-similarity-threshold",
        type=float,
        default=MORPH_SIMILARITY_THRESHOLD,
    )
    return p.parse_args()


# =====================================================================
# General utilities
# =====================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def decode_string_array(arr):
    out = []
    for x in np.asarray(arr):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def write_str_dataset(group, name, values):
    dt = h5py.string_dtype("utf-8")
    group.create_dataset(
        name,
        data=np.asarray([str(x) for x in values], dtype=object),
        dtype=dt,
        compression="gzip",
        compression_opts=4,
    )


def clean_id(x):
    if pd.isna(x):
        return ""

    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace").strip()

    s = str(x).strip()

    if (
        (s.startswith("b'") and s.endswith("'"))
        or (s.startswith('b"') and s.endswith('"'))
    ):
        try:
            value = ast.literal_eval(s)
            if isinstance(value, bytes):
                return value.decode(
                    "utf-8",
                    errors="replace",
                ).strip()
        except Exception:
            pass

    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def torch_load_checkpoint(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def time_axis_ms():
    return (
        -PRE_MS
        + np.arange(BEAT_SAMPLES, dtype=float)
        * 1000.0
        / FS_HZ
    )


# =====================================================================
# Exact Phase-I filter
# =====================================================================

def make_filter():
    return butter(
        FILTER_ORDER,
        [LOW_HZ, HIGH_HZ],
        btype="bandpass",
        fs=FS_HZ,
        output="sos",
    )


# =====================================================================
# Median-beat construction
# =====================================================================

def robust_scale_per_lead(x):
    med = np.median(x, axis=1, keepdims=True)
    centered = x - med

    mad = np.median(
        np.abs(centered),
        axis=1,
        keepdims=True,
    )
    scale = 1.4826 * mad

    std = np.std(
        centered,
        axis=1,
        keepdims=True,
    )

    scale = np.where(scale > 1e-6, scale, std)
    scale = np.where(scale > 1e-6, scale, 1.0)

    return centered / scale


def detect_multilead_qrs(x_mv, fs_hz=FS_HZ):
    if x_mv.shape[0] != 12:
        raise ValueError("Expected 12 leads.")

    z = robust_scale_per_lead(x_mv)

    grad = np.gradient(z, axis=1)
    energy = np.median(grad ** 2, axis=0)

    smooth_samples = max(
        3,
        int(round(0.080 * fs_hz)),
    )
    detection_score = uniform_filter1d(
        energy,
        size=smooth_samples,
        mode="nearest",
    )

    med = float(np.median(detection_score))
    mad = float(
        1.4826
        * np.median(
            np.abs(detection_score - med)
        )
    )
    p99 = float(
        np.percentile(
            detection_score,
            99.0,
        )
    )

    min_distance = max(
        1,
        int(round(0.25 * fs_hz)),
    )

    height = med + 2.5 * mad
    prominence = max(
        1.5 * mad,
        0.05 * max(p99 - med, 1e-12),
        1e-12,
    )

    peaks, _ = find_peaks(
        detection_score,
        distance=min_distance,
        height=height,
        prominence=prominence,
    )

    if len(peaks) < 2:
        height = med + 1.5 * mad
        prominence = max(
            0.75 * mad,
            0.025 * max(p99 - med, 1e-12),
            1e-12,
        )
        peaks, _ = find_peaks(
            detection_score,
            distance=min_distance,
            height=height,
            prominence=prominence,
        )

    if len(peaks) == 0:
        prominence = max(
            0.01 * max(p99 - med, 1e-12),
            1e-12,
        )
        peaks, _ = find_peaks(
            detection_score,
            distance=min_distance,
            prominence=prominence,
        )

        if len(peaks) > 0:
            order = np.argsort(
                detection_score[peaks]
            )[::-1]
            peaks = np.sort(
                peaks[order[:16]]
            )

    composite = np.sqrt(
        np.mean(z ** 2, axis=0)
    )

    refine_radius = max(
        1,
        int(round(0.060 * fs_hz)),
    )

    refined = []

    for p in peaks:
        a = max(
            0,
            int(p) - refine_radius,
        )
        b = min(
            x_mv.shape[1],
            int(p) + refine_radius + 1,
        )

        local = composite[a:b]

        if len(local) == 0:
            continue

        refined.append(
            a + int(np.argmax(local))
        )

    if not refined:
        return np.asarray([], dtype=np.int32)

    refined = np.asarray(
        sorted(set(refined)),
        dtype=np.int32,
    )

    min_refined_distance = max(
        1,
        int(round(0.20 * fs_hz)),
    )

    dedup = []

    for p in refined:
        if not dedup:
            dedup.append(int(p))
            continue

        if p - dedup[-1] >= min_refined_distance:
            dedup.append(int(p))
        else:
            prev = dedup[-1]
            if composite[p] > composite[prev]:
                dedup[-1] = int(p)

    return np.asarray(
        dedup,
        dtype=np.int32,
    )


def extract_complete_beats(
    x_mv,
    peaks,
    pre_samples,
    post_samples,
):
    beats = []
    used_peaks = []
    excluded_edge = 0
    excluded_nonfinite = 0

    n_samples = x_mv.shape[1]

    for p in peaks:
        start = int(p) - pre_samples
        stop = int(p) + post_samples

        if (
            start < 0
            or stop > n_samples
        ):
            excluded_edge += 1
            continue

        beat = x_mv[:, start:stop]

        if (
            beat.shape[1]
            != pre_samples + post_samples
        ):
            excluded_edge += 1
            continue

        if not np.isfinite(beat).all():
            excluded_nonfinite += 1
            continue

        beats.append(
            beat.astype(
                np.float32,
                copy=False,
            )
        )
        used_peaks.append(int(p))

    if beats:
        beats = np.stack(
            beats,
            axis=0,
        )
    else:
        beats = np.empty(
            (
                0,
                12,
                pre_samples + post_samples,
            ),
            dtype=np.float32,
        )

    return (
        beats,
        np.asarray(
            used_peaks,
            dtype=np.int32,
        ),
        excluded_edge,
        excluded_nonfinite,
    )


def beat_shape_vectors(beats_mv):
    if len(beats_mv) == 0:
        return np.empty(
            (0, 0),
            dtype=np.float64,
        )

    b = beats_mv.astype(np.float64)

    centered = (
        b
        - np.mean(
            b,
            axis=2,
            keepdims=True,
        )
    )

    vectors = centered.reshape(
        centered.shape[0],
        -1,
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )
    norms = np.where(
        norms > 1e-12,
        norms,
        1.0,
    )

    return vectors / norms


def morphology_similarity_matrix(beats_mv):
    vectors = beat_shape_vectors(beats_mv)

    if len(vectors) == 0:
        return np.empty(
            (0, 0),
            dtype=np.float64,
        )

    return np.clip(
        vectors @ vectors.T,
        -1.0,
        1.0,
    )


def connected_components_from_similarity(
    similarity,
    threshold,
):
    n = similarity.shape[0]

    if n == 0:
        return []

    visited = np.zeros(
        n,
        dtype=bool,
    )
    components = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        comp = []

        while stack:
            i = stack.pop()
            comp.append(i)

            neighbors = np.where(
                similarity[i] >= threshold
            )[0]

            for j in neighbors:
                if not visited[j]:
                    visited[j] = True
                    stack.append(int(j))

        components.append(
            np.asarray(
                sorted(comp),
                dtype=np.int32,
            )
        )

    components.sort(
        key=lambda c: (
            -len(c),
            int(c[0]),
        )
    )

    return components


def dominant_morphology_indices(
    similarity,
    threshold,
):
    n = similarity.shape[0]

    if n == 0:
        return np.asarray(
            [],
            dtype=np.int32,
        )

    if n == 1:
        return np.asarray(
            [0],
            dtype=np.int32,
        )

    neighbor_counts = np.sum(
        similarity >= threshold,
        axis=1,
    )

    median_similarity = np.median(
        similarity,
        axis=1,
    )

    best_count = np.max(
        neighbor_counts
    )

    candidates = np.where(
        neighbor_counts == best_count
    )[0]

    if len(candidates) > 1:
        best = candidates[
            np.argmax(
                median_similarity[candidates]
            )
        ]
    else:
        best = candidates[0]

    dominant = np.where(
        similarity[best] >= threshold
    )[0]

    return np.asarray(
        dominant,
        dtype=np.int32,
    )


def compute_rr_metrics(peaks, fs_hz=FS_HZ):
    if len(peaks) < 2:
        return {
            "rr_count": 0,
            "rr_mean_ms": np.nan,
            "rr_median_ms": np.nan,
            "rr_sd_ms": np.nan,
            "rr_mad_ms": np.nan,
            "rr_cv": np.nan,
            "rmssd_ms": np.nan,
            "heart_rate_bpm": np.nan,
        }

    rr_ms = (
        np.diff(
            peaks.astype(np.float64)
        )
        / fs_hz
        * 1000.0
    )

    rr_mean = float(
        np.mean(rr_ms)
    )
    rr_median = float(
        np.median(rr_ms)
    )

    rr_sd = (
        float(
            np.std(
                rr_ms,
                ddof=1,
            )
        )
        if len(rr_ms) >= 2
        else 0.0
    )

    rr_mad = float(
        np.median(
            np.abs(
                rr_ms - rr_median
            )
        )
    )

    rr_cv = (
        rr_sd / rr_mean
        if rr_mean > 0
        else np.nan
    )

    drr = np.diff(rr_ms)

    rmssd = (
        float(
            np.sqrt(
                np.mean(
                    drr ** 2
                )
            )
        )
        if len(drr) > 0
        else np.nan
    )

    hr = (
        60000.0 / rr_median
        if rr_median > 0
        else np.nan
    )

    return {
        "rr_count": int(len(rr_ms)),
        "rr_mean_ms": rr_mean,
        "rr_median_ms": rr_median,
        "rr_sd_ms": rr_sd,
        "rr_mad_ms": rr_mad,
        "rr_cv": rr_cv,
        "rmssd_ms": rmssd,
        "heart_rate_bpm": hr,
    }


def compute_morph_metrics(
    beats,
    med_all,
    similarity,
    dominant_idx,
    threshold,
):
    n = len(beats)

    if n == 0:
        return {
            "morph_pairwise_similarity_median": np.nan,
            "morph_pairwise_similarity_min": np.nan,
            "morph_rms_to_median_mv": np.nan,
            "morph_rms_relative_to_template": np.nan,
            "dominant_cluster_size": 0,
            "dominant_cluster_fraction": np.nan,
            "morph_components_at_threshold": 0,
        }

    delta = (
        beats.astype(np.float64)
        - med_all[None, :, :].astype(np.float64)
    )

    rms_per_beat = np.sqrt(
        np.mean(
            delta ** 2,
            axis=(1, 2),
        )
    )

    template_rms = float(
        np.sqrt(
            np.mean(
                med_all.astype(np.float64) ** 2
            )
        )
    )

    relative = (
        float(
            np.median(rms_per_beat)
            / template_rms
        )
        if template_rms > 1e-12
        else np.nan
    )

    if n >= 2:
        tri = similarity[
            np.triu_indices(
                n,
                k=1,
            )
        ]
        sim_median = float(
            np.median(tri)
        )
        sim_min = float(
            np.min(tri)
        )
    else:
        sim_median = np.nan
        sim_min = np.nan

    components = (
        connected_components_from_similarity(
            similarity,
            threshold,
        )
    )

    return {
        "morph_pairwise_similarity_median": sim_median,
        "morph_pairwise_similarity_min": sim_min,
        "morph_rms_to_median_mv": float(
            np.median(rms_per_beat)
        ),
        "morph_rms_relative_to_template": relative,
        "dominant_cluster_size": int(
            len(dominant_idx)
        ),
        "dominant_cluster_fraction": float(
            len(dominant_idx) / n
        ),
        "morph_components_at_threshold": int(
            len(components)
        ),
    }


def qc_label(n_usable):
    if n_usable >= 3:
        return "good"
    if n_usable == 2:
        return "acceptable"
    if n_usable == 1:
        return "limited"
    return "failed"


# =====================================================================
# Model 1: exact Normal median-beat VAE architecture
# =====================================================================

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
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
        c_in,
        c_out,
        output_padding=0,
        activate=True,
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
    def __init__(self, latent_dim=16):
        super().__init__()

        self.latent_dim = latent_dim

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
            latent_dim,
        )
        self.fc_logvar = nn.Linear(
            96,
            latent_dim,
        )

        self.dec_fc = nn.Sequential(
            nn.Linear(
                latent_dim,
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
                0,
                True,
            ),
            DeconvBlock(
                96,
                64,
                1,
                True,
            ),
            DeconvBlock(
                64,
                32,
                0,
                True,
            ),
            DeconvBlock(
                32,
                12,
                0,
                False,
            ),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(
            start_dim=1
        )
        h = self.enc_fc(h)
        return (
            self.fc_mu(h),
            self.fc_logvar(h),
        )

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(
            -1,
            128,
            31,
        )
        return self.decoder(h)


# =====================================================================
# Model 2: exact HCM-static VAE architecture
# =====================================================================

class HCMStaticBetaVAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(
                12,
                16,
                4,
                2,
                1,
            ),
            nn.GroupNorm(
                4,
                16,
            ),
            nn.SiLU(),

            nn.Conv1d(
                16,
                32,
                4,
                2,
                1,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.SiLU(),

            nn.Conv1d(
                32,
                48,
                4,
                2,
                1,
            ),
            nn.GroupNorm(
                8,
                48,
            ),
            nn.SiLU(),

            nn.Conv1d(
                48,
                64,
                4,
                2,
                1,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.SiLU(),
        )

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * 31,
                48,
            ),
            nn.SiLU(),
        )

        self.fc_mu = nn.Linear(
            48,
            latent_dim,
        )
        self.fc_logvar = nn.Linear(
            48,
            latent_dim,
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(
                latent_dim,
                48,
            ),
            nn.SiLU(),
            nn.Linear(
                48,
                64 * 31,
            ),
            nn.SiLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(
                64,
                48,
                4,
                2,
                1,
                output_padding=0,
            ),
            nn.GroupNorm(
                8,
                48,
            ),
            nn.SiLU(),

            nn.ConvTranspose1d(
                48,
                32,
                4,
                2,
                1,
                output_padding=1,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.SiLU(),

            nn.ConvTranspose1d(
                32,
                16,
                4,
                2,
                1,
                output_padding=0,
            ),
            nn.GroupNorm(
                4,
                16,
            ),
            nn.SiLU(),

            nn.ConvTranspose1d(
                16,
                12,
                4,
                2,
                1,
                output_padding=0,
            ),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = self.encoder_fc(h)
        return (
            self.fc_mu(h),
            self.fc_logvar(h),
        )

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(
            z.shape[0],
            64,
            31,
        )
        return self.decoder_conv(h)


# =====================================================================
# Load frozen models and scaling
# =====================================================================

def load_frozen_models(
    normal_checkpoint,
    hcm_checkpoint,
    hcm_prepared_h5,
    device,
):
    nckpt = torch_load_checkpoint(
        normal_checkpoint,
        device,
    )

    normal_dim = int(
        nckpt.get(
            "latent_dim",
            NORMAL_LATENT_DIM,
        )
    )

    if normal_dim != NORMAL_LATENT_DIM:
        raise RuntimeError(
            f"Unexpected Normal latent dim: {normal_dim}"
        )

    normal_mean = float(
        nckpt[
            "normal_train_scaling_mean_mv"
        ]
    )
    normal_std = float(
        nckpt[
            "normal_train_scaling_std_mv"
        ]
    )

    normal_model = MedianBeatBetaVAE(
        latent_dim=normal_dim
    ).to(device)

    normal_model.load_state_dict(
        nckpt["model_state_dict"]
    )
    normal_model.eval()

    normal_params = sum(
        p.numel()
        for p in normal_model.parameters()
    )

    if normal_params != 939116:
        raise RuntimeError(
            "Normal-VAE architecture mismatch: "
            f"{normal_params:,} parameters, "
            "expected 939,116."
        )

    hckpt = torch_load_checkpoint(
        hcm_checkpoint,
        device,
    )

    hcm_dim = int(
        hckpt.get(
            "latent_dim",
            HCM_LATENT_DIM,
        )
    )

    if hcm_dim != HCM_LATENT_DIM:
        raise RuntimeError(
            f"Unexpected HCM latent dim: {hcm_dim}"
        )

    hcm_model = HCMStaticBetaVAE(
        latent_dim=hcm_dim
    ).to(device)

    hcm_model.load_state_dict(
        hckpt["model_state_dict"]
    )
    hcm_model.eval()

    hcm_params = sum(
        p.numel()
        for p in hcm_model.parameters()
    )

    if hcm_params != 236988:
        raise RuntimeError(
            "HCM-VAE architecture mismatch: "
            f"{hcm_params:,} parameters, "
            "expected 236,988."
        )

    with h5py.File(
        hcm_prepared_h5,
        "r",
    ) as h5:
        hcm_mean = float(
            h5.attrs[
                "hcm_train_scaling_mean_mv"
            ]
        )
        hcm_std = float(
            h5.attrs[
                "hcm_train_scaling_std_mv"
            ]
        )

    return {
        "normal_model": normal_model,
        "hcm_model": hcm_model,
        "normal_mean_mv": normal_mean,
        "normal_std_mv": normal_std,
        "hcm_mean_mv": hcm_mean,
        "hcm_std_mv": hcm_std,
        "normal_params": normal_params,
        "hcm_params": hcm_params,
    }


# =====================================================================
# Encode / reconstruct
# =====================================================================

@torch.inference_mode()
def encode_reconstruct(
    model,
    x_mv,
    scale_mean,
    scale_std,
    batch_size,
    device,
):
    x_mv = np.asarray(
        x_mv,
        dtype=np.float32,
    )

    x_scaled = (
        (
            x_mv
            - scale_mean
        )
        / scale_std
    ).astype(np.float32)

    latent_dim = model.latent_dim

    mu_all = np.empty(
        (
            len(x_mv),
            latent_dim,
        ),
        dtype=np.float32,
    )

    logvar_all = np.empty_like(
        mu_all
    )

    recon_mv = np.empty_like(
        x_mv,
        dtype=np.float32,
    )

    for start in range(
        0,
        len(x_mv),
        batch_size,
    ):
        stop = min(
            len(x_mv),
            start + batch_size,
        )

        xb = torch.from_numpy(
            x_scaled[start:stop]
        ).to(device)

        mu, logvar = model.encode(xb)
        recon_scaled = model.decode(mu)

        mu_all[start:stop] = (
            mu.cpu().numpy()
        )
        logvar_all[start:stop] = (
            logvar.cpu().numpy()
        )

        recon_mv[start:stop] = (
            recon_scaled.cpu().numpy()
            * scale_std
            + scale_mean
        ).astype(np.float32)

    return (
        mu_all,
        logvar_all,
        recon_mv,
    )


def compute_reconstruction_metrics(
    x_mv,
    recon_mv,
):
    residual = (
        x_mv.astype(np.float64)
        - recon_mv.astype(np.float64)
    )

    sq = residual ** 2

    t = time_axis_ms()

    masks = {
        "preQRS": (
            (t >= -300.0)
            & (t < -60.0)
        ),
        "QRS": (
            (t >= -60.0)
            & (t <= 80.0)
        ),
        "STT": (
            (t > 80.0)
            & (t <= 500.0)
        ),
    }

    out = {
        "mse": np.mean(
            sq,
            axis=(1, 2),
        ),
        "rmse": np.sqrt(
            np.mean(
                sq,
                axis=(1, 2),
            )
        ),
    }

    for region, mask in masks.items():
        out[
            f"mse_{region}"
        ] = np.mean(
            sq[:, :, mask],
            axis=(1, 2),
        )

    out["STT_to_QRS_mse_ratio"] = (
        out["mse_STT"]
        / (
            out["mse_QRS"]
            + EPS
        )
    )

    return out


# =====================================================================
# Latent reference distributions
# =====================================================================

def get_latent_columns(df, expected):
    cols = [
        c
        for c in df.columns
        if c.startswith("z_")
        and c[2:].isdigit()
    ]

    cols = sorted(
        cols,
        key=lambda c: int(
            c.split("_")[1]
        ),
    )

    if len(cols) != expected:
        raise RuntimeError(
            f"Expected {expected} latent columns; "
            f"found {len(cols)}."
        )

    return cols


def fit_reference_distances(
    normal_latents_path,
    hcm_latents_path,
):
    normal = pd.read_csv(
        normal_latents_path
    )
    hcm = pd.read_csv(
        hcm_latents_path
    )

    zn_cols = get_latent_columns(
        normal,
        NORMAL_LATENT_DIM,
    )
    zh_cols = get_latent_columns(
        hcm,
        HCM_LATENT_DIM,
    )

    nref = normal[
        normal["split_name"]
        .astype(str)
        .eq("normal_train")
    ][zn_cols].to_numpy(
        dtype=float
    )

    href = hcm[
        hcm["split_name"]
        .astype(str)
        .eq("hcm_train")
    ][zh_cols].to_numpy(
        dtype=float
    )

    lw_n = LedoitWolf().fit(
        nref
    )
    lw_h = LedoitWolf().fit(
        href
    )

    return (
        normal,
        hcm,
        zn_cols,
        zh_cols,
        lw_n,
        lw_h,
    )


def mahalanobis_distance(
    X,
    estimator,
):
    X = np.asarray(
        X,
        dtype=float,
    )

    d = (
        X
        - estimator.location_[None, :]
    )

    d2 = np.einsum(
        "ni,ij,nj->n",
        d,
        estimator.precision_,
        d,
    )

    d2 = np.maximum(
        d2,
        0.0,
    )

    return (
        np.sqrt(d2),
        d2,
    )


# =====================================================================
# Longitudinal timing
# =====================================================================

def add_longitudinal_time_columns(df):
    dt = pd.to_datetime(
        df["ecg_time"],
        errors="coerce",
    )

    df["ecg_datetime"] = dt

    subject = df[
        "analysis_subject_key"
    ].astype(str)

    first_dt = (
        df.assign(
            _dt=dt
        )
        .groupby(
            "analysis_subject_key"
        )["_dt"]
        .transform("min")
    )

    df["time_from_first_ecg_days"] = (
        dt - first_dt
    ).dt.total_seconds() / 86400.0

    df["time_from_first_ecg_years"] = (
        df["time_from_first_ecg_days"]
        / 365.25
    )

    order_df = df[
        [
            "analysis_subject_key",
            "ecg_datetime",
            "longitudinal_row_index",
        ]
    ].copy()

    order_df["_time_sort"] = (
        order_df["ecg_datetime"]
    )

    order_df = order_df.sort_values(
        [
            "analysis_subject_key",
            "_time_sort",
            "longitudinal_row_index",
        ],
        na_position="last",
    )

    order_df["visit_order_all"] = (
        order_df
        .groupby(
            "analysis_subject_key"
        )
        .cumcount()
        + 1
    )

    df = df.merge(
        order_df[
            [
                "longitudinal_row_index",
                "visit_order_all",
            ]
        ],
        on="longitudinal_row_index",
        how="left",
        validate="one_to_one",
    )

    valid = df[
        df["valid_for_latent_projection"]
    ].copy()

    if len(valid):
        valid_first = (
            valid.groupby(
                "analysis_subject_key"
            )["ecg_datetime"]
            .min()
        )

        df["first_valid_datetime"] = (
            df["analysis_subject_key"]
            .map(valid_first)
        )

        df[
            "time_from_first_valid_ecg_years"
        ] = (
            df["ecg_datetime"]
            - df["first_valid_datetime"]
        ).dt.total_seconds() / (
            86400.0
            * 365.25
        )

        valid_order = (
            valid[
                [
                    "analysis_subject_key",
                    "ecg_datetime",
                    "longitudinal_row_index",
                ]
            ]
            .sort_values(
                [
                    "analysis_subject_key",
                    "ecg_datetime",
                    "longitudinal_row_index",
                ],
                na_position="last",
            )
        )

        valid_order[
            "visit_order_valid"
        ] = (
            valid_order
            .groupby(
                "analysis_subject_key"
            )
            .cumcount()
            + 1
        )

        df = df.merge(
            valid_order[
                [
                    "longitudinal_row_index",
                    "visit_order_valid",
                ]
            ],
            on="longitudinal_row_index",
            how="left",
            validate="one_to_one",
        )

    else:
        df["first_valid_datetime"] = pd.NaT
        df[
            "time_from_first_valid_ecg_years"
        ] = np.nan
        df["visit_order_valid"] = np.nan

    df["is_first_recorded_visit"] = (
        df["visit_order_all"] == 1
    )

    df["is_first_valid_visit"] = (
        df["visit_order_valid"] == 1
    )

    return df


# =====================================================================
# Baseline equivalence check
# =====================================================================

def baseline_consistency(
    long_df,
    normal_baseline_latents,
    hcm_baseline_latents,
    zn_cols,
    zh_cols,
):
    valid = long_df[
        long_df["valid_for_latent_projection"]
    ].copy()

    # Existing baseline Model-1 HCM rows.
    nbase = normal_baseline_latents[
        normal_baseline_latents["split_name"]
        .astype(str)
        .eq("hcm_eval")
    ].copy()

    nbase["_ecg_clean"] = (
        nbase["ecg_id"]
        .map(clean_id)
    )

    # Existing baseline Model-2 HCM rows.
    hbase = hcm_baseline_latents[
        hcm_baseline_latents["split_name"]
        .astype(str)
        .str.startswith("hcm_")
    ].copy()

    hbase["_ecg_clean"] = (
        hbase["ecg_id"]
        .map(clean_id)
    )

    valid["_ecg_clean"] = (
        valid["ecg_id"]
        .map(clean_id)
    )

    nmap = (
        nbase
        .drop_duplicates(
            "_ecg_clean"
        )
        .set_index(
            "_ecg_clean"
        )
    )

    hmap = (
        hbase
        .drop_duplicates(
            "_ecg_clean"
        )
        .set_index(
            "_ecg_clean"
        )
    )

    rows = []

    for _, r in valid.iterrows():
        key = r["_ecg_clean"]

        if (
            key not in nmap.index
            and key not in hmap.index
        ):
            continue

        row = {
            "ecg_id": r["ecg_id"],
            "analysis_subject_key": r[
                "analysis_subject_key"
            ],
            "longitudinal_row_index": int(
                r["longitudinal_row_index"]
            ),
            "matched_normal_baseline": (
                key in nmap.index
            ),
            "matched_hcm_baseline": (
                key in hmap.index
            ),
        }

        if key in nmap.index:
            ref = nmap.loc[key]
            diff = np.asarray(
                [
                    r[f"N_z_{k:02d}"]
                    - ref[
                        f"z_{k:02d}"
                    ]
                    for k in range(
                        NORMAL_LATENT_DIM
                    )
                ],
                dtype=float,
            )

            row[
                "normal_latent_max_abs_difference"
            ] = float(
                np.max(
                    np.abs(diff)
                )
            )

            row[
                "normal_latent_rms_difference"
            ] = float(
                np.sqrt(
                    np.mean(
                        diff ** 2
                    )
                )
            )

        if key in hmap.index:
            ref = hmap.loc[key]
            diff = np.asarray(
                [
                    r[f"H_z_{k:02d}"]
                    - ref[
                        f"z_{k:02d}"
                    ]
                    for k in range(
                        HCM_LATENT_DIM
                    )
                ],
                dtype=float,
            )

            row[
                "hcm_latent_max_abs_difference"
            ] = float(
                np.max(
                    np.abs(diff)
                )
            )

            row[
                "hcm_latent_rms_difference"
            ] = float(
                np.sqrt(
                    np.mean(
                        diff ** 2
                    )
                )
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Subject summary
# =====================================================================

def build_subject_summary(df):
    rows = []

    for subject, g in df.groupby(
        "analysis_subject_key",
        sort=False,
    ):
        g = g.sort_values(
            [
                "ecg_datetime",
                "longitudinal_row_index",
            ],
            na_position="last",
        )

        valid = g[
            g["valid_for_latent_projection"]
        ]

        t = pd.to_numeric(
            g[
                "time_from_first_ecg_years"
            ],
            errors="coerce",
        )

        tv = pd.to_numeric(
            valid[
                "time_from_first_ecg_years"
            ],
            errors="coerce",
        )

        rows.append({
            "analysis_subject_key": subject,
            "subject_index": (
                int(g["subject_index"].iloc[0])
                if pd.notna(
                    g["subject_index"].iloc[0]
                )
                else np.nan
            ),
            "n_visits_total": int(
                len(g)
            ),
            "n_visits_valid": int(
                len(valid)
            ),
            "n_visits_invalid": int(
                len(g) - len(valid)
            ),
            "followup_years_total": (
                float(
                    np.nanmax(t)
                    - np.nanmin(t)
                )
                if np.sum(
                    np.isfinite(t)
                ) >= 2
                else 0.0
            ),
            "followup_years_valid": (
                float(
                    np.nanmax(tv)
                    - np.nanmin(tv)
                )
                if np.sum(
                    np.isfinite(tv)
                ) >= 2
                else 0.0
            ),
            "has_2plus_valid_visits": (
                len(valid) >= 2
            ),
            "has_3plus_valid_visits": (
                len(valid) >= 3
            ),
            "first_ecg_time": (
                str(
                    g[
                        "ecg_datetime"
                    ].min()
                )
            ),
            "last_ecg_time": (
                str(
                    g[
                        "ecg_datetime"
                    ].max()
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Plots
# =====================================================================

def make_plots(
    subject_summary,
    outdir,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    vals = (
        subject_summary[
            "n_visits_valid"
        ]
        .to_numpy(
            dtype=float
        )
    )

    bins = np.arange(
        0.5,
        max(
            2.5,
            np.nanmax(vals) + 1.5,
        ),
        1.0,
    )

    ax.hist(
        vals[
            np.isfinite(vals)
        ],
        bins=bins,
    )

    ax.set_xlabel(
        "Valid longitudinal ECG visits per subject"
    )
    ax.set_ylabel(
        "Subjects"
    )
    ax.set_title(
        "Longitudinal visit distribution"
    )

    fig.tight_layout()
    fig.savefig(
        outdir
        / "visits_per_subject.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    follow = subject_summary[
        "followup_years_valid"
    ].to_numpy(
        dtype=float
    )

    ax.hist(
        follow[
            np.isfinite(follow)
        ],
        bins=30,
    )

    ax.set_xlabel(
        "Valid-visit follow-up duration (years)"
    )
    ax.set_ylabel(
        "Subjects"
    )
    ax.set_title(
        "Longitudinal follow-up"
    )

    fig.tight_layout()
    fig.savefig(
        outdir
        / "followup_duration_years.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    if args.min_beats < 1:
        raise ValueError(
            "--min-beats must be >= 1."
        )

    if not (
        0
        < args.morph_similarity_threshold
        <= 1
    ):
        raise ValueError(
            "--morph-similarity-threshold must be in (0,1]."
        )

    required = [
        args.input,
        args.normal_checkpoint,
        args.normal_latents,
        args.hcm_prepared_h5,
        args.hcm_checkpoint,
        args.hcm_latents,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Required input not found: {p}"
            )

    args.output_h5.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device(
        args.device
    )

    print("=" * 78)
    print(
        "16 - Build longitudinal dual-manifold dataset"
    )
    print("=" * 78)
    print(
        "Input              :",
        args.input,
    )
    print(
        "Normal checkpoint  :",
        args.normal_checkpoint,
    )
    print(
        "HCM checkpoint     :",
        args.hcm_checkpoint,
    )
    print(
        "Output H5          :",
        args.output_h5,
    )
    print(
        "Results            :",
        args.results_dir,
    )
    print(
        "Device             :",
        device,
    )
    print()

    models = load_frozen_models(
        args.normal_checkpoint,
        args.hcm_checkpoint,
        args.hcm_prepared_h5,
        device,
    )

    print(
        "Normal scaling     :",
        models["normal_mean_mv"],
        models["normal_std_mv"],
    )
    print(
        "HCM scaling        :",
        models["hcm_mean_mv"],
        models["hcm_std_mv"],
    )
    print()

    (
        normal_baseline_latents,
        hcm_baseline_latents,
        zn_cols,
        zh_cols,
        lw_n,
        lw_h,
    ) = fit_reference_distances(
        args.normal_latents,
        args.hcm_latents,
    )

    sos = make_filter()

    pre_samples = int(
        round(
            PRE_MS
            / 1000.0
            * FS_HZ
        )
    )

    post_samples = int(
        round(
            POST_MS
            / 1000.0
            * FS_HZ
        )
    )

    if (
        pre_samples
        + post_samples
        != BEAT_SAMPLES
    ):
        raise RuntimeError(
            "Unexpected beat sample count."
        )

    # -------------------------------------------------------------
    # Read metadata and construct longitudinal median beats
    # -------------------------------------------------------------
    with h5py.File(
        args.input,
        "r",
    ) as h5:

        if "signals/ecg" not in h5:
            raise KeyError(
                "Input HDF5 has no signals/ecg dataset."
            )

        signal_ds = h5[
            "signals/ecg"
        ]

        if (
            signal_ds.ndim != 3
            or tuple(
                signal_ds.shape[1:]
            )
            != (
                12,
                2000,
            )
        ):
            raise ValueError(
                "Expected input ECG shape [N,12,2000]; "
                f"found {signal_ds.shape}."
            )

        n = int(
            signal_ds.shape[0]
        )

        def meta_str(name):
            path = f"metadata/{name}"
            if path not in h5:
                return np.asarray(
                    [""] * n,
                    dtype=object,
                )
            return decode_string_array(
                h5[path][:]
            )

        ecg_id = meta_str(
            "ecg_id"
        )
        patient_id = meta_str(
            "patient_id"
        )
        subject_key = meta_str(
            "subject_key"
        )
        ecg_time = meta_str(
            "ecg_time"
        )
        birthdate = meta_str(
            "birthdate"
        )

        if "metadata/subject_index" not in h5:
            raise KeyError(
                "Longitudinal HDF5 has no metadata/subject_index."
            )

        subject_index = np.asarray(
            h5[
                "metadata/subject_index"
            ][:]
        ).astype(np.int64)

        if (
            "metadata/ecg_number_per_patient"
            in h5
        ):
            ecg_number_per_patient = np.asarray(
                h5[
                    "metadata/ecg_number_per_patient"
                ][:]
            ).astype(np.int32)
        else:
            ecg_number_per_patient = np.full(
                n,
                -1,
                dtype=np.int32,
            )

        print(
            f"Longitudinal ECG records: {n:,}"
        )
        print(
            "Coherent subject_index values:",
            len(
                np.unique(
                    subject_index
                )
            ),
        )
        print()

        median_all = np.full(
            (
                n,
                12,
                BEAT_SAMPLES,
            ),
            np.nan,
            dtype=np.float32,
        )

        median_dominant = np.full_like(
            median_all,
            np.nan,
        )

        qc_rows = []

        for i in range(n):
            raw = np.asarray(
                signal_ds[i],
                dtype=np.float32,
            )

            if not np.isfinite(
                raw
            ).all():
                filtered = np.full_like(
                    raw,
                    np.nan,
                )
                peaks = np.asarray(
                    [],
                    dtype=np.int32,
                )
                beats = np.empty(
                    (
                        0,
                        12,
                        BEAT_SAMPLES,
                    ),
                    dtype=np.float32,
                )
                used_peaks = np.asarray(
                    [],
                    dtype=np.int32,
                )
                excluded_edge = 0
                excluded_nonfinite = 0

            else:
                # Exact Phase-I filter.
                filtered = sosfiltfilt(
                    sos,
                    raw,
                    axis=-1,
                ).astype(np.float32)

                peaks = detect_multilead_qrs(
                    filtered,
                    FS_HZ,
                )

                (
                    beats,
                    used_peaks,
                    excluded_edge,
                    excluded_nonfinite,
                ) = extract_complete_beats(
                    filtered,
                    peaks,
                    pre_samples,
                    post_samples,
                )

            n_detected = int(
                len(peaks)
            )

            n_usable = int(
                len(beats)
            )

            rr = compute_rr_metrics(
                peaks,
                FS_HZ,
            )

            morph = {
                "morph_pairwise_similarity_median": np.nan,
                "morph_pairwise_similarity_min": np.nan,
                "morph_rms_to_median_mv": np.nan,
                "morph_rms_relative_to_template": np.nan,
                "dominant_cluster_size": 0,
                "dominant_cluster_fraction": np.nan,
                "morph_components_at_threshold": 0,
            }

            if n_usable > 0:
                med_all = np.median(
                    beats,
                    axis=0,
                ).astype(np.float32)

                similarity = (
                    morphology_similarity_matrix(
                        beats
                    )
                )

                dominant_idx = (
                    dominant_morphology_indices(
                        similarity,
                        args.morph_similarity_threshold,
                    )
                )

                med_dom = np.median(
                    beats[
                        dominant_idx
                    ],
                    axis=0,
                ).astype(np.float32)

                median_all[i] = med_all
                median_dominant[i] = med_dom

                morph = compute_morph_metrics(
                    beats,
                    med_all,
                    similarity,
                    dominant_idx,
                    args.morph_similarity_threshold,
                )

            valid = (
                n_usable
                >= args.min_beats
                and np.isfinite(
                    median_all[i]
                ).all()
            )

            row = {
                "longitudinal_row_index": i,
                "ecg_id": ecg_id[i],
                "patient_id": patient_id[i],
                "subject_key": subject_key[i],
                "subject_index": int(
                    subject_index[i]
                ),
                "ecg_time": ecg_time[i],
                "birthdate": birthdate[i],
                "ecg_number_per_patient": int(
                    ecg_number_per_patient[i]
                ),
                "n_detected_qrs": n_detected,
                "n_usable_beats": n_usable,
                "n_excluded_edge": int(
                    excluded_edge
                ),
                "n_excluded_nonfinite": int(
                    excluded_nonfinite
                ),
                "qc_label": qc_label(
                    n_usable
                ),
                "valid_for_latent_projection": bool(
                    valid
                ),
                **rr,
                **morph,
            }

            qc_rows.append(row)

            if (
                i == 0
                or (i + 1) % 250 == 0
                or i == n - 1
            ):
                print(
                    f"[median beats] {i+1:,}/{n:,}"
                )

    qc_df = pd.DataFrame(
        qc_rows
    )

    # Prefer subject_key, but do not allow empty strings to collapse subjects.
    subject_clean = (
        qc_df["subject_key"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    qc_df[
        "analysis_subject_key"
    ] = np.where(
        subject_clean.str.len() > 0,
        subject_clean,
        "SUBJECT_INDEX_"
        + qc_df[
            "subject_index"
        ].astype(str),
    )

    n_subjects = qc_df[
        "analysis_subject_key"
    ].nunique()

    if n_subjects != 650:
        warnings.warn(
            "Expected 650 coherent HCM subjects; "
            f"found {n_subjects}."
        )

    valid_mask = qc_df[
        "valid_for_latent_projection"
    ].to_numpy(
        dtype=bool
    )

    valid_idx = np.where(
        valid_mask
    )[0]

    print()
    print(
        "Valid visits for dual projection:",
        f"{len(valid_idx):,}/{n:,}",
    )

    # -------------------------------------------------------------
    # Project valid visits through both frozen VAEs
    # -------------------------------------------------------------
    x_valid = median_all[
        valid_idx
    ]

    (
        n_mu,
        n_logvar,
        n_recon_mv,
    ) = encode_reconstruct(
        models["normal_model"],
        x_valid,
        models["normal_mean_mv"],
        models["normal_std_mv"],
        args.batch_size,
        device,
    )

    (
        h_mu,
        h_logvar,
        h_recon_mv,
    ) = encode_reconstruct(
        models["hcm_model"],
        x_valid,
        models["hcm_mean_mv"],
        models["hcm_std_mv"],
        args.batch_size,
        device,
    )

    n_res = compute_reconstruction_metrics(
        x_valid,
        n_recon_mv,
    )

    h_res = compute_reconstruction_metrics(
        x_valid,
        h_recon_mv,
    )

    d_n, d_n2 = mahalanobis_distance(
        n_mu,
        lw_n,
    )

    d_h, d_h2 = mahalanobis_distance(
        h_mu,
        lw_h,
    )

    # Create full-length output arrays.
    def full_nan(shape):
        return np.full(
            shape,
            np.nan,
            dtype=np.float32,
        )

    N_mu_full = full_nan(
        (
            n,
            NORMAL_LATENT_DIM,
        )
    )
    N_logvar_full = full_nan(
        (
            n,
            NORMAL_LATENT_DIM,
        )
    )
    H_mu_full = full_nan(
        (
            n,
            HCM_LATENT_DIM,
        )
    )
    H_logvar_full = full_nan(
        (
            n,
            HCM_LATENT_DIM,
        )
    )

    N_mu_full[valid_idx] = n_mu
    N_logvar_full[valid_idx] = n_logvar
    H_mu_full[valid_idx] = h_mu
    H_logvar_full[valid_idx] = h_logvar

    scalar_outputs = {
        "d_N": np.full(
            n,
            np.nan,
            dtype=np.float32,
        ),
        "d_N2": np.full(
            n,
            np.nan,
            dtype=np.float32,
        ),
        "d_HCM": np.full(
            n,
            np.nan,
            dtype=np.float32,
        ),
        "d_HCM2": np.full(
            n,
            np.nan,
            dtype=np.float32,
        ),
    }

    scalar_outputs[
        "d_N"
    ][valid_idx] = d_n

    scalar_outputs[
        "d_N2"
    ][valid_idx] = d_n2

    scalar_outputs[
        "d_HCM"
    ][valid_idx] = d_h

    scalar_outputs[
        "d_HCM2"
    ][valid_idx] = d_h2

    for prefix, metrics in [
        ("RE_N", n_res),
        ("RE_H", h_res),
    ]:
        for key, vals in metrics.items():
            name = (
                f"{prefix}_{key}"
            )

            scalar_outputs[name] = np.full(
                n,
                np.nan,
                dtype=np.float32,
            )

            scalar_outputs[name][
                valid_idx
            ] = vals

    scalar_outputs[
        "Delta_RE_mse"
    ] = (
        scalar_outputs["RE_N_mse"]
        - scalar_outputs["RE_H_mse"]
    )

    scalar_outputs[
        "log_RE_N_over_RE_H"
    ] = np.log(
        (
            scalar_outputs["RE_N_mse"]
            + EPS
        )
        / (
            scalar_outputs["RE_H_mse"]
            + EPS
        )
    ).astype(np.float32)

    # -------------------------------------------------------------
    # Assemble longitudinal table
    # -------------------------------------------------------------
    long_df = qc_df.copy()

    for k in range(
        NORMAL_LATENT_DIM
    ):
        long_df[
            f"N_z_{k:02d}"
        ] = N_mu_full[:, k]

        long_df[
            f"N_logvar_{k:02d}"
        ] = N_logvar_full[:, k]

    for k in range(
        HCM_LATENT_DIM
    ):
        long_df[
            f"H_z_{k:02d}"
        ] = H_mu_full[:, k]

        long_df[
            f"H_logvar_{k:02d}"
        ] = H_logvar_full[:, k]

    for name, vals in (
        scalar_outputs.items()
    ):
        long_df[name] = vals

    long_df = (
        add_longitudinal_time_columns(
            long_df
        )
    )

    # -------------------------------------------------------------
    # Baseline consistency with existing cross-sectional projections
    # -------------------------------------------------------------
    baseline_qc = baseline_consistency(
        long_df,
        normal_baseline_latents,
        hcm_baseline_latents,
        zn_cols,
        zh_cols,
    )

    baseline_qc.to_csv(
        args.results_dir
        / "baseline_projection_consistency.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Subject summary
    # -------------------------------------------------------------
    subject_summary = (
        build_subject_summary(
            long_df
        )
    )

    subject_summary.to_csv(
        args.results_dir
        / "longitudinal_subject_summary.csv",
        index=False,
    )

    long_df.to_csv(
        args.results_dir
        / "longitudinal_dual_manifold.csv",
        index=False,
    )

    qc_cols = [
        "longitudinal_row_index",
        "ecg_id",
        "analysis_subject_key",
        "subject_index",
        "ecg_time",
        "n_detected_qrs",
        "n_usable_beats",
        "qc_label",
        "valid_for_latent_projection",
        "rr_median_ms",
        "rr_sd_ms",
        "rr_cv",
        "rmssd_ms",
        "heart_rate_bpm",
        "morph_pairwise_similarity_median",
        "morph_pairwise_similarity_min",
        "morph_rms_to_median_mv",
        "morph_rms_relative_to_template",
        "dominant_cluster_fraction",
        "morph_components_at_threshold",
    ]

    long_df[
        [
            c
            for c in qc_cols
            if c in long_df.columns
        ]
    ].to_csv(
        args.results_dir
        / "longitudinal_qc.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Write HDF5
    # -------------------------------------------------------------
    with h5py.File(
        args.output_h5,
        "w",
    ) as h5:

        h5.attrs[
            "description"
        ] = (
            "SMASH-HCM longitudinal frozen dual-manifold representation"
        )

        h5.attrs[
            "sampling_rate_hz"
        ] = FS_HZ

        h5.attrs[
            "filter"
        ] = (
            "Butterworth band-pass 0.5-40 Hz, order 4, zero-phase"
        )

        h5.attrs[
            "beat_window_ms"
        ] = (
            "-350 to +650 ms"
        )

        h5.attrs[
            "minimum_usable_beats"
        ] = int(
            args.min_beats
        )

        h5.attrs[
            "normal_train_scaling_mean_mv"
        ] = models[
            "normal_mean_mv"
        ]

        h5.attrs[
            "normal_train_scaling_std_mv"
        ] = models[
            "normal_std_mv"
        ]

        h5.attrs[
            "hcm_train_scaling_mean_mv"
        ] = models[
            "hcm_mean_mv"
        ]

        h5.attrs[
            "hcm_train_scaling_std_mv"
        ] = models[
            "hcm_std_mv"
        ]

        h5.attrs[
            "models_frozen"
        ] = True

        sg = h5.create_group(
            "signals"
        )

        sg.create_dataset(
            "median_all_mv",
            data=median_all,
            compression="gzip",
            compression_opts=4,
            chunks=(
                1,
                12,
                BEAT_SAMPLES,
            ),
        )

        sg.create_dataset(
            "median_dominant_mv",
            data=median_dominant,
            compression="gzip",
            compression_opts=4,
            chunks=(
                1,
                12,
                BEAT_SAMPLES,
            ),
        )

        write_str_dataset(
            sg,
            "lead_names",
            LEAD_NAMES,
        )

        mg = h5.create_group(
            "metadata"
        )

        for name in [
            "ecg_id",
            "patient_id",
            "subject_key",
            "analysis_subject_key",
            "ecg_time",
            "birthdate",
        ]:
            write_str_dataset(
                mg,
                name,
                long_df[name].values,
            )

        for name in [
            "subject_index",
            "ecg_number_per_patient",
            "visit_order_all",
            "visit_order_valid",
        ]:
            vals = pd.to_numeric(
                long_df[name],
                errors="coerce",
            ).fillna(-1).to_numpy(
                dtype=np.int32
            )

            mg.create_dataset(
                name,
                data=vals,
                compression="gzip",
                compression_opts=4,
            )

        for name in [
            "time_from_first_ecg_days",
            "time_from_first_ecg_years",
            "time_from_first_valid_ecg_years",
        ]:
            vals = pd.to_numeric(
                long_df[name],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32
            )

            mg.create_dataset(
                name,
                data=vals,
                compression="gzip",
                compression_opts=4,
            )

        qg = h5.create_group(
            "quality"
        )

        qg.create_dataset(
            "valid_for_latent_projection",
            data=valid_mask.astype(
                np.uint8
            ),
            compression="gzip",
            compression_opts=4,
        )

        for name in [
            "n_detected_qrs",
            "n_usable_beats",
            "dominant_cluster_size",
            "morph_components_at_threshold",
        ]:
            qg.create_dataset(
                name,
                data=pd.to_numeric(
                    long_df[name],
                    errors="coerce",
                ).fillna(-1).to_numpy(
                    dtype=np.int16
                ),
                compression="gzip",
                compression_opts=4,
            )

        rg = h5.create_group(
            "rhythm"
        )

        for name in [
            "rr_mean_ms",
            "rr_median_ms",
            "rr_sd_ms",
            "rr_mad_ms",
            "rr_cv",
            "rmssd_ms",
            "heart_rate_bpm",
        ]:
            rg.create_dataset(
                name,
                data=pd.to_numeric(
                    long_df[name],
                    errors="coerce",
                ).to_numpy(
                    dtype=np.float32
                ),
                compression="gzip",
                compression_opts=4,
            )

        vg = h5.create_group(
            "variability"
        )

        for name in [
            "morph_pairwise_similarity_median",
            "morph_pairwise_similarity_min",
            "morph_rms_to_median_mv",
            "morph_rms_relative_to_template",
            "dominant_cluster_fraction",
        ]:
            vg.create_dataset(
                name,
                data=pd.to_numeric(
                    long_df[name],
                    errors="coerce",
                ).to_numpy(
                    dtype=np.float32
                ),
                compression="gzip",
                compression_opts=4,
            )

        ng = h5.create_group(
            "normal_manifold"
        )

        ng.create_dataset(
            "z_mu",
            data=N_mu_full,
            compression="gzip",
            compression_opts=4,
        )

        ng.create_dataset(
            "logvar",
            data=N_logvar_full,
            compression="gzip",
            compression_opts=4,
        )

        ng.create_dataset(
            "d_N",
            data=scalar_outputs["d_N"],
            compression="gzip",
            compression_opts=4,
        )

        for key in [
            "RE_N_mse",
            "RE_N_rmse",
            "RE_N_mse_preQRS",
            "RE_N_mse_QRS",
            "RE_N_mse_STT",
            "RE_N_STT_to_QRS_mse_ratio",
        ]:
            ng.create_dataset(
                key.replace(
                    "RE_N_",
                    "",
                ),
                data=scalar_outputs[key],
                compression="gzip",
                compression_opts=4,
            )

        hg = h5.create_group(
            "hcm_manifold"
        )

        hg.create_dataset(
            "z_mu",
            data=H_mu_full,
            compression="gzip",
            compression_opts=4,
        )

        hg.create_dataset(
            "logvar",
            data=H_logvar_full,
            compression="gzip",
            compression_opts=4,
        )

        hg.create_dataset(
            "d_HCM",
            data=scalar_outputs["d_HCM"],
            compression="gzip",
            compression_opts=4,
        )

        for key in [
            "RE_H_mse",
            "RE_H_rmse",
            "RE_H_mse_preQRS",
            "RE_H_mse_QRS",
            "RE_H_mse_STT",
            "RE_H_STT_to_QRS_mse_ratio",
        ]:
            hg.create_dataset(
                key.replace(
                    "RE_H_",
                    "",
                ),
                data=scalar_outputs[key],
                compression="gzip",
                compression_opts=4,
            )

        cg = h5.create_group(
            "cross_manifold"
        )

        cg.create_dataset(
            "Delta_RE_mse",
            data=scalar_outputs[
                "Delta_RE_mse"
            ],
            compression="gzip",
            compression_opts=4,
        )

        cg.create_dataset(
            "log_RE_N_over_RE_H",
            data=scalar_outputs[
                "log_RE_N_over_RE_H"
            ],
            compression="gzip",
            compression_opts=4,
        )

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------
    make_plots(
        subject_summary,
        args.results_dir,
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    n_valid = int(
        valid_mask.sum()
    )

    n_subjects_valid = int(
        subject_summary[
            "n_visits_valid"
        ].gt(0).sum()
    )

    n_2plus = int(
        subject_summary[
            "has_2plus_valid_visits"
        ].sum()
    )

    n_3plus = int(
        subject_summary[
            "has_3plus_valid_visits"
        ].sum()
    )

    follow = subject_summary[
        "followup_years_valid"
    ].to_numpy(
        dtype=float
    )

    visits = subject_summary[
        "n_visits_valid"
    ].to_numpy(
        dtype=float
    )

    baseline_stats = {
        "n_rows": int(
            len(baseline_qc)
        ),
        "n_normal_latent_matches": int(
            baseline_qc[
                "matched_normal_baseline"
            ].sum()
        )
        if len(baseline_qc)
        else 0,
        "n_hcm_latent_matches": int(
            baseline_qc[
                "matched_hcm_baseline"
            ].sum()
        )
        if len(baseline_qc)
        else 0,
    }

    if (
        len(baseline_qc)
        and "normal_latent_max_abs_difference"
        in baseline_qc
    ):
        baseline_stats[
            "normal_latent_max_abs_difference_median"
        ] = float(
            np.nanmedian(
                baseline_qc[
                    "normal_latent_max_abs_difference"
                ]
            )
        )

        baseline_stats[
            "normal_latent_max_abs_difference_max"
        ] = float(
            np.nanmax(
                baseline_qc[
                    "normal_latent_max_abs_difference"
                ]
            )
        )

    if (
        len(baseline_qc)
        and "hcm_latent_max_abs_difference"
        in baseline_qc
    ):
        baseline_stats[
            "hcm_latent_max_abs_difference_median"
        ] = float(
            np.nanmedian(
                baseline_qc[
                    "hcm_latent_max_abs_difference"
                ]
            )
        )

        baseline_stats[
            "hcm_latent_max_abs_difference_max"
        ] = float(
            np.nanmax(
                baseline_qc[
                    "hcm_latent_max_abs_difference"
                ]
            )
        )

    summary = {
        "script": (
            "16_build_longitudinal_dual_manifold_dataset.py"
        ),
        "version": "1.0",
        "scientific_role": (
            "Frozen dual-manifold projection of all longitudinal HCM ECG visits"
        ),
        "input": str(
            args.input
        ),
        "output_h5": str(
            args.output_h5
        ),
        "results_dir": str(
            args.results_dir
        ),
        "n_longitudinal_records": int(
            n
        ),
        "n_coherent_subjects": int(
            n_subjects
        ),
        "n_valid_visits": n_valid,
        "valid_visit_fraction": float(
            n_valid / n
        ),
        "n_subjects_with_valid_visits": n_subjects_valid,
        "n_subjects_with_2plus_valid_visits": n_2plus,
        "n_subjects_with_3plus_valid_visits": n_3plus,
        "median_valid_visits_per_subject": float(
            np.nanmedian(
                visits
            )
        ),
        "median_valid_followup_years": float(
            np.nanmedian(
                follow
            )
        ),
        "max_valid_followup_years": float(
            np.nanmax(
                follow
            )
        ),
        "preprocessing": {
            "sampling_rate_hz": FS_HZ,
            "filter": (
                "Butterworth band-pass 0.5-40 Hz, order 4, zero-phase"
            ),
            "beat_window_ms": [
                -PRE_MS,
                POST_MS,
            ],
            "beat_samples": BEAT_SAMPLES,
            "minimum_usable_beats": int(
                args.min_beats
            ),
            "morph_similarity_threshold": float(
                args.morph_similarity_threshold
            ),
            "morphologically_different_beats_removed": False,
        },
        "normal_model": {
            "checkpoint": str(
                args.normal_checkpoint
            ),
            "latent_dim": NORMAL_LATENT_DIM,
            "parameters": int(
                models[
                    "normal_params"
                ]
            ),
            "scaling_mean_mv": float(
                models[
                    "normal_mean_mv"
                ]
            ),
            "scaling_std_mv": float(
                models[
                    "normal_std_mv"
                ]
            ),
            "reference_for_d_N": (
                "Normal-train posterior means with Ledoit-Wolf covariance"
            ),
        },
        "hcm_model": {
            "checkpoint": str(
                args.hcm_checkpoint
            ),
            "latent_dim": HCM_LATENT_DIM,
            "parameters": int(
                models[
                    "hcm_params"
                ]
            ),
            "scaling_mean_mv": float(
                models[
                    "hcm_mean_mv"
                ]
            ),
            "scaling_std_mv": float(
                models[
                    "hcm_std_mv"
                ]
            ),
            "reference_for_d_HCM": (
                "HCM-train posterior means with Ledoit-Wolf covariance"
            ),
        },
        "baseline_projection_consistency": baseline_stats,
        "longitudinal_time_origin": (
            "earliest recorded ECG within each coherent analysis_subject_key"
        ),
        "important_cautions": [
            "The HCM sampling rate remains a working 500-Hz assumption inherited from the harmonization stage.",
            "Both VAEs are frozen; no longitudinal ECG contributes to representation learning.",
            "RE_N and RE_H are physical MSE values but arise from models with different capacity and training populations.",
            "Delta_RE is descriptive and is not interpreted as a direct measure of HCM specificity.",
            "Median-beat morphology does not represent rhythm or beat-to-beat variability; those descriptors are retained separately.",
            "Longitudinal mixed-effects modeling should use only visits passing the minimum-beat QC criterion.",
        ],
    }

    (
        args.results_dir
        / "run_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "STEP 16 COMPLETE"
    )
    print("=" * 78)
    print(
        "Longitudinal records:",
        n,
    )
    print(
        "Coherent subjects    :",
        n_subjects,
    )
    print(
        "Valid visits         :",
        n_valid,
        f"({100*n_valid/n:.2f}%)",
    )
    print(
        "Subjects >=2 valid   :",
        n_2plus,
    )
    print(
        "Subjects >=3 valid   :",
        n_3plus,
    )
    print(
        "Median follow-up     :",
        f"{np.nanmedian(follow):.3f} years",
    )
    print(
        "Maximum follow-up    :",
        f"{np.nanmax(follow):.3f} years",
    )

    if len(baseline_qc):
        print()
        print(
            "[baseline projection consistency]"
        )

        if (
            "normal_latent_max_abs_difference"
            in baseline_qc
        ):
            x = baseline_qc[
                "normal_latent_max_abs_difference"
            ]
            print(
                "Normal latent max |difference|:",
                f"median={np.nanmedian(x):.6g}",
                f"max={np.nanmax(x):.6g}",
            )

        if (
            "hcm_latent_max_abs_difference"
            in baseline_qc
        ):
            x = baseline_qc[
                "hcm_latent_max_abs_difference"
            ]
            print(
                "HCM latent max |difference|   :",
                f"median={np.nanmedian(x):.6g}",
                f"max={np.nanmax(x):.6g}",
            )

    print()
    print(
        "HDF5:",
        args.output_h5,
    )
    print(
        "Results:",
        args.results_dir,
    )


if __name__ == "__main__":
    main()
