#!/usr/bin/env python
"""
build_phase1_median_beats.py

SMASH-HCM -- Phase I median-beat construction with preservation of
beat-to-beat morphology and rhythm variability.

Concept
-------
The 4-s 12-lead ECG is NOT replaced by a median beat without qualification.

For each ECG this script produces:
1. an all-beat median waveform:
       median_all(t, lead) = median across every usable aligned beat
2. a dominant-morphology median waveform:
       median_dominant(t, lead) = median across the largest coherent
       morphology neighborhood
3. beat-to-beat morphology variability metrics
4. rhythm / RR variability metrics
5. QC labels and beat counts

Morphologically different beats are NEVER discarded merely because they
differ from the dominant waveform. The only beats excluded from template
construction are those that cannot provide a complete extraction window
or contain non-finite samples.

This preserves the distinction:

    ECG phenotype =
        dominant morphology
        + beat-to-beat morphology variability
        + rhythm variability

Input
-----
By default:
    Results/03_Phase1_Normal_Anchored/Phase1_NormalAnchored_preprocessed.h5

This is the previously filtered (0.5--40 Hz) and globally standardized
Phase-I dataset. The script restores the signals to mV before beat
construction using the Phase-I global scaling constants.

The same detected R fiducials are used for all 12 leads.

Output
------
Results/04_Phase1_Median_Beats/Phase1_MedianBeats_v1.h5

Results/04_Phase1_Median_Beats/
    median_beat_qc.csv
    median_beat_summary.json
    qc_figures/*.png

HDF5 signal datasets
--------------------
/signals/median_all_mv          [N, 12, 500]
/signals/median_dominant_mv     [N, 12, 500]

The extraction window is, by default:
    -350 ms to +650 ms relative to the common multilead R fiducial.

At 500 Hz this gives exactly 500 samples.

QC convention
-------------
good        : >= 3 complete usable beats
acceptable  : 2 complete usable beats
limited     : 1 complete usable beat
failed      : 0 complete usable beats

The script does NOT remove ECGs from the dataset. A provisional
"valid_for_vae" flag is stored separately and is configurable through
--min-beats-for-vae (default 2). This can be changed later after examining
the actual beat-count distribution.

Requirements
------------
pip install numpy pandas h5py scipy matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


SEED = 20261004
FS_HZ_DEFAULT = 500.0
PRE_MS_DEFAULT = 350.0
POST_MS_DEFAULT = 650.0

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

# Exact values previously estimated from normal_train only in Phase I.
FALLBACK_GLOBAL_MEAN_MV = 1.5854158125689152e-05
FALLBACK_GLOBAL_STD_MV = 0.1798839974179138


def infer_project_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if (root / "Data").exists():
        return root

    cwd = Path.cwd().resolve()
    if (cwd / "Data").exists():
        return cwd

    return root


def decode_h5_scalar_or_array(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x


def load_scaling(root: Path, explicit_summary: Path | None):
    candidates = []

    if explicit_summary is not None:
        candidates.append(explicit_summary)

    candidates += [
        root / "Results" / "03_Phase1_Normal_Anchored" / "phase1_preprocessing_summary.json",
        root / "Results" / "Step_03" / "phase1_preprocessing_summary.json",
        root / "Data" / "Harmonized" / "phase1_preprocessing_summary.json",
        root / "phase1_preprocessing_summary.json",
    ]

    for p in candidates:
        if p is None or not p.exists():
            continue

        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            s = d.get("scaling", {})
            mean_mv = s.get("mean_mv")
            std_mv = s.get("std_mv")

            if mean_mv is not None and std_mv is not None:
                std_mv = float(std_mv)

                if std_mv > 0:
                    return (
                        float(mean_mv),
                        std_mv,
                        f"preprocessing summary: {p}",
                    )
        except Exception:
            pass

    return (
        FALLBACK_GLOBAL_MEAN_MV,
        FALLBACK_GLOBAL_STD_MV,
        "fallback Phase-I scaling constants",
    )


def get_signal_dataset(h5: h5py.File):
    candidates = [
        "signals/ecg",
        "signals",
        "ecg",
    ]

    for key in candidates:
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            ds = h5[key]
            if ds.ndim == 3:
                return key, ds

    # Recursive fallback.
    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            if tuple(obj.shape[1:]) == (12, 2000):
                found.append((name, obj))

    h5.visititems(visit)

    if not found:
        raise KeyError(
            "Could not locate a 3-D ECG signal dataset in the input HDF5."
        )

    return found[0]


def robust_scale_per_lead(x: np.ndarray) -> np.ndarray:
    """
    Robust lead-wise scaling FOR QRS DETECTION ONLY.

    This does not modify the waveform used for the median beat.
    """
    med = np.median(x, axis=1, keepdims=True)
    centered = x - med

    mad = np.median(
        np.abs(centered),
        axis=1,
        keepdims=True,
    )

    scale = 1.4826 * mad

    # Fall back to standard deviation for extremely flat/low-MAD leads.
    std = np.std(centered, axis=1, keepdims=True)

    scale = np.where(scale > 1e-6, scale, std)
    scale = np.where(scale > 1e-6, scale, 1.0)

    return centered / scale


def detect_multilead_qrs(
    x_mv: np.ndarray,
    fs_hz: float,
):
    """
    Detect common multilead QRS fiducials.

    Strategy
    --------
    1. Robustly standardize each lead for detection only.
    2. Compute multilead derivative energy.
    3. Smooth over ~80 ms.
    4. Detect energy peaks with an adaptive robust threshold.
    5. Refine each fiducial to the local maximum of multilead voltage
       magnitude within +/-60 ms.
    6. Deduplicate close detections.

    Returns
    -------
    peaks : int ndarray
        Common R/QRS fiducial sample indices.
    detection_score : ndarray
        Smoothed multilead QRS activity.
    composite_magnitude : ndarray
        Robust multilead voltage magnitude used for refinement.
    """

    if x_mv.shape[0] != 12:
        raise ValueError("Expected 12 leads.")

    z = robust_scale_per_lead(x_mv)

    grad = np.gradient(z, axis=1)
    energy_leads = grad ** 2

    # Median across leads limits the influence of one noisy channel.
    energy = np.median(energy_leads, axis=0)

    smooth_samples = max(3, int(round(0.080 * fs_hz)))
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

    p99 = float(np.percentile(detection_score, 99.0))

    min_distance = max(
        1,
        int(round(0.25 * fs_hz))
    )

    # First, a conservative threshold.
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

    # Relax adaptively if a 4-s ECG has fewer than two candidates.
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

    # Last-resort morphology-agnostic peak search.
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

        # Retain only the strongest plausible candidates.
        if len(peaks) > 0:
            order = np.argsort(
                detection_score[peaks]
            )[::-1]

            # 4 seconds; allow up to 16 candidates (~240 bpm).
            peaks = np.sort(
                peaks[order[:16]]
            )

    composite = np.sqrt(
        np.mean(z ** 2, axis=0)
    )

    refine_radius = max(
        1,
        int(round(0.060 * fs_hz))
    )

    refined = []

    for p in peaks:
        a = max(0, int(p) - refine_radius)
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
        return (
            np.asarray([], dtype=np.int32),
            detection_score,
            composite,
        )

    refined = np.asarray(
        sorted(set(refined)),
        dtype=np.int32,
    )

    # Deduplicate refinements that landed on the same QRS.
    min_refined_distance = max(
        1,
        int(round(0.20 * fs_hz))
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

    return (
        np.asarray(dedup, dtype=np.int32),
        detection_score,
        composite,
    )


def extract_complete_beats(
    x_mv: np.ndarray,
    peaks: np.ndarray,
    pre_samples: int,
    post_samples: int,
):
    """
    Extract only complete finite windows.

    IMPORTANT:
    No beat is rejected because its morphology differs from the others.
    """
    beats = []
    used_peaks = []
    excluded_edge = 0
    excluded_nonfinite = 0

    n_samples = x_mv.shape[1]

    for p in peaks:
        start = int(p) - pre_samples
        stop = int(p) + post_samples

        if start < 0 or stop > n_samples:
            excluded_edge += 1
            continue

        beat = x_mv[:, start:stop]

        if beat.shape[1] != pre_samples + post_samples:
            excluded_edge += 1
            continue

        if not np.isfinite(beat).all():
            excluded_nonfinite += 1
            continue

        beats.append(
            beat.astype(np.float32, copy=False)
        )
        used_peaks.append(int(p))

    if beats:
        beats_arr = np.stack(beats, axis=0)
    else:
        beats_arr = np.empty(
            (
                0,
                12,
                pre_samples + post_samples,
            ),
            dtype=np.float32,
        )

    return (
        beats_arr,
        np.asarray(used_peaks, dtype=np.int32),
        excluded_edge,
        excluded_nonfinite,
    )


def beat_shape_vectors(
    beats_mv: np.ndarray,
):
    """
    Create vectors ONLY for morphology similarity estimation.

    Each beat:
    - subtracts its per-lead temporal mean
    - preserves relative amplitudes across leads
    - is normalized by one global L2 norm

    This removes arbitrary baseline offsets while avoiding lead-by-lead
    amplitude normalization.
    """
    if len(beats_mv) == 0:
        return np.empty((0, 0), dtype=np.float64)

    b = beats_mv.astype(np.float64)

    centered = b - np.mean(
        b,
        axis=2,
        keepdims=True,
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


def morphology_similarity_matrix(
    beats_mv: np.ndarray,
):
    vectors = beat_shape_vectors(beats_mv)

    if len(vectors) == 0:
        return np.empty((0, 0), dtype=np.float64)

    sim = vectors @ vectors.T

    return np.clip(
        sim,
        -1.0,
        1.0,
    )


def connected_components_from_similarity(
    similarity: np.ndarray,
    threshold: float,
):
    """
    Simple graph components at a similarity threshold.

    These are exploratory morphology components, not diagnostic classes.
    """
    n = similarity.shape[0]

    if n == 0:
        return []

    visited = np.zeros(n, dtype=bool)
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
            np.asarray(sorted(comp), dtype=np.int32)
        )

    components.sort(
        key=lambda c: (-len(c), int(c[0]))
    )

    return components


def dominant_morphology_indices(
    similarity: np.ndarray,
    threshold: float,
):
    """
    Choose a dominant morphology neighborhood without deleting any beats.

    The seed is the beat with the largest number of direct neighbors whose
    morphology similarity is >= threshold. Ties are resolved by the highest
    median similarity to all beats.

    The dominant set contains the seed plus all direct neighbors above the
    threshold. It is used ONLY to define an additional dominant template.
    All usable beats remain represented in variability metrics.
    """
    n = similarity.shape[0]

    if n == 0:
        return np.asarray([], dtype=np.int32)

    if n == 1:
        return np.asarray([0], dtype=np.int32)

    neighbor_counts = np.sum(
        similarity >= threshold,
        axis=1,
    )

    median_similarity = np.median(
        similarity,
        axis=1,
    )

    best_count = np.max(neighbor_counts)

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


def safe_stat(x, fn, default=np.nan):
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return float(default)

    return float(fn(x))


def compute_rr_metrics(
    peaks: np.ndarray,
    fs_hz: float,
):
    if len(peaks) < 2:
        return {
            "rr_count": 0,
            "rr_mean_ms": np.nan,
            "rr_median_ms": np.nan,
            "rr_sd_ms": np.nan,
            "rr_mad_ms": np.nan,
            "rr_cv": np.nan,
            "rmssd_ms": np.nan,
            "heart_rate_from_median_rr_bpm": np.nan,
        }

    rr_ms = (
        np.diff(peaks.astype(np.float64))
        / fs_hz
        * 1000.0
    )

    rr_mean = float(np.mean(rr_ms))
    rr_median = float(np.median(rr_ms))

    rr_sd = (
        float(np.std(rr_ms, ddof=1))
        if len(rr_ms) >= 2
        else 0.0
    )

    rr_mad = float(
        np.median(
            np.abs(rr_ms - rr_median)
        )
    )

    rr_cv = (
        rr_sd / rr_mean
        if rr_mean > 0
        else np.nan
    )

    rr_diff = np.diff(rr_ms)

    rmssd = (
        float(
            np.sqrt(
                np.mean(rr_diff ** 2)
            )
        )
        if len(rr_diff) > 0
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
        "heart_rate_from_median_rr_bpm": hr,
    }


def compute_morphology_metrics(
    beats_mv: np.ndarray,
    median_all_mv: np.ndarray,
    similarity: np.ndarray,
    dominant_idx: np.ndarray,
    similarity_threshold: float,
):
    n = len(beats_mv)

    if n == 0:
        return {
            "morph_pairwise_similarity_median": np.nan,
            "morph_pairwise_similarity_min": np.nan,
            "morph_pairwise_similarity_p10": np.nan,
            "morph_rms_to_all_median_mv": np.nan,
            "morph_rms_to_all_median_p90_mv": np.nan,
            "morph_rms_relative_to_template": np.nan,
            "dominant_cluster_size": 0,
            "dominant_cluster_fraction": np.nan,
            "morph_components_at_threshold": 0,
        }

    delta = (
        beats_mv.astype(np.float64)
        - median_all_mv[None, :, :].astype(np.float64)
    )

    rms_per_beat = np.sqrt(
        np.mean(delta ** 2, axis=(1, 2))
    )

    template_rms = float(
        np.sqrt(
            np.mean(
                median_all_mv.astype(np.float64) ** 2
            )
        )
    )

    relative = (
        float(np.median(rms_per_beat) / template_rms)
        if template_rms > 1e-12
        else np.nan
    )

    if n >= 2:
        tri = similarity[
            np.triu_indices(n, k=1)
        ]

        sim_median = float(np.median(tri))
        sim_min = float(np.min(tri))
        sim_p10 = float(np.percentile(tri, 10.0))
    else:
        sim_median = np.nan
        sim_min = np.nan
        sim_p10 = np.nan

    components = connected_components_from_similarity(
        similarity,
        similarity_threshold,
    )

    return {
        "morph_pairwise_similarity_median":
            sim_median,
        "morph_pairwise_similarity_min":
            sim_min,
        "morph_pairwise_similarity_p10":
            sim_p10,
        "morph_rms_to_all_median_mv":
            float(np.median(rms_per_beat)),
        "morph_rms_to_all_median_p90_mv":
            float(np.percentile(rms_per_beat, 90.0)),
        "morph_rms_relative_to_template":
            relative,
        "dominant_cluster_size":
            int(len(dominant_idx)),
        "dominant_cluster_fraction":
            float(len(dominant_idx) / n),
        "morph_components_at_threshold":
            int(len(components)),
    }


def qc_label_from_count(n_usable: int):
    if n_usable >= 3:
        return "good", 3

    if n_usable == 2:
        return "acceptable", 2

    if n_usable == 1:
        return "limited", 1

    return "failed", 0


def read_metadata_value(
    h5: h5py.File,
    candidate_paths: list[str],
    index: int,
    default="",
):
    for path in candidate_paths:
        if path in h5 and isinstance(h5[path], h5py.Dataset):
            ds = h5[path]

            if ds.ndim == 1 and index < len(ds):
                value = ds[index]

                if isinstance(value, bytes):
                    return value.decode(
                        "utf-8",
                        errors="replace",
                    )

                if isinstance(value, np.generic):
                    return value.item()

                return value

    return default


def copy_group_if_present(
    src: h5py.File,
    dst: h5py.File,
    group_name: str,
):
    if group_name in src:
        src.copy(
            group_name,
            dst,
            name=group_name,
        )


def plot_qc_example(
    x_mv: np.ndarray,
    detected_peaks: np.ndarray,
    used_peaks: np.ndarray,
    median_all_mv: np.ndarray,
    median_dominant_mv: np.ndarray,
    row: dict,
    fs_hz: float,
    pre_samples: int,
    out_path: Path,
):
    """
    Compact QC figure:
    - original lead II over 4 s with detected and usable R fiducials
    - all-beat vs dominant median waveform in all 12 leads
    """
    t = np.arange(x_mv.shape[1]) / fs_hz

    beat_t = (
        np.arange(median_all_mv.shape[1])
        - pre_samples
    ) / fs_hz

    fig = plt.figure(
        figsize=(15, 12),
    )

    gs = fig.add_gridspec(
        5,
        3,
        height_ratios=[1.1, 1, 1, 1, 1],
        hspace=0.48,
        wspace=0.28,
    )

    ax0 = fig.add_subplot(
        gs[0, :]
    )

    lead_ii = 1

    ax0.plot(
        t,
        x_mv[lead_ii],
        linewidth=1.0,
    )

    for p in detected_peaks:
        ax0.axvline(
            p / fs_hz,
            linewidth=0.8,
            alpha=0.35,
        )

    for p in used_peaks:
        ax0.axvline(
            p / fs_hz,
            linewidth=1.3,
            alpha=0.8,
        )

    ax0.set_title(
        "Lead II: detected QRS fiducials "
        "(thin) and complete usable beats (strong)"
    )
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("mV")
    ax0.grid(alpha=0.15)

    axes = []

    for r in range(4):
        for c in range(3):
            axes.append(
                fig.add_subplot(
                    gs[r + 1, c]
                )
            )

    for i, ax in enumerate(axes):
        ax.plot(
            beat_t,
            median_all_mv[i],
            linewidth=1.3,
            label="median all beats",
        )

        ax.plot(
            beat_t,
            median_dominant_mv[i],
            linewidth=1.0,
            alpha=0.8,
            label="dominant morphology median",
        )

        ax.axvline(
            0.0,
            linewidth=0.7,
            alpha=0.45,
        )

        ax.axhline(
            0.0,
            linewidth=0.6,
            alpha=0.35,
        )

        ax.set_title(
            LEAD_NAMES[i]
        )

        ax.set_xlabel(
            "Time from fiducial (s)"
        )

        ax.set_ylabel("mV")
        ax.grid(alpha=0.15)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.745),
    )

    fig.suptitle(
        "Median-beat QC | "
        f"QC={row['qc_label']} | "
        f"detected={row['n_detected_qrs']} | "
        f"usable={row['n_usable_beats']} | "
        f"dominant={row['dominant_cluster_size']} | "
        f"HR={row['heart_rate_from_median_rr_bpm']:.1f} bpm"
        if np.isfinite(
            row["heart_rate_from_median_rr_bpm"]
        )
        else
        "Median-beat QC | "
        f"QC={row['qc_label']} | "
        f"detected={row['n_detected_qrs']} | "
        f"usable={row['n_usable_beats']} | "
        f"dominant={row['dominant_cluster_size']}",
        y=0.995,
    )

    fig.subplots_adjust(
        top=0.95,
        bottom=0.05,
        left=0.06,
        right=0.98,
    )

    fig.savefig(
        out_path,
        dpi=160,
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
            / "03_Phase1_Normal_Anchored"
            / "Phase1_NormalAnchored_preprocessed.h5"
        ),
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "Results"
            / "04_Phase1_Median_Beats"
            / "Phase1_MedianBeats_v1.h5"
        ),
    )

    ap.add_argument(
        "--results-dir",
        type=Path,
        default=(
            root
            / "Results"
            / "04_Phase1_Median_Beats"
        ),
    )

    ap.add_argument(
        "--preprocessing-summary",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--fs",
        type=float,
        default=FS_HZ_DEFAULT,
    )

    ap.add_argument(
        "--pre-ms",
        type=float,
        default=PRE_MS_DEFAULT,
    )

    ap.add_argument(
        "--post-ms",
        type=float,
        default=POST_MS_DEFAULT,
    )

    ap.add_argument(
        "--morph-similarity-threshold",
        type=float,
        default=0.90,
        help=(
            "Exploratory shape-similarity threshold used only "
            "to define the dominant morphology neighborhood. "
            "No beat is discarded based on this threshold."
        ),
    )

    ap.add_argument(
        "--min-beats-for-vae",
        type=int,
        default=2,
        help=(
            "Provisional QC flag only. Records are not removed."
        ),
    )

    ap.add_argument(
        "--qc-figures",
        type=int,
        default=40,
        help=(
            "Maximum number of example QC figures to create."
        ),
    )

    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    if not (0.0 < args.morph_similarity_threshold <= 1.0):
        raise ValueError(
            "--morph-similarity-threshold must be in (0,1]."
        )

    if args.min_beats_for_vae < 1:
        raise ValueError(
            "--min-beats-for-vae must be >= 1."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    qc_fig_dir = (
        args.results_dir
        / "qc_figures"
    )

    qc_fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pre_samples = int(
        round(
            args.pre_ms
            / 1000.0
            * args.fs
        )
    )

    post_samples = int(
        round(
            args.post_ms
            / 1000.0
            * args.fs
        )
    )

    beat_samples = (
        pre_samples
        + post_samples
    )

    if beat_samples <= 0:
        raise ValueError(
            "Invalid beat window."
        )

    mean_mv, std_mv, scaling_source = (
        load_scaling(
            root,
            args.preprocessing_summary,
        )
    )

    print(f"Input                : {args.input}")
    print(f"Output               : {args.output}")
    print(f"Sampling rate        : {args.fs} Hz")
    print(
        f"Beat window          : "
        f"-{args.pre_ms:g} to +{args.post_ms:g} ms "
        f"({beat_samples} samples)"
    )
    print(
        f"Morph similarity thr : "
        f"{args.morph_similarity_threshold:.3f}"
    )
    print(f"Scaling source       : {scaling_source}")
    print(f"Global mean          : {mean_mv:.12g} mV")
    print(f"Global std           : {std_mv:.12g} mV")
    print()

    rows = []

    # Small cache only for later selected QC figures.
    qc_cache = {}

    with h5py.File(
        args.input,
        "r",
    ) as src:

        signal_path, signal_ds = (
            get_signal_dataset(src)
        )

        n_records = int(
            signal_ds.shape[0]
        )

        if tuple(signal_ds.shape[1:]) != (12, 2000):
            raise ValueError(
                "Expected ECG input shape [N,12,2000], "
                f"found {signal_ds.shape} at {signal_path}."
            )

        print(
            f"Signal dataset       : {signal_path}"
        )
        print(
            f"Records              : {n_records:,}"
        )
        print()

        median_all = np.full(
            (
                n_records,
                12,
                beat_samples,
            ),
            np.nan,
            dtype=np.float32,
        )

        median_dominant = np.full_like(
            median_all,
            np.nan,
        )

        n_detected_arr = np.zeros(
            n_records,
            dtype=np.int16,
        )

        n_usable_arr = np.zeros(
            n_records,
            dtype=np.int16,
        )

        n_dominant_arr = np.zeros(
            n_records,
            dtype=np.int16,
        )

        qc_code_arr = np.zeros(
            n_records,
            dtype=np.int8,
        )

        provisional_valid_arr = np.zeros(
            n_records,
            dtype=np.uint8,
        )

        rr_median_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        rr_sd_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        rr_mad_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        rr_cv_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        rmssd_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        hr_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        morph_sim_median_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        morph_sim_min_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        morph_rms_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        morph_relative_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        dominant_fraction_arr = np.full(
            n_records,
            np.nan,
            dtype=np.float32,
        )

        morph_components_arr = np.zeros(
            n_records,
            dtype=np.int16,
        )

        for i in range(n_records):
            x_scaled = np.asarray(
                signal_ds[i],
                dtype=np.float32,
            )

            # Restore the already filtered Phase-I signal to mV.
            x_mv = (
                x_scaled.astype(np.float64)
                * std_mv
                + mean_mv
            ).astype(np.float32)

            if not np.isfinite(x_mv).all():
                detected_peaks = np.asarray(
                    [],
                    dtype=np.int32,
                )

                beats = np.empty(
                    (
                        0,
                        12,
                        beat_samples,
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
                detected_peaks, _, _ = (
                    detect_multilead_qrs(
                        x_mv,
                        args.fs,
                    )
                )

                (
                    beats,
                    used_peaks,
                    excluded_edge,
                    excluded_nonfinite,
                ) = extract_complete_beats(
                    x_mv,
                    detected_peaks,
                    pre_samples,
                    post_samples,
                )

            n_detected = int(
                len(detected_peaks)
            )

            n_usable = int(
                len(beats)
            )

            n_detected_arr[i] = n_detected
            n_usable_arr[i] = n_usable

            qc_label, qc_code = (
                qc_label_from_count(
                    n_usable
                )
            )

            qc_code_arr[i] = qc_code

            provisional_valid = (
                n_usable
                >= args.min_beats_for_vae
            )

            provisional_valid_arr[i] = (
                1
                if provisional_valid
                else 0
            )

            rr = compute_rr_metrics(
                detected_peaks,
                args.fs,
            )

            rr_median_arr[i] = (
                rr["rr_median_ms"]
            )

            rr_sd_arr[i] = (
                rr["rr_sd_ms"]
            )

            rr_mad_arr[i] = (
                rr["rr_mad_ms"]
            )

            rr_cv_arr[i] = (
                rr["rr_cv"]
            )

            rmssd_arr[i] = (
                rr["rmssd_ms"]
            )

            hr_arr[i] = (
                rr[
                    "heart_rate_from_median_rr_bpm"
                ]
            )

            dominant_idx = np.asarray(
                [],
                dtype=np.int32,
            )

            similarity = np.empty(
                (0, 0),
                dtype=np.float64,
            )

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
                    beats[dominant_idx],
                    axis=0,
                ).astype(np.float32)

                median_all[i] = med_all
                median_dominant[i] = med_dom

                morph = compute_morphology_metrics(
                    beats,
                    med_all,
                    similarity,
                    dominant_idx,
                    args.morph_similarity_threshold,
                )

                n_dominant_arr[i] = (
                    morph[
                        "dominant_cluster_size"
                    ]
                )

                morph_sim_median_arr[i] = (
                    morph[
                        "morph_pairwise_similarity_median"
                    ]
                )

                morph_sim_min_arr[i] = (
                    morph[
                        "morph_pairwise_similarity_min"
                    ]
                )

                morph_rms_arr[i] = (
                    morph[
                        "morph_rms_to_all_median_mv"
                    ]
                )

                morph_relative_arr[i] = (
                    morph[
                        "morph_rms_relative_to_template"
                    ]
                )

                dominant_fraction_arr[i] = (
                    morph[
                        "dominant_cluster_fraction"
                    ]
                )

                morph_components_arr[i] = (
                    morph[
                        "morph_components_at_threshold"
                    ]
                )

            ecg_id = read_metadata_value(
                src,
                [
                    "metadata/ecg_id",
                    "ecg_id",
                ],
                i,
                default=str(i),
            )

            patient_id = read_metadata_value(
                src,
                [
                    "metadata/patient_id",
                    "patient_id",
                ],
                i,
                default="",
            )

            subject_key = read_metadata_value(
                src,
                [
                    "metadata/subject_key",
                    "subject_key",
                ],
                i,
                default="",
            )

            source_name = read_metadata_value(
                src,
                [
                    "metadata/source",
                    "source",
                ],
                i,
                default="",
            )

            hcm = read_metadata_value(
                src,
                [
                    "labels/hcm",
                    "hcm",
                ],
                i,
                default=np.nan,
            )

            split_code = read_metadata_value(
                src,
                [
                    "labels/split_code",
                    "split_code",
                ],
                i,
                default=np.nan,
            )

            split_name = read_metadata_value(
                src,
                [
                    "labels/split_name",
                    "split_name",
                ],
                i,
                default="",
            )

            row = {
                "record_index": i,
                "ecg_id": ecg_id,
                "patient_id": patient_id,
                "subject_key": subject_key,
                "source": source_name,
                "hcm": hcm,
                "split_code": split_code,
                "split_name": split_name,
                "n_detected_qrs": n_detected,
                "n_usable_beats": n_usable,
                "n_excluded_edge": int(
                    excluded_edge
                ),
                "n_excluded_nonfinite": int(
                    excluded_nonfinite
                ),
                "qc_label": qc_label,
                "qc_code": qc_code,
                "provisional_valid_for_vae":
                    bool(provisional_valid),
                "r_peaks_detected_samples":
                    ";".join(
                        str(int(p))
                        for p in detected_peaks
                    ),
                "r_peaks_used_samples":
                    ";".join(
                        str(int(p))
                        for p in used_peaks
                    ),
                "rr_count": rr["rr_count"],
                "rr_mean_ms": rr["rr_mean_ms"],
                "rr_median_ms": rr["rr_median_ms"],
                "rr_sd_ms": rr["rr_sd_ms"],
                "rr_mad_ms": rr["rr_mad_ms"],
                "rr_cv": rr["rr_cv"],
                "rmssd_ms": rr["rmssd_ms"],
                "heart_rate_from_median_rr_bpm":
                    rr[
                        "heart_rate_from_median_rr_bpm"
                    ],
                "morph_pairwise_similarity_median":
                    (
                        float(
                            morph_sim_median_arr[i]
                        )
                        if np.isfinite(
                            morph_sim_median_arr[i]
                        )
                        else np.nan
                    ),
                "morph_pairwise_similarity_min":
                    (
                        float(
                            morph_sim_min_arr[i]
                        )
                        if np.isfinite(
                            morph_sim_min_arr[i]
                        )
                        else np.nan
                    ),
                "morph_rms_to_all_median_mv":
                    (
                        float(
                            morph_rms_arr[i]
                        )
                        if np.isfinite(
                            morph_rms_arr[i]
                        )
                        else np.nan
                    ),
                "morph_rms_relative_to_template":
                    (
                        float(
                            morph_relative_arr[i]
                        )
                        if np.isfinite(
                            morph_relative_arr[i]
                        )
                        else np.nan
                    ),
                "dominant_cluster_size":
                    int(
                        n_dominant_arr[i]
                    ),
                "dominant_cluster_fraction":
                    (
                        float(
                            dominant_fraction_arr[i]
                        )
                        if np.isfinite(
                            dominant_fraction_arr[i]
                        )
                        else np.nan
                    ),
                "morph_components_at_threshold":
                    int(
                        morph_components_arr[i]
                    ),
            }

            rows.append(row)

            # Cache only a small deterministic set of candidate examples.
            # Final stratified selection happens after all records are processed.
            if len(qc_cache) < max(
                args.qc_figures * 4,
                80,
            ):
                qc_cache[i] = {
                    "x_mv": x_mv,
                    "detected_peaks":
                        detected_peaks.copy(),
                    "used_peaks":
                        used_peaks.copy(),
                    "median_all":
                        (
                            median_all[i].copy()
                            if n_usable > 0
                            else None
                        ),
                    "median_dominant":
                        (
                            median_dominant[i].copy()
                            if n_usable > 0
                            else None
                        ),
                }

            if (
                (i + 1) % 250 == 0
                or i + 1 == n_records
            ):
                print(
                    f"Processed "
                    f"{i + 1:,}/{n_records:,}"
                )

        qc_df = pd.DataFrame(rows)

        # Write final HDF5 only after processing successfully.
        with h5py.File(
            args.output,
            "w",
        ) as dst:

            gsig = dst.create_group(
                "signals"
            )

            gsig.create_dataset(
                "median_all_mv",
                data=median_all,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            gsig.create_dataset(
                "median_dominant_mv",
                data=median_dominant,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            gq = dst.create_group(
                "quality"
            )

            gq.create_dataset(
                "n_detected_qrs",
                data=n_detected_arr,
            )

            gq.create_dataset(
                "n_usable_beats",
                data=n_usable_arr,
            )

            gq.create_dataset(
                "n_dominant_beats",
                data=n_dominant_arr,
            )

            gq.create_dataset(
                "qc_code",
                data=qc_code_arr,
            )

            gq.create_dataset(
                "provisional_valid_for_vae",
                data=provisional_valid_arr,
            )

            gq.create_dataset(
                "morph_pairwise_similarity_median",
                data=morph_sim_median_arr,
            )

            gq.create_dataset(
                "morph_pairwise_similarity_min",
                data=morph_sim_min_arr,
            )

            gq.create_dataset(
                "morph_rms_to_all_median_mv",
                data=morph_rms_arr,
            )

            gq.create_dataset(
                "morph_rms_relative_to_template",
                data=morph_relative_arr,
            )

            gq.create_dataset(
                "dominant_cluster_fraction",
                data=dominant_fraction_arr,
            )

            gq.create_dataset(
                "morph_components_at_threshold",
                data=morph_components_arr,
            )

            gr = dst.create_group(
                "rhythm"
            )

            gr.create_dataset(
                "rr_median_ms",
                data=rr_median_arr,
            )

            gr.create_dataset(
                "rr_sd_ms",
                data=rr_sd_arr,
            )

            gr.create_dataset(
                "rr_mad_ms",
                data=rr_mad_arr,
            )

            gr.create_dataset(
                "rr_cv",
                data=rr_cv_arr,
            )

            gr.create_dataset(
                "rmssd_ms",
                data=rmssd_arr,
            )

            gr.create_dataset(
                "heart_rate_from_median_rr_bpm",
                data=hr_arr,
            )

            # Preserve original metadata and split/label information.
            copy_group_if_present(
                src,
                dst,
                "metadata",
            )

            copy_group_if_present(
                src,
                dst,
                "labels",
            )

            dst.attrs[
                "design"
            ] = (
                "median beat + beat-to-beat morphology "
                "+ rhythm preservation"
            )

            dst.attrs[
                "source_file"
            ] = str(args.input)

            dst.attrs[
                "source_signal_dataset"
            ] = signal_path

            dst.attrs[
                "sampling_rate_hz"
            ] = float(args.fs)

            dst.attrs[
                "pre_ms"
            ] = float(args.pre_ms)

            dst.attrs[
                "post_ms"
            ] = float(args.post_ms)

            dst.attrs[
                "beat_samples"
            ] = int(beat_samples)

            dst.attrs[
                "morph_similarity_threshold"
            ] = float(
                args.morph_similarity_threshold
            )

            dst.attrs[
                "morphologically_different_beats_removed"
            ] = False

            dst.attrs[
                "provisional_min_beats_for_vae"
            ] = int(
                args.min_beats_for_vae
            )

            dst.attrs[
                "input_global_scaling_mean_mv"
            ] = float(mean_mv)

            dst.attrs[
                "input_global_scaling_std_mv"
            ] = float(std_mv)

            dst.attrs[
                "input_scaling_source"
            ] = scaling_source

    qc_csv = (
        args.results_dir
        / "median_beat_qc.csv"
    )

    qc_df.to_csv(
        qc_csv,
        index=False,
    )

    # Aggregate summary.
    counts = (
        qc_df["qc_label"]
        .value_counts()
        .to_dict()
    )

    usable_counts = (
        qc_df[
            "n_usable_beats"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "records": int(len(qc_df)),
        "sampling_rate_hz": float(args.fs),
        "beat_window_ms": {
            "pre": float(args.pre_ms),
            "post": float(args.post_ms),
        },
        "beat_samples": int(beat_samples),
        "morph_similarity_threshold":
            float(
                args.morph_similarity_threshold
            ),
        "morphologically_different_beats_removed":
            False,
        "template_definitions": {
            "median_all_mv":
                "sample-wise median across every complete finite aligned beat",
            "median_dominant_mv":
                "sample-wise median across the dominant morphology neighborhood; all other beats remain in variability metrics",
        },
        "qc_definition": {
            "good": ">=3 usable beats",
            "acceptable": "2 usable beats",
            "limited": "1 usable beat",
            "failed": "0 usable beats",
        },
        "provisional_min_beats_for_vae":
            int(
                args.min_beats_for_vae
            ),
        "provisional_valid_for_vae_records":
            int(
                qc_df[
                    "provisional_valid_for_vae"
                ].sum()
            ),
        "qc_counts": {
            str(k): int(v)
            for k, v in counts.items()
        },
        "usable_beat_count_distribution": {
            str(k): int(v)
            for k, v in usable_counts.items()
        },
        "median_detected_qrs":
            float(
                qc_df[
                    "n_detected_qrs"
                ].median()
            ),
        "median_usable_beats":
            float(
                qc_df[
                    "n_usable_beats"
                ].median()
            ),
        "median_rr_ms":
            (
                float(
                    np.nanmedian(
                        qc_df[
                            "rr_median_ms"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )
                if qc_df[
                    "rr_median_ms"
                ].notna().any()
                else None
            ),
        "median_heart_rate_bpm":
            (
                float(
                    np.nanmedian(
                        qc_df[
                            "heart_rate_from_median_rr_bpm"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )
                if qc_df[
                    "heart_rate_from_median_rr_bpm"
                ].notna().any()
                else None
            ),
        "median_pairwise_morph_similarity":
            (
                float(
                    np.nanmedian(
                        qc_df[
                            "morph_pairwise_similarity_median"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )
                if qc_df[
                    "morph_pairwise_similarity_median"
                ].notna().any()
                else None
            ),
        "median_dominant_cluster_fraction":
            (
                float(
                    np.nanmedian(
                        qc_df[
                            "dominant_cluster_fraction"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )
                if qc_df[
                    "dominant_cluster_fraction"
                ].notna().any()
                else None
            ),
        "input_scaling": {
            "mean_mv": float(mean_mv),
            "std_mv": float(std_mv),
            "source": scaling_source,
        },
    }

    summary_path = (
        args.results_dir
        / "median_beat_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Generate a small stratified QC sample from cached early examples.
    if args.qc_figures > 0:
        per_class_target = max(
            1,
            int(
                math.ceil(
                    args.qc_figures
                    / 4
                )
            ),
        )

        selected = []

        for qc_name in [
            "good",
            "acceptable",
            "limited",
            "failed",
        ]:
            subset = qc_df[
                qc_df[
                    "qc_label"
                ] == qc_name
            ]

            for idx in subset[
                "record_index"
            ].astype(int):
                if idx in qc_cache:
                    selected.append(idx)

                if (
                    sum(
                        qc_df.loc[
                            qc_df[
                                "record_index"
                            ].isin(selected),
                            "qc_label",
                        ]
                        == qc_name
                    )
                    >= per_class_target
                ):
                    break

        selected = selected[
            : args.qc_figures
        ]

        for idx in selected:
            cached = qc_cache[idx]
            row = qc_df.loc[
                qc_df[
                    "record_index"
                ] == idx
            ].iloc[0].to_dict()

            if (
                cached["median_all"]
                is None
            ):
                continue

            out = (
                qc_fig_dir
                / f"record_{idx:04d}_{row['qc_label']}.png"
            )

            plot_qc_example(
                x_mv=cached["x_mv"],
                detected_peaks=cached[
                    "detected_peaks"
                ],
                used_peaks=cached[
                    "used_peaks"
                ],
                median_all_mv=cached[
                    "median_all"
                ],
                median_dominant_mv=cached[
                    "median_dominant"
                ],
                row=row,
                fs_hz=args.fs,
                pre_samples=pre_samples,
                out_path=out,
            )

    print()
    print("Median-beat construction complete.")
    print(f"HDF5 output : {args.output}")
    print(f"QC CSV      : {qc_csv}")
    print(f"Summary     : {summary_path}")
    print(f"QC figures  : {qc_fig_dir}")
    print()
    print("IMPORTANT:")
    print(
        "Morphologically different beats were retained. "
        "The dominant template is an additional representation, "
        "not an outlier-filtered replacement."
    )


if __name__ == "__main__":
    main()
