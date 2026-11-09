#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
28_prepare_mimic_smash_median_beats.py

SMASH-HCM -- External MIMIC-IV HCM preprocessing.

Purpose
-------
Recursively read MIMIC-IV 12-lead WFDB ECG records organized as:

    Data/MIMIC-IV/MIMIC_HCM_patients_ecg/
        p1002/
            p10021109/
                s41296929/
                    41296929.hea
                    41296929.dat
                s41562386/
                    41562386.hea
                    41562386.dat
                ...

and reconstruct SMASH-HCM-compatible median beats while preserving patient
identity, acquisition time, and longitudinal ordering.

The preprocessing intentionally mirrors the frozen SMASH-HCM pipeline:

1. read physical ECG amplitudes from WFDB;
2. harmonize the 12 leads to:
       I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
3. use the first 4 seconds;
4. resample to 500 Hz only if necessary;
5. apply a 4th-order Butterworth 0.5-40 Hz band-pass filter, zero phase;
6. detect common multilead QRS fiducials;
7. extract complete windows from -350 ms to +650 ms;
8. compute the pointwise median across all usable aligned beats;
9. require >=2 usable beats for projection eligibility;
10. retain all records in the output, including failed/limited records, with
    explicit QC flags.

No per-record or per-lead amplitude normalization is applied.

IMPORTANT
---------
This script does NOT apply the Normal-training global z-scaling and does NOT
project the beats into any VAE. The output median beats remain in physical mV.
Global scaling and frozen-model projection should be a separate next step.

Outputs
-------
Results/28_MIMIC_SMASH_MedianBeats/
    mimic_smash_median_beats.h5
    mimic_smash_median_beat_qc.csv
    mimic_smash_longitudinal_manifest.csv
    mimic_smash_summary.json

Requirements
------------
pip install wfdb numpy pandas scipy h5py tqdm
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import wfdb
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

STEP = 28
STEP_NAME = "MIMIC_SMASH_MedianBeats"

TARGET_FS_HZ = 500.0
WINDOW_SECONDS = 4.0
N_SAMPLES_4S = int(round(TARGET_FS_HZ * WINDOW_SECONDS))

LOW_HZ = 0.5
HIGH_HZ = 40.0
FILTER_ORDER = 4

PRE_MS = 350.0
POST_MS = 650.0
PRE_SAMPLES = int(round(PRE_MS * TARGET_FS_HZ / 1000.0))
POST_SAMPLES = int(round(POST_MS * TARGET_FS_HZ / 1000.0))
BEAT_SAMPLES = PRE_SAMPLES + POST_SAMPLES

MIN_BEATS_FOR_PROJECTION = 2

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


# ============================================================================
# Paths
# ============================================================================

def infer_project_root() -> Path:
    """
    Expected placement:
        <project>/Scripts/28_prepare_mimic_smash_median_beats.py
    """
    here = Path(__file__).resolve()

    for p in [here.parent, *here.parents]:
        if (p / "Data").exists() and (p / "Results").exists():
            return p

    # Fallback if Results has not yet been created.
    for p in [here.parent, *here.parents]:
        if (p / "Data").exists():
            return p

    return here.parent.parent


def parse_args():
    root = infer_project_root()

    default_input = (
        root
        / "Data"
        / "MIMIC-IV"
        / "MIMIC_HCM_patients_ecg"
    )

    default_results = (
        root
        / "Results"
        / f"{STEP:02d}_{STEP_NAME}"
    )

    ap = argparse.ArgumentParser(
        description="Build SMASH-HCM-compatible median beats from MIMIC-IV WFDB ECGs."
    )

    ap.add_argument(
        "--input-root",
        type=Path,
        default=default_input,
        help="Root folder containing pXXXX/pXXXXXXXX/sXXXXXXXX WFDB records.",
    )

    ap.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help="Output directory. Default uses the same progressive number as this script.",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of .hea records to process for testing.",
    )

    return ap.parse_args()


# ============================================================================
# Small utilities
# ============================================================================

def write_str_dataset(group: h5py.Group, name: str, values) -> None:
    dt = h5py.string_dtype("utf-8")
    group.create_dataset(
        name,
        data=np.asarray(
            ["" if pd.isna(x) else str(x) for x in values],
            dtype=object,
        ),
        dtype=dt,
        compression="gzip",
        compression_opts=4,
    )


def canonical_lead_name(name: str) -> str | None:
    """
    Canonicalize common WFDB lead-name variants.
    """
    s = str(name).strip()
    s_compact = re.sub(r"[\s_\-]", "", s).upper()

    aliases = {
        "I": "I",
        "LEADI": "I",
        "ECGI": "I",
        "II": "II",
        "LEADII": "II",
        "ECGII": "II",
        "III": "III",
        "LEADIII": "III",
        "ECGIII": "III",
        "AVR": "aVR",
        "LEADAVR": "aVR",
        "ECGAVR": "aVR",
        "AVL": "aVL",
        "LEADAVL": "aVL",
        "ECGAVL": "aVL",
        "AVF": "aVF",
        "LEADAVF": "aVF",
        "ECGAVF": "aVF",
        "V1": "V1",
        "V2": "V2",
        "V3": "V3",
        "V4": "V4",
        "V5": "V5",
        "V6": "V6",
        "LEADV1": "V1",
        "LEADV2": "V2",
        "LEADV3": "V3",
        "LEADV4": "V4",
        "LEADV5": "V5",
        "LEADV6": "V6",
        "ECGV1": "V1",
        "ECGV2": "V2",
        "ECGV3": "V3",
        "ECGV4": "V4",
        "ECGV5": "V5",
        "ECGV6": "V6",
    }

    return aliases.get(s_compact)


def parse_subject_from_comments(comments) -> str:
    if comments is None:
        return ""

    for c in comments:
        text = str(c)
        m = re.search(
            r"subject[_\s-]*id\s*[:=]\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1)

    return ""


def folder_metadata(hea_path: Path) -> dict:
    """
    Expected:
        .../p1002/p10021109/s41296929/41296929.hea
    """
    record_folder = hea_path.parent.name
    patient_folder = hea_path.parent.parent.name if len(hea_path.parents) >= 2 else ""
    group_folder = hea_path.parent.parent.parent.name if len(hea_path.parents) >= 3 else ""

    return {
        "group_folder": group_folder,
        "patient_folder": patient_folder,
        "record_folder": record_folder,
        "subject_id_folder": (
            patient_folder[1:]
            if re.fullmatch(r"p\d+", patient_folder, flags=re.IGNORECASE)
            else ""
        ),
        "study_id_folder": (
            record_folder[1:]
            if re.fullmatch(r"s\d+", record_folder, flags=re.IGNORECASE)
            else ""
        ),
        "record_id": hea_path.stem,
    }


def header_datetime(header) -> pd.Timestamp:
    """
    Prefer WFDB-parsed base_date/base_time rather than reparsing the raw line.
    """
    base_date = getattr(header, "base_date", None)
    base_time = getattr(header, "base_time", None)

    if base_date is None:
        return pd.NaT

    if base_time is None:
        base_time = datetime.min.time()

    try:
        return pd.Timestamp.combine(base_date, base_time)
    except Exception:
        return pd.NaT


def unit_factor_to_mv(unit: str) -> float | None:
    """
    Convert physical units returned by WFDB to mV.
    """
    u = str(unit or "").strip().replace("µ", "u").replace("μ", "u").lower()

    if u == "mv":
        return 1.0
    if u == "uv":
        return 1e-3
    if u == "v":
        return 1e3

    return None


def harmonize_leads_to_mv(record) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Return [12, samples] physical signal in mV and the original/canonical order.
    """
    if record.p_signal is None:
        raise ValueError("WFDB record has no physical p_signal.")

    sig = np.asarray(record.p_signal, dtype=np.float64)  # [samples, leads]

    if sig.ndim != 2:
        raise ValueError(f"Expected 2D p_signal; got shape {sig.shape}.")

    raw_names = [str(x) for x in record.sig_name]
    canon_names = [canonical_lead_name(x) for x in raw_names]

    index_by_lead = {}

    for idx, lead in enumerate(canon_names):
        if lead is None:
            continue
        if lead in index_by_lead:
            raise ValueError(
                f"Duplicate canonical lead {lead!r} in {raw_names}."
            )
        index_by_lead[lead] = idx

    missing = [lead for lead in LEAD_NAMES if lead not in index_by_lead]

    if missing:
        raise ValueError(
            "Missing required leads: "
            + ", ".join(missing)
            + f". Original leads: {raw_names}"
        )

    units = list(record.units) if record.units is not None else [""] * sig.shape[1]

    out = np.empty((12, sig.shape[0]), dtype=np.float64)

    for j, lead in enumerate(LEAD_NAMES):
        src = index_by_lead[lead]
        factor = unit_factor_to_mv(units[src])

        if factor is None:
            raise ValueError(
                f"Unsupported/unknown physical unit {units[src]!r} for lead {lead}."
            )

        out[j] = sig[:, src] * factor

    return out, raw_names, canon_names


# ============================================================================
# SMASH-HCM signal preprocessing
# ============================================================================

def resample_first_4s(
    x_mv: np.ndarray,
    original_fs: float,
) -> tuple[np.ndarray, bool]:
    """
    Take the first 4 seconds and convert to exactly 500 Hz if needed.
    """
    if not np.isfinite(original_fs) or original_fs <= 0:
        raise ValueError(f"Invalid sampling rate: {original_fs}")

    n_needed_src = int(round(WINDOW_SECONDS * float(original_fs)))

    if x_mv.shape[1] < n_needed_src:
        raise ValueError(
            f"Record shorter than {WINDOW_SECONDS:g} s: "
            f"{x_mv.shape[1]} samples at {original_fs:g} Hz."
        )

    x4 = x_mv[:, :n_needed_src]

    if abs(float(original_fs) - TARGET_FS_HZ) < 1e-9:
        if x4.shape[1] != N_SAMPLES_4S:
            raise ValueError(
                f"Unexpected 4-s shape at 500 Hz: {x4.shape}."
            )
        return x4.astype(np.float32), False

    ratio = Fraction(TARGET_FS_HZ / float(original_fs)).limit_denominator(1000)

    xr = resample_poly(
        x4,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=1,
    )

    # For ordinary rational ECG sampling rates this should be exact.
    if xr.shape[1] < N_SAMPLES_4S:
        raise ValueError(
            f"Resampling produced only {xr.shape[1]} samples; "
            f"expected at least {N_SAMPLES_4S}."
        )

    if xr.shape[1] > N_SAMPLES_4S:
        xr = xr[:, :N_SAMPLES_4S]

    if xr.shape[1] != N_SAMPLES_4S:
        raise ValueError(
            f"Resampled 4-s signal has shape {xr.shape}; "
            f"expected [12, {N_SAMPLES_4S}]."
        )

    return xr.astype(np.float32), True


def make_filter():
    return butter(
        FILTER_ORDER,
        [LOW_HZ, HIGH_HZ],
        btype="bandpass",
        fs=TARGET_FS_HZ,
        output="sos",
    )


def filter_ecg(x_mv: np.ndarray, sos: np.ndarray) -> np.ndarray:
    if not np.isfinite(x_mv).all():
        raise ValueError("Signal contains NaN/Inf before filtering.")

    return sosfiltfilt(
        sos,
        x_mv,
        axis=-1,
    ).astype(np.float32)


# ============================================================================
# QRS detection copied from the SMASH-HCM median-beat logic
# ============================================================================

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
        int(round(0.25 * fs_hz)),
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

        if len(peaks) > 0:
            order = np.argsort(
                detection_score[peaks]
            )[::-1]

            # Four seconds; allow up to 16 candidates (~240 bpm).
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


def rr_metrics(peaks: np.ndarray, fs_hz: float) -> dict:
    """Compute the same detected-QRS rhythm descriptors used in SMASH-HCM."""
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

    rr_ms = np.diff(peaks.astype(np.float64)) / fs_hz * 1000.0
    rr_mean = float(np.mean(rr_ms))
    rr_median = float(np.median(rr_ms))
    rr_sd = float(np.std(rr_ms, ddof=1)) if len(rr_ms) >= 2 else 0.0
    rr_mad = float(np.median(np.abs(rr_ms - rr_median)))
    rr_cv = rr_sd / rr_mean if rr_mean > 0 else np.nan
    drr = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(drr ** 2))) if len(drr) > 0 else np.nan

    return {
        "rr_count": int(len(rr_ms)),
        "rr_mean_ms": rr_mean,
        "rr_median_ms": rr_median,
        "rr_sd_ms": rr_sd,
        "rr_mad_ms": rr_mad,
        "rr_cv": rr_cv,
        "rmssd_ms": rmssd,
        "heart_rate_from_median_rr_bpm": (
            60000.0 / rr_median
            if rr_median > 0
            else np.nan
        ),
    }


def qc_label_from_beats(n_usable: int) -> tuple[str, int]:
    if n_usable >= 3:
        return "good", 3
    if n_usable == 2:
        return "acceptable", 2
    if n_usable == 1:
        return "limited", 1
    return "failed", 0


# ============================================================================
# One-record processing
# ============================================================================

def process_record(
    hea_path: Path,
    input_root: Path,
    sos: np.ndarray,
) -> tuple[dict, np.ndarray]:
    folder = folder_metadata(hea_path)

    row = {
        **folder,
        "source_relpath": str(hea_path.relative_to(input_root)),
        "hea_path": str(hea_path),
        "dat_path": str(hea_path.with_suffix(".dat")),
        "dat_exists": hea_path.with_suffix(".dat").exists(),
        "subject_id_header": "",
        "subject_id": folder["subject_id_folder"],
        "subject_id_folder_header_match": False,
        "acquisition_datetime": pd.NaT,
        "original_fs_hz": np.nan,
        "original_n_samples": np.nan,
        "original_n_sig": np.nan,
        "original_lead_order": "",
        "harmonized_lead_order": ";".join(LEAD_NAMES),
        "physical_units_original": "",
        "resampled_to_500hz": False,
        "n_detected_qrs": 0,
        "n_usable_beats": 0,
        "n_excluded_edge": 0,
        "n_excluded_nonfinite": 0,
        "qc_label": "failed",
        "qc_code": 0,
        "valid_for_projection": False,
        "r_peaks_detected_samples": "",
        "r_peaks_used_samples": "",
        "rr_count": 0,
        "rr_mean_ms": np.nan,
        "rr_median_ms": np.nan,
        "rr_sd_ms": np.nan,
        "rr_mad_ms": np.nan,
        "rr_cv": np.nan,
        "rmssd_ms": np.nan,
        "heart_rate_from_median_rr_bpm": np.nan,
        "error": "",
    }

    median_beat = np.full(
        (12, BEAT_SAMPLES),
        np.nan,
        dtype=np.float32,
    )

    if not row["dat_exists"]:
        row["error"] = "missing_dat"
        return row, median_beat

    record_base = hea_path.with_suffix("")

    try:
        header = wfdb.rdheader(str(record_base))

        row["subject_id_header"] = parse_subject_from_comments(
            getattr(header, "comments", None)
        )

        if not row["subject_id"] and row["subject_id_header"]:
            row["subject_id"] = row["subject_id_header"]

        row["subject_id_folder_header_match"] = bool(
            row["subject_id"]
            and row["subject_id_header"]
            and row["subject_id"] == row["subject_id_header"]
        )

        row["acquisition_datetime"] = header_datetime(header)
        row["original_fs_hz"] = float(header.fs)
        row["original_n_sig"] = int(header.n_sig)
        row["original_n_samples"] = int(header.sig_len)
        row["original_lead_order"] = ";".join(map(str, header.sig_name))
        row["physical_units_original"] = ";".join(map(str, header.units))

    except Exception as exc:
        row["error"] = f"header_read_error: {type(exc).__name__}: {exc}"
        return row, median_beat

    try:
        record = wfdb.rdrecord(
            str(record_base),
            physical=True,
        )

        x_mv, raw_names, _ = harmonize_leads_to_mv(record)

        if not np.isfinite(x_mv).all():
            raise ValueError("Raw physical signal contains NaN/Inf.")

        x4_mv, resampled = resample_first_4s(
            x_mv,
            float(record.fs),
        )

        row["resampled_to_500hz"] = bool(resampled)

        x4_filtered = filter_ecg(
            x4_mv,
            sos,
        )

        detected_peaks, _, _ = detect_multilead_qrs(
            x4_filtered,
            TARGET_FS_HZ,
        )

        (
            beats,
            used_peaks,
            excluded_edge,
            excluded_nonfinite,
        ) = extract_complete_beats(
            x4_filtered,
            detected_peaks,
            PRE_SAMPLES,
            POST_SAMPLES,
        )

        n_detected = int(len(detected_peaks))
        n_usable = int(len(beats))

        if n_usable > 0:
            median_beat = np.median(
                beats,
                axis=0,
            ).astype(np.float32)

        label, code = qc_label_from_beats(n_usable)
        valid = n_usable >= MIN_BEATS_FOR_PROJECTION

        row.update({
            "n_detected_qrs": n_detected,
            "n_usable_beats": n_usable,
            "n_excluded_edge": int(excluded_edge),
            "n_excluded_nonfinite": int(excluded_nonfinite),
            "qc_label": label,
            "qc_code": code,
            "valid_for_projection": bool(valid),
            "r_peaks_detected_samples": ";".join(
                str(int(p)) for p in detected_peaks
            ),
            "r_peaks_used_samples": ";".join(
                str(int(p)) for p in used_peaks
            ),
        })

        row.update(
            rr_metrics(
                detected_peaks,
                TARGET_FS_HZ,
            )
        )

    except Exception as exc:
        row["error"] = f"signal_processing_error: {type(exc).__name__}: {exc}"

    return row, median_beat


# ============================================================================
# Longitudinal organization
# ============================================================================

def add_longitudinal_fields(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["acquisition_datetime"] = pd.to_datetime(
        d["acquisition_datetime"],
        errors="coerce",
    )

    # Stable chronological ordering inside each subject.
    d = d.sort_values(
        ["subject_id", "acquisition_datetime", "record_id"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)

    d["record_index_sorted"] = np.arange(len(d), dtype=np.int64)

    # All-record acquisition order.
    d["visit_number_all_records"] = (
        d.groupby("subject_id", dropna=False)
        .cumcount()
        .add(1)
        .astype(np.int32)
    )

    d["n_records_subject"] = (
        d.groupby("subject_id", dropna=False)["record_id"]
        .transform("count")
        .astype(np.int32)
    )

    # Time from first recorded ECG.
    first_dt = (
        d.groupby("subject_id", dropna=False)["acquisition_datetime"]
        .transform("min")
    )

    d["time_from_first_ecg_days"] = (
        d["acquisition_datetime"] - first_dt
    ).dt.total_seconds() / 86400.0

    d["time_from_first_ecg_years"] = (
        d["time_from_first_ecg_days"] / 365.2425
    )

    d["delta_from_previous_ecg_days"] = (
        d.groupby("subject_id", dropna=False)["acquisition_datetime"]
        .diff()
        .dt.total_seconds()
        / 86400.0
    )

    # Valid-for-projection visit structure.
    d["n_valid_records_subject"] = (
        d["valid_for_projection"]
        .astype(int)
        .groupby(d["subject_id"], dropna=False)
        .transform("sum")
        .astype(np.int32)
    )

    d["valid_visit_number"] = pd.Series(
        pd.NA,
        index=d.index,
        dtype="Int64",
    )

    valid_mask = d["valid_for_projection"].astype(bool)

    d.loc[valid_mask, "valid_visit_number"] = (
        d.loc[valid_mask]
        .groupby("subject_id", dropna=False)
        .cumcount()
        .add(1)
        .astype("Int64")
    )

    d["time_from_first_valid_ecg_days"] = np.nan
    d["time_from_first_valid_ecg_years"] = np.nan
    d["delta_from_previous_valid_ecg_days"] = np.nan

    if valid_mask.any():
        valid = d.loc[
            valid_mask,
            ["subject_id", "acquisition_datetime"]
        ].copy()

        first_valid = (
            valid.groupby("subject_id", dropna=False)["acquisition_datetime"]
            .transform("min")
        )

        valid["time_from_first_valid_ecg_days"] = (
            valid["acquisition_datetime"] - first_valid
        ).dt.total_seconds() / 86400.0

        valid["time_from_first_valid_ecg_years"] = (
            valid["time_from_first_valid_ecg_days"] / 365.2425
        )

        valid["delta_from_previous_valid_ecg_days"] = (
            valid.groupby("subject_id", dropna=False)["acquisition_datetime"]
            .diff()
            .dt.total_seconds()
            / 86400.0
        )

        d.loc[
            valid.index,
            "time_from_first_valid_ecg_days",
        ] = valid["time_from_first_valid_ecg_days"]

        d.loc[
            valid.index,
            "time_from_first_valid_ecg_years",
        ] = valid["time_from_first_valid_ecg_years"]

        d.loc[
            valid.index,
            "delta_from_previous_valid_ecg_days",
        ] = valid["delta_from_previous_valid_ecg_days"]

    d["has_acquisition_datetime"] = d["acquisition_datetime"].notna()

    return d


# ============================================================================
# HDF5 export
# ============================================================================

def save_hdf5(
    path: Path,
    df: pd.DataFrame,
    median_beats: np.ndarray,
) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["description"] = (
            "MIMIC-IV HCM median beats harmonized to the SMASH-HCM preprocessing."
        )
        h5.attrs["script"] = Path(__file__).name
        h5.attrs["step"] = STEP
        h5.attrs["sampling_rate_hz"] = TARGET_FS_HZ
        h5.attrs["source_window_seconds"] = WINDOW_SECONDS
        h5.attrs["source_window_samples"] = N_SAMPLES_4S
        h5.attrs["beat_pre_ms"] = PRE_MS
        h5.attrs["beat_post_ms"] = POST_MS
        h5.attrs["beat_samples"] = BEAT_SAMPLES
        h5.attrs["bandpass_low_hz"] = LOW_HZ
        h5.attrs["bandpass_high_hz"] = HIGH_HZ
        h5.attrs["bandpass_order"] = FILTER_ORDER
        h5.attrs["zero_phase_filter"] = True
        h5.attrs["min_beats_for_projection"] = MIN_BEATS_FOR_PROJECTION
        h5.attrs["per_record_amplitude_normalization"] = False
        h5.attrs["per_lead_amplitude_normalization"] = False
        h5.attrs["global_vae_scaling_applied"] = False

        gs = h5.create_group("signals")
        gs.create_dataset(
            "median_all_mv",
            data=median_beats.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        gm = h5.create_group("metadata")

        str_cols = [
            "subject_id",
            "record_id",
            "study_id_folder",
            "group_folder",
            "patient_folder",
            "record_folder",
            "acquisition_datetime",
            "source_relpath",
            "original_lead_order",
            "physical_units_original",
            "qc_label",
            "error",
        ]

        for c in str_cols:
            if c in df.columns:
                values = df[c]
                if c == "acquisition_datetime":
                    values = values.dt.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ).fillna("")
                write_str_dataset(gm, c, values)

        numeric_cols = [
            "record_index_sorted",
            "visit_number_all_records",
            "n_records_subject",
            "valid_visit_number",
            "n_valid_records_subject",
            "time_from_first_ecg_days",
            "time_from_first_ecg_years",
            "delta_from_previous_ecg_days",
            "time_from_first_valid_ecg_days",
            "time_from_first_valid_ecg_years",
            "delta_from_previous_valid_ecg_days",
            "original_fs_hz",
            "original_n_samples",
            "original_n_sig",
            "n_detected_qrs",
            "n_usable_beats",
            "n_excluded_edge",
            "n_excluded_nonfinite",
            "rr_count",
            "rr_mean_ms",
            "rr_median_ms",
            "rr_sd_ms",
            "rr_mad_ms",
            "rr_cv",
            "rmssd_ms",
            "heart_rate_from_median_rr_bpm",
        ]

        for c in numeric_cols:
            if c not in df.columns:
                continue

            arr = pd.to_numeric(
                df[c],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            gm.create_dataset(
                c,
                data=arr,
                compression="gzip",
                compression_opts=4,
            )

        bool_cols = [
            "dat_exists",
            "subject_id_folder_header_match",
            "resampled_to_500hz",
            "valid_for_projection",
            "has_acquisition_datetime",
        ]

        for c in bool_cols:
            if c in df.columns:
                gm.create_dataset(
                    c,
                    data=df[c].fillna(False).astype(bool).to_numpy(),
                    compression="gzip",
                    compression_opts=4,
                )

        write_str_dataset(
            h5.create_group("preprocessing"),
            "lead_names",
            LEAD_NAMES,
        )


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    input_root = args.input_root.resolve()
    results_dir = args.results_dir.resolve()

    if not input_root.exists():
        raise FileNotFoundError(
            f"MIMIC input root not found:\n{input_root}"
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hea_files = sorted(
        input_root.rglob("*.hea")
    )

    if args.limit is not None:
        hea_files = hea_files[: args.limit]

    if not hea_files:
        raise RuntimeError(
            f"No .hea files found recursively under:\n{input_root}"
        )

    print("=" * 86)
    print("28 - MIMIC-IV -> SMASH-HCM compatible median beats")
    print("=" * 86)
    print(f"Input root : {input_root}")
    print(f"Results    : {results_dir}")
    print(f".hea files : {len(hea_files):,}")
    print()

    sos = make_filter()

    rows = []
    beats = []

    for hea_path in tqdm(
        hea_files,
        desc="Processing MIMIC ECGs",
        unit="ECG",
    ):
        row, median_beat = process_record(
            hea_path=hea_path,
            input_root=input_root,
            sos=sos,
        )
        rows.append(row)
        beats.append(median_beat)

    raw_df = pd.DataFrame(rows)
    raw_beats = np.stack(beats, axis=0).astype(np.float32)

    # Add longitudinal fields and establish chronological output order.
    df = add_longitudinal_fields(raw_df)

    # Reorder median beats exactly as the sorted manifest.
    order = df["record_index_sorted"].to_numpy()

    # record_index_sorted currently reflects row position AFTER sorting, so we
    # need an explicit mapping from record identity back to raw processing order.
    raw_lookup = {
        (
            str(r["subject_id"]),
            str(r["record_id"]),
            str(r["source_relpath"]),
        ): i
        for i, r in raw_df.iterrows()
    }

    source_indices = []

    for _, r in df.iterrows():
        key = (
            str(r["subject_id"]),
            str(r["record_id"]),
            str(r["source_relpath"]),
        )
        source_indices.append(raw_lookup[key])

    median_beats = raw_beats[
        np.asarray(source_indices, dtype=int)
    ]

    # Rewrite record_index_sorted to be unambiguous after reordering.
    df["record_index_sorted"] = np.arange(
        len(df),
        dtype=np.int64,
    )

    qc_path = (
        results_dir
        / "mimic_smash_median_beat_qc.csv"
    )

    longitudinal_path = (
        results_dir
        / "mimic_smash_longitudinal_manifest.csv"
    )

    h5_path = (
        results_dir
        / "mimic_smash_median_beats.h5"
    )

    summary_path = (
        results_dir
        / "mimic_smash_summary.json"
    )

    # Complete audit/QC table.
    df.to_csv(
        qc_path,
        index=False,
    )

    # Compact longitudinal table.
    longitudinal_cols = [
        "subject_id",
        "record_id",
        "study_id_folder",
        "acquisition_datetime",
        "visit_number_all_records",
        "n_records_subject",
        "valid_for_projection",
        "valid_visit_number",
        "n_valid_records_subject",
        "time_from_first_ecg_days",
        "time_from_first_ecg_years",
        "delta_from_previous_ecg_days",
        "time_from_first_valid_ecg_days",
        "time_from_first_valid_ecg_years",
        "delta_from_previous_valid_ecg_days",
        "n_detected_qrs",
        "n_usable_beats",
        "qc_label",
        "source_relpath",
        "error",
    ]

    df[
        [c for c in longitudinal_cols if c in df.columns]
    ].to_csv(
        longitudinal_path,
        index=False,
    )

    save_hdf5(
        path=h5_path,
        df=df,
        median_beats=median_beats,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    valid = df["valid_for_projection"].astype(bool)

    subject_counts_all = (
        df.groupby("subject_id")["record_id"]
        .count()
    )

    subject_counts_valid = (
        df.loc[valid]
        .groupby("subject_id")["record_id"]
        .count()
    )

    dated_subject_counts_valid = (
        df.loc[
            valid & df["has_acquisition_datetime"].astype(bool)
        ]
        .groupby("subject_id")["acquisition_datetime"]
        .nunique()
    )

    summary = {
        "script": Path(__file__).name,
        "step": STEP,
        "input_root": str(input_root),
        "results_dir": str(results_dir),
        "n_records_discovered": int(len(df)),
        "n_unique_subjects": int(df["subject_id"].nunique()),
        "n_records_with_datetime": int(
            df["has_acquisition_datetime"].sum()
        ),
        "n_records_valid_for_projection": int(valid.sum()),
        "n_subjects_with_at_least_2_records": int(
            (subject_counts_all >= 2).sum()
        ),
        "n_subjects_with_at_least_3_records": int(
            (subject_counts_all >= 3).sum()
        ),
        "n_subjects_with_at_least_2_valid_records": int(
            (subject_counts_valid >= 2).sum()
        ),
        "n_subjects_with_at_least_3_valid_records": int(
            (subject_counts_valid >= 3).sum()
        ),
        "n_subjects_with_at_least_2_valid_dated_visits": int(
            (dated_subject_counts_valid >= 2).sum()
        ),
        "n_subjects_with_at_least_3_valid_dated_visits": int(
            (dated_subject_counts_valid >= 3).sum()
        ),
        "qc_counts": {
            str(k): int(v)
            for k, v in df["qc_label"]
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "records_with_errors": int(
            df["error"].fillna("").astype(str).str.len().gt(0).sum()
        ),
        "preprocessing": {
            "target_sampling_rate_hz": TARGET_FS_HZ,
            "first_window_seconds": WINDOW_SECONDS,
            "filter": {
                "type": "Butterworth band-pass",
                "order": FILTER_ORDER,
                "low_hz": LOW_HZ,
                "high_hz": HIGH_HZ,
                "zero_phase": True,
            },
            "median_beat": {
                "pre_ms": PRE_MS,
                "post_ms": POST_MS,
                "samples": BEAT_SAMPLES,
                "common_multilead_qrs_fiducials": True,
                "pointwise_median_all_usable_beats": True,
                "temporal_warping": False,
                "beatwise_amplitude_normalization": False,
                "minimum_usable_beats_for_projection": MIN_BEATS_FOR_PROJECTION,
            },
        },
        "outputs": {
            "hdf5": str(h5_path),
            "qc_csv": str(qc_path),
            "longitudinal_csv": str(longitudinal_path),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 86)
    print("Completed")
    print("=" * 86)
    print(
        f"Records discovered                    : {summary['n_records_discovered']:,}"
    )
    print(
        f"Unique subjects                       : {summary['n_unique_subjects']:,}"
    )
    print(
        f"Valid median beats                    : {summary['n_records_valid_for_projection']:,}"
    )
    print(
        f"Subjects >=2 valid dated ECG visits   : "
        f"{summary['n_subjects_with_at_least_2_valid_dated_visits']:,}"
    )
    print(
        f"Subjects >=3 valid dated ECG visits   : "
        f"{summary['n_subjects_with_at_least_3_valid_dated_visits']:,}"
    )
    print(
        f"Records with errors                   : {summary['records_with_errors']:,}"
    )
    print()
    print(f"HDF5        : {h5_path}")
    print(f"QC CSV      : {qc_path}")
    print(f"Longitudinal: {longitudinal_path}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
