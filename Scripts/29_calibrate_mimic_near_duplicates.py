#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
29_calibrate_mimic_near_duplicates.py

SMASH-HCM -- MIMIC-IV near-duplicate ECG calibration.

Purpose
-------
Quantify within-subject similarity among valid MIMIC-IV median beats produced by
Step 28 and derive a data-driven candidate threshold for approximately identical
ECG waveforms before any longitudinal temporal collapse.

This step DOES NOT remove records and DOES NOT project ECGs through a VAE.
It is an audit/calibration step only.

Primary waveform distance
-------------------------
For two median beats X1 and X2 in physical mV, each with shape [12, 500],

    symmetric NRMSE = RMSE(X1-X2) /
        sqrt(0.5 * (RMS(X1)^2 + RMS(X2)^2))

No per-record or per-lead amplitude normalization is applied. Therefore true
amplitude differences remain part of the distance.

Complementary metrics
---------------------
- flattened Pearson correlation across all 12 x 500 samples
- per-lead symmetric NRMSE (median and maximum across leads)
- global RMSE / MAE / maximum absolute difference in mV
- RMS amplitude ratio
- exact float32 waveform identity
- acquisition-time separation

Calibration strategy
--------------------
All valid within-subject pairs are measured. Time separation is used only to
stratify the empirical distributions; it is NOT part of the waveform definition
of near-identity.

A two-component Gaussian mixture is fit to log10(NRMSE) among short-interval
pairs (default <=1 day). If a clearly separated low-distance component is found,
the intersection of the two fitted component densities is reported as a
candidate NRMSE threshold. This is a recommendation for inspection, not an
automatic deduplication rule.

Default input
-------------
Results/28_MIMIC_SMASH_MedianBeats/mimic_smash_median_beats.h5

Default outputs
---------------
Results/29_MIMIC_NearDuplicateCalibration/
    mimic_within_subject_pair_metrics.csv
    mimic_pair_distance_by_timebin.csv
    mimic_near_duplicate_threshold_diagnostics.csv
    mimic_near_duplicate_threshold.json
    mimic_near_duplicate_calibration_report.txt
    mimic_nrmse_by_timebin.png
    mimic_nrmse_vs_time.png
    mimic_corr_vs_nrmse.png

Run
---
python Scripts/29_calibrate_mimic_near_duplicates.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


STEP = 29
STEP_NAME = "MIMIC_NearDuplicateCalibration"
EPS = 1e-12
GMM_SEED = 20261021

TIME_BIN_ORDER = [
    "same_timestamp",
    "gt0_to_5min",
    "gt5min_to_1h",
    "gt1h_to_1d",
    "gt1d_to_7d",
    "gt7d_to_30d",
    "gt30d",
    "unknown_time",
]


def infer_project_root() -> Path:
    """Locate the SMASH-HCM project root from script location or cwd."""
    candidates = []

    try:
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    except NameError:
        pass

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
        "Cannot locate SMASH-HCM project root (expected Data/ and Results/)."
    )


def parse_args():
    root = infer_project_root()

    default_h5 = (
        root
        / "Results"
        / "28_MIMIC_SMASH_MedianBeats"
        / "mimic_smash_median_beats.h5"
    )

    default_out = (
        root
        / "Results"
        / f"{STEP:02d}_{STEP_NAME}"
    )

    ap = argparse.ArgumentParser(
        description="Calibrate a near-duplicate waveform threshold for MIMIC-IV ECG median beats."
    )
    ap.add_argument(
        "--input-h5",
        type=Path,
        default=default_h5,
        help="Step-28 median-beat HDF5.",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=default_out,
        help="Output directory.",
    )
    ap.add_argument(
        "--gmm-max-days",
        type=float,
        default=1.0,
        help="Maximum time separation used for the short-interval GMM calibration set.",
    )
    ap.add_argument(
        "--gmm-min-separation",
        type=float,
        default=2.0,
        help="Minimum standardized separation between GMM component means required to recommend a threshold.",
    )
    return ap.parse_args()


def decode_str_array(arr) -> np.ndarray:
    out = []
    for x in np.asarray(arr):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def safe_metadata_str(h5: h5py.File, name: str, n: int) -> np.ndarray:
    path = f"metadata/{name}"
    if path not in h5:
        return np.asarray([""] * n, dtype=object)
    return decode_str_array(h5[path][:])


def safe_metadata_bool(h5: h5py.File, name: str, n: int) -> np.ndarray:
    path = f"metadata/{name}"
    if path not in h5:
        return np.zeros(n, dtype=bool)
    return np.asarray(h5[path][:]).astype(bool)


def time_bin(delta_seconds: float) -> str:
    if not np.isfinite(delta_seconds):
        return "unknown_time"
    if abs(delta_seconds) < 0.5:
        return "same_timestamp"
    if delta_seconds <= 5.0 * 60.0:
        return "gt0_to_5min"
    if delta_seconds <= 60.0 * 60.0:
        return "gt5min_to_1h"
    if delta_seconds <= 86400.0:
        return "gt1h_to_1d"
    if delta_seconds <= 7.0 * 86400.0:
        return "gt1d_to_7d"
    if delta_seconds <= 30.0 * 86400.0:
        return "gt7d_to_30d"
    return "gt30d"


def waveform_hash(x: np.ndarray) -> str:
    """Exact-content hash after canonical float32 contiguous conversion."""
    y = np.ascontiguousarray(x, dtype=np.float32)
    return hashlib.sha256(y.tobytes()).hexdigest()


def pearson_flat(x1: np.ndarray, x2: np.ndarray) -> float:
    a = np.asarray(x1, dtype=np.float64).ravel()
    b = np.asarray(x2, dtype=np.float64).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da <= EPS or db <= EPS:
        return np.nan
    return float(np.dot(a, b) / (da * db))


def pair_metrics(x1: np.ndarray, x2: np.ndarray) -> dict:
    a = np.asarray(x1, dtype=np.float64)
    b = np.asarray(x2, dtype=np.float64)
    d = a - b

    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    max_abs = float(np.max(np.abs(d)))

    rms1 = float(np.sqrt(np.mean(a * a)))
    rms2 = float(np.sqrt(np.mean(b * b)))
    denom = math.sqrt(0.5 * (rms1 * rms1 + rms2 * rms2))
    nrmse = rmse / denom if denom > EPS else np.nan

    amp_ratio = (
        max(rms1, rms2) / max(min(rms1, rms2), EPS)
        if max(rms1, rms2) > EPS
        else np.nan
    )

    lead_rmse = np.sqrt(np.mean(d * d, axis=1))
    lead_rms1 = np.sqrt(np.mean(a * a, axis=1))
    lead_rms2 = np.sqrt(np.mean(b * b, axis=1))
    lead_denom = np.sqrt(0.5 * (lead_rms1 * lead_rms1 + lead_rms2 * lead_rms2))

    lead_nrmse = np.full(a.shape[0], np.nan, dtype=np.float64)
    good = lead_denom > EPS
    lead_nrmse[good] = lead_rmse[good] / lead_denom[good]

    return {
        "rmse_mv": rmse,
        "mae_mv": mae,
        "max_abs_diff_mv": max_abs,
        "symmetric_nrmse": float(nrmse),
        "pearson_r": pearson_flat(a, b),
        "rms1_mv": rms1,
        "rms2_mv": rms2,
        "rms_amplitude_ratio": float(amp_ratio),
        "median_lead_nrmse": float(np.nanmedian(lead_nrmse)),
        "max_lead_nrmse": float(np.nanmax(lead_nrmse)),
    }


def gaussian_intersection(m1, s1, w1, m2, s2, w2):
    """Return a density intersection between component means, if available."""
    s1 = max(float(s1), 1e-9)
    s2 = max(float(s2), 1e-9)
    w1 = max(float(w1), 1e-12)
    w2 = max(float(w2), 1e-12)

    # Solve log(w1/s1) - (x-m1)^2/(2s1^2)
    #     = log(w2/s2) - (x-m2)^2/(2s2^2)
    A = 1.0 / (2.0 * s2 * s2) - 1.0 / (2.0 * s1 * s1)
    B = m1 / (s1 * s1) - m2 / (s2 * s2)
    C = (
        (m2 * m2) / (2.0 * s2 * s2)
        - (m1 * m1) / (2.0 * s1 * s1)
        + math.log((w1 / s1) / (w2 / s2))
    )

    roots = []
    if abs(A) < 1e-12:
        if abs(B) > 1e-12:
            roots = [-C / B]
    else:
        disc = B * B - 4.0 * A * C
        if disc >= 0:
            sq = math.sqrt(disc)
            roots = [(-B - sq) / (2.0 * A), (-B + sq) / (2.0 * A)]

    lo = min(m1, m2)
    hi = max(m1, m2)
    between = [x for x in roots if lo <= x <= hi]
    if between:
        return float(between[0])
    return np.nan


def fit_gmm_threshold(pair_df: pd.DataFrame, max_days: float, min_separation: float) -> dict:
    d = pair_df.copy()
    d = d[
        np.isfinite(d["symmetric_nrmse"].to_numpy(float))
        & np.isfinite(d["delta_days"].to_numpy(float))
        & (d["delta_days"].to_numpy(float) <= float(max_days))
    ].copy()

    if len(d) < 20:
        return {
            "status": "insufficient_short_interval_pairs",
            "n_calibration_pairs": int(len(d)),
            "recommended_threshold_nrmse": None,
        }

    vals = d["symmetric_nrmse"].to_numpy(float)
    vals = np.clip(vals, 1e-12, None)
    logv = np.log10(vals).reshape(-1, 1)

    gm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=GMM_SEED,
        n_init=20,
        max_iter=1000,
    )
    gm.fit(logv)

    means = gm.means_.ravel()
    vars_ = gm.covariances_.reshape(-1)
    weights = gm.weights_.ravel()
    order = np.argsort(means)

    lo_idx, hi_idx = int(order[0]), int(order[1])
    m_lo, m_hi = float(means[lo_idx]), float(means[hi_idx])
    s_lo = math.sqrt(max(float(vars_[lo_idx]), 1e-12))
    s_hi = math.sqrt(max(float(vars_[hi_idx]), 1e-12))
    w_lo, w_hi = float(weights[lo_idx]), float(weights[hi_idx])

    pooled_sd = math.sqrt(0.5 * (s_lo * s_lo + s_hi * s_hi))
    separation = (m_hi - m_lo) / pooled_sd if pooled_sd > 0 else np.inf

    x_int = gaussian_intersection(m_lo, s_lo, w_lo, m_hi, s_hi, w_hi)
    threshold = 10.0 ** x_int if np.isfinite(x_int) else np.nan

    recommend = (
        np.isfinite(threshold)
        and separation >= float(min_separation)
        and 0.001 <= w_lo <= 0.75
    )

    labels = gm.predict(logv)
    low_mask = labels == lo_idx

    out = {
        "status": "recommended" if recommend else "gmm_fit_but_no_stable_recommendation",
        "n_calibration_pairs": int(len(d)),
        "calibration_max_days": float(max_days),
        "gmm_low_component_weight": w_lo,
        "gmm_high_component_weight": w_hi,
        "gmm_low_log10_nrmse_mean": m_lo,
        "gmm_high_log10_nrmse_mean": m_hi,
        "gmm_low_log10_nrmse_sd": s_lo,
        "gmm_high_log10_nrmse_sd": s_hi,
        "gmm_component_separation_sd": float(separation),
        "gmm_intersection_log10_nrmse": float(x_int) if np.isfinite(x_int) else None,
        "gmm_intersection_nrmse": float(threshold) if np.isfinite(threshold) else None,
        "recommended_threshold_nrmse": float(threshold) if recommend else None,
        "low_component_pair_count": int(np.sum(low_mask)),
        "low_component_nrmse_median": float(np.median(vals[low_mask])) if np.any(low_mask) else None,
        "low_component_nrmse_p95": float(np.percentile(vals[low_mask], 95)) if np.any(low_mask) else None,
        "low_component_nrmse_p99": float(np.percentile(vals[low_mask], 99)) if np.any(low_mask) else None,
    }
    return out


def summarize_by_timebin(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in TIME_BIN_ORDER:
        g = pair_df[pair_df["time_bin"] == label]
        if g.empty:
            continue
        x = g["symmetric_nrmse"].to_numpy(float)
        r = g["pearson_r"].to_numpy(float)
        rows.append({
            "time_bin": label,
            "n_pairs": int(len(g)),
            "n_exact_waveform_matches": int(g["exact_waveform_match"].sum()),
            "nrmse_min": float(np.nanmin(x)),
            "nrmse_p01": float(np.nanpercentile(x, 1)),
            "nrmse_p05": float(np.nanpercentile(x, 5)),
            "nrmse_median": float(np.nanmedian(x)),
            "nrmse_p95": float(np.nanpercentile(x, 95)),
            "pearson_median": float(np.nanmedian(r)),
            "pearson_p05": float(np.nanpercentile(r, 5)),
            "pearson_p95": float(np.nanpercentile(r, 95)),
        })
    return pd.DataFrame(rows)


def threshold_diagnostics(pair_df: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if threshold is None or not np.isfinite(threshold):
        return pd.DataFrame()

    rows = []
    for label in ["all_pairs", *TIME_BIN_ORDER]:
        g = pair_df if label == "all_pairs" else pair_df[pair_df["time_bin"] == label]
        if g.empty:
            continue
        hit = g["symmetric_nrmse"] <= float(threshold)
        strict_corr = hit & (g["pearson_r"] >= 0.995)
        rows.append({
            "group": label,
            "n_pairs": int(len(g)),
            "threshold_nrmse": float(threshold),
            "n_below_threshold": int(hit.sum()),
            "fraction_below_threshold": float(hit.mean()),
            "n_below_threshold_and_r_ge_0_995": int(strict_corr.sum()),
            "fraction_below_threshold_and_r_ge_0_995": float(strict_corr.mean()),
        })
    return pd.DataFrame(rows)


def make_figures(pair_df: pd.DataFrame, out_dir: Path, threshold: float | None) -> None:
    finite = pair_df[np.isfinite(pair_df["symmetric_nrmse"])].copy()
    finite = finite[finite["symmetric_nrmse"] > 0].copy()

    if not finite.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = []
        labels = []
        for label in TIME_BIN_ORDER:
            vals = finite.loc[finite["time_bin"] == label, "symmetric_nrmse"].to_numpy(float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                data.append(np.log10(vals))
                labels.append(label)
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            if threshold is not None and np.isfinite(threshold) and threshold > 0:
                ax.axhline(np.log10(threshold), linestyle="--", linewidth=1.2)
            ax.set_ylabel("log10 symmetric NRMSE")
            ax.set_xlabel("Acquisition-time separation")
            ax.tick_params(axis="x", rotation=35)
            ax.set_title("Within-subject ECG waveform distance by time separation")
            fig.tight_layout()
            fig.savefig(out_dir / "mimic_nrmse_by_timebin.png", dpi=180)
        plt.close(fig)

        timed = finite[
            np.isfinite(finite["delta_days"].to_numpy(float))
            & (finite["delta_days"] >= 0)
        ].copy()
        if not timed.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            x = np.log10(timed["delta_days"].to_numpy(float) + 1.0 / 1440.0)
            y = np.log10(timed["symmetric_nrmse"].to_numpy(float))
            ax.scatter(x, y, s=8, alpha=0.25)
            if threshold is not None and np.isfinite(threshold) and threshold > 0:
                ax.axhline(np.log10(threshold), linestyle="--", linewidth=1.2)
            ax.set_xlabel("log10(time separation in days + 1 minute)")
            ax.set_ylabel("log10 symmetric NRMSE")
            ax.set_title("Within-subject ECG similarity versus acquisition interval")
            fig.tight_layout()
            fig.savefig(out_dir / "mimic_nrmse_vs_time.png", dpi=180)
            plt.close(fig)

        corr = finite[np.isfinite(finite["pearson_r"])].copy()
        if not corr.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(
                np.log10(corr["symmetric_nrmse"].to_numpy(float)),
                corr["pearson_r"].to_numpy(float),
                s=8,
                alpha=0.25,
            )
            if threshold is not None and np.isfinite(threshold) and threshold > 0:
                ax.axvline(np.log10(threshold), linestyle="--", linewidth=1.2)
            ax.set_xlabel("log10 symmetric NRMSE")
            ax.set_ylabel("Flattened Pearson correlation")
            ax.set_title("Correlation versus amplitude-preserving waveform distance")
            fig.tight_layout()
            fig.savefig(out_dir / "mimic_corr_vs_nrmse.png", dpi=180)
            plt.close(fig)


def main():
    args = parse_args()
    input_h5 = args.input_h5.resolve()
    out_dir = args.results_dir.resolve()

    if not input_h5.exists():
        raise FileNotFoundError(f"Step-28 HDF5 not found:\n{input_h5}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as h5:
        if "signals/median_all_mv" not in h5:
            raise KeyError("HDF5 missing signals/median_all_mv")

        beats = np.asarray(h5["signals/median_all_mv"][:], dtype=np.float32)
        if beats.ndim != 3 or beats.shape[1:] != (12, 500):
            raise ValueError(f"Expected median beats [N,12,500], found {beats.shape}")

        n = int(beats.shape[0])
        subject_id = safe_metadata_str(h5, "subject_id", n)
        record_id = safe_metadata_str(h5, "record_id", n)
        source_relpath = safe_metadata_str(h5, "source_relpath", n)
        acquisition_datetime = pd.to_datetime(
            safe_metadata_str(h5, "acquisition_datetime", n),
            errors="coerce",
        )
        valid = safe_metadata_bool(h5, "valid_for_projection", n)

    finite_waveform = np.isfinite(beats).all(axis=(1, 2))
    keep = valid & finite_waveform & (pd.Series(subject_id).astype(str).str.len().to_numpy() > 0)

    meta = pd.DataFrame({
        "row_index": np.arange(n, dtype=np.int64),
        "subject_id": subject_id,
        "record_id": record_id,
        "source_relpath": source_relpath,
        "acquisition_datetime": acquisition_datetime,
        "valid_for_projection": valid,
        "finite_waveform": finite_waveform,
    })
    meta = meta.loc[keep].copy().reset_index(drop=True)

    hashes = {}
    for idx in tqdm(meta["row_index"].to_numpy(int), desc="Hashing valid median beats", unit="ECG"):
        hashes[int(idx)] = waveform_hash(beats[int(idx)])

    groups = list(meta.groupby("subject_id", sort=True))
    pair_rows = []

    total_pairs = int(sum(len(g) * (len(g) - 1) // 2 for _, g in groups))
    progress = tqdm(total=total_pairs, desc="Measuring within-subject pairs", unit="pair")

    for sid, g in groups:
        if len(g) < 2:
            continue

        rows = list(g.itertuples(index=False))
        for a, b in combinations(rows, 2):
            i = int(a.row_index)
            j = int(b.row_index)

            dt1 = a.acquisition_datetime
            dt2 = b.acquisition_datetime
            if pd.notna(dt1) and pd.notna(dt2):
                delta_seconds = abs((dt2 - dt1).total_seconds())
                delta_days = delta_seconds / 86400.0
            else:
                delta_seconds = np.nan
                delta_days = np.nan

            m = pair_metrics(beats[i], beats[j])
            exact = hashes[i] == hashes[j]

            pair_rows.append({
                "subject_id": str(sid),
                "row_index_1": i,
                "row_index_2": j,
                "record_id_1": str(a.record_id),
                "record_id_2": str(b.record_id),
                "acquisition_datetime_1": dt1,
                "acquisition_datetime_2": dt2,
                "source_relpath_1": str(a.source_relpath),
                "source_relpath_2": str(b.source_relpath),
                "delta_seconds": float(delta_seconds) if np.isfinite(delta_seconds) else np.nan,
                "delta_days": float(delta_days) if np.isfinite(delta_days) else np.nan,
                "time_bin": time_bin(delta_seconds),
                "exact_waveform_match": bool(exact),
                **m,
            })
            progress.update(1)

    progress.close()

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        raise RuntimeError("No within-subject valid ECG pairs were available.")

    pair_df["time_bin"] = pd.Categorical(
        pair_df["time_bin"],
        categories=TIME_BIN_ORDER,
        ordered=True,
    )
    pair_df = pair_df.sort_values(
        ["subject_id", "acquisition_datetime_1", "acquisition_datetime_2", "record_id_1", "record_id_2"],
        kind="mergesort",
    ).reset_index(drop=True)

    gmm = fit_gmm_threshold(
        pair_df,
        max_days=float(args.gmm_max_days),
        min_separation=float(args.gmm_min_separation),
    )
    threshold = gmm.get("recommended_threshold_nrmse")

    if threshold is not None:
        pair_df["below_recommended_nrmse_threshold"] = (
            pair_df["symmetric_nrmse"] <= float(threshold)
        )
        pair_df["below_threshold_and_r_ge_0_995"] = (
            pair_df["below_recommended_nrmse_threshold"]
            & (pair_df["pearson_r"] >= 0.995)
        )
    else:
        pair_df["below_recommended_nrmse_threshold"] = False
        pair_df["below_threshold_and_r_ge_0_995"] = False

    by_time = summarize_by_timebin(pair_df)
    diag = threshold_diagnostics(pair_df, threshold)

    pair_path = out_dir / "mimic_within_subject_pair_metrics.csv"
    time_path = out_dir / "mimic_pair_distance_by_timebin.csv"
    diag_path = out_dir / "mimic_near_duplicate_threshold_diagnostics.csv"
    json_path = out_dir / "mimic_near_duplicate_threshold.json"
    report_path = out_dir / "mimic_near_duplicate_calibration_report.txt"

    pair_df.to_csv(pair_path, index=False)
    by_time.to_csv(time_path, index=False)
    diag.to_csv(diag_path, index=False)

    summary = {
        "script": Path(__file__).name,
        "step": STEP,
        "input_h5": str(input_h5),
        "results_dir": str(out_dir),
        "n_h5_records": int(n),
        "n_valid_finite_records": int(len(meta)),
        "n_subjects_with_valid_records": int(meta["subject_id"].nunique()),
        "n_within_subject_pairs": int(len(pair_df)),
        "n_exact_waveform_matches": int(pair_df["exact_waveform_match"].sum()),
        "metric": {
            "primary": "symmetric_nrmse",
            "definition": "RMSE(X1-X2) / sqrt(0.5*(RMS(X1)^2+RMS(X2)^2))",
            "amplitude_normalization": "none",
            "complementary": [
                "pearson_r",
                "median_lead_nrmse",
                "max_lead_nrmse",
                "rmse_mv",
                "mae_mv",
                "max_abs_diff_mv",
                "rms_amplitude_ratio",
            ],
        },
        "gmm_calibration": gmm,
        "interpretation": (
            "Time separation is used only for empirical stratification. "
            "The GMM intersection, when recommended, is a candidate waveform-distance threshold "
            "for manual scientific review and is not itself an automatic record-deduplication rule."
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_figures(pair_df, out_dir, threshold)

    lines = []
    lines.append("SMASH-HCM Step 29 -- MIMIC near-duplicate ECG calibration\n")
    lines.append(f"Input HDF5: {input_h5}\n")
    lines.append(f"Valid finite ECGs: {len(meta):,}\n")
    lines.append(f"Subjects with valid ECGs: {meta['subject_id'].nunique():,}\n")
    lines.append(f"Within-subject pairs: {len(pair_df):,}\n")
    lines.append(f"Exact waveform matches: {int(pair_df['exact_waveform_match'].sum()):,}\n\n")

    lines.append("Time-bin counts:\n")
    for _, r in by_time.iterrows():
        lines.append(
            f"  {r['time_bin']}: n={int(r['n_pairs']):,}, "
            f"median NRMSE={r['nrmse_median']:.6g}, "
            f"p01={r['nrmse_p01']:.6g}, "
            f"median r={r['pearson_median']:.6g}\n"
        )

    lines.append("\nGMM calibration:\n")
    for key, value in gmm.items():
        lines.append(f"  {key}: {value}\n")

    if threshold is not None:
        hits = int((pair_df["symmetric_nrmse"] <= float(threshold)).sum())
        hits_corr = int(pair_df["below_threshold_and_r_ge_0_995"].sum())
        lines.append(
            f"\nCandidate NRMSE threshold: {float(threshold):.8g}\n"
            f"Pairs below threshold: {hits:,}\n"
            f"Pairs below threshold AND r>=0.995: {hits_corr:,}\n"
        )
    else:
        lines.append(
            "\nNo stable automatic NRMSE threshold was recommended by the GMM. "
            "Inspect the exported distributions before choosing a cut-off.\n"
        )

    lines.append(
        "\nImportant: waveform near-identity is not equivalent to record duplication. "
        "Temporal collapse/deduplication should be defined only after reviewing these waveform-distance results.\n"
    )

    report_path.write_text("".join(lines), encoding="utf-8")

    print("".join(lines))
    print(f"Pair metrics : {pair_path}")
    print(f"Time bins    : {time_path}")
    print(f"Diagnostics  : {diag_path}")
    print(f"Threshold    : {json_path}")
    print(f"Report       : {report_path}")


if __name__ == "__main__":
    main()
