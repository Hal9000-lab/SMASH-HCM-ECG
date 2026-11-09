#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
14_quantify_hcm_static_latent_ecg_features.py

SMASH-HCM -- Model 2 / Step 14
Quantitative physiological interpretation of HCM-static beta-VAE latent axes.

PURPOSE
-------
For each HCM-specific latent coordinate z_k^HCM:

1) quantify associations between z_k and classical ECG morphology descriptors
   measured on REAL HCM median beats;
2) quantify how the same descriptors change along the decoder traversal
   (-3 ... +3 HCM-train SD);
3) compare the sign and monotonicity of real-data associations with decoder
   traversal effects;
4) produce a compact interpretation table for each latent dimension.

PRIMARY INFERENCE POPULATION
----------------------------
HCM-train only.

SECONDARY / CONFIRMATORY POPULATIONS
------------------------------------
HCM-validation, HCM-test, and all HCM baseline ECGs.

IMPORTANT
---------
- No Normal ECG is used for primary Model-2 latent interpretation.
- All morphology amplitudes are measured in physical mV.
- No per-record or per-lead amplitude normalization is applied.
- One multilead QRS onset/offset is used for all leads.
- Raw R/S ratios are NOT used because they are numerically unstable when S
  approaches zero. A bounded R/S balance is used instead:

      RS_balance = (R - Sdepth) / (R + Sdepth + eps)

- Sokolow-Lyon and Cornell-like values are continuous morphology descriptors,
  not diagnostic labels.
- QT-end is a heuristic research surrogate and not a clinical QT measurement.

Expected inputs
---------------
Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5
Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv
Results/13_HCM_Static_LatentTraversal/latent_traversals_hcm_static.npz

Outputs
-------
Results/14_HCM_Static_LatentECGFeatures/
├── ecg_classical_features.csv
├── feature_extraction_qc.csv
├── latent_feature_associations_hcm_train.csv
├── latent_feature_associations_hcm_val.csv
├── latent_feature_associations_hcm_test.csv
├── latent_feature_associations_hcm_all.csv
├── latent_traversal_features.csv
├── latent_traversal_feature_slopes.csv
├── latent_real_vs_traversal_concordance.csv
├── latent_interpretation_priority_summary.csv
├── run_summary.json
└── figures/
    ├── latent_feature_heatmap_hcm_train.png
    └── latent_traversal_concordance.png

Run
---
python Scripts\\14_quantify_hcm_static_latent_ecg_features.py
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats


SEED = 20261004

FS_HZ = 500.0
BEAT_START_MS = -350.0
N_LEADS = 12

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"]

SPLIT_NAMES = {
    0: "hcm_train",
    1: "hcm_val",
    2: "hcm_test",
    3: "normal_eval",
}

EPS_AMP = 0.05  # mV regularization for amplitude ratios
EPS_NUM = 1e-12


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--h5",
        type=Path,
        default=Path("Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5"),
    )
    p.add_argument(
        "--latents",
        type=Path,
        default=Path("Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv"),
    )
    p.add_argument(
        "--traversals",
        type=Path,
        default=Path(
            "Results/13_HCM_Static_LatentTraversal/"
            "latent_traversals_hcm_static.npz"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/14_HCM_Static_LatentECGFeatures"),
    )
    p.add_argument("--fs", type=float, default=FS_HZ)
    p.add_argument("--beat-start-ms", type=float, default=BEAT_START_MS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--top-n", type=int, default=8)
    return p.parse_args()


# =====================================================================
# Utilities
# =====================================================================

def decode_array(a):
    arr = np.asarray(a)
    if arr.dtype.kind in ("S", "O", "U"):
        out = []
        for x in arr.ravel():
            if isinstance(x, bytes):
                out.append(x.decode("utf-8", errors="replace"))
            else:
                out.append(str(x))
        return np.asarray(out, dtype=object).reshape(arr.shape)
    return arr


def time_axis_ms(n_samples: int, fs: float, beat_start_ms: float):
    return beat_start_ms + np.arange(n_samples, dtype=float) * 1000.0 / fs


def nearest_index(t_ms, target_ms):
    return int(np.argmin(np.abs(t_ms - target_ms)))


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else np.nan


def safe_rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nanmean(x * x))) if np.any(np.isfinite(x)) else np.nan


def integrate(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 2 or not np.all(np.isfinite(y)):
        return np.nan
    return float(np.trapezoid(y, x))


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    finite = np.isfinite(p)
    pf = p[finite]
    if len(pf) == 0:
        return q

    order = np.argsort(pf)
    ranked = pf[order]
    m = len(ranked)

    qrank = ranked * m / np.arange(1, m + 1)
    qrank = np.minimum.accumulate(qrank[::-1])[::-1]
    qrank = np.clip(qrank, 0.0, 1.0)

    tmp = np.empty(m, dtype=float)
    tmp[order] = qrank
    q[finite] = tmp
    return q


def robust_corr(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if np.sum(mask) < 10:
        return np.nan, np.nan, int(np.sum(mask))

    xx = x[mask]
    yy = y[mask]

    if np.std(xx) <= 0 or np.std(yy) <= 0:
        return np.nan, np.nan, int(len(xx))

    if method == "spearman":
        r, p = stats.spearmanr(xx, yy)
    elif method == "pearson":
        r, p = stats.pearsonr(xx, yy)
    else:
        raise ValueError(method)

    return float(r), float(p), int(len(xx))


def standardized_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)

    if np.sum(m) < 10:
        return np.nan, np.nan

    xx = x[m]
    yy = y[m]

    sx = np.std(xx, ddof=1)
    sy = np.std(yy, ddof=1)

    if sx <= 0 or sy <= 0:
        return np.nan, np.nan

    lr = stats.linregress(xx, yy)
    slope_std = float(lr.slope * sx / sy)
    return float(lr.slope), slope_std


# =====================================================================
# QRS delineation
# =====================================================================

def smooth_1d(x, n):
    n = max(1, int(n))
    if n == 1:
        return np.asarray(x, dtype=float)
    kernel = np.ones(n, dtype=float) / n
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="same")


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-8:
        scale = np.std(x)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (x - med) / scale


def detect_multilead_qrs_boundaries(
    beat_mv: np.ndarray,
    fs: float,
    beat_start_ms: float,
):
    """
    Robust common QRS boundary detector.

    Uses normalized multilead gradient energy only for boundary localization.
    Final morphology features always use original mV signals.

    Accepted physiological duration: 45--180 ms.
    Fallback: -60 to +100 ms.
    """
    n_leads, n_samples = beat_mv.shape
    t = time_axis_ms(n_samples, fs, beat_start_ms)

    dt_s = 1.0 / fs

    norm_grad = []
    for li in range(n_leads):
        g = np.gradient(beat_mv[li], dt_s)
        # Detect normalization only; does not alter morphology measurements.
        g = robust_scale(g)
        norm_grad.append(g * g)

    energy = np.median(np.vstack(norm_grad), axis=0)
    energy = smooth_1d(energy, max(1, round(0.012 * fs)))

    search = (t >= -120.0) & (t <= 160.0)
    idx_search = np.where(search)[0]

    fallback_on = nearest_index(t, -60.0)
    fallback_off = nearest_index(t, 100.0)

    if len(idx_search) < 10:
        return fallback_on, fallback_off, False, "fallback_short_search"

    peak = idx_search[np.argmax(energy[idx_search])]
    peak_val = float(energy[peak])

    baseline_mask = ((t >= -300) & (t <= -150)) | ((t >= 180) & (t <= 300))
    baseline_vals = energy[baseline_mask]
    if len(baseline_vals):
        bmed = float(np.median(baseline_vals))
        bmad = float(np.median(np.abs(baseline_vals - bmed)))
    else:
        bmed = float(np.median(energy))
        bmad = float(np.median(np.abs(energy - bmed)))

    threshold = max(
        bmed + 3.0 * 1.4826 * bmad,
        bmed + 0.08 * max(peak_val - bmed, 0.0),
    )

    left_limit = nearest_index(t, -140)
    right_limit = nearest_index(t, 180)

    on = peak
    while on > left_limit and energy[on] > threshold:
        on -= 1

    off = peak
    while off < right_limit and energy[off] > threshold:
        off += 1

    # Slight extension to include low-gradient beginning/end of QRS.
    extension = int(round(0.008 * fs))
    on = max(0, on - extension)
    off = min(n_samples - 1, off + extension)

    duration_ms = t[off] - t[on]

    accepted = (
        np.isfinite(duration_ms)
        and 45.0 <= duration_ms <= 180.0
        and t[on] < 0.0
        and t[off] > 20.0
    )

    if not accepted:
        return fallback_on, fallback_off, False, "fallback_duration"

    return int(on), int(off), True, "gradient_energy"


# =====================================================================
# ECG feature extraction
# =====================================================================

def baseline_for_qrs(x, t, qrs_on_idx):
    qrs_on_ms = t[qrs_on_idx]

    lo = qrs_on_ms - 70.0
    hi = qrs_on_ms - 20.0
    m = (t >= lo) & (t <= hi)

    if np.sum(m) < 5:
        m = (t >= -140.0) & (t <= -90.0)

    if np.sum(m) < 5:
        m = (t >= -300.0) & (t <= -200.0)

    return float(np.median(x[m])) if np.any(m) else 0.0


def contiguous_below_duration(y, peak_idx, fs):
    """
    Approximate Q-wave duration immediately before R.
    y is baseline-corrected QRS segment.
    """
    if peak_idx <= 0:
        return 0.0

    pre = y[:peak_idx + 1]
    if len(pre) == 0:
        return 0.0

    q_idx = int(np.argmin(pre))
    if pre[q_idx] >= -0.02:
        return 0.0

    left = q_idx
    while left > 0 and pre[left] < -0.01:
        left -= 1

    right = q_idx
    while right < peak_idx and pre[right] < -0.01:
        right += 1

    return float(max(0, right - left) * 1000.0 / fs)


def heuristic_t_end(
    global_t_envelope,
    t,
    t_peak_idx,
    qrs_off_idx,
):
    """
    Research-only T-end surrogate.

    Search after global T peak for first sustained return toward baseline
    below 15% of peak envelope. Must occur after QRS offset + 100 ms.
    """
    env = np.asarray(global_t_envelope, dtype=float)
    if not np.all(np.isfinite(env)):
        return np.nan, False

    peak_val = float(env[t_peak_idx])
    if peak_val <= 0:
        return np.nan, False

    threshold = 0.15 * peak_val
    min_idx = max(
        t_peak_idx + 1,
        qrs_off_idx + 1,
    )

    sustain = 8  # 16 ms at 500 Hz
    for i in range(min_idx, len(env) - sustain):
        if t[i] > 580:
            break
        if np.all(env[i:i+sustain] <= threshold):
            return float(t[i]), True

    return np.nan, False


def extract_ecg_features(
    beat_mv: np.ndarray,
    fs: float,
    beat_start_ms: float,
    rr_ms: float = np.nan,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    x = np.asarray(beat_mv, dtype=float)

    if x.shape[0] != N_LEADS:
        raise ValueError(f"Expected 12 leads, got {x.shape}")

    n_samples = x.shape[1]
    t = time_axis_ms(n_samples, fs, beat_start_ms)

    qrs_on, qrs_off, qrs_ok, qrs_method = detect_multilead_qrs_boundaries(
        x, fs, beat_start_ms
    )

    qrs_on_ms = float(t[qrs_on])
    qrs_off_ms = float(t[qrs_off])
    qrs_duration_ms = float(qrs_off_ms - qrs_on_ms)

    feats: Dict[str, float] = {}
    qc = {
        "qrs_boundary_ok": bool(qrs_ok),
        "qrs_boundary_method": qrs_method,
        "qrs_on_ms": qrs_on_ms,
        "qrs_off_ms": qrs_off_ms,
        "qrs_duration_ms": qrs_duration_ms,
        "tend_ok": False,
    }

    feats["qrs_on_ms"] = qrs_on_ms
    feats["qrs_off_ms"] = qrs_off_ms
    feats["qrs_duration_ms"] = qrs_duration_ms

    qrs_net_areas = {}
    qrs_abs_areas = {}
    r_amp = {}
    s_depth = {}

    corrected = np.zeros_like(x)

    for li, lead in enumerate(LEAD_NAMES):
        sig = x[li]
        baseline = baseline_for_qrs(sig, t, qrs_on)
        y = sig - baseline
        corrected[li] = y

        qseg = y[qrs_on:qrs_off + 1]
        tq = t[qrs_on:qrs_off + 1]

        if len(qseg) < 3:
            continue

        r_local = int(np.argmax(qseg))
        r = float(max(0.0, qseg[r_local]))

        pre_r = qseg[:r_local + 1]
        post_r = qseg[r_local:]

        q_val = float(np.min(pre_r)) if len(pre_r) else 0.0
        s_val = float(np.min(post_r)) if len(post_r) else 0.0

        q_depth = float(max(0.0, -q_val))
        s_dep = float(max(0.0, -s_val))

        r_amp[lead] = r
        s_depth[lead] = s_dep

        qdur = contiguous_below_duration(qseg, r_local, fs)

        net_area = integrate(qseg, tq)
        abs_area = integrate(np.abs(qseg), tq)
        p2p = float(np.ptp(qseg))

        qrs_net_areas[lead] = net_area
        qrs_abs_areas[lead] = abs_area

        feats[f"R_mV_{lead}"] = r
        feats[f"Q_mV_{lead}"] = -q_depth
        feats[f"Q_depth_mV_{lead}"] = q_depth
        feats[f"S_mV_{lead}"] = -s_dep
        feats[f"S_depth_mV_{lead}"] = s_dep
        feats[f"Q_duration_ms_{lead}"] = qdur
        feats[f"Q_over_R_reg_{lead}"] = q_depth / (r + EPS_AMP)
        feats[f"RS_balance_{lead}"] = (
            (r - s_dep) / (r + s_dep + EPS_AMP)
        )
        feats[f"QRS_peak_to_peak_mV_{lead}"] = p2p
        feats[f"QRS_net_area_mVms_{lead}"] = net_area
        feats[f"QRS_abs_area_mVms_{lead}"] = abs_area

    # Global QRS morphology
    qrs_corr = corrected[:, qrs_on:qrs_off + 1]
    tq = t[qrs_on:qrs_off + 1]

    feats["QRS_global_rms_mV"] = safe_rms(qrs_corr)
    feats["QRS_global_peak_to_peak_mV"] = float(np.ptp(qrs_corr))
    feats["QRS_global_abs_area_mVms"] = integrate(
        np.sqrt(np.mean(qrs_corr ** 2, axis=0)),
        tq,
    )

    # Continuous LV-voltage descriptors.
    if all(k in s_depth for k in ["V1"]) and all(k in r_amp for k in ["V5", "V6"]):
        feats["SokolowLyon_mV"] = (
            s_depth["V1"] + max(r_amp["V5"], r_amp["V6"])
        )
    else:
        feats["SokolowLyon_mV"] = np.nan

    if "aVL" in r_amp and "V3" in s_depth:
        feats["CornellLike_mV"] = r_amp["aVL"] + s_depth["V3"]
    else:
        feats["CornellLike_mV"] = np.nan

    # Frontal QRS axis from net areas I / aVF.
    ai = qrs_net_areas.get("I", np.nan)
    avf = qrs_net_areas.get("aVF", np.nan)

    if np.isfinite(ai) and np.isfinite(avf) and (abs(ai) + abs(avf) > 1e-9):
        axis_rad = float(np.arctan2(avf, ai))
        axis_deg = float(np.degrees(axis_rad))
        feats["QRS_axis_deg"] = axis_deg
        feats["QRS_axis_cos"] = float(np.cos(axis_rad))
        feats["QRS_axis_sin"] = float(np.sin(axis_rad))
    else:
        feats["QRS_axis_deg"] = np.nan
        feats["QRS_axis_cos"] = np.nan
        feats["QRS_axis_sin"] = np.nan

    # R/S precordial transition as continuous-ish lead index.
    transition = np.nan
    for j, lead in enumerate(PRECORDIAL, start=1):
        if lead in r_amp and lead in s_depth:
            if r_amp[lead] >= s_depth[lead]:
                transition = float(j)
                break
    feats["RS_transition_V_index"] = transition

    # ST and T-wave morphology.
    st40_idx = nearest_index(t, qrs_off_ms + 40.0)
    st60_idx = nearest_index(t, qrs_off_ms + 60.0)
    st80_idx = nearest_index(t, qrs_off_ms + 80.0)

    # T search starts at least 80 ms after QRS offset.
    t_start_ms = max(qrs_off_ms + 80.0, 100.0)
    t_end_ms = min(float(t[-1]), 500.0)

    t_mask = (t >= t_start_ms) & (t <= t_end_ms)
    t_indices = np.where(t_mask)[0]

    perlead_t_amp = []
    perlead_t_peak = []
    perlead_t_area = []

    for li, lead in enumerate(LEAD_NAMES):
        y = corrected[li]

        feats[f"ST60_mV_{lead}"] = (
            float(y[st60_idx]) if 0 <= st60_idx < n_samples else np.nan
        )

        lo = min(st40_idx, st80_idx)
        hi = max(st40_idx, st80_idx) + 1
        if 0 <= lo < hi <= n_samples:
            feats[f"ST_mean_40_80_mV_{lead}"] = float(np.mean(y[lo:hi]))
        else:
            feats[f"ST_mean_40_80_mV_{lead}"] = np.nan

        if len(t_indices) >= 5:
            yt = y[t_indices]
            tt = t[t_indices]

            imax = int(np.argmax(np.abs(yt)))
            tamp = float(yt[imax])
            tabs = float(abs(tamp))
            tpeak = float(tt[imax])
            tarea = integrate(yt, tt)

            feats[f"T_amp_mV_{lead}"] = tamp
            feats[f"T_abs_amp_mV_{lead}"] = tabs
            feats[f"T_peak_ms_{lead}"] = tpeak
            feats[f"T_area_mVms_{lead}"] = tarea
            feats[f"T_polarity_{lead}"] = float(np.sign(tamp))

            perlead_t_amp.append(tamp)
            perlead_t_peak.append(tpeak)
            perlead_t_area.append(tarea)
        else:
            feats[f"T_amp_mV_{lead}"] = np.nan
            feats[f"T_abs_amp_mV_{lead}"] = np.nan
            feats[f"T_peak_ms_{lead}"] = np.nan
            feats[f"T_area_mVms_{lead}"] = np.nan
            feats[f"T_polarity_{lead}"] = np.nan

    if len(t_indices) >= 5:
        global_env = np.sqrt(np.mean(corrected[:, t_indices] ** 2, axis=0))
        ig = int(np.argmax(global_env))
        global_peak_idx = int(t_indices[ig])
        global_t_peak_ms = float(t[global_peak_idx])

        feats["T_global_peak_ms"] = global_t_peak_ms
        feats["T_global_rms_mV"] = safe_rms(corrected[:, t_indices])
        feats["T_global_peak_rms_mV"] = float(global_env[ig])

        qtpeak = global_t_peak_ms - qrs_on_ms
        feats["QTpeak_global_ms"] = float(qtpeak)

        full_env = np.sqrt(np.mean(corrected ** 2, axis=0))
        tend_ms, tend_ok = heuristic_t_end(
            full_env,
            t,
            global_peak_idx,
            qrs_off,
        )
        qc["tend_ok"] = bool(tend_ok)
        qc["tend_ms_est"] = tend_ms

        feats["Tend_global_ms_est"] = tend_ms
        feats["QT_global_ms_est"] = (
            tend_ms - qrs_on_ms if np.isfinite(tend_ms) else np.nan
        )
    else:
        feats["T_global_peak_ms"] = np.nan
        feats["T_global_rms_mV"] = np.nan
        feats["T_global_peak_rms_mV"] = np.nan
        feats["QTpeak_global_ms"] = np.nan
        feats["Tend_global_ms_est"] = np.nan
        feats["QT_global_ms_est"] = np.nan

    # Repolarization-to-depolarization relative energy.
    if len(t_indices) >= 5:
        qrs_rms = feats["QRS_global_rms_mV"]
        t_rms = feats["T_global_rms_mV"]
        feats["T_to_QRS_rms_ratio_reg"] = (
            t_rms / (qrs_rms + EPS_AMP)
        )
    else:
        feats["T_to_QRS_rms_ratio_reg"] = np.nan

    # RR / HR and corrected QTpeak if available.
    feats["RR_ms"] = float(rr_ms) if np.isfinite(rr_ms) else np.nan
    if np.isfinite(rr_ms) and rr_ms > 0:
        rr_s = rr_ms / 1000.0
        feats["HR_bpm"] = 60000.0 / rr_ms

        qtp = feats.get("QTpeak_global_ms", np.nan)
        if np.isfinite(qtp):
            feats["QTpeakc_Bazett_ms"] = qtp / math.sqrt(rr_s)
            feats["QTpeakc_Fridericia_ms"] = qtp / (rr_s ** (1.0/3.0))
        else:
            feats["QTpeakc_Bazett_ms"] = np.nan
            feats["QTpeakc_Fridericia_ms"] = np.nan

        qt = feats.get("QT_global_ms_est", np.nan)
        if np.isfinite(qt):
            feats["QTc_Bazett_ms_est"] = qt / math.sqrt(rr_s)
            feats["QTc_Fridericia_ms_est"] = qt / (rr_s ** (1.0/3.0))
        else:
            feats["QTc_Bazett_ms_est"] = np.nan
            feats["QTc_Fridericia_ms_est"] = np.nan
    else:
        feats["HR_bpm"] = np.nan
        feats["QTpeakc_Bazett_ms"] = np.nan
        feats["QTpeakc_Fridericia_ms"] = np.nan
        feats["QTc_Bazett_ms_est"] = np.nan
        feats["QTc_Fridericia_ms_est"] = np.nan

    return feats, qc


# =====================================================================
# Input handling
# =====================================================================

def load_h5(path: Path):
    with h5py.File(path, "r") as h5:
        if "median_all_mv" not in h5:
            raise KeyError("HDF5 has no median_all_mv dataset.")

        x = np.asarray(h5["median_all_mv"], dtype=np.float32)
        if x.ndim != 3 or x.shape[1] != 12:
            raise ValueError(f"Unexpected median_all_mv shape: {x.shape}")

        split = np.asarray(h5["split_code"], dtype=np.int8)

        metadata = {}
        for k, ds in h5.items():
            if k in {"median_all_mv", "median_all_scaled"}:
                continue

            arr = np.asarray(ds)
            if arr.ndim == 1 and len(arr) == len(x):
                metadata[k] = decode_array(arr)

        attrs = {k: h5.attrs[k] for k in h5.attrs.keys()}

    return x, split, metadata, attrs


def find_rr_ms(metadata, n):
    candidates = [
        "median_rr_ms",
        "rr_ms",
        "RR_ms",
        "median_RR_ms",
    ]

    for c in candidates:
        if c in metadata:
            vals = pd.to_numeric(
                pd.Series(metadata[c]),
                errors="coerce",
            ).to_numpy(dtype=float)
            if len(vals) == n and np.sum(np.isfinite(vals)) > 0:
                return vals, c

    # HR fallback.
    for c in ["heart_rate_bpm", "HR_bpm", "median_heart_rate_bpm"]:
        if c in metadata:
            hr = pd.to_numeric(
                pd.Series(metadata[c]),
                errors="coerce",
            ).to_numpy(dtype=float)
            rr = np.where(
                np.isfinite(hr) & (hr > 0),
                60000.0 / hr,
                np.nan,
            )
            if np.sum(np.isfinite(rr)) > 0:
                return rr, f"derived from {c}"

    return np.full(n, np.nan), None


def load_latents(path: Path, n_records: int):
    df = pd.read_csv(path)

    if len(df) != n_records:
        raise RuntimeError(
            f"Latent CSV has {len(df)} rows but HDF5 has {n_records}."
        )

    zcols = sorted(
        [
            c for c in df.columns
            if c.startswith("z_") and c[2:].isdigit()
        ],
        key=lambda x: int(x.split("_")[1]),
    )

    if not zcols:
        raise RuntimeError("No z_XX latent columns found.")

    return df, zcols


def load_traversals(path: Path):
    z = np.load(path, allow_pickle=True)

    required = [
        "decoded_mv",
        "levels_sd",
        "lead_names",
        "sampling_rate_hz",
    ]
    for k in required:
        if k not in z:
            raise KeyError(f"Traversal NPZ missing '{k}'.")

    decoded_mv = np.asarray(z["decoded_mv"], dtype=np.float32)
    levels_sd = np.asarray(z["levels_sd"], dtype=float)
    lead_names = [str(x) for x in z["lead_names"]]
    fs = float(np.ravel(z["sampling_rate_hz"])[0])

    if decoded_mv.ndim != 4:
        raise ValueError(f"Unexpected decoded_mv shape: {decoded_mv.shape}")

    if decoded_mv.shape[2] != 12:
        raise ValueError(f"Traversal signals do not have 12 leads: {decoded_mv.shape}")

    return decoded_mv, levels_sd, lead_names, fs


# =====================================================================
# Real ECG feature table
# =====================================================================

def build_real_feature_table(
    x_mv,
    split,
    metadata,
    latent_df,
    zcols,
    fs,
    beat_start_ms,
):
    n = len(x_mv)
    rr, rr_source = find_rr_ms(metadata, n)

    rows = []
    qc_rows = []

    metadata_copy_keys = [
        "ecg_id",
        "patient_id",
        "subject_key",
        "age",
        "sex",
        "gender",
        "source_group",
        "phase1_row_index",
    ]

    for i in range(n):
        try:
            feats, qc = extract_ecg_features(
                x_mv[i],
                fs=fs,
                beat_start_ms=beat_start_ms,
                rr_ms=rr[i],
            )
            success = True
            error = ""
        except Exception as exc:
            feats = {}
            qc = {
                "qrs_boundary_ok": False,
                "qrs_boundary_method": "exception",
                "qrs_on_ms": np.nan,
                "qrs_off_ms": np.nan,
                "qrs_duration_ms": np.nan,
                "tend_ok": False,
            }
            success = False
            error = repr(exc)

        row = {
            "row_index": i,
            "split_code": int(split[i]),
            "split_name": SPLIT_NAMES.get(int(split[i]), f"split_{split[i]}"),
            "feature_extraction_success": success,
        }

        for key in metadata_copy_keys:
            if key in metadata:
                row[key] = metadata[key][i]

        for zc in zcols:
            row[zc] = latent_df.loc[i, zc]

        row.update(feats)
        rows.append(row)

        qc_rows.append({
            "row_index": i,
            "split_code": int(split[i]),
            "split_name": SPLIT_NAMES.get(int(split[i]), f"split_{split[i]}"),
            "success": success,
            "error": error,
            **qc,
        })

        if i == 0 or (i + 1) % 250 == 0 or i == n - 1:
            print(f"[real features] {i+1}/{n}")

    return pd.DataFrame(rows), pd.DataFrame(qc_rows), rr_source


# =====================================================================
# Association analysis
# =====================================================================

NON_FEATURE_COLUMNS = {
    "row_index",
    "split_code",
    "split_name",
    "feature_extraction_success",
    "ecg_id",
    "patient_id",
    "subject_key",
    "age",
    "sex",
    "gender",
    "source_group",
    "phase1_row_index",
}


def select_feature_columns(df, zcols):
    candidates = []

    for c in df.columns:
        if c in NON_FEATURE_COLUMNS:
            continue
        if c in zcols:
            continue
        if c.startswith("logvar_"):
            continue

        vals = pd.to_numeric(df[c], errors="coerce")
        nfinite = int(np.sum(np.isfinite(vals)))

        if nfinite < 30:
            continue

        if np.nanstd(vals) <= 1e-10:
            continue

        # Explicitly exclude circular raw axis in degrees from correlation
        # ranking; sin/cos carry the circular information robustly.
        if c == "QRS_axis_deg":
            continue

        candidates.append(c)

    return candidates


def association_table(
    df,
    zcols,
    feature_cols,
    group_name,
):
    rows = []

    for zc in zcols:
        z = pd.to_numeric(df[zc], errors="coerce").to_numpy(dtype=float)

        for feat in feature_cols:
            y = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)

            sr, sp, n = robust_corr(z, y, "spearman")
            pr, pp, _ = robust_corr(z, y, "pearson")
            raw_slope, std_slope = standardized_slope(z, y)

            rows.append({
                "group": group_name,
                "latent": zc,
                "feature": feat,
                "n": n,
                "spearman_rho": sr,
                "spearman_p": sp,
                "pearson_r": pr,
                "pearson_p": pp,
                "raw_slope_feature_per_latent_unit": raw_slope,
                "standardized_slope": std_slope,
                "abs_spearman_rho": abs(sr) if np.isfinite(sr) else np.nan,
            })

    out = pd.DataFrame(rows)
    out["spearman_q_fdr"] = bh_fdr(out["spearman_p"].to_numpy(dtype=float))
    out["pearson_q_fdr"] = bh_fdr(out["pearson_p"].to_numpy(dtype=float))
    return out


# =====================================================================
# Traversal feature extraction / slopes
# =====================================================================

def build_traversal_feature_table(
    decoded_mv,
    levels_sd,
    fs,
    beat_start_ms,
):
    rows = []
    qcs = []

    latent_dim = decoded_mv.shape[0]

    for k in range(latent_dim):
        for j, level in enumerate(levels_sd):
            feats, qc = extract_ecg_features(
                decoded_mv[k, j],
                fs=fs,
                beat_start_ms=beat_start_ms,
                rr_ms=np.nan,
            )

            row = {
                "latent": f"z_{k:02d}",
                "latent_index": k,
                "level_sd": float(level),
            }
            row.update(feats)
            rows.append(row)

            qcs.append({
                "latent": f"z_{k:02d}",
                "latent_index": k,
                "level_sd": float(level),
                **qc,
            })

    return pd.DataFrame(rows), pd.DataFrame(qcs)


def traversal_slope_table(trav_df, feature_cols):
    rows = []

    for latent, g in trav_df.groupby("latent", sort=True):
        levels = g["level_sd"].to_numpy(dtype=float)

        for feat in feature_cols:
            if feat not in g.columns:
                continue

            y = pd.to_numeric(g[feat], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(levels) & np.isfinite(y)

            if np.sum(m) < 4 or np.std(y[m]) <= 1e-12:
                continue

            lr = stats.linregress(levels[m], y[m])
            sr, sp = stats.spearmanr(levels[m], y[m])

            # Prefer explicit -3 / +3 values when available.
            y_minus = np.nan
            y_plus = np.nan

            mm = np.where(np.isclose(levels, -3.0) & np.isfinite(y))[0]
            pp = np.where(np.isclose(levels, +3.0) & np.isfinite(y))[0]

            if len(mm):
                y_minus = float(y[mm[0]])
            if len(pp):
                y_plus = float(y[pp[0]])

            delta = (
                y_plus - y_minus
                if np.isfinite(y_plus) and np.isfinite(y_minus)
                else np.nan
            )

            rows.append({
                "latent": latent,
                "feature": feat,
                "n_levels": int(np.sum(m)),
                "feature_at_minus3sd": y_minus,
                "feature_at_plus3sd": y_plus,
                "delta_minus3_to_plus3": delta,
                "slope_per_sd": float(lr.slope),
                "intercept": float(lr.intercept),
                "r_squared": float(lr.rvalue ** 2),
                "linear_p": float(lr.pvalue),
                "spearman_level_rho": float(sr),
                "spearman_level_p": float(sp),
                "abs_spearman_level_rho": abs(float(sr)),
            })

    out = pd.DataFrame(rows)
    if len(out):
        out["linear_q_fdr"] = bh_fdr(out["linear_p"].to_numpy(dtype=float))
        out["spearman_level_q_fdr"] = bh_fdr(
            out["spearman_level_p"].to_numpy(dtype=float)
        )
    return out


def concordance_table(real_assoc, trav_slopes):
    cols = [
        "latent",
        "feature",
        "spearman_rho",
        "spearman_p",
        "spearman_q_fdr",
        "abs_spearman_rho",
    ]

    a = real_assoc[cols].copy()
    b = trav_slopes[
        [
            "latent",
            "feature",
            "delta_minus3_to_plus3",
            "slope_per_sd",
            "r_squared",
            "spearman_level_rho",
            "spearman_level_q_fdr",
        ]
    ].copy()

    m = a.merge(b, on=["latent", "feature"], how="inner")

    m["sign_concordant"] = (
        np.sign(m["spearman_rho"])
        == np.sign(m["slope_per_sd"])
    )

    m["strong_internal_validation"] = (
        (m["abs_spearman_rho"] >= 0.20)
        & (m["r_squared"] >= 0.80)
        & m["sign_concordant"]
    )

    m["priority_score"] = (
        m["abs_spearman_rho"]
        * np.sqrt(np.clip(m["r_squared"], 0, 1))
    )

    return m.sort_values(
        ["latent", "priority_score"],
        ascending=[True, False],
    )


# =====================================================================
# Compact interpretation summary
# =====================================================================

def priority_summary(
    hcm_train_assoc,
    traversal_slopes,
    concordance,
    zcols,
    top_n,
):
    rows = []

    for zc in zcols:
        a = (
            hcm_train_assoc[hcm_train_assoc["latent"] == zc]
            .sort_values("abs_spearman_rho", ascending=False)
        )

        # Prefer features that are both empirically associated and traversal-consistent.
        c = concordance[
            (concordance["latent"] == zc)
            & np.isfinite(concordance["spearman_rho"])
        ].sort_values(
            ["strong_internal_validation", "priority_score"],
            ascending=[False, False],
        )

        tr = (
            traversal_slopes[traversal_slopes["latent"] == zc]
            .assign(
                traversal_rank_score=lambda d:
                    d["abs_spearman_level_rho"]
                    * np.sqrt(np.clip(d["r_squared"], 0, 1))
            )
            .sort_values("traversal_rank_score", ascending=False)
        )

        row = {"latent": zc}

        for rank in range(top_n):
            if rank < len(a):
                rr = a.iloc[rank]
                row[f"real_feature_{rank+1}"] = rr["feature"]
                row[f"real_spearman_{rank+1}"] = rr["spearman_rho"]
                row[f"real_q_{rank+1}"] = rr["spearman_q_fdr"]

            if rank < len(c):
                rr = c.iloc[rank]
                row[f"concordant_feature_{rank+1}"] = rr["feature"]
                row[f"concordant_real_rho_{rank+1}"] = rr["spearman_rho"]
                row[f"concordant_traversal_slope_{rank+1}"] = rr["slope_per_sd"]
                row[f"concordant_traversal_r2_{rank+1}"] = rr["r_squared"]
                row[f"concordant_sign_{rank+1}"] = rr["sign_concordant"]

            if rank < len(tr):
                rr = tr.iloc[rank]
                row[f"traversal_feature_{rank+1}"] = rr["feature"]
                row[f"traversal_delta_{rank+1}"] = rr["delta_minus3_to_plus3"]
                row[f"traversal_slope_per_sd_{rank+1}"] = rr["slope_per_sd"]
                row[f"traversal_r2_{rank+1}"] = rr["r_squared"]

        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# Figures
# =====================================================================

def plot_latent_feature_heatmap(
    assoc_df,
    zcols,
    out_path,
    max_features=30,
):
    # Rank globally by maximum absolute association across latents.
    piv = assoc_df.pivot_table(
        index="feature",
        columns="latent",
        values="spearman_rho",
        aggfunc="first",
    )

    for zc in zcols:
        if zc not in piv.columns:
            piv[zc] = np.nan

    piv = piv[zcols]
    rank = piv.abs().max(axis=1).sort_values(ascending=False)
    feats = rank.head(max_features).index.tolist()
    mat = piv.loc[feats, zcols].to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(9, max(7, 0.27 * len(feats)))
    )
    im = ax.imshow(
        mat,
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )
    ax.set_xticks(np.arange(len(zcols)))
    ax.set_xticklabels(zcols)
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels(feats, fontsize=7)
    ax.set_title("HCM-train latent-feature Spearman associations")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_concordance(concordance, out_path):
    d = concordance[
        np.isfinite(concordance["spearman_rho"])
        & np.isfinite(concordance["spearman_level_rho"])
    ].copy()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        d["spearman_rho"],
        d["spearman_level_rho"],
        s=14,
        alpha=0.55,
    )
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Real HCM-train: latent-feature Spearman rho")
    ax.set_ylabel("Traversal: SD-level-feature Spearman rho")
    ax.set_title("Real-data vs decoder-traversal direction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    print("=" * 78)
    print("14 - Quantify HCM-static latent ECG features")
    print("=" * 78)
    print("HDF5     :", args.h5)
    print("Latents  :", args.latents)
    print("Traversal:", args.traversals)
    print("Output   :", args.output_dir)

    x_mv, split, metadata, attrs = load_h5(args.h5)
    latent_df, zcols = load_latents(args.latents, len(x_mv))

    print(f"[data] shape      : {x_mv.shape}")
    print(f"[data] latent dim : {len(zcols)}")
    print(f"[data] HCM train  : {np.sum(split == 0)}")
    print(f"[data] HCM val    : {np.sum(split == 1)}")
    print(f"[data] HCM test   : {np.sum(split == 2)}")
    print(f"[data] Normal eval: {np.sum(split == 3)}")

    # -------------------------------------------------------------
    # Real HCM ECG features
    # -------------------------------------------------------------
    feature_df, qc_df, rr_source = build_real_feature_table(
        x_mv=x_mv,
        split=split,
        metadata=metadata,
        latent_df=latent_df,
        zcols=zcols,
        fs=args.fs,
        beat_start_ms=args.beat_start_ms,
    )

    feature_path = args.output_dir / "ecg_classical_features.csv"
    feature_df.to_csv(feature_path, index=False)
    qc_df.to_csv(
        args.output_dir / "feature_extraction_qc.csv",
        index=False,
    )

    feature_cols = select_feature_columns(feature_df, zcols)

    # HCM only groups.
    groups = {
        "hcm_train": feature_df["split_code"] == 0,
        "hcm_val": feature_df["split_code"] == 1,
        "hcm_test": feature_df["split_code"] == 2,
        "hcm_all": feature_df["split_code"].isin([0, 1, 2]),
    }

    assoc_tables = {}

    for name, mask in groups.items():
        sub = feature_df.loc[mask].copy()
        assoc = association_table(
            sub,
            zcols=zcols,
            feature_cols=feature_cols,
            group_name=name,
        )
        assoc_tables[name] = assoc
        assoc.to_csv(
            args.output_dir / f"latent_feature_associations_{name}.csv",
            index=False,
        )

    # -------------------------------------------------------------
    # Traversal features
    # -------------------------------------------------------------
    decoded_mv, levels_sd, traversal_leads, traversal_fs = load_traversals(
        args.traversals
    )

    if decoded_mv.shape[0] != len(zcols):
        raise RuntimeError(
            f"Traversal latent dim {decoded_mv.shape[0]} != CSV latent dim {len(zcols)}."
        )

    if [x.lower() for x in traversal_leads] != [x.lower() for x in LEAD_NAMES]:
        warnings.warn(
            f"Traversal lead names differ from expected order: {traversal_leads}"
        )

    if abs(traversal_fs - args.fs) > 1e-6:
        warnings.warn(
            f"Traversal fs={traversal_fs}, command fs={args.fs}; using traversal fs."
        )

    trav_df, trav_qc = build_traversal_feature_table(
        decoded_mv=decoded_mv,
        levels_sd=levels_sd,
        fs=traversal_fs,
        beat_start_ms=args.beat_start_ms,
    )

    trav_df.to_csv(
        args.output_dir / "latent_traversal_features.csv",
        index=False,
    )
    trav_qc.to_csv(
        args.output_dir / "latent_traversal_feature_qc.csv",
        index=False,
    )

    # Restrict slopes to the same morphology features used for real associations.
    traversal_feature_cols = [
        c for c in feature_cols
        if c in trav_df.columns
        and c not in {
            "RR_ms",
            "HR_bpm",
            "QTpeakc_Bazett_ms",
            "QTpeakc_Fridericia_ms",
            "QTc_Bazett_ms_est",
            "QTc_Fridericia_ms_est",
        }
    ]

    trav_slopes = traversal_slope_table(
        trav_df,
        traversal_feature_cols,
    )
    trav_slopes.to_csv(
        args.output_dir / "latent_traversal_feature_slopes.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Real-vs-traversal internal validation
    # -------------------------------------------------------------
    concordance = concordance_table(
        assoc_tables["hcm_train"],
        trav_slopes,
    )
    concordance.to_csv(
        args.output_dir / "latent_real_vs_traversal_concordance.csv",
        index=False,
    )

    priority = priority_summary(
        hcm_train_assoc=assoc_tables["hcm_train"],
        traversal_slopes=trav_slopes,
        concordance=concordance,
        zcols=zcols,
        top_n=args.top_n,
    )
    priority.to_csv(
        args.output_dir / "latent_interpretation_priority_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Internal-validation headline
    # -------------------------------------------------------------
    moderate = concordance[
        (concordance["abs_spearman_rho"] >= 0.20)
        & (concordance["r_squared"] >= 0.80)
    ].copy()

    n_internal = int(len(moderate))
    n_sign = int(np.sum(moderate["sign_concordant"])) if n_internal else 0
    sign_fraction = (
        float(n_sign / n_internal)
        if n_internal else np.nan
    )

    # QC
    real_success = float(qc_df["success"].mean())
    qrs_qc = float(qc_df["qrs_boundary_ok"].mean())
    tend_qc = float(qc_df["tend_ok"].mean())

    trav_qrs_qc = float(trav_qc["qrs_boundary_ok"].mean())
    trav_tend_qc = float(trav_qc["tend_ok"].mean())

    # -------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------
    plot_latent_feature_heatmap(
        assoc_tables["hcm_train"],
        zcols,
        fig_dir / "latent_feature_heatmap_hcm_train.png",
    )

    plot_concordance(
        concordance,
        fig_dir / "latent_traversal_concordance.png",
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    summary = {
        "script": "14_quantify_hcm_static_latent_ecg_features.py",
        "version": "1.0",
        "purpose": (
            "Quantitative physiological interpretation of HCM-static "
            "beta-VAE latent coordinates using real HCM morphology and "
            "decoder traversals"
        ),
        "input_h5": str(args.h5),
        "input_latents": str(args.latents),
        "input_traversals": str(args.traversals),
        "n_records_total": int(len(feature_df)),
        "groups": {
            "hcm_train": int(np.sum(split == 0)),
            "hcm_val": int(np.sum(split == 1)),
            "hcm_test": int(np.sum(split == 2)),
            "hcm_all": int(np.sum(np.isin(split, [0,1,2]))),
            "normal_eval": int(np.sum(split == 3)),
        },
        "latent_dim": int(len(zcols)),
        "latent_columns": zcols,
        "n_ecg_features_used_for_association": int(len(feature_cols)),
        "feature_extraction_success_fraction": real_success,
        "qrs_boundary_qc_fraction": qrs_qc,
        "heuristic_tend_qc_fraction": tend_qc,
        "rr_source": rr_source,
        "traversal_qrs_boundary_qc_fraction": trav_qrs_qc,
        "traversal_heuristic_tend_qc_fraction": trav_tend_qc,
        "traversal_levels_sd": levels_sd.tolist(),
        "traversal_analysis_completed": True,
        "internal_validation_filter": {
            "minimum_abs_real_spearman": 0.20,
            "minimum_traversal_r_squared": 0.80,
        },
        "internal_validation_pairs": n_internal,
        "internal_validation_sign_concordant_pairs": n_sign,
        "internal_validation_sign_concordance_fraction": sign_fraction,
        "methodological_notes": [
            "HCM-train associations are the primary Model-2 physiological interpretation analysis.",
            "HCM validation/test/all associations are secondary confirmation analyses.",
            "Normal-eval ECGs are not used for primary latent interpretation.",
            "All amplitudes are measured on median beats in physical mV.",
            "No per-ECG or per-lead amplitude normalization is applied.",
            "One multilead QRS onset/offset is used for all leads.",
            "QRS frontal axis is estimated from net QRS areas in leads I and aVF.",
            "Raw R/S ratio is excluded; bounded RS_balance is used to avoid numerical instability.",
            "Sokolow-Lyon and Cornell-like voltages are continuous morphology descriptors, not diagnostic labels.",
            "QT-end is a heuristic research surrogate and not a clinical QT measurement.",
            "Traversal slopes vary one HCM latent coordinate while all others remain fixed at the HCM-train centroid.",
            "Real-vs-traversal concordance is an internal consistency check, not independent external validation.",
        ],
    }

    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("STEP 14 COMPLETE")
    print("=" * 78)
    print("Features used              :", len(feature_cols))
    print("Real feature success       :", f"{real_success:.4f}")
    print("Real QRS-boundary QC       :", f"{qrs_qc:.4f}")
    print("Real heuristic Tend QC     :", f"{tend_qc:.4f}")
    print("Traversal QRS-boundary QC  :", f"{trav_qrs_qc:.4f}")
    print("Internal validation pairs  :", n_internal)
    if n_internal:
        print(
            "Sign-concordant pairs      :",
            f"{n_sign}/{n_internal} ({100*sign_fraction:.1f}%)",
        )

    print()
    print("[Top HCM-train associations per latent]")
    for zc in zcols:
        a = (
            assoc_tables["hcm_train"]
            [assoc_tables["hcm_train"]["latent"] == zc]
            .sort_values("abs_spearman_rho", ascending=False)
            .head(5)
        )
        print()
        print(zc)
        for _, r in a.iterrows():
            print(
                f"  {r['feature']:<36s} "
                f"rho={r['spearman_rho']:+.3f} "
                f"q={r['spearman_q_fdr']:.3g}"
            )

    print()
    print("Results:", args.output_dir)


if __name__ == "__main__":
    main()
