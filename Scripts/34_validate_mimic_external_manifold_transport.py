#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
34_validate_mimic_external_manifold_transport.py

SMASH-HCM -- Step 34
External MIMIC-IV baseline manifold transport / coverage analysis.

Scientific role
---------------
This step evaluates whether the two FROZEN SMASH-HCM latent representations
transport to an independent MIMIC-IV HCM cohort without any retraining,
fine-tuning, rescaling, or refitting on MIMIC.

The analysis is deliberately baseline-to-baseline to avoid pseudoreplication:
    - MIMIC: first valid projected ECG per subject from Step 33
    - SHaRe: first valid projected ECG per coherent subject from Step 16

Primary external-transport questions
------------------------------------
1. HCM-manifold coverage:
   What fraction of MIMIC baseline ECGs lies within empirical 95%, 97.5%, and
   99% Mahalanobis-distance contours defined ONLY by HCM-train posterior means?

2. Normal-manifold displacement:
   What fraction of MIMIC HCM baseline ECGs lies outside corresponding empirical
   Normal-train contours?

3. Frozen HCM latent-coordinate transport:
   Compare MIMIC vs internal SHaRe HCM baseline distributions for H_z_00..H_z_07
   using effect sizes, bootstrap CIs, Wasserstein distance, and secondary KS tests
   with BH-FDR correction across the eight coordinates.

4. Prespecified coordinate H_z_06:
   Report its baseline distribution explicitly as a priority external phenotype.

5. Cross-manifold reconstruction behavior:
   Compare d_N, d_HCM, RE_N, RE_H, Delta_RE and log(RE_N/RE_H) descriptively.
   Reconstruction error is NOT interpreted as a diagnostic classifier.

IMPORTANT
---------
- MIMIC is never used to fit the VAE models.
- MIMIC is never used to estimate scaling parameters.
- MIMIC is never used to estimate latent reference covariance.
- Empirical coverage thresholds are derived from original Normal-train and
  HCM-train latent distributions only.
- The primary sampling unit is the SUBJECT, one baseline ECG per subject.
- Statistical significance is secondary to effect size and coverage.
- Similarity is not claimed from a non-significant two-sample test.

Canonical inputs
----------------
Results/33_MIMIC_FrozenDualManifoldProjection/mimic_frozen_dual_manifold_valid.csv
Results/16_Longitudinal_DualManifold/longitudinal_dual_manifold.csv
Results/06_Phase1_MedianBeat_Beta_VAE/phase1_medianbeat_latents.csv
Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv

Outputs
-------
Results/34_MIMIC_ExternalManifoldTransport/
    mimic_baseline_external.csv
    share_baseline_internal.csv
    manifold_coverage.csv
    hcm_latent_transport.csv
    cross_manifold_metric_transport.csv
    h_z06_external_summary.csv
    hcm_latent_covariance_transport.json
    run_summary.json
    external_transport_report.txt
    figures/
        hcm_latent_baseline_distributions.png
        manifold_distance_ecdf.png
        h_z06_baseline_distribution.png
        hcm_latent_correlation_difference.png

Run
---
python Scripts/34_validate_mimic_external_manifold_transport.py

Dependencies
------------
numpy pandas scipy scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.spatial.distance import cdist, pdist
from scipy.stats import chi2, wasserstein_distance
from sklearn.covariance import LedoitWolf


STEP = 34
SEED = 20261004
BOOTSTRAP_REPS = 2000
NORMAL_LATENT_DIM = 16
HCM_LATENT_DIM = 8
EPS = 1e-12

N_COLS = [f"N_z_{i:02d}" for i in range(NORMAL_LATENT_DIM)]
H_COLS = [f"H_z_{i:02d}" for i in range(HCM_LATENT_DIM)]
TRAIN_N_COLS = [f"z_{i:02d}" for i in range(NORMAL_LATENT_DIM)]
TRAIN_H_COLS = [f"z_{i:02d}" for i in range(HCM_LATENT_DIM)]

TRANSPORT_METRICS = [
    "d_N",
    "d_HCM",
    "RE_N_mse",
    "RE_H_mse",
    "Delta_RE_mse",
    "log_RE_N_over_RE_H",
]


# =============================================================================
# Paths / arguments
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


def parse_args() -> argparse.Namespace:
    root = infer_project_root()

    mimic_default = first_existing(
        [
            root / "Results" / "33_MIMIC_FrozenDualManifoldProjection" / "mimic_frozen_dual_manifold_valid.csv",
        ],
        "Step-33 MIMIC valid projection CSV",
    )

    share_default = first_existing(
        [
            root / "Results" / "16_Longitudinal_DualManifold" / "longitudinal_dual_manifold.csv",
            root / "Results" / "Phase3_Longitudinal_DualManifold" / "longitudinal_dual_manifold.csv",
        ],
        "Step-16 SHaRe longitudinal dual-manifold CSV",
    )

    normal_latents_default = first_existing(
        [
            root / "Results" / "06_Phase1_MedianBeat_Beta_VAE" / "phase1_medianbeat_latents.csv",
            root / "Results" / "Phase1_MedianBeat_VAE" / "phase1_medianbeat_latents.csv",
        ],
        "Step-06 Normal-anchored latent CSV",
    )

    hcm_latents_default = first_existing(
        [
            root / "Results" / "12_HCM_Static_Beta_VAE" / "hcm_static_latents.csv",
            root / "Results" / "Phase2_HCMStatic_VAE" / "hcm_static_latents.csv",
        ],
        "Step-12 HCM-static latent CSV",
    )

    p = argparse.ArgumentParser(
        description="Validate frozen SMASH-HCM manifold transport in MIMIC-IV HCM."
    )
    p.add_argument("--mimic", type=Path, default=mimic_default)
    p.add_argument("--share", type=Path, default=share_default)
    p.add_argument("--normal-latents", type=Path, default=normal_latents_default)
    p.add_argument("--hcm-latents", type=Path, default=hcm_latents_default)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=root / "Results" / "34_MIMIC_ExternalManifoldTransport",
    )
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def numeric_finite(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def describe_array(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q025": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q975": np.nan,
        }
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "median": float(np.median(x)),
        "q025": float(np.quantile(x, 0.025)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q975": float(np.quantile(x, 0.975)),
    }


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    pf = p[finite]
    if len(pf) == 0:
        return out
    order = np.argsort(pf)
    ranked = pf[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out[finite] = q[inv]
    return out


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    y = np.arange(1, len(x) + 1, dtype=float) / max(len(x), 1)
    return x, y


def fit_reference(z: np.ndarray) -> LedoitWolf:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z).all(axis=1)]
    if len(z) <= z.shape[1] + 2:
        raise RuntimeError("Too few finite rows to fit latent reference covariance.")
    return LedoitWolf().fit(z)


def mahalanobis(z: np.ndarray, model: LedoitWolf) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    d = z - model.location_[None, :]
    d2 = np.einsum("ni,ij,nj->n", d, model.precision_, d)
    return np.sqrt(np.maximum(d2, 0.0))


def bootstrap_standardized_shift(
    internal: np.ndarray,
    external: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """(mean_external - mean_internal) / SD_internal, with percentile bootstrap CI."""
    a = np.asarray(internal, dtype=float)
    b = np.asarray(external, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan

    sd_a = np.std(a, ddof=1)
    point = (np.mean(b) - np.mean(a)) / sd_a if sd_a > EPS else np.nan
    if reps <= 0:
        return float(point), np.nan, np.nan

    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        aa = a[rng.integers(0, len(a), size=len(a))]
        bb = b[rng.integers(0, len(b), size=len(b))]
        sd = np.std(aa, ddof=1)
        vals[i] = (np.mean(bb) - np.mean(aa)) / sd if sd > EPS else np.nan
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float(point), np.nan, np.nan
    return float(point), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def multivariate_energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Multivariate energy statistic (descriptive; no permutation p-value)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    y = y[np.isfinite(y).all(axis=1)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    cross = float(np.mean(cdist(x, y, metric="euclidean")))
    # E||X-X'|| and E||Y-Y'|| include the zero diagonal and both pair orders.
    px = pdist(x, metric="euclidean")
    py = pdist(y, metric="euclidean")
    within_x = float(2.0 * np.sum(px) / (len(x) ** 2))
    within_y = float(2.0 * np.sum(py) / (len(y) ** 2))
    val = 2.0 * cross - within_x - within_y
    return float(max(val, 0.0))


# =============================================================================
# Baseline extraction
# =============================================================================

def extract_mimic_baseline(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["subject_id", *N_COLS, *H_COLS], "MIMIC Step-33 table")
    d = df.copy()
    d["subject_id"] = d["subject_id"].astype(str)

    if "projected_frozen_dual_manifold" in d.columns:
        mask = d["projected_frozen_dual_manifold"].astype(bool)
        d = d.loc[mask].copy()

    if "acquisition_datetime" in d.columns:
        d["_dt"] = pd.to_datetime(d["acquisition_datetime"], errors="coerce")
    else:
        d["_dt"] = pd.NaT

    if "projection_row_index" not in d.columns:
        d["projection_row_index"] = np.arange(len(d), dtype=int)

    # Prefer the pre-existing valid visit order when available; otherwise chronological.
    if "valid_visit_number" in d.columns:
        v = pd.to_numeric(d["valid_visit_number"], errors="coerce")
        base = d.loc[v.eq(1)].copy()
        # If a malformed file misses some subjects, fill from chronological first valid.
        missing_subjects = sorted(set(d["subject_id"]) - set(base["subject_id"]))
        if missing_subjects:
            fill = (
                d[d["subject_id"].isin(missing_subjects)]
                .sort_values(["subject_id", "_dt", "projection_row_index"], na_position="last", kind="mergesort")
                .groupby("subject_id", as_index=False, sort=False)
                .head(1)
            )
            base = pd.concat([base, fill], ignore_index=True)
    else:
        base = (
            d.sort_values(["subject_id", "_dt", "projection_row_index"], na_position="last", kind="mergesort")
            .groupby("subject_id", as_index=False, sort=False)
            .head(1)
            .copy()
        )

    base = base.sort_values("subject_id", kind="mergesort").reset_index(drop=True)
    if base["subject_id"].duplicated().any():
        raise RuntimeError("MIMIC baseline extraction produced duplicate subjects.")
    return base.drop(columns=["_dt"], errors="ignore")


def extract_share_baseline(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        ["analysis_subject_key", "valid_for_latent_projection", *N_COLS, *H_COLS],
        "SHaRe Step-16 table",
    )
    d = df.loc[df["valid_for_latent_projection"].astype(bool)].copy()
    d["analysis_subject_key"] = d["analysis_subject_key"].astype(str)

    if "is_first_valid_visit" in d.columns:
        mask = d["is_first_valid_visit"].astype(bool)
        base = d.loc[mask].copy()
    elif "visit_order_valid" in d.columns:
        base = d.loc[pd.to_numeric(d["visit_order_valid"], errors="coerce").eq(1)].copy()
    else:
        if "ecg_datetime" in d.columns:
            d["_dt"] = pd.to_datetime(d["ecg_datetime"], errors="coerce")
        elif "ecg_time" in d.columns:
            d["_dt"] = pd.to_datetime(d["ecg_time"], errors="coerce")
        else:
            d["_dt"] = pd.NaT
        row_col = "longitudinal_row_index" if "longitudinal_row_index" in d.columns else d.index.name
        if row_col is None:
            d["_row"] = np.arange(len(d), dtype=int)
            row_col = "_row"
        base = (
            d.sort_values(["analysis_subject_key", "_dt", row_col], na_position="last", kind="mergesort")
            .groupby("analysis_subject_key", as_index=False, sort=False)
            .head(1)
            .copy()
        )

    base = base.sort_values("analysis_subject_key", kind="mergesort").reset_index(drop=True)
    if base["analysis_subject_key"].duplicated().any():
        raise RuntimeError("SHaRe baseline extraction produced duplicate subjects.")
    return base.drop(columns=["_dt", "_row"], errors="ignore")


# =============================================================================
# Analyses
# =============================================================================

def build_reference_models(
    normal_latents: pd.DataFrame,
    hcm_latents: pd.DataFrame,
) -> Tuple[LedoitWolf, LedoitWolf, np.ndarray, np.ndarray, Dict[str, object]]:
    require_columns(normal_latents, ["split_name", *TRAIN_N_COLS], "Normal latent CSV")
    require_columns(hcm_latents, ["split_name", *TRAIN_H_COLS], "HCM latent CSV")

    nref = normal_latents.loc[
        normal_latents["split_name"].astype(str).eq("normal_train"), TRAIN_N_COLS
    ].to_numpy(dtype=float)
    href = hcm_latents.loc[
        hcm_latents["split_name"].astype(str).eq("hcm_train"), TRAIN_H_COLS
    ].to_numpy(dtype=float)
    nref = nref[np.isfinite(nref).all(axis=1)]
    href = href[np.isfinite(href).all(axis=1)]

    lw_n = fit_reference(nref)
    lw_h = fit_reference(href)
    dn_ref = mahalanobis(nref, lw_n)
    dh_ref = mahalanobis(href, lw_h)

    info = {
        "normal_train_n": int(len(nref)),
        "hcm_train_n": int(len(href)),
        "normal_train_distance_quantiles": {
            str(q): float(np.quantile(dn_ref, q)) for q in [0.95, 0.975, 0.99]
        },
        "hcm_train_distance_quantiles": {
            str(q): float(np.quantile(dh_ref, q)) for q in [0.95, 0.975, 0.99]
        },
    }
    return lw_n, lw_h, dn_ref, dh_ref, info


def coverage_table(
    mimic: pd.DataFrame,
    share: pd.DataFrame,
    dn_ref: np.ndarray,
    dh_ref: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    specs = [
        ("Normal", "d_N", NORMAL_LATENT_DIM, dn_ref),
        ("HCM", "d_HCM", HCM_LATENT_DIM, dh_ref),
    ]

    for manifold, col, dim, ref in specs:
        require_columns(mimic, [col], "MIMIC baseline")
        require_columns(share, [col], "SHaRe baseline")
        for q in [0.95, 0.975, 0.99]:
            empirical_thr = float(np.quantile(ref, q))
            theoretical_thr = float(math.sqrt(chi2.ppf(q, df=dim)))
            for cohort_name, cohort in [("MIMIC", mimic), ("SHaRe", share)]:
                x = numeric_finite(cohort[col])
                rows.append({
                    "manifold": manifold,
                    "distance_variable": col,
                    "latent_dim": dim,
                    "reference_quantile": q,
                    "threshold_type": "empirical_training",
                    "threshold": empirical_thr,
                    "cohort": cohort_name,
                    "n": int(len(x)),
                    "n_inside": int(np.sum(x <= empirical_thr)),
                    "fraction_inside": float(np.mean(x <= empirical_thr)),
                    "n_outside": int(np.sum(x > empirical_thr)),
                    "fraction_outside": float(np.mean(x > empirical_thr)),
                    "theoretical_chi_threshold": theoretical_thr,
                })
    return pd.DataFrame(rows)


def transport_table(
    mimic: pd.DataFrame,
    share: pd.DataFrame,
    columns: Sequence[str],
    reps: int,
    seed: int,
    fdr: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    pvals: List[float] = []

    for col in columns:
        require_columns(mimic, [col], "MIMIC baseline")
        require_columns(share, [col], "SHaRe baseline")
        ext = numeric_finite(mimic[col])
        inte = numeric_finite(share[col])
        di = describe_array(inte)
        de = describe_array(ext)

        shift, ci_lo, ci_hi = bootstrap_standardized_shift(
            inte, ext, reps=reps, rng=rng
        )
        int_sd = di["sd"]
        wd = float(wasserstein_distance(inte, ext)) if len(inte) and len(ext) else np.nan
        wd_std = wd / int_sd if np.isfinite(int_sd) and int_sd > EPS else np.nan

        if len(inte) and len(ext):
            ks = stats.ks_2samp(inte, ext, alternative="two-sided", method="auto")
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
        else:
            ks_stat = np.nan
            ks_p = np.nan

        pvals.append(ks_p)
        rows.append({
            "variable": col,
            "internal_n": di["n"],
            "external_n": de["n"],
            "internal_mean": di["mean"],
            "external_mean": de["mean"],
            "internal_sd": di["sd"],
            "external_sd": de["sd"],
            "external_to_internal_sd_ratio": de["sd"] / di["sd"] if np.isfinite(di["sd"]) and di["sd"] > EPS else np.nan,
            "internal_median": di["median"],
            "external_median": de["median"],
            "standardized_mean_shift_external_minus_internal": shift,
            "standardized_mean_shift_ci025": ci_lo,
            "standardized_mean_shift_ci975": ci_hi,
            "wasserstein_distance": wd,
            "wasserstein_over_internal_sd": wd_std,
            "ks_statistic": ks_stat,
            "ks_p": ks_p,
        })

    out = pd.DataFrame(rows)
    if fdr:
        out["ks_q_bh"] = bh_fdr(pvals)
    else:
        out["ks_q_bh"] = np.nan
    return out


def h_z06_summary(mimic: pd.DataFrame, share: pd.DataFrame, latent_table: pd.DataFrame) -> pd.DataFrame:
    row = latent_table.loc[latent_table["variable"].eq("H_z_06")]
    if len(row) != 1:
        raise RuntimeError("Could not identify exactly one H_z_06 row in latent transport table.")
    row = row.iloc[0]
    return pd.DataFrame([
        {
            "phenotype": "H_z_06",
            "role": "prespecified priority HCM electrical phenotype",
            "internal_n": int(row["internal_n"]),
            "external_n": int(row["external_n"]),
            "internal_mean": row["internal_mean"],
            "external_mean": row["external_mean"],
            "internal_sd": row["internal_sd"],
            "external_sd": row["external_sd"],
            "internal_median": row["internal_median"],
            "external_median": row["external_median"],
            "standardized_mean_shift_external_minus_internal": row["standardized_mean_shift_external_minus_internal"],
            "shift_ci025": row["standardized_mean_shift_ci025"],
            "shift_ci975": row["standardized_mean_shift_ci975"],
            "wasserstein_over_internal_sd": row["wasserstein_over_internal_sd"],
            "ks_statistic": row["ks_statistic"],
            "ks_p": row["ks_p"],
            "ks_q_bh_across_8_hcm_latents": row["ks_q_bh"],
        }
    ])


def covariance_transport(mimic: pd.DataFrame, share: pd.DataFrame) -> Dict[str, object]:
    x = share[H_COLS].to_numpy(dtype=float)
    y = mimic[H_COLS].to_numpy(dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    y = y[np.isfinite(y).all(axis=1)]

    corr_x = np.corrcoef(x, rowvar=False)
    corr_y = np.corrcoef(y, rowvar=False)
    iu = np.triu_indices(HCM_LATENT_DIM, k=1)
    vx = corr_x[iu]
    vy = corr_y[iu]
    corr_similarity = float(np.corrcoef(vx, vy)[0, 1]) if np.std(vx) > EPS and np.std(vy) > EPS else np.nan
    fro = float(np.linalg.norm(corr_y - corr_x, ord="fro"))

    cov_x = np.cov(x, rowvar=False, ddof=1)
    mean_shift = np.mean(y, axis=0) - np.mean(x, axis=0)
    cov_reg = LedoitWolf().fit(x)
    centroid_d = float(np.sqrt(max(mean_shift @ cov_reg.precision_ @ mean_shift, 0.0)))

    energy = multivariate_energy_distance(x, y)

    return {
        "internal_n": int(len(x)),
        "external_n": int(len(y)),
        "pairwise_correlation_pattern_similarity": corr_similarity,
        "correlation_matrix_frobenius_difference": fro,
        "external_centroid_shift_in_internal_ledoitwolf_metric": centroid_d,
        "multivariate_energy_statistic": energy,
        "internal_correlation_matrix": corr_x.tolist(),
        "external_correlation_matrix": corr_y.tolist(),
        "correlation_difference_external_minus_internal": (corr_y - corr_x).tolist(),
    }


# =============================================================================
# Figures
# =============================================================================

def plot_hcm_latents(mimic: pd.DataFrame, share: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    for j, ax in enumerate(axes.ravel()):
        col = H_COLS[j]
        a = numeric_finite(share[col])
        b = numeric_finite(mimic[col])
        lo = min(np.quantile(a, 0.005), np.quantile(b, 0.005))
        hi = max(np.quantile(a, 0.995), np.quantile(b, 0.995))
        bins = np.linspace(lo, hi, 36)
        ax.hist(a, bins=bins, density=True, histtype="step", linewidth=1.6, label="SHaRe baseline")
        ax.hist(b, bins=bins, density=True, histtype="step", linewidth=1.6, label="MIMIC baseline")
        ax.axvline(np.median(a), linewidth=1.0, linestyle="--", alpha=0.8)
        ax.axvline(np.median(b), linewidth=1.0, linestyle=":", alpha=0.8)
        ax.set_title(col)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.15, linewidth=0.5)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Frozen HCM-manifold latent distributions: internal vs external baseline")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_distances(mimic: pd.DataFrame, share: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, col, title in zip(axes, ["d_N", "d_HCM"], ["Distance from Normal reference", "Distance from HCM reference"]):
        for label, df in [("SHaRe baseline", share), ("MIMIC baseline", mimic)]:
            x, y = ecdf(numeric_finite(df[col]))
            ax.plot(x, y, linewidth=1.8, label=label)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Empirical CDF")
        ax.grid(alpha=0.15, linewidth=0.5)
        ax.legend(frameon=False)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_z06(mimic: pd.DataFrame, share: pd.DataFrame, out: Path, dpi: int) -> None:
    a = numeric_finite(share["H_z_06"])
    b = numeric_finite(mimic["H_z_06"])
    lo = min(np.quantile(a, 0.005), np.quantile(b, 0.005))
    hi = max(np.quantile(a, 0.995), np.quantile(b, 0.995))
    bins = np.linspace(lo, hi, 42)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.hist(a, bins=bins, density=True, histtype="step", linewidth=1.8, label="SHaRe baseline")
    ax.hist(b, bins=bins, density=True, histtype="step", linewidth=1.8, label="MIMIC baseline")
    ax.axvline(np.median(a), linewidth=1.1, linestyle="--", alpha=0.8)
    ax.axvline(np.median(b), linewidth=1.1, linestyle=":", alpha=0.8)
    ax.set_xlabel("H_z_06")
    ax.set_ylabel("Density")
    ax.set_title("External transport of the priority HCM latent coordinate H_z_06")
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.legend(frameon=False)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_corr_difference(cov_info: Dict[str, object], out: Path, dpi: int) -> None:
    diff = np.asarray(cov_info["correlation_difference_external_minus_internal"], dtype=float)
    vmax = max(float(np.nanmax(np.abs(diff))), 0.05)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    im = ax.imshow(diff, vmin=-vmax, vmax=vmax, cmap="coolwarm", aspect="equal")
    ax.set_xticks(range(HCM_LATENT_DIM), labels=H_COLS, rotation=45, ha="right")
    ax.set_yticks(range(HCM_LATENT_DIM), labels=H_COLS)
    ax.set_title("HCM latent correlation difference: MIMIC minus SHaRe")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Delta correlation")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be >= 0")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("34 - MIMIC external manifold transport / baseline coverage")
    print("=" * 88)
    print("MIMIC Step 33 :", args.mimic)
    print("SHaRe Step 16 :", args.share)
    print("Normal latents:", args.normal_latents)
    print("HCM latents   :", args.hcm_latents)
    print("Results       :", args.results_dir)
    print()

    mimic_all = pd.read_csv(args.mimic)
    share_all = pd.read_csv(args.share)
    normal_latents = pd.read_csv(args.normal_latents)
    hcm_latents = pd.read_csv(args.hcm_latents)

    mimic = extract_mimic_baseline(mimic_all)
    share = extract_share_baseline(share_all)

    require_columns(mimic, [*N_COLS, *H_COLS, *TRANSPORT_METRICS], "MIMIC baseline")
    require_columns(share, [*N_COLS, *H_COLS, *TRANSPORT_METRICS], "SHaRe baseline")

    # Preserve baseline audit tables.
    mimic.to_csv(args.results_dir / "mimic_baseline_external.csv", index=False)
    share.to_csv(args.results_dir / "share_baseline_internal.csv", index=False)

    _, _, dn_ref, dh_ref, ref_info = build_reference_models(normal_latents, hcm_latents)

    coverage = coverage_table(mimic, share, dn_ref, dh_ref)
    coverage.to_csv(args.results_dir / "manifold_coverage.csv", index=False)

    latent_transport = transport_table(
        mimic=mimic,
        share=share,
        columns=H_COLS,
        reps=args.bootstrap_reps,
        seed=args.seed,
        fdr=True,
    )
    latent_transport.to_csv(args.results_dir / "hcm_latent_transport.csv", index=False)

    metric_transport = transport_table(
        mimic=mimic,
        share=share,
        columns=TRANSPORT_METRICS,
        reps=args.bootstrap_reps,
        seed=args.seed + 1,
        fdr=False,
    )
    metric_transport.to_csv(args.results_dir / "cross_manifold_metric_transport.csv", index=False)

    z06 = h_z06_summary(mimic, share, latent_transport)
    z06.to_csv(args.results_dir / "h_z06_external_summary.csv", index=False)

    cov_info = covariance_transport(mimic, share)
    (args.results_dir / "hcm_latent_covariance_transport.json").write_text(
        json.dumps(cov_info, indent=2), encoding="utf-8"
    )

    # Figures.
    plot_hcm_latents(mimic, share, fig_dir / "hcm_latent_baseline_distributions.png", args.dpi)
    plot_distances(mimic, share, fig_dir / "manifold_distance_ecdf.png", args.dpi)
    plot_z06(mimic, share, fig_dir / "h_z06_baseline_distribution.png", args.dpi)
    plot_corr_difference(cov_info, fig_dir / "hcm_latent_correlation_difference.png", args.dpi)

    # Compact summary values.
    def cov_pick(manifold: str, q: float, cohort: str) -> Dict[str, float]:
        x = coverage[
            coverage["manifold"].eq(manifold)
            & np.isclose(coverage["reference_quantile"], q)
            & coverage["cohort"].eq(cohort)
        ]
        if len(x) != 1:
            return {}
        r = x.iloc[0]
        return {
            "threshold": float(r["threshold"]),
            "fraction_inside": float(r["fraction_inside"]),
            "fraction_outside": float(r["fraction_outside"]),
        }

    summary = {
        "script": Path(__file__).name,
        "step": STEP,
        "scientific_role": "External MIMIC-IV HCM baseline manifold transport and coverage",
        "design": "one first valid baseline ECG per subject in both MIMIC and SHaRe",
        "n_mimic_baseline_subjects": int(len(mimic)),
        "n_share_baseline_subjects": int(len(share)),
        "bootstrap_reps": int(args.bootstrap_reps),
        "reference_distributions": ref_info,
        "coverage_95": {
            "HCM_MIMIC": cov_pick("HCM", 0.95, "MIMIC"),
            "HCM_SHaRe": cov_pick("HCM", 0.95, "SHaRe"),
            "Normal_MIMIC": cov_pick("Normal", 0.95, "MIMIC"),
            "Normal_SHaRe": cov_pick("Normal", 0.95, "SHaRe"),
        },
        "h_z06": z06.iloc[0].to_dict(),
        "hcm_latent_covariance_transport": {
            k: v for k, v in cov_info.items()
            if not k.endswith("matrix") and not k.startswith("correlation_difference")
        },
        "constraints": {
            "baseline_only": True,
            "subject_is_sampling_unit": True,
            "mimic_used_for_model_fitting": False,
            "mimic_used_for_scaling": False,
            "mimic_used_for_reference_covariance": False,
            "coverage_thresholds_from_original_training_only": True,
            "reconstruction_error_not_used_as_classifier": True,
        },
        "inputs": {
            "mimic": str(args.mimic),
            "share": str(args.share),
            "normal_latents": str(args.normal_latents),
            "hcm_latents": str(args.hcm_latents),
        },
    }
    (args.results_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    z = z06.iloc[0]
    hcm_m = cov_pick("HCM", 0.95, "MIMIC")
    hcm_s = cov_pick("HCM", 0.95, "SHaRe")
    norm_m = cov_pick("Normal", 0.95, "MIMIC")
    norm_s = cov_pick("Normal", 0.95, "SHaRe")

    lines = [
        "SMASH-HCM Step 34 -- External MIMIC manifold transport",
        f"MIMIC baseline subjects: {len(mimic):,}",
        f"SHaRe baseline subjects: {len(share):,}",
        "",
        "PRIMARY: empirical training-reference coverage",
        f"HCM manifold 95% coverage -- MIMIC: {hcm_m.get('fraction_inside', np.nan):.3f}; SHaRe: {hcm_s.get('fraction_inside', np.nan):.3f}",
        f"Normal manifold 95% outside -- MIMIC: {norm_m.get('fraction_outside', np.nan):.3f}; SHaRe: {norm_s.get('fraction_outside', np.nan):.3f}",
        "",
        "PRIORITY LATENT: H_z_06",
        f"SHaRe mean/SD: {float(z['internal_mean']):.4f} / {float(z['internal_sd']):.4f}",
        f"MIMIC mean/SD: {float(z['external_mean']):.4f} / {float(z['external_sd']):.4f}",
        f"Standardized external-minus-internal shift: {float(z['standardized_mean_shift_external_minus_internal']):.3f} "
        f"(95% bootstrap CI {float(z['shift_ci025']):.3f} to {float(z['shift_ci975']):.3f})",
        f"KS D={float(z['ks_statistic']):.3f}, raw p={float(z['ks_p']):.4g}, BH q={float(z['ks_q_bh_across_8_hcm_latents']):.4g}",
        "",
        "MULTIVARIATE HCM LATENT STRUCTURE",
        f"Pairwise-correlation pattern similarity: {cov_info['pairwise_correlation_pattern_similarity']:.3f}",
        f"Correlation-matrix Frobenius difference: {cov_info['correlation_matrix_frobenius_difference']:.3f}",
        f"External centroid shift in internal Ledoit-Wolf metric: {cov_info['external_centroid_shift_in_internal_ledoitwolf_metric']:.3f}",
        f"Multivariate energy statistic: {cov_info['multivariate_energy_statistic']:.3f}",
        "",
        "Interpretation guardrails:",
        "- High HCM-manifold coverage supports transport of the learned disease representation; it does not prove cohort equivalence.",
        "- Normal-manifold displacement remains descriptive and is not a diagnostic classification analysis.",
        "- Reconstruction-error differences are secondary domain-shift descriptors, not classifier scores.",
        "- Longitudinal replication of H_z_06 is intentionally deferred to the next step.",
    ]
    (args.results_dir / "external_transport_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("DONE")
    print(f"MIMIC baseline subjects: {len(mimic):,}")
    print(f"SHaRe baseline subjects: {len(share):,}")
    print(f"HCM 95% coverage MIMIC: {hcm_m.get('fraction_inside', np.nan):.3f}")
    print(f"Normal 95% outside MIMIC: {norm_m.get('fraction_outside', np.nan):.3f}")
    print(f"H_z_06 standardized shift: {float(z['standardized_mean_shift_external_minus_internal']):.3f}")
    print("Results:", args.results_dir)


if __name__ == "__main__":
    main()
