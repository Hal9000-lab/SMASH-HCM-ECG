#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
15_cross_manifold_analysis.py

SMASH-HCM -- Model 1 + Model 2 / Step 15
Cross-manifold integration of the Normal-anchored and HCM-static ECG latent spaces.

SCIENTIFIC QUESTION
-------------------
For the same HCM baseline ECGs, Model 1 and Model 2 provide complementary views:

    z^N    : displacement along physiological modes learned from Normal ECGs
    z^HCM  : electrical heterogeneity learned within HCM

This script asks:

1) Which HCM-specific latent axes correspond to combinations of normal physiological
   modes?
2) Which HCM-specific axes are poorly predictable from the Normal latent space?
3) Which HCM axes are associated with morphology that is poorly explained by the
   Normal decoder but better represented by the HCM decoder?
4) Can the HCM axes be provisionally categorized as:
       - normal-extension-like
       - hybrid
       - HCM-specific/off-normal candidate
       - indeterminate

PRIMARY ANALYSIS
----------------
HCM-train (n ~454), because this is the population used to define the HCM-specific
latent manifold.

INDEPENDENT CHECKS
------------------
HCM-validation (n ~97)
HCM-test       (n ~98)

The HCM-test subset is especially important for analyses involving RE_H because
HCM-train reconstruction error is in-sample for the disease-specific VAE.

MERGE STRATEGY
--------------
The primary key is:

    Model-2 phase1_row_index
        ->
    Model-1 record_index / residual_record_summary row_index

This deliberately avoids relying on patient_id, which is not a safe identity key in
the original cohort.

INPUTS
------
Results/06_Phase1_MedianBeat_Beta_VAE/phase1_medianbeat_latents.csv
Results/10_Phase1_MedianBeat_NormalManifoldResiduals/residual_record_summary.csv
Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv

OUTPUTS
-------
Results/15_CrossManifold/
├── cross_manifold_hcm_merged.csv
├── normal_latent_reference_stats.csv
├── hcm_latent_reference_stats.csv
├── latent_cross_correlations_hcm_train.csv
├── latent_cross_correlations_hcm_all.csv
├── latent_cross_correlations_hcm_test.csv
├── latent_cross_correlation_matrix_hcm_train.csv
├── latent_cross_correlation_matrix_hcm_all.csv
├── ridge_mapping_summary.csv
├── ridge_mapping_coefficients.csv
├── ridge_oof_predictions_hcm_train.csv
├── hcm_latent_vs_manifold_metrics.csv
├── phenotype_cross_manifold_summary.csv
├── run_summary.json
└── figures/
    ├── cross_latent_heatmap_hcm_train.png
    ├── cross_latent_heatmap_hcm_all.png
    ├── ridge_explainability.png
    └── delta_re_associations.png

RUN
---
python Scripts\\15_cross_manifold_analysis.py

DEPENDENCIES
------------
numpy pandas scipy scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.covariance import LedoitWolf
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


SEED = 20261004

NORMAL_LATENT_DIM = 16
HCM_LATENT_DIM = 8

RIDGE_ALPHAS = np.logspace(-4, 4, 33)
OUTER_FOLDS = 5
INNER_FOLDS = 5

EPS = 1e-12


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-manifold integration of Model 1 and Model 2."
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
        "--normal-residuals",
        type=Path,
        default=Path(
            "Results/10_Phase1_MedianBeat_NormalManifoldResiduals/"
            "residual_record_summary.csv"
        ),
    )
    p.add_argument(
        "--hcm-latents",
        type=Path,
        default=Path(
            "Results/12_HCM_Static_Beta_VAE/"
            "hcm_static_latents.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/15_CrossManifold"),
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--outer-folds", type=int, default=OUTER_FOLDS)
    p.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    return p.parse_args()


# =====================================================================
# Generic utilities
# =====================================================================

def clean_scalar_id(x):
    """
    Normalize IDs such as:
        b'9889'
        "b'9889'"
        9889.0
        '9889'
    to a stable string.

    This is only a secondary diagnostic key. The primary merge uses row indices.
    """
    if pd.isna(x):
        return ""

    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace").strip()

    s = str(x).strip()

    # Decode string representation of Python bytes.
    if (s.startswith("b'") and s.endswith("'")) or (
        s.startswith('b"') and s.endswith('"')
    ):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace").strip()
        except Exception:
            pass

    # Remove simple numeric .0 artifacts.
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def latent_cols(df: pd.DataFrame, prefix="z_", expected=None):
    cols = [
        c for c in df.columns
        if c.startswith(prefix)
        and c[len(prefix):].isdigit()
    ]
    cols = sorted(cols, key=lambda c: int(c.split("_")[1]))

    if expected is not None and len(cols) != expected:
        raise RuntimeError(
            f"Expected {expected} latent columns, found {len(cols)}: {cols}"
        )
    return cols


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
    qrank = np.clip(qrank, 0, 1)

    tmp = np.empty(m, dtype=float)
    tmp[order] = qrank
    q[finite] = tmp
    return q


def corr_with_n(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(m))

    if n < 10:
        return np.nan, np.nan, n

    xx = x[m]
    yy = y[m]

    if np.std(xx) <= 0 or np.std(yy) <= 0:
        return np.nan, np.nan, n

    if method == "spearman":
        r, p = stats.spearmanr(xx, yy)
    elif method == "pearson":
        r, p = stats.pearsonr(xx, yy)
    else:
        raise ValueError(method)

    return float(r), float(p), n


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)

    if np.sum(m) < 3:
        return np.nan
    if np.var(y_true[m]) <= 0:
        return np.nan

    return float(r2_score(y_true[m], y_pred[m]))


# =====================================================================
# Load and align Model 1 and Model 2
# =====================================================================

def load_inputs(
    normal_latents_path: Path,
    normal_residuals_path: Path,
    hcm_latents_path: Path,
):
    for p in [
        normal_latents_path,
        normal_residuals_path,
        hcm_latents_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required input not found:\n  {p}\n"
                "Use the corresponding command-line option if your file is elsewhere."
            )

    normal = pd.read_csv(normal_latents_path)
    residual = pd.read_csv(normal_residuals_path)
    hcm = pd.read_csv(hcm_latents_path)

    zn = latent_cols(normal, expected=NORMAL_LATENT_DIM)
    zh = latent_cols(hcm, expected=HCM_LATENT_DIM)

    required_normal = ["record_index", "split_name"]
    for c in required_normal:
        if c not in normal.columns:
            raise KeyError(
                f"Model-1 latent file is missing required column '{c}'."
            )

    if "phase1_row_index" not in hcm.columns:
        raise KeyError(
            "Model-2 latent file has no phase1_row_index. "
            "This key is required for safe cross-manifold alignment."
        )

    if "row_index" not in residual.columns:
        raise KeyError(
            "Normal-manifold residual file has no row_index."
        )

    return normal, residual, hcm, zn, zh


def align_hcm_records(
    normal: pd.DataFrame,
    residual: pd.DataFrame,
    hcm: pd.DataFrame,
    zn_cols: Sequence[str],
    zh_cols: Sequence[str],
):
    """
    Build a single HCM baseline table.

    Primary key:
        hcm.phase1_row_index
        -> normal.record_index
        -> residual.row_index
    """

    # Model-1 fields.
    keep_n = [
        "record_index",
        "ecg_id",
        "patient_id",
        "subject_key",
        "source",
        "hcm",
        "split_name",
        "rr_median_ms",
        "heart_rate_bpm",
    ]
    keep_n = [c for c in keep_n if c in normal.columns] + list(zn_cols)

    n = normal[keep_n].copy()
    n = n.rename(
        columns={
            "record_index": "phase1_row_index",
            "split_name": "normal_model_split_name",
            **{z: f"N_{z}" for z in zn_cols},
        }
    )

    # Residual fields from Model 1.
    residual_priority = [
        "row_index",
        "residual_mse_mv2",
        "residual_rmse_mv",
        "residual_mae_mv",
        "residual_mse_mv2_preQRS",
        "residual_mse_mv2_QRS",
        "residual_mse_mv2_STT",
        "STT_to_QRS_mse_ratio",
        "residual_fraction_preQRS",
        "residual_fraction_QRS",
        "residual_fraction_STT",
        "residual_mse_mv2_QRS_individual",
        "residual_mse_mv2_STT_individual",
        "STT_to_QRS_mse_ratio_individual",
    ]

    rcols = [c for c in residual_priority if c in residual.columns]
    r = residual[rcols].copy()
    r = r.rename(columns={"row_index": "phase1_row_index"})

    # Model-2 HCM fields.
    keep_h = [
        "row_index",
        "phase1_row_index",
        "split_code",
        "split_name",
        "recon_mse_scaled",
        "recon_mse_mv2",
        "recon_rmse_mv",
        "ecg_id",
        "patient_id",
        "source_group",
    ] + list(zh_cols)

    keep_h = [c for c in keep_h if c in hcm.columns]
    h = hcm[keep_h].copy()

    h = h.rename(
        columns={
            "row_index": "hcm_static_row_index",
            "split_code": "hcm_split_code",
            "split_name": "hcm_split_name",
            "recon_mse_scaled": "RE_H_scaled",
            "recon_mse_mv2": "RE_H_mse_mv2",
            "recon_rmse_mv": "RE_H_rmse_mv",
            "ecg_id": "hcm_ecg_id_raw",
            "patient_id": "hcm_patient_id_raw",
            **{z: f"H_{z}" for z in zh_cols},
        }
    )

    # Enforce one-to-one row keys.
    if h["phase1_row_index"].duplicated().any():
        dup = h.loc[
            h["phase1_row_index"].duplicated(keep=False),
            "phase1_row_index"
        ].tolist()
        raise RuntimeError(
            "Model-2 phase1_row_index is not unique. "
            f"Example duplicates: {dup[:10]}"
        )

    if n["phase1_row_index"].duplicated().any():
        raise RuntimeError("Model-1 record_index is not unique.")

    if r["phase1_row_index"].duplicated().any():
        raise RuntimeError("Residual row_index is not unique.")

    merged = h.merge(
        n,
        on="phase1_row_index",
        how="left",
        validate="one_to_one",
        indicator="_merge_normal",
    )

    if not (merged["_merge_normal"] == "both").all():
        bad = merged.loc[
            merged["_merge_normal"] != "both",
            "phase1_row_index"
        ].tolist()
        raise RuntimeError(
            "Some HCM-static records could not be mapped into Model 1. "
            f"Example row indices: {bad[:10]}"
        )

    merged = merged.drop(columns="_merge_normal")

    merged = merged.merge(
        r,
        on="phase1_row_index",
        how="left",
        validate="one_to_one",
        indicator="_merge_residual",
    )

    if not (merged["_merge_residual"] == "both").all():
        bad = merged.loc[
            merged["_merge_residual"] != "both",
            "phase1_row_index"
        ].tolist()
        raise RuntimeError(
            "Some HCM-static records could not be mapped to Model-1 residuals. "
            f"Example row indices: {bad[:10]}"
        )

    merged = merged.drop(columns="_merge_residual")

    # Rename Model-1 global residual to explicit RE_N terminology.
    if "residual_mse_mv2" in merged.columns:
        merged = merged.rename(
            columns={
                "residual_mse_mv2": "RE_N_mse_mv2",
                "residual_rmse_mv": "RE_N_rmse_mv",
                "residual_mae_mv": "RE_N_mae_mv",
            }
        )

    # Secondary diagnostic identity check.
    if "ecg_id" in merged.columns and "hcm_ecg_id_raw" in merged.columns:
        a = merged["ecg_id"].map(clean_scalar_id)
        b = merged["hcm_ecg_id_raw"].map(clean_scalar_id)
        merged["ecg_id_crosscheck_match"] = (a == b)
    else:
        merged["ecg_id_crosscheck_match"] = np.nan

    # Keep only actual HCM baseline rows from Model 2.
    hcm_mask = merged["hcm_split_name"].astype(str).str.startswith("hcm_")
    merged = merged.loc[hcm_mask].copy().reset_index(drop=True)

    if len(merged) != 649:
        warnings.warn(
            f"Expected 649 HCM baseline ECGs after alignment, found {len(merged)}."
        )

    return merged


# =====================================================================
# Normal-manifold and HCM-manifold distances
# =====================================================================

def mahalanobis_from_reference(
    reference_matrix: np.ndarray,
    target_matrix: np.ndarray,
):
    reference_matrix = np.asarray(reference_matrix, dtype=float)
    target_matrix = np.asarray(target_matrix, dtype=float)

    if not np.all(np.isfinite(reference_matrix)):
        raise ValueError("Reference latent matrix contains non-finite values.")
    if not np.all(np.isfinite(target_matrix)):
        raise ValueError("Target latent matrix contains non-finite values.")

    lw = LedoitWolf().fit(reference_matrix)
    mean = lw.location_
    precision = lw.precision_

    delta = target_matrix - mean[None, :]
    d2 = np.einsum("ni,ij,nj->n", delta, precision, delta)
    d2 = np.maximum(d2, 0.0)
    d = np.sqrt(d2)

    return d, d2, lw


def compute_manifold_distances(
    normal: pd.DataFrame,
    merged: pd.DataFrame,
    zn_cols: Sequence[str],
    zh_cols: Sequence[str],
):
    # Normal reference: Model-1 normal_train only.
    ntrain = normal[
        normal["split_name"].astype(str).eq("normal_train")
    ].copy()

    Xn_ref = ntrain[list(zn_cols)].to_numpy(dtype=float)
    Xn_hcm = merged[
        [f"N_{z}" for z in zn_cols]
    ].to_numpy(dtype=float)

    dN, dN2, lwN = mahalanobis_from_reference(
        Xn_ref,
        Xn_hcm,
    )

    merged["d_N"] = dN
    merged["d_N2"] = dN2

    # HCM reference: Model-2 HCM-train only.
    htrain = merged[
        merged["hcm_split_name"].astype(str).eq("hcm_train")
    ].copy()

    Xh_ref = htrain[
        [f"H_{z}" for z in zh_cols]
    ].to_numpy(dtype=float)

    Xh_all = merged[
        [f"H_{z}" for z in zh_cols]
    ].to_numpy(dtype=float)

    dH, dH2, lwH = mahalanobis_from_reference(
        Xh_ref,
        Xh_all,
    )

    merged["d_HCM"] = dH
    merged["d_HCM2"] = dH2

    normal_stats = pd.DataFrame({
        "latent": [f"N_{z}" for z in zn_cols],
        "reference": "normal_train",
        "mean": lwN.location_,
        "variance_diag": np.diag(lwN.covariance_),
    })

    hcm_stats = pd.DataFrame({
        "latent": [f"H_{z}" for z in zh_cols],
        "reference": "hcm_train",
        "mean": lwH.location_,
        "variance_diag": np.diag(lwH.covariance_),
    })

    return merged, normal_stats, hcm_stats


# =====================================================================
# Residual integration
# =====================================================================

def add_residual_integration_metrics(df: pd.DataFrame):
    if "RE_N_mse_mv2" not in df.columns:
        raise KeyError(
            "Aligned table has no RE_N_mse_mv2 from Model-1 residual analysis."
        )
    if "RE_H_mse_mv2" not in df.columns:
        raise KeyError(
            "Aligned table has no RE_H_mse_mv2 from Model-2 latent output."
        )

    re_n = pd.to_numeric(df["RE_N_mse_mv2"], errors="coerce").to_numpy(float)
    re_h = pd.to_numeric(df["RE_H_mse_mv2"], errors="coerce").to_numpy(float)

    df["Delta_RE_mse_mv2"] = re_n - re_h
    df["RE_N_over_RE_H"] = re_n / (re_h + EPS)
    df["log_RE_N_over_RE_H"] = np.log(
        (re_n + EPS) / (re_h + EPS)
    )

    return df


# =====================================================================
# Cross-latent correlations
# =====================================================================

def cross_latent_correlations(
    df: pd.DataFrame,
    zn_prefixed: Sequence[str],
    zh_prefixed: Sequence[str],
    group_name: str,
):
    rows = []

    for hz in zh_prefixed:
        for nz in zn_prefixed:
            sr, sp, n = corr_with_n(df[hz], df[nz], "spearman")
            pr, pp, _ = corr_with_n(df[hz], df[nz], "pearson")

            rows.append({
                "group": group_name,
                "hcm_latent": hz,
                "normal_latent": nz,
                "n": n,
                "spearman_rho": sr,
                "spearman_p": sp,
                "pearson_r": pr,
                "pearson_p": pp,
                "abs_spearman_rho": abs(sr) if np.isfinite(sr) else np.nan,
            })

    out = pd.DataFrame(rows)
    out["spearman_q_fdr"] = bh_fdr(out["spearman_p"])
    out["pearson_q_fdr"] = bh_fdr(out["pearson_p"])
    return out


def correlation_matrix(corr_table, zh_prefixed, zn_prefixed):
    piv = corr_table.pivot(
        index="hcm_latent",
        columns="normal_latent",
        values="spearman_rho",
    )
    return piv.reindex(
        index=list(zh_prefixed),
        columns=list(zn_prefixed),
    )


# =====================================================================
# Nested ridge mapping zHCM <- zN
# =====================================================================

def standardize_train_apply(X_train, X_other):
    mu = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)

    return (
        (X_train - mu) / sd,
        (X_other - mu) / sd,
        mu,
        sd,
    )


def standardize_y_train_apply(y_train, y_other):
    mu = float(np.mean(y_train))
    sd = float(np.std(y_train, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0

    return (
        (y_train - mu) / sd,
        (y_other - mu) / sd,
        mu,
        sd,
    )


def nested_ridge_for_latent(
    train_df,
    val_df,
    test_df,
    zn_prefixed,
    target_col,
    outer_folds,
    inner_folds,
    seed,
):
    X = train_df[list(zn_prefixed)].to_numpy(dtype=float)
    y = train_df[target_col].to_numpy(dtype=float)

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError(
            f"Non-finite values in ridge inputs for {target_col}."
        )

    kf = KFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=seed,
    )

    oof = np.full(len(train_df), np.nan)
    fold_alpha = np.full(len(train_df), np.nan)

    for fold, (itr, iva) in enumerate(kf.split(X), start=1):
        Xtr = X[itr]
        Xva = X[iva]
        ytr = y[itr]
        yva = y[iva]

        Xtr_z, Xva_z, _, _ = standardize_train_apply(Xtr, Xva)
        ytr_z, _, y_mu, y_sd = standardize_y_train_apply(ytr, yva)

        model = RidgeCV(
            alphas=RIDGE_ALPHAS,
            cv=inner_folds,
            scoring="neg_mean_squared_error",
        )
        model.fit(Xtr_z, ytr_z)

        pred_z = model.predict(Xva_z)
        pred = pred_z * y_sd + y_mu

        oof[iva] = pred
        fold_alpha[iva] = model.alpha_

    oof_r2 = safe_r2(y, oof)
    oof_r, oof_p, _ = corr_with_n(y, oof, "pearson")
    oof_s, oof_sp, _ = corr_with_n(y, oof, "spearman")

    # Final fit on all HCM-train, standardized within HCM-train.
    Xtrain_z, _, x_mu, x_sd = standardize_train_apply(X, X)
    ytrain_z, _, y_mu, y_sd = standardize_y_train_apply(y, y)

    final_model = RidgeCV(
        alphas=RIDGE_ALPHAS,
        cv=inner_folds,
        scoring="neg_mean_squared_error",
    )
    final_model.fit(Xtrain_z, ytrain_z)

    def predict_external(d):
        if len(d) == 0:
            return np.array([], dtype=float), np.nan

        Xo = d[list(zn_prefixed)].to_numpy(dtype=float)
        yo = d[target_col].to_numpy(dtype=float)
        Xo_z = (Xo - x_mu) / x_sd
        pred_z = final_model.predict(Xo_z)
        pred = pred_z * y_sd + y_mu
        return pred, safe_r2(yo, pred)

    val_pred, val_r2 = predict_external(val_df)
    test_pred, test_r2 = predict_external(test_df)

    coefficients_standardized = final_model.coef_.astype(float)

    return {
        "oof_pred": oof,
        "oof_fold_alpha": fold_alpha,
        "hcm_train_oof_r2": oof_r2,
        "hcm_train_oof_pearson_r": oof_r,
        "hcm_train_oof_pearson_p": oof_p,
        "hcm_train_oof_spearman_rho": oof_s,
        "hcm_train_oof_spearman_p": oof_sp,
        "final_alpha": float(final_model.alpha_),
        "val_pred": val_pred,
        "hcm_val_r2": val_r2,
        "test_pred": test_pred,
        "hcm_test_r2": test_r2,
        "coefficients_standardized": coefficients_standardized,
    }


def ridge_mapping_analysis(
    merged,
    zn_prefixed,
    zh_prefixed,
    outer_folds,
    inner_folds,
    seed,
):
    train = merged[
        merged["hcm_split_name"].astype(str).eq("hcm_train")
    ].copy().reset_index(drop=False)

    val = merged[
        merged["hcm_split_name"].astype(str).eq("hcm_val")
    ].copy().reset_index(drop=False)

    test = merged[
        merged["hcm_split_name"].astype(str).eq("hcm_test")
    ].copy().reset_index(drop=False)

    summary_rows = []
    coef_rows = []
    oof_rows = []

    for j, hz in enumerate(zh_prefixed):
        res = nested_ridge_for_latent(
            train_df=train,
            val_df=val,
            test_df=test,
            zn_prefixed=zn_prefixed,
            target_col=hz,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            seed=seed + j,
        )

        summary_rows.append({
            "hcm_latent": hz,
            "hcm_train_oof_r2": res["hcm_train_oof_r2"],
            "hcm_train_oof_pearson_r": res["hcm_train_oof_pearson_r"],
            "hcm_train_oof_spearman_rho": res["hcm_train_oof_spearman_rho"],
            "hcm_val_r2": res["hcm_val_r2"],
            "hcm_test_r2": res["hcm_test_r2"],
            "final_ridge_alpha": res["final_alpha"],
            "median_outer_selected_alpha": float(
                np.nanmedian(res["oof_fold_alpha"])
            ),
        })

        for nz, coef in zip(
            zn_prefixed,
            res["coefficients_standardized"],
        ):
            coef_rows.append({
                "hcm_latent": hz,
                "normal_latent": nz,
                "standardized_ridge_coefficient": float(coef),
                "abs_standardized_ridge_coefficient": abs(float(coef)),
            })

        for i in range(len(train)):
            oof_rows.append({
                "hcm_latent": hz,
                "merged_row_index": int(train.loc[i, "index"]),
                "phase1_row_index": int(train.loc[i, "phase1_row_index"]),
                "observed": float(train.loc[i, hz]),
                "oof_predicted": float(res["oof_pred"][i]),
                "outer_fold_selected_alpha": float(res["oof_fold_alpha"][i]),
            })

    ridge_summary = pd.DataFrame(summary_rows)
    ridge_coef = pd.DataFrame(coef_rows)
    ridge_oof = pd.DataFrame(oof_rows)

    return ridge_summary, ridge_coef, ridge_oof


# =====================================================================
# HCM latent vs cross-manifold metrics
# =====================================================================

def latent_vs_metrics(
    df,
    zh_prefixed,
    group_name,
):
    metrics = [
        "d_N",
        "d_N2",
        "d_HCM",
        "RE_N_mse_mv2",
        "RE_H_mse_mv2",
        "Delta_RE_mse_mv2",
        "RE_N_over_RE_H",
        "log_RE_N_over_RE_H",
        "residual_mse_mv2_QRS",
        "residual_mse_mv2_STT",
        "STT_to_QRS_mse_ratio",
    ]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for hz in zh_prefixed:
        for metric in metrics:
            sr, sp, n = corr_with_n(df[hz], df[metric], "spearman")
            pr, pp, _ = corr_with_n(df[hz], df[metric], "pearson")

            rows.append({
                "group": group_name,
                "hcm_latent": hz,
                "metric": metric,
                "n": n,
                "spearman_rho": sr,
                "spearman_p": sp,
                "pearson_r": pr,
                "pearson_p": pp,
                "abs_spearman_rho": abs(sr) if np.isfinite(sr) else np.nan,
            })

    out = pd.DataFrame(rows)
    out["spearman_q_fdr"] = bh_fdr(out["spearman_p"])
    out["pearson_q_fdr"] = bh_fdr(out["pearson_p"])
    return out


# =====================================================================
# Exploratory cross-manifold phenotype categorization
# =====================================================================

def explainability_level(r2):
    if not np.isfinite(r2):
        return "unknown"
    if r2 >= 0.50:
        return "high"
    if r2 >= 0.25:
        return "moderate"
    return "low"


def offnormal_level(rho):
    if not np.isfinite(rho):
        return "unknown"
    a = abs(rho)
    if a >= 0.30:
        return "strong"
    if a >= 0.20:
        return "moderate"
    return "weak"


def exploratory_class(r2, delta_rho):
    """
    Exploratory only. Thresholds are descriptive, not clinical.
    """
    e = explainability_level(r2)
    o = offnormal_level(delta_rho)

    if e == "high" and o in {"weak", "moderate"}:
        return "normal-extension-like"

    if e == "low" and o in {"moderate", "strong"}:
        return "HCM-specific/off-normal candidate"

    if (
        e == "moderate"
        or (e == "high" and o == "strong")
    ):
        return "hybrid"

    return "indeterminate"


def build_phenotype_summary(
    ridge_summary,
    cross_train,
    metric_all,
    metric_test,
    ridge_coef,
    zh_prefixed,
):
    rows = []

    for hz in zh_prefixed:
        rr = ridge_summary[ridge_summary["hcm_latent"] == hz].iloc[0]

        # Strongest pairwise normal-latent relation in HCM-train.
        cc = (
            cross_train[cross_train["hcm_latent"] == hz]
            .sort_values("abs_spearman_rho", ascending=False)
            .iloc[0]
        )

        # Strongest standardized ridge coefficient.
        rc = (
            ridge_coef[ridge_coef["hcm_latent"] == hz]
            .sort_values(
                "abs_standardized_ridge_coefficient",
                ascending=False,
            )
            .iloc[0]
        )

        def metric_rho(table, metric):
            q = table[
                (table["hcm_latent"] == hz)
                & (table["metric"] == metric)
            ]
            if len(q) == 0:
                return np.nan, np.nan
            q = q.iloc[0]
            return float(q["spearman_rho"]), float(q["spearman_q_fdr"])

        dN_rho, dN_q = metric_rho(metric_all, "d_N")
        reN_rho, reN_q = metric_rho(metric_all, "RE_N_mse_mv2")
        reH_rho, reH_q = metric_rho(metric_all, "RE_H_mse_mv2")
        delta_rho, delta_q = metric_rho(metric_all, "Delta_RE_mse_mv2")
        delta_test_rho, delta_test_q = metric_rho(
            metric_test,
            "Delta_RE_mse_mv2",
        )

        row = {
            "hcm_latent": hz,
            "strongest_normal_latent_pair": cc["normal_latent"],
            "strongest_pairwise_spearman_rho": cc["spearman_rho"],
            "strongest_pairwise_q_fdr": cc["spearman_q_fdr"],
            "strongest_ridge_normal_latent": rc["normal_latent"],
            "strongest_standardized_ridge_coefficient": rc[
                "standardized_ridge_coefficient"
            ],
            "hcm_train_oof_r2_from_normal_latents": rr[
                "hcm_train_oof_r2"
            ],
            "hcm_val_r2_from_normal_latents": rr["hcm_val_r2"],
            "hcm_test_r2_from_normal_latents": rr["hcm_test_r2"],
            "rho_d_N_hcm_all": dN_rho,
            "q_d_N_hcm_all": dN_q,
            "rho_RE_N_hcm_all": reN_rho,
            "q_RE_N_hcm_all": reN_q,
            "rho_RE_H_hcm_all": reH_rho,
            "q_RE_H_hcm_all": reH_q,
            "rho_Delta_RE_hcm_all": delta_rho,
            "q_Delta_RE_hcm_all": delta_q,
            "rho_Delta_RE_hcm_test": delta_test_rho,
            "q_Delta_RE_hcm_test": delta_test_q,
            "normal_explainability_level": explainability_level(
                rr["hcm_train_oof_r2"]
            ),
            "off_normal_association_level": offnormal_level(
                delta_rho
            ),
            "exploratory_cross_manifold_class": exploratory_class(
                rr["hcm_train_oof_r2"],
                delta_rho,
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# Figures
# =====================================================================

def plot_heatmap(matrix: pd.DataFrame, title: str, path: Path):
    mat = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(
        mat,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [c.replace("N_z_", "N z") for c in matrix.columns],
        rotation=45,
        ha="right",
    )

    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(
        [c.replace("H_z_", "HCM z") for c in matrix.index]
    )

    ax.set_xlabel("Normal-anchored latent coordinates")
    ax.set_ylabel("HCM-specific latent coordinates")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ridge_summary(ridge_summary: pd.DataFrame, path: Path):
    d = ridge_summary.sort_values(
        "hcm_train_oof_r2",
        ascending=False,
    ).reset_index(drop=True)

    x = np.arange(len(d))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width,
        d["hcm_train_oof_r2"],
        width,
        label="HCM-train OOF",
    )
    ax.bar(
        x,
        d["hcm_val_r2"],
        width,
        label="HCM-val",
    )
    ax.bar(
        x + width,
        d["hcm_test_r2"],
        width,
        label="HCM-test",
    )

    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [z.replace("H_z_", "z") for z in d["hcm_latent"]]
    )
    ax.set_ylabel(r"$R^2$ predicting HCM latent from 16 Normal latents")
    ax.set_xlabel("HCM-specific latent coordinate")
    ax.set_title("Cross-manifold explainability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_re_associations(
    phenotype_summary: pd.DataFrame,
    path: Path,
):
    d = phenotype_summary.copy()

    x = np.arange(len(d))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        d["rho_Delta_RE_hcm_all"],
        width,
        label="All HCM baseline",
    )
    ax.bar(
        x + width / 2,
        d["rho_Delta_RE_hcm_test"],
        width,
        label="HCM-test",
    )

    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [z.replace("H_z_", "z") for z in d["hcm_latent"]]
    )
    ax.set_ylabel(r"Spearman $\rho$ with $\Delta RE$")
    ax.set_xlabel("HCM-specific latent coordinate")
    ax.set_title(
        r"HCM latent association with $\Delta RE = RE_N - RE_H$"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
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
    print("15 - Cross-manifold integration")
    print("=" * 78)
    print("Normal latents  :", args.normal_latents)
    print("Normal residuals:", args.normal_residuals)
    print("HCM latents     :", args.hcm_latents)
    print("Output          :", args.output_dir)

    normal, residual, hcm, zn_cols, zh_cols = load_inputs(
        args.normal_latents,
        args.normal_residuals,
        args.hcm_latents,
    )

    print()
    print("[input]")
    print("Model-1 records:", len(normal))
    print("Model-2 records:", len(hcm))
    print("Normal latents :", len(zn_cols))
    print("HCM latents    :", len(zh_cols))

    merged = align_hcm_records(
        normal=normal,
        residual=residual,
        hcm=hcm,
        zn_cols=zn_cols,
        zh_cols=zh_cols,
    )

    print()
    print("[alignment]")
    print("Aligned HCM baseline ECGs:", len(merged))
    if "ecg_id_crosscheck_match" in merged.columns:
        valid = merged["ecg_id_crosscheck_match"].dropna()
        if len(valid):
            print(
                "ECG-ID secondary cross-check:",
                f"{int(valid.sum())}/{len(valid)} match"
            )

    merged, normal_stats, hcm_stats = compute_manifold_distances(
        normal=normal,
        merged=merged,
        zn_cols=zn_cols,
        zh_cols=zh_cols,
    )

    merged = add_residual_integration_metrics(merged)

    # Explicit prefixed latent names in merged table.
    zn = [f"N_{z}" for z in zn_cols]
    zh = [f"H_{z}" for z in zh_cols]

    # Split masks.
    mask_train = merged["hcm_split_name"].astype(str).eq("hcm_train")
    mask_val = merged["hcm_split_name"].astype(str).eq("hcm_val")
    mask_test = merged["hcm_split_name"].astype(str).eq("hcm_test")

    if int(mask_train.sum()) != 454:
        warnings.warn(
            f"Expected 454 HCM-train rows, found {int(mask_train.sum())}."
        )
    if int(mask_test.sum()) != 98:
        warnings.warn(
            f"Expected 98 HCM-test rows, found {int(mask_test.sum())}."
        )

    # -------------------------------------------------------------
    # Cross-latent correlation matrices
    # -------------------------------------------------------------
    cross_train = cross_latent_correlations(
        merged.loc[mask_train],
        zn,
        zh,
        "hcm_train",
    )
    cross_all = cross_latent_correlations(
        merged,
        zn,
        zh,
        "hcm_all",
    )
    cross_test = cross_latent_correlations(
        merged.loc[mask_test],
        zn,
        zh,
        "hcm_test",
    )

    cross_train.to_csv(
        args.output_dir / "latent_cross_correlations_hcm_train.csv",
        index=False,
    )
    cross_all.to_csv(
        args.output_dir / "latent_cross_correlations_hcm_all.csv",
        index=False,
    )
    cross_test.to_csv(
        args.output_dir / "latent_cross_correlations_hcm_test.csv",
        index=False,
    )

    matrix_train = correlation_matrix(
        cross_train,
        zh,
        zn,
    )
    matrix_all = correlation_matrix(
        cross_all,
        zh,
        zn,
    )

    matrix_train.to_csv(
        args.output_dir / "latent_cross_correlation_matrix_hcm_train.csv"
    )
    matrix_all.to_csv(
        args.output_dir / "latent_cross_correlation_matrix_hcm_all.csv"
    )

    # -------------------------------------------------------------
    # Ridge mapping
    # -------------------------------------------------------------
    print()
    print("[ridge mapping: HCM latent <- 16 Normal latents]")

    ridge_summary, ridge_coef, ridge_oof = ridge_mapping_analysis(
        merged=merged,
        zn_prefixed=zn,
        zh_prefixed=zh,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )

    ridge_summary.to_csv(
        args.output_dir / "ridge_mapping_summary.csv",
        index=False,
    )
    ridge_coef.to_csv(
        args.output_dir / "ridge_mapping_coefficients.csv",
        index=False,
    )
    ridge_oof.to_csv(
        args.output_dir / "ridge_oof_predictions_hcm_train.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # HCM latent vs manifold distances / residuals
    # -------------------------------------------------------------
    metric_train = latent_vs_metrics(
        merged.loc[mask_train],
        zh,
        "hcm_train",
    )
    metric_all = latent_vs_metrics(
        merged,
        zh,
        "hcm_all",
    )
    metric_test = latent_vs_metrics(
        merged.loc[mask_test],
        zh,
        "hcm_test",
    )

    metric_tables = pd.concat(
        [metric_train, metric_all, metric_test],
        ignore_index=True,
    )
    metric_tables.to_csv(
        args.output_dir / "hcm_latent_vs_manifold_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Integrated phenotype table
    # -------------------------------------------------------------
    phenotype = build_phenotype_summary(
        ridge_summary=ridge_summary,
        cross_train=cross_train,
        metric_all=metric_all,
        metric_test=metric_test,
        ridge_coef=ridge_coef,
        zh_prefixed=zh,
    )
    phenotype.to_csv(
        args.output_dir / "phenotype_cross_manifold_summary.csv",
        index=False,
    )

    # Save merged patient-level representation.
    merged.to_csv(
        args.output_dir / "cross_manifold_hcm_merged.csv",
        index=False,
    )
    normal_stats.to_csv(
        args.output_dir / "normal_latent_reference_stats.csv",
        index=False,
    )
    hcm_stats.to_csv(
        args.output_dir / "hcm_latent_reference_stats.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------
    plot_heatmap(
        matrix_train,
        "Cross-manifold latent associations: HCM-train",
        fig_dir / "cross_latent_heatmap_hcm_train.png",
    )
    plot_heatmap(
        matrix_all,
        "Cross-manifold latent associations: all HCM baseline ECGs",
        fig_dir / "cross_latent_heatmap_hcm_all.png",
    )
    plot_ridge_summary(
        ridge_summary,
        fig_dir / "ridge_explainability.png",
    )
    plot_delta_re_associations(
        phenotype,
        fig_dir / "delta_re_associations.png",
    )

    # -------------------------------------------------------------
    # Headline diagnostics
    # -------------------------------------------------------------
    re_n = merged["RE_N_mse_mv2"].to_numpy(dtype=float)
    re_h = merged["RE_H_mse_mv2"].to_numpy(dtype=float)
    delta = merged["Delta_RE_mse_mv2"].to_numpy(dtype=float)

    re_summary = {
        "all_hcm": {
            "n": int(len(merged)),
            "RE_N_median_mV2": float(np.nanmedian(re_n)),
            "RE_H_median_mV2": float(np.nanmedian(re_h)),
            "Delta_RE_median_mV2": float(np.nanmedian(delta)),
            "fraction_Delta_RE_positive": float(
                np.nanmean(delta > 0)
            ),
        },
        "hcm_test": {
            "n": int(mask_test.sum()),
            "RE_N_median_mV2": float(
                np.nanmedian(merged.loc[mask_test, "RE_N_mse_mv2"])
            ),
            "RE_H_median_mV2": float(
                np.nanmedian(merged.loc[mask_test, "RE_H_mse_mv2"])
            ),
            "Delta_RE_median_mV2": float(
                np.nanmedian(
                    merged.loc[mask_test, "Delta_RE_mse_mv2"]
                )
            ),
            "fraction_Delta_RE_positive": float(
                np.nanmean(
                    merged.loc[mask_test, "Delta_RE_mse_mv2"] > 0
                )
            ),
        },
    }

    # Strongest cross-space correlation per HCM latent.
    strongest_pairs = []
    for hz in zh:
        q = (
            cross_train[cross_train["hcm_latent"] == hz]
            .sort_values("abs_spearman_rho", ascending=False)
            .iloc[0]
        )
        strongest_pairs.append({
            "hcm_latent": hz,
            "normal_latent": q["normal_latent"],
            "spearman_rho": float(q["spearman_rho"]),
            "q_fdr": float(q["spearman_q_fdr"]),
        })

    summary = {
        "script": "15_cross_manifold_analysis.py",
        "version": "1.0",
        "scientific_role": (
            "Integration of the Normal-anchored and HCM-static ECG manifolds"
        ),
        "inputs": {
            "normal_latents": str(args.normal_latents),
            "normal_residuals": str(args.normal_residuals),
            "hcm_latents": str(args.hcm_latents),
        },
        "merge_key": (
            "Model-2 phase1_row_index -> Model-1 record_index / residual row_index"
        ),
        "n_aligned_hcm_baseline": int(len(merged)),
        "groups": {
            "hcm_train": int(mask_train.sum()),
            "hcm_val": int(mask_val.sum()),
            "hcm_test": int(mask_test.sum()),
        },
        "normal_latent_dim": int(len(zn)),
        "hcm_latent_dim": int(len(zh)),
        "normal_distance": (
            "Ledoit-Wolf Mahalanobis distance fit on Model-1 normal_train latent means"
        ),
        "hcm_distance": (
            "Ledoit-Wolf Mahalanobis distance fit on Model-2 HCM-train latent means"
        ),
        "delta_re_definition": "Delta_RE = RE_N_mse_mv2 - RE_H_mse_mv2",
        "ridge_mapping": {
            "target": "each HCM-specific latent coordinate",
            "predictors": "all 16 Normal-anchored latent coordinates",
            "primary_population": "HCM-train",
            "outer_cv_folds": int(args.outer_folds),
            "inner_ridge_cv_folds": int(args.inner_folds),
            "alphas": RIDGE_ALPHAS.tolist(),
            "validation": "final HCM-train fit evaluated on HCM-val and HCM-test",
        },
        "reconstruction_error_summary": re_summary,
        "strongest_pairwise_normal_latent_per_hcm_axis": strongest_pairs,
        "phenotype_classes_are_exploratory": True,
        "classification_thresholds": {
            "normal_explainability_high": "OOF R2 >= 0.50",
            "normal_explainability_moderate": "0.25 <= OOF R2 < 0.50",
            "normal_explainability_low": "OOF R2 < 0.25",
            "off_normal_strong": "|rho(Delta_RE)| >= 0.30",
            "off_normal_moderate": "0.20 <= |rho(Delta_RE)| < 0.30",
            "off_normal_weak": "|rho(Delta_RE)| < 0.20",
        },
        "important_cautions": [
            "Cross-manifold associations are descriptive statistical relationships, not mechanistic equivalence.",
            "The two VAEs differ in architecture, training population and capacity.",
            "RE_N and RE_H are expressed in the same physical MSE units but arise from different models.",
            "RE_H is in-sample for HCM-train and held-out for HCM-val/test; HCM-test is therefore the cleanest subset for Delta_RE interpretation.",
            "Delta_RE does not by itself prove disease specificity or biological substrate.",
            "Exploratory phenotype classes use prespecified descriptive thresholds and are not clinical categories.",
        ],
    }

    (
        args.output_dir / "run_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------
    print()
    print("=" * 78)
    print("STEP 15 COMPLETE")
    print("=" * 78)

    print()
    print("[cross-manifold explainability]")
    disp = ridge_summary[
        [
            "hcm_latent",
            "hcm_train_oof_r2",
            "hcm_val_r2",
            "hcm_test_r2",
        ]
    ].copy()
    print(disp.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print()
    print("[integrated phenotype summary]")
    cols = [
        "hcm_latent",
        "strongest_normal_latent_pair",
        "strongest_pairwise_spearman_rho",
        "hcm_train_oof_r2_from_normal_latents",
        "rho_Delta_RE_hcm_all",
        "rho_Delta_RE_hcm_test",
        "exploratory_cross_manifold_class",
    ]
    print(
        phenotype[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:+.3f}",
        )
    )

    print()
    print("[reconstruction comparison]")
    print(
        "All HCM:  median RE_N =",
        f"{re_summary['all_hcm']['RE_N_median_mV2']:.6g}",
        "| median RE_H =",
        f"{re_summary['all_hcm']['RE_H_median_mV2']:.6g}",
        "| median Delta_RE =",
        f"{re_summary['all_hcm']['Delta_RE_median_mV2']:.6g}",
    )
    print(
        "HCM-test: median RE_N =",
        f"{re_summary['hcm_test']['RE_N_median_mV2']:.6g}",
        "| median RE_H =",
        f"{re_summary['hcm_test']['RE_H_median_mV2']:.6g}",
        "| median Delta_RE =",
        f"{re_summary['hcm_test']['Delta_RE_median_mV2']:.6g}",
    )

    print()
    print("Results:", args.output_dir)


if __name__ == "__main__":
    main()
