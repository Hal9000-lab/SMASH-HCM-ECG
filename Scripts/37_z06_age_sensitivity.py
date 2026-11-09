#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
37_z06_age_sensitivity.py

SMASH-HCM -- Step 37
Is the longitudinal drift of H_z_06 compatible with physiological ageing?

WHY THIS STEP EXISTS
--------------------
Within one subject, age = baseline_age + time. Time and ageing therefore cannot
be separated inside the HCM cohort alone, and adjusting for baseline age does
not remove ageing from the time slope. An external age reference is required.

ANALYSES
--------
A. Normal age gradient
   PTB-XL Normal ECGs are projected through the frozen HCM-oriented VAE. A
   restricted cubic spline of age (plus sex, when available) is fitted, and the
   annual age gradient is averaged over the observed HCM ECGs aged 18-90.
B. HCM within/between decomposition
   y ~ t + baseline_age, with random intercept and slope. The cross-sectional
   baseline-age coefficient is contrasted with the longitudinal time slope.
C. Age-residualized trajectory
   Each HCM value is reduced by the value expected from the Normal age curve,
   and the mixed model is refitted on the residual.
D. Physiological anchors
   The same analyses are repeated on QRS net area in aVL and frontal QRS axis,
   measured with the Step-14 extractor. They do not depend on the VAE.
E. Prespecified summary
   R = Normal age gradient / HCM longitudinal slope, with a Monte Carlo 95% CI.
       CI lower bound >= 0.75 -> slope compatible with the ageing gradient
       CI upper bound <= 0.25 -> ageing explains little of the slope
       otherwise              -> partial or indeterminate

INTERPRETIVE LIMITS
-------------------
- PTB-XL is cross-sectional: the gradient mixes ageing and birth-cohort effects.
- PTB-XL NORM selection tends to exclude older subjects with left-axis deviation,
  so the Normal gradient may be biased towards zero. A small R does not by itself
  exclude ageing; a longitudinal non-HCM control is still the stronger test.
- H_z_02 is a negative control: it showed no linear trend in Step 17.
- Normal ECGs are out-of-distribution for the HCM VAE. Only the age gradient is
  used here, not the absolute level of H_z_06 in Normal ECGs.

INPUTS
------
Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5
Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5
Results/12_HCM_Static_Beta_VAE/best_model.pt
Results/12_HCM_Static_Beta_VAE/hcm_static_latents.csv
Results/16_Longitudinal_DualManifold/Phase3_HCM_Longitudinal_DualManifold.h5
Results/16_Longitudinal_DualManifold/longitudinal_dual_manifold.csv
Results/18_Genotype_Longitudinal/genotype_subject_cohort.csv   (sex, optional)
Data/SHaReData-FLO.xlsx                                        (sex, optional)
Scripts/12_train_hcm_static_beta_vae.py
Scripts/14_quantify_hcm_static_latent_ecg_features.py

OUTPUTS
-------
Results/37_Z06_Age_Sensitivity/
    age_sensitivity_summary.csv
    age_sensitivity_models.csv
    normal_reference_values.csv
    hcm_visit_values.csv
    age_sensitivity_report.txt
    run_summary.json
    figures/hz06_age_gradient.png

RUN
---
python Scripts/37_z06_age_sensitivity.py

DEPENDENCIES
------------
numpy pandas h5py statsmodels matplotlib torch openpyxl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 20261103
DAYS_PER_YEAR = 365.2425
FS_HZ = 500.0
BEAT_START_MS = -350.0
PTB_MIN_AGE = 18.0
PTB_MAX_AGE = 90.0
PAPER_SLOPE_HZ06 = -0.0531

LATENT_OUTCOMES = ["H_z_06", "H_z_01", "H_z_05", "H_z_02"]
ANCHOR_OUTCOMES = ["QRS_net_area_aVL", "QRS_axis_wrapped_deg"]
NEGATIVE_CONTROLS = {"H_z_02"}


# =============================================================================
# Paths and generic helpers
# =============================================================================

def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve(), here.parent, *here.parents]
    for p in candidates:
        if (p / "Data").exists() and (p / "Results").exists():
            return p
    return here.parent.parent


def parse_args() -> argparse.Namespace:
    root = infer_project_root()
    r = root / "Results"
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--normal-h5", type=Path, default=r / "05_Phase1_MedianBeat_VAE" / "Phase1_MedianBeat_NormalAnchored_preprocessed.h5")
    p.add_argument("--hcm-prepared-h5", type=Path, default=r / "11_HCM_Static_VAE_Preparation" / "Phase2_HCMStatic_preprocessed.h5")
    p.add_argument("--hcm-checkpoint", type=Path, default=r / "12_HCM_Static_Beta_VAE" / "best_model.pt")
    p.add_argument("--hcm-latents", type=Path, default=r / "12_HCM_Static_Beta_VAE" / "hcm_static_latents.csv")
    p.add_argument("--longitudinal-h5", type=Path, default=r / "16_Longitudinal_DualManifold" / "Phase3_HCM_Longitudinal_DualManifold.h5")
    p.add_argument("--longitudinal-csv", type=Path, default=r / "16_Longitudinal_DualManifold" / "longitudinal_dual_manifold.csv")
    p.add_argument("--crosswalk", type=Path, default=r / "18_Genotype_Longitudinal" / "genotype_subject_cohort.csv")
    p.add_argument("--share-xlsx", type=Path, default=root / "Data" / "SHaReData-FLO.xlsx")
    p.add_argument("--vae-module", type=Path, default=root / "Scripts" / "12_train_hcm_static_beta_vae.py")
    p.add_argument("--feature-module", type=Path, default=root / "Scripts" / "14_quantify_hcm_static_latent_ecg_features.py")
    p.add_argument("--results-dir", type=Path, default=r / "37_Z06_Age_Sensitivity")
    p.add_argument("--ptb-male-code", type=float, default=0.0, help="PTB-XL sex code meaning male (PTB-XL convention: 0).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-draws", type=int, default=200000)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_strings(values) -> np.ndarray:
    return np.asarray(
        [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in values],
        dtype=object,
    )


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def wrap_axis_deg(deg) -> np.ndarray:
    """Wrap the frontal axis to [-120, 240) so left-axis values stay continuous."""
    return ((np.asarray(deg, dtype=float) + 120.0) % 360.0) - 120.0


# =============================================================================
# Normal age reference
# =============================================================================

def rcs_basis(x, knots) -> np.ndarray:
    """Restricted cubic spline basis (Harrell), linear beyond the boundary knots."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    k = len(t)
    if k < 3:
        raise ValueError("At least three knots are required.")
    scale = (t[-1] - t[0]) ** 2
    columns = [x]
    for j in range(k - 2):
        term = (
            np.clip(x - t[j], 0, None) ** 3
            - np.clip(x - t[k - 2], 0, None) ** 3 * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + np.clip(x - t[k - 1], 0, None) ** 3 * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        )
        columns.append(term / scale)
    return np.column_stack(columns)


def reference_design(age, knots, male=None) -> np.ndarray:
    age = np.asarray(age, dtype=float)
    parts = [np.ones((len(age), 1)), rcs_basis(age, knots)]
    if male is not None:
        parts.append(np.asarray(male, dtype=float).reshape(-1, 1))
    return np.hstack(parts)


def fit_age_reference(age, y, male=None) -> dict:
    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(age) & np.isfinite(y)
    if male is not None:
        male = np.asarray(male, dtype=float)
        mask &= np.isfinite(male)
    if mask.sum() < 50:
        raise RuntimeError(f"Too few Normal observations for an age reference: {int(mask.sum())}")
    knots = np.unique(np.quantile(age[mask], [0.05, 0.35, 0.65, 0.95]))
    X = reference_design(age[mask], knots, None if male is None else male[mask])
    fit = sm.OLS(y[mask], X).fit(cov_type="HC3")
    return {
        "beta": np.asarray(fit.params, dtype=float),
        "cov": np.asarray(fit.cov_params(), dtype=float),
        "knots": knots,
        "uses_sex": male is not None,
        "n": int(mask.sum()),
    }


def predict_reference(ref: dict, age, male=None) -> np.ndarray:
    age = np.asarray(age, dtype=float)
    out = np.full(len(age), np.nan)
    if ref["uses_sex"]:
        if male is None:
            raise ValueError("This reference requires sex.")
        male = np.asarray(male, dtype=float)
        mask = np.isfinite(age) & np.isfinite(male)
        out[mask] = reference_design(age[mask], ref["knots"], male[mask]) @ ref["beta"]
    else:
        mask = np.isfinite(age)
        out[mask] = reference_design(age[mask], ref["knots"]) @ ref["beta"]
    return out


def average_age_gradient(ref: dict, ages) -> tuple[float, float]:
    """Annual gradient averaged over the supplied ages, with its standard error."""
    ages = np.asarray(ages, dtype=float)
    ages = ages[np.isfinite(ages)]
    male = np.zeros(len(ages)) if ref["uses_sex"] else None
    upper = reference_design(ages + 0.5, ref["knots"], male)
    lower = reference_design(ages - 0.5, ref["knots"], male)
    c = np.mean(upper - lower, axis=0)
    return float(c @ ref["beta"]), float(np.sqrt(max(c @ ref["cov"] @ c, 0.0)))


# =============================================================================
# Mixed models
# =============================================================================

def eligible_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """Same rule as Step 17: >=2 valid visits at >=2 distinct times."""
    grouped = df.groupby("subject")["t"]
    keep = grouped.nunique()
    return df[df["subject"].isin(keep[keep >= 2].index)].copy()


def fit_mixed(df: pd.DataFrame, outcome: str, rhs: str, maxiter: int = 2000) -> dict:
    """Random intercept/slope model; y is rescaled for numerical stability."""
    terms = [c for c in ["t", "age0_c", "male"] if c in rhs]
    d = df[["subject", outcome, *dict.fromkeys(["t", *terms])]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = eligible_subjects(d)
    if d["subject"].nunique() < 10:
        return {"success": False, "failure_reason": "insufficient_subjects", "n_subjects": int(d["subject"].nunique())}

    first = d.sort_values(["subject", "t"]).groupby("subject")[outcome].first()
    scale = float(first.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(d[outcome].std(ddof=1))
    d["_y"] = d[outcome] / scale

    attempts = [("random_intercept_slope", "~t", m) for m in ("lbfgs", "powell", "cg")]
    attempts += [("random_intercept_only", "1", m) for m in ("lbfgs", "powell")]
    for structure, re_formula, method in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = smf.mixedlm(f"_y ~ {rhs}", data=d, groups=d["subject"], re_formula=re_formula).fit(
                    reml=True, method=method, maxiter=maxiter, disp=False
                )
        except Exception:
            continue
        if not bool(getattr(result, "converged", False)):
            continue
        names = list(result.fe_params.index)
        return {
            "success": True,
            "structure": structure,
            "optimizer": method,
            "n_obs": int(len(d)),
            "n_subjects": int(d["subject"].nunique()),
            "baseline_sd": scale,
            "params": {n: float(result.fe_params[n]) * scale for n in names},
            "cov": pd.DataFrame(result.cov_params().loc[names, names].to_numpy(dtype=float) * scale ** 2, index=names, columns=names),
        }
    return {"success": False, "failure_reason": "no_convergence", "n_subjects": int(d["subject"].nunique())}


def term_stats(fit: dict, term: str) -> tuple[float, float]:
    if not fit.get("success") or term not in fit["params"]:
        return np.nan, np.nan
    return fit["params"][term], float(np.sqrt(max(fit["cov"].loc[term, term], 0.0)))


def model_row(outcome: str, label: str, fit: dict, term: str = "t") -> dict:
    estimate, se = term_stats(fit, term)
    return {
        "outcome": outcome,
        "model": label,
        "term": term,
        "success": bool(fit.get("success", False)),
        "structure": fit.get("structure", ""),
        "n_subjects": fit.get("n_subjects", np.nan),
        "n_obs": fit.get("n_obs", np.nan),
        "estimate_per_year": estimate,
        "se": se,
        "ci95_low": estimate - 1.96 * se,
        "ci95_high": estimate + 1.96 * se,
        "z": estimate / se if se and np.isfinite(se) and se > 0 else np.nan,
    }


def within_between_contrast(fit: dict) -> tuple[float, float]:
    if not fit.get("success") or not {"t", "age0_c"} <= set(fit["params"]):
        return np.nan, np.nan
    cov = fit["cov"]
    variance = cov.loc["t", "t"] + cov.loc["age0_c", "age0_c"] - 2.0 * cov.loc["t", "age0_c"]
    return fit["params"]["t"] - fit["params"]["age0_c"], float(np.sqrt(max(variance, 0.0)))


# =============================================================================
# Prespecified summary
# =============================================================================

def ratio_interval(gradient, gradient_se, slope, slope_se, n_draws, rng) -> tuple[float, float, float]:
    if not all(np.isfinite([gradient, gradient_se, slope, slope_se])) or slope == 0:
        return np.nan, np.nan, np.nan
    numerator = rng.normal(gradient, gradient_se, n_draws)
    denominator = rng.normal(slope, slope_se, n_draws)
    low, high = np.quantile(numerator / denominator, [0.025, 0.975])
    return float(gradient / slope), float(low), float(high)


def classify_ratio(low: float, high: float) -> str:
    if not np.isfinite(low) or not np.isfinite(high):
        return "not_estimable"
    if low >= 0.75:
        return "compatible_with_ageing_gradient"
    if high <= 0.25 and low >= -0.25:
        return "ageing_explains_little"
    return "partial_or_indeterminate"


# =============================================================================
# Core analysis, independent from file formats
# =============================================================================

def analyse_outcome(outcome, normal, hcm, n_draws, rng, maxiter) -> tuple[dict, list]:
    """Run analyses A-C and the summary ratio for one outcome."""
    rows = []
    ref_age = fit_age_reference(normal["age"], normal[outcome])
    # The reference spans 18-90 years: HCM visits outside it must not weight the
    # gradient through a linear extrapolation of the spline.
    in_range_ages = hcm.loc[hcm["age"].between(PTB_MIN_AGE, PTB_MAX_AGE), "age"]
    gradient, gradient_se = average_age_gradient(ref_age, in_range_ages)
    n_finite = normal[outcome].notna() & normal["age"].notna()
    simple = sm.OLS(
        normal.loc[n_finite, outcome].to_numpy(dtype=float),
        sm.add_constant(normal.loc[n_finite, "age"].to_numpy(dtype=float)),
    ).fit(cov_type="HC3")

    hcm = hcm.copy()
    base = fit_mixed(hcm, outcome, "t", maxiter)
    decomposition = fit_mixed(hcm, outcome, "t + age0_c", maxiter)
    interaction = fit_mixed(hcm, outcome, "t + age0_c + t:age0_c", maxiter)

    hcm["_residual"] = hcm[outcome] - predict_reference(ref_age, hcm["age"])
    residual = fit_mixed(hcm, "_residual", "t", maxiter)

    within_range = hcm[hcm["age"].between(PTB_MIN_AGE, PTB_MAX_AGE)]
    residual_range = fit_mixed(within_range, "_residual", "t", maxiter)

    rows += [
        model_row(outcome, "time_only", base),
        model_row(outcome, "time_plus_baseline_age", decomposition),
        model_row(outcome, "time_plus_baseline_age", decomposition, "age0_c"),
        model_row(outcome, "time_by_baseline_age", interaction, "t:age0_c"),
        model_row(outcome, "normal_age_residual", residual),
        model_row(outcome, "normal_age_residual_ages_18_90", residual_range),
    ]

    sex_slope = sex_residual_slope = np.nan
    if hcm["male"].notna().any() and normal["male"].notna().any():
        ref_sex = fit_age_reference(normal["age"], normal[outcome], normal["male"])
        hcm["_residual_sex"] = hcm[outcome] - predict_reference(ref_sex, hcm["age"], hcm["male"])
        sex_model = fit_mixed(hcm, outcome, "t + age0_c + male", maxiter)
        sex_residual = fit_mixed(hcm[hcm["male"].notna()], "_residual_sex", "t", maxiter)
        rows += [
            model_row(outcome, "time_plus_baseline_age_sex", sex_model),
            model_row(outcome, "normal_age_sex_residual", sex_residual),
        ]
        sex_slope = term_stats(sex_model, "t")[0]
        sex_residual_slope = term_stats(sex_residual, "t")[0]

    slope, slope_se = term_stats(base, "t")
    between, between_se = term_stats(decomposition, "age0_c")
    contrast, contrast_se = within_between_contrast(decomposition)
    residual_slope, residual_se = term_stats(residual, "t")
    ratio, ratio_low, ratio_high = ratio_interval(gradient, gradient_se, slope, slope_se, n_draws, rng)

    summary = {
        "outcome": outcome,
        "negative_control": outcome in NEGATIVE_CONTROLS,
        "n_normal": ref_age["n"],
        "normal_gradient_per_year": gradient,
        "normal_gradient_se": gradient_se,
        "normal_linear_gradient_per_year": float(simple.params[1]),
        "hcm_n_subjects": base.get("n_subjects", np.nan),
        "hcm_n_obs": base.get("n_obs", np.nan),
        "hcm_slope_per_year": slope,
        "hcm_slope_se": slope_se,
        "hcm_between_age_per_year": between,
        "hcm_between_age_se": between_se,
        "within_minus_between": contrast,
        "within_minus_between_se": contrast_se,
        "residual_slope_per_year": residual_slope,
        "residual_slope_se": residual_se,
        "hcm_slope_adjusted_age_sex": sex_slope,
        "residual_slope_age_sex": sex_residual_slope,
        "ratio_gradient_over_slope": ratio,
        "ratio_ci95_low": ratio_low,
        "ratio_ci95_high": ratio_high,
        "classification": classify_ratio(ratio_low, ratio_high),
    }
    return summary, rows


# =============================================================================
# Data loading
# =============================================================================

def load_normals(path: Path, male_code: float) -> tuple[np.ndarray, pd.DataFrame]:
    with h5py.File(path, "r") as h5:
        beats = np.asarray(h5["signals/median_all_mv"][:], dtype=np.float32)
        labels = np.asarray(h5["labels/hcm"][:]).astype(int)
        split = np.asarray(h5["labels/split_code"][:]).astype(int)
        age = np.asarray(h5["metadata/age_ptbxl_only"][:], dtype=float)
        sex = decode_strings(h5["metadata/sex_ptbxl_only"][:])
        ecg_id = decode_strings(h5["metadata/ecg_id"][:])

    meta = pd.DataFrame({"phase1_row_index": np.arange(len(labels)), "ecg_id": ecg_id, "split_code": split, "age": age})
    sex_num = pd.to_numeric(pd.Series(sex), errors="coerce")
    meta["male"] = np.where(sex_num.notna(), (sex_num == male_code).astype(float), np.nan)

    keep = (labels == 0) & (age >= PTB_MIN_AGE) & (age <= PTB_MAX_AGE)
    if keep.sum() == 0:
        raise RuntimeError(
            "No PTB-XL age is available in metadata/age_ptbxl_only. "
            "Check that the Step-02 manifest contains an 'age' column."
        )
    return beats[keep], meta.loc[keep].reset_index(drop=True)


def project_normals(beats, meta, args) -> pd.DataFrame:
    import torch

    module = load_module(args.vae_module, "smash_step12_vae")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.hcm_checkpoint, map_location=device, weights_only=False)
    latent_dim = int(checkpoint.get("latent_dim", 8))
    model = module.HCMStaticBetaVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with h5py.File(args.hcm_prepared_h5, "r") as h5:
        mean = float(h5.attrs["hcm_train_scaling_mean_mv"])
        std = float(h5.attrs["hcm_train_scaling_std_mv"])

    scaled = ((beats - mean) / std).astype(np.float32)
    mu = np.empty((len(scaled), latent_dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(scaled), args.batch_size):
            stop = min(start + args.batch_size, len(scaled))
            batch = torch.from_numpy(scaled[start:stop]).to(device)
            mu[start:stop] = model.encode(batch)[0].cpu().numpy()

    out = meta.copy()
    for k in range(latent_dim):
        out[f"H_z_{k:02d}"] = mu[:, k]

    saved = pd.read_csv(args.hcm_latents)
    saved = saved[saved["split_name"].astype(str).eq("normal_eval")]
    cols = [f"z_{k:02d}" for k in range(latent_dim)]
    check = out.merge(saved[["phase1_row_index", *cols]], on="phase1_row_index", how="inner")
    if len(check):
        diff = np.abs(
            check[[f"H_z_{k:02d}" for k in range(latent_dim)]].to_numpy(dtype=float)
            - check[cols].to_numpy(dtype=float)
        )
        print(f"[cross-check] {len(check)} normal_eval ECGs, max |z difference| = {diff.max():.3g}")
        if float(np.median(diff)) > 1e-3 or float(diff.max()) > 5e-2:
            raise RuntimeError("Projection does not reproduce the saved Step-12 latents.")
    return out


def extract_anchors(beats, module, label) -> pd.DataFrame:
    rows = []
    for i, beat in enumerate(beats):
        area = axis = np.nan
        if np.isfinite(beat).all():
            try:
                feats, qc = module.extract_ecg_features(beat, fs=FS_HZ, beat_start_ms=BEAT_START_MS, rr_ms=np.nan)
                if qc.get("qrs_boundary_ok", False):
                    area = feats.get("QRS_net_area_mVms_aVL", np.nan)
                    axis = feats.get("QRS_axis_deg", np.nan)
            except Exception:
                pass
        rows.append({"QRS_net_area_aVL": area, "QRS_axis_wrapped_deg": float(wrap_axis_deg(axis))})
        if (i + 1) % 1000 == 0 or i + 1 == len(beats):
            print(f"[anchors:{label}] {i + 1:,}/{len(beats):,}")
    return pd.DataFrame(rows)


def load_sex_map(crosswalk: Path, xlsx: Path) -> dict:
    if not crosswalk.exists() or not xlsx.exists():
        warnings.warn("Crosswalk or SHaRe workbook not found: HCM sex is unavailable.")
        return {}

    def pid(value):
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value).strip()

    cross = pd.read_csv(crosswalk)
    cross = cross[cross["mapping_status"].astype(str).str.strip().eq("matched")]
    sheet = pd.read_excel(xlsx, sheet_name="HCM_PatientKeyOutcomes", engine="openpyxl", dtype=object)
    codes = {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0}
    by_pid = {
        pid(p): codes.get(str(s).strip().lower(), np.nan)
        for p, s in zip(sheet["HCM_PatientKeyOutcomes_PID"], sheet["Sex"])
    }
    return {
        str(subject).strip(): by_pid.get(pid(p), np.nan)
        for subject, p in zip(cross["analysis_subject_key"], cross["SHaRe_PID"])
    }


def load_hcm(csv_path: Path, sex_map: dict) -> pd.DataFrame:
    latent_cols = [f"H_z_{k:02d}" for k in range(8)]
    d = pd.read_csv(csv_path)
    d = d[as_bool(d["valid_for_latent_projection"])].copy()
    d["subject"] = d["analysis_subject_key"].astype(str).str.strip()
    d["t"] = pd.to_numeric(d["time_from_first_valid_ecg_years"], errors="coerce")
    ecg_time = pd.to_datetime(d["ecg_time"], errors="coerce")
    birth = pd.to_datetime(d["birthdate"], errors="coerce")
    d["age"] = (ecg_time - birth).dt.total_seconds() / (86400.0 * DAYS_PER_YEAR)
    d = d[np.isfinite(d["t"]) & np.isfinite(d["age"])].copy()
    d = eligible_subjects(d)

    first = d.sort_values(["subject", "t"]).groupby("subject")["age"].first()
    d["age0"] = d["subject"].map(first)
    d["age0_c"] = d["age0"] - float(first.mean())
    d["male"] = d["subject"].map(sex_map).astype(float)
    keep = ["longitudinal_row_index", "subject", "t", "age", "age0", "age0_c", "male", *latent_cols]
    return d[keep].reset_index(drop=True)


# =============================================================================
# Outputs
# =============================================================================

def plot_gradient(normal: pd.DataFrame, hcm: pd.DataFrame, path: Path) -> None:
    ref = fit_age_reference(normal["age"], normal["H_z_06"])
    grid = np.linspace(normal["age"].min(), normal["age"].max(), 200)
    first = hcm.sort_values(["subject", "t"]).groupby("subject").first()
    bins = pd.cut(first["age"], bins=np.arange(10, 95, 10))
    grouped = first.groupby(bins, observed=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(normal["age"], normal["H_z_06"], s=4, alpha=0.12, label="PTB-XL Normal")
    ax.plot(grid, predict_reference(ref, grid), linewidth=2, label="Normal age curve")
    ax.errorbar(
        grouped["age"].mean(), grouped["H_z_06"].mean(), yerr=1.96 * grouped["H_z_06"].sem(),
        fmt="o", capsize=3, label="HCM baseline, mean by decade",
    )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("H_z_06")
    ax.set_title("H_z_06 and age")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report(summary: pd.DataFrame) -> str:
    lines = ["SMASH-HCM Step 37 -- H_z_06 and physiological ageing", ""]
    for _, r in summary.iterrows():
        lines += [
            r["outcome"] + (" (negative control)" if r["negative_control"] else ""),
            f"  HCM slope            : {r['hcm_slope_per_year']:+.5f} per year (SE {r['hcm_slope_se']:.5f})",
            f"  Normal age gradient  : {r['normal_gradient_per_year']:+.5f} per year (SE {r['normal_gradient_se']:.5f})",
            f"  HCM baseline-age term: {r['hcm_between_age_per_year']:+.5f} per year (SE {r['hcm_between_age_se']:.5f})",
            f"  Within minus between : {r['within_minus_between']:+.5f} (SE {r['within_minus_between_se']:.5f})",
            f"  Residual HCM slope   : {r['residual_slope_per_year']:+.5f} per year (SE {r['residual_slope_se']:.5f})",
            f"  Ratio gradient/slope : {r['ratio_gradient_over_slope']:+.3f} "
            f"(95% CI {r['ratio_ci95_low']:+.3f} to {r['ratio_ci95_high']:+.3f})",
            f"  Classification       : {r['classification']}",
            "",
        ]
    lines += [
        "Guardrails:",
        "- PTB-XL is cross-sectional and NORM selection may shrink the age gradient.",
        "- A ratio near 0 does not exclude ageing without a longitudinal non-HCM control.",
        "- The ratio is unstable when the HCM slope is close to zero (see H_z_02).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[1/5] Loading and projecting Normal ECGs")
    normal_beats, normal_meta = load_normals(args.normal_h5, args.ptb_male_code)
    normal = project_normals(normal_beats, normal_meta, args)
    print(f"      Normal ECGs: {len(normal):,}; age {normal['age'].min():.0f}-{normal['age'].max():.0f}")
    print(f"      Sex codes: {normal['male'].value_counts(dropna=False).to_dict()}")

    print("[2/5] Loading HCM visits")
    hcm = load_hcm(args.longitudinal_csv, load_sex_map(args.crosswalk, args.share_xlsx))
    print(f"      HCM: {hcm['subject'].nunique()} subjects, {len(hcm):,} visits (paper: 493 / 2,437)")

    print("[3/5] Measuring physiological anchors")
    features = load_module(args.feature_module, "smash_step14_features")
    normal = pd.concat([normal, extract_anchors(normal_beats, features, "normal")], axis=1)
    with h5py.File(args.longitudinal_h5, "r") as h5:
        hcm_beats = np.asarray(h5["signals/median_all_mv"][:], dtype=np.float32)
    selected = hcm_beats[hcm["longitudinal_row_index"].to_numpy(dtype=int)]
    hcm = pd.concat([hcm, extract_anchors(selected, features, "hcm")], axis=1)

    print("[4/5] Fitting models")
    summaries, rows = [], []
    for outcome in [*LATENT_OUTCOMES, *ANCHOR_OUTCOMES]:
        print(f"      {outcome}")
        summary, model_rows = analyse_outcome(outcome, normal, hcm, args.n_draws, rng, args.maxiter)
        summaries.append(summary)
        rows += model_rows
    summary_df = pd.DataFrame(summaries)
    models_df = pd.DataFrame(rows)

    reference = summary_df.loc[summary_df["outcome"].eq("H_z_06"), "hcm_slope_per_year"].iloc[0]
    if abs(reference - PAPER_SLOPE_HZ06) > 1e-3:
        warnings.warn(f"H_z_06 slope {reference:.4f} differs from the paper value {PAPER_SLOPE_HZ06}.")

    print("[5/5] Writing outputs")
    summary_df.to_csv(args.results_dir / "age_sensitivity_summary.csv", index=False)
    models_df.to_csv(args.results_dir / "age_sensitivity_models.csv", index=False)
    normal.to_csv(args.results_dir / "normal_reference_values.csv", index=False)
    hcm.to_csv(args.results_dir / "hcm_visit_values.csv", index=False)
    plot_gradient(normal, hcm, args.results_dir / "figures" / "hz06_age_gradient.png")

    report = build_report(summary_df)
    (args.results_dir / "age_sensitivity_report.txt").write_text(report, encoding="utf-8")
    (args.results_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "script": Path(__file__).name,
                "n_normal": int(len(normal)),
                "n_hcm_subjects": int(hcm["subject"].nunique()),
                "n_hcm_visits": int(len(hcm)),
                "seed": args.seed,
                "models_frozen": True,
                "ratio_rule": "CI low >= 0.75 compatible; CI within [-0.25, 0.25] little; otherwise indeterminate",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(report)
    print("Results:", args.results_dir)


if __name__ == "__main__":
    main()
