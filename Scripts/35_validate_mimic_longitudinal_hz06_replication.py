#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
35_validate_mimic_longitudinal_hz06_replication.py

SMASH-HCM -- Step 35
External longitudinal replication of the prespecified HCM-oriented z06 phenotype
in MIMIC-IV.

SCIENTIFIC QUESTION
-------------------
Does the HCM-oriented electrical phenotype H_z_06 reproduce the negative
population-average longitudinal trajectory observed in SHaRe when the frozen
SMASH-HCM encoders are applied to an independent MIMIC-IV HCM cohort?

PRIMARY ANALYSIS
----------------
MIMIC often contains multiple ECGs acquired within the same short clinical
period. To avoid overweighting densely sampled hospital episodes, the primary
analysis first collapses projected ECGs into bounded 30-day clinical windows.

Window construction is NON-TRANSITIVE:
    - take the earliest unused ECG as the anchor;
    - include ECGs acquired from anchor time through anchor + 30 days;
    - choose one representative ECG using a prespecified QC hierarchy;
    - start the next window at the earliest ECG not already assigned.

An ECG can therefore enter a window only by its absolute time from that
window's anchor. Chained gaps cannot extend a window.

Representative hierarchy within each window:
    1. QC: good > acceptable > limited > failed > other;
    2. greater n_usable_beats;
    3. greater n_detected_qrs;
    4. earlier acquisition time;
    5. smaller record_id as deterministic tie-breaker.

The primary cohort contains subjects with >=2 representative visits at >=2
distinct time points. Time is measured continuously in years from each
subject's first representative visit.

Primary model:
    H_z_06_ij = beta0 + beta1 * time_ij + b0_i + b1_i * time_ij + error_ij

A random-intercept model is used only as an explicitly flagged fallback if no
random-intercept/random-slope model converges. REML is used by default to match
Step 17.

SENSITIVITY ANALYSES
--------------------
- 7-day bounded clinical windows, >=2 visits
- 90-day bounded clinical windows, >=2 visits
- 30-day windows, >=3 visits
- 30-day windows, >=2 visits and >=1 year follow-up
- all projected ECGs without clinical-window collapse (audit only)

INTERNAL COMPARATOR
-------------------
The script reads the Step-17 SHaRe mixed-effects result for H_z_06 and reports:
- raw slope and 95% CI in each cohort;
- standardized slope using each cohort's own baseline SD;
- same/different slope direction;
- independent-cohort difference in raw slopes, when both standard errors are
  available.

The between-cohort slope-difference test is a consistency descriptor. Failure
to detect a difference is NOT an equivalence test.

SECONDARY CONTEXT
-----------------
The primary 30-day cohort is also analyzed descriptively for:
    H_z_01, H_z_05, d_N
which showed longitudinal change internally, and H_z_02 as a stable-coordinate
context/negative-control phenotype. No multiplicity-adjusted confirmatory claim
is attached to these contextual models.

IMPORTANT
---------
- No VAE training, fine-tuning, scaling, or latent-reference fitting occurs.
- Step 33 projections are used exactly as exported.
- H_z_06 is prespecified before this external longitudinal analysis.
- The primary inferential test is the MIMIC H_z_06 time slope in the 30-day
  bounded-window cohort; no FDR correction is applied to this single
  prespecified external replication test.
- Population-average longitudinal change does not imply deterministic disease
  progression in every individual patient.

DEFAULT INPUTS
--------------
Results/33_MIMIC_FrozenDualManifoldProjection/
    mimic_frozen_dual_manifold_valid.csv
Results/17_Longitudinal_Trajectories/
    mixed_effects_primary.csv

DEFAULT OUTPUTS
---------------
Results/35_MIMIC_Longitudinal_Hz06_Replication/
    mimic_visit_windows_30d.csv
    mimic_visit_window_sensitivity_summary.csv
    hz06_replication_models.csv
    hz06_subject_empirical_slopes.csv
    share_mimic_hz06_slope_comparison.csv
    secondary_context_models.csv
    longitudinal_replication_report.txt
    run_summary.json
    figures/
        hz06_mimic_trajectories_30d.png
        hz06_share_mimic_slope_comparison.png
        hz06_subject_slope_distribution.png

RUN
---
python Scripts/35_validate_mimic_longitudinal_hz06_replication.py

DEPENDENCIES
------------
numpy pandas scipy statsmodels matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


SEED = 20261021
DAYS_PER_YEAR = 365.2425
PRIMARY_WINDOW_DAYS = 30.0
SENSITIVITY_WINDOW_DAYS = (7.0, 90.0)
PRIMARY_OUTCOME = "H_z_06"
CONTEXT_OUTCOMES = ["H_z_01", "H_z_05", "d_N", "H_z_02"]

QC_RANK = {
    "good": 0,
    "acceptable": 1,
    "limited": 2,
    "failed": 3,
}


# =============================================================================
# Paths / CLI
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


def first_existing(paths: Sequence[Path], label: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    tried = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{tried}")


def parse_args() -> argparse.Namespace:
    root = infer_project_root()

    mimic_default = first_existing(
        [
            root / "Results" / "33_MIMIC_FrozenDualManifoldProjection" / "mimic_frozen_dual_manifold_valid.csv",
        ],
        "Step-33 valid MIMIC projection CSV",
    )

    share_step17_default = first_existing(
        [
            root / "Results" / "17_Longitudinal_Trajectories" / "mixed_effects_primary.csv",
            root / "Results" / "Phase3_Longitudinal_Trajectories" / "mixed_effects_primary.csv",
        ],
        "Step-17 SHaRe mixed-effects primary results",
    )

    p = argparse.ArgumentParser(
        description="Externally replicate the longitudinal H_z_06 trajectory in MIMIC-IV."
    )
    p.add_argument("--mimic", type=Path, default=mimic_default)
    p.add_argument("--share-step17", type=Path, default=share_step17_default)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=root / "Results" / "35_MIMIC_Longitudinal_Hz06_Replication",
    )
    p.add_argument("--primary-window-days", type=float, default=PRIMARY_WINDOW_DAYS)
    p.add_argument("--min-visits-primary", type=int, default=2)
    p.add_argument("--min-visits-sensitivity", type=int, default=3)
    p.add_argument("--min-followup-years-sensitivity", type=float, default=1.0)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--reml", action="store_true", default=True)
    p.add_argument("--ml", dest="reml", action="store_false")
    p.add_argument("--trajectory-plot-subjects", type=int, default=60)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def safe_num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if np.isfinite(v) else float(default)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def normal_p_two_sided(z: float) -> float:
    if not np.isfinite(z):
        return np.nan
    return float(2.0 * stats.norm.sf(abs(z)))


# =============================================================================
# Input preparation
# =============================================================================

def prepare_mimic_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "subject_id",
        "record_id",
        "acquisition_datetime",
        PRIMARY_OUTCOME,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"MIMIC input missing required columns: {missing}")

    if "projected_frozen_dual_manifold" in df.columns:
        df = df[to_bool(df["projected_frozen_dual_manifold"])].copy()

    if "valid_for_projection" in df.columns:
        df = df[to_bool(df["valid_for_projection"])].copy()

    df["acquisition_datetime"] = pd.to_datetime(
        df["acquisition_datetime"], errors="coerce"
    )
    df[PRIMARY_OUTCOME] = safe_num(df[PRIMARY_OUTCOME])

    df = df[
        df["acquisition_datetime"].notna()
        & np.isfinite(df[PRIMARY_OUTCOME])
    ].copy()

    if len(df) == 0:
        raise RuntimeError("No dated finite projected MIMIC records remain.")

    # Deterministic type for grouping / tie-breaking.
    df["subject_id"] = df["subject_id"].astype(str)
    df["record_id_str"] = df["record_id"].astype(str)

    if "qc_label" not in df.columns:
        df["qc_label"] = ""
    if "n_usable_beats" not in df.columns:
        df["n_usable_beats"] = np.nan
    if "n_detected_qrs" not in df.columns:
        df["n_detected_qrs"] = np.nan

    for c in CONTEXT_OUTCOMES:
        if c in df.columns:
            df[c] = safe_num(df[c])

    return df.sort_values(
        ["subject_id", "acquisition_datetime", "record_id_str"],
        kind="mergesort",
    ).reset_index(drop=True)


# =============================================================================
# Clinical-window collapse
# =============================================================================

def representative_sort_key(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["_qc_rank"] = (
        x["qc_label"].astype(str).str.lower().map(QC_RANK).fillna(4).astype(int)
    )
    x["_n_usable"] = safe_num(x["n_usable_beats"]).fillna(-1)
    x["_n_qrs"] = safe_num(x["n_detected_qrs"]).fillna(-1)
    x["_record_num"] = safe_num(x["record_id"]).fillna(np.inf)
    return x.sort_values(
        ["_qc_rank", "_n_usable", "_n_qrs", "acquisition_datetime", "_record_num", "record_id_str"],
        ascending=[True, False, False, True, True, True],
        kind="mergesort",
    )


def collapse_bounded_windows(df: pd.DataFrame, window_days: float) -> pd.DataFrame:
    if window_days <= 0:
        out = df.copy()
        out["window_days"] = float(window_days)
        out["window_index"] = (
            out.groupby("subject_id", sort=False).cumcount().add(1).astype(int)
        )
        out["window_anchor_datetime"] = out["acquisition_datetime"]
        out["window_end_datetime"] = out["acquisition_datetime"]
        out["window_n_records"] = 1
        out["representative_datetime"] = out["acquisition_datetime"]
        out["representative_record_id"] = out["record_id"]
        return add_window_time(out)

    rows: List[Dict] = []
    delta = pd.Timedelta(days=float(window_days))

    for subject, g in df.groupby("subject_id", sort=False):
        g = g.sort_values(
            ["acquisition_datetime", "record_id_str"], kind="mergesort"
        ).reset_index(drop=True)

        start = 0
        window_index = 1
        while start < len(g):
            anchor_time = g.loc[start, "acquisition_datetime"]
            end_time = anchor_time + delta

            stop = start
            while stop + 1 < len(g) and g.loc[stop + 1, "acquisition_datetime"] <= end_time:
                stop += 1

            members = g.iloc[start : stop + 1].copy()
            rep = representative_sort_key(members).iloc[0].copy()
            rec = rep.to_dict()
            rec.update({
                "window_days": float(window_days),
                "window_index": int(window_index),
                "window_anchor_datetime": anchor_time,
                "window_end_datetime": end_time,
                "window_n_records": int(len(members)),
                "representative_datetime": rep["acquisition_datetime"],
                "representative_record_id": rep["record_id"],
            })
            rows.append(rec)

            start = stop + 1
            window_index += 1

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["subject_id", "representative_datetime", "record_id_str"], kind="mergesort"
    ).reset_index(drop=True)
    return add_window_time(out)


def add_window_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    first = out.groupby("subject_id", sort=False)["representative_datetime"].transform("min")
    out["time_years"] = (
        (out["representative_datetime"] - first).dt.total_seconds()
        / (86400.0 * DAYS_PER_YEAR)
    )
    out["visit_order_windowed"] = (
        out.groupby("subject_id", sort=False).cumcount().add(1).astype(int)
    )
    return out


def cohort_eligibility(
    visits: pd.DataFrame,
    min_visits: int,
    min_followup_years: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    keep_subjects = []

    for subject, g in visits.groupby("subject_id", sort=False):
        t = safe_num(g["time_years"]).to_numpy(dtype=float)
        t = t[np.isfinite(t)]
        n = int(len(t))
        distinct = int(len(np.unique(t)))
        follow = float(np.max(t) - np.min(t)) if n else np.nan
        eligible = (
            n >= int(min_visits)
            and distinct >= 2
            and np.isfinite(follow)
            and follow >= float(min_followup_years)
        )
        rows.append({
            "subject_id": subject,
            "n_visits": n,
            "n_distinct_timepoints": distinct,
            "followup_years": follow,
            "eligible": bool(eligible),
        })
        if eligible:
            keep_subjects.append(subject)

    elig = pd.DataFrame(rows)
    analysis = visits[visits["subject_id"].isin(keep_subjects)].copy()
    return analysis, elig


# =============================================================================
# Mixed-effects model
# =============================================================================

def baseline_sd(df: pd.DataFrame, outcome: str) -> float:
    rows = []
    for _, g in df.groupby("subject_id", sort=False):
        gg = g.sort_values(["time_years", "representative_datetime"], kind="mergesort")
        vals = safe_num(gg[outcome]).to_numpy(dtype=float)
        tt = safe_num(gg["time_years"]).to_numpy(dtype=float)
        m = np.isfinite(vals) & np.isfinite(tt)
        if m.any():
            rows.append(float(vals[m][np.argmin(tt[m])]))
    if len(rows) < 2:
        return np.nan
    sd = float(np.std(rows, ddof=1))
    return sd if np.isfinite(sd) and sd > 0 else np.nan


def summarize_fit(result, data: pd.DataFrame, outcome: str, structure: str, method: str,
                  attempts: List[Dict], convergence_warning: bool, warning_text: str) -> Dict:
    beta = safe_float(result.params.get("time_years", np.nan))
    se = safe_float(result.bse.get("time_years", np.nan))
    p = safe_float(result.pvalues.get("time_years", np.nan))
    intercept = safe_float(result.params.get("Intercept", np.nan))

    try:
        ci = result.conf_int().loc["time_years"]
        lo, hi = float(ci.iloc[0]), float(ci.iloc[1])
    except Exception:
        lo = beta - 1.96 * se if np.isfinite(beta) and np.isfinite(se) else np.nan
        hi = beta + 1.96 * se if np.isfinite(beta) and np.isfinite(se) else np.nan

    bsd = baseline_sd(data, outcome)
    std_beta = beta / bsd if np.isfinite(beta) and np.isfinite(bsd) and bsd > 0 else np.nan

    cov_re = np.asarray(result.cov_re, dtype=float)
    random_intercept_var = np.nan
    random_slope_var = np.nan
    random_intercept_slope_cov = np.nan
    if cov_re.size:
        random_intercept_var = safe_float(cov_re[0, 0])
        if cov_re.shape[0] >= 2:
            random_slope_var = safe_float(cov_re[1, 1])
            random_intercept_slope_cov = safe_float(cov_re[0, 1])

    return {
        "success": True,
        "outcome": outcome,
        "model_structure": structure,
        "optimizer": method,
        "converged": bool(getattr(result, "converged", False)),
        "has_convergence_warning": bool(convergence_warning),
        "warning_text": warning_text,
        "n_obs": int(len(data)),
        "n_subjects": int(data["subject_id"].nunique()),
        "fixed_intercept": intercept,
        "fixed_slope_per_year": beta,
        "fixed_slope_se": se,
        "fixed_slope_ci_low": lo,
        "fixed_slope_ci_high": hi,
        "fixed_slope_p": p,
        "baseline_sd": bsd,
        "fixed_slope_baseline_sd_per_year": std_beta,
        "random_intercept_variance": random_intercept_var,
        "random_slope_variance": random_slope_var,
        "random_intercept_slope_covariance": random_intercept_slope_cov,
        "attempts_json": json.dumps(attempts),
    }


def fit_mixed_model(df: pd.DataFrame, outcome: str, maxiter: int, reml: bool) -> Dict:
    d = df[["subject_id", "time_years", outcome, "representative_datetime"]].copy()
    d["time_years"] = safe_num(d["time_years"])
    d[outcome] = safe_num(d[outcome])
    d = d[np.isfinite(d["time_years"]) & np.isfinite(d[outcome])].copy()

    keep = []
    for subject, g in d.groupby("subject_id", sort=False):
        if len(g) >= 2 and g["time_years"].nunique() >= 2:
            keep.append(subject)
    d = d[d["subject_id"].isin(keep)].copy()

    if len(d) < 20 or d["subject_id"].nunique() < 10:
        return {
            "success": False,
            "outcome": outcome,
            "model_structure": "not_fit",
            "failure_reason": "insufficient_data",
            "n_obs": int(len(d)),
            "n_subjects": int(d["subject_id"].nunique()),
        }

    formula = f"{outcome} ~ time_years"
    attempts: List[Dict] = []
    warned_candidate = None

    for method in ["lbfgs", "powell", "cg"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    data=d,
                    groups=d["subject_id"],
                    re_formula="~time_years",
                )
                result = model.fit(
                    reml=reml,
                    method=method,
                    maxiter=maxiter,
                    disp=False,
                )

            conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            warn_text = " | ".join(str(w.message) for w in caught)
            attempt = {
                "structure": "random_intercept_slope",
                "method": method,
                "converged": bool(getattr(result, "converged", False)),
                "convergence_warning": bool(conv_warn),
                "warnings": warn_text,
            }
            attempts.append(attempt)

            valid = (
                bool(getattr(result, "converged", False))
                and np.isfinite(safe_float(result.params.get("time_years", np.nan)))
                and np.isfinite(safe_float(result.bse.get("time_years", np.nan)))
            )
            if valid:
                candidate = summarize_fit(
                    result, d, outcome, "random_intercept_slope", method,
                    attempts.copy(), conv_warn, warn_text,
                )
                if not conv_warn:
                    return candidate
                if warned_candidate is None:
                    warned_candidate = candidate
        except Exception as exc:
            attempts.append({
                "structure": "random_intercept_slope",
                "method": method,
                "converged": False,
                "convergence_warning": False,
                "warnings": f"{type(exc).__name__}: {exc}",
            })

    # If random-slope fits converged only with ConvergenceWarning, they are not
    # treated as clean convergence. Try the explicit random-intercept fallback
    # before deciding whether a warned random-slope fit must be retained.

    # Explicit fallback: random intercept only.
    warned_ri = None
    for method in ["lbfgs", "powell"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    data=d,
                    groups=d["subject_id"],
                    re_formula="1",
                )
                result = model.fit(
                    reml=reml,
                    method=method,
                    maxiter=maxiter,
                    disp=False,
                )

            conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            warn_text = " | ".join(str(w.message) for w in caught)
            attempts.append({
                "structure": "random_intercept_only",
                "method": method,
                "converged": bool(getattr(result, "converged", False)),
                "convergence_warning": bool(conv_warn),
                "warnings": warn_text,
            })

            valid = (
                bool(getattr(result, "converged", False))
                and np.isfinite(safe_float(result.params.get("time_years", np.nan)))
                and np.isfinite(safe_float(result.bse.get("time_years", np.nan)))
            )
            if valid:
                candidate = summarize_fit(
                    result, d, outcome, "random_intercept_only", method,
                    attempts.copy(), conv_warn, warn_text,
                )
                if not conv_warn:
                    return candidate
                if warned_ri is None:
                    warned_ri = candidate
        except Exception as exc:
            attempts.append({
                "structure": "random_intercept_only",
                "method": method,
                "converged": False,
                "convergence_warning": False,
                "warnings": f"{type(exc).__name__}: {exc}",
            })

    # Prefer a warned random-slope result over a warned random-intercept result
    # only when no warning-free model of either structure was available.
    if warned_candidate is not None:
        warned_candidate["attempts_json"] = json.dumps(attempts)
        return warned_candidate

    if warned_ri is not None:
        warned_ri["attempts_json"] = json.dumps(attempts)
        return warned_ri

    return {
        "success": False,
        "outcome": outcome,
        "model_structure": "failed",
        "failure_reason": json.dumps(attempts),
        "n_obs": int(len(d)),
        "n_subjects": int(d["subject_id"].nunique()),
    }


# =============================================================================
# Empirical slopes / comparator
# =============================================================================

def empirical_subject_slopes(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows = []
    for subject, g in df.groupby("subject_id", sort=False):
        gg = g.sort_values("time_years")
        t = safe_num(gg["time_years"]).to_numpy(dtype=float)
        y = safe_num(gg[outcome]).to_numpy(dtype=float)
        m = np.isfinite(t) & np.isfinite(y)
        t, y = t[m], y[m]
        if len(t) < 2 or len(np.unique(t)) < 2:
            continue
        lr = stats.linregress(t, y)
        i0, i1 = int(np.argmin(t)), int(np.argmax(t))
        dt = float(t[i1] - t[i0])
        rows.append({
            "subject_id": subject,
            "n_visits": int(len(t)),
            "followup_years": dt,
            "ols_slope_per_year": float(lr.slope),
            "ols_r_squared": float(lr.rvalue ** 2),
            "first_value": float(y[i0]),
            "last_value": float(y[i1]),
            "first_to_last_slope_per_year": float((y[i1] - y[i0]) / dt) if dt > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def pick_first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def read_share_hz06_result(path: Path) -> Dict:
    df = pd.read_csv(path)
    if "outcome" not in df.columns:
        raise KeyError("Step-17 file has no outcome column.")
    row = df[df["outcome"].astype(str) == PRIMARY_OUTCOME]
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one {PRIMARY_OUTCOME} row in Step 17, found {len(row)}.")
    r = row.iloc[0]

    slope_col = pick_first_existing_column(df, ["fixed_slope_per_year", "slope_per_year"])
    se_col = pick_first_existing_column(df, ["fixed_slope_se", "slope_se"])
    lo_col = pick_first_existing_column(df, ["fixed_slope_ci_low", "ci_low", "fixed_slope_ci025"])
    hi_col = pick_first_existing_column(df, ["fixed_slope_ci_high", "ci_high", "fixed_slope_ci975"])
    p_col = pick_first_existing_column(df, ["fixed_slope_p", "p"])
    std_col = pick_first_existing_column(df, ["fixed_slope_baseline_sd_per_year", "standardized_slope_per_year"])
    model_col = pick_first_existing_column(df, ["model_structure"])

    if slope_col is None:
        raise KeyError("Could not find the Step-17 fixed slope column.")

    slope = safe_float(r[slope_col])
    lo = safe_float(r[lo_col]) if lo_col else np.nan
    hi = safe_float(r[hi_col]) if hi_col else np.nan
    se = safe_float(r[se_col]) if se_col else np.nan
    if not np.isfinite(se) and np.isfinite(lo) and np.isfinite(hi):
        se = float((hi - lo) / (2.0 * 1.96))

    return {
        "cohort": "SHaRe",
        "outcome": PRIMARY_OUTCOME,
        "fixed_slope_per_year": slope,
        "fixed_slope_se": se,
        "fixed_slope_ci_low": lo,
        "fixed_slope_ci_high": hi,
        "fixed_slope_p": safe_float(r[p_col]) if p_col else np.nan,
        "fixed_slope_baseline_sd_per_year": safe_float(r[std_col]) if std_col else np.nan,
        "model_structure": str(r[model_col]) if model_col else "",
    }


def compare_share_mimic(share: Dict, mimic: Dict) -> pd.DataFrame:
    beta_s = safe_float(share.get("fixed_slope_per_year"))
    se_s = safe_float(share.get("fixed_slope_se"))
    beta_m = safe_float(mimic.get("fixed_slope_per_year"))
    se_m = safe_float(mimic.get("fixed_slope_se"))

    diff = beta_m - beta_s if np.isfinite(beta_m) and np.isfinite(beta_s) else np.nan
    diff_se = math.sqrt(se_m ** 2 + se_s ** 2) if np.isfinite(se_m) and np.isfinite(se_s) else np.nan
    z = diff / diff_se if np.isfinite(diff) and np.isfinite(diff_se) and diff_se > 0 else np.nan
    diff_lo = diff - 1.96 * diff_se if np.isfinite(z) else np.nan
    diff_hi = diff + 1.96 * diff_se if np.isfinite(z) else np.nan

    same_direction = (
        bool(np.sign(beta_m) == np.sign(beta_s))
        if np.isfinite(beta_m) and np.isfinite(beta_s) and beta_m != 0 and beta_s != 0
        else False
    )

    rows = [
        {**share, "analysis": "internal_reference"},
        {**mimic, "cohort": "MIMIC", "analysis": "external_primary_30d"},
    ]
    out = pd.DataFrame(rows)
    out["same_slope_direction"] = same_direction
    out["external_minus_internal_slope"] = diff
    out["slope_difference_se"] = diff_se
    out["slope_difference_ci_low"] = diff_lo
    out["slope_difference_ci_high"] = diff_hi
    out["slope_difference_z"] = z
    out["slope_difference_p"] = normal_p_two_sided(z)
    out["difference_test_is_equivalence_test"] = False
    return out


# =============================================================================
# Plots
# =============================================================================

def plot_trajectories(df: pd.DataFrame, fit: Dict, out: Path, n_subjects: int, seed: int) -> None:
    subjects = np.asarray(sorted(df["subject_id"].unique()))
    rng = np.random.default_rng(seed)
    if len(subjects) > n_subjects:
        subjects = rng.choice(subjects, size=n_subjects, replace=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for subject in subjects:
        g = df[df["subject_id"] == subject].sort_values("time_years")
        ax.plot(g["time_years"], g[PRIMARY_OUTCOME], marker="o", markersize=2.5, linewidth=0.7, alpha=0.25)

    beta0 = safe_float(fit.get("fixed_intercept"))
    beta1 = safe_float(fit.get("fixed_slope_per_year"))
    if np.isfinite(beta0) and np.isfinite(beta1):
        xmax = float(np.nanmax(df["time_years"]))
        xx = np.linspace(0.0, xmax, 200)
        ax.plot(xx, beta0 + beta1 * xx, linewidth=2.5, label="MIMIC population trend")
        ax.legend(frameon=False)

    ax.set_xlabel("Years from first representative ECG")
    ax.set_ylabel(r"$H\_z\_{06}$")
    ax.set_title("MIMIC external H_z_06 trajectories (30-day bounded windows)")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_slope_comparison(comp: pd.DataFrame, out: Path) -> None:
    x = []
    y = []
    lo = []
    hi = []
    labels = []
    for cohort in ["SHaRe", "MIMIC"]:
        r = comp[comp["cohort"] == cohort]
        if len(r) != 1:
            continue
        rr = r.iloc[0]
        beta = safe_float(rr.get("fixed_slope_per_year"))
        l = safe_float(rr.get("fixed_slope_ci_low"))
        h = safe_float(rr.get("fixed_slope_ci_high"))
        if not np.isfinite(beta):
            continue
        labels.append(cohort)
        x.append(len(labels) - 1)
        y.append(beta)
        lo.append(beta - l if np.isfinite(l) else 0.0)
        hi.append(h - beta if np.isfinite(h) else 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    if x:
        ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="o", capsize=5, linewidth=1.5)
    ax.axhline(0.0, linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$H\_z\_{06}$ slope (latent units/year)")
    ax.set_title("Internal and external population slopes")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_subject_slopes(emp: pd.DataFrame, out: Path) -> None:
    vals = safe_num(emp["ols_slope_per_year"]).to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=35)
    if len(vals):
        ax.axvline(np.median(vals), linewidth=1.5, label=f"median={np.median(vals):.3f}")
        ax.legend(frameon=False)
    ax.axvline(0.0, linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"Subject OLS slope of $H\_z\_{06}$ (units/year)")
    ax.set_ylabel("Subjects")
    ax.set_title("External subject-level empirical H_z_06 slopes")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("35 - MIMIC external longitudinal H_z_06 replication")
    print("=" * 88)
    print("MIMIC input :", args.mimic)
    print("SHaRe Step17:", args.share_step17)
    print("Results     :", args.results_dir)
    print("Estimation  :", "REML" if args.reml else "ML")
    print()

    raw = prepare_mimic_input(args.mimic)

    # Construct all needed window definitions once.
    window_defs: Dict[str, pd.DataFrame] = {
        "primary_30d_2plus": collapse_bounded_windows(raw, args.primary_window_days),
        "sensitivity_7d_2plus": collapse_bounded_windows(raw, SENSITIVITY_WINDOW_DAYS[0]),
        "sensitivity_90d_2plus": collapse_bounded_windows(raw, SENSITIVITY_WINDOW_DAYS[1]),
        "audit_all_records_2plus": collapse_bounded_windows(raw, 0.0),
    }

    primary_visits = window_defs["primary_30d_2plus"]
    primary_visits.to_csv(args.results_dir / "mimic_visit_windows_30d.csv", index=False)

    scenario_specs = [
        ("primary_30d_2plus", args.min_visits_primary, 0.0),
        ("sensitivity_7d_2plus", args.min_visits_primary, 0.0),
        ("sensitivity_90d_2plus", args.min_visits_primary, 0.0),
        ("sensitivity_30d_3plus", args.min_visits_sensitivity, 0.0),
        ("sensitivity_30d_2plus_followup1y", args.min_visits_primary, args.min_followup_years_sensitivity),
        ("audit_all_records_2plus", args.min_visits_primary, 0.0),
    ]

    model_rows = []
    window_summary_rows = []
    analysis_sets: Dict[str, pd.DataFrame] = {}

    for scenario, min_visits, min_followup in scenario_specs:
        if scenario.startswith("sensitivity_30d"):
            visits = primary_visits
        else:
            visits = window_defs[scenario]

        analysis, elig = cohort_eligibility(
            visits,
            min_visits=min_visits,
            min_followup_years=min_followup,
        )
        analysis_sets[scenario] = analysis

        fit = fit_mixed_model(
            analysis,
            PRIMARY_OUTCOME,
            maxiter=args.maxiter,
            reml=args.reml,
        )
        fit.update({
            "scenario": scenario,
            "window_days": (
                args.primary_window_days if "30d" in scenario
                else SENSITIVITY_WINDOW_DAYS[0] if "7d" in scenario
                else SENSITIVITY_WINDOW_DAYS[1] if "90d" in scenario
                else 0.0
            ),
            "min_visits": int(min_visits),
            "min_followup_years": float(min_followup),
            "estimation": "REML" if args.reml else "ML",
            "prespecified_primary": scenario == "primary_30d_2plus",
        })
        model_rows.append(fit)

        eligible_elig = elig[elig["eligible"] == True]
        window_summary_rows.append({
            "scenario": scenario,
            "window_days": fit["window_days"],
            "min_visits": int(min_visits),
            "min_followup_years": float(min_followup),
            "n_input_records": int(len(raw)),
            "n_windowed_visits": int(len(visits)),
            "n_subjects_total": int(visits["subject_id"].nunique()),
            "n_subjects_eligible": int(analysis["subject_id"].nunique()),
            "n_visits_eligible": int(len(analysis)),
            "median_visits_per_eligible_subject": float(eligible_elig["n_visits"].median()) if len(eligible_elig) else np.nan,
            "median_followup_years_eligible": float(eligible_elig["followup_years"].median()) if len(eligible_elig) else np.nan,
            "max_followup_years_eligible": float(eligible_elig["followup_years"].max()) if len(eligible_elig) else np.nan,
        })

    models = pd.DataFrame(model_rows)
    models.to_csv(args.results_dir / "hz06_replication_models.csv", index=False)

    window_summary = pd.DataFrame(window_summary_rows)
    window_summary.to_csv(
        args.results_dir / "mimic_visit_window_sensitivity_summary.csv", index=False
    )

    primary_analysis = analysis_sets["primary_30d_2plus"]
    primary_fit_row = models[models["scenario"] == "primary_30d_2plus"].iloc[0].to_dict()

    empirical = empirical_subject_slopes(primary_analysis, PRIMARY_OUTCOME)
    empirical.to_csv(args.results_dir / "hz06_subject_empirical_slopes.csv", index=False)

    # Secondary context in the exact same primary 30-day cohort.
    context_rows = []
    for outcome in CONTEXT_OUTCOMES:
        if outcome not in primary_analysis.columns:
            continue
        res = fit_mixed_model(
            primary_analysis,
            outcome,
            maxiter=args.maxiter,
            reml=args.reml,
        )
        res.update({
            "scenario": "primary_30d_2plus_context",
            "confirmatory_role": "contextual_secondary",
        })
        context_rows.append(res)
    context = pd.DataFrame(context_rows)
    context.to_csv(args.results_dir / "secondary_context_models.csv", index=False)

    # Internal-vs-external comparison.
    share = read_share_hz06_result(args.share_step17)
    comparison = compare_share_mimic(share, primary_fit_row)
    comparison.to_csv(
        args.results_dir / "share_mimic_hz06_slope_comparison.csv", index=False
    )

    # Figures.
    if len(primary_analysis):
        plot_trajectories(
            primary_analysis,
            primary_fit_row,
            fig_dir / "hz06_mimic_trajectories_30d.png",
            n_subjects=args.trajectory_plot_subjects,
            seed=args.seed,
        )
    plot_slope_comparison(
        comparison,
        fig_dir / "hz06_share_mimic_slope_comparison.png",
    )
    if len(empirical):
        plot_subject_slopes(
            empirical,
            fig_dir / "hz06_subject_slope_distribution.png",
        )

    # Summary / report.
    ext_beta = safe_float(primary_fit_row.get("fixed_slope_per_year"))
    ext_lo = safe_float(primary_fit_row.get("fixed_slope_ci_low"))
    ext_hi = safe_float(primary_fit_row.get("fixed_slope_ci_high"))
    ext_p = safe_float(primary_fit_row.get("fixed_slope_p"))
    int_beta = safe_float(share.get("fixed_slope_per_year"))

    same_direction = bool(
        np.isfinite(ext_beta) and np.isfinite(int_beta)
        and ext_beta != 0 and int_beta != 0
        and np.sign(ext_beta) == np.sign(int_beta)
    )
    external_negative = bool(np.isfinite(ext_beta) and ext_beta < 0)
    external_ci_excludes_zero_negative = bool(
        np.isfinite(ext_hi) and ext_hi < 0
    )

    frac_emp_negative = (
        float((safe_num(empirical["ols_slope_per_year"]) < 0).mean())
        if len(empirical) else np.nan
    )
    med_emp = (
        float(safe_num(empirical["ols_slope_per_year"]).median())
        if len(empirical) else np.nan
    )

    comp_first = comparison.iloc[0]
    diff = safe_float(comp_first.get("external_minus_internal_slope"))
    diff_lo = safe_float(comp_first.get("slope_difference_ci_low"))
    diff_hi = safe_float(comp_first.get("slope_difference_ci_high"))
    diff_p = safe_float(comp_first.get("slope_difference_p"))

    summary = {
        "script": Path(__file__).name,
        "step": 35,
        "scientific_role": "External longitudinal replication of prespecified H_z_06 in MIMIC-IV",
        "primary_analysis": {
            "window_days": float(args.primary_window_days),
            "window_algorithm": "bounded anchor; no transitive chaining",
            "minimum_visits": int(args.min_visits_primary),
            "minimum_followup_years": 0.0,
            "n_subjects": int(primary_analysis["subject_id"].nunique()),
            "n_visits": int(len(primary_analysis)),
            "model": "H_z_06 ~ time_years with subject random intercept and random slope; RI fallback allowed",
            "estimation": "REML" if args.reml else "ML",
            "fixed_slope_per_year": ext_beta,
            "fixed_slope_ci_low": ext_lo,
            "fixed_slope_ci_high": ext_hi,
            "fixed_slope_p": ext_p,
            "fixed_slope_baseline_sd_per_year": safe_float(primary_fit_row.get("fixed_slope_baseline_sd_per_year")),
            "model_structure": primary_fit_row.get("model_structure", ""),
            "optimizer": primary_fit_row.get("optimizer", ""),
            "has_convergence_warning": bool(primary_fit_row.get("has_convergence_warning", False)),
        },
        "internal_reference": share,
        "replication_descriptors": {
            "same_slope_direction_as_share": same_direction,
            "external_slope_negative": external_negative,
            "external_95ci_entirely_negative": external_ci_excludes_zero_negative,
            "external_primary_p_lt_0_05": bool(np.isfinite(ext_p) and ext_p < 0.05),
            "subject_empirical_slope_median": med_emp,
            "fraction_subject_empirical_slopes_negative": frac_emp_negative,
        },
        "independent_cohort_slope_difference": {
            "external_minus_internal": diff,
            "ci_low": diff_lo,
            "ci_high": diff_hi,
            "p": diff_p,
            "note": "This is not an equivalence test.",
        },
        "sensitivity_scenarios": window_summary.to_dict(orient="records"),
        "constraints": {
            "models_frozen": True,
            "mimic_used_for_model_fitting": False,
            "mimic_used_for_scaling": False,
            "primary_external_outcome_prespecified": True,
            "primary_multiplicity_correction": "none; one prespecified external replication test",
            "clinical_window_collapse_separate_from_step31_duplicate_deduplication": True,
        },
        "inputs": {
            "mimic": str(args.mimic),
            "share_step17": str(args.share_step17),
        },
    }

    (args.results_dir / "run_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )

    lines = [
        "SMASH-HCM Step 35 -- MIMIC external longitudinal H_z_06 replication",
        f"Primary window: bounded {args.primary_window_days:g}-day anchor windows; no transitive chaining",
        f"Primary eligible cohort: {primary_analysis['subject_id'].nunique():,} subjects, {len(primary_analysis):,} representative visits",
        f"MIMIC H_z_06 slope: {ext_beta:.6f} units/year (95% CI {ext_lo:.6f} to {ext_hi:.6f}), p={ext_p:.6g}",
        f"SHaRe H_z_06 slope: {int_beta:.6f} units/year",
        f"Same slope direction: {same_direction}",
        f"External 95% CI entirely negative: {external_ci_excludes_zero_negative}",
        f"Median empirical subject slope: {med_emp:.6f} units/year",
        f"Fraction empirical subject slopes negative: {frac_emp_negative:.3f}",
        f"External-minus-internal slope difference: {diff:.6f} (95% CI {diff_lo:.6f} to {diff_hi:.6f}), p={diff_p:.6g}",
        "",
        "Interpretation guardrails:",
        "- Directional and statistical agreement can support external longitudinal replication of the prespecified phenotype.",
        "- A non-significant between-cohort slope difference is not evidence of formal equivalence.",
        "- Bounded clinical-window collapse is distinct from Step-31 duplicate-acquisition removal.",
        "- Population-average change does not imply deterministic progression in each patient.",
    ]
    (args.results_dir / "longitudinal_replication_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print()
    print("DONE")
    print(f"Primary subjects : {primary_analysis['subject_id'].nunique():,}")
    print(f"Primary visits   : {len(primary_analysis):,}")
    print(f"MIMIC H_z_06 beta: {ext_beta:.6f} / year")
    print(f"95% CI           : {ext_lo:.6f} to {ext_hi:.6f}")
    print(f"p                : {ext_p:.6g}")
    print(f"SHaRe beta       : {int_beta:.6f} / year")
    print(f"Same direction   : {same_direction}")
    print(f"Results          : {args.results_dir}")


if __name__ == "__main__":
    main()
