#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17_analyze_longitudinal_dual_manifold_trajectories.py

SMASH-HCM -- Longitudinal Phase / Step 17
Linear mixed-effects analysis of dual-manifold ECG trajectories.

SCIENTIFIC QUESTION
-------------------
For each frozen latent coordinate / manifold metric:

    Does the HCM electrical phenotype show a systematic population-level
    temporal trend, and how heterogeneous are patient-specific trajectories?

Primary model:

    y_ij = beta_0 + beta_1 * t_ij
           + b_0i + b_1i * t_ij
           + epsilon_ij

where:
    t_ij = years from the subject's first VALID ECG
    beta_1 = population-average annual change
    b_1i = subject-specific deviation from the population slope

PRIMARY POPULATION
------------------
Subjects with:
    - valid_for_latent_projection == True
    - >= 2 valid visits
    - >= 2 distinct valid time points

SENSITIVITY POPULATION
----------------------
Subjects with:
    - >= 3 valid visits
    - >= 2 distinct valid time points

PRIMARY OUTCOMES
----------------
16 Normal-anchored latent coordinates:
    N_z_00 ... N_z_15

8 HCM-oriented latent coordinates:
    H_z_00 ... H_z_07

Global manifold metrics:
    d_N
    d_HCM
    RE_N_mse
    RE_H_mse

Thus, 28 primary longitudinal tests are FDR-corrected together.

SECONDARY / DESCRIPTIVE OUTCOMES
--------------------------------
If available:
    Delta_RE_mse
    log_RE_N_over_RE_H
    RE_N_mse_QRS
    RE_N_mse_STT
    RE_H_mse_QRS
    RE_H_mse_STT
    heart_rate_bpm
    rr_median_ms
    morph_rms_to_median_mv
    morph_rms_relative_to_template
    dominant_cluster_fraction

No multiplicity claim is attached to these exploratory outcomes.

IMPORTANT
---------
- Both VAEs remain frozen.
- Time is continuous in years; visit number is not used as the longitudinal
  exposure.
- The primary mixed model is linear. Non-linearity is not introduced before
  establishing whether a reproducible linear temporal component exists.
- Patient-specific BLUP slopes are exported when the random-slope model
  converges.
- Independent subject-wise OLS slopes and first-to-last annualized changes
  are also exported as descriptive, model-independent trajectory summaries.
- RE_N and RE_H are not directly interchangeable measures of disease
  specificity; Delta_RE remains descriptive.
- A random-intercept fallback is allowed only when the random-slope model
  fails. Such outcomes are explicitly flagged and are NOT considered to
  provide mixed-model patient-specific slope estimates.

EXPECTED INPUT
--------------
Results/16_Longitudinal_DualManifold/longitudinal_dual_manifold.csv

OUTPUTS
-------
Results/17_Longitudinal_Trajectories/
    mixed_effects_primary.csv
    mixed_effects_sensitivity_3plus.csv
    mixed_effects_secondary.csv
    patient_specific_blup_slopes.csv
    patient_empirical_slopes.csv
    patient_trajectory_summary.csv
    fixed_effect_direction_summary.csv
    run_summary.json
    figures/
        fixed_slopes_normal_latents.png
        fixed_slopes_hcm_latents.png
        fixed_slopes_global_metrics.png
        hcm_latent_patient_slope_distributions.png
        selected_hcm_trajectories.png

RUN
---
python Scripts\\17_analyze_longitudinal_dual_manifold_trajectories.py

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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.formula.api as smf


SEED = 20261005
TIME_COL = "time_from_first_valid_ecg_years"
SUBJECT_COL = "analysis_subject_key"

NORMAL_LATENTS = [f"N_z_{k:02d}" for k in range(16)]
HCM_LATENTS = [f"H_z_{k:02d}" for k in range(8)]

PRIMARY_GLOBAL = [
    "d_N",
    "d_HCM",
    "RE_N_mse",
    "RE_H_mse",
]

SECONDARY_CANDIDATES = [
    "Delta_RE_mse",
    "log_RE_N_over_RE_H",
    "RE_N_mse_preQRS",
    "RE_N_mse_QRS",
    "RE_N_mse_STT",
    "RE_H_mse_preQRS",
    "RE_H_mse_QRS",
    "RE_H_mse_STT",
    "RE_N_STT_to_QRS_mse_ratio",
    "RE_H_STT_to_QRS_mse_ratio",
    "heart_rate_bpm",
    "rr_median_ms",
    "rr_sd_ms",
    "rr_cv",
    "rmssd_ms",
    "morph_pairwise_similarity_median",
    "morph_rms_to_median_mv",
    "morph_rms_relative_to_template",
    "dominant_cluster_fraction",
]

PRIORITY_HCM_FOR_TRAJECTORY_PLOT = [
    "H_z_02",
    "H_z_05",
    "H_z_06",
    "H_z_07",
]


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "Results/16_Longitudinal_DualManifold/"
            "longitudinal_dual_manifold.csv"
        ),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "Results/17_Longitudinal_Trajectories"
        ),
    )
    p.add_argument(
        "--min-visits-primary",
        type=int,
        default=2,
    )
    p.add_argument(
        "--min-visits-sensitivity",
        type=int,
        default=3,
    )
    p.add_argument(
        "--reml",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--ml",
        dest="reml",
        action="store_false",
        help="Use maximum likelihood instead of REML.",
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=2000,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    p.add_argument(
        "--trajectory-plot-subjects",
        type=int,
        default=60,
    )
    return p.parse_args()


# =====================================================================
# Utilities
# =====================================================================

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


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_sd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    sd = float(
        np.std(
            x,
            ddof=1,
        )
    )

    return (
        sd
        if sd > 0
        else np.nan
    )


def prepare_input(df):
    required = [
        SUBJECT_COL,
        TIME_COL,
        "valid_for_latent_projection",
    ]

    for c in required:
        if c not in df.columns:
            raise KeyError(
                f"Input is missing required column: {c}"
            )

    valid = df[
        df["valid_for_latent_projection"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    ].copy()

    valid[TIME_COL] = safe_numeric(
        valid[TIME_COL]
    )

    valid = valid[
        np.isfinite(
            valid[TIME_COL]
        )
    ].copy()

    return valid


def subject_eligibility(
    valid_df,
    min_visits,
):
    rows = []

    for subject, g in valid_df.groupby(
        SUBJECT_COL,
        sort=False,
    ):
        t = safe_numeric(
            g[TIME_COL]
        ).to_numpy(
            dtype=float
        )

        finite_t = t[
            np.isfinite(t)
        ]

        rows.append({
            SUBJECT_COL: subject,
            "n_valid_visits": int(
                len(g)
            ),
            "n_distinct_times": int(
                len(
                    np.unique(
                        np.round(
                            finite_t,
                            10,
                        )
                    )
                )
            ),
            "followup_years": (
                float(
                    np.max(finite_t)
                    - np.min(finite_t)
                )
                if len(finite_t)
                else np.nan
            ),
        })

    s = pd.DataFrame(
        rows
    )

    s["eligible"] = (
        (s["n_valid_visits"] >= min_visits)
        & (s["n_distinct_times"] >= 2)
        & (s["followup_years"] > 0)
    )

    return s


def subset_eligible(
    valid_df,
    eligibility,
):
    keep = set(
        eligibility.loc[
            eligibility["eligible"],
            SUBJECT_COL,
        ].astype(str)
    )

    return valid_df[
        valid_df[SUBJECT_COL]
        .astype(str)
        .isin(keep)
    ].copy()


def outcome_baseline_sd(
    df,
    outcome,
):
    first = (
        df.sort_values(
            [
                SUBJECT_COL,
                TIME_COL,
            ]
        )
        .groupby(
            SUBJECT_COL,
            as_index=False,
        )
        .first()
    )

    return safe_sd(
        safe_numeric(
            first[outcome]
        )
    )


def count_outcome_data(
    df,
    outcome,
):
    d = df[
        [
            SUBJECT_COL,
            TIME_COL,
            outcome,
        ]
    ].copy()

    d[outcome] = safe_numeric(
        d[outcome]
    )

    d = d[
        np.isfinite(d[outcome])
        & np.isfinite(d[TIME_COL])
    ].copy()

    n_obs = int(
        len(d)
    )
    n_subjects = int(
        d[SUBJECT_COL].nunique()
    )

    return d, n_obs, n_subjects


# =====================================================================
# Mixed-effects model
# =====================================================================

def extract_random_effect_components(
    result,
    time_name,
):
    """
    Return per-subject random intercept and random slope.

    statsmodels may label the random-intercept entry as Group,
    Intercept, or the grouping variable. We detect the slope by
    exact match to the time variable where possible.
    """
    rows = []

    for subject, re in result.random_effects.items():
        if hasattr(
            re,
            "index",
        ):
            idx = [
                str(x)
                for x in re.index
            ]
            vals = np.asarray(
                re,
                dtype=float,
            )

            slope = np.nan
            intercept = np.nan

            for name, value in zip(
                idx,
                vals,
            ):
                if name == time_name:
                    slope = float(
                        value
                    )
                elif (
                    name.lower()
                    in {
                        "group",
                        "intercept",
                    }
                    or "intercept"
                    in name.lower()
                ):
                    intercept = float(
                        value
                    )

            if not np.isfinite(
                intercept
            ):
                nonslope = [
                    float(v)
                    for name, v in zip(
                        idx,
                        vals,
                    )
                    if name != time_name
                ]

                if nonslope:
                    intercept = nonslope[0]

        else:
            arr = np.asarray(
                re,
                dtype=float,
            ).ravel()

            intercept = (
                float(arr[0])
                if len(arr) >= 1
                else np.nan
            )

            slope = (
                float(arr[1])
                if len(arr) >= 2
                else np.nan
            )

        rows.append({
            SUBJECT_COL: str(subject),
            "random_intercept": intercept,
            "random_slope": slope,
        })

    return pd.DataFrame(
        rows
    )


def fit_mixed_model(
    df,
    outcome,
    maxiter,
    reml,
):
    """
    Primary attempt:
        random intercept + random slope.

    Fallback:
        random intercept only.

    The fallback is explicitly flagged and is not used to claim
    patient-specific mixed-model slopes.
    """
    d, n_obs, n_subjects = (
        count_outcome_data(
            df,
            outcome,
        )
    )

    if (
        n_obs < 20
        or n_subjects < 10
    ):
        return {
            "success": False,
            "outcome": outcome,
            "model_structure": "not_fit",
            "failure_reason": "insufficient_data",
            "n_obs": n_obs,
            "n_subjects": n_subjects,
            "random_effects": pd.DataFrame(),
        }

    # Ensure each retained subject contributes at least 2 usable
    # outcome observations at distinct times.
    keep_subjects = []

    for subject, g in d.groupby(
        SUBJECT_COL,
    ):
        if (
            len(g) >= 2
            and g[TIME_COL].nunique() >= 2
        ):
            keep_subjects.append(
                subject
            )

    d = d[
        d[SUBJECT_COL].isin(
            keep_subjects
        )
    ].copy()

    n_obs = int(
        len(d)
    )
    n_subjects = int(
        d[SUBJECT_COL].nunique()
    )

    if n_subjects < 10:
        return {
            "success": False,
            "outcome": outcome,
            "model_structure": "not_fit",
            "failure_reason": (
                "insufficient_subjects_after_outcome_filter"
            ),
            "n_obs": n_obs,
            "n_subjects": n_subjects,
            "random_effects": pd.DataFrame(),
        }

    formula = (
        f"{outcome} ~ {TIME_COL}"
    )

    attempts = []

    # Random intercept + random slope.
    for method in [
        "lbfgs",
        "powell",
        "cg",
    ]:
        try:
            with warnings.catch_warnings(
                record=True
            ) as caught:
                warnings.simplefilter(
                    "always"
                )

                model = smf.mixedlm(
                    formula,
                    data=d,
                    groups=d[
                        SUBJECT_COL
                    ],
                    re_formula=(
                        f"~{TIME_COL}"
                    ),
                )

                result = model.fit(
                    reml=reml,
                    method=method,
                    maxiter=maxiter,
                    disp=False,
                )

            attempts.append({
                "structure": "random_intercept_slope",
                "method": method,
                "converged": bool(
                    getattr(
                        result,
                        "converged",
                        False,
                    )
                ),
                "warnings": " | ".join(
                    str(w.message)
                    for w in caught
                ),
            })

            if bool(
                getattr(
                    result,
                    "converged",
                    False,
                )
            ):
                return summarize_mixed_result(
                    result=result,
                    data=d,
                    outcome=outcome,
                    model_structure=(
                        "random_intercept_slope"
                    ),
                    optimizer=method,
                    attempts=attempts,
                )

        except Exception as exc:
            attempts.append({
                "structure": "random_intercept_slope",
                "method": method,
                "converged": False,
                "warnings": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    # Explicit fallback: random intercept only.
    for method in [
        "lbfgs",
        "powell",
    ]:
        try:
            with warnings.catch_warnings(
                record=True
            ) as caught:
                warnings.simplefilter(
                    "always"
                )

                model = smf.mixedlm(
                    formula,
                    data=d,
                    groups=d[
                        SUBJECT_COL
                    ],
                    re_formula="1",
                )

                result = model.fit(
                    reml=reml,
                    method=method,
                    maxiter=maxiter,
                    disp=False,
                )

            attempts.append({
                "structure": "random_intercept_only",
                "method": method,
                "converged": bool(
                    getattr(
                        result,
                        "converged",
                        False,
                    )
                ),
                "warnings": " | ".join(
                    str(w.message)
                    for w in caught
                ),
            })

            if bool(
                getattr(
                    result,
                    "converged",
                    False,
                )
            ):
                return summarize_mixed_result(
                    result=result,
                    data=d,
                    outcome=outcome,
                    model_structure=(
                        "random_intercept_only"
                    ),
                    optimizer=method,
                    attempts=attempts,
                )

        except Exception as exc:
            attempts.append({
                "structure": "random_intercept_only",
                "method": method,
                "converged": False,
                "warnings": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    return {
        "success": False,
        "outcome": outcome,
        "model_structure": "failed",
        "failure_reason": json.dumps(
            attempts
        ),
        "n_obs": n_obs,
        "n_subjects": n_subjects,
        "random_effects": pd.DataFrame(),
    }


def summarize_mixed_result(
    result,
    data,
    outcome,
    model_structure,
    optimizer,
    attempts,
):
    beta0 = float(
        result.params.get(
            "Intercept",
            np.nan,
        )
    )

    beta1 = float(
        result.params.get(
            TIME_COL,
            np.nan,
        )
    )

    se = float(
        result.bse.get(
            TIME_COL,
            np.nan,
        )
    )

    p = float(
        result.pvalues.get(
            TIME_COL,
            np.nan,
        )
    )

    try:
        ci = result.conf_int().loc[
            TIME_COL
        ]
        ci_low = float(
            ci.iloc[0]
        )
        ci_high = float(
            ci.iloc[1]
        )
    except Exception:
        ci_low = (
            beta1 - 1.96 * se
            if np.isfinite(se)
            else np.nan
        )
        ci_high = (
            beta1 + 1.96 * se
            if np.isfinite(se)
            else np.nan
        )

    baseline_sd = outcome_baseline_sd(
        data,
        outcome,
    )

    standardized_slope = (
        beta1 / baseline_sd
        if (
            np.isfinite(
                baseline_sd
            )
            and baseline_sd > 0
        )
        else np.nan
    )

    cov_re = np.asarray(
        result.cov_re,
        dtype=float,
    )

    random_intercept_var = (
        float(
            cov_re[0, 0]
        )
        if (
            cov_re.ndim == 2
            and cov_re.shape[0] >= 1
        )
        else np.nan
    )

    random_slope_var = (
        float(
            cov_re[1, 1]
        )
        if (
            model_structure
            == "random_intercept_slope"
            and cov_re.ndim == 2
            and cov_re.shape[0] >= 2
        )
        else np.nan
    )

    random_intercept_slope_cov = (
        float(
            cov_re[0, 1]
        )
        if (
            model_structure
            == "random_intercept_slope"
            and cov_re.ndim == 2
            and cov_re.shape[0] >= 2
        )
        else np.nan
    )

    residual_var = float(
        result.scale
    )

    icc_at_baseline = (
        random_intercept_var
        / (
            random_intercept_var
            + residual_var
        )
        if (
            np.isfinite(
                random_intercept_var
            )
            and random_intercept_var >= 0
            and residual_var >= 0
            and (
                random_intercept_var
                + residual_var
            ) > 0
        )
        else np.nan
    )

    try:
        llf = float(
            result.llf
        )
        aic = float(
            result.aic
        )
        bic = float(
            result.bic
        )
    except Exception:
        llf = np.nan
        aic = np.nan
        bic = np.nan

    if (
        model_structure
        == "random_intercept_slope"
    ):
        re_df = (
            extract_random_effect_components(
                result,
                TIME_COL,
            )
        )

        re_df[
            "population_fixed_slope"
        ] = beta1

        re_df[
            "patient_blup_slope"
        ] = (
            beta1
            + re_df[
                "random_slope"
            ]
        )

        re_df[
            "population_fixed_intercept"
        ] = beta0

        re_df[
            "patient_blup_intercept"
        ] = (
            beta0
            + re_df[
                "random_intercept"
            ]
        )

    else:
        re_df = pd.DataFrame()

    return {
        "success": True,
        "outcome": outcome,
        "model_structure": model_structure,
        "optimizer": optimizer,
        "converged": bool(
            getattr(
                result,
                "converged",
                False,
            )
        ),
        "n_obs": int(
            len(data)
        ),
        "n_subjects": int(
            data[
                SUBJECT_COL
            ].nunique()
        ),
        "fixed_intercept": beta0,
        "fixed_slope_per_year": beta1,
        "fixed_slope_se": se,
        "fixed_slope_ci95_low": ci_low,
        "fixed_slope_ci95_high": ci_high,
        "fixed_slope_p": p,
        "baseline_between_subject_sd": baseline_sd,
        "fixed_slope_baseline_sd_per_year": standardized_slope,
        "random_intercept_variance": random_intercept_var,
        "random_slope_variance": random_slope_var,
        "random_slope_sd": (
            math.sqrt(
                max(
                    random_slope_var,
                    0.0,
                )
            )
            if np.isfinite(
                random_slope_var
            )
            else np.nan
        ),
        "random_intercept_slope_covariance": random_intercept_slope_cov,
        "residual_variance": residual_var,
        "icc_at_baseline": icc_at_baseline,
        "log_likelihood": llf,
        "aic": aic,
        "bic": bic,
        "fit_attempts": json.dumps(
            attempts
        ),
        "random_effects": re_df,
    }


def fit_outcome_family(
    df,
    outcomes,
    cohort_name,
    maxiter,
    reml,
):
    rows = []
    blups = []

    for i, outcome in enumerate(
        outcomes,
        start=1,
    ):
        print(
            f"[{cohort_name}] "
            f"{i:02d}/{len(outcomes):02d} "
            f"{outcome}"
        )

        res = fit_mixed_model(
            df=df,
            outcome=outcome,
            maxiter=maxiter,
            reml=reml,
        )

        row = {
            k: v
            for k, v in res.items()
            if k != "random_effects"
        }

        row["cohort"] = (
            cohort_name
        )

        rows.append(row)

        if (
            res.get(
                "success",
                False,
            )
            and len(
                res["random_effects"]
            )
        ):
            x = (
                res[
                    "random_effects"
                ].copy()
            )

            x["outcome"] = (
                outcome
            )

            x["cohort"] = (
                cohort_name
            )

            blups.append(
                x
            )

    summary = pd.DataFrame(
        rows
    )

    blup_df = (
        pd.concat(
            blups,
            ignore_index=True,
        )
        if blups
        else pd.DataFrame()
    )

    return (
        summary,
        blup_df,
    )


# =====================================================================
# Empirical per-patient trajectories
# =====================================================================

def empirical_subject_slopes(
    df,
    outcomes,
):
    rows = []

    for subject, g in df.groupby(
        SUBJECT_COL,
        sort=False,
    ):
        g = g.sort_values(
            TIME_COL
        )

        for outcome in outcomes:
            y = safe_numeric(
                g[outcome]
            ).to_numpy(
                dtype=float
            )

            t = safe_numeric(
                g[TIME_COL]
            ).to_numpy(
                dtype=float
            )

            m = (
                np.isfinite(y)
                & np.isfinite(t)
            )

            yy = y[m]
            tt = t[m]

            if (
                len(yy) < 2
                or len(
                    np.unique(tt)
                ) < 2
            ):
                continue

            lr = stats.linregress(
                tt,
                yy,
            )

            first_idx = int(
                np.argmin(tt)
            )
            last_idx = int(
                np.argmax(tt)
            )

            dt = float(
                tt[last_idx]
                - tt[first_idx]
            )

            first_last_slope = (
                float(
                    (
                        yy[last_idx]
                        - yy[first_idx]
                    )
                    / dt
                )
                if dt > 0
                else np.nan
            )

            rows.append({
                SUBJECT_COL: subject,
                "outcome": outcome,
                "n_visits": int(
                    len(yy)
                ),
                "followup_years": dt,
                "ols_intercept": float(
                    lr.intercept
                ),
                "ols_slope_per_year": float(
                    lr.slope
                ),
                "ols_r_squared": float(
                    lr.rvalue ** 2
                ),
                "first_value": float(
                    yy[first_idx]
                ),
                "last_value": float(
                    yy[last_idx]
                ),
                "first_to_last_slope_per_year": (
                    first_last_slope
                ),
            })

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Multiplicity and summaries
# =====================================================================

def add_primary_fdr(df):
    out = df.copy()

    p = safe_numeric(
        out[
            "fixed_slope_p"
        ]
    ).to_numpy(
        dtype=float
    )

    out[
        "fixed_slope_q_fdr"
    ] = bh_fdr(
        p
    )

    out[
        "fixed_slope_significant_fdr05"
    ] = (
        out[
            "fixed_slope_q_fdr"
        ] < 0.05
    )

    out[
        "fixed_slope_direction"
    ] = np.where(
        safe_numeric(
            out[
                "fixed_slope_per_year"
            ]
        ) > 0,
        "increase",
        np.where(
            safe_numeric(
                out[
                    "fixed_slope_per_year"
                ]
            ) < 0,
            "decrease",
            "flat",
        ),
    )

    return out


def classify_outcome(outcome):
    if outcome.startswith(
        "N_z_"
    ):
        return "normal_latent"

    if outcome.startswith(
        "H_z_"
    ):
        return "hcm_latent"

    return "global_metric"


def direction_summary(
    primary,
    blups,
    empirical,
):
    rows = []

    for _, r in primary.iterrows():
        outcome = r[
            "outcome"
        ]

        b = blups[
            blups["outcome"]
            == outcome
        ].copy()

        e = empirical[
            empirical["outcome"]
            == outcome
        ].copy()

        row = {
            "outcome": outcome,
            "outcome_family": classify_outcome(
                outcome
            ),
            "fixed_slope_per_year": r.get(
                "fixed_slope_per_year",
                np.nan,
            ),
            "fixed_slope_q_fdr": r.get(
                "fixed_slope_q_fdr",
                np.nan,
            ),
            "model_structure": r.get(
                "model_structure",
                "",
            ),
        }

        if len(b):
            slopes = safe_numeric(
                b[
                    "patient_blup_slope"
                ]
            ).to_numpy(
                dtype=float
            )

            slopes = slopes[
                np.isfinite(
                    slopes
                )
            ]

            if len(slopes):
                row.update({
                    "blup_slope_median": float(
                        np.median(
                            slopes
                        )
                    ),
                    "blup_slope_iqr_low": float(
                        np.percentile(
                            slopes,
                            25,
                        )
                    ),
                    "blup_slope_iqr_high": float(
                        np.percentile(
                            slopes,
                            75,
                        )
                    ),
                    "blup_fraction_positive": float(
                        np.mean(
                            slopes > 0
                        )
                    ),
                    "blup_fraction_negative": float(
                        np.mean(
                            slopes < 0
                        )
                    ),
                })

        if len(e):
            slopes = safe_numeric(
                e[
                    "ols_slope_per_year"
                ]
            ).to_numpy(
                dtype=float
            )

            slopes = slopes[
                np.isfinite(
                    slopes
                )
            ]

            if len(slopes):
                row.update({
                    "empirical_slope_median": float(
                        np.median(
                            slopes
                        )
                    ),
                    "empirical_slope_iqr_low": float(
                        np.percentile(
                            slopes,
                            25,
                        )
                    ),
                    "empirical_slope_iqr_high": float(
                        np.percentile(
                            slopes,
                            75,
                        )
                    ),
                    "empirical_fraction_positive": float(
                        np.mean(
                            slopes > 0
                        )
                    ),
                    "empirical_fraction_negative": float(
                        np.mean(
                            slopes < 0
                        )
                    ),
                })

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Figures
# =====================================================================

def plot_fixed_slopes(
    table,
    outcomes,
    title,
    path,
):
    d = (
        table[
            table["outcome"]
            .isin(
                outcomes
            )
        ]
        .copy()
        .set_index(
            "outcome"
        )
        .reindex(
            outcomes
        )
        .reset_index()
    )

    y = safe_numeric(
        d[
            "fixed_slope_baseline_sd_per_year"
        ]
    ).to_numpy(
        dtype=float
    )

    lo_raw = safe_numeric(
        d[
            "fixed_slope_ci95_low"
        ]
    ).to_numpy(
        dtype=float
    )

    hi_raw = safe_numeric(
        d[
            "fixed_slope_ci95_high"
        ]
    ).to_numpy(
        dtype=float
    )

    baseline_sd = safe_numeric(
        d[
            "baseline_between_subject_sd"
        ]
    ).to_numpy(
        dtype=float
    )

    lo = (
        lo_raw
        / baseline_sd
    )
    hi = (
        hi_raw
        / baseline_sd
    )

    lower_err = (
        y - lo
    )
    upper_err = (
        hi - y
    )

    x = np.arange(
        len(d)
    )

    fig, ax = plt.subplots(
        figsize=(
            max(
                8,
                0.65 * len(d),
            ),
            5,
        )
    )

    ax.errorbar(
        x,
        y,
        yerr=np.vstack(
            [
                lower_err,
                upper_err,
            ]
        ),
        fmt="o",
        capsize=3,
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    sig = (
        safe_numeric(
            d[
                "fixed_slope_q_fdr"
            ]
        ).to_numpy(
            dtype=float
        )
        < 0.05
    )

    for i, yes in enumerate(
        sig
    ):
        if yes and np.isfinite(
            y[i]
        ):
            ax.text(
                i,
                y[i],
                "*",
                ha="center",
                va="bottom",
            )

    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        d["outcome"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Annual change (baseline SD/year)"
    )
    ax.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_hcm_slope_distributions(
    blups,
    path,
):
    available = [
        z
        for z in HCM_LATENTS
        if z in set(
            blups["outcome"]
        )
    ]

    if not available:
        return

    data = []

    for z in available:
        vals = safe_numeric(
            blups.loc[
                blups["outcome"] == z,
                "patient_blup_slope",
            ]
        ).dropna().to_numpy()

        data.append(
            vals
        )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    # Matplotlib >=3.9 uses ``tick_labels``; older versions use ``labels``.
    # Try the current API first and fall back for backward compatibility.
    try:
        ax.boxplot(
            data,
            tick_labels=available,
            showfliers=False,
        )
    except TypeError:
        ax.boxplot(
            data,
            labels=available,
            showfliers=False,
        )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_ylabel(
        "Patient-specific BLUP slope / year"
    )

    ax.set_title(
        "HCM-oriented latent trajectory distributions"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_selected_hcm_trajectories(
    df,
    outcomes,
    path,
    max_subjects,
    seed,
):
    outcomes = [
        o
        for o in outcomes
        if o in df.columns
    ]

    if not outcomes:
        return

    subjects = np.asarray(
        sorted(
            df[
                SUBJECT_COL
            ].astype(str).unique()
        ),
        dtype=object,
    )

    rng = np.random.default_rng(
        seed
    )

    if len(subjects) > max_subjects:
        subjects = rng.choice(
            subjects,
            size=max_subjects,
            replace=False,
        )

    fig, axes = plt.subplots(
        len(outcomes),
        1,
        figsize=(
            10,
            3.2 * len(outcomes),
        ),
        sharex=True,
    )

    if len(outcomes) == 1:
        axes = [
            axes
        ]

    for ax, outcome in zip(
        axes,
        outcomes,
    ):
        for subject in subjects:
            g = df[
                df[
                    SUBJECT_COL
                ].astype(str)
                == str(subject)
            ].sort_values(
                TIME_COL
            )

            t = safe_numeric(
                g[
                    TIME_COL
                ]
            ).to_numpy(
                dtype=float
            )

            y = safe_numeric(
                g[
                    outcome
                ]
            ).to_numpy(
                dtype=float
            )

            m = (
                np.isfinite(t)
                & np.isfinite(y)
            )

            if np.sum(m) >= 2:
                ax.plot(
                    t[m],
                    y[m],
                    linewidth=0.7,
                    alpha=0.25,
                )

        ax.set_ylabel(
            outcome
        )
        ax.axhline(
            0,
            linewidth=0.5,
            alpha=0.4,
        )

    axes[-1].set_xlabel(
        "Years from first valid ECG"
    )

    fig.suptitle(
        "Representative individual HCM-oriented trajectories"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()

    np.random.seed(
        args.seed
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig_dir = (
        args.results_dir
        / "figures"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input not found: {args.input}"
        )

    print(
        "=" * 78
    )
    print(
        "17 - Longitudinal dual-manifold trajectories"
    )
    print(
        "=" * 78
    )
    print(
        "Input   :",
        args.input,
    )
    print(
        "Results :",
        args.results_dir,
    )
    print(
        "REML    :",
        args.reml,
    )
    print()

    raw = pd.read_csv(
        args.input
    )

    valid = prepare_input(
        raw
    )

    primary_elig = (
        subject_eligibility(
            valid,
            args.min_visits_primary,
        )
    )

    sensitivity_elig = (
        subject_eligibility(
            valid,
            args.min_visits_sensitivity,
        )
    )

    primary_df = subset_eligible(
        valid,
        primary_elig,
    )

    sensitivity_df = subset_eligible(
        valid,
        sensitivity_elig,
    )

    primary_outcomes = (
        NORMAL_LATENTS
        + HCM_LATENTS
        + PRIMARY_GLOBAL
    )

    missing_primary = [
        c
        for c in primary_outcomes
        if c not in raw.columns
    ]

    if missing_primary:
        raise KeyError(
            "Missing primary longitudinal columns: "
            + ", ".join(
                missing_primary
            )
        )

    secondary_outcomes = [
        c
        for c in SECONDARY_CANDIDATES
        if c in raw.columns
    ]

    print(
        "[cohorts]"
    )
    print(
        "Valid visits total        :",
        len(valid),
    )
    print(
        "Primary subjects >=2      :",
        primary_df[
            SUBJECT_COL
        ].nunique(),
    )
    print(
        "Primary visits            :",
        len(primary_df),
    )
    print(
        "Sensitivity subjects >=3  :",
        sensitivity_df[
            SUBJECT_COL
        ].nunique(),
    )
    print(
        "Sensitivity visits        :",
        len(sensitivity_df),
    )
    print()

    # -------------------------------------------------------------
    # Primary mixed models
    # -------------------------------------------------------------
    primary_results, primary_blups = (
        fit_outcome_family(
            primary_df,
            primary_outcomes,
            "primary_2plus",
            args.maxiter,
            args.reml,
        )
    )

    primary_results = (
        add_primary_fdr(
            primary_results
        )
    )

    primary_results[
        "outcome_family"
    ] = primary_results[
        "outcome"
    ].map(
        classify_outcome
    )

    primary_results.to_csv(
        args.results_dir
        / "mixed_effects_primary.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # >=3-visit sensitivity
    # -------------------------------------------------------------
    sensitivity_results, sensitivity_blups = (
        fit_outcome_family(
            sensitivity_df,
            primary_outcomes,
            "sensitivity_3plus",
            args.maxiter,
            args.reml,
        )
    )

    sensitivity_results = (
        add_primary_fdr(
            sensitivity_results
        )
    )

    sensitivity_results[
        "outcome_family"
    ] = sensitivity_results[
        "outcome"
    ].map(
        classify_outcome
    )

    sensitivity_results.to_csv(
        args.results_dir
        / "mixed_effects_sensitivity_3plus.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Secondary outcomes
    # -------------------------------------------------------------
    if secondary_outcomes:
        secondary_results, secondary_blups = (
            fit_outcome_family(
                primary_df,
                secondary_outcomes,
                "secondary_2plus",
                args.maxiter,
                args.reml,
            )
        )

        # FDR provided for reference only within this exploratory family.
        secondary_results[
            "fixed_slope_q_fdr_exploratory_family"
        ] = bh_fdr(
            safe_numeric(
                secondary_results[
                    "fixed_slope_p"
                ]
            ).to_numpy(
                dtype=float
            )
        )

        secondary_results.to_csv(
            args.results_dir
            / "mixed_effects_secondary.csv",
            index=False,
        )
    else:
        secondary_results = pd.DataFrame()
        secondary_blups = pd.DataFrame()

    # -------------------------------------------------------------
    # Patient-specific BLUPs
    # -------------------------------------------------------------
    all_blups = []

    if len(
        primary_blups
    ):
        all_blups.append(
            primary_blups
        )

    if len(
        sensitivity_blups
    ):
        all_blups.append(
            sensitivity_blups
        )

    if len(
        secondary_blups
    ):
        all_blups.append(
            secondary_blups
        )

    blup_df = (
        pd.concat(
            all_blups,
            ignore_index=True,
        )
        if all_blups
        else pd.DataFrame()
    )

    blup_df.to_csv(
        args.results_dir
        / "patient_specific_blup_slopes.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Empirical patient slopes
    # -------------------------------------------------------------
    empirical = (
        empirical_subject_slopes(
            primary_df,
            (
                primary_outcomes
                + secondary_outcomes
            ),
        )
    )

    empirical.to_csv(
        args.results_dir
        / "patient_empirical_slopes.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Subject-level trajectory eligibility summary
    # -------------------------------------------------------------
    subject_summary = (
        primary_elig.merge(
            sensitivity_elig[
                [
                    SUBJECT_COL,
                    "eligible",
                ]
            ].rename(
                columns={
                    "eligible": (
                        "eligible_3plus"
                    )
                }
            ),
            on=SUBJECT_COL,
            how="left",
        )
    )

    subject_summary = (
        subject_summary.rename(
            columns={
                "eligible": (
                    "eligible_primary_2plus"
                )
            }
        )
    )

    subject_summary.to_csv(
        args.results_dir
        / "patient_trajectory_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Direction summary
    # -------------------------------------------------------------
    primary_blups_only = (
        primary_blups
        if len(
            primary_blups
        )
        else pd.DataFrame(
            columns=[
                "outcome",
                "patient_blup_slope",
            ]
        )
    )

    direction = direction_summary(
        primary_results,
        primary_blups_only,
        empirical,
    )

    direction.to_csv(
        args.results_dir
        / "fixed_effect_direction_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------
    plot_fixed_slopes(
        primary_results,
        NORMAL_LATENTS,
        "Normal-anchored latent population slopes",
        fig_dir
        / "fixed_slopes_normal_latents.png",
    )

    plot_fixed_slopes(
        primary_results,
        HCM_LATENTS,
        "HCM-oriented latent population slopes",
        fig_dir
        / "fixed_slopes_hcm_latents.png",
    )

    plot_fixed_slopes(
        primary_results,
        PRIMARY_GLOBAL,
        "Global manifold metric population slopes",
        fig_dir
        / "fixed_slopes_global_metrics.png",
    )

    if len(
        primary_blups_only
    ):
        plot_hcm_slope_distributions(
            primary_blups_only,
            fig_dir
            / "hcm_latent_patient_slope_distributions.png",
        )

    plot_selected_hcm_trajectories(
        primary_df,
        PRIORITY_HCM_FOR_TRAJECTORY_PLOT,
        fig_dir
        / "selected_hcm_trajectories.png",
        args.trajectory_plot_subjects,
        args.seed,
    )

    # -------------------------------------------------------------
    # Run summary
    # -------------------------------------------------------------
    converged_primary = primary_results[
        primary_results[
            "success"
        ] == True
    ]

    random_slope_primary = primary_results[
        primary_results[
            "model_structure"
        ]
        == "random_intercept_slope"
    ]

    sig = primary_results[
        primary_results[
            "fixed_slope_q_fdr"
        ] < 0.05
    ].copy()

    significant_rows = []

    for _, r in sig.iterrows():
        significant_rows.append({
            "outcome": r["outcome"],
            "family": r[
                "outcome_family"
            ],
            "slope_per_year": (
                float(
                    r[
                        "fixed_slope_per_year"
                    ]
                )
            ),
            "slope_baseline_sd_per_year": (
                float(
                    r[
                        "fixed_slope_baseline_sd_per_year"
                    ]
                )
                if np.isfinite(
                    r[
                        "fixed_slope_baseline_sd_per_year"
                    ]
                )
                else None
            ),
            "p": float(
                r[
                    "fixed_slope_p"
                ]
            ),
            "q_fdr": float(
                r[
                    "fixed_slope_q_fdr"
                ]
            ),
            "model_structure": r[
                "model_structure"
            ],
        })

    sensitivity_map = (
        sensitivity_results[
            [
                "outcome",
                "fixed_slope_per_year",
                "fixed_slope_q_fdr",
                "model_structure",
            ]
        ]
        .rename(
            columns={
                "fixed_slope_per_year": (
                    "sensitivity_slope"
                ),
                "fixed_slope_q_fdr": (
                    "sensitivity_q"
                ),
                "model_structure": (
                    "sensitivity_structure"
                ),
            }
        )
    )

    comparison = (
        primary_results[
            [
                "outcome",
                "fixed_slope_per_year",
                "fixed_slope_q_fdr",
                "model_structure",
            ]
        ]
        .merge(
            sensitivity_map,
            on="outcome",
            how="left",
        )
    )

    comparison[
        "same_slope_sign"
    ] = (
        np.sign(
            safe_numeric(
                comparison[
                    "fixed_slope_per_year"
                ]
            )
        )
        == np.sign(
            safe_numeric(
                comparison[
                    "sensitivity_slope"
                ]
            )
        )
    )

    n_same_sign = int(
        comparison[
            "same_slope_sign"
        ].sum()
    )

    summary = {
        "script": (
            "17_analyze_longitudinal_dual_manifold_trajectories.py"
        ),
        "version": "1.1",
        "scientific_role": (
            "Population-average and patient-specific longitudinal trajectory "
            "analysis in the frozen Normal-anchored and HCM-oriented ECG spaces"
        ),
        "input": str(
            args.input
        ),
        "time_variable": TIME_COL,
        "time_unit": "years",
        "primary_model": (
            "linear mixed-effects model: outcome ~ time with subject-specific "
            "random intercept and random slope"
        ),
        "random_intercept_fallback_allowed": True,
        "estimation": (
            "REML"
            if args.reml
            else "ML"
        ),
        "primary_population": {
            "minimum_valid_visits": int(
                args.min_visits_primary
            ),
            "n_subjects": int(
                primary_df[
                    SUBJECT_COL
                ].nunique()
            ),
            "n_visits": int(
                len(
                    primary_df
                )
            ),
        },
        "sensitivity_population": {
            "minimum_valid_visits": int(
                args.min_visits_sensitivity
            ),
            "n_subjects": int(
                sensitivity_df[
                    SUBJECT_COL
                ].nunique()
            ),
            "n_visits": int(
                len(
                    sensitivity_df
                )
            ),
        },
        "primary_outcomes": (
            primary_outcomes
        ),
        "n_primary_tests": int(
            len(
                primary_outcomes
            )
        ),
        "multiplicity": (
            "Benjamini-Hochberg FDR across all 28 primary outcomes"
        ),
        "n_primary_models_successful": int(
            len(
                converged_primary
            )
        ),
        "n_primary_random_intercept_slope_models": int(
            len(
                random_slope_primary
            )
        ),
        "n_primary_fdr_significant": int(
            len(
                sig
            )
        ),
        "primary_fdr_significant_outcomes": (
            significant_rows
        ),
        "sensitivity_same_slope_sign": {
            "n_same_sign": n_same_sign,
            "n_total": int(
                len(
                    comparison
                )
            ),
            "fraction": float(
                n_same_sign
                / len(
                    comparison
                )
            )
            if len(
                comparison
            )
            else None,
        },
        "secondary_outcomes": (
            secondary_outcomes
        ),
        "interpretive_notes": [
            "A significant fixed time slope indicates an average population-level longitudinal trend, not deterministic progression in every patient.",
            "Random-slope variance and patient BLUP slopes quantify heterogeneity around the population-average trajectory.",
            "Patient BLUP slopes are shrinkage estimates and should not be interpreted as independent raw subject slopes.",
            "Subject-wise OLS and first-to-last slopes are exported as descriptive model-independent complements.",
            "The >=3-visit analysis is a sensitivity analysis for trajectory stability.",
            "Latent coordinates are model-derived electrical phenotypes and are not direct biological mechanisms.",
            "RE_N and RE_H come from different frozen VAEs; Delta_RE remains descriptive.",
            "This step tests linear time trends only. Non-linear trajectories should be considered later only if residual patterns or clinical hypotheses justify them.",
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

    # -------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------
    print()
    print(
        "=" * 78
    )
    print(
        "STEP 17 COMPLETE"
    )
    print(
        "=" * 78
    )
    print(
        "Primary subjects          :",
        primary_df[
            SUBJECT_COL
        ].nunique(),
    )
    print(
        "Sensitivity subjects      :",
        sensitivity_df[
            SUBJECT_COL
        ].nunique(),
    )
    print(
        "Successful primary models :",
        len(
            converged_primary
        ),
        "/",
        len(
            primary_outcomes
        ),
    )
    print(
        "Random-slope primary fits :",
        len(
            random_slope_primary
        ),
        "/",
        len(
            primary_outcomes
        ),
    )
    print(
        "FDR-significant outcomes  :",
        len(
            sig
        ),
    )
    print(
        "Primary/sensitivity same sign:",
        f"{n_same_sign}/{len(comparison)}",
    )

    if len(
        sig
    ):
        print()
        print(
            "[FDR-significant population slopes]"
        )

        show = sig[
            [
                "outcome",
                "fixed_slope_per_year",
                "fixed_slope_baseline_sd_per_year",
                "fixed_slope_ci95_low",
                "fixed_slope_ci95_high",
                "fixed_slope_p",
                "fixed_slope_q_fdr",
                "random_slope_sd",
                "model_structure",
            ]
        ].copy()

        print(
            show.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:+.5g}"
                ),
            )
        )

    print()
    print(
        "Results:",
        args.results_dir,
    )


if __name__ == "__main__":
    main()
