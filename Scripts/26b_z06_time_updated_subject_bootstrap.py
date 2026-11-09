#!/usr/bin/env python3
"""
26b_z06_time_updated_subject_bootstrap.py

SMASH-HCM -- Step 26b
Subject-bootstrap sensitivity analysis for the age/sex-adjusted time-updated
H_z_06 Cox models from Step 26.

The bootstrap resamples subjects with replacement and retains each selected
subject's complete set of non-overlapping counting-process intervals.

Important convergence handling
------------------------------
statsmodels PHReg can return a finite coefficient while also issuing a
ConvergenceWarning. Such replicates are NOT treated as clean successful fits.
The primary bootstrap interval below is computed from warning-free finite fits.
Finite estimates returned with a convergence warning are counted separately and
summarized only for audit/sensitivity. This avoids silently calling a warned fit
"successful" merely because no Python exception was raised.

The bootstrap is used as a stability/sensitivity analysis. The percentile CI and
the fraction of bootstrap coefficients below zero are reported; the latter is a
descriptive sign-stability measure, not a formal null-hypothesis p-value.

Default input
-------------
Results/26_Z06_TimeUpdatedCox/z06_time_updated_cox_intervals.csv

Default output
--------------
Results/26b_Z06_TimeUpdated_SubjectBootstrap/
    z06_time_updated_subject_bootstrap.csv

Run from project root
---------------------
    python Scripts\\26b_z06_time_updated_subject_bootstrap.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.sm_exceptions import ConvergenceWarning


ENDPOINTS = ["Composite_HF", "Composite_VArrhythmia"]
N_BOOTSTRAP = 2000
SEED = 20261010
COVARIATES = ["z06_per_sd", "baseline_age_10y", "male"]


def infer_project_root() -> Path:
    """Locate the SMASH-HCM project root from the script location or cwd."""
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
        if (p / "Results").exists() and (p / "Scripts").exists():
            return p

    raise RuntimeError(
        "Cannot locate SMASH-HCM project root (expected Results/ and Scripts/)."
    )


def fit_bootstrap_beta(sample: pd.DataFrame) -> tuple[float, bool]:
    """Fit adjusted counting-process Cox model; return z06 log-HR and warning flag."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        res = PHReg(
            sample["stop"].astype(float),
            sample[COVARIATES].astype(float),
            status=sample["event"].astype(int),
            entry=sample["start"].astype(float),
            ties="breslow",
        ).fit(disp=0)

    had_convergence_warning = any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )
    return float(res.params[0]), had_convergence_warning


def percentile_summary(values: list[float]) -> dict[str, float]:
    """Return median and percentile 95% interval for finite log-HR values."""
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {
            "beta_median": np.nan,
            "beta_ci95_low": np.nan,
            "beta_ci95_high": np.nan,
            "hr_median": np.nan,
            "hr_ci95_low": np.nan,
            "hr_ci95_high": np.nan,
            "fraction_beta_negative": np.nan,
        }

    lo, med, hi = np.percentile(a, [2.5, 50.0, 97.5])
    return {
        "beta_median": float(med),
        "beta_ci95_low": float(lo),
        "beta_ci95_high": float(hi),
        "hr_median": float(np.exp(med)),
        "hr_ci95_low": float(np.exp(lo)),
        "hr_ci95_high": float(np.exp(hi)),
        "fraction_beta_negative": float(np.mean(a < 0.0)),
    }


def main() -> None:
    root = infer_project_root()
    input_path = (
        root
        / "Results"
        / "26_Z06_TimeUpdatedCox"
        / "z06_time_updated_cox_intervals.csv"
    )
    output_dir = root / "Results" / "26b_Z06_TimeUpdated_SubjectBootstrap"
    output_path = output_dir / "z06_time_updated_subject_bootstrap.csv"

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Cannot find Step 26 intervals file:\n{input_path}\n"
            "Run 26_z06_time_updated_cox.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {root}")
    print(f"Step 26 input: {input_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Bootstrap B  : {N_BOOTSTRAP}")

    df = pd.read_csv(input_path)
    required = {
        "analysis_subject_key",
        "endpoint",
        "lag",
        "start",
        "stop",
        "event",
        *COVARIATES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Step 26 interval table is missing columns: {missing}")

    seed_sequence = np.random.SeedSequence(SEED)
    endpoint_seeds = seed_sequence.spawn(len(ENDPOINTS))
    rows = []

    for endpoint, endpoint_seed in zip(ENDPOINTS, endpoint_seeds):
        d = df[
            (df["endpoint"].astype(str) == endpoint)
            & (df["lag"].astype(str) == "no_lag")
        ].dropna(subset=COVARIATES + ["start", "stop", "event"]).copy()

        groups = [
            g[["start", "stop", "event"] + COVARIATES].reset_index(drop=True)
            for _, g in d.groupby("analysis_subject_key", sort=False)
        ]

        n_subjects = len(groups)
        n_events = int(pd.to_numeric(d["event"], errors="coerce").fillna(0).sum())
        if n_subjects < 5:
            raise RuntimeError(
                f"Too few subjects for bootstrap endpoint {endpoint}: {n_subjects}"
            )
        if n_events < 1:
            raise RuntimeError(
                f"No events available for bootstrap endpoint {endpoint}."
            )

        rng = np.random.default_rng(endpoint_seed)
        clean_betas: list[float] = []
        warned_betas: list[float] = []
        exception_failures = 0
        nonfinite_failures = 0
        zero_event_resamples = 0

        for _ in range(N_BOOTSTRAP):
            sampled_idx = rng.integers(0, n_subjects, size=n_subjects)
            sample = pd.concat(
                [groups[i] for i in sampled_idx],
                ignore_index=True,
            )

            # A bootstrap draw can contain no event subjects for very sparse endpoints.
            if int(pd.to_numeric(sample["event"], errors="coerce").fillna(0).sum()) == 0:
                zero_event_resamples += 1
                continue

            try:
                beta, warned = fit_bootstrap_beta(sample)
                if not np.isfinite(beta):
                    nonfinite_failures += 1
                elif warned:
                    warned_betas.append(beta)
                else:
                    clean_betas.append(beta)
            except Exception:
                exception_failures += 1

        clean = percentile_summary(clean_betas)
        warned = percentile_summary(warned_betas)

        n_clean = len(clean_betas)
        n_warned = len(warned_betas)
        n_failed = exception_failures + nonfinite_failures + zero_event_resamples
        accounted = n_clean + n_warned + n_failed
        if accounted != N_BOOTSTRAP:
            raise RuntimeError(
                f"Internal bootstrap accounting error for {endpoint}: "
                f"{accounted} != {N_BOOTSTRAP}"
            )
        if n_clean == 0:
            raise RuntimeError(
                f"No warning-free finite bootstrap fits for endpoint {endpoint}."
            )

        rows.append(
            {
                "endpoint": endpoint,
                "model": "age_sex_adjusted",
                "lag": "no_lag",
                "n_subjects_original": n_subjects,
                "n_events_original": n_events,
                "bootstrap_reps_requested": N_BOOTSTRAP,
                "bootstrap_reps_clean_used": n_clean,
                "bootstrap_reps_convergence_warning": n_warned,
                "convergence_warning_fraction": float(n_warned / N_BOOTSTRAP),
                "bootstrap_reps_zero_event": zero_event_resamples,
                "bootstrap_reps_nonfinite": nonfinite_failures,
                "bootstrap_reps_exception": exception_failures,
                "bootstrap_reps_total_failed": n_failed,
                "failure_fraction": float(n_failed / N_BOOTSTRAP),
                **clean,
                "warned_fit_beta_median_audit": warned["beta_median"],
                "warned_fit_hr_median_audit": warned["hr_median"],
                "warned_fit_fraction_beta_negative_audit": warned[
                    "fraction_beta_negative"
                ],
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)

    print()
    print(out.to_string(index=False))
    print()
    for _, r in out.iterrows():
        print(
            f"{r['endpoint']}: clean={int(r['bootstrap_reps_clean_used'])}/{N_BOOTSTRAP}, "
            f"convergence warnings={int(r['bootstrap_reps_convergence_warning'])} "
            f"({100.0 * r['convergence_warning_fraction']:.1f}%), "
            f"failed={int(r['bootstrap_reps_total_failed'])}."
        )
        if r["convergence_warning_fraction"] > 0.10:
            print(
                "  CAUTION: >10% of bootstrap replicates returned a convergence "
                "warning; interpret this endpoint as numerically unstable/sparse."
            )

    print()
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
