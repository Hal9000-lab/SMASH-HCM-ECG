#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
36_validate_mimic_subject_balanced_hz06_replication.py

SMASH-HCM -- Step 36
Subject-balanced robustness analysis of the external MIMIC-IV longitudinal
replication of the prespecified HCM-oriented phenotype H_z_06.

Scientific aim
--------------
Step 35 established the external population-average H_z_06 trajectory with a
linear mixed-effects model after bounded 30-day clinical-window collapse. This
step asks a complementary question:

    Does the negative longitudinal direction remain evident when each subject
    contributes exactly one empirical slope, irrespective of how many ECG
    visits that subject has?

This is a robustness/sensitivity analysis, not a replacement for the Step-35
mixed-effects model.

Primary subject-balanced input
------------------------------
Results/35_MIMIC_Longitudinal_Hz06_Replication/
    hz06_subject_empirical_slopes.csv

The table contains one row per Step-35 primary-cohort subject and was derived
from the bounded 30-day visit representation.

Internal descriptive comparator
--------------------------------
Results/17_Longitudinal_Trajectories/patient_empirical_slopes.csv

Only outcome == H_z_06 is used. The SHaRe comparison is descriptive because its
visit construction is not identical to the MIMIC 30-day clinical-window
collapse. No formal equivalence claim is made. Because empirical annualized
subject slopes can contain extreme values when follow-up is short, cross-cohort
comparison deliberately uses robust/rank-based summaries (median, IQR,
fraction negative, Mann-Whitney, and bootstrap median difference) rather than
means, SDs, or Wasserstein distance.

Analyses
--------
For MIMIC, summarize subject-wise:
    - OLS slope per year
    - first-to-last annualized slope

for four prespecified robustness cohorts:
    1. all Step-35 primary subjects (>=2 visits)
    2. >=3 visits
    3. >=1 year follow-up
    4. >=3 visits AND >=1 year follow-up

For each cohort/metric report:
    - median and IQR
    - mean and SD (within-MIMIC descriptive only)
    - 95% subject bootstrap CI for the median
    - fraction of slopes < 0
    - exact binomial sign test versus 0.5 (two-sided and directional <0)
    - Wilcoxon signed-rank test versus 0 (two-sided and directional <0)

The directional tests are justified only as sensitivity checks because the
negative direction was prespecified by the independent SHaRe result before
external evaluation.

Outputs
-------
Results/36_MIMIC_SubjectBalanced_Hz06_Replication/
    subject_balanced_sensitivity.csv
    share_mimic_empirical_slope_comparison.csv
    subject_balanced_report.txt
    run_summary.json
    figures/
        mimic_ols_subject_slopes.png
        mimic_first_last_subject_slopes.png
        share_mimic_ols_slope_ecdf.png

Run
---
python Scripts/36_validate_mimic_subject_balanced_hz06_replication.py

Dependencies
------------
numpy pandas scipy matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 20261021
BOOTSTRAP_REPS = 10000


def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "Results").exists():
            return p
    return here.parent


def parse_args() -> argparse.Namespace:
    root = infer_project_root()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mimic-slopes",
        type=Path,
        default=root / "Results" / "35_MIMIC_Longitudinal_Hz06_Replication" / "hz06_subject_empirical_slopes.csv",
    )
    p.add_argument(
        "--share-slopes",
        type=Path,
        default=root / "Results" / "17_Longitudinal_Trajectories" / "patient_empirical_slopes.csv",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=root / "Results" / "36_MIMIC_SubjectBalanced_Hz06_Replication",
    )
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def finite(x) -> np.ndarray:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    return a[np.isfinite(a)]


def bootstrap_median_ci(x: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    vals = np.empty(reps, dtype=float)
    n = len(x)
    for i in range(reps):
        vals[i] = np.median(x[rng.integers(0, n, size=n)])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def safe_wilcoxon(x: np.ndarray, alternative: str) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0.0):
        return np.nan
    try:
        return float(stats.wilcoxon(x, alternative=alternative, zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


def summarize_vector(
    x: np.ndarray,
    cohort: str,
    metric: str,
    reps: int,
    rng: np.random.Generator,
) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {
            "cohort": cohort,
            "metric": metric,
            "n_subjects": 0,
        }

    n_neg = int(np.sum(x < 0))
    frac_neg = float(n_neg / n)
    ci_lo, ci_hi = bootstrap_median_ci(x, reps, rng)

    sign_two = float(stats.binomtest(n_neg, n, p=0.5, alternative="two-sided").pvalue)
    sign_neg = float(stats.binomtest(n_neg, n, p=0.5, alternative="greater").pvalue)

    return {
        "cohort": cohort,
        "metric": metric,
        "n_subjects": int(n),
        "mean_slope_per_year": float(np.mean(x)),
        "sd_slope_per_year": float(np.std(x, ddof=1)) if n > 1 else np.nan,
        "median_slope_per_year": float(np.median(x)),
        "q25_slope_per_year": float(np.quantile(x, 0.25)),
        "q75_slope_per_year": float(np.quantile(x, 0.75)),
        "bootstrap_median_ci_low": ci_lo,
        "bootstrap_median_ci_high": ci_hi,
        "n_negative": n_neg,
        "fraction_negative": frac_neg,
        "sign_test_p_two_sided": sign_two,
        "sign_test_p_directional_negative": sign_neg,
        "wilcoxon_p_two_sided": safe_wilcoxon(x, "two-sided"),
        "wilcoxon_p_directional_negative": safe_wilcoxon(x, "less"),
    }


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def bootstrap_median_difference_ci(
    x: np.ndarray,
    y: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap MIMIC-minus-SHaRe difference in medians.

    This is a descriptive independent-cohort comparison, not an equivalence
    procedure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan
    obs = float(np.median(x) - np.median(y))
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        xb = x[rng.integers(0, len(x), size=len(x))]
        yb = y[rng.integers(0, len(y), size=len(y))]
        vals[i] = np.median(xb) - np.median(yb)
    return (
        obs,
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )


def plot_hist(x: np.ndarray, median: float, path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.hist(x, bins=35, alpha=0.8)
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    ax.axvline(median, linestyle="-", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Subjects")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not args.mimic_slopes.exists():
        raise FileNotFoundError(f"MIMIC subject slopes not found: {args.mimic_slopes}")

    mimic = pd.read_csv(args.mimic_slopes)
    required = {
        "n_visits",
        "followup_years",
        "ols_slope_per_year",
        "first_to_last_slope_per_year",
    }
    missing = sorted(required - set(mimic.columns))
    if missing:
        raise KeyError("MIMIC slopes table is missing: " + ", ".join(missing))

    for c in required:
        mimic[c] = pd.to_numeric(mimic[c], errors="coerce")

    cohorts = {
        "primary_30d_2plus": np.ones(len(mimic), dtype=bool),
        "sensitivity_30d_3plus": mimic["n_visits"].ge(3).to_numpy(),
        "sensitivity_30d_followup1y": mimic["followup_years"].ge(1.0).to_numpy(),
        "sensitivity_30d_3plus_followup1y": (
            mimic["n_visits"].ge(3) & mimic["followup_years"].ge(1.0)
        ).to_numpy(),
    }

    metrics = {
        "ols_slope_per_year": "ols_slope_per_year",
        "first_to_last_slope_per_year": "first_to_last_slope_per_year",
    }

    rng = np.random.default_rng(args.seed)
    rows = []
    for cohort, mask in cohorts.items():
        d = mimic.loc[mask].copy()
        for metric_name, col in metrics.items():
            x = finite(d[col])
            rows.append(
                summarize_vector(
                    x=x,
                    cohort=cohort,
                    metric=metric_name,
                    reps=args.bootstrap_reps,
                    rng=rng,
                )
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.results_dir / "subject_balanced_sensitivity.csv", index=False)

    # Internal SHaRe comparator: descriptive and robust only.
    # Do not use cross-cohort means/SDs or Wasserstein distance here: empirical
    # annualized slopes can have extreme values, and the two cohorts do not use
    # identical visit construction.
    compare_rows = []
    robust_comparisons = []
    if args.share_slopes.exists():
        share = pd.read_csv(args.share_slopes)
        if "outcome" not in share.columns:
            raise KeyError("SHaRe empirical slopes table lacks 'outcome'.")
        share = share[share["outcome"].astype(str).eq("H_z_06")].copy()

        for cohort_name, d in [
            ("MIMIC_primary_30d", mimic),
            ("SHaRe_Step17", share),
        ]:
            for metric_name, col in metrics.items():
                if col not in d.columns:
                    continue
                x = finite(d[col])
                compare_rows.append({
                    "cohort": cohort_name,
                    "metric": metric_name,
                    "n_subjects": int(len(x)),
                    "median_slope_per_year": float(np.median(x)) if len(x) else np.nan,
                    "q25_slope_per_year": float(np.quantile(x, 0.25)) if len(x) else np.nan,
                    "q75_slope_per_year": float(np.quantile(x, 0.75)) if len(x) else np.nan,
                    "fraction_negative": float(np.mean(x < 0)) if len(x) else np.nan,
                })

        for metric_name, col in metrics.items():
            if col not in mimic.columns or col not in share.columns:
                continue
            xm = finite(mimic[col])
            xs = finite(share[col])
            if len(xm) and len(xs):
                mw = stats.mannwhitneyu(xm, xs, alternative="two-sided")
                diff, diff_lo, diff_hi = bootstrap_median_difference_ci(
                    xm, xs, args.bootstrap_reps, rng
                )
                robust_comparisons.append({
                    "metric": metric_name,
                    "mimic_median": float(np.median(xm)),
                    "share_median": float(np.median(xs)),
                    "mimic_minus_share_median_difference": diff,
                    "bootstrap_difference_ci_low": diff_lo,
                    "bootstrap_difference_ci_high": diff_hi,
                    "mann_whitney_u": float(mw.statistic),
                    "mann_whitney_p": float(mw.pvalue),
                    "note": (
                        "Descriptive robust comparison only; visit construction differs "
                        "between cohorts and this is not an equivalence test."
                    ),
                })
        descriptive_cmp = robust_comparisons
    else:
        share = None
        descriptive_cmp = None

    pd.DataFrame(compare_rows).to_csv(
        args.results_dir / "share_mimic_empirical_slope_comparison.csv",
        index=False,
    )

    primary_ols = summary[
        (summary["cohort"] == "primary_30d_2plus")
        & (summary["metric"] == "ols_slope_per_year")
    ].iloc[0]

    primary_fl = summary[
        (summary["cohort"] == "primary_30d_2plus")
        & (summary["metric"] == "first_to_last_slope_per_year")
    ].iloc[0]

    x_ols = finite(mimic["ols_slope_per_year"])
    x_fl = finite(mimic["first_to_last_slope_per_year"])
    plot_hist(
        x_ols,
        float(primary_ols["median_slope_per_year"]),
        fig_dir / "mimic_ols_subject_slopes.png",
        "MIMIC subject-balanced H_z_06 OLS slopes",
        "H_z_06 slope (latent units/year)",
    )
    plot_hist(
        x_fl,
        float(primary_fl["median_slope_per_year"]),
        fig_dir / "mimic_first_last_subject_slopes.png",
        "MIMIC subject-balanced H_z_06 first-to-last slopes",
        "H_z_06 annualized first-to-last change",
    )

    if share is not None and "ols_slope_per_year" in share.columns:
        xs = finite(share["ols_slope_per_year"])
        if len(xs):
            fig, ax = plt.subplots(figsize=(7.0, 4.6))
            xx, yy = ecdf(x_ols)
            ax.plot(xx, yy, label="MIMIC")
            xx, yy = ecdf(xs)
            ax.plot(xx, yy, label="SHaRe")
            ax.axvline(0.0, linestyle="--", linewidth=1.2)
            pooled = np.concatenate([x_ols, xs])
            if len(pooled) >= 20:
                lo, hi = np.quantile(pooled, [0.01, 0.99])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ax.set_xlim(lo, hi)
            ax.set_xlabel("Subject OLS H_z_06 slope (latent units/year)")
            ax.set_ylabel("Empirical cumulative probability")
            ax.set_title("Subject-level H_z_06 slope distributions (robust display range)")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "share_mimic_ols_slope_ecdf.png", dpi=300)
            plt.close(fig)

    report_lines = [
        "SMASH-HCM Step 36 -- Subject-balanced MIMIC H_z_06 replication",
        f"Primary subjects: {int(primary_ols['n_subjects'])}",
        (
            "Primary OLS median slope: "
            f"{primary_ols['median_slope_per_year']:.6f} units/year "
            f"(95% subject-bootstrap median CI "
            f"{primary_ols['bootstrap_median_ci_low']:.6f} to "
            f"{primary_ols['bootstrap_median_ci_high']:.6f})"
        ),
        f"Fraction OLS slopes negative: {primary_ols['fraction_negative']:.3f}",
        f"Exact sign-test p (two-sided): {primary_ols['sign_test_p_two_sided']:.6g}",
        f"Exact sign-test p (prespecified negative direction): {primary_ols['sign_test_p_directional_negative']:.6g}",
        f"Wilcoxon p (two-sided): {primary_ols['wilcoxon_p_two_sided']:.6g}",
        f"Wilcoxon p (prespecified negative direction): {primary_ols['wilcoxon_p_directional_negative']:.6g}",
        "",
        (
            "Primary first-to-last median slope: "
            f"{primary_fl['median_slope_per_year']:.6f} units/year "
            f"(95% subject-bootstrap median CI "
            f"{primary_fl['bootstrap_median_ci_low']:.6f} to "
            f"{primary_fl['bootstrap_median_ci_high']:.6f})"
        ),
        f"Fraction first-to-last slopes negative: {primary_fl['fraction_negative']:.3f}",
        "",
        "Interpretation guardrails:",
        "- Step 35 remains the primary external longitudinal replication analysis.",
        "- This step gives each subject one slope and tests robustness to unequal visit density.",
        "- Directional tests are sensitivity checks because the negative direction was prespecified from SHaRe.",
        "- SHaRe-versus-MIMIC subject-slope comparisons use robust/rank-based summaries only and are descriptive, not equivalence tests.",
        "- Population-average and subject-level trends do not imply deterministic progression in every patient.",
    ]
    (args.results_dir / "subject_balanced_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    run_summary = {
        "script": Path(__file__).name,
        "step": 36,
        "scientific_role": "Subject-balanced robustness of external MIMIC H_z_06 longitudinal replication",
        "primary_input": str(args.mimic_slopes.resolve()),
        "internal_descriptive_comparator": str(args.share_slopes.resolve()) if args.share_slopes.exists() else None,
        "bootstrap_reps": int(args.bootstrap_reps),
        "seed": int(args.seed),
        "primary_ols": primary_ols.to_dict(),
        "primary_first_to_last": primary_fl.to_dict(),
        "sensitivity_cohorts": summary.to_dict(orient="records"),
        "share_mimic_robust_descriptive_comparisons": descriptive_cmp,
        "constraints": {
            "one_empirical_slope_per_subject": True,
            "step35_mixed_model_remains_primary": True,
            "negative_direction_prespecified_from_share": True,
            "between_cohort_subject_slope_comparison_is_equivalence_test": False,
            "cross_cohort_mean_sd_wasserstein_omitted_due_to_extreme_empirical_slopes": True,
        },
    }
    (args.results_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=lambda x: None if pd.isna(x) else x),
        encoding="utf-8",
    )

    print("=" * 80)
    print("36 - Subject-balanced MIMIC H_z_06 replication")
    print("=" * 80)
    print(f"Primary subjects                : {int(primary_ols['n_subjects'])}")
    print(f"Median OLS slope                : {primary_ols['median_slope_per_year']:.6f} units/year")
    print(
        "Bootstrap median 95% CI         : "
        f"{primary_ols['bootstrap_median_ci_low']:.6f} to "
        f"{primary_ols['bootstrap_median_ci_high']:.6f}"
    )
    print(f"Fraction OLS slopes negative    : {primary_ols['fraction_negative']:.3f}")
    print(f"Directional exact sign-test p   : {primary_ols['sign_test_p_directional_negative']:.6g}")
    print(f"Directional Wilcoxon p          : {primary_ols['wilcoxon_p_directional_negative']:.6g}")
    print(f"Results                         : {args.results_dir}")


if __name__ == "__main__":
    main()
