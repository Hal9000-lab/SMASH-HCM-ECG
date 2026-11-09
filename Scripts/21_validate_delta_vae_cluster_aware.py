#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
21_validate_delta_vae_cluster_aware.py

SMASH-HCM -- Step 21
Cluster-aware validation of the four prespecified physiological associations
from the HCM-only longitudinal Delta-VAE (VAE-3).

PRIMARY AIM
-----------
The held-out VAE-3 test set contains multiple consecutive visit pairs from some
subjects. Pair-level Spearman correlations are therefore descriptive but their
usual p-values assume independent observations. This script validates the four
prespecified VAE-3 interpretations while respecting subject clustering.

PRESPECIFIED ASSOCIATIONS
-------------------------
D_z_00 <-> Delta T_amp_mV_V4
D_z_01 <-> Delta QRS_net_area_mVms_V3
D_z_02 <-> Delta QRS_net_area_mVms_V5
D_z_03 <-> Delta QRS_net_area_mVms_V3

THREE COMPLEMENTARY ANALYSES
----------------------------
1) Pair-level Spearman rho on the held-out test set
   - retained only as the reference effect estimate

2) Subject-cluster bootstrap
   - resample subjects with replacement
   - keep all pairs belonging to each sampled subject
   - recompute pair-level Spearman rho
   - report percentile 95% CI
   - inference is based on the CI, not on naive pair-level p-values

3) Subject-level aggregation
   - median latent and median Delta-feature within each subject
   - one independent observation per subject
   - Spearman rho
   - subject-level permutation p-value
   - BH-FDR across the four prespecified tests

4) One-pair-per-subject Monte Carlo sensitivity
   - in each repetition select one random pair per subject
   - compute Spearman rho across independent subjects
   - report median and 95% interval across repetitions
   - report fraction of repetitions preserving the prespecified sign

INPUT
-----
Results/20_Longitudinal_DeltaVAE_Interpretation/pair_delta_ecg_features.csv

OUTPUT
------
Results/21_DeltaVAE_ClusterAware/
    cluster_aware_summary.csv
    subject_level_medians.csv
    subject_level_associations.csv
    cluster_bootstrap_rho.csv
    one_pair_per_subject_rho.csv
    run_summary.json
    figures/
        cluster_aware_forest.png

RUN
---
python Scripts\\21_validate_delta_vae_cluster_aware.py
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


SEED = 20261005

PRESPECIFIED = [
    {
        "latent": "D_z_00",
        "feature": "delta__T_amp_mV_V4",
        "short_feature": "Delta T amplitude V4",
        "expected_sign": -1,
        "interpretation": "precordial/lateral repolarization change",
    },
    {
        "latent": "D_z_01",
        "feature": "delta__QRS_net_area_mVms_V3",
        "short_feature": "Delta QRS net area V3",
        "expected_sign": -1,
        "interpretation": "precordial depolarization-ST/T coupled change",
    },
    {
        "latent": "D_z_02",
        "feature": "delta__QRS_net_area_mVms_V5",
        "short_feature": "Delta QRS net area V5",
        "expected_sign": +1,
        "interpretation": "lateral/LV-oriented depolarization-force change",
    },
    {
        "latent": "D_z_03",
        "feature": "delta__QRS_net_area_mVms_V3",
        "short_feature": "Delta QRS net area V3",
        "expected_sign": -1,
        "interpretation": "precordial activation redistribution / R-S balance change",
    },
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Cluster-aware validation of VAE-3 longitudinal latent interpretations."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "Results/20_Longitudinal_DeltaVAE_Interpretation/"
            "pair_delta_ecg_features.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Results/21_DeltaVAE_ClusterAware"
        ),
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="Primary split. Default: test.",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of subject-cluster bootstrap replicates.",
    )
    p.add_argument(
        "--n-one-pair",
        type=int,
        default=10000,
        help="Number of one-pair-per-subject Monte Carlo repetitions.",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=20000,
        help="Subject-level permutation repetitions per prespecified test.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    return p.parse_args()


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    finite = np.isfinite(p)
    if not np.any(finite):
        return q

    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    n = len(ranked)

    qr = ranked * n / np.arange(1, n + 1)
    qr = np.minimum.accumulate(qr[::-1])[::-1]
    qr = np.clip(qr, 0.0, 1.0)

    restored = np.empty(n, dtype=float)
    restored[order] = qr
    q[finite] = restored
    return q


def spearman_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)

    n = int(np.sum(m))
    if n < 5:
        return np.nan, np.nan, n

    xx = x[m]
    yy = y[m]

    if np.std(xx) <= 1e-12 or np.std(yy) <= 1e-12:
        return np.nan, np.nan, n

    rho, p = stats.spearmanr(xx, yy)
    return float(rho), float(p), n


def percentile_interval(values, alpha=0.05):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return np.nan, np.nan

    lo = float(np.quantile(v, alpha / 2.0))
    hi = float(np.quantile(v, 1.0 - alpha / 2.0))
    return lo, hi


def cluster_bootstrap(
    df,
    subject_col,
    x_col,
    y_col,
    n_bootstrap,
    rng,
):
    d = df[[subject_col, x_col, y_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna()

    grouped = {
        str(s): g[[x_col, y_col]].to_numpy(dtype=float)
        for s, g in d.groupby(subject_col, sort=False)
    }

    subjects = np.asarray(list(grouped.keys()), dtype=object)
    n_subjects = len(subjects)

    if n_subjects < 5:
        return np.full(n_bootstrap, np.nan)

    out = np.full(n_bootstrap, np.nan, dtype=float)

    for b in range(n_bootstrap):
        sampled = rng.choice(subjects, size=n_subjects, replace=True)
        blocks = [grouped[str(s)] for s in sampled]
        arr = np.concatenate(blocks, axis=0)

        rho, _, _ = spearman_safe(arr[:, 0], arr[:, 1])
        out[b] = rho

    return out


def subject_level_table(df, subject_col):
    cols = [subject_col]
    for spec in PRESPECIFIED:
        cols.extend([spec["latent"], spec["feature"]])

    cols = list(dict.fromkeys(cols))
    d = df[cols].copy()

    for c in cols:
        if c == subject_col:
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce")

    agg = d.groupby(subject_col, as_index=False).median(numeric_only=True)
    return agg


def spearman_permutation_p(x, y, n_perm, rng):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    n = len(x)
    if n < 5:
        return np.nan, np.nan, n

    rx = stats.rankdata(x)
    ry = stats.rankdata(y)

    rx = rx - np.mean(rx)
    ry = ry - np.mean(ry)

    denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    if denom <= 0:
        return np.nan, np.nan, n

    observed = float(np.sum(rx * ry) / denom)

    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(rx)
        rho_perm = float(np.sum(perm * ry) / denom)
        if abs(rho_perm) >= abs(observed) - 1e-15:
            extreme += 1

    p = (extreme + 1.0) / (n_perm + 1.0)
    return observed, float(p), n


def one_pair_per_subject(
    df,
    subject_col,
    x_col,
    y_col,
    n_reps,
    rng,
):
    d = df[[subject_col, x_col, y_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna()

    grouped = [
        g[[x_col, y_col]].to_numpy(dtype=float)
        for _, g in d.groupby(subject_col, sort=False)
    ]

    if len(grouped) < 5:
        return np.full(n_reps, np.nan)

    out = np.full(n_reps, np.nan, dtype=float)

    for r in range(n_reps):
        chosen = np.asarray(
            [g[rng.integers(0, len(g))] for g in grouped],
            dtype=float,
        )
        rho, _, _ = spearman_safe(chosen[:, 0], chosen[:, 1])
        out[r] = rho

    return out


def forest_plot(summary, out_path):
    d = summary.copy()
    y = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Cluster-bootstrap pair-level effect.
    x = d["pair_level_rho"].to_numpy(float)
    lo = d["cluster_boot_ci_low"].to_numpy(float)
    hi = d["cluster_boot_ci_high"].to_numpy(float)

    ax.errorbar(
        x,
        y + 0.12,
        xerr=np.vstack([x - lo, hi - x]),
        fmt="o",
        capsize=3,
        label="Pair-level rho + subject-cluster bootstrap 95% CI",
    )

    # Subject-level independent aggregation.
    xs = d["subject_level_rho"].to_numpy(float)
    ax.scatter(
        xs,
        y - 0.12,
        marker="s",
        label="Subject-level median association",
    )

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{r.latent}: {r.short_feature}"
            for r in d.itertuples()
        ]
    )
    ax.set_xlabel("Spearman rho")
    ax.set_title("VAE-3 cluster-aware validation on held-out test subjects")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.input)

    required_base = {
        "analysis_subject_key",
        "split",
        "pair_id",
    }

    missing = required_base - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    for spec in PRESPECIFIED:
        for col in [spec["latent"], spec["feature"]]:
            if col not in df.columns:
                raise KeyError(
                    f"Missing prespecified column '{col}' in {args.input}"
                )

    test = df[df["split"].astype(str) == args.split].copy()

    if len(test) == 0:
        raise RuntimeError(f"No rows found for split='{args.split}'.")

    subject_col = "analysis_subject_key"
    n_pairs = int(len(test))
    n_subjects = int(test[subject_col].nunique())

    print("=" * 84)
    print("21 - Cluster-aware validation of HCM longitudinal Delta-VAE")
    print("=" * 84)
    print("Input        :", args.input)
    print("Split        :", args.split)
    print("Pairs        :", n_pairs)
    print("Subjects     :", n_subjects)
    print("Bootstrap    :", args.n_bootstrap)
    print("One-pair MC  :", args.n_one_pair)
    print("Permutations :", args.n_permutations)

    subject_df = subject_level_table(test, subject_col)
    subject_df.to_csv(
        args.output_dir / "subject_level_medians.csv",
        index=False,
    )

    summary_rows = []
    subject_rows = []
    boot_long = []
    one_pair_long = []

    # Use independent RNG streams per test for reproducibility.
    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(len(PRESPECIFIED) * 3)

    for i, spec in enumerate(PRESPECIFIED):
        latent = spec["latent"]
        feature = spec["feature"]

        x = pd.to_numeric(test[latent], errors="coerce").to_numpy(float)
        y = pd.to_numeric(test[feature], errors="coerce").to_numpy(float)

        pair_rho, pair_p_naive, pair_n = spearman_safe(x, y)

        rng_boot = np.random.default_rng(child_seeds[3 * i])
        boot = cluster_bootstrap(
            test,
            subject_col,
            latent,
            feature,
            args.n_bootstrap,
            rng_boot,
        )
        boot_lo, boot_hi = percentile_interval(boot)
        boot_med = float(np.nanmedian(boot))
        boot_sign_fraction = float(
            np.mean(
                np.sign(boot[np.isfinite(boot)])
                == spec["expected_sign"]
            )
        )

        for b, rho in enumerate(boot):
            boot_long.append({
                "latent": latent,
                "feature": feature,
                "bootstrap_index": b,
                "rho": rho,
            })

        sx = pd.to_numeric(
            subject_df[latent],
            errors="coerce",
        ).to_numpy(float)
        sy = pd.to_numeric(
            subject_df[feature],
            errors="coerce",
        ).to_numpy(float)

        rng_perm = np.random.default_rng(child_seeds[3 * i + 1])
        subject_rho, subject_perm_p, subject_n = spearman_permutation_p(
            sx,
            sy,
            args.n_permutations,
            rng_perm,
        )

        subject_rows.append({
            "latent": latent,
            "feature": feature,
            "short_feature": spec["short_feature"],
            "interpretation": spec["interpretation"],
            "n_subjects": subject_n,
            "subject_level_rho": subject_rho,
            "subject_level_permutation_p": subject_perm_p,
        })

        rng_one = np.random.default_rng(child_seeds[3 * i + 2])
        one = one_pair_per_subject(
            test,
            subject_col,
            latent,
            feature,
            args.n_one_pair,
            rng_one,
        )
        one_lo, one_hi = percentile_interval(one)
        one_med = float(np.nanmedian(one))
        one_sign_fraction = float(
            np.mean(
                np.sign(one[np.isfinite(one)])
                == spec["expected_sign"]
            )
        )

        for r, rho in enumerate(one):
            one_pair_long.append({
                "latent": latent,
                "feature": feature,
                "repetition": r,
                "rho": rho,
            })

        summary_rows.append({
            "latent": latent,
            "feature": feature,
            "short_feature": spec["short_feature"],
            "interpretation": spec["interpretation"],
            "expected_sign": spec["expected_sign"],
            "n_test_pairs": pair_n,
            "n_test_subjects": subject_n,
            "pair_level_rho": pair_rho,
            "pair_level_naive_p_descriptive_only": pair_p_naive,
            "cluster_boot_median_rho": boot_med,
            "cluster_boot_ci_low": boot_lo,
            "cluster_boot_ci_high": boot_hi,
            "cluster_boot_expected_sign_fraction": boot_sign_fraction,
            "cluster_boot_ci_excludes_zero": bool(
                np.isfinite(boot_lo)
                and np.isfinite(boot_hi)
                and (
                    (boot_lo > 0 and boot_hi > 0)
                    or (boot_lo < 0 and boot_hi < 0)
                )
            ),
            "subject_level_rho": subject_rho,
            "subject_level_permutation_p": subject_perm_p,
            "one_pair_median_rho": one_med,
            "one_pair_ci_low": one_lo,
            "one_pair_ci_high": one_hi,
            "one_pair_expected_sign_fraction": one_sign_fraction,
        })

        print()
        print(latent, "<->", spec["short_feature"])
        print(f"  pair-level rho             : {pair_rho:+.4f}")
        print(
            "  cluster bootstrap 95% CI   : "
            f"[{boot_lo:+.4f}, {boot_hi:+.4f}]"
        )
        print(
            f"  subject-level rho           : {subject_rho:+.4f}"
            f"  perm-p={subject_perm_p:.6g}"
        )
        print(
            "  one-pair/subject median rho : "
            f"{one_med:+.4f} [{one_lo:+.4f}, {one_hi:+.4f}]"
        )

    summary = pd.DataFrame(summary_rows)
    subject_assoc = pd.DataFrame(subject_rows)

    # FDR correction only across the four prespecified independent-subject tests.
    q = bh_fdr(
        subject_assoc["subject_level_permutation_p"].to_numpy(float)
    )
    subject_assoc["subject_level_permutation_q_fdr_4"] = q

    summary = summary.merge(
        subject_assoc[
            [
                "latent",
                "feature",
                "subject_level_permutation_q_fdr_4",
            ]
        ],
        on=["latent", "feature"],
        how="left",
        validate="one_to_one",
    )

    summary["subject_level_expected_sign"] = (
        np.sign(summary["subject_level_rho"])
        == summary["expected_sign"]
    )

    summary["one_pair_interval_excludes_zero"] = (
        (
            (summary["one_pair_ci_low"] > 0)
            & (summary["one_pair_ci_high"] > 0)
        )
        |
        (
            (summary["one_pair_ci_low"] < 0)
            & (summary["one_pair_ci_high"] < 0)
        )
    )

    # Conservative consolidation rule:
    # - cluster-bootstrap CI excludes zero
    # - independent subject-level estimate preserves sign
    # - subject-level FDR q < 0.05
    # - at least 90% of one-pair-per-subject repetitions preserve sign
    summary["cluster_aware_confirmed"] = (
        summary["cluster_boot_ci_excludes_zero"]
        & summary["subject_level_expected_sign"]
        & (summary["subject_level_permutation_q_fdr_4"] < 0.05)
        & (summary["one_pair_expected_sign_fraction"] >= 0.90)
    )

    summary.to_csv(
        args.output_dir / "cluster_aware_summary.csv",
        index=False,
    )
    subject_assoc.to_csv(
        args.output_dir / "subject_level_associations.csv",
        index=False,
    )
    pd.DataFrame(boot_long).to_csv(
        args.output_dir / "cluster_bootstrap_rho.csv",
        index=False,
    )
    pd.DataFrame(one_pair_long).to_csv(
        args.output_dir / "one_pair_per_subject_rho.csv",
        index=False,
    )

    forest_plot(
        summary,
        fig_dir / "cluster_aware_forest.png",
    )

    run_summary = {
        "script": "21_validate_delta_vae_cluster_aware.py",
        "version": "1.0",
        "scientific_role": (
            "Cluster-aware held-out validation of the four prespecified "
            "VAE-3 physiological associations"
        ),
        "input": str(args.input),
        "split": args.split,
        "n_test_pairs": n_pairs,
        "n_test_subjects": n_subjects,
        "n_prespecified_tests": len(PRESPECIFIED),
        "n_cluster_bootstrap": args.n_bootstrap,
        "n_one_pair_per_subject_repetitions": args.n_one_pair,
        "n_subject_level_permutations": args.n_permutations,
        "fdr_family": (
            "BH correction across the four prespecified subject-level "
            "permutation tests"
        ),
        "confirmation_rule": (
            "cluster bootstrap 95% CI excludes zero; subject-level median "
            "association preserves the prespecified sign and has q<0.05; "
            ">=90% of one-pair-per-subject repetitions preserve the sign"
        ),
        "n_confirmed": int(
            np.sum(summary["cluster_aware_confirmed"])
        ),
        "results": [
            {
                "latent": str(r["latent"]),
                "feature": str(r["feature"]),
                "interpretation": str(r["interpretation"]),
                "pair_level_rho": float(r["pair_level_rho"]),
                "cluster_boot_ci_95": [
                    float(r["cluster_boot_ci_low"]),
                    float(r["cluster_boot_ci_high"]),
                ],
                "subject_level_rho": float(r["subject_level_rho"]),
                "subject_level_permutation_q_fdr_4": float(
                    r["subject_level_permutation_q_fdr_4"]
                ),
                "one_pair_median_rho": float(r["one_pair_median_rho"]),
                "one_pair_expected_sign_fraction": float(
                    r["one_pair_expected_sign_fraction"]
                ),
                "cluster_aware_confirmed": bool(
                    r["cluster_aware_confirmed"]
                ),
            }
            for _, r in summary.iterrows()
        ],
        "notes": [
            (
                "Usual pair-level Spearman p-values are retained only as "
                "descriptive quantities because repeated pairs within a subject "
                "are not independent."
            ),
            (
                "The subject-cluster bootstrap resamples subjects and retains "
                "all within-subject pairs."
            ),
            (
                "The subject-level analysis uses one median latent value and "
                "one median Delta-feature value per test subject."
            ),
            (
                "The one-pair-per-subject sensitivity repeatedly selects a "
                "single pair per subject, guaranteeing independence across "
                "observations within each repetition."
            ),
            (
                "All four tests were prespecified from the held-out Step-20 "
                "physiological interpretation and no feature selection is "
                "performed in Step 21."
            ),
        ],
    }

    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 84)
    print("STEP 21 COMPLETE")
    print("=" * 84)
    print(
        "Cluster-aware confirmed:",
        int(np.sum(summary["cluster_aware_confirmed"])),
        "/",
        len(summary),
    )
    print("Results:", args.output_dir)


if __name__ == "__main__":
    main()
