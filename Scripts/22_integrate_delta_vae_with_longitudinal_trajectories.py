#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
22_integrate_delta_vae_with_longitudinal_trajectories.py

SMASH-HCM -- Step 22
Integration of HCM-only longitudinal Delta-VAE modes (VAE-3) with the
patient-specific longitudinal trajectories estimated in Step 17.

SCIENTIFIC QUESTION
-------------------
Which individual ECG morphological change modes captured by the Delta-VAE
are aligned with the subject-specific longitudinal drift observed in the
frozen HCM-oriented manifold and in distance from the Normal manifold?

PRIMARY INTEGRATION
-------------------
At subject level:
    median pairwise D_z_00 ... D_z_03
versus Step-17 patient BLUP slopes for:
    H_z_01, H_z_05, H_z_06, d_N

This gives 4 x 4 = 16 prespecified associations.

The Delta-VAE subject summary uses the median signed latent score across all
consecutive visit pairs from the same subject. This is intended to represent
the subject's typical direction of expression of each learned change mode.

PRIMARY STATISTICS
------------------
- Spearman rho across independent subjects
- subject bootstrap 95% CI
- Benjamini-Hochberg FDR across the 16 prespecified primary associations

SENSITIVITY ANALYSES
--------------------
1) Held-out VAE-3 test subjects only:
   the same subject-level median Delta-VAE scores versus Step-17 BLUP slopes.
   This checks that the integration is not limited to VAE-3 training subjects.

2) Empirical OLS slopes in subjects with >=3 visits:
   the same Delta-VAE summaries versus Step-17 subject-wise OLS slopes.
   This is a model-independent sensitivity to BLUP shrinkage.

IMPORTANT INTERPRETIVE CAUTIONS
-------------------------------
- This is an integration analysis, not an external validation analysis.
- BLUP slopes and Delta-VAE scores are derived from overlapping longitudinal
  ECG information, although the Delta-VAE was never supervised on the slopes.
- Patient BLUP slopes are shrinkage estimates.
- Associations indicate alignment between two longitudinal representations;
  they do not imply mechanistic equivalence or causal biological progression.
- A Delta-VAE latent is a model-dependent ECG morphological change mode, not
  a clinical progression score.
- The signed orientation of a latent axis is model-dependent.

DEFAULT INPUTS
--------------
Results/20_Longitudinal_DeltaVAE_Interpretation/pair_delta_ecg_features.csv
Results/17_Longitudinal_Trajectories/patient_specific_blup_slopes.csv
Results/17_Longitudinal_Trajectories/patient_empirical_slopes.csv

OUTPUTS
-------
Results/22_DeltaVAE_TrajectoryIntegration/
    subject_delta_vae_summary.csv
    primary_blup_associations.csv
    heldout_test_blup_associations.csv
    sensitivity_ols_3plus_associations.csv
    strongest_target_per_delta_latent.csv
    strongest_delta_latent_per_target.csv
    run_summary.json
    figures/
        heatmap_primary_blup_rho.png
        heatmap_heldout_test_blup_rho.png
        heatmap_ols3_rho.png

RUN
---
python Scripts\\22_integrate_delta_vae_with_longitudinal_trajectories.py
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
SUBJECT_COL = "analysis_subject_key"

DELTA_LATENTS = [
    "D_z_00",
    "D_z_01",
    "D_z_02",
    "D_z_03",
]

TRAJECTORY_TARGETS = [
    "H_z_01",
    "H_z_05",
    "H_z_06",
    "d_N",
]

DELTA_LABELS = {
    "D_z_00": "repolarization change",
    "D_z_01": "precordial depol-ST/T change",
    "D_z_02": "lateral depolarization-force change",
    "D_z_03": "precordial activation redistribution",
}

TARGET_LABELS = {
    "H_z_01": "H_z_01",
    "H_z_05": "H_z_05",
    "H_z_06": "H_z_06",
    "d_N": "d_N",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Integrate VAE-3 subject-level change-mode summaries with "
            "Step-17 patient-specific longitudinal slopes."
        )
    )
    p.add_argument(
        "--pairs",
        type=Path,
        default=Path(
            "Results/20_Longitudinal_DeltaVAE_Interpretation/"
            "pair_delta_ecg_features.csv"
        ),
    )
    p.add_argument(
        "--blup-slopes",
        type=Path,
        default=Path(
            "Results/17_Longitudinal_Trajectories/"
            "patient_specific_blup_slopes.csv"
        ),
    )
    p.add_argument(
        "--empirical-slopes",
        type=Path,
        default=Path(
            "Results/17_Longitudinal_Trajectories/"
            "patient_empirical_slopes.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Results/22_DeltaVAE_TrajectoryIntegration"
        ),
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Subject bootstrap replicates for Spearman 95%% CI.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    return p.parse_args()


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


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

    x = x[m]
    y = y[m]
    n = len(x)

    if n < 5:
        return np.nan, np.nan, n

    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan, np.nan, n

    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p), int(n)


def bootstrap_spearman(x, y, n_bootstrap, rng):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)

    x = x[m]
    y = y[m]

    n = len(x)
    out = np.full(n_bootstrap, np.nan, dtype=float)

    if n < 5:
        return out

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]

        if np.std(xb) <= 1e-12 or np.std(yb) <= 1e-12:
            continue

        out[b] = stats.spearmanr(xb, yb).statistic

    return out


def percentile_ci(values, alpha=0.05):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return np.nan, np.nan

    return (
        float(np.quantile(v, alpha / 2.0)),
        float(np.quantile(v, 1.0 - alpha / 2.0)),
    )


def check_required_columns(df, cols, source_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{source_name} is missing required columns: {missing}"
        )


def build_subject_delta_summary(pair_df):
    required = [
        SUBJECT_COL,
        "split",
        *DELTA_LATENTS,
    ]
    check_required_columns(
        pair_df,
        required,
        "Delta-VAE pair table",
    )

    d = pair_df.copy()
    d[SUBJECT_COL] = d[SUBJECT_COL].astype(str)

    for z in DELTA_LATENTS:
        d[z] = safe_numeric(d[z])

    # Subject-wise split should be internally consistent.
    split_nunique = (
        d.groupby(SUBJECT_COL)["split"]
        .nunique(dropna=False)
    )
    bad = split_nunique[split_nunique != 1]
    if len(bad):
        raise RuntimeError(
            "VAE-3 split is not subject-wise for "
            f"{len(bad)} subjects."
        )

    rows = []

    for subject, g in d.groupby(
        SUBJECT_COL,
        sort=False,
    ):
        row = {
            SUBJECT_COL: subject,
            "vae3_split": str(g["split"].iloc[0]),
            "n_delta_pairs": int(len(g)),
        }

        if "delta_t_years" in g.columns:
            dt = safe_numeric(g["delta_t_years"])
            row["median_delta_t_years"] = float(
                np.nanmedian(dt)
            )
            row["total_pair_delta_t_years"] = float(
                np.nansum(dt)
            )

        for z in DELTA_LATENTS:
            vals = safe_numeric(g[z]).to_numpy(float)
            vals = vals[np.isfinite(vals)]

            if len(vals):
                row[f"median_{z}"] = float(
                    np.median(vals)
                )
                row[f"mean_{z}"] = float(
                    np.mean(vals)
                )
                row[f"sd_{z}"] = (
                    float(np.std(vals, ddof=1))
                    if len(vals) >= 2
                    else np.nan
                )
                row[f"median_abs_{z}"] = float(
                    np.median(np.abs(vals))
                )
                row[f"rms_{z}"] = float(
                    np.sqrt(np.mean(vals ** 2))
                )
            else:
                row[f"median_{z}"] = np.nan
                row[f"mean_{z}"] = np.nan
                row[f"sd_{z}"] = np.nan
                row[f"median_abs_{z}"] = np.nan
                row[f"rms_{z}"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def build_blup_wide(blup_df):
    required = [
        SUBJECT_COL,
        "outcome",
        "cohort",
        "patient_blup_slope",
    ]
    check_required_columns(
        blup_df,
        required,
        "Step-17 BLUP table",
    )

    d = blup_df.copy()
    d[SUBJECT_COL] = d[SUBJECT_COL].astype(str)

    d = d[
        (d["cohort"].astype(str) == "primary_2plus")
        & d["outcome"].isin(TRAJECTORY_TARGETS)
    ].copy()

    d["patient_blup_slope"] = safe_numeric(
        d["patient_blup_slope"]
    )

    dup = d.duplicated(
        [SUBJECT_COL, "outcome"],
        keep=False,
    )
    if dup.any():
        raise RuntimeError(
            "Duplicate subject/outcome rows found in primary BLUP table."
        )

    wide = d.pivot(
        index=SUBJECT_COL,
        columns="outcome",
        values="patient_blup_slope",
    ).reset_index()

    wide.columns.name = None
    wide = wide.rename(
        columns={
            target: f"blup_{target}"
            for target in TRAJECTORY_TARGETS
        }
    )
    return wide


def build_ols3_wide(empirical_df):
    required = [
        SUBJECT_COL,
        "outcome",
        "n_visits",
        "ols_slope_per_year",
    ]
    check_required_columns(
        empirical_df,
        required,
        "Step-17 empirical slope table",
    )

    d = empirical_df.copy()
    d[SUBJECT_COL] = d[SUBJECT_COL].astype(str)
    d["n_visits"] = safe_numeric(d["n_visits"])
    d["ols_slope_per_year"] = safe_numeric(
        d["ols_slope_per_year"]
    )

    d = d[
        d["outcome"].isin(TRAJECTORY_TARGETS)
        & (d["n_visits"] >= 3)
    ].copy()

    dup = d.duplicated(
        [SUBJECT_COL, "outcome"],
        keep=False,
    )
    if dup.any():
        raise RuntimeError(
            "Duplicate subject/outcome rows found in empirical slope table."
        )

    wide = d.pivot(
        index=SUBJECT_COL,
        columns="outcome",
        values="ols_slope_per_year",
    ).reset_index()

    wide.columns.name = None
    wide = wide.rename(
        columns={
            target: f"ols3_{target}"
            for target in TRAJECTORY_TARGETS
        }
    )
    return wide


def association_grid(
    merged,
    target_prefix,
    analysis_name,
    n_bootstrap,
    seed_sequence,
):
    rows = []

    children = seed_sequence.spawn(
        len(DELTA_LATENTS) * len(TRAJECTORY_TARGETS)
    )
    child_idx = 0

    for z in DELTA_LATENTS:
        x_col = f"median_{z}"

        for target in TRAJECTORY_TARGETS:
            y_col = f"{target_prefix}{target}"

            x = safe_numeric(
                merged[x_col]
            ).to_numpy(float)

            y = safe_numeric(
                merged[y_col]
            ).to_numpy(float)

            rho, p, n = spearman_safe(x, y)

            rng = np.random.default_rng(
                children[child_idx]
            )
            child_idx += 1

            boot = bootstrap_spearman(
                x,
                y,
                n_bootstrap,
                rng,
            )
            ci_low, ci_high = percentile_ci(boot)

            rows.append({
                "analysis": analysis_name,
                "delta_latent": z,
                "delta_latent_interpretation": DELTA_LABELS[z],
                "trajectory_target": target,
                "n_subjects": n,
                "spearman_rho": rho,
                "spearman_p": p,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "bootstrap_ci_excludes_zero": bool(
                    np.isfinite(ci_low)
                    and np.isfinite(ci_high)
                    and (
                        (ci_low > 0 and ci_high > 0)
                        or (ci_low < 0 and ci_high < 0)
                    )
                ),
            })

    out = pd.DataFrame(rows)
    out["q_fdr_16"] = bh_fdr(
        out["spearman_p"].to_numpy(float)
    )
    out["significant_fdr05"] = (
        out["q_fdr_16"] < 0.05
    )
    out["abs_rho"] = np.abs(
        out["spearman_rho"]
    )
    return out


def make_heatmap(table, title, out_path):
    mat = np.full(
        (
            len(DELTA_LATENTS),
            len(TRAJECTORY_TARGETS),
        ),
        np.nan,
        dtype=float,
    )

    qmat = np.full_like(
        mat,
        np.nan,
    )

    for i, z in enumerate(DELTA_LATENTS):
        for j, target in enumerate(TRAJECTORY_TARGETS):
            d = table[
                (table["delta_latent"] == z)
                & (
                    table["trajectory_target"]
                    == target
                )
            ]
            if len(d) == 1:
                mat[i, j] = float(
                    d["spearman_rho"].iloc[0]
                )
                qmat[i, j] = float(
                    d["q_fdr_16"].iloc[0]
                )

    fig, ax = plt.subplots(
        figsize=(7.6, 5.2)
    )

    im = ax.imshow(
        mat,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(
        np.arange(
            len(TRAJECTORY_TARGETS)
        )
    )
    ax.set_xticklabels(
        [
            TARGET_LABELS[t]
            for t in TRAJECTORY_TARGETS
        ]
    )

    ax.set_yticks(
        np.arange(
            len(DELTA_LATENTS)
        )
    )
    ax.set_yticklabels(
        DELTA_LATENTS
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                star = (
                    "*"
                    if (
                        np.isfinite(qmat[i, j])
                        and qmat[i, j] < 0.05
                    )
                    else ""
                )
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:+.2f}{star}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    ax.set_xlabel(
        "Step-17 longitudinal trajectory"
    )
    ax.set_ylabel(
        "VAE-3 longitudinal change mode"
    )
    ax.set_title(title)

    cbar = fig.colorbar(
        im,
        ax=ax,
        shrink=0.85,
    )
    cbar.set_label(
        "Spearman rho"
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def strongest_summaries(
    primary,
    test_only,
    ols3,
):
    # Strongest Step-17 trajectory target per Delta-VAE latent.
    rows_latent = []

    for z in DELTA_LATENTS:
        d = primary[
            primary["delta_latent"] == z
        ].copy()

        d = d.sort_values(
            "abs_rho",
            ascending=False,
        )

        if len(d) == 0:
            continue

        r = d.iloc[0]
        target = r["trajectory_target"]

        t = test_only[
            (test_only["delta_latent"] == z)
            & (
                test_only["trajectory_target"]
                == target
            )
        ]

        o = ols3[
            (ols3["delta_latent"] == z)
            & (
                ols3["trajectory_target"]
                == target
            )
        ]

        rows_latent.append({
            "delta_latent": z,
            "interpretation": DELTA_LABELS[z],
            "strongest_primary_target": target,
            "primary_rho": float(
                r["spearman_rho"]
            ),
            "primary_q_fdr_16": float(
                r["q_fdr_16"]
            ),
            "primary_ci95_low": float(
                r["bootstrap_ci95_low"]
            ),
            "primary_ci95_high": float(
                r["bootstrap_ci95_high"]
            ),
            "heldout_test_rho_same_target": (
                float(
                    t["spearman_rho"].iloc[0]
                )
                if len(t)
                else np.nan
            ),
            "heldout_test_q_fdr_16_same_target": (
                float(
                    t["q_fdr_16"].iloc[0]
                )
                if len(t)
                else np.nan
            ),
            "ols3_rho_same_target": (
                float(
                    o["spearman_rho"].iloc[0]
                )
                if len(o)
                else np.nan
            ),
            "ols3_q_fdr_16_same_target": (
                float(
                    o["q_fdr_16"].iloc[0]
                )
                if len(o)
                else np.nan
            ),
        })

    # Strongest Delta-VAE latent per Step-17 target.
    rows_target = []

    for target in TRAJECTORY_TARGETS:
        d = primary[
            primary["trajectory_target"]
            == target
        ].copy()

        d = d.sort_values(
            "abs_rho",
            ascending=False,
        )

        if len(d) == 0:
            continue

        r = d.iloc[0]
        z = r["delta_latent"]

        rows_target.append({
            "trajectory_target": target,
            "strongest_delta_latent": z,
            "delta_latent_interpretation": DELTA_LABELS[z],
            "primary_rho": float(
                r["spearman_rho"]
            ),
            "primary_q_fdr_16": float(
                r["q_fdr_16"]
            ),
            "primary_ci95_low": float(
                r["bootstrap_ci95_low"]
            ),
            "primary_ci95_high": float(
                r["bootstrap_ci95_high"]
            ),
        })

    return (
        pd.DataFrame(rows_latent),
        pd.DataFrame(rows_target),
    )


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig_dir = (
        args.output_dir
        / "figures"
    )
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        args.pairs,
        args.blup_slopes,
        args.empirical_slopes,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input not found: {path}"
            )

    print("=" * 86)
    print(
        "22 - Delta-VAE integration with Step-17 longitudinal trajectories"
    )
    print("=" * 86)
    print("Pairs      :", args.pairs)
    print("BLUP       :", args.blup_slopes)
    print("Empirical  :", args.empirical_slopes)
    print("Output     :", args.output_dir)
    print("Bootstrap  :", args.n_bootstrap)

    pairs = pd.read_csv(
        args.pairs
    )
    blup = pd.read_csv(
        args.blup_slopes
    )
    empirical = pd.read_csv(
        args.empirical_slopes
    )

    subject_delta = (
        build_subject_delta_summary(
            pairs
        )
    )
    subject_delta.to_csv(
        args.output_dir
        / "subject_delta_vae_summary.csv",
        index=False,
    )

    blup_wide = build_blup_wide(
        blup
    )
    ols3_wide = build_ols3_wide(
        empirical
    )

    merged_blup = subject_delta.merge(
        blup_wide,
        on=SUBJECT_COL,
        how="inner",
        validate="one_to_one",
    )

    merged_ols3 = subject_delta.merge(
        ols3_wide,
        on=SUBJECT_COL,
        how="inner",
        validate="one_to_one",
    )

    merged_test = merged_blup[
        merged_blup["vae3_split"]
        .astype(str)
        .str.lower()
        .eq("test")
    ].copy()

    print()
    print("[subject-level integration cohorts]")
    print(
        "Delta-VAE subjects      :",
        subject_delta[SUBJECT_COL].nunique(),
    )
    print(
        "BLUP overlap            :",
        merged_blup[SUBJECT_COL].nunique(),
    )
    print(
        "Held-out test overlap   :",
        merged_test[SUBJECT_COL].nunique(),
    )
    print(
        "OLS >=3 overlap         :",
        merged_ols3[SUBJECT_COL].nunique(),
    )

    master_seed = np.random.SeedSequence(
        args.seed
    )
    seed_primary, seed_test, seed_ols = (
        master_seed.spawn(3)
    )

    primary = association_grid(
        merged=merged_blup,
        target_prefix="blup_",
        analysis_name="primary_blup_all_subjects",
        n_bootstrap=args.n_bootstrap,
        seed_sequence=seed_primary,
    )

    heldout = association_grid(
        merged=merged_test,
        target_prefix="blup_",
        analysis_name="sensitivity_heldout_test_blup",
        n_bootstrap=args.n_bootstrap,
        seed_sequence=seed_test,
    )

    ols3 = association_grid(
        merged=merged_ols3,
        target_prefix="ols3_",
        analysis_name="sensitivity_ols_3plus",
        n_bootstrap=args.n_bootstrap,
        seed_sequence=seed_ols,
    )

    primary.to_csv(
        args.output_dir
        / "primary_blup_associations.csv",
        index=False,
    )
    heldout.to_csv(
        args.output_dir
        / "heldout_test_blup_associations.csv",
        index=False,
    )
    ols3.to_csv(
        args.output_dir
        / "sensitivity_ols_3plus_associations.csv",
        index=False,
    )

    strongest_latent, strongest_target = (
        strongest_summaries(
            primary,
            heldout,
            ols3,
        )
    )

    strongest_latent.to_csv(
        args.output_dir
        / "strongest_target_per_delta_latent.csv",
        index=False,
    )
    strongest_target.to_csv(
        args.output_dir
        / "strongest_delta_latent_per_target.csv",
        index=False,
    )

    make_heatmap(
        primary,
        (
            "Delta-VAE modes vs Step-17 patient BLUP slopes\n"
            "all subjects; * FDR q < 0.05"
        ),
        fig_dir
        / "heatmap_primary_blup_rho.png",
    )

    make_heatmap(
        heldout,
        (
            "Delta-VAE modes vs Step-17 BLUP slopes\n"
            "held-out VAE-3 test subjects; * FDR q < 0.05"
        ),
        fig_dir
        / "heatmap_heldout_test_blup_rho.png",
    )

    make_heatmap(
        ols3,
        (
            "Delta-VAE modes vs empirical OLS slopes\n"
            "subjects with >=3 visits; * FDR q < 0.05"
        ),
        fig_dir
        / "heatmap_ols3_rho.png",
    )

    n_primary_sig = int(
        primary["significant_fdr05"].sum()
    )
    n_heldout_sig = int(
        heldout["significant_fdr05"].sum()
    )
    n_ols3_sig = int(
        ols3["significant_fdr05"].sum()
    )

    # How many primary-significant associations preserve their sign
    # in the two sensitivity analyses?
    primary_sig = primary[
        primary["significant_fdr05"]
    ].copy()

    sign_rows = []

    for _, r in primary_sig.iterrows():
        z = r["delta_latent"]
        target = r["trajectory_target"]
        sign_primary = np.sign(
            r["spearman_rho"]
        )

        rt = heldout[
            (heldout["delta_latent"] == z)
            & (
                heldout["trajectory_target"]
                == target
            )
        ]

        ro = ols3[
            (ols3["delta_latent"] == z)
            & (
                ols3["trajectory_target"]
                == target
            )
        ]

        sign_rows.append({
            "delta_latent": z,
            "trajectory_target": target,
            "primary_rho": r["spearman_rho"],
            "heldout_test_rho": (
                rt["spearman_rho"].iloc[0]
                if len(rt)
                else np.nan
            ),
            "ols3_rho": (
                ro["spearman_rho"].iloc[0]
                if len(ro)
                else np.nan
            ),
            "same_sign_heldout_test": (
                bool(
                    np.sign(
                        rt["spearman_rho"].iloc[0]
                    )
                    == sign_primary
                )
                if len(rt)
                and np.isfinite(
                    rt["spearman_rho"].iloc[0]
                )
                else False
            ),
            "same_sign_ols3": (
                bool(
                    np.sign(
                        ro["spearman_rho"].iloc[0]
                    )
                    == sign_primary
                )
                if len(ro)
                and np.isfinite(
                    ro["spearman_rho"].iloc[0]
                )
                else False
            ),
        })

    sign_df = pd.DataFrame(
        sign_rows
    )
    sign_df.to_csv(
        args.output_dir
        / "primary_significant_sensitivity_signs.csv",
        index=False,
    )

    summary = {
        "script": (
            "22_integrate_delta_vae_with_longitudinal_trajectories.py"
        ),
        "version": "1.0",
        "scientific_role": (
            "Subject-level integration of HCM-only Delta-VAE change modes "
            "with Step-17 patient-specific longitudinal trajectory slopes"
        ),
        "inputs": {
            "delta_vae_pairs": str(args.pairs),
            "blup_slopes": str(args.blup_slopes),
            "empirical_slopes": str(args.empirical_slopes),
        },
        "prespecified_delta_latents": DELTA_LATENTS,
        "prespecified_trajectory_targets": TRAJECTORY_TARGETS,
        "primary_estimand": (
            "Spearman association between subject median signed Delta-VAE "
            "latent score and Step-17 primary patient BLUP slope"
        ),
        "multiplicity": (
            "Benjamini-Hochberg FDR across 16 prespecified associations "
            "within each analysis family"
        ),
        "n_subjects_delta_vae": int(
            subject_delta[SUBJECT_COL].nunique()
        ),
        "n_subjects_primary_blup_overlap": int(
            merged_blup[SUBJECT_COL].nunique()
        ),
        "n_subjects_heldout_test_overlap": int(
            merged_test[SUBJECT_COL].nunique()
        ),
        "n_subjects_ols3_overlap": int(
            merged_ols3[SUBJECT_COL].nunique()
        ),
        "n_primary_fdr_significant": n_primary_sig,
        "n_heldout_test_fdr_significant": n_heldout_sig,
        "n_ols3_fdr_significant": n_ols3_sig,
        "strongest_primary_target_per_delta_latent": (
            strongest_latent.to_dict(
                orient="records"
            )
        ),
        "strongest_primary_delta_latent_per_target": (
            strongest_target.to_dict(
                orient="records"
            )
        ),
        "interpretive_notes": [
            (
                "The primary analysis is subject-level, so each subject "
                "contributes one Delta-VAE summary and one trajectory slope."
            ),
            (
                "The signed median Delta-VAE latent across consecutive pairs "
                "is used as the primary subject-level change-mode summary."
            ),
            (
                "The held-out test-subject analysis is a sensitivity analysis "
                "for representation generalization, not a separate external cohort."
            ),
            (
                "The >=3-visit empirical OLS analysis is a model-independent "
                "sensitivity to BLUP shrinkage."
            ),
            (
                "BLUP slopes and Delta-VAE scores are derived from overlapping "
                "longitudinal ECG information; their association is an integration "
                "of representations, not independent validation."
            ),
            (
                "Associations describe alignment between model-derived electrical "
                "change phenotypes and trajectory heterogeneity; they do not imply "
                "causal biological progression."
            ),
        ],
    }

    (
        args.output_dir
        / "run_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 86)
    print("STEP 22 COMPLETE")
    print("=" * 86)
    print(
        "Primary FDR-significant associations:",
        n_primary_sig,
        "/ 16",
    )
    print(
        "Held-out test FDR-significant associations:",
        n_heldout_sig,
        "/ 16",
    )
    print(
        "OLS >=3 FDR-significant associations:",
        n_ols3_sig,
        "/ 16",
    )

    if len(strongest_latent):
        print()
        print("[strongest primary target per Delta-VAE latent]")
        print(
            strongest_latent[
                [
                    "delta_latent",
                    "strongest_primary_target",
                    "primary_rho",
                    "primary_q_fdr_16",
                    "heldout_test_rho_same_target",
                    "ols3_rho_same_target",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:+.4f}",
            )
        )

    print()
    print("Results:", args.output_dir)


if __name__ == "__main__":
    main()
