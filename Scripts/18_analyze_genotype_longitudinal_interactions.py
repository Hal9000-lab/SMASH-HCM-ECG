#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18_analyze_genotype_longitudinal_interactions.py

SMASH-HCM -- Step 18
Genotype-by-time interaction analysis in frozen ECG phenotype manifolds.

SCIENTIFIC AIM
--------------
Test whether sarcomeric genotype modifies the longitudinal electrical trajectory
identified in Step 17, while keeping baseline phenotype and trajectory effects
conceptually separate.

PRIMARY GENETIC CONTRAST
------------------------
Strict sarcomeric P/LP positive (Sarc+) versus genetically negative (Sarc-).

Sarc+:
    - exact, non-ambiguous SHaRe mapping
    - Primary_Diagnosis == HCM
    - at least one Pathogenic / Likely Pathogenic variant in a core sarcomeric
      HCM gene:
          MYBPC3, MYH7, TNNT2, TNNI3, TPM1, ACTC1, MYL2, MYL3

Sarc-:
    - exact, non-ambiguous SHaRe mapping
    - Primary_Diagnosis == HCM
    - genetic_status_raw == G(-)
    - no P/LP variant identified in the crosswalk

G(0), G(U), VUS-only, non-core P/LP, unmatched and ambiguous mappings are NOT
forced into either primary group. They remain available for cohort accounting.

PRIMARY LONGITUDINAL ENDPOINTS
------------------------------
The targeted trajectory-sensitive phenotypes identified in Step 17:
    H_z_01  precordial R/S transition-balance phenotype
    H_z_05  lateral/LV-oriented depolarization-force phenotype
    H_z_06  ventricular activation redistribution / QRS spatial orientation
    d_N     distance from the Normal-anchored latent distribution

The key coefficient is the genotype-by-time interaction:

    y_ij = beta0 + beta1*time_ij + beta2*genotype_i
           + beta3*(time_ij*genotype_i)
           + beta4*baseline_age_i
           + beta5*(time_ij*baseline_age_i)
           + b0_i + b1_i*time_ij + epsilon_ij

where genotype=0 for Sarc- and genotype=1 for Sarc+.

Interpretation:
    beta2  = baseline Sarc+ vs Sarc- difference at mean baseline age
    beta1  = annual slope in Sarc-
    beta3  = DIFFERENCE in annual slope, Sarc+ minus Sarc-
    beta1 + beta3 = annual slope in Sarc+

The primary statistical question is therefore:

    H0: beta3 = 0

Benjamini-Hochberg FDR is applied across the four primary interaction tests.

SENSITIVITY / SECONDARY ANALYSES
--------------------------------
1. Unadjusted genotype-by-time mixed model on the same primary endpoints.
2. Any P/LP variant versus G(-), irrespective of gene, as a broader secondary
   genetic definition.
3. H_z_02 (repolarization phenotype) is retained as a prespecified comparator:
   it was strongly phenotype-defining cross-sectionally but showed no average
   temporal drift in Step 17.
4. Exploratory genotype-by-time interactions are estimated across all available
   dual-manifold primary outcomes from Step 17. These are explicitly marked
   exploratory and FDR-corrected only within that exploratory family.
5. If sample size permits, gene-specific secondary models are run for MYBPC3
   and MYH7 carriers versus Sarc-.
6. Subject-wise empirical OLS slopes are summarized by genotype as a
   model-independent cross-check of the mixed-effects interaction.

IMPORTANT CAUTIONS
------------------
- This is an observational genotype-trajectory association analysis.
- A genotype-by-time interaction does not establish a causal molecular
  mechanism.
- Latent coordinates are model-derived electrical phenotype coordinates.
- d_N is a model-based distance from the Normal reference manifold, not a
  clinical severity score.
- RE_N, RE_H and Delta_RE, if explored, remain model-dependent reconstruction
  quantities and are not direct measures of HCM specificity.
- Genetic status is reconstructed conservatively from the crosswalk. Raw G(+)
  is NOT automatically equated with a P/LP sarcomeric genotype.

EXPECTED INPUTS
---------------
Data/SMASH_HCM_SHaRe_genetics_crosswalk.xlsx
    sheet: Subject_Genetics

Results/16_Longitudinal_DualManifold/longitudinal_dual_manifold.csv

OUTPUT DIRECTORY
----------------
Results/18_Genotype_Longitudinal/

KEY OUTPUTS
-----------
genotype_subject_cohort.csv
cohort_flow.csv
primary_genotype_time_interactions.csv
primary_unadjusted_sensitivity.csv
secondary_any_plp_interactions.csv
comparator_hz02_interaction.csv
exploratory_all_outcomes_interactions.csv
gene_specific_interactions.csv
empirical_subject_slopes_by_genotype.csv
empirical_slope_group_summary.csv
run_summary.json
figures/
    primary_interaction_effects.png
    primary_group_slopes.png
    primary_empirical_slope_distributions.png

RUN
---
python Scripts\\18_analyze_genotype_longitudinal_interactions.py

DEPENDENCIES
------------
numpy pandas scipy statsmodels matplotlib openpyxl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf


SEED = 20261005
SUBJECT_LONG = "analysis_subject_key"
SUBJECT_GEN = "subject_key"
TIME_SOURCE = "time_from_first_valid_ecg_years"
TIME_MODEL = "time_years"

CORE_SARCOMERE_GENES = {
    "MYBPC3",
    "MYH7",
    "TNNT2",
    "TNNI3",
    "TPM1",
    "ACTC1",
    "MYL2",
    "MYL3",
}

PRIMARY_OUTCOMES = [
    "H_z_01",
    "H_z_05",
    "H_z_06",
    "d_N",
]

COMPARATOR_OUTCOMES = [
    "H_z_02",
]

NORMAL_LATENTS = [f"N_z_{k:02d}" for k in range(16)]
HCM_LATENTS = [f"H_z_{k:02d}" for k in range(8)]
GLOBAL_OUTCOMES = [
    "d_N",
    "d_HCM",
    "RE_N_mse",
    "RE_H_mse",
]
EXPLORATORY_CANDIDATES = NORMAL_LATENTS + HCM_LATENTS + GLOBAL_OUTCOMES

OUTCOME_LABELS = {
    "H_z_01": "Precordial R/S transition-balance phenotype",
    "H_z_02": "Precordial/lateral repolarization phenotype",
    "H_z_05": "Lateral/LV-oriented depolarization-force phenotype",
    "H_z_06": "Ventricular activation redistribution / QRS spatial orientation",
    "d_N": "Normal-manifold distance",
    "d_HCM": "HCM-manifold distance",
    "RE_N_mse": "Normal-VAE reconstruction MSE",
    "RE_H_mse": "HCM-VAE reconstruction MSE",
}

GENO_NEG = "Sarc-"
GENO_POS = "Sarc+"
ANY_NEG = "G(-)"
ANY_POS = "Any_PLP+"


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--genetics",
        type=Path,
        default=Path("Data/SMASH_HCM_SHaRe_genetics_crosswalk.xlsx"),
    )
    p.add_argument(
        "--longitudinal",
        type=Path,
        default=Path(
            "Results/16_Longitudinal_DualManifold/"
            "longitudinal_dual_manifold.csv"
        ),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Results/18_Genotype_Longitudinal"),
    )
    p.add_argument("--min-visits", type=int, default=2)
    p.add_argument("--min-gene-carriers", type=int, default=25)
    p.add_argument("--min-negative-controls", type=int, default=40)
    p.add_argument("--maxiter", type=int, default=2500)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--ml",
        action="store_true",
        help="Use ML instead of REML for mixed-model estimation.",
    )
    return p.parse_args()


# ============================================================================
# General utilities
# ============================================================================


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def clean_string(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def boolish(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_gene_set(x) -> set:
    s = clean_string(x).upper()
    if not s:
        return set()
    parts = re.split(r"[;,|/]+", s)
    return {p.strip() for p in parts if p.strip()}


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


def safe_sd(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return np.nan
    sd = float(np.std(a, ddof=1))
    return sd if sd > 0 else np.nan


def z975():
    return float(stats.norm.ppf(0.975))


def finite_or_none(x):
    try:
        y = float(x)
    except Exception:
        return None
    return y if np.isfinite(y) else None


# ============================================================================
# Genetics loading and conservative classification
# ============================================================================


def load_genetics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Genetics workbook not found: {path}")

    try:
        g = pd.read_excel(path, sheet_name="Subject_Genetics", engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError(
            "Reading the genetics workbook requires openpyxl. Install with: "
            "python -m pip install openpyxl"
        ) from exc

    required = {
        "subject_key",
        "mapping_status",
        "Primary_Diagnosis",
        "genetic_status_raw",
        "has_P_or_LP_variant",
        "P_LP_genes",
    }
    missing = sorted(required - set(g.columns))
    if missing:
        raise KeyError(
            "Subject_Genetics is missing required columns: " + ", ".join(missing)
        )

    g = g.copy()
    g[SUBJECT_GEN] = g[SUBJECT_GEN].astype(str).str.strip()
    g["mapping_status"] = g["mapping_status"].fillna("").astype(str).str.strip()
    g["Primary_Diagnosis"] = g["Primary_Diagnosis"].fillna("").astype(str).str.strip()
    g["genetic_status_raw"] = g["genetic_status_raw"].fillna("").astype(str).str.strip()
    g["has_P_or_LP_variant_bool"] = g["has_P_or_LP_variant"].map(boolish)
    g["P_LP_gene_set"] = g["P_LP_genes"].map(parse_gene_set)
    g["sarcomere_PLP_gene_set"] = g["P_LP_gene_set"].map(
        lambda s: set(s).intersection(CORE_SARCOMERE_GENES)
    )
    g["has_core_sarcomere_PLP"] = g["sarcomere_PLP_gene_set"].map(lambda s: len(s) > 0)
    g["sarcomere_PLP_genes"] = g["sarcomere_PLP_gene_set"].map(
        lambda s: "; ".join(sorted(s)) if s else ""
    )

    exact = g["mapping_status"].eq("matched")
    hcm_dx = g["Primary_Diagnosis"].str.upper().eq("HCM")
    genetic_negative = g["genetic_status_raw"].eq("G(-)") & (~g["has_P_or_LP_variant_bool"])

    g["primary_genotype_group"] = "Excluded"
    g.loc[exact & hcm_dx & genetic_negative, "primary_genotype_group"] = GENO_NEG
    g.loc[exact & hcm_dx & g["has_core_sarcomere_PLP"], "primary_genotype_group"] = GENO_POS

    # Broad secondary definition: any P/LP variant versus G(-).
    g["any_plp_group"] = "Excluded"
    g.loc[exact & hcm_dx & genetic_negative, "any_plp_group"] = ANY_NEG
    g.loc[exact & hcm_dx & g["has_P_or_LP_variant_bool"], "any_plp_group"] = ANY_POS

    def exclusion_reason(r):
        if r["primary_genotype_group"] != "Excluded":
            return "included_primary"
        if r["mapping_status"] != "matched":
            return f"mapping_{r['mapping_status'] or 'missing'}"
        if str(r["Primary_Diagnosis"]).upper() != "HCM":
            return "primary_diagnosis_not_HCM"
        if r["genetic_status_raw"] == "G(U)":
            return "uncertain_genetic_status_GU"
        if r["genetic_status_raw"] == "G(0)":
            return "no_informative_genetic_result_G0"
        if r["has_P_or_LP_variant_bool"] and not r["has_core_sarcomere_PLP"]:
            return "PLP_non_core_sarcomere_only"
        if r["genetic_status_raw"] == "G(+)" and not r["has_core_sarcomere_PLP"]:
            return "raw_Gplus_without_core_sarcomere_PLP"
        return "other_exclusion"

    g["primary_exclusion_reason"] = g.apply(exclusion_reason, axis=1)
    return g


# ============================================================================
# Longitudinal loading / merge / baseline age
# ============================================================================


def load_longitudinal(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Longitudinal CSV not found: {path}")
    d = pd.read_csv(path)
    required = {SUBJECT_LONG, TIME_SOURCE, "valid_for_latent_projection"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise KeyError("Longitudinal file is missing: " + ", ".join(missing))

    d = d.copy()
    d[SUBJECT_LONG] = d[SUBJECT_LONG].astype(str).str.strip()
    d[TIME_MODEL] = numeric(d[TIME_SOURCE])
    d["valid_projection_bool"] = d["valid_for_latent_projection"].map(boolish)
    d = d[d["valid_projection_bool"] & np.isfinite(d[TIME_MODEL])].copy()
    return d


def derive_baseline_age(merged: pd.DataFrame) -> pd.DataFrame:
    d = merged.copy()

    if "ecg_time" in d.columns:
        d["ecg_time_parsed"] = pd.to_datetime(d["ecg_time"], errors="coerce")
    else:
        d["ecg_time_parsed"] = pd.NaT

    if "birthdate_signal" in d.columns:
        d["birthdate_parsed"] = pd.to_datetime(d["birthdate_signal"], errors="coerce")
    else:
        # subject_key convention: <id>_YYYY-MM-DD
        birth = d[SUBJECT_LONG].str.extract(r"(\d{4}-\d{2}-\d{2})$")[0]
        d["birthdate_parsed"] = pd.to_datetime(birth, errors="coerce")

    baseline_rows = (
        d.sort_values([SUBJECT_LONG, TIME_MODEL])
        .groupby(SUBJECT_LONG, as_index=False)
        .first()
    )

    baseline_age = (
        (baseline_rows["ecg_time_parsed"] - baseline_rows["birthdate_parsed"])
        .dt.total_seconds()
        / (365.2425 * 24 * 3600)
    )

    age_map = dict(zip(baseline_rows[SUBJECT_LONG], baseline_age))
    d["baseline_age_years"] = d[SUBJECT_LONG].map(age_map)

    return d


def merge_data(longitudinal: pd.DataFrame, genetics: pd.DataFrame) -> pd.DataFrame:
    gen_cols = [c for c in genetics.columns if c != "P_LP_gene_set" and c != "sarcomere_PLP_gene_set"]
    g = genetics[gen_cols].copy()

    if g[SUBJECT_GEN].duplicated().any():
        dup = g.loc[g[SUBJECT_GEN].duplicated(keep=False), SUBJECT_GEN].unique().tolist()
        raise ValueError(
            "Subject_Genetics contains duplicate subject_key values; cannot perform one-to-one merge. "
            f"Examples: {dup[:10]}"
        )

    d = longitudinal.merge(
        g,
        left_on=SUBJECT_LONG,
        right_on=SUBJECT_GEN,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    return derive_baseline_age(d)


# ============================================================================
# Cohort selection
# ============================================================================


def subject_visit_counts(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, g in d.groupby(SUBJECT_LONG, sort=False):
        t = numeric(g[TIME_MODEL]).to_numpy(float)
        t = t[np.isfinite(t)]
        rows.append(
            {
                SUBJECT_LONG: subject,
                "n_valid_visits": int(len(g)),
                "n_distinct_times": int(len(np.unique(np.round(t, 10)))) if len(t) else 0,
                "followup_years": float(np.max(t) - np.min(t)) if len(t) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def select_group_cohort(
    merged: pd.DataFrame,
    group_col: str,
    groups: Sequence[str],
    min_visits: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = merged[merged[group_col].isin(groups)].copy()
    counts = subject_visit_counts(d)
    counts["eligible_longitudinal"] = (
        (counts["n_valid_visits"] >= min_visits)
        & (counts["n_distinct_times"] >= 2)
        & (counts["followup_years"] > 0)
    )
    keep = set(counts.loc[counts["eligible_longitudinal"], SUBJECT_LONG].astype(str))
    return d[d[SUBJECT_LONG].astype(str).isin(keep)].copy(), counts


def add_binary_and_age_center(
    d: pd.DataFrame,
    group_col: str,
    positive_label: str,
) -> pd.DataFrame:
    x = d.copy()
    x["genotype_binary"] = (x[group_col] == positive_label).astype(float)

    subject_age = (
        x[[SUBJECT_LONG, "baseline_age_years"]]
        .drop_duplicates(SUBJECT_LONG)
        .copy()
    )
    mean_age = float(numeric(subject_age["baseline_age_years"]).mean())
    x["baseline_age_c"] = numeric(x["baseline_age_years"]) - mean_age
    x["baseline_age_center"] = mean_age
    return x


# ============================================================================
# Mixed-model engine
# ============================================================================


def prepare_outcome_data(d: pd.DataFrame, outcome: str, adjusted: bool) -> pd.DataFrame:
    cols = [SUBJECT_LONG, TIME_MODEL, "genotype_binary", outcome]
    if adjusted:
        cols.append("baseline_age_c")
    x = d[cols].copy()
    x[outcome] = numeric(x[outcome])
    x[TIME_MODEL] = numeric(x[TIME_MODEL])
    x["genotype_binary"] = numeric(x["genotype_binary"])
    if adjusted:
        x["baseline_age_c"] = numeric(x["baseline_age_c"])

    finite = np.isfinite(x[outcome]) & np.isfinite(x[TIME_MODEL]) & np.isfinite(x["genotype_binary"])
    if adjusted:
        finite &= np.isfinite(x["baseline_age_c"])
    x = x[finite].copy()

    keep = []
    for subject, g in x.groupby(SUBJECT_LONG):
        if len(g) >= 2 and g[TIME_MODEL].nunique() >= 2:
            keep.append(subject)
    x = x[x[SUBJECT_LONG].isin(keep)].copy()
    return x


def baseline_outcome_sd(x: pd.DataFrame, outcome: str) -> float:
    b = (
        x.sort_values([SUBJECT_LONG, TIME_MODEL])
        .groupby(SUBJECT_LONG, as_index=False)
        .first()
    )
    return safe_sd(numeric(b[outcome]))


def fit_interaction_model(
    d: pd.DataFrame,
    outcome: str,
    adjusted: bool,
    reml: bool,
    maxiter: int,
) -> Dict:
    x = prepare_outcome_data(d, outcome, adjusted=adjusted)
    n_obs = int(len(x))
    n_subjects = int(x[SUBJECT_LONG].nunique())
    n_pos = int(x.loc[x["genotype_binary"] == 1, SUBJECT_LONG].nunique())
    n_neg = int(x.loc[x["genotype_binary"] == 0, SUBJECT_LONG].nunique())

    if min(n_pos, n_neg) < 10 or n_subjects < 30:
        return {
            "success": False,
            "outcome": outcome,
            "adjusted_for_baseline_age": adjusted,
            "failure_reason": "insufficient_group_sample_size",
            "n_obs": n_obs,
            "n_subjects": n_subjects,
            "n_positive_subjects": n_pos,
            "n_negative_subjects": n_neg,
        }

    if adjusted:
        formula = (
            f"{outcome} ~ {TIME_MODEL} + genotype_binary + "
            f"{TIME_MODEL}:genotype_binary + baseline_age_c + "
            f"{TIME_MODEL}:baseline_age_c"
        )
    else:
        formula = (
            f"{outcome} ~ {TIME_MODEL} + genotype_binary + "
            f"{TIME_MODEL}:genotype_binary"
        )

    attempts = []
    result = None
    structure = None
    optimizer = None

    for method in ["lbfgs", "powell", "cg"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    data=x,
                    groups=x[SUBJECT_LONG],
                    re_formula=f"~{TIME_MODEL}",
                )
                res = model.fit(
                    reml=reml,
                    method=method,
                    maxiter=maxiter,
                    disp=False,
                )
            attempts.append(
                {
                    "structure": "random_intercept_slope",
                    "optimizer": method,
                    "converged": bool(getattr(res, "converged", False)),
                    "warnings": " | ".join(str(w.message) for w in caught),
                }
            )
            if bool(getattr(res, "converged", False)):
                result = res
                structure = "random_intercept_slope"
                optimizer = method
                break
        except Exception as exc:
            attempts.append(
                {
                    "structure": "random_intercept_slope",
                    "optimizer": method,
                    "converged": False,
                    "warnings": f"{type(exc).__name__}: {exc}",
                }
            )

    # Explicit fallback. Interaction inference remains available, but no
    # patient-specific random-slope interpretation should be made from it.
    if result is None:
        for method in ["lbfgs", "powell"]:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = smf.mixedlm(
                        formula,
                        data=x,
                        groups=x[SUBJECT_LONG],
                        re_formula="1",
                    )
                    res = model.fit(
                        reml=reml,
                        method=method,
                        maxiter=maxiter,
                        disp=False,
                    )
                attempts.append(
                    {
                        "structure": "random_intercept_only",
                        "optimizer": method,
                        "converged": bool(getattr(res, "converged", False)),
                        "warnings": " | ".join(str(w.message) for w in caught),
                    }
                )
                if bool(getattr(res, "converged", False)):
                    result = res
                    structure = "random_intercept_only"
                    optimizer = method
                    break
            except Exception as exc:
                attempts.append(
                    {
                        "structure": "random_intercept_only",
                        "optimizer": method,
                        "converged": False,
                        "warnings": f"{type(exc).__name__}: {exc}",
                    }
                )

    if result is None:
        return {
            "success": False,
            "outcome": outcome,
            "adjusted_for_baseline_age": adjusted,
            "failure_reason": json.dumps(attempts),
            "n_obs": n_obs,
            "n_subjects": n_subjects,
            "n_positive_subjects": n_pos,
            "n_negative_subjects": n_neg,
        }

    params = result.params
    cov = result.cov_params()
    b_time = float(params.get(TIME_MODEL, np.nan))
    b_geno = float(params.get("genotype_binary", np.nan))
    int_name = f"{TIME_MODEL}:genotype_binary"
    b_int = float(params.get(int_name, np.nan))

    se_time = float(result.bse.get(TIME_MODEL, np.nan))
    se_geno = float(result.bse.get("genotype_binary", np.nan))
    se_int = float(result.bse.get(int_name, np.nan))
    p_geno = float(result.pvalues.get("genotype_binary", np.nan))
    p_int = float(result.pvalues.get(int_name, np.nan))

    # Group-specific annual slopes.
    slope_neg = b_time
    slope_pos = b_time + b_int

    var_time = float(cov.loc[TIME_MODEL, TIME_MODEL]) if TIME_MODEL in cov.index else np.nan
    var_int = float(cov.loc[int_name, int_name]) if int_name in cov.index else np.nan
    cov_ti = (
        float(cov.loc[TIME_MODEL, int_name])
        if TIME_MODEL in cov.index and int_name in cov.columns
        else np.nan
    )

    se_slope_neg = math.sqrt(max(var_time, 0.0)) if np.isfinite(var_time) else np.nan
    var_pos = var_time + var_int + 2.0 * cov_ti if all(np.isfinite([var_time, var_int, cov_ti])) else np.nan
    se_slope_pos = math.sqrt(max(var_pos, 0.0)) if np.isfinite(var_pos) else np.nan

    z = z975()
    baseline_sd = baseline_outcome_sd(x, outcome)

    cov_re = np.asarray(result.cov_re, dtype=float)
    random_slope_var = (
        float(cov_re[1, 1])
        if structure == "random_intercept_slope" and cov_re.ndim == 2 and cov_re.shape[0] >= 2
        else np.nan
    )

    row = {
        "success": True,
        "outcome": outcome,
        "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
        "adjusted_for_baseline_age": adjusted,
        "model_structure": structure,
        "optimizer": optimizer,
        "converged": bool(getattr(result, "converged", False)),
        "n_obs": n_obs,
        "n_subjects": n_subjects,
        "n_positive_subjects": n_pos,
        "n_negative_subjects": n_neg,
        "fixed_intercept": float(params.get("Intercept", np.nan)),
        "baseline_genotype_effect_pos_minus_neg": b_geno,
        "baseline_genotype_effect_se": se_geno,
        "baseline_genotype_effect_ci95_low": b_geno - z * se_geno,
        "baseline_genotype_effect_ci95_high": b_geno + z * se_geno,
        "baseline_genotype_effect_p": p_geno,
        "slope_negative_per_year": slope_neg,
        "slope_negative_se": se_slope_neg,
        "slope_negative_ci95_low": slope_neg - z * se_slope_neg,
        "slope_negative_ci95_high": slope_neg + z * se_slope_neg,
        "slope_positive_per_year": slope_pos,
        "slope_positive_se": se_slope_pos,
        "slope_positive_ci95_low": slope_pos - z * se_slope_pos,
        "slope_positive_ci95_high": slope_pos + z * se_slope_pos,
        "genotype_time_interaction_pos_minus_neg_per_year": b_int,
        "genotype_time_interaction_se": se_int,
        "genotype_time_interaction_ci95_low": b_int - z * se_int,
        "genotype_time_interaction_ci95_high": b_int + z * se_int,
        "genotype_time_interaction_p": p_int,
        "baseline_between_subject_sd": baseline_sd,
        "interaction_baseline_sd_per_year": (
            b_int / baseline_sd if np.isfinite(baseline_sd) and baseline_sd > 0 else np.nan
        ),
        "random_slope_variance": random_slope_var,
        "random_slope_sd": (
            math.sqrt(max(random_slope_var, 0.0)) if np.isfinite(random_slope_var) else np.nan
        ),
        "residual_variance": float(result.scale),
        "log_likelihood": float(result.llf),
        "fit_attempts": json.dumps(attempts),
    }

    if adjusted:
        age_name = "baseline_age_c"
        age_time_name = f"{TIME_MODEL}:baseline_age_c"
        row.update(
            {
                "baseline_age_effect": float(params.get(age_name, np.nan)),
                "baseline_age_effect_p": float(result.pvalues.get(age_name, np.nan)),
                "time_by_baseline_age_effect": float(params.get(age_time_name, np.nan)),
                "time_by_baseline_age_effect_p": float(result.pvalues.get(age_time_name, np.nan)),
            }
        )

    return row


def fit_family(
    d: pd.DataFrame,
    outcomes: Sequence[str],
    adjusted: bool,
    reml: bool,
    maxiter: int,
    label: str,
) -> pd.DataFrame:
    rows = []
    for i, outcome in enumerate(outcomes, 1):
        print(f"[{label}] {i:02d}/{len(outcomes):02d} {outcome}")
        rows.append(
            fit_interaction_model(
                d=d,
                outcome=outcome,
                adjusted=adjusted,
                reml=reml,
                maxiter=maxiter,
            )
        )
    return pd.DataFrame(rows)


# ============================================================================
# Empirical subject-wise slopes
# ============================================================================


def empirical_slopes(d: pd.DataFrame, outcomes: Sequence[str], group_col: str) -> pd.DataFrame:
    rows = []
    for subject, g in d.groupby(SUBJECT_LONG, sort=False):
        g = g.sort_values(TIME_MODEL)
        group_values = g[group_col].dropna().astype(str).unique()
        if len(group_values) != 1:
            continue
        group = group_values[0]
        age = numeric(g["baseline_age_years"]).dropna()
        age0 = float(age.iloc[0]) if len(age) else np.nan
        genes = clean_string(g["sarcomere_PLP_genes"].iloc[0]) if "sarcomere_PLP_genes" in g.columns else ""

        for outcome in outcomes:
            if outcome not in g.columns:
                continue
            t = numeric(g[TIME_MODEL]).to_numpy(float)
            y = numeric(g[outcome]).to_numpy(float)
            m = np.isfinite(t) & np.isfinite(y)
            t = t[m]
            y = y[m]
            if len(y) < 2 or len(np.unique(t)) < 2:
                continue
            lr = stats.linregress(t, y)
            rows.append(
                {
                    SUBJECT_LONG: subject,
                    "genotype_group": group,
                    "sarcomere_PLP_genes": genes,
                    "baseline_age_years": age0,
                    "outcome": outcome,
                    "n_visits": int(len(y)),
                    "followup_years": float(np.max(t) - np.min(t)),
                    "ols_slope_per_year": float(lr.slope),
                    "ols_intercept": float(lr.intercept),
                    "ols_r_squared": float(lr.rvalue ** 2),
                }
            )
    return pd.DataFrame(rows)


def rank_biserial_from_u(u: float, n1: int, n0: int) -> float:
    # Positive means group 1 tends to have larger values than group 0.
    if n1 <= 0 or n0 <= 0:
        return np.nan
    return 2.0 * u / (n1 * n0) - 1.0


def summarize_empirical_slopes(emp: pd.DataFrame, positive_label: str, negative_label: str) -> pd.DataFrame:
    rows = []
    if emp.empty:
        return pd.DataFrame()

    for outcome, g in emp.groupby("outcome"):
        a = numeric(g.loc[g["genotype_group"] == positive_label, "ols_slope_per_year"]).dropna().to_numpy()
        b = numeric(g.loc[g["genotype_group"] == negative_label, "ols_slope_per_year"]).dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "outcome": outcome,
                "positive_label": positive_label,
                "negative_label": negative_label,
                "n_positive": int(len(a)),
                "n_negative": int(len(b)),
                "positive_median_slope": float(np.median(a)),
                "positive_iqr_low": float(np.percentile(a, 25)),
                "positive_iqr_high": float(np.percentile(a, 75)),
                "negative_median_slope": float(np.median(b)),
                "negative_iqr_low": float(np.percentile(b, 25)),
                "negative_iqr_high": float(np.percentile(b, 75)),
                "median_difference_pos_minus_neg": float(np.median(a) - np.median(b)),
                "mannwhitney_u": float(mw.statistic),
                "mannwhitney_p": float(mw.pvalue),
                "rank_biserial_pos_vs_neg": float(rank_biserial_from_u(mw.statistic, len(a), len(b))),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["mannwhitney_q_fdr"] = bh_fdr(out["mannwhitney_p"].to_numpy(float))
    return out


# ============================================================================
# Gene-specific secondary analyses
# ============================================================================


def gene_carrier_flag(gene_string: str, gene: str) -> bool:
    return gene.upper() in parse_gene_set(gene_string)


def run_gene_specific(
    strict_merged: pd.DataFrame,
    outcomes: Sequence[str],
    min_carriers: int,
    min_negatives: int,
    reml: bool,
    maxiter: int,
) -> pd.DataFrame:
    results = []
    negatives = strict_merged[strict_merged["primary_genotype_group"] == GENO_NEG].copy()
    n_neg = negatives[SUBJECT_LONG].nunique()
    if n_neg < min_negatives:
        return pd.DataFrame()

    positives = strict_merged[strict_merged["primary_genotype_group"] == GENO_POS].copy()

    for gene in ["MYBPC3", "MYH7"]:
        carrier_subjects = set(
            positives.loc[
                positives["sarcomere_PLP_genes"].fillna("").map(lambda x: gene_carrier_flag(x, gene)),
                SUBJECT_LONG,
            ].astype(str)
        )
        if len(carrier_subjects) < min_carriers:
            continue

        carrier = positives[positives[SUBJECT_LONG].astype(str).isin(carrier_subjects)].copy()
        d = pd.concat([negatives, carrier], ignore_index=True)
        d["gene_group"] = np.where(d[SUBJECT_LONG].astype(str).isin(carrier_subjects), f"{gene}+", GENO_NEG)
        d = add_binary_and_age_center(d, "gene_group", f"{gene}+")

        tab = fit_family(
            d,
            outcomes,
            adjusted=True,
            reml=reml,
            maxiter=maxiter,
            label=f"gene_{gene}",
        )
        tab["gene"] = gene
        tab["positive_definition"] = f"P/LP carrier in {gene}"
        tab["negative_definition"] = GENO_NEG
        tab["n_gene_carrier_subjects_source"] = len(carrier_subjects)
        results.append(tab)

    if not results:
        return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    out["interaction_q_fdr_within_gene_specific_family"] = bh_fdr(
        numeric(out["genotype_time_interaction_p"]).to_numpy(float)
    )
    return out


# ============================================================================
# Figures
# ============================================================================


def plot_primary_interactions(tab: pd.DataFrame, path: Path):
    d = tab[tab["success"] == True].copy()
    if d.empty:
        return
    d = d.set_index("outcome").reindex(PRIMARY_OUTCOMES).dropna(subset=["genotype_time_interaction_pos_minus_neg_per_year"]).reset_index()
    if d.empty:
        return

    y = numeric(d["interaction_baseline_sd_per_year"]).to_numpy(float)
    sd = numeric(d["baseline_between_subject_sd"]).to_numpy(float)
    lo = numeric(d["genotype_time_interaction_ci95_low"]).to_numpy(float) / sd
    hi = numeric(d["genotype_time_interaction_ci95_high"]).to_numpy(float) / sd
    err = np.vstack([y - lo, hi - y])

    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y, yerr=err, fmt="o", capsize=4)
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(d["outcome"], rotation=30, ha="right")
    ax.set_ylabel("Sarc+ minus Sarc- slope difference (baseline SD/year)")
    ax.set_title("Genotype-by-time interaction: primary ECG phenotypes")
    for i, q in enumerate(numeric(d["interaction_q_fdr_primary"]).to_numpy(float)):
        if np.isfinite(q) and q < 0.05:
            ax.text(i, y[i], "*", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_group_slopes(tab: pd.DataFrame, path: Path):
    d = tab[tab["success"] == True].copy()
    d = d.set_index("outcome").reindex(PRIMARY_OUTCOMES).reset_index()
    if d.empty:
        return

    x = np.arange(len(d), dtype=float)
    off = 0.16
    fig, ax = plt.subplots(figsize=(9, 5))

    for shift, label, slope_col, lo_col, hi_col in [
        (-off, GENO_NEG, "slope_negative_per_year", "slope_negative_ci95_low", "slope_negative_ci95_high"),
        (+off, GENO_POS, "slope_positive_per_year", "slope_positive_ci95_low", "slope_positive_ci95_high"),
    ]:
        y = numeric(d[slope_col]).to_numpy(float)
        lo = numeric(d[lo_col]).to_numpy(float)
        hi = numeric(d[hi_col]).to_numpy(float)
        err = np.vstack([y - lo, hi - y])
        ax.errorbar(x + shift, y, yerr=err, fmt="o", capsize=3, label=label)

    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(d["outcome"], rotation=30, ha="right")
    ax.set_ylabel("Estimated annual latent/metric change")
    ax.set_title("Genotype-stratified longitudinal slopes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_empirical_slope_distributions(emp: pd.DataFrame, path: Path):
    if emp.empty:
        return
    outcomes = [o for o in PRIMARY_OUTCOMES if o in set(emp["outcome"])]
    if not outcomes:
        return

    fig, axes = plt.subplots(len(outcomes), 1, figsize=(8, 3.0 * len(outcomes)), sharex=False)
    if len(outcomes) == 1:
        axes = [axes]

    for ax, outcome in zip(axes, outcomes):
        g = emp[emp["outcome"] == outcome]
        neg = numeric(g.loc[g["genotype_group"] == GENO_NEG, "ols_slope_per_year"]).dropna().to_numpy()
        pos = numeric(g.loc[g["genotype_group"] == GENO_POS, "ols_slope_per_year"]).dropna().to_numpy()
        if len(neg) and len(pos):
            ax.boxplot([neg, pos], tick_labels=[GENO_NEG, GENO_POS], showfliers=False)
        ax.axhline(0, linewidth=0.8)
        ax.set_ylabel(f"{outcome} slope/year")
    axes[-1].set_xlabel("Genotype group")
    fig.suptitle("Empirical patient-specific slopes by genotype")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Cohort flow and subject export
# ============================================================================


def build_cohort_flow(genetics: pd.DataFrame, merged: pd.DataFrame, strict: pd.DataFrame, min_visits: int) -> pd.DataFrame:
    subject_gen = genetics.drop_duplicates(SUBJECT_GEN)
    rows = []

    def add(stage, n, note=""):
        rows.append({"stage": stage, "n_subjects": int(n), "note": note})

    add("genetics_crosswalk_total", subject_gen[SUBJECT_GEN].nunique())
    add("matched_exact", subject_gen.loc[subject_gen["mapping_status"] == "matched", SUBJECT_GEN].nunique())
    add("matched_ambiguous_excluded", subject_gen.loc[subject_gen["mapping_status"] == "matched_ambiguous_signal_id", SUBJECT_GEN].nunique())
    add("unmatched", subject_gen.loc[subject_gen["mapping_status"] == "unmatched", SUBJECT_GEN].nunique())
    add("primary_Sarc_plus_before_longitudinal", subject_gen.loc[subject_gen["primary_genotype_group"] == GENO_POS, SUBJECT_GEN].nunique())
    add("primary_Sarc_minus_before_longitudinal", subject_gen.loc[subject_gen["primary_genotype_group"] == GENO_NEG, SUBJECT_GEN].nunique())
    add("primary_excluded_genetic_definition", subject_gen.loc[subject_gen["primary_genotype_group"] == "Excluded", SUBJECT_GEN].nunique())
    add(
        f"primary_longitudinal_eligible_min_{min_visits}_visits",
        strict[SUBJECT_LONG].nunique(),
        "Sarc+ and Sarc- combined",
    )
    add("primary_longitudinal_Sarc_plus", strict.loc[strict["primary_genotype_group"] == GENO_POS, SUBJECT_LONG].nunique())
    add("primary_longitudinal_Sarc_minus", strict.loc[strict["primary_genotype_group"] == GENO_NEG, SUBJECT_LONG].nunique())
    return pd.DataFrame(rows)


def make_subject_cohort_export(merged: pd.DataFrame) -> pd.DataFrame:
    cols = [
        SUBJECT_LONG,
        "subject_index",
        "patient_id_signal",
        "SHaRe_ID",
        "SHaRe_PID",
        "mapping_status",
        "Primary_Diagnosis",
        "genetic_status_raw",
        "has_P_or_LP_variant_bool",
        "P_LP_genes",
        "sarcomere_PLP_genes",
        "primary_genotype_group",
        "any_plp_group",
        "primary_exclusion_reason",
        "negative_test_record_present",
        "negative_test_age",
        "negative_test_panel",
        "baseline_age_years",
    ]
    cols = [c for c in cols if c in merged.columns]
    s = merged.sort_values([SUBJECT_LONG, TIME_MODEL]).groupby(SUBJECT_LONG, as_index=False).first()
    counts = subject_visit_counts(merged)
    s = s[cols].merge(counts, on=SUBJECT_LONG, how="left")
    return s


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()
    np.random.seed(args.seed)
    reml = not args.ml

    args.results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("18 - Genotype x longitudinal ECG phenotype interactions")
    print("=" * 82)
    print("Genetics     :", args.genetics)
    print("Longitudinal :", args.longitudinal)
    print("Results      :", args.results_dir)
    print("Estimator    :", "REML" if reml else "ML")
    print()

    genetics = load_genetics(args.genetics)
    longitudinal = load_longitudinal(args.longitudinal)
    merged = merge_data(longitudinal, genetics)

    # Export complete subject-level merged genetics/QC table first.
    subject_export = make_subject_cohort_export(merged)
    subject_export.to_csv(args.results_dir / "genotype_subject_cohort.csv", index=False)

    # ------------------------------------------------------------------
    # Primary strict sarcomeric cohort
    # ------------------------------------------------------------------
    strict, strict_counts = select_group_cohort(
        merged,
        group_col="primary_genotype_group",
        groups=[GENO_NEG, GENO_POS],
        min_visits=args.min_visits,
    )
    strict = add_binary_and_age_center(strict, "primary_genotype_group", GENO_POS)

    if strict.empty:
        raise RuntimeError("No subjects satisfy the strict primary genetic/longitudinal cohort definition.")

    n_pos = strict.loc[strict["primary_genotype_group"] == GENO_POS, SUBJECT_LONG].nunique()
    n_neg = strict.loc[strict["primary_genotype_group"] == GENO_NEG, SUBJECT_LONG].nunique()
    print(f"Primary strict cohort: {strict[SUBJECT_LONG].nunique()} subjects / {len(strict)} visits")
    print(f"  {GENO_POS}: {n_pos} subjects")
    print(f"  {GENO_NEG}: {n_neg} subjects")
    print()

    missing_primary = [o for o in PRIMARY_OUTCOMES if o not in strict.columns]
    if missing_primary:
        raise KeyError("Missing primary outcomes: " + ", ".join(missing_primary))

    # Age-adjusted primary models.
    primary = fit_family(
        strict,
        PRIMARY_OUTCOMES,
        adjusted=True,
        reml=reml,
        maxiter=args.maxiter,
        label="primary_strict_age_adjusted",
    )
    primary["interaction_q_fdr_primary"] = bh_fdr(
        numeric(primary["genotype_time_interaction_p"]).to_numpy(float)
    )
    primary["interaction_significant_fdr05"] = primary["interaction_q_fdr_primary"] < 0.05
    primary.to_csv(args.results_dir / "primary_genotype_time_interactions.csv", index=False)

    # Unadjusted sensitivity on exactly the same cohort/endpoints.
    primary_unadj = fit_family(
        strict,
        PRIMARY_OUTCOMES,
        adjusted=False,
        reml=reml,
        maxiter=args.maxiter,
        label="primary_strict_unadjusted",
    )
    primary_unadj["interaction_q_fdr_primary_unadjusted"] = bh_fdr(
        numeric(primary_unadj["genotype_time_interaction_p"]).to_numpy(float)
    )
    primary_unadj.to_csv(args.results_dir / "primary_unadjusted_sensitivity.csv", index=False)

    # Prespecified H_z02 comparator.
    comp_available = [o for o in COMPARATOR_OUTCOMES if o in strict.columns]
    if comp_available:
        comparator = fit_family(
            strict,
            comp_available,
            adjusted=True,
            reml=reml,
            maxiter=args.maxiter,
            label="comparator_Hz02",
        )
        comparator.to_csv(args.results_dir / "comparator_hz02_interaction.csv", index=False)
    else:
        comparator = pd.DataFrame()

    # ------------------------------------------------------------------
    # Broad secondary: any P/LP versus G(-)
    # ------------------------------------------------------------------
    broad, broad_counts = select_group_cohort(
        merged,
        group_col="any_plp_group",
        groups=[ANY_NEG, ANY_POS],
        min_visits=args.min_visits,
    )
    broad = add_binary_and_age_center(broad, "any_plp_group", ANY_POS)

    secondary_any = fit_family(
        broad,
        PRIMARY_OUTCOMES,
        adjusted=True,
        reml=reml,
        maxiter=args.maxiter,
        label="secondary_any_PLP",
    )
    secondary_any["interaction_q_fdr_secondary_any_plp"] = bh_fdr(
        numeric(secondary_any["genotype_time_interaction_p"]).to_numpy(float)
    )
    secondary_any.to_csv(args.results_dir / "secondary_any_plp_interactions.csv", index=False)

    # ------------------------------------------------------------------
    # Exploratory all Step-17 primary outcomes, strict genetic cohort
    # ------------------------------------------------------------------
    exploratory_outcomes = [o for o in EXPLORATORY_CANDIDATES if o in strict.columns]
    exploratory = fit_family(
        strict,
        exploratory_outcomes,
        adjusted=True,
        reml=reml,
        maxiter=args.maxiter,
        label="exploratory_all_outcomes",
    )
    exploratory["interaction_q_fdr_exploratory_family"] = bh_fdr(
        numeric(exploratory["genotype_time_interaction_p"]).to_numpy(float)
    )
    exploratory.to_csv(args.results_dir / "exploratory_all_outcomes_interactions.csv", index=False)

    # ------------------------------------------------------------------
    # Gene-specific secondary models
    # ------------------------------------------------------------------
    gene_specific = run_gene_specific(
        strict_merged=strict,
        outcomes=PRIMARY_OUTCOMES,
        min_carriers=args.min_gene_carriers,
        min_negatives=args.min_negative_controls,
        reml=reml,
        maxiter=args.maxiter,
    )
    gene_specific.to_csv(args.results_dir / "gene_specific_interactions.csv", index=False)

    # ------------------------------------------------------------------
    # Model-independent empirical slopes
    # ------------------------------------------------------------------
    empirical = empirical_slopes(
        strict,
        PRIMARY_OUTCOMES + COMPARATOR_OUTCOMES,
        group_col="primary_genotype_group",
    )
    empirical.to_csv(args.results_dir / "empirical_subject_slopes_by_genotype.csv", index=False)

    empirical_summary = summarize_empirical_slopes(empirical, GENO_POS, GENO_NEG)
    empirical_summary.to_csv(args.results_dir / "empirical_slope_group_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Cohort accounting
    # ------------------------------------------------------------------
    flow = build_cohort_flow(genetics, merged, strict, args.min_visits)
    flow.to_csv(args.results_dir / "cohort_flow.csv", index=False)

    exclusion = (
        genetics.drop_duplicates(SUBJECT_GEN)
        .groupby(["primary_genotype_group", "primary_exclusion_reason"], dropna=False)
        .size()
        .reset_index(name="n_subjects")
        .sort_values("n_subjects", ascending=False)
    )
    exclusion.to_csv(args.results_dir / "genetic_classification_counts.csv", index=False)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_primary_interactions(primary, fig_dir / "primary_interaction_effects.png")
    plot_group_slopes(primary, fig_dir / "primary_group_slopes.png")
    plot_empirical_slope_distributions(empirical, fig_dir / "primary_empirical_slope_distributions.png")

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    sig = primary[primary["interaction_q_fdr_primary"] < 0.05].copy()
    significant = []
    for _, r in sig.iterrows():
        significant.append(
            {
                "outcome": r["outcome"],
                "interaction_pos_minus_neg_per_year": finite_or_none(
                    r["genotype_time_interaction_pos_minus_neg_per_year"]
                ),
                "interaction_baseline_sd_per_year": finite_or_none(
                    r["interaction_baseline_sd_per_year"]
                ),
                "p": finite_or_none(r["genotype_time_interaction_p"]),
                "q_fdr": finite_or_none(r["interaction_q_fdr_primary"]),
                "slope_Sarc_minus_per_year": finite_or_none(r["slope_negative_per_year"]),
                "slope_Sarc_plus_per_year": finite_or_none(r["slope_positive_per_year"]),
            }
        )

    unadj_map = primary_unadj.set_index("outcome") if len(primary_unadj) else pd.DataFrame()
    adjusted_unadjusted_sign = {}
    for _, r in primary.iterrows():
        o = r["outcome"]
        if len(primary_unadj) and o in unadj_map.index:
            a = r["genotype_time_interaction_pos_minus_neg_per_year"]
            b = unadj_map.loc[o, "genotype_time_interaction_pos_minus_neg_per_year"]
            adjusted_unadjusted_sign[o] = bool(np.sign(a) == np.sign(b)) if np.isfinite(a) and np.isfinite(b) else None

    summary = {
        "script": "18_analyze_genotype_longitudinal_interactions.py",
        "version": "1.0",
        "scientific_role": (
            "Test whether genotype modifies longitudinal electrical phenotype trajectories "
            "in frozen Normal-anchored and HCM-oriented ECG representations"
        ),
        "inputs": {
            "genetics": str(args.genetics),
            "longitudinal": str(args.longitudinal),
            "genetics_sheet": "Subject_Genetics",
        },
        "primary_genetic_definition": {
            "positive_label": GENO_POS,
            "positive_definition": (
                "Exact non-ambiguous HCM mapping with >=1 P/LP variant in core sarcomeric genes"
            ),
            "negative_label": GENO_NEG,
            "negative_definition": (
                "Exact non-ambiguous HCM mapping with raw G(-) and no P/LP variant"
            ),
            "core_sarcomere_genes": sorted(CORE_SARCOMERE_GENES),
            "excluded_from_primary": [
                "G(0)",
                "G(U)",
                "VUS-only",
                "raw G(+) without core sarcomeric P/LP",
                "P/LP only in non-core genes",
                "unmatched mappings",
                "ambiguous signal-ID mappings",
                "Primary_Diagnosis other than HCM",
            ],
        },
        "primary_model": (
            "Linear mixed-effects model with random intercept and random time slope: "
            "outcome ~ time + genotype + time:genotype + baseline_age + time:baseline_age"
        ),
        "primary_estimand": (
            "genotype-by-time interaction: annual slope difference Sarc+ minus Sarc-"
        ),
        "estimation": "REML" if reml else "ML",
        "minimum_visits": int(args.min_visits),
        "primary_outcomes": PRIMARY_OUTCOMES,
        "multiplicity": "Benjamini-Hochberg FDR across four primary genotype-by-time interactions",
        "primary_cohort": {
            "n_subjects": int(strict[SUBJECT_LONG].nunique()),
            "n_visits": int(len(strict)),
            "n_Sarc_plus_subjects": int(n_pos),
            "n_Sarc_minus_subjects": int(n_neg),
            "baseline_age_mean_years": finite_or_none(
                strict[[SUBJECT_LONG, "baseline_age_years"]].drop_duplicates(SUBJECT_LONG)["baseline_age_years"].mean()
            ),
        },
        "broad_any_plp_secondary_cohort": {
            "n_subjects": int(broad[SUBJECT_LONG].nunique()),
            "n_visits": int(len(broad)),
            "n_any_PLP_positive_subjects": int(
                broad.loc[broad["any_plp_group"] == ANY_POS, SUBJECT_LONG].nunique()
            ),
            "n_G_negative_subjects": int(
                broad.loc[broad["any_plp_group"] == ANY_NEG, SUBJECT_LONG].nunique()
            ),
        },
        "n_primary_models_successful": int(primary["success"].fillna(False).sum()),
        "n_primary_interactions_fdr_significant": int(len(sig)),
        "primary_significant_interactions": significant,
        "adjusted_vs_unadjusted_interaction_sign_concordance": adjusted_unadjusted_sign,
        "gene_specific_models_run_for": sorted(gene_specific["gene"].unique().tolist()) if len(gene_specific) else [],
        "interpretive_notes": [
            "The genotype-by-time interaction is the primary longitudinal genetic effect; baseline genotype effects are secondary.",
            "Sarc+ is defined from P/LP variants in core sarcomeric genes rather than raw G(+) status.",
            "Ambiguous patient-ID mappings are excluded from the primary genetic analysis.",
            "G(0), G(U), and VUS-only subjects are not treated as genotype-negative.",
            "Age-adjusted primary models include baseline age and a time-by-baseline-age term to reduce confounding by age-dependent trajectory differences.",
            "Empirical OLS slopes are provided as a model-independent cross-check and are not substitutes for the mixed-effects interaction.",
            "Gene-specific MYBPC3/MYH7 analyses are secondary and run only when prespecified minimum carrier counts are met.",
            "A significant genotype-by-time interaction indicates different average electrical trajectory rates, not causal molecular progression.",
        ],
    }

    (args.results_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 82)
    print("STEP 18 COMPLETE")
    print("=" * 82)
    print(f"Primary cohort: {strict[SUBJECT_LONG].nunique()} subjects / {len(strict)} visits")
    print(f"  {GENO_POS}: {n_pos}")
    print(f"  {GENO_NEG}: {n_neg}")
    print(f"Primary interaction models successful: {int(primary['success'].fillna(False).sum())}/{len(PRIMARY_OUTCOMES)}")
    print(f"FDR-significant genotype x time interactions: {len(sig)}/{len(PRIMARY_OUTCOMES)}")

    if len(sig):
        cols = [
            "outcome",
            "slope_negative_per_year",
            "slope_positive_per_year",
            "genotype_time_interaction_pos_minus_neg_per_year",
            "interaction_baseline_sd_per_year",
            "genotype_time_interaction_p",
            "interaction_q_fdr_primary",
        ]
        print()
        print("[Primary FDR-significant genotype x time interactions]")
        print(sig[cols].to_string(index=False, float_format=lambda v: f"{v:+.5g}"))

    print()
    print("Results:", args.results_dir)


if __name__ == "__main__":
    main()
