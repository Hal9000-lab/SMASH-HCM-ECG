#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
38_z06_piecewise_trajectory.py

SMASH-HCM -- Step 38
Is the H_z_06 drift steeper early in follow-up, and what produces that pattern?

WHY THIS STEP EXISTS
--------------------
Step 37 follow-up checks suggested a steeper H_z_06 slope in the first two years.
That pattern was seen after looking at the data, so this analysis is EXPLORATORY.
MIMIC-IV is the independent check: the 2-year knot is applied unchanged.

MODELS
------
Linear    : y ~ t
Piecewise : y ~ t + t_k, with t_k = max(t - k, 0)
            early slope = b_t; late slope = b_t + b_tk; change = b_tk
Drop-first: first visit removed, time re-zeroed at the second visit, y ~ t
All models use a subject random intercept and slope. A random-slope fit with a
convergence warning is kept and flagged; random intercept is a last resort.

AGEING COMPARISON
-----------------
The Step-37 comparison is repeated inside every SHaRe cohort: time plus baseline
age, normal-age residual slope, and both age effects as a share of the slope.

COHORTS
-------
SHaRe: all subjects, exact mapped, pre-SRT, pre-AF, pre-pacing, and pre-core-clean
       (censored at the earliest of SRT, AF and pacing, with the Step-23 rules).
MIMIC: Step-35 bounded 30-day representative visits, subjects with >=2 time points.

OUTCOMES
--------
H_z_06 (primary; also knots at 1 and 3 years), H_z_01, H_z_05, d_N,
H_z_02 (negative control).

READING RULE (fixed before seeing these results)
------------------------------------------------
Non-linearity is supported if the change CI excludes zero, with a more negative
early slope, in both SHaRe pre_core_clean and MIMIC. Then:
- gone after censoring only  -> pattern linked to SRT, AF or pacing;
- gone after drop-first only -> pattern linked to the first ECG, for example
  selection of the first recording or regression to the mean;
- present in all analyses    -> early electrical phase, or a clinical-care effect
  that these analyses cannot separate.

INPUTS
------
Results/16_Longitudinal_DualManifold/longitudinal_dual_manifold.csv
Results/18_Genotype_Longitudinal/genotype_subject_cohort.csv
Results/35_MIMIC_Longitudinal_Hz06_Replication/mimic_visit_windows_30d.csv
Results/37_Z06_Age_Sensitivity/normal_reference_values.csv
Data/SHaReData-FLO.xlsx
Scripts/23_sensitivity_major_events_longitudinal.py
Scripts/37_z06_age_sensitivity.py

OUTPUTS
-------
Results/38_Z06_Piecewise_Trajectory/
    piecewise_models.csv
    cohort_counts.csv
    piecewise_report.txt
    figures/hz06_binned_trajectory.png

RUN
---
python Scripts/38_z06_piecewise_trajectory.py

DEPENDENCIES
------------
numpy pandas scipy statsmodels matplotlib
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DAYS_PER_YEAR = 365.2425
PRIMARY_KNOT = 2.0
SENSITIVITY_KNOTS = (1.0, 3.0)
PRIMARY_OUTCOME = "H_z_06"
OUTCOMES = ["H_z_06", "H_z_01", "H_z_05", "d_N", "H_z_02"]
CENSORING = {
    "pre_SRT": ["srt"],
    "pre_AF": ["af"],
    "pre_pacer": ["pacer"],
    "pre_core_clean": ["srt", "af", "pacer"],
}


def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [Path.cwd().resolve(), here.parent, *here.parents]:
        if (p / "Data").exists() and (p / "Results").exists():
            return p
    return here.parent.parent


def parse_args() -> argparse.Namespace:
    root = infer_project_root()
    r = root / "Results"
    p = argparse.ArgumentParser(description="Piecewise H_z_06 trajectory analysis.")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--longitudinal", type=Path, default=r / "16_Longitudinal_DualManifold" / "longitudinal_dual_manifold.csv")
    p.add_argument("--mimic-visits", type=Path, default=r / "35_MIMIC_Longitudinal_Hz06_Replication" / "mimic_visit_windows_30d.csv")
    p.add_argument("--step23", type=Path, default=root / "Scripts" / "23_sensitivity_major_events_longitudinal.py")
    p.add_argument("--step37", type=Path, default=root / "Scripts" / "37_z06_age_sensitivity.py")
    p.add_argument("--normal-reference", type=Path, default=r / "37_Z06_Age_Sensitivity" / "normal_reference_values.csv")
    p.add_argument("--results-dir", type=Path, default=r / "38_Z06_Piecewise_Trajectory")
    p.add_argument("--maxiter", type=int, default=2000)
    return p.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Keep subjects with at least two distinct time points."""
    n_times = df.groupby("subject")["t"].transform("nunique")
    return df[n_times >= 2].copy()


# =============================================================================
# Cohorts
# =============================================================================

def load_share_visits(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[as_bool(d["valid_for_latent_projection"])].copy()
    d["subject"] = d["analysis_subject_key"].astype(str).str.strip()
    d["t"] = pd.to_numeric(d["time_from_first_valid_ecg_years"], errors="coerce")
    seconds = (pd.to_datetime(d["ecg_time"], errors="coerce") - pd.to_datetime(d["birthdate"], errors="coerce")).dt.total_seconds()
    d["age"] = seconds / (86400.0 * DAYS_PER_YEAR)
    d = d[np.isfinite(d["t"])].copy()
    return eligible(d[["subject", "t", "age", *OUTCOMES]])


def load_mimic_visits(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["subject"] = d["subject_id"].astype(str)
    d["t"] = pd.to_numeric(d["time_years"], errors="coerce")
    d = d[np.isfinite(d["t"])].copy()
    return eligible(d[["subject", "t", *[c for c in OUTCOMES if c in d.columns]]])


def censor(df: pd.DataFrame, events: dict, keys: list[str]) -> tuple[pd.DataFrame, dict]:
    """Same rules as Step 23: drop visits from the event onwards, and drop subjects
    with a positive event of unknown age."""
    keep = []
    subjects_missing_age = set()
    visits_censored = 0
    for row in df.itertuples(index=False):
        event = events.get(row.subject, {})
        ages, missing = [], False
        for key in keys:
            if event.get(f"{key}_pos", False):
                age = event.get(f"{key}_age")
                if age is None:
                    missing = True
                else:
                    ages.append(age)
        if missing:
            subjects_missing_age.add(row.subject)
            keep.append(False)
        elif ages and row.age >= min(ages):
            visits_censored += 1
            keep.append(False)
        else:
            keep.append(True)
    out = eligible(df[np.asarray(keep, dtype=bool)])
    return out, {"visits_censored": visits_censored, "subjects_missing_event_age": len(subjects_missing_age)}


def build_share_cohorts(visits: pd.DataFrame, s2p: dict, events: dict) -> tuple[dict, list]:
    exact = eligible(visits[visits["subject"].isin(s2p)])
    cohorts = {"SHaRe_all": visits, "SHaRe_exact_mapped": exact}
    audit = []
    for name, keys in CENSORING.items():
        cohorts[f"SHaRe_{name}"], info = censor(exact, events, keys)
        audit.append({"cohort": f"SHaRe_{name}", **info})
    return cohorts, audit


# =============================================================================
# Models
# =============================================================================

def fit_formula(d: pd.DataFrame, rhs: str, maxiter: int) -> dict | None:
    """Keep the random slope whenever it can be fitted, as in Steps 17 and 37.

    A warning-free random-slope fit is preferred; a warned one is kept and flagged.
    The random-intercept model is a last resort: it ignores between-subject slope
    variation and gives standard errors that are too small for the time terms.
    """
    warned = None
    specs = [("random_intercept_slope", "~t", m) for m in ("lbfgs", "powell", "cg")]
    specs += [("random_intercept_only", "1", m) for m in ("lbfgs", "powell")]
    for structure, re_formula, method in specs:
        if structure == "random_intercept_only" and warned is not None:
            return warned
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = smf.mixedlm(f"_y ~ {rhs}", data=d, groups=d["subject"], re_formula=re_formula).fit(
                    reml=True, method=method, maxiter=maxiter, disp=False
                )
        except Exception:
            continue
        if not bool(getattr(result, "converged", False)):
            continue
        names = list(result.fe_params.index)
        fit = {
            "structure": structure,
            "optimizer": method,
            "params": result.fe_params,
            "cov": result.cov_params().loc[names, names],
            "convergence_warning": any("Convergence" in w.category.__name__ for w in caught),
        }
        if not fit["convergence_warning"]:
            return fit
        warned = warned or fit
    return warned


def linear_combination(fit: dict, weights: dict, scale: float) -> tuple[float, float]:
    names = list(weights)
    w = np.array([weights[n] for n in names], dtype=float)
    beta = fit["params"].loc[names].to_numpy(dtype=float)
    cov = fit["cov"].loc[names, names].to_numpy(dtype=float)
    return float(w @ beta) * scale, float(np.sqrt(max(w @ cov @ w, 0.0))) * scale


def effect_columns(prefix: str, estimate: float, se: float) -> dict:
    return {
        prefix: estimate,
        f"{prefix}_se": se,
        f"{prefix}_ci_low": estimate - 1.96 * se,
        f"{prefix}_ci_high": estimate + 1.96 * se,
        f"{prefix}_p": float(2 * stats.norm.sf(abs(estimate / se))) if se > 0 else np.nan,
    }


def prepare(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, float]:
    d = df[["subject", "t", outcome]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = eligible(d)
    first = d.sort_values(["subject", "t"]).groupby("subject")[outcome].first()
    scale = float(first.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(d[outcome].std(ddof=1))
    d["_y"] = d[outcome] / scale
    return d, scale


def base_row(cohort: str, outcome: str, model: str, knot: float, d: pd.DataFrame, fit: dict | None) -> dict:
    return {
        "cohort": cohort,
        "outcome": outcome,
        "model": model,
        "knot_years": knot,
        "n_subjects": int(d["subject"].nunique()),
        "n_obs": int(len(d)),
        "structure": "" if fit is None else fit["structure"],
        "optimizer": "" if fit is None else fit["optimizer"],
        "convergence_warning": None if fit is None else fit["convergence_warning"],
    }


def run_linear(cohort, outcome, df, maxiter, model="linear") -> dict:
    d, scale = prepare(df, outcome)
    fit = fit_formula(d, "t", maxiter) if d["subject"].nunique() >= 10 else None
    row = base_row(cohort, outcome, model, np.nan, d, fit)
    if fit is not None:
        row.update(effect_columns("slope", *linear_combination(fit, {"t": 1.0}, scale)))
    return row


def run_piecewise(cohort, outcome, df, knot, maxiter) -> dict:
    d, scale = prepare(df, outcome)
    d["t_k"] = np.clip(d["t"] - knot, 0.0, None)
    fit = None
    if d["subject"].nunique() >= 10 and (d["t"] > knot).any() and (d["t"] < knot).any():
        fit = fit_formula(d, "t + t_k", maxiter)
    row = base_row(cohort, outcome, "piecewise", knot, d, fit)
    has_early = d[d["t"] < knot].groupby("subject")["t"].nunique() >= 2
    has_late = d[d["t"] >= knot].groupby("subject")["t"].nunique() >= 2
    row["n_subjects_2plus_early_visits"] = int(has_early.sum())
    row["n_subjects_2plus_late_visits"] = int(has_late.sum())
    if fit is not None:
        row.update(effect_columns("early_slope", *linear_combination(fit, {"t": 1.0}, scale)))
        row.update(effect_columns("late_slope", *linear_combination(fit, {"t": 1.0, "t_k": 1.0}, scale)))
        row.update(effect_columns("slope_change", *linear_combination(fit, {"t_k": 1.0}, scale)))
    return row


def drop_first_visit(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["subject", "t"]).copy()
    d = d[d.groupby("subject").cumcount() > 0].copy()
    d["t"] = d["t"] - d.groupby("subject")["t"].transform("min")
    return eligible(d)


def run_age_decomposition(cohort, outcome, df, s37, reference, maxiter) -> dict:
    """Repeat the Step-37 ageing comparison inside one cohort."""
    d = df[["subject", "t", "age", outcome]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = eligible(d)
    ordered = d.sort_values(["subject", "t"]).groupby("subject")
    first_age, first_y = ordered["age"].first(), ordered[outcome].first()
    d["age0_c"] = d["subject"].map(first_age) - float(first_age.mean())
    scale = float(first_y.std(ddof=1))

    ref = s37.fit_age_reference(reference["age"], reference[outcome])
    gradient, gradient_se = s37.average_age_gradient(ref, d.loc[d["age"].between(18.0, 90.0), "age"])

    d["_y"] = d[outcome] / scale
    decomposition = fit_formula(d, "t + age0_c", maxiter)
    d["_y"] = (d[outcome] - s37.predict_reference(ref, d["age"])) / scale
    residual = fit_formula(d, "t", maxiter)

    row = base_row(cohort, outcome, "age_decomposition", np.nan, d, decomposition)
    row.update({"normal_gradient": gradient, "normal_gradient_se": gradient_se})
    if decomposition is not None:
        slope, slope_se = linear_combination(decomposition, {"t": 1.0}, scale)
        between, between_se = linear_combination(decomposition, {"age0_c": 1.0}, scale)
        row.update(effect_columns("slope", slope, slope_se))
        row.update(effect_columns("between_age", between, between_se))
        row.update(effect_columns("within_minus_between", *linear_combination(decomposition, {"t": 1.0, "age0_c": -1.0}, scale)))
        row["share_normal_gradient"] = gradient / slope
        row["share_between_age"] = between / slope
    if residual is not None:
        row.update(effect_columns("residual_slope", *linear_combination(residual, {"t": 1.0}, scale)))
        row["residual_structure"] = residual["structure"]
    return row


def analyse(cohorts: dict, maxiter: int, s37=None, reference=None) -> pd.DataFrame:
    rows = []
    for cohort, df in cohorts.items():
        for outcome in [o for o in OUTCOMES if o in df.columns]:
            print(f"  {cohort} | {outcome}")
            rows.append(run_linear(cohort, outcome, df, maxiter))
            rows.append(run_piecewise(cohort, outcome, df, PRIMARY_KNOT, maxiter))
            rows.append(run_linear(cohort, outcome, drop_first_visit(df), maxiter, model="drop_first_linear"))
            if outcome == PRIMARY_OUTCOME:
                for knot in SENSITIVITY_KNOTS:
                    rows.append(run_piecewise(cohort, outcome, df, knot, maxiter))
            if s37 is not None and "age" in df.columns and outcome in reference.columns:
                rows.append(run_age_decomposition(cohort, outcome, df, s37, reference, maxiter))
    return pd.DataFrame(rows)


# =============================================================================
# Outputs
# =============================================================================

def pick(models: pd.DataFrame, cohort: str, model: str, outcome: str = PRIMARY_OUTCOME, knot: float | None = None):
    d = models[models["cohort"].eq(cohort) & models["model"].eq(model) & models["outcome"].eq(outcome)]
    if knot is not None:
        d = d[np.isclose(d["knot_years"], knot)]
    return None if d.empty else d.iloc[0]


def supports_nonlinearity(row) -> bool:
    return row is not None and bool(row.get("slope_change_ci_low", np.nan) > 0)


def build_report(models: pd.DataFrame) -> str:
    lines = ["SMASH-HCM Step 38 -- Piecewise H_z_06 trajectory (exploratory)", ""]
    cohorts = list(dict.fromkeys(models["cohort"]))
    for cohort in cohorts:
        linear = pick(models, cohort, "linear")
        piece = pick(models, cohort, "piecewise", knot=PRIMARY_KNOT)
        first = pick(models, cohort, "drop_first_linear")
        if linear is None or piece is None or "early_slope" not in piece.index or pd.isna(piece["early_slope"]):
            lines += [cohort, "  piecewise model not estimable", ""]
            continue
        lines += [
            f"{cohort}: {int(linear['n_subjects'])} subjects, {int(linear['n_obs'])} visits",
            f"  Linear slope : {linear['slope']:+.4f} ({linear['slope_ci_low']:+.4f} to {linear['slope_ci_high']:+.4f})",
            f"  Early slope  : {piece['early_slope']:+.4f} ({piece['early_slope_ci_low']:+.4f} to {piece['early_slope_ci_high']:+.4f})",
            f"  Late slope   : {piece['late_slope']:+.4f} ({piece['late_slope_ci_low']:+.4f} to {piece['late_slope_ci_high']:+.4f})",
            f"  Slope change : {piece['slope_change']:+.4f} ({piece['slope_change_ci_low']:+.4f} to {piece['slope_change_ci_high']:+.4f}), p={piece['slope_change_p']:.3g}",
        ]
        if first is not None and pd.notna(first.get("slope", np.nan)):
            lines.append(f"  Drop first   : {first['slope']:+.4f} ({first['slope_ci_low']:+.4f} to {first['slope_ci_high']:+.4f}), {int(first['n_subjects'])} subjects")
        structures = models[models["cohort"].eq(cohort) & models["outcome"].eq(PRIMARY_OUTCOME)]
        lines.append(f"  Structures   : {', '.join(sorted(set(structures['structure'].dropna())))}; warned fits: {int((structures['convergence_warning'] == True).sum())}")
        for outcome in [PRIMARY_OUTCOME, "H_z_02"]:
            age = pick(models, cohort, "age_decomposition", outcome=outcome)
            if age is not None and pd.notna(age.get("slope", np.nan)):
                lines += [
                    f"  Age, {outcome}: slope {age['slope']:+.4f}, normal gradient {age['normal_gradient']:+.4f} "
                    f"({100 * age['share_normal_gradient']:.0f}% of slope), baseline-age term {age['between_age']:+.4f} "
                    f"({100 * age['share_between_age']:.0f}% of slope)",
                    f"      residual slope {age['residual_slope']:+.4f} ({age['residual_slope_ci_low']:+.4f} to {age['residual_slope_ci_high']:+.4f}), "
                    f"within-minus-between {age['within_minus_between']:+.4f} ({age['within_minus_between_ci_low']:+.4f} to {age['within_minus_between_ci_high']:+.4f})",
                ]
        lines.append("")

    clean = pick(models, "SHaRe_pre_core_clean", "piecewise", knot=PRIMARY_KNOT)
    mimic = pick(models, "MIMIC_30d", "piecewise", knot=PRIMARY_KNOT)
    lines += [
        "Prespecified reading rule",
        f"  Non-linearity in SHaRe pre_core_clean: {supports_nonlinearity(clean)}",
        f"  Non-linearity in MIMIC               : {supports_nonlinearity(mimic)}",
        "",
        "Guardrails:",
        "- The 2-year knot was chosen after seeing the data; only MIMIC is an independent check.",
        "- Censoring at an event is informative and removes sicker follow-up.",
        "- No multiplicity correction is applied in this exploratory step.",
    ]
    return "\n".join(lines) + "\n"


def plot_binned(cohorts: dict, path: Path) -> None:
    edges = np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 20], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["SHaRe_all", "SHaRe_pre_core_clean", "MIMIC_30d"]:
        if name not in cohorts:
            continue
        d = cohorts[name][["subject", "t", PRIMARY_OUTCOME]].dropna().copy()
        d["change"] = d[PRIMARY_OUTCOME] - d.groupby("subject")[PRIMARY_OUTCOME].transform("first")
        d = d[d["t"] > 0]
        groups = d.groupby(pd.cut(d["t"], edges), observed=True)
        ax.errorbar(groups["t"].mean(), groups["change"].mean(), yerr=1.96 * groups["change"].sem(), marker="o", capsize=3, label=name)
    ax.axhline(0, linewidth=0.8, color="0.5")
    ax.axvline(PRIMARY_KNOT, linewidth=0.8, linestyle="--", color="0.5")
    ax.set_xlabel("Years from first visit")
    ax.set_ylabel("Change in H_z_06 from first visit")
    ax.set_title("H_z_06 change over follow-up")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_step23_events(path: Path, root: Path) -> tuple[dict, dict]:
    """Step 23 uses paths relative to the project root."""
    if not path.exists():
        raise FileNotFoundError(f"Step-23 script not found: {path}")
    os.chdir(root)
    spec = importlib.util.spec_from_file_location("smash_step23", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    s2p, p2s = module.load_cross()
    return s2p, module.eventmap(s2p, module.load_events(p2s))


def main() -> None:
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    (args.results_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading cohorts")
    visits = load_share_visits(args.longitudinal.resolve())
    mimic_path = args.mimic_visits.resolve()
    step37_path, reference_path = args.step37.resolve(), args.normal_reference.resolve()
    s2p, events = load_step23_events(args.step23.resolve(), args.root.resolve())
    args.step37, args.normal_reference = step37_path, reference_path
    cohorts, audit = build_share_cohorts(visits, s2p, events)
    if mimic_path.exists():
        cohorts["MIMIC_30d"] = load_mimic_visits(mimic_path)
    else:
        warnings.warn(f"MIMIC visits not found: {mimic_path}")

    counts = pd.DataFrame([
        {"cohort": k, "n_subjects": d["subject"].nunique(), "n_visits": len(d)} for k, d in cohorts.items()
    ]).merge(pd.DataFrame(audit, columns=["cohort", "visits_censored", "subjects_missing_event_age"]), on="cohort", how="left")
    print(counts.to_string(index=False))

    print("[2/4] Fitting models")
    spec = importlib.util.spec_from_file_location("smash_step37", args.step37.resolve())
    s37 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s37)
    reference = pd.read_csv(args.normal_reference.resolve())
    models = analyse(cohorts, args.maxiter, s37, reference)

    print("[3/4] Writing outputs")
    models.to_csv(args.results_dir / "piecewise_models.csv", index=False)
    counts.to_csv(args.results_dir / "cohort_counts.csv", index=False)
    plot_binned(cohorts, args.results_dir / "figures" / "hz06_binned_trajectory.png")

    print("[4/4] Report")
    report = build_report(models)
    (args.results_dir / "piecewise_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print("Results:", args.results_dir)


if __name__ == "__main__":
    main()
