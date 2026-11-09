#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
40_z06_clinical_adjustment.py

SMASH-HCM -- Step 40
Is H_z_06 associated with heart failure independently of established clinical and
echocardiographic markers?

WHY THIS STEP EXISTS
--------------------
Step 39 showed that the association survives SRT, AF, pacing and stricter age
handling. Independence from imaging and clinical markers remains untested.

DATA
----
All clinical data come from the SHaRe workbook (Data/SHaReData-FLO.xlsx):
HCM_Echo        Ech_DIM_Max_LVT, Ech_DIM_LA, Ech_DIM_LVEF, Ech_LVOTO_Rest
HCM_ClinicVisit Cli_NYHA
HCM_Events      prior AF (through Step 23)
Risk sets, exposure and z06 scaling are reused from Step 26, as in Step 39.

MATCHING
--------
Each variable takes the value closest to the baseline ECG within +/-1 year, and
never after the endpoint or a post-baseline SRT, so that outcomes and procedures do
not leak into the covariates. Equal distance prefers the earlier measurement.
Sensitivity: prior_only uses measurements from -1 year to baseline + 1 day.
Only the main numeric fields are used. Hypovalue fields are audited, not parsed,
and values outside plausible ranges are set to missing and counted.

MODELS (time-updated and baseline, always on the same complete cases)
---------------------------------------------------------------------
age_sex          z06 + age + sex; reference for the reading rule
each marker      age_sex + one marker
clinical         age_sex + MaxLVT + LA + LVEF + log(1 + LVOT gradient)
clinical_nyha    clinical + NYHA II or higher
clinical_af      clinical + prior AF
clinical_censor_srt  clinical, follow-up censored at SRT
clinical_prior_only  clinical, prior_only values

Baseline discrimination compares the clinical model with and without z06: apparent
Harrell C with a subject-bootstrap interval for the difference. It is not corrected
for optimism.

READING RULE (fixed before seeing the results)
----------------------------------------------
As in Step 39, on the same subjects: preserved if at least 70% of the log HR is kept
and the CI excludes 1; preserved_low_precision if kept and the CI includes 1;
attenuated otherwise. LA and LVEF may lie on the causal pathway, so attenuation is
not by itself evidence of confounding. Complete-case analysis; no imputation.

OUTPUT: Results/40_Z06_Clinical_Adjustment/
Usage: python Scripts\\40_z06_clinical_adjustment.py
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ENDPOINT = "Composite_HF"
WINDOW = (-1.0, 1.0)
PRIOR_WINDOW = (-1.0, 1.0 / 365.2425)
ONE_DAY_YEARS = 1.0 / 365.2425
EVENTS_PER_PARAMETER = 10
ECHO = {  # variable: (source column, lowest, highest, divisor for the reported HR)
    "max_lvt": ("Ech_DIM_Max_LVT", 5.0, 60.0, 5.0),
    "la": ("Ech_DIM_LA", 15.0, 90.0, 5.0),
    "lvef": ("Ech_DIM_LVEF", 10.0, 90.0, 5.0),
    "lvot": ("Ech_LVOTO_Rest", 0.0, 200.0, None),
}
AUDIT_TEXT = ["ECH_DIM_Max_LVT__Hypovalue", "ECH_DIM_LA__Hypovalue", "Ech_DIM_LVEF__Hypovalue", "Ech_LVOTO_Rest__Hypovalue"]
TERMS = ["max_lvt_per5mm", "la_per5mm", "lvef_per5pct", "lvot_log1p"]
BASE = ["z06_per_sd", "baseline_age_10y", "male"]


def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / "Results").exists() and (p / "Scripts").exists():
            return p
    raise RuntimeError("Cannot locate the project root (expected Results/ and Scripts/).")


def parse_args() -> argparse.Namespace:
    root = infer_project_root()
    p = argparse.ArgumentParser(description="Clinical adjustment of the H_z_06 association.")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--step23", type=Path, default=root / "Scripts" / "23_sensitivity_major_events_longitudinal.py")
    p.add_argument("--step26", type=Path, default=root / "Scripts" / "26_z06_time_updated_cox.py")
    p.add_argument("--step39", type=Path, default=root / "Scripts" / "39_z06_risk_sensitivity.py")
    p.add_argument("--results-dir", type=Path, default=root / "Results" / "40_Z06_Clinical_Adjustment")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261103)
    return p.parse_args()


def parse_nyha(value) -> float:
    """Return 1-4 from entries such as 2, 2.0, II or Class II; otherwise NaN."""
    text = str(value).strip().upper()
    match = re.search(r"\b(IV|III|II|I|[1-4])(?:\.0+)?\b", text)
    if not match:
        return np.nan
    return float({"I": 1, "II": 2, "III": 3, "IV": 4}.get(match.group(1), match.group(1)))


def load_echo(rows: list[dict], norm_pid) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(rows)
    out = pd.DataFrame({"pid": raw["HCM_Echo_PID"].map(norm_pid),
                        "age": pd.to_numeric(raw["HCM_Echo_Patient_Age"], errors="coerce")})
    audit = []
    for name, (col, low, high, _) in ECHO.items():
        text = raw[col].astype(str).str.strip() if col in raw else pd.Series("", index=raw.index)
        values = pd.to_numeric(text, errors="coerce")
        bad = values.notna() & ~values.between(low, high)
        audit.append({"variable": name, "source": col, "n_filled": int(text.ne("").sum() - text.eq("nan").sum()),
                      "n_numeric": int(values.notna().sum()), "n_out_of_range": int(bad.sum()),
                      "p01": values.quantile(.01), "median": values.median(), "p99": values.quantile(.99),
                      "text_values": "; ".join(f"{k}={v}" for k, v in text[values.isna() & text.ne("") & text.ne("nan")].value_counts().head(8).items())})
        out[name] = values.mask(bad)
    for col in AUDIT_TEXT:
        text = raw[col].astype(str).str.strip() if col in raw else pd.Series(dtype=str)
        text = text[text.ne("") & text.ne("nan")]
        audit.append({"variable": "hypovalue", "source": col, "n_filled": len(text),
                      "text_values": "; ".join(f"{k}={v}" for k, v in text.value_counts().head(8).items())})
    return out[out["age"].notna()], pd.DataFrame(audit)


def load_nyha(rows: list[dict], norm_pid) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(rows)
    out = pd.DataFrame({"pid": raw["HCM_ClinicVisit_PID"].map(norm_pid),
                        "age": pd.to_numeric(raw["HCM_ClinicVisit_Patient_Age"], errors="coerce"),
                        "nyha": raw["Cli_NYHA"].map(parse_nyha)})
    text = raw["Cli_NYHA"].astype(str).str.strip()
    audit = pd.DataFrame([{"variable": "nyha", "source": "Cli_NYHA", "n_filled": int((text.ne("") & text.ne("nan")).sum()),
                           "n_numeric": int(out["nyha"].notna().sum()),
                           "text_values": "; ".join(f"{k}={v}" for k, v in text.value_counts().head(8).items())}])
    return out[out["age"].notna()], audit


def nearest_value(measure: pd.DataFrame, column: str, keys: pd.DataFrame, window: tuple[float, float]) -> pd.DataFrame:
    """Closest value per subject inside the window and not after limit_t."""
    m = measure[["pid", "age", column]].dropna().merge(keys, on="pid")
    m["delta"] = m["age"] - m["baseline_age"]
    m = m[m["delta"].between(*window) & (m["delta"] <= m["limit_t"])].copy()
    m["distance"] = m["delta"].abs()
    m = m.sort_values(["pid", "distance", "delta"]).drop_duplicates("pid")
    return m.set_index("pid")[[column, "delta"]]


def add_clinical(risk: pd.DataFrame, echo: pd.DataFrame, nyha: pd.DataFrame, table: pd.DataFrame,
                 window: tuple[float, float], norm_pid) -> tuple[pd.DataFrame, list[dict]]:
    out = risk.copy()
    out["pid"] = out["SHaRe_PID"].map(norm_pid)
    srt_t = table.loc[out["analysis_subject_key"].astype(str), "srt_t"].to_numpy(float)
    limit = out["endpoint_time"].to_numpy(float).copy()
    post = np.isfinite(srt_t) & (srt_t > ONE_DAY_YEARS)
    limit[post] = np.minimum(limit[post], srt_t[post])
    keys = pd.DataFrame({"pid": out["pid"].to_numpy(), "baseline_age": pd.to_numeric(out["baseline_age"]).to_numpy(float), "limit_t": limit})
    audit = []
    for name, source in [*((k, echo) for k in ECHO), ("nyha", nyha)]:
        found = nearest_value(source, name, keys, window)
        out[name] = out["pid"].map(found[name])
        delta = found["delta"]
        audit.append({"variable": name, "window": str(window), "n_matched": int(out[name].notna().sum()), "n_risk_set": len(out),
                      "events_matched": int(out.loc[out[name].notna(), "endpoint_event"].sum()),
                      "median_delta_years": delta.median(), "p05_delta": delta.quantile(.05), "p95_delta": delta.quantile(.95)})
    out["max_lvt_per5mm"] = out["max_lvt"] / ECHO["max_lvt"][3]
    out["la_per5mm"] = out["la"] / ECHO["la"][3]
    out["lvef_per5pct"] = out["lvef"] / ECHO["lvef"][3]
    out["lvot_log1p"] = np.log1p(out["lvot"])
    out["nyha_2plus"] = np.where(out["nyha"].isna(), np.nan, (out["nyha"] >= 2).astype(float))
    af_t = table.loc[out["analysis_subject_key"].astype(str), "af_t"].to_numpy(float)
    unknown = table.loc[out["analysis_subject_key"].astype(str), "af_missing_age"].to_numpy(bool)
    out["prior_af"] = np.where(unknown, np.nan, (af_t <= ONE_DAY_YEARS).astype(float))
    return out, audit


def fit_cox(phreg, cp: pd.DataFrame, covariates: list[str]) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = phreg(endog=cp["stop"].astype(float), exog=cp[covariates].astype(float), status=cp["event"].astype(int),
                    entry=cp["start"].astype(float), ties="breslow").fit()
    ci = np.asarray(res.conf_int())
    terms = [{"term": name, "hr": math.exp(float(res.params[i])), "ci95_low": math.exp(float(ci[i, 0])),
              "ci95_high": math.exp(float(ci[i, 1])), "p": float(res.pvalues[i]), "log_hr": float(res.params[i])}
             for i, name in enumerate(covariates)]
    return {"terms": terms, "beta": np.asarray(res.params, float),
            "warning": any("Convergence" in w.category.__name__ for w in caught)}


def harrell_c(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> float:
    """Comparable pairs: the shorter time must be an event."""
    idx = np.flatnonzero(event == 1)
    concordant = 0.0
    total = 0
    for i in idx:
        later = time > time[i]
        total += int(later.sum())
        concordant += float((score[i] > score[later]).sum()) + 0.5 * float((score[i] == score[later]).sum())
    return concordant / total if total else np.nan


def delta_c(phreg, d: pd.DataFrame, clinical: list[str]) -> tuple[float, float]:
    time, event = d["stop"].to_numpy(float), d["event"].to_numpy(int)
    without = fit_cox(phreg, d, clinical)
    with_z = fit_cox(phreg, d, ["z06_per_sd", *clinical])
    c0 = harrell_c(time, event, d[clinical].to_numpy(float) @ without["beta"])
    c1 = harrell_c(time, event, d[["z06_per_sd", *clinical]].to_numpy(float) @ with_z["beta"])
    return c0, c1


def discrimination(phreg, d: pd.DataFrame, clinical: list[str], n_boot: int, seed: int) -> dict:
    c0, c1 = delta_c(phreg, d, clinical)
    rng = np.random.default_rng(seed)
    diffs, failed = [], 0
    for _ in range(n_boot):
        sample = d.iloc[rng.integers(0, len(d), len(d))]
        try:
            a, b = delta_c(phreg, sample, clinical)
            diffs.append(b - a)
        except Exception:
            failed += 1
    low, high = np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)
    return {"c_clinical": c0, "c_clinical_z06": c1, "delta_c": c1 - c0, "delta_c_ci95_low": low, "delta_c_ci95_high": high,
            "bootstrap_failed": failed, "n_subjects": len(d), "events": int(d["event"].sum())}


def read_row(log_hr: float, ref: float, ci_low: float, ci_high: float) -> tuple[float, str]:
    retained = log_hr / ref if ref else np.nan
    if not np.isfinite(retained):
        return np.nan, "not_estimable"
    if retained < 0.70:
        return retained, "attenuated"
    return retained, "preserved" if ci_high < 1 or ci_low > 1 else "preserved_low_precision"


def run_analysis(analysis: str, build, risk: pd.DataFrame, risk_prior: pd.DataFrame, censored: pd.DataFrame, phreg) -> tuple[list, list]:
    """All models share the subjects complete for the fullest covariate set they are compared with."""
    clinical = [*BASE, *TERMS]
    specs = [("age_sex", BASE, risk), *[(f"plus_{t}", [*BASE, t], risk) for t in TERMS], ("clinical", clinical, risk)]
    rows, term_rows = [], []

    def complete(r: pd.DataFrame, covariates: list[str]) -> set:
        return set(r.dropna(subset=covariates)["analysis_subject_key"].astype(str))

    def fit(name, group, cp, covariates, reference):
        d = cp.dropna(subset=covariates)
        row = {"analysis": analysis, "comparison": group, "model": name, "n_subjects": d["analysis_subject_key"].nunique(),
               "events": int(d["event"].sum()), "n_parameters": len(covariates), "hr": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
               "p": np.nan, "log_hr": np.nan, "retained_log_hr": np.nan, "reading": "", "warning": False, "error": ""}
        try:
            result = fit_cox(phreg, d, covariates)
        except Exception as exc:  # a failed model is reported, never hidden
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            return None
        z = result["terms"][0]
        row.update({k: z[k] for k in ("hr", "ci95_low", "ci95_high", "p", "log_hr")}, warning=result["warning"])
        if reference is not None:
            row["retained_log_hr"], row["reading"] = read_row(z["log_hr"], reference, z["ci95_low"], z["ci95_high"])
        rows.append(row)
        term_rows.extend({"analysis": analysis, "comparison": group, "model": name, **t} for t in result["terms"])
        return z["log_hr"]

    def same_subjects(r: pd.DataFrame, ids: set) -> pd.DataFrame:
        return r[r["analysis_subject_key"].astype(str).isin(ids)]

    def attach(cp, r):
        columns = ["analysis_subject_key", *TERMS, "nyha_2plus", "prior_af"]
        return cp.merge(r[columns].assign(analysis_subject_key=r["analysis_subject_key"].astype(str)), on="analysis_subject_key", how="left")

    ids = complete(risk, clinical)
    primary = same_subjects(risk, ids)
    cp = attach(build(primary), primary)
    ref = None
    for name, covariates, _ in specs:
        value = fit(name, "clinical", cp, covariates, ref)
        ref = value if name == "age_sex" else ref

    for name, extra in (("nyha", "nyha_2plus"), ("prior_af", "prior_af")):
        ids2 = complete(risk, [*clinical, extra])
        sub = same_subjects(risk, ids2)
        sub_cp = attach(build(sub), sub)
        ref2 = fit(f"age_sex_{name}_subset", f"clinical_{name}", sub_cp, BASE, None)
        fit(f"clinical_plus_{name}", f"clinical_{name}", sub_cp, [*clinical, extra], ref2)

    for group, source, label in (("censor_srt", censored, "clinical_censor_srt"), ("prior_only", risk_prior, "clinical_prior_only")):
        ids3 = complete(source, clinical)
        sub = same_subjects(source, ids3)
        sub_cp = attach(build(sub), sub)
        ref3 = fit(f"age_sex_{group}_subset", group, sub_cp, BASE, None)
        fit(label, group, sub_cp, clinical, ref3)
    return rows, term_rows


def build_report(models: pd.DataFrame, terms: pd.DataFrame, disc: dict, match: pd.DataFrame, values: pd.DataFrame,
                 full: dict) -> str:
    lines = ["SMASH-HCM Step 40 -- H_z_06, heart failure and clinical markers", "",
             f"Full risk set: {full['n']} subjects, {full['events']} events", "", "Matching (+/-1 year)"]
    for _, r in match[match["window"].eq(str(WINDOW))].iterrows():
        lines.append(f"  {r['variable']:8s} matched {r['n_matched']}/{r['n_risk_set']}, events {r['events_matched']}, "
                     f"median delta {r['median_delta_years']:+.2f} y ({r['p05_delta']:+.2f} to {r['p95_delta']:+.2f})")
    lines += ["", "Value audit (check units before reading the models)"]
    for _, r in values[values["variable"].ne("hypovalue")].iterrows():
        lines.append(f"  {r['variable']:8s} numeric {r['n_numeric']} of {r['n_filled']}, out of range {r.get('n_out_of_range', 0)}, "
                     f"p1/median/p99 {r.get('p01', np.nan)}/{r.get('median', np.nan)}/{r.get('p99', np.nan)}")
    for (analysis, comparison), g in models.groupby(["analysis", "comparison"], sort=False):
        lines += ["", f"{analysis} | {comparison}"]
        for _, r in g.iterrows():
            if r["error"]:
                lines.append(f"  {r['model']:28s} not fitted: {r['error']}")
                continue
            kept = "" if not np.isfinite(r["retained_log_hr"]) else f"  kept {r['retained_log_hr']:.0%} {r['reading']}"
            flags = (" [warning]" if r["warning"] else "") + (" [few events per parameter]" if r["events"] < EVENTS_PER_PARAMETER * r["n_parameters"] else "")
            lines.append(f"  {r['model']:28s} n={r['n_subjects']:4d} ev={r['events']:3d} HR {r['hr']:.3f} "
                         f"({r['ci95_low']:.3f}-{r['ci95_high']:.3f}) p={r['p']:.3g}{kept}{flags}")
    lines += ["", "Covariates in the clinical model"]
    t = terms[terms["model"].eq("clinical") & terms["comparison"].eq("clinical")]
    for _, r in t.iterrows():
        lines.append(f"  {r['analysis']:12s} {r['term']:18s} HR {r['hr']:.3f} ({r['ci95_low']:.3f}-{r['ci95_high']:.3f}) p={r['p']:.3g}")
    lines += ["", f"Baseline discrimination on {disc['n_subjects']} subjects, {disc['events']} events (apparent, not optimism-corrected)",
              f"  C clinical {disc['c_clinical']:.3f}, C clinical + z06 {disc['c_clinical_z06']:.3f}, "
              f"difference {disc['delta_c']:+.3f} ({disc['delta_c_ci95_low']:+.3f} to {disc['delta_c_ci95_high']:+.3f}), "
              f"failed resamples {disc['bootstrap_failed']}",
              "", "Guardrails:", "- Complete-case analysis; subjects with echo data may differ from the full cohort.",
              "- LA and LVEF may lie on the causal pathway; attenuation is not proof of confounding.",
              "- Echo is static in the time-updated models. No multiplicity correction."]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    os.chdir(root)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.step39.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("smash_step39", args.step39.resolve())
    s39 = importlib.util.module_from_spec(spec)
    sys.modules["smash_step39"] = s39
    spec.loader.exec_module(s39)
    s26 = s39.load_module(args.step26.resolve(), "smash_step26")
    s23 = s39.load_module(args.step23.resolve(), "smash_step23")

    subj = pd.read_csv(s26.SUBJECT_CSV)
    subj["analysis_subject_key"] = subj["analysis_subject_key"].astype(str).str.strip()
    subj = s26.add_sex(subj)
    subj["baseline_age_10y"] = pd.to_numeric(subj["baseline_age"], errors="coerce") / 10.0
    baseline_z = pd.to_numeric(subj["baseline_z06"], errors="coerce")
    zmu, zsd = float(baseline_z.mean()), float(baseline_z.std(ddof=1))
    subj["z06_per_sd"] = (baseline_z - zmu) / zsd

    s2p, p2s = s23.load_cross()
    table = s39.build_event_table(subj, s23.eventmap(s2p, s23.load_events(p2s)))
    long = s26.prepare_longitudinal(subj, zmu, zsd)
    risk = s26.endpoint_subject_riskset(subj, ENDPOINT)

    print("Reading HCM_Echo and HCM_ClinicVisit")
    echo, echo_audit = load_echo(s26.read_xlsx_sheet_rows(s26.XLSX, "HCM_Echo"), s26.norm_pid)
    nyha, nyha_audit = load_nyha(s26.read_xlsx_sheet_rows(s26.XLSX, "HCM_ClinicVisit"), s26.norm_pid)
    values = pd.concat([echo_audit, nyha_audit], ignore_index=True)

    primary, match = add_clinical(risk, echo, nyha, table, WINDOW, s26.norm_pid)
    prior, match_prior = add_clinical(risk, echo, nyha, table, PRIOR_WINDOW, s26.norm_pid)
    censored = s39.censor_risk(primary, table, ["srt"])
    match_table = pd.DataFrame([*match, *match_prior])

    def tv(r):
        return s26.build_counting_process(r, long)

    def base(r):
        return s39.baseline_intervals(r)

    rows, term_rows = [], []
    for analysis, build in (("time_updated", tv), ("baseline", base)):
        print(f"Fitting {analysis}")
        r, t = run_analysis(analysis, build, primary, prior, censored, s26.PHReg)
        rows += r
        term_rows += t
    models, terms = pd.DataFrame(rows), pd.DataFrame(term_rows)

    clinical = ["baseline_age_10y", "male", *TERMS]
    complete = primary.dropna(subset=["z06_per_sd", *clinical])
    d = s39.baseline_intervals(complete)
    for col in TERMS:
        d[col] = complete[col].to_numpy(float)
    print("Bootstrapping discrimination")
    disc = discrimination(s26.PHReg, d, clinical, args.bootstrap, args.seed)

    models.to_csv(args.results_dir / "clinical_cox_models.csv", index=False)
    terms.to_csv(args.results_dir / "clinical_cox_terms.csv", index=False)
    match_table.to_csv(args.results_dir / "clinical_matching_audit.csv", index=False)
    values.to_csv(args.results_dir / "clinical_value_audit.csv", index=False)
    pd.DataFrame([disc]).to_csv(args.results_dir / "clinical_discrimination.csv", index=False)
    report = build_report(models, terms, disc, match_table, values, {"n": len(risk), "events": int(risk["endpoint_event"].sum())})
    (args.results_dir / "clinical_adjustment_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.results_dir}")


if __name__ == "__main__":
    main()
