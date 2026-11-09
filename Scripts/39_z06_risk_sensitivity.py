#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
39_z06_risk_sensitivity.py

SMASH-HCM -- Step 39
Does the H_z_06 / clinical-event association survive septal reduction therapy
(SRT), atrial fibrillation (AF), pacing and a stricter handling of age?

WHY THIS STEP EXISTS
--------------------
Step 38 showed that SRT lowers H_z_06. Steps 25/26 neither censor nor adjust for
SRT, so the time-updated HR may partly reflect the procedure or the patients who
receive it. A prior procedure may also affect the baseline value.

Risk sets, scaling, exposure construction and Cox fitting are REUSED from Step 26.
Event ages are REUSED from Step 23. The reference rows must reproduce Steps 25/26.

SCENARIOS
---------
reference              as published
exclude_prior_srt      subjects with SRT before baseline removed
exclude_prior_core     subjects with SRT, AF or pacing before baseline removed
censor_srt/af/pacer    follow-up ends at that event; earlier events are kept
censor_core            follow-up ends at the first of the three events
censor_core_lag30      as above with the 30-day exposure lag (time-updated)
freeze_pre_srt         follow-up kept, exposure stops updating at SRT
adjust_srt             SRT as a time-varying covariate
adjust_core            SRT, AF and pacing as time-varying covariates
age_timescale          attained age as time axis, left-truncated at baseline age
spline_age             baseline age as a 4-knot restricted cubic spline
genotype               core sarcomeric genotype added (complete cases)
genotype_reference     age/sex model on those same complete cases

A positive event with unknown age removes the subject, as in Steps 23 and 38.

READING RULE (fixed before seeing the results)
----------------------------------------------
retained = log HR / reference log HR, with the reference taken on the published
cohort (or on the same complete cases for genotype).
preserved                retained >= 0.70 and the 95% CI excludes 1
preserved_low_precision  retained >= 0.70 and the 95% CI includes 1
attenuated               retained < 0.70
The 0.70 threshold is a convention for reading this table, not a test. Events are
lost under censoring, so a wider CI alone is not evidence against the association.
Censoring is informative. Results are secondary and uncorrected for multiplicity.

Clinical covariates are not fitted yet: their names and timing must be checked.
share_workbook_columns.csv lists the workbook columns for that decision.

OUTPUT: Results/39_Z06_Risk_Sensitivity/

Usage: python Scripts\\39_z06_risk_sensitivity.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ENDPOINTS = [("Composite_HF", "primary"), ("Composite_VArrhythmia", "exploratory_sparse")]
ONE_DAY_YEARS = 1.0 / 365.2425
RETAINED_THRESHOLD = 0.70
SPARSE_EVENTS = 30
CORE = ["srt", "af", "pacer"]
GENOTYPE_POSITIVE = "Sarc+"
EXCLUSION_LABEL = "Excluded"


def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / "Results").exists() and (p / "Scripts").exists():
            return p
    raise RuntimeError("Cannot locate the project root (expected Results/ and Scripts/).")


def parse_args() -> argparse.Namespace:
    root = infer_project_root()
    p = argparse.ArgumentParser(description="Sensitivity of the H_z_06 event association.")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--step23", type=Path, default=root / "Scripts" / "23_sensitivity_major_events_longitudinal.py")
    p.add_argument("--step26", type=Path, default=root / "Scripts" / "26_z06_time_updated_cox.py")
    p.add_argument("--results-dir", type=Path, default=root / "Results" / "39_Z06_Risk_Sensitivity")
    return p.parse_args()


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_event_table(subj: pd.DataFrame, events: dict) -> pd.DataFrame:
    """Event times from baseline. t <= 1 day counts as prior, as in Step 24."""
    rows = []
    for row in subj.itertuples(index=False):
        sid = str(row.analysis_subject_key)
        event = events.get(sid)
        out = {"analysis_subject_key": sid, "in_step23_map": event is not None}
        event = event or {}
        for key in CORE:
            positive = bool(event.get(f"{key}_pos", False))
            age = event.get(f"{key}_age")
            out[f"{key}_pos"] = positive
            out[f"{key}_missing_age"] = positive and age is None
            out[f"{key}_t"] = age - float(row.baseline_age) if positive and age is not None else np.nan
        rows.append(out)
    return pd.DataFrame(rows).set_index("analysis_subject_key")


def missing_age_subjects(table: pd.DataFrame, keys: list[str]) -> set:
    mask = table[[f"{k}_missing_age" for k in keys]].any(axis=1)
    return set(table.index[mask])


def exclude_prior(risk: pd.DataFrame, table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    drop = missing_age_subjects(table, keys)
    for key in keys:
        drop |= set(table.index[table[f"{key}_t"] <= ONE_DAY_YEARS])
    return risk[~risk["analysis_subject_key"].astype(str).isin(drop)].copy()


def censor_risk(risk: pd.DataFrame, table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """End follow-up at the first listed event. An endpoint before it stays an event."""
    out = exclude_prior(risk, table, keys)
    sids = out["analysis_subject_key"].astype(str)
    first = table.loc[sids, [f"{k}_t" for k in keys]].min(axis=1).to_numpy(float)
    hit = np.isfinite(first) & (first < out["endpoint_time"].to_numpy(float))
    out["endpoint_time"] = np.where(hit, first, out["endpoint_time"].to_numpy(float))
    out["endpoint_event"] = np.where(hit, 0, out["endpoint_event"].to_numpy(int))
    return out


def freeze_exposure(long: pd.DataFrame, table: pd.DataFrame, key: str) -> pd.DataFrame:
    """Remove ECGs from the event onwards, so the pre-event value is carried forward."""
    t = long["analysis_subject_key"].astype(str).map(table[f"{key}_t"]).to_numpy(float)
    keep = ~(np.isfinite(t) & (long["ecg_time_y"].to_numpy(float) >= t))
    return long[keep].copy()


def add_time_varying(cp: pd.DataFrame, table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Split intervals at event times and add post_<event> indicators."""
    cp = cp[~cp["analysis_subject_key"].astype(str).isin(missing_age_subjects(table, keys))]
    rows = []
    for row in cp.to_dict("records"):
        times = table.loc[str(row["analysis_subject_key"]), [f"{k}_t" for k in keys]].to_numpy(float)
        cuts = sorted({float(t) for t in times if np.isfinite(t) and row["start"] < t < row["stop"]})
        edges = [row["start"], *cuts, row["stop"]]
        for i in range(len(edges) - 1):
            piece = dict(row)
            piece["start"], piece["stop"] = edges[i], edges[i + 1]
            piece["event"] = row["event"] if i == len(edges) - 2 else 0
            for key, t in zip(keys, times):
                piece[f"post_{key}"] = float(np.isfinite(t) and t <= edges[i])
            rows.append(piece)
    return pd.DataFrame(rows)


def rcs_terms(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Non-linear terms of a restricted cubic spline (Harrell)."""
    k = np.asarray(knots, float)
    scale = (k[-1] - k[0]) ** 2
    pos = lambda v: np.clip(v, 0.0, None) ** 3
    terms = []
    for j in range(len(k) - 2):
        terms.append((pos(x - k[j])
                      - pos(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                      + pos(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / scale)
    return np.column_stack(terms)


def add_spline(cp: pd.DataFrame, knots: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    out = cp.copy()
    terms = rcs_terms(out["baseline_age_10y"].to_numpy(float) * 10.0, knots)
    names = [f"age_rcs{j + 1}" for j in range(terms.shape[1])]
    for j, name in enumerate(names):
        out[name] = terms[:, j]
    return out, names


def to_age_scale(cp: pd.DataFrame) -> pd.DataFrame:
    out = cp.copy()
    age0 = out["baseline_age_10y"].to_numpy(float) * 10.0
    out["start"] = out["start"].to_numpy(float) + age0
    out["stop"] = out["stop"].to_numpy(float) + age0
    return out


def baseline_intervals(risk: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "analysis_subject_key": risk["analysis_subject_key"].astype(str).to_numpy(),
        "start": 0.0, "stop": risk["endpoint_time"].to_numpy(float),
        "event": risk["endpoint_event"].to_numpy(int), "z06_per_sd": risk["z06_per_sd"].to_numpy(float),
        "baseline_age_10y": risk["baseline_age_10y"].to_numpy(float), "male": risk["male"].to_numpy(float),
    })


def fit_row(fit_cox, cp: pd.DataFrame, covariates: list[str], meta: dict) -> dict:
    row = dict(meta, n_subjects=0, n_intervals=0, events=0, hr=np.nan, ci95_low=np.nan, ci95_high=np.nan,
               p=np.nan, log_hr=np.nan, se=np.nan, convergence_warning=False, error="")
    if cp.empty:
        row["error"] = "no intervals"
        return row
    d = cp.dropna(subset=covariates).copy()
    row.update(n_subjects=int(d["analysis_subject_key"].nunique()), n_intervals=int(len(d)), events=int(d["event"].sum()))
    if row["events"] < 3:
        row["error"] = "fewer than 3 events"
        return row
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            terms = fit_cox(d, covariates)
        row["convergence_warning"] = any("Convergence" in w.category.__name__ for w in caught)
    except Exception as exc:  # a failed sensitivity model is reported, never hidden
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    z = next(t for t in terms if t["term"] == "z06_per_sd")
    row.update({k: z[k] for k in ("hr", "ci95_low", "ci95_high", "p", "log_hr", "se")})
    for t in terms:
        if t["term"].startswith(("post_", "genotype")):
            row[f"hr_{t['term']}"] = t["hr"]
            row[f"ci95_low_{t['term']}"] = t["ci95_low"]
            row[f"ci95_high_{t['term']}"] = t["ci95_high"]
    return row


def classify(row, reference_log_hr: float) -> tuple[float, str]:
    if not np.isfinite(row["log_hr"]) or not np.isfinite(reference_log_hr) or reference_log_hr == 0:
        return np.nan, "not_estimable"
    retained = row["log_hr"] / reference_log_hr
    if retained < RETAINED_THRESHOLD:
        return retained, "attenuated"
    return retained, "preserved" if row["ci95_high"] < 1 or row["ci95_low"] > 1 else "preserved_low_precision"


def run_endpoint(endpoint: str, family: str, subj: pd.DataFrame, long: pd.DataFrame, table: pd.DataFrame,
                 s26, knots: np.ndarray) -> tuple[list, dict]:
    risk = s26.endpoint_subject_riskset(subj, endpoint)
    fit = s26.fit_tv_cox
    adj = ["z06_per_sd", "baseline_age_10y", "male"]
    age_scale = ["z06_per_sd", "male"]
    rows = []

    def add(analysis, scenario, cp, covariates):
        rows.append(fit_row(fit, cp, covariates,
                            {"endpoint": endpoint, "family": family, "analysis": analysis, "scenario": scenario}))

    tv = lambda r, lag=0.0, data=long: s26.build_counting_process(r, data, lag_years=lag)
    base = lambda r: baseline_intervals(r)
    ref_tv, ref_base = tv(risk), base(risk)

    for analysis, make, ref in (("time_updated", tv, ref_tv), ("baseline", base, ref_base)):
        add(analysis, "reference", ref, adj)
        add(analysis, "exclude_prior_srt", make(exclude_prior(risk, table, ["srt"])), adj)
        add(analysis, "exclude_prior_core", make(exclude_prior(risk, table, CORE)), adj)
        for key in CORE:
            add(analysis, f"censor_{key}", make(censor_risk(risk, table, [key])), adj)
        add(analysis, "censor_core", make(censor_risk(risk, table, CORE)), adj)
        add(analysis, "age_timescale", to_age_scale(ref), age_scale)
        add(analysis, "censor_core_age_timescale", to_age_scale(make(censor_risk(risk, table, CORE))), age_scale)
        spline, names = add_spline(ref.dropna(subset=["baseline_age_10y"]), knots)
        add(analysis, "spline_age", spline, [*adj, *names])
        with_genotype = ref.merge(subj[["analysis_subject_key", "genotype_positive"]], on="analysis_subject_key", how="left")
        add(analysis, "genotype_reference", with_genotype.dropna(subset=["genotype_positive"]), adj)
        add(analysis, "genotype", with_genotype, [*adj, "genotype_positive"])

    add("time_updated", "censor_core_lag30", tv(censor_risk(risk, table, CORE), s26.LAG_YEARS), adj)
    frozen = exclude_prior(risk, table, ["srt"])
    add("time_updated", "freeze_pre_srt", tv(frozen, data=freeze_exposure(long, table, "srt")), adj)
    for name, keys in (("adjust_srt", ["srt"]), ("adjust_core", CORE)):
        add("time_updated", name, add_time_varying(ref_tv, table, keys), [*adj, *[f"post_{k}" for k in keys]])

    events = risk[risk["endpoint_event"].eq(1)]
    audit = {"endpoint": endpoint, "n_risk_set": len(risk), "events": len(events)}
    ids_all, ids_event = risk["analysis_subject_key"].astype(str), events["analysis_subject_key"].astype(str)
    for key in CORE:
        t_all, t_event = table.loc[ids_all, f"{key}_t"], table.loc[ids_event, f"{key}_t"].to_numpy(float)
        end = events["endpoint_time"].to_numpy(float)
        audit[f"{key}_prior_to_baseline"] = int((t_all <= ONE_DAY_YEARS).sum())
        audit[f"{key}_during_followup"] = int(((t_all > ONE_DAY_YEARS) & (t_all < risk["endpoint_time"].to_numpy(float))).sum())
        audit[f"{key}_missing_age"] = int(table.loc[ids_all, f"{key}_missing_age"].sum())
        audit[f"events_after_{key}"] = int((t_event < end).sum())
    return rows, audit


def catalog_workbook(path: Path, reader) -> pd.DataFrame:
    """Column names and fill counts only; no patient-level value is written."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        sheets = [s.attrib["name"] for s in ET.fromstring(z.read("xl/workbook.xml")).find(f"{ns}sheets")]
    rows = []
    for sheet in sheets:
        try:
            data = pd.DataFrame(reader(path, sheet))
        except Exception as exc:
            rows.append({"sheet": sheet, "column": "", "n_rows": 0, "n_filled": 0, "note": f"{type(exc).__name__}: {exc}"})
            continue
        for col in data.columns:
            filled = data[col].notna() & data[col].astype(str).str.strip().ne("")
            rows.append({"sheet": sheet, "column": col, "n_rows": len(data), "n_filled": int(filled.sum()), "note": ""})
    return pd.DataFrame(rows)


def add_classification(models: pd.DataFrame) -> pd.DataFrame:
    out = models.copy()
    out["retained_log_hr"], out["reading"] = np.nan, ""
    for (endpoint, analysis), group in out.groupby(["endpoint", "analysis"]):
        ref = group.loc[group["scenario"].eq("reference"), "log_hr"].iloc[0]
        gen = group.loc[group["scenario"].eq("genotype_reference"), "log_hr"]
        for idx, row in group.iterrows():
            if row["scenario"] == "reference":
                continue
            base = float(gen.iloc[0]) if row["scenario"] == "genotype" and len(gen) else ref
            out.loc[idx, ["retained_log_hr", "reading"]] = classify(row, base)
    return out


def build_report(models: pd.DataFrame, audits: pd.DataFrame, published: list[str], clinical_note: str) -> str:
    lines = ["SMASH-HCM Step 39 -- Sensitivity of the H_z_06 event association", "", "Published values check"]
    lines += [f"  {x}" for x in published]
    for _, a in audits.iterrows():
        lines += ["", f"{a['endpoint']}: {a['n_risk_set']} subjects, {a['events']} events"]
        for key in CORE:
            lines.append(f"  {key:5s}: prior {a[f'{key}_prior_to_baseline']}, during follow-up {a[f'{key}_during_followup']}, "
                         f"missing age {a[f'{key}_missing_age']}, endpoint events after it {a[f'events_after_{key}']}")
    for (endpoint, analysis), g in models.groupby(["endpoint", "analysis"], sort=False):
        lines += ["", f"{endpoint} | {analysis}"]
        if g["family"].iloc[0] != "primary":
            lines.append("  EXPLORATORY: sparse events, read direction only")
        for _, r in g.iterrows():
            if r["error"]:
                lines.append(f"  {r['scenario']:27s} not fitted: {r['error']}")
                continue
            flags = (" [warning]" if r["convergence_warning"] else "") + (" [sparse]" if r["events"] < SPARSE_EVENTS else "")
            retained = "" if not np.isfinite(r["retained_log_hr"]) else f"  kept {r['retained_log_hr']:.0%} {r['reading']}"
            lines.append(f"  {r['scenario']:27s} n={r['n_subjects']:4d} ev={r['events']:3d} "
                         f"HR {r['hr']:.3f} ({r['ci95_low']:.3f}-{r['ci95_high']:.3f}) p={r['p']:.3g}{retained}{flags}")
            for col in r.index:
                if col.startswith("hr_") and np.isfinite(r[col]):
                    term = col[3:]
                    lines.append(f"      {term}: HR {r[col]:.2f} ({r[f'ci95_low_{term}']:.2f}-{r[f'ci95_high_{term}']:.2f})")
    lines += ["", "Reading rule: preserved if at least 70% of the reference log HR is kept and the CI excludes 1;",
              "preserved_low_precision if kept but the CI includes 1; attenuated otherwise.",
              "", "Guardrails:", "- Censoring is informative and removes events; these are secondary analyses.",
              "- No multiplicity correction. Event ages may be coarse in the registry.",
              "- Post-event indicators are descriptive, not treatment effects.", f"- {clinical_note}"]
    return "\n".join(lines) + "\n"


def check_published(models: pd.DataFrame, root: Path) -> list[str]:
    out = []
    sources = (("baseline", root / "Results" / "25_Z06_BaselineCox" / "z06_baseline_cox_results.csv", {}),
               ("time_updated", root / "Results" / "26_Z06_TimeUpdatedCox" / "z06_time_updated_cox_results.csv", {"lag": "no_lag"}))
    for analysis, path, extra in sources:
        if not path.is_file():
            out.append(f"{analysis}: {path.name} not found, not checked")
            continue
        old = pd.read_csv(path)
        old = old[old["endpoint"].eq("Composite_HF") & old["model"].eq("age_sex_adjusted")]
        for col, value in extra.items():
            old = old[old[col].eq(value)]
        new = models[models["endpoint"].eq("Composite_HF") & models["analysis"].eq(analysis) & models["scenario"].eq("reference")]
        if old.empty or new.empty:
            out.append(f"{analysis}: no comparable row")
            continue
        a, b = float(old.iloc[0]["hr"]), float(new.iloc[0]["hr"])
        out.append(f"{analysis}: previous HR {a:.3f}, this run {b:.3f}, {'MATCH' if abs(a - b) < 0.0015 else 'MISMATCH: do not use this run'}")
    return out


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    os.chdir(root)  # Step 23 uses paths relative to the project root.
    args.results_dir.mkdir(parents=True, exist_ok=True)
    s26 = load_module(args.step26.resolve(), "smash_step26")
    s23 = load_module(args.step23.resolve(), "smash_step23")

    subj = pd.read_csv(s26.SUBJECT_CSV)
    subj["analysis_subject_key"] = subj["analysis_subject_key"].astype(str).str.strip()
    subj = s26.add_sex(subj)
    subj["baseline_age_10y"] = pd.to_numeric(subj["baseline_age"], errors="coerce") / 10.0
    baseline_z = pd.to_numeric(subj["baseline_z06"], errors="coerce")
    zmu, zsd = float(baseline_z.mean()), float(baseline_z.std(ddof=1))
    subj["z06_per_sd"] = (baseline_z - zmu) / zsd

    genotype = pd.read_csv(args.root / "Results" / "18_Genotype_Longitudinal" / "genotype_subject_cohort.csv",
                           usecols=["analysis_subject_key", "primary_genotype_group"]).drop_duplicates("analysis_subject_key")
    genotype["analysis_subject_key"] = genotype["analysis_subject_key"].astype(str).str.strip()
    group = subj["analysis_subject_key"].map(genotype.set_index("analysis_subject_key")["primary_genotype_group"])
    subj["genotype_positive"] = np.where(group.isna() | group.eq(EXCLUSION_LABEL), np.nan, (group == GENOTYPE_POSITIVE).astype(float))

    s2p, p2s = s23.load_cross()
    events = s23.eventmap(s2p, s23.load_events(p2s))
    table = build_event_table(subj, events)
    if not table["in_step23_map"].all():
        print(f"WARNING: {(~table['in_step23_map']).sum()} subjects have no Step-23 event record; treated as event-free.")
    if "has_srt" in subj:
        a = subj["has_srt"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        print(f"SRT agreement Step 24 vs Step 23: {(a == table.loc[subj['analysis_subject_key'], 'srt_pos'].to_numpy()).mean():.1%}")

    long = s26.prepare_longitudinal(subj, zmu, zsd)
    knots = np.percentile(pd.to_numeric(subj["baseline_age"], errors="coerce").dropna(), [5, 35, 65, 95])

    rows, audits = [], []
    for endpoint, family in ENDPOINTS:
        print(f"Fitting {endpoint}")
        r, a = run_endpoint(endpoint, family, subj, long, table, s26, knots)
        rows += r
        audits.append(a)
    models = add_classification(pd.DataFrame(rows))
    audit = pd.DataFrame(audits)
    published = check_published(models, root)
    catalog = catalog_workbook(s26.XLSX, s26.read_xlsx_sheet_rows)

    models.to_csv(args.results_dir / "z06_risk_sensitivity_models.csv", index=False)
    audit.to_csv(args.results_dir / "z06_risk_event_audit.csv", index=False)
    catalog.to_csv(args.results_dir / "share_workbook_columns.csv", index=False)
    report = build_report(models, audit, published, "Clinical covariates not fitted; see share_workbook_columns.csv.")
    (args.results_dir / "z06_risk_sensitivity_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.results_dir}")


if __name__ == "__main__":
    main()
