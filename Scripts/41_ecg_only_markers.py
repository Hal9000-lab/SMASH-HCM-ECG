#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
41_ecg_only_markers.py

SMASH-HCM -- Step 41
Electrical markers defined from ECG latents alone, then tested once against clinic.

STAGE 1: DEFINITION. Uses H_z_00..07 only: no echo, no events, no treatment data.
The only group information is the cohort itself (HCM ECG versus PTB-XL normal ECG).
Definitions come from the Step-37 files, the same visits used for the ageing work.
z02, z06    single coordinates (not reoriented; HR comparable with Steps 25/26)
signature   linear discriminant HCM first visit versus normal ECGs
trait       direction with the highest between/within-subject variance ratio
distance    Mahalanobis distance from the normal centroid in H space
Composites are oriented so that higher means more HCM-like. Weights are saved to
ecg_only_marker_weights.csv before any clinical file is opened.

STAGE 2: TEST. The frozen markers are applied to the Step-24 cohorts.
Heart failure : age/sex in the full risk set, then on complete cases age/sex,
                + LA + LVEF, + all four echo markers, and censored at SRT.
AF, VA        : age/sex only; VA is descriptive because of sparse events.
Echo overlap  : R2 of each baseline marker on age, sex and the four echo markers.
Discrimination: apparent C of the clinical model with and without the marker.

READING RULE (fixed before seeing the results)
----------------------------------------------
candidate = BH q < 0.05 across the five markers for heart failure (age/sex), AND
            at least 70% of the log HR kept with all four echo markers on the same
            subjects, AND the 95% CI excludes 1 in that model.
overlaps_structure = q < 0.05 but the second or third condition fails.
no_association     = q >= 0.05.
Secondary, exploratory analysis. HCM versus PTB-XL also differs by centre and device,
so AUC is not diagnostic performance. Findings need independent replication.

OUTPUT: Results/41_ECG_Only_Markers/
Usage: python Scripts\\41_ecg_only_markers.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import rankdata

Z_COLS = [f"H_z_{k:02d}" for k in range(8)]
MARKERS = ["H_z_02", "H_z_06", "signature", "trait", "distance"]
ECHO_TERMS = ["max_lvt_per5mm", "la_per5mm", "lvef_per5pct", "lvot_log1p"]
BASE = ["marker_per_sd", "baseline_age_10y", "male"]
RETAINED = 0.70
Q_LIMIT = 0.05


def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / "Results").exists() and (p / "Scripts").exists():
            return p
    raise RuntimeError("Cannot locate the project root (expected Results/ and Scripts/).")


def parse_args() -> argparse.Namespace:
    root = infer_project_root()
    r = root / "Results"
    p = argparse.ArgumentParser(description="ECG-only electrical markers.")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--hcm-visits", type=Path, default=r / "37_Z06_Age_Sensitivity" / "hcm_visit_values.csv")
    p.add_argument("--normal", type=Path, default=r / "37_Z06_Age_Sensitivity" / "normal_reference_values.csv")
    p.add_argument("--mimic-visits", type=Path, default=r / "35_MIMIC_Longitudinal_Hz06_Replication" / "mimic_visit_windows_30d.csv")
    p.add_argument("--step23", type=Path, default=root / "Scripts" / "23_sensitivity_major_events_longitudinal.py")
    p.add_argument("--step26", type=Path, default=root / "Scripts" / "26_z06_time_updated_cox.py")
    p.add_argument("--step39", type=Path, default=root / "Scripts" / "39_z06_risk_sensitivity.py")
    p.add_argument("--step40", type=Path, default=root / "Scripts" / "40_z06_clinical_adjustment.py")
    p.add_argument("--results-dir", type=Path, default=r / "41_ECG_Only_Markers")
    return p.parse_args()


def auc(a: np.ndarray, b: np.ndarray) -> float:
    r = rankdata(np.r_[a, b])
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def icc(values: pd.Series, subject: pd.Series) -> float:
    """One-way ICC for unbalanced repeated measures."""
    d = pd.DataFrame({"v": np.asarray(values, float), "s": np.asarray(subject)}).dropna()
    g = d.groupby("s")["v"]
    sizes = g.size()
    d = d[d["s"].isin(sizes.index[sizes >= 1])]
    mean = g.transform("mean")
    n_subjects, n = len(sizes), len(d)
    msb = float((sizes * (g.mean() - d["v"].mean()) ** 2).sum()) / (n_subjects - 1)
    msw = float(((d["v"] - mean) ** 2).sum()) / (n - n_subjects)
    n0 = (n - float((sizes ** 2).sum()) / n) / (n_subjects - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def define_markers(hcm: pd.DataFrame, normal: pd.DataFrame) -> dict:
    """ECG-only stage. It receives only latents, subject keys and cohort membership."""
    first = hcm.sort_values("t").drop_duplicates("subject")
    F, N = first[Z_COLS].to_numpy(float), normal[Z_COLS].to_numpy(float)
    pooled = (np.cov(F.T) + np.cov(N.T)) / 2.0
    signature = np.linalg.solve(pooled, F.mean(0) - N.mean(0))
    signature = signature / np.linalg.norm(signature)

    X = hcm[Z_COLS].to_numpy(float)
    within = np.cov((X - hcm.groupby("subject")[Z_COLS].transform("mean").to_numpy(float)).T)
    between = np.cov(hcm.groupby("subject")[Z_COLS].mean().to_numpy(float).T)
    values, vectors = eigh(between, within)
    trait = vectors[:, int(np.argmax(values))]
    trait = trait / np.linalg.norm(trait)
    if (F @ trait).mean() < (N @ trait).mean():
        trait = -trait
    return {"signature": signature, "trait": trait, "normal_mean": N.mean(0), "normal_precision": np.linalg.inv(np.cov(N.T))}


def apply_markers(d: pd.DataFrame, spec: dict) -> pd.DataFrame:
    X = d[Z_COLS].to_numpy(float)
    out = pd.DataFrame(index=d.index)
    out["H_z_02"], out["H_z_06"] = d["H_z_02"], d["H_z_06"]
    out["signature"], out["trait"] = X @ spec["signature"], X @ spec["trait"]
    delta = X - spec["normal_mean"]
    out["distance"] = np.sqrt(np.einsum("ij,jk,ik->i", delta, spec["normal_precision"], delta))
    return out


def ecg_summary(hcm: pd.DataFrame, normal: pd.DataFrame, spec: dict, mimic: pd.DataFrame | None) -> pd.DataFrame:
    h, n = apply_markers(hcm, spec), apply_markers(normal, spec)
    first = hcm.sort_values("t").drop_duplicates("subject").index
    rows = []
    for m in MARKERS:
        row = {"marker": m, "auc_hcm_vs_normal": auc(h.loc[first, m].to_numpy(), n[m].to_numpy()), "icc_share": icc(h[m], hcm["subject"])}
        if mimic is not None:
            x = apply_markers(mimic, spec)
            row["icc_mimic"] = icc(x[m], mimic["subject"])
            row["auc_mimic_vs_normal"] = auc(x.groupby(mimic["subject"])[m].mean().to_numpy(), n[m].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_mimic(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    if not set(Z_COLS) <= set(d.columns):
        return None
    d = d.dropna(subset=Z_COLS).copy()
    d["subject"] = d["subject_id"].astype(str)
    return d


def bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    out = np.empty(len(p))
    out[order] = q
    return out


def r_squared(y: np.ndarray, X: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(X)), X])
    fitted = X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def marker_rows(marker, risk, primary, censored, long, s26, s39, s40, va_risk, af_risk) -> tuple[list, dict]:
    """long carries the marker in z06_per_sd, which is how Step 26 reads the exposure."""
    rows = []
    phreg = s26.PHReg

    def fit(analysis, label, cp, covariates, ref=None):
        d = cp.dropna(subset=covariates)
        row = {"marker": marker, "analysis": analysis, "model": label, "n_subjects": d["analysis_subject_key"].nunique(), "events": int(d["event"].sum()),
               "hr": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p": np.nan, "log_hr": np.nan, "retained_log_hr": np.nan, "warning": False, "error": ""}
        try:
            result = s40.fit_cox(phreg, d, covariates)
            t = result["terms"][0]
            row.update({k: t[k] for k in ("hr", "ci95_low", "ci95_high", "p", "log_hr")}, warning=result["warning"])
            if ref is not None:
                row["retained_log_hr"] = t["log_hr"] / ref
        except Exception as exc:  # a failed model is reported, never hidden
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        return row["log_hr"]

    def tv(r):
        return s26.build_counting_process(r, long).rename(columns={"z06_per_sd": "marker_per_sd"})

    def base(r):
        return s39.baseline_intervals(r).rename(columns={"z06_per_sd": "marker_per_sd"})

    def attach(cp, r):
        cols = r[["analysis_subject_key", *ECHO_TERMS]].assign(analysis_subject_key=r["analysis_subject_key"].astype(str))
        return cp.merge(cols, on="analysis_subject_key", how="left")

    complete = primary.dropna(subset=["z06_per_sd", "baseline_age_10y", "male", *ECHO_TERMS])
    keep = set(complete["analysis_subject_key"].astype(str))
    censored_complete = censored[censored["analysis_subject_key"].astype(str).isin(keep)]
    for analysis, build in (("time_updated", tv), ("baseline", base)):
        fit(analysis, "age_sex_full", build(risk), BASE)
        cp = attach(build(complete), complete)
        ref = fit(analysis, "age_sex_complete", cp, BASE)
        fit(analysis, "plus_la_lvef", cp, [*BASE, "la_per5mm", "lvef_per5pct"], ref)
        fit(analysis, "plus_four_echo", cp, [*BASE, *ECHO_TERMS], ref)
        fit(analysis, "plus_four_echo_censor_srt", attach(build(censored_complete), censored_complete), [*BASE, *ECHO_TERMS], ref)
        fit(analysis, "af_age_sex", build(af_risk), BASE)
        fit(analysis, "va_age_sex_descriptive", build(va_risk), BASE)

    b = complete[["z06_per_sd", "baseline_age_10y", "male", *ECHO_TERMS]].to_numpy(float)
    d = base(complete)
    for t in ECHO_TERMS:
        d[t] = complete[t].to_numpy(float)
    clinical = ["baseline_age_10y", "male", *ECHO_TERMS]
    d = d.rename(columns={"marker_per_sd": "z06_per_sd"})
    c0, c1 = s40.delta_c(phreg, d, clinical)
    extra = {"marker": marker, "r2_on_age_sex_echo": r_squared(b[:, 0], b[:, 1:]), "c_clinical": c0, "c_clinical_marker": c1, "delta_c": c1 - c0}
    return rows, extra


def classify(models: pd.DataFrame) -> pd.DataFrame:
    a = models[models["analysis"].eq("time_updated") & models["model"].eq("age_sex_full")].set_index("marker")
    e = models[models["analysis"].eq("time_updated") & models["model"].eq("plus_four_echo")].set_index("marker")
    out = pd.DataFrame({"hr_age_sex": a["hr"], "p": a["p"]})
    out["q"] = bh(out["p"].fillna(1.0).to_numpy())
    out["hr_echo"], out["echo_low"], out["echo_high"], out["retained"] = e["hr"], e["ci95_low"], e["ci95_high"], e["retained_log_hr"]
    keeps = (out["retained"] >= RETAINED) & ((out["echo_high"] < 1) | (out["echo_low"] > 1))
    out["reading"] = np.where(out["q"] >= Q_LIMIT, "no_association", np.where(keeps, "candidate", "overlaps_structure"))
    return out.reset_index()


def build_report(summary, reading, models, extra, weights) -> str:
    lines = ["SMASH-HCM Step 41 -- Markers defined from ECG latents alone", "", "Stage 1 weights (fixed before clinical data)"]
    lines += [f"  {r['marker']:10s} " + " ".join(f"{c[-2:]}:{r[c]:+.2f}" for c in Z_COLS) for _, r in weights.iterrows()]
    lines += ["", "ECG properties"]
    for _, r in summary.iterrows():
        tail = f"  MIMIC ICC {r['icc_mimic']:.3f}, AUC {r['auc_mimic_vs_normal']:.3f}" if "icc_mimic" in r else ""
        lines.append(f"  {r['marker']:10s} AUC HCM vs PTB-XL {r['auc_hcm_vs_normal']:.3f}, ICC SHaRe {r['icc_share']:.3f}{tail}")
    lines += ["", "Reading (time-updated heart failure)"]
    for _, r in reading.iterrows():
        lines.append(f"  {r['marker']:10s} HR {r['hr_age_sex']:.3f} q={r['q']:.3g}; with four echo HR {r['hr_echo']:.3f} ({r['echo_low']:.3f}-{r['echo_high']:.3f}), kept {r['retained']:.0%}: {r['reading']}")
    lines += ["", "Echo overlap and discrimination on complete cases"]
    for _, r in extra.iterrows():
        lines.append(f"  {r['marker']:10s} R2 {r['r2_on_age_sex_echo']:.3f}; C clinical {r['c_clinical']:.3f} -> {r['c_clinical_marker']:.3f} ({r['delta_c']:+.3f}, apparent)")
    for (marker, analysis), g in models.groupby(["marker", "analysis"], sort=False):
        lines += ["", f"{marker} | {analysis}"]
        for _, r in g.iterrows():
            if r["error"]:
                lines.append(f"  {r['model']:26s} not fitted: {r['error']}")
                continue
            kept = "" if not np.isfinite(r["retained_log_hr"]) else f"  kept {r['retained_log_hr']:.0%}"
            lines.append(f"  {r['model']:26s} n={r['n_subjects']:4d} ev={r['events']:3d} HR {r['hr']:.3f} ({r['ci95_low']:.3f}-{r['ci95_high']:.3f}) p={r['p']:.3g}{kept}{' [warning]' if r['warning'] else ''}")
    lines += ["", "Guardrails:", "- Exploratory. HCM and PTB-XL differ by centre and device; AUC is not diagnostic accuracy.",
              "- HRs for z02 and z06 keep their native sign; composites are oriented so higher is more HCM-like.",
              "- Complete-case echo analysis. No optimism correction for C. Replication is required."]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: only the two latent files are opened here.
    hcm = pd.read_csv(args.hcm_visits, usecols=["subject", "t", *Z_COLS]).dropna(subset=Z_COLS).reset_index(drop=True)
    normal = pd.read_csv(args.normal, usecols=Z_COLS).dropna(subset=Z_COLS).reset_index(drop=True)
    spec = define_markers(hcm, normal)
    weights = pd.DataFrame([{"marker": k, **dict(zip(Z_COLS, spec[k]))} for k in ("signature", "trait")])
    weights.to_csv(args.results_dir / "ecg_only_marker_weights.csv", index=False)
    summary = ecg_summary(hcm, normal, spec, load_mimic(args.mimic_visits))
    summary.to_csv(args.results_dir / "ecg_only_marker_properties.csv", index=False)

    # Stage 2: clinical data enter only now, with the markers frozen.
    os.chdir(root)
    s39 = load_module(args.step39.resolve(), "smash_step39")
    s40 = load_module(args.step40.resolve(), "smash_step40")
    s26 = s39.load_module(args.step26.resolve(), "smash_step26")
    s23 = s39.load_module(args.step23.resolve(), "smash_step23")

    subj = pd.read_csv(s26.SUBJECT_CSV)
    subj["analysis_subject_key"] = subj["analysis_subject_key"].astype(str).str.strip()
    subj = s26.add_sex(subj)
    subj["baseline_age_10y"] = pd.to_numeric(subj["baseline_age"], errors="coerce") / 10.0
    s2p, p2s = s23.load_cross()
    table = s39.build_event_table(subj, s23.eventmap(s2p, s23.load_events(p2s)))

    latents = pd.read_csv(s26.LONG_CSV, usecols=["ecg_id", *Z_COLS]).drop_duplicates("ecg_id")
    long0 = s26.prepare_longitudinal(subj, 0.0, 1.0).merge(latents, on="ecg_id", how="left", suffixes=("_step26", ""))
    long0 = long0.dropna(subset=Z_COLS).reset_index(drop=True)
    values = apply_markers(long0, spec)
    baseline_rows = long0.groupby("analysis_subject_key")["ecg_time_y"].idxmin()

    echo, _ = s40.load_echo(s26.read_xlsx_sheet_rows(s26.XLSX, "HCM_Echo"), s26.norm_pid)
    nyha, _ = s40.load_nyha(s26.read_xlsx_sheet_rows(s26.XLSX, "HCM_ClinicVisit"), s26.norm_pid)

    rows, extras = [], []
    for marker in MARKERS:
        print(f"Testing {marker}")
        at_baseline = pd.Series(values.loc[baseline_rows, marker].to_numpy(), index=long0.loc[baseline_rows, "analysis_subject_key"].to_numpy())
        mu, sd = float(at_baseline.mean()), float(at_baseline.std(ddof=1))
        long = long0.copy()
        long["z06_per_sd"] = (values[marker] - mu) / sd
        s = subj.copy()
        s["z06_per_sd"] = (s["analysis_subject_key"].map(at_baseline) - mu) / sd
        s = s[s["z06_per_sd"].notna()]
        risk = s26.endpoint_subject_riskset(s, "Composite_HF")
        primary, _ = s40.add_clinical(risk, echo, nyha, table, s40.WINDOW, s26.norm_pid)
        censored = s39.censor_risk(primary, table, ["srt"])
        va = s26.endpoint_subject_riskset(s, "Composite_VArrhythmia")
        af = s26.endpoint_subject_riskset(s, "Atrial_Fibrillation")
        r, e = marker_rows(marker, risk, primary, censored, long, s26, s39, s40, va, af)
        rows += r
        extras.append(e)

    models, extra = pd.DataFrame(rows), pd.DataFrame(extras)
    reading = classify(models)
    models.to_csv(args.results_dir / "ecg_only_marker_models.csv", index=False)
    reading.to_csv(args.results_dir / "ecg_only_marker_reading.csv", index=False)
    extra.to_csv(args.results_dir / "ecg_only_marker_echo_overlap.csv", index=False)
    report = build_report(summary, reading, models, extra, weights)
    (args.results_dir / "ecg_only_marker_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.results_dir}")


if __name__ == "__main__":
    main()
