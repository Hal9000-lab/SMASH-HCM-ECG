#!/usr/bin/env python3
"""
Step 26 -- Time-updated HCM z06 and future clinical events (Cox PH).

Purpose
-------
Test whether the current value of the HCM-oriented lateral/frontal QRS-force
coordinate H_z_06, updated at each valid ECG, is associated with subsequent
clinical events.

Design
------
- exact ECG-to-SHaRe mappings from Step 24a
- time origin = first valid ECG with finite H_z_06
- counting-process (start, stop] intervals with last-observation-carried-forward
  H_z_06; no ECG occurring after an event contributes to exposure
- endpoint-specific clean risk sets inherited from Step 24a: prevalent events,
  positive events without usable timing, and subjects without positive follow-up
  are excluded
- common z06 scaling uses the baseline mean/SD in the exact-mapped cohort, so
  HRs are directly comparable with Step 25 baseline-Cox HRs
- Model 1: time-updated z06 only
- Model 2: time-updated z06 + baseline age + sex
- standard Cox information-matrix standard errors
- primary focus: Composite_HF
- exploratory sparse endpoint: Composite_VArrhythmia
- additional context endpoints: Atrial_Fibrillation,
  Unplanned_CV_Hospitalization, Composite_Overall
- 30-day lag sensitivity: each ECG value becomes active 30 days after recording;
  follow-up begins at day 30, avoiding use of ECG measurements immediately
  preceding an event

This script does not retrain any latent representation.

Canonical project inputs / output
---------------------------------
- Results/24_Z06_EventRiskFeasibility/z06_event_feasibility_subject_level.csv
- Results/16_Longitudinal_DualManifold/longitudinal_dual_manifold.csv
- Data/SHaReData-FLO.xlsx
- outputs: Results/26_Z06_TimeUpdatedCox/
"""
from __future__ import annotations

import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests

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
    raise RuntimeError("Cannot locate SMASH-HCM project root (expected Results/ and Scripts/).")


def locate_required_file(label: str, *candidates: Path) -> Path:
    """Return the first existing candidate path, otherwise raise a clear error."""
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    tried = "\n".join(f"  - {Path(p)}" for p in candidates)
    raise FileNotFoundError(f"Cannot locate {label}. Tried:\n{tried}")


ROOT = infer_project_root()
RESULTS = ROOT / "Results"
DATA = ROOT / "Data"

SUBJECT_CSV = RESULTS / "24_Z06_EventRiskFeasibility" / "z06_event_feasibility_subject_level.csv"
LONG_CSV = RESULTS / "16_Longitudinal_DualManifold" / "longitudinal_dual_manifold.csv"
XLSX = DATA / "SHaReData-FLO.xlsx"
OUT = RESULTS / "26_Z06_TimeUpdatedCox"
OUT.mkdir(parents=True, exist_ok=True)

for label, path in (
    ("Step 24 event-feasibility subject table", SUBJECT_CSV),
    ("Step 16 longitudinal dual-manifold table", LONG_CSV),
    ("SHaRe workbook", XLSX),
):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}:\n{path}")

print(f"Project root   : {ROOT}")
print(f"Step 24 input  : {SUBJECT_CSV}")
print(f"Step 16 input  : {LONG_CSV}")
print(f"SHaRe workbook : {XLSX}")
print(f"Step 26 output : {OUT}")

ENDPOINTS = [
    ("Composite_HF", "primary"),
    ("Composite_VArrhythmia", "exploratory_sparse"),
    ("Atrial_Fibrillation", "context"),
    ("Unplanned_CV_Hospitalization", "context"),
    ("Composite_Overall", "context"),
]
DAYS_PER_YEAR = 365.2425
LAG_DAYS = 30.0
LAG_YEARS = LAG_DAYS / DAYS_PER_YEAR

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def col_index(cell_ref: str) -> int:
    m = re.match(r"([A-Z]+)", cell_ref)
    n = 0
    for ch in m.group(1):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx_sheet_rows(path: Path, sheet_name: str):
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join((t.text or "") for t in si.iter(f"{{{NS}}}t")) for si in root.findall(f"{{{NS}}}si")]
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rel = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relmap = {r.attrib["Id"]: r.attrib["Target"] for r in rel}
        target = None
        for s in wb.find(f"{{{NS}}}sheets"):
            if s.attrib.get("name") == sheet_name:
                target = relmap[s.attrib[f"{{{RNS}}}id"]]
                break
        if target is None:
            raise KeyError(sheet_name)
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        sheet = ET.fromstring(z.read(target))
        rows = sheet.find(f"{{{NS}}}sheetData").findall(f"{{{NS}}}row")

        def cell_value(c):
            typ = c.attrib.get("t")
            if typ == "inlineStr":
                inline = c.find(f"{{{NS}}}is")
                return "" if inline is None else "".join((t.text or "") for t in inline.iter(f"{{{NS}}}t"))
            v = c.find(f"{{{NS}}}v")
            if v is None:
                return ""
            raw = v.text or ""
            if typ == "s":
                try:
                    return shared[int(raw)]
                except Exception:
                    return raw
            return raw

        header_map = {col_index(c.attrib["r"]): cell_value(c) for c in rows[0].findall(f"{{{NS}}}c")}
        out = []
        for row in rows[1:]:
            d = {}
            for c in row.findall(f"{{{NS}}}c"):
                idx = col_index(c.attrib["r"])
                h = header_map.get(idx)
                if h:
                    d[h] = cell_value(c)
            out.append(d)
        return out


def norm_pid(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def add_sex(df: pd.DataFrame) -> pd.DataFrame:
    rows = read_xlsx_sheet_rows(XLSX, "HCM_PatientKeyOutcomes")
    sex_map = {norm_pid(r.get("HCM_PatientKeyOutcomes_PID")): str(r.get("Sex", "")).strip() for r in rows}
    out = df.copy()
    out["SHaRe_PID_key"] = out["SHaRe_PID"].map(norm_pid)
    out["sex"] = out["SHaRe_PID_key"].map(sex_map)
    sex_norm = out["sex"].astype(str).str.strip().str.lower()
    out["male"] = sex_norm.map({
        "male": 1.0, "m": 1.0, "1": 1.0, "1.0": 1.0,
        "female": 0.0, "f": 0.0, "0": 0.0, "0.0": 0.0,
    })
    return out


def fit_tv_cox(d: pd.DataFrame, covariates: list[str]):
    """Fit counting-process Cox PH with standard information-matrix covariance."""
    model = PHReg(
        endog=d["stop"].astype(float),
        exog=d[covariates].astype(float),
        status=d["event"].astype(int),
        entry=d["start"].astype(float),
        ties="breslow",
    )
    # Standard Cox information-matrix covariance. For a single terminal event,
    # splitting each subject into non-overlapping counting-process intervals is
    # the standard representation of a time-dependent covariate and does not
    # create recurrent-event pseudo-observations.
    res = model.fit()
    rows = []
    ci = np.asarray(res.conf_int())
    for i, name in enumerate(covariates):
        beta = float(res.params[i])
        rows.append({
            "term": name,
            "log_hr": beta,
            "se": float(res.bse[i]),
            "hr": math.exp(beta),
            "ci95_low": math.exp(float(ci[i, 0])),
            "ci95_high": math.exp(float(ci[i, 1])),
            "p": float(res.pvalues[i]),
        })
    return rows


def endpoint_subject_riskset(subj: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    cls = f"{endpoint}__baseline_class"
    et = f"{endpoint}__time_from_baseline_years"
    d = subj[subj[cls].isin(["none", "incident"])].copy()
    d = d[d["baseline_to_censor_years"].notna()].copy()
    d["endpoint_event"] = (d[cls] == "incident").astype(int)
    d["endpoint_time"] = np.where(
        d["endpoint_event"].eq(1),
        pd.to_numeric(d[et], errors="coerce"),
        pd.to_numeric(d["baseline_to_censor_years"], errors="coerce"),
    )
    d = d[np.isfinite(d["endpoint_time"]) & (d["endpoint_time"] > 0)].copy()
    return d


def prepare_longitudinal(subj: pd.DataFrame, zmu: float, zsd: float) -> pd.DataFrame:
    keep_subjects = set(subj["analysis_subject_key"].astype(str))
    header = pd.read_csv(LONG_CSV, nrows=0).columns.tolist()
    time_col = "ecg_datetime" if "ecg_datetime" in header else "ecg_time" if "ecg_time" in header else None
    if time_col is None:
        raise KeyError("Step 16 longitudinal CSV must contain ecg_datetime or ecg_time")
    required = [
        "analysis_subject_key", "ecg_id", time_col,
        "valid_for_latent_projection", "H_z_06"
    ]
    missing = [c for c in required if c not in header]
    if missing:
        raise KeyError(f"Step 16 longitudinal CSV is missing required columns: {missing}")
    long = pd.read_csv(LONG_CSV, usecols=required)
    if time_col != "ecg_datetime":
        long = long.rename(columns={time_col: "ecg_datetime"})
    long["analysis_subject_key"] = long["analysis_subject_key"].astype(str)
    long = long[long["analysis_subject_key"].isin(keep_subjects)].copy()
    valid = long["valid_for_latent_projection"]
    if valid.dtype != bool:
        valid = valid.astype(str).str.lower().isin(["true", "1", "yes"])
    long = long[valid].copy()
    long["H_z_06"] = pd.to_numeric(long["H_z_06"], errors="coerce")
    long["ecg_datetime"] = pd.to_datetime(long["ecg_datetime"], errors="coerce")
    long = long[long["H_z_06"].notna() & long["ecg_datetime"].notna()].copy()

    base_dates = subj.set_index(subj["analysis_subject_key"].astype(str))["baseline_ecg_date"]
    base_dates = pd.to_datetime(base_dates, errors="coerce")
    long["baseline_ecg_date"] = long["analysis_subject_key"].map(base_dates)
    long["ecg_time_y"] = (long["ecg_datetime"] - long["baseline_ecg_date"]).dt.total_seconds() / (86400.0 * DAYS_PER_YEAR)
    # Tolerate tiny negative discrepancies caused by date/time serialization only.
    long.loc[(long["ecg_time_y"] < 0) & (long["ecg_time_y"] > -1e-4), "ecg_time_y"] = 0.0
    long = long[long["ecg_time_y"] >= 0].copy()
    long["z06_per_sd"] = (long["H_z_06"] - zmu) / zsd
    long = long.sort_values(["analysis_subject_key", "ecg_time_y", "ecg_id"])
    # If two ECGs share exactly the same time, retain the last row deterministically.
    long = long.drop_duplicates(["analysis_subject_key", "ecg_time_y"], keep="last")
    return long


def build_counting_process(risk: pd.DataFrame, long: pd.DataFrame, lag_years: float = 0.0) -> pd.DataFrame:
    rows = []
    long_groups = {k: g.copy() for k, g in long.groupby("analysis_subject_key", sort=False)}

    for _, s in risk.iterrows():
        sid = str(s["analysis_subject_key"])
        if sid not in long_groups:
            continue
        end_time = float(s["endpoint_time"])
        event = int(s["endpoint_event"])
        if not np.isfinite(end_time) or end_time <= 0:
            continue
        g = long_groups[sid].copy()
        # Measurements after event/censor are never used.
        g = g[g["ecg_time_y"] < end_time + 1e-12].copy()
        if g.empty:
            continue

        # Exposure activation time = ECG time + lag.  At lag=0 baseline exposure
        # starts at time zero.  With lag>0, follow-up is left-truncated at lag.
        g["activation"] = g["ecg_time_y"] + lag_years
        g = g[g["activation"] < end_time - 1e-12].copy()
        if g.empty:
            continue

        # Must have a baseline measurement; Step24a guarantees this, verify here.
        # For lag sensitivity the baseline value becomes active at lag_years.
        start_follow = lag_years
        if end_time <= start_follow:
            continue

        # Keep activations at/after follow-up start and before endpoint/censor.
        g = g[g["activation"] >= start_follow - 1e-12].sort_values(["activation", "ecg_id"])
        g = g.drop_duplicates("activation", keep="last")
        # Explicit copies are required here because recent pandas/NumPy versions
        # may expose read-only views through Series.to_numpy().  We modify acts[0]
        # below to anchor the first interval exactly at the prespecified follow-up
        # start, so the array must be writable.
        acts = g["activation"].to_numpy(dtype=float, copy=True)
        zvals = g["z06_per_sd"].to_numpy(dtype=float, copy=True)
        rawz = g["H_z_06"].to_numpy(dtype=float, copy=True)
        ids = g["ecg_id"].to_numpy(copy=True)

        # Ensure first active exposure starts at start_follow.  Usually baseline ECG.
        if acts[0] > start_follow + 1e-8:
            continue
        acts[0] = start_follow

        for j in range(len(acts)):
            st = float(acts[j])
            sp = float(acts[j + 1]) if j + 1 < len(acts) else end_time
            sp = min(sp, end_time)
            if sp <= st + 1e-12:
                continue
            evt = int(event == 1 and abs(sp - end_time) < 1e-9)
            rows.append({
                "analysis_subject_key": sid,
                "SHaRe_PID": s["SHaRe_PID"],
                "start": st,
                "stop": sp,
                "event": evt,
                "z06_per_sd": float(zvals[j]),
                "H_z_06": float(rawz[j]),
                "source_ecg_id": ids[j],
                "baseline_age_10y": float(s["baseline_age_10y"]) if np.isfinite(s["baseline_age_10y"]) else np.nan,
                "male": float(s["male"]) if np.isfinite(s["male"]) else np.nan,
                "sex": s["sex"],
                "endpoint_time": end_time,
            })
    return pd.DataFrame(rows)


def summarize_intervals(d: pd.DataFrame):
    if d.empty:
        return dict(n_subjects=0, n_intervals=0, events=0, median_intervals_per_subject=np.nan,
                    median_followup_y=np.nan, subjects_with_z06_update=0)
    by = d.groupby("analysis_subject_key")
    n_intervals = by.size()
    # subjects with >1 source ECG contributing during risk time
    nupdates = by["source_ecg_id"].nunique()
    follow = by["stop"].max() - by["start"].min()
    return dict(
        n_subjects=int(d["analysis_subject_key"].nunique()),
        n_intervals=int(len(d)),
        events=int(d["event"].sum()),
        median_intervals_per_subject=float(n_intervals.median()),
        median_followup_y=float(follow.median()),
        subjects_with_z06_update=int((nupdates > 1).sum()),
    )


def main():
    subj = pd.read_csv(SUBJECT_CSV)
    subj["analysis_subject_key"] = subj["analysis_subject_key"].astype(str)
    subj = add_sex(subj)
    subj["baseline_age_10y"] = pd.to_numeric(subj["baseline_age"], errors="coerce") / 10.0

    # Same common baseline scaling as Step 25.
    baseline_z = pd.to_numeric(subj["baseline_z06"], errors="coerce")
    zmu = float(baseline_z.mean())
    zsd = float(baseline_z.std(ddof=1))
    if not np.isfinite(zsd) or zsd <= 0:
        raise RuntimeError(f"Invalid baseline H_z_06 SD for standardization: {zsd}")
    long = prepare_longitudinal(subj, zmu, zsd)

    results = []
    summaries = []
    audit_parts = []

    for endpoint, family in ENDPOINTS:
        risk = endpoint_subject_riskset(subj, endpoint)
        for lag_label, lag_y in [("no_lag", 0.0), ("lag_30d", LAG_YEARS)]:
            cp = build_counting_process(risk, long, lag_years=lag_y)
            if cp.empty:
                continue
            cp["endpoint"] = endpoint
            cp["family"] = family
            cp["lag"] = lag_label
            audit_parts.append(cp.copy())

            summary = summarize_intervals(cp)
            summaries.append({"endpoint": endpoint, "family": family, "lag": lag_label, **summary})

            for model_name, covs in [
                ("unadjusted", ["z06_per_sd"]),
                ("age_sex_adjusted", ["z06_per_sd", "baseline_age_10y", "male"]),
            ]:
                dm = cp.dropna(subset=covs).copy()
                # Remove subjects/intervals lost due to covariates; event count updated.
                terms = fit_tv_cox(dm, covs)
                zterm = next(x for x in terms if x["term"] == "z06_per_sd")
                results.append({
                    "endpoint": endpoint,
                    "family": family,
                    "lag": lag_label,
                    "model": model_name,
                    "n_subjects": int(dm["analysis_subject_key"].nunique()),
                    "n_intervals": int(len(dm)),
                    "events": int(dm["event"].sum()),
                    **zterm,
                })

    res = pd.DataFrame(results)
    # Contextual BH family across the 4 non-sparse endpoints, separately by lag/model.
    # Composite_VArrhythmia stays exploratory because of sparse events.
    res["q_context4"] = np.nan
    fdr_endpoints = ["Composite_HF", "Atrial_Fibrillation", "Unplanned_CV_Hospitalization", "Composite_Overall"]
    for lag_label in res["lag"].unique():
        for model_name in res["model"].unique():
            idx = res.index[(res["lag"] == lag_label) & (res["model"] == model_name) & res["endpoint"].isin(fdr_endpoints)]
            if len(idx):
                pvals = pd.to_numeric(res.loc[idx, "p"], errors="coerce")
                finite = np.isfinite(pvals.to_numpy(dtype=float))
                if finite.any():
                    q = multipletests(pvals.to_numpy(dtype=float)[finite], method="fdr_bh")[1]
                    res.loc[idx[finite], "q_context4"] = q

    res.to_csv(OUT / "z06_time_updated_cox_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "z06_time_updated_cox_risksets.csv", index=False)
    if audit_parts:
        pd.concat(audit_parts, ignore_index=True).to_csv(OUT / "z06_time_updated_cox_intervals.csv", index=False)

    # Compare directly with Step25 baseline results where available.
    baseline_path = RESULTS / "25_Z06_BaselineCox" / "z06_baseline_cox_results.csv"
    compare_rows = []
    if baseline_path.exists():
        b = pd.read_csv(baseline_path)
        for endpoint, family in ENDPOINTS:
            for model_name in ["unadjusted", "age_sex_adjusted"]:
                br = b[(b["endpoint"] == endpoint) & (b["model"] == model_name)]
                tr = res[(res["endpoint"] == endpoint) & (res["model"] == model_name) & (res["lag"] == "no_lag")]
                lr = res[(res["endpoint"] == endpoint) & (res["model"] == model_name) & (res["lag"] == "lag_30d")]
                if len(br) and len(tr):
                    compare_rows.append({
                        "endpoint": endpoint, "family": family, "model": model_name,
                        "baseline_hr": float(br.iloc[0]["hr"]), "baseline_p": float(br.iloc[0]["p"]),
                        "time_updated_hr": float(tr.iloc[0]["hr"]), "time_updated_p": float(tr.iloc[0]["p"]),
                        "time_updated_30d_lag_hr": float(lr.iloc[0]["hr"]) if len(lr) else np.nan,
                        "time_updated_30d_lag_p": float(lr.iloc[0]["p"]) if len(lr) else np.nan,
                    })
        pd.DataFrame(compare_rows).to_csv(OUT / "z06_baseline_vs_time_updated_comparison.csv", index=False)

    report = []
    report.append("Step 26: time-updated HCM z06 -> future events (counting-process Cox PH)\n")
    report.append(f"Common z06 scaling from exact-mapped baseline cohort: mean={zmu:.6f}, SD={zsd:.6f}.\n")
    report.append("SEs are standard Cox information-matrix estimates for non-overlapping counting-process intervals.\n")
    report.append("30-day lag sensitivity activates each ECG measurement 30 days after acquisition and left-truncates follow-up at day 30.\n\n")
    for _, r in res.iterrows():
        q = f", q_context4={r['q_context4']:.4g}" if np.isfinite(r["q_context4"]) else ""
        report.append(
            f"{r['endpoint']} | {r['lag']} | {r['model']}: subjects={int(r['n_subjects'])}, intervals={int(r['n_intervals'])}, "
            f"events={int(r['events'])}, HR per 1 SD current z06={r['hr']:.3f} "
            f"(95% CI {r['ci95_low']:.3f}-{r['ci95_high']:.3f}), p={r['p']:.4g}{q}\n"
        )
    (OUT / "z06_time_updated_cox_report.txt").write_text("".join(report), encoding="utf-8")
    print("".join(report))


if __name__ == "__main__":
    main()
