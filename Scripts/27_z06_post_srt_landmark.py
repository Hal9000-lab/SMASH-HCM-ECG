#!/usr/bin/env python3
"""
Step 27 -- Exploratory post-SRT landmark analysis of HCM z06 and subsequent events.

Purpose
-------
Assess whether the HCM-oriented lateral/frontal QRS-force coordinate H_z_06
measured after septal reduction therapy (SRT) is associated with subsequent
clinical events, i.e. whether there is evidence compatible with residual
post-treatment electrical risk.

Important design constraints
----------------------------
- exact ECG-to-SHaRe mappings only, inherited from Step 24 subject table
- SRT = earliest surgical myectomy or alcohol septal ablation, inherited from Step 24
- primary landmark window = first valid ECG 30--180 days after SRT
- sensitivity landmark window = first valid ECG 30--365 days after SRT
- endpoint events occurring before/on the post-SRT landmark are excluded from
  that endpoint's risk set
- positive endpoints with unavailable timing or events after recorded censoring
  are excluded
- no ECG after the landmark is used to define the landmark exposure
- H_z_06 is scaled using the SD of baseline H_z_06 in the full exact-mapped cohort,
  so HRs remain comparable with Steps 25--26
- because post-SRT event counts are small, analyses are explicitly exploratory;
  no multivariable adjustment and no multiplicity-based claim is made
- Cox models are fit only when >=3 incident events are available; otherwise the
  endpoint is reported as descriptive/not estimable

Secondary exploratory exposures
-------------------------------
Among subjects with a valid pre-SRT ECG within 365 days before SRT:
- pre-SRT z06
- peri-SRT delta z06 = post-SRT z06 - pre-SRT z06
These are not interpreted causally.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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



ROOT = infer_project_root()
RESULTS = ROOT / "Results"

# Canonical upstream dependencies.
SUBJECT_CSV = RESULTS / "24_Z06_EventRiskFeasibility" / "z06_event_feasibility_subject_level.csv"
LONG_CSV = RESULTS / "16_Longitudinal_DualManifold" / "longitudinal_dual_manifold.csv"

# Dedicated Step-27 output directory.
OUT = RESULTS / "27_Z06_PostSRT_Landmark"
OUT.mkdir(parents=True, exist_ok=True)

for label, path in (
    ("Step 24 event-feasibility subject table", SUBJECT_CSV),
    ("Step 16 longitudinal dual-manifold table", LONG_CSV),
):
    if not path.is_file():
        raise FileNotFoundError(f"Cannot locate {label}:\n{path}")

print(f"Project root  : {ROOT}")
print(f"Step 24 input : {SUBJECT_CSV}")
print(f"Step 16 input : {LONG_CSV}")
print(f"Step 27 output: {OUT}")

DAYS_PER_YEAR = 365.2425
ONE_DAY_Y = 1.0 / DAYS_PER_YEAR

ENDPOINTS = [
    "Composite_HF",
    "Composite_VArrhythmia",
    "Composite_Overall",
    "Atrial_Fibrillation",
    "Unplanned_CV_Hospitalization",
]

WINDOWS = [
    ("30_180d", 30.0, 180.0),
    ("30_365d", 30.0, 365.0),
]


def valid_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def prepare_longitudinal(subj: pd.DataFrame) -> pd.DataFrame:
    """Load valid Step-16 visits required for the post-SRT landmark analysis."""
    header = pd.read_csv(LONG_CSV, nrows=0).columns.tolist()
    time_col = "ecg_datetime" if "ecg_datetime" in header else "ecg_time" if "ecg_time" in header else None
    if time_col is None:
        raise KeyError("Step 16 longitudinal table contains neither 'ecg_datetime' nor 'ecg_time'.")

    required = {
        "analysis_subject_key", "ecg_id", "birthdate",
        "valid_for_latent_projection", "H_z_06", time_col,
    }
    missing = sorted(required.difference(header))
    if missing:
        raise KeyError(f"Missing required Step 16 columns: {missing}")

    long = pd.read_csv(LONG_CSV, usecols=list(required))
    if time_col != "ecg_datetime":
        long = long.rename(columns={time_col: "ecg_datetime"})

    long["analysis_subject_key"] = long["analysis_subject_key"].astype(str).str.strip()
    keep = set(subj["analysis_subject_key"].astype(str).str.strip())
    long = long[long["analysis_subject_key"].isin(keep)].copy()
    long = long[valid_bool(long["valid_for_latent_projection"])].copy()
    long["H_z_06"] = pd.to_numeric(long["H_z_06"], errors="coerce")
    long["ecg_datetime"] = pd.to_datetime(long["ecg_datetime"], errors="coerce")
    long["birthdate"] = pd.to_datetime(long["birthdate"], errors="coerce")
    long = long[
        long["H_z_06"].notna()
        & long["ecg_datetime"].notna()
        & long["birthdate"].notna()
    ].copy()
    long["ecg_age"] = (
        long["ecg_datetime"] - long["birthdate"]
    ).dt.total_seconds() / (86400.0 * DAYS_PER_YEAR)
    long = long.sort_values(["analysis_subject_key", "ecg_datetime", "ecg_id"])
    return long


def build_landmarks(subj: pd.DataFrame, long: pd.DataFrame, min_days: float, max_days: float) -> pd.DataFrame:
    rows = []
    groups = {sid: g.copy() for sid, g in long.groupby("analysis_subject_key", sort=False)}
    for _, s in subj.iterrows():
        sid = str(s["analysis_subject_key"])
        srt_age = pd.to_numeric(pd.Series([s.get("srt_age")]), errors="coerce").iloc[0]
        if not np.isfinite(srt_age) or sid not in groups:
            continue
        g = groups[sid].copy()
        g["days_from_srt"] = (g["ecg_age"] - float(srt_age)) * DAYS_PER_YEAR
        post = g[(g["days_from_srt"] >= min_days - 1e-9) & (g["days_from_srt"] <= max_days + 1e-9)].copy()
        if post.empty:
            continue
        post = post.sort_values(["days_from_srt", "ecg_datetime", "ecg_id"]).iloc[0]

        pre = g[(g["days_from_srt"] < -1.0) & (g["days_from_srt"] >= -365.0 - 1e-9)].copy()
        if not pre.empty:
            pre = pre.sort_values(["days_from_srt", "ecg_datetime", "ecg_id"], ascending=[False, False, False]).iloc[0]
            pre_age = float(pre["ecg_age"])
            pre_z = float(pre["H_z_06"])
            pre_id = pre["ecg_id"]
        else:
            pre_age = np.nan
            pre_z = np.nan
            pre_id = np.nan

        row = s.to_dict()
        row.update({
            "landmark_ecg_id": post["ecg_id"],
            "landmark_ecg_date": post["ecg_datetime"],
            "landmark_age": float(post["ecg_age"]),
            "days_srt_to_landmark": float(post["days_from_srt"]),
            "post_srt_z06_recomputed": float(post["H_z_06"]),
            "pre_srt_ecg_id_recomputed": pre_id,
            "pre_srt_ecg_age_recomputed": pre_age,
            "pre_srt_z06_recomputed": pre_z,
            "delta_z06_recomputed": float(post["H_z_06"]) - pre_z if np.isfinite(pre_z) else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def endpoint_riskset(land: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    rows = []
    for _, s in land.iterrows():
        lm_age = float(s["landmark_age"])
        censor_age = pd.to_numeric(pd.Series([s.get("censor_age")]), errors="coerce").iloc[0]
        present_raw = s.get(f"{endpoint}__present")
        present = str(present_raw).strip().lower() in ["true", "1", "1.0", "yes"] if not isinstance(present_raw, (bool, np.bool_)) else bool(present_raw)
        event_age = pd.to_numeric(pd.Series([s.get(f"{endpoint}__age")]), errors="coerce").iloc[0]

        status = None
        event = 0
        follow = np.nan
        if not np.isfinite(censor_age) or censor_age <= lm_age + ONE_DAY_Y:
            status = "no_positive_followup_after_landmark"
        elif present and not np.isfinite(event_age):
            status = "positive_missing_event_age"
        elif present and event_age <= lm_age + ONE_DAY_Y:
            status = "event_before_or_at_landmark"
        elif present and event_age > censor_age + ONE_DAY_Y:
            status = "event_after_recorded_censor"
        elif present:
            status = "incident"
            event = 1
            follow = float(event_age - lm_age)
        else:
            status = "none"
            follow = float(censor_age - lm_age)

        r = s.to_dict()
        r.update({
            "endpoint": endpoint,
            "riskset_status": status,
            "event": event,
            "followup_years": follow,
        })
        rows.append(r)
    d = pd.DataFrame(rows)
    d = d[d["riskset_status"].isin(["none", "incident"])].copy()
    d = d[np.isfinite(pd.to_numeric(d["followup_years"], errors="coerce")) & (pd.to_numeric(d["followup_years"], errors="coerce") > 0)].copy()
    return d


def fit_cox(d: pd.DataFrame, exposure: str):
    x = pd.to_numeric(d[exposure], errors="coerce")
    y = pd.to_numeric(d["followup_years"], errors="coerce")
    e = pd.to_numeric(d["event"], errors="coerce")
    use = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (y > 0)
    dd = d.loc[use].copy()
    if len(dd) == 0:
        return None
    nev = int(dd["event"].sum())
    if nev < 3:
        return {
            "n": len(dd), "events": nev, "status": "not_estimated_lt3_events",
            "log_hr": np.nan, "se": np.nan, "hr": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan, "p": np.nan,
        }
    try:
        mod = PHReg(
            dd["followup_years"].to_numpy(float),
            dd[[exposure]].to_numpy(float),
            status=dd["event"].to_numpy(int),
            ties="breslow",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            res = mod.fit(disp=0)
        converged_cleanly = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        beta = float(res.params[0])
        ci = np.asarray(res.conf_int())[0]
        return {
            "n": len(dd), "events": nev,
            "status": "estimated" if converged_cleanly else "estimated_with_convergence_warning",
            "log_hr": beta, "se": float(res.bse[0]), "hr": math.exp(beta),
            "ci95_low": math.exp(float(ci[0])), "ci95_high": math.exp(float(ci[1])),
            "p": float(res.pvalues[0]),
        }
    except Exception as exc:
        return {
            "n": len(dd), "events": nev, "status": f"fit_failed:{type(exc).__name__}",
            "log_hr": np.nan, "se": np.nan, "hr": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan, "p": np.nan,
        }


def medians_by_event(d: pd.DataFrame, exposure: str):
    x = pd.to_numeric(d[exposure], errors="coerce")
    ev = d.loc[d["event"].eq(1) & x.notna(), exposure].astype(float)
    ne = d.loc[d["event"].eq(0) & x.notna(), exposure].astype(float)
    return (
        float(ev.median()) if len(ev) else np.nan,
        float(ne.median()) if len(ne) else np.nan,
        int(len(ev)), int(len(ne))
    )


def main():
    subj = pd.read_csv(SUBJECT_CSV)
    required_subject_cols = {"analysis_subject_key", "srt_age", "baseline_z06", "censor_age"}
    for endpoint in ENDPOINTS:
        required_subject_cols.update({f"{endpoint}__present", f"{endpoint}__age"})
    missing_subject = sorted(required_subject_cols.difference(subj.columns))
    if missing_subject:
        raise KeyError(f"Missing required Step 24 columns: {missing_subject}")
    subj["analysis_subject_key"] = subj["analysis_subject_key"].astype(str).str.strip()
    subj = subj[pd.to_numeric(subj["srt_age"], errors="coerce").notna()].copy()

    # Preserve scaling used by Steps 25--26: SD from all exact-mapped baseline z06.
    allsub = pd.read_csv(SUBJECT_CSV)
    zbase = pd.to_numeric(allsub["baseline_z06"], errors="coerce")
    zmu = float(zbase.mean())
    zsd = float(zbase.std(ddof=1))

    long = prepare_longitudinal(subj)
    all_results = []
    riskset_summaries = []
    audit = []
    landmark_summaries = []

    for wlabel, mind, maxd in WINDOWS:
        land = build_landmarks(subj, long, mind, maxd)
        if land.empty:
            continue
        land["post_z06_per_global_sd"] = (land["post_srt_z06_recomputed"] - zmu) / zsd
        land["pre_z06_per_global_sd"] = (land["pre_srt_z06_recomputed"] - zmu) / zsd
        land["delta_z06_per_global_sd"] = land["delta_z06_recomputed"] / zsd
        landmark_summaries.append({
            "window": wlabel,
            "n_landmarks": int(len(land)),
            "n_with_pre_srt_ecg_365d": int(land["pre_srt_z06_recomputed"].notna().sum()),
            "median_days_srt_to_landmark": float(land["days_srt_to_landmark"].median()),
            "median_post_z06": float(land["post_srt_z06_recomputed"].median()),
            "median_delta_z06": float(land["delta_z06_recomputed"].dropna().median()) if land["delta_z06_recomputed"].notna().any() else np.nan,
        })

        for endpoint in ENDPOINTS:
            risk = endpoint_riskset(land, endpoint)
            if risk.empty:
                continue
            risk["window"] = wlabel
            audit.append(risk.copy())
            riskset_summaries.append({
                "window": wlabel,
                "endpoint": endpoint,
                "n_at_risk": int(len(risk)),
                "events": int(risk["event"].sum()),
                "total_person_years": float(risk["followup_years"].sum()),
                "median_followup_years": float(risk["followup_years"].median()),
                "n_with_pre_srt_ecg": int(risk["pre_srt_z06_recomputed"].notna().sum()),
            })

            for exposure, label in [
                ("post_z06_per_global_sd", "post_srt_z06"),
                ("pre_z06_per_global_sd", "pre_srt_z06"),
                ("delta_z06_per_global_sd", "peri_srt_delta_z06"),
            ]:
                fit = fit_cox(risk, exposure)
                if fit is None:
                    continue
                med_e, med_ne, n_e, n_ne = medians_by_event(risk, exposure)
                all_results.append({
                    "window": wlabel,
                    "endpoint": endpoint,
                    "exposure": label,
                    "scale": "per 1 SD of baseline z06 in exact-mapped cohort",
                    **fit,
                    "median_exposure_events": med_e,
                    "median_exposure_nonevents": med_ne,
                    "n_events_with_exposure": n_e,
                    "n_nonevents_with_exposure": n_ne,
                })

    res = pd.DataFrame(all_results)
    rsum = pd.DataFrame(riskset_summaries)
    lsum = pd.DataFrame(landmark_summaries)
    res.to_csv(OUT / "z06_post_srt_landmark_cox_results.csv", index=False)
    rsum.to_csv(OUT / "z06_post_srt_landmark_risksets.csv", index=False)
    lsum.to_csv(OUT / "z06_post_srt_landmark_summary.csv", index=False)
    if audit:
        pd.concat(audit, ignore_index=True).to_csv(OUT / "z06_post_srt_landmark_subject_rows.csv", index=False)

    # Concise report.
    lines = []
    lines.append("SMASH-HCM z06 post-SRT landmark analysis (exploratory)\n")
    lines.append(f"Global baseline z06 scaling: mean={zmu:.6f}, SD={zsd:.6f}\n")
    for _, lr in lsum.iterrows():
        lines.append(
            f"Window {lr['window']}: {int(lr['n_landmarks'])} post-SRT landmarks, "
            f"{int(lr['n_with_pre_srt_ecg_365d'])} with pre-SRT ECG; "
            f"median SRT-to-landmark {lr['median_days_srt_to_landmark']:.1f} days.\n"
        )
    lines.append("\nPrimary residual-risk exposure = post-SRT z06. Cox is not fit when <3 events.\n")
    pri = res[(res["exposure"] == "post_srt_z06") & (res["endpoint"].isin(["Composite_HF", "Composite_VArrhythmia", "Composite_Overall"]))]
    for _, rr in pri.iterrows():
        if str(rr["status"]).startswith("estimated"):
            lines.append(
                f"{rr['window']} {rr['endpoint']}: n={int(rr['n'])}, events={int(rr['events'])}, "
                f"HR/SD={rr['hr']:.3f} (95% CI {rr['ci95_low']:.3f}-{rr['ci95_high']:.3f}), "
                f"p={rr['p']:.4g}, status={rr['status']}.\n"
            )
        else:
            lines.append(
                f"{rr['window']} {rr['endpoint']}: n={int(rr['n'])}, events={int(rr['events'])}; "
                f"{rr['status']}.\n"
            )
    lines.append("\nInterpretation constraint: event counts are too small for adjusted post-SRT prognostic modelling. Any estimated HR is exploratory and cannot establish residual risk independently of treatment selection or post-treatment clinical state.\n")
    (OUT / "z06_post_srt_landmark_report.txt").write_text("".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nLandmark summary:\n", lsum.to_string(index=False))
    print("\nRisksets:\n", rsum.to_string(index=False))
    print("\nCox results:\n", res.to_string(index=False))


if __name__ == "__main__":
    main()
