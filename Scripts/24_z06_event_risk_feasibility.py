#!/usr/bin/env python3
"""
Step 24a -- Feasibility analysis for future-event association of HCM z06.

Purpose
-------
Before fitting Cox or joint longitudinal-survival models, quantify whether the
available SHaRe event data provide enough incident events after an ECG landmark,
and enough post-SRT events for a residual-risk analysis.

The script deliberately does NOT fit prognostic models. It produces an auditable
subject-level table and summary counts that can be used to choose endpoint(s) for
subsequent survival analysis.

Primary analysis population
---------------------------
- exact ECG-to-SHaRe mappings only (mapping_status == 'matched')
- at least one valid longitudinal ECG with non-missing H_z_06
- baseline = first such valid ECG
- incident events must occur strictly after baseline (1-day tolerance)
- subjects with the endpoint before/at baseline are excluded from that endpoint's
  at-risk population

Post-SRT feasibility
--------------------
- SRT = earliest of surgical myectomy or alcohol septal ablation
- post-SRT landmark ECG = first valid z06 ECG 30--180 days after SRT
- optional pre-SRT ECG for change analysis = closest valid z06 ECG within
  365 days before SRT
- future post-SRT events must occur after the post-SRT landmark

Inputs
------
Data/SHaReData-FLO.xls
  sheets: HCM_Events, HCM_PatientKeyOutcomes
Results/18_Genotype_Longitudinal/genotype_subject_cohort.csv
Results/16_Longitudinal_DualManifold/longitudinal_dual_manifold.csv

Outputs
-------
Results/24_Z06_EventRiskFeasibility/
    z06_event_feasibility_summary.csv
    z06_post_srt_feasibility_summary.csv
    z06_event_feasibility_subject_level.csv
    z06_event_feasibility_report.txt

No model is trained and no latent representation is modified.

Run from project root:
    python Scripts\\24_z06_event_risk_feasibility.py

Dependencies:
    numpy-free standard library processing plus pandas and xlrd for the legacy .xls source.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# -----------------------------------------------------------------------------
# Project-root aware paths.
# -----------------------------------------------------------------------------
def infer_project_root() -> Path:
    """Infer the SMASH-HCM project root from the script location or CWD."""
    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve(), here.parent, *here.parents]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "Data").exists() and (candidate / "Results").exists():
            return candidate
    # Standard layout: <project>/Scripts/24_...py
    return here.parent.parent


def resolve_share_file(project_root: Path) -> Path:
    """Locate the exact legacy SHaReData-FLO.xls workbook within the project."""
    preferred = [
        project_root / "Data" / "SHaReData-FLO.xlsx",
        project_root / "SHaReData-FLO.xlsx",
        project_root / "Scripts" / "SHaReData-FLO.xlsx",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    # Last-resort project-local search, excluding the virtual environment.
    matches = []
    for candidate in project_root.rglob("SHaReData-FLO.xls"):
        try:
            rel_parts = candidate.relative_to(project_root).parts
        except ValueError:
            rel_parts = candidate.parts
        if any(part.lower() in {"venv", ".venv"} for part in rel_parts):
            continue
        matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        listed = "\n  ".join(str(x) for x in matches)
        raise RuntimeError(
            "Multiple SHaReData-FLO.xls files found; keep one canonical copy or "
            f"set SHARE_FILE explicitly:\n  {listed}"
        )

    checked = "\n  ".join(str(x) for x in preferred)
    raise FileNotFoundError(
        "Could not find the exact source workbook 'SHaReData-FLO.xls'. "
        f"Checked:\n  {checked}"
    )


PROJECT_ROOT = infer_project_root()
SHARE_FILE = resolve_share_file(PROJECT_ROOT)
CROSS = PROJECT_ROOT / "Results" / "18_Genotype_Longitudinal" / "genotype_subject_cohort.csv"
LONG = PROJECT_ROOT / "Results" / "16_Longitudinal_DualManifold" / "longitudinal_dual_manifold.csv"
OUT = PROJECT_ROOT / "Results" / "24_Z06_EventRiskFeasibility"
OUT.mkdir(parents=True, exist_ok=True)

# Main electrical coordinate of interest.
Z06_COL = "H_z_06"

# Timing tolerances / landmark windows.
DAYS_PER_YEAR = 365.2425
ONE_DAY_YEARS = 1.0 / DAYS_PER_YEAR
POST_SRT_MIN_DAYS = 30.0
POST_SRT_MAX_DAYS = 180.0
PRE_SRT_MAX_DAYS = 365.0

# Candidate endpoints.  Keep the native SHaRe composites because they are the
# cleanest first-pass feasibility targets; individual endpoints are included to
# show where event counts may be too sparse for multivariable modelling.
ENDPOINTS = [
    {
        "name": "Composite_HF",
        "source": "HCM_PatientKeyOutcomes",
        "flag": "Composite_HF",
        "age": "Composite_HF_Age",
        "category": "primary_candidate",
    },
    {
        "name": "Composite_VArrhythmia",
        "source": "HCM_PatientKeyOutcomes",
        "flag": "Composite_VArrhythmia",
        "age": "Composite_VArrhythmia_Age",
        "category": "primary_candidate",
    },
    {
        "name": "Composite_Overall",
        "source": "HCM_PatientKeyOutcomes",
        "flag": "Composite_Overall",
        "age": "Composite_Overall_Age",
        "category": "primary_candidate",
    },
    {
        "name": "Atrial_Fibrillation",
        "source": "HCM_PatientKeyOutcomes",
        "flag": "IsAtrial_Fibrillation",
        "age": "Atrial_Fibrillation_Age",
        "category": "secondary",
    },
    {
        "name": "Cardiac_Arrest",
        "source": "HCM_PatientKeyOutcomes",
        "flag": "IsCardiac_Arrest",
        "age": "Cardiac_Arrest_Age",
        "category": "secondary",
    },
    {
        "name": "ICD_Appropriate_Firing",
        "source": "HCM_PatientKeyOutcomes",
        "flag": "IsICD_Appropriate_Firing",
        "age": "ICD_Appropriate_Firing_Age",
        "category": "secondary",
    },
    {
        "name": "Stroke",
        "source": "HCM_PatientKeyOutcomes",
        "flag": "IsStroke",
        "age": "Stroke_Age",
        "category": "secondary",
    },
    {
        "name": "Unplanned_CV_Hospitalization",
        "source": "HCM_Events",
        "flag": "Unplanned_Cardiovascular_Hospitalization",
        "age": "Unplanned_Cardiovascular_Hospitalization_Age",
        "category": "secondary",
    },
]


# XLSX XML namespaces.
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def safe_float(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def yes(x) -> bool:
    return str(x or "").strip().lower() in {
        "yes", "true", "1", "1.0", "y", "t", "positive", "present"
    }


def parse_datetime(x):
    s = str(x or "").strip()
    if not s:
        return None
    # The longitudinal files used in SMASH-HCM are ISO-like.  Strip trailing Z.
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        # Conservative fallback for common date-only forms.
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(s[:10], fmt)
            except ValueError:
                pass
    return None


def normalize_pid(x):
    s = str(x or "").strip()
    if not s:
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def median_or_blank(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(vals) if vals else ""


def pct(num, den):
    return (100.0 * num / den) if den else ""


def col_index(cell_ref: str) -> int:
    m = re.match(r"([A-Z]+)", cell_ref)
    if not m:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    n = 0
    for ch in m.group(1):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_excel_sheet_rows(path: Path, sheet_name: str):
    """Read one Excel sheet as dictionaries.

    The project source is a legacy BIFF ``.xls`` workbook, which requires
    pandas + xlrd. A lightweight ZIP/XML fallback is retained for ``.xlsx``
    files so the script remains usable if the source format changes later.
    """
    if path.suffix.lower() == ".xls":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Reading Data/SHaReData-FLO.xls requires pandas."
            ) from exc
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, engine="xlrd", dtype=object)
        except ImportError as exc:
            raise RuntimeError(
                "Reading legacy .xls files requires xlrd. Install it with: "
                "python -m pip install xlrd"
            ) from exc
        # Preserve the original workbook field names and represent missing cells
        # as empty strings, matching the previous ZIP/XML reader semantics.
        rows = []
        for _, row in df.iterrows():
            rows.append({
                str(col): ("" if pd.isna(row[col]) else row[col])
                for col in df.columns
            })
        return rows

    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Unsupported Excel format: {path.suffix}")

    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = [
                "".join((t.text or "") for t in si.iter(f"{{{NS}}}t"))
                for si in root.findall(f"{{{NS}}}si")
            ]

        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rel = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relmap = {r.attrib["Id"]: r.attrib["Target"] for r in rel}

        target = None
        for s in wb.find(f"{{{NS}}}sheets"):
            if s.attrib.get("name") == sheet_name:
                target = relmap[s.attrib[f"{{{RNS}}}id"]]
                break
        if target is None:
            raise KeyError(f"Sheet not found: {sheet_name}")

        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        sheet = ET.fromstring(z.read(target))
        sheet_data = sheet.find(f"{{{NS}}}sheetData")
        rows = sheet_data.findall(f"{{{NS}}}row") if sheet_data is not None else []
        if not rows:
            return []

        def cell_value(c):
            typ = c.attrib.get("t")
            if typ == "inlineStr":
                inline = c.find(f"{{{NS}}}is")
                if inline is None:
                    return ""
                return "".join((t.text or "") for t in inline.iter(f"{{{NS}}}t"))
            v = c.find(f"{{{NS}}}v")
            if v is None:
                return ""
            raw = v.text or ""
            if typ == "s":
                try:
                    return shared[int(raw)]
                except (ValueError, IndexError):
                    return raw
            return raw

        header_map = {
            col_index(c.attrib["r"]): cell_value(c)
            for c in rows[0].findall(f"{{{NS}}}c")
        }

        output = []
        for row in rows[1:]:
            d = {}
            for c in row.findall(f"{{{NS}}}c"):
                idx = col_index(c.attrib["r"])
                h = header_map.get(idx)
                if h:
                    d[h] = cell_value(c)
            output.append(d)
        return output


def load_exact_crosswalk():
    """Return exact mapping analysis_subject_key -> SHaRe PID."""
    s2p = {}
    p2s = {}
    with CROSS.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("mapping_status", "")).strip() != "matched":
                continue
            s = str(r.get("analysis_subject_key", "")).strip()
            p = normalize_pid(r.get("SHaRe_PID"))
            if not s or not p:
                continue
            s2p[s] = p
            p2s[p] = s
    return s2p, p2s


def load_longitudinal_z06(s2p):
    """Load all valid exact-mapped visits with a finite H_z_06."""
    by_subject = defaultdict(list)
    with LONG.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("valid_for_latent_projection", "")).strip().lower() not in {
                "true", "1", "yes"
            }:
                continue
            s = str(r.get("analysis_subject_key", "")).strip()
            if s not in s2p:
                continue
            z06 = safe_float(r.get(Z06_COL))
            ecg_dt = parse_datetime(r.get("ecg_time") or r.get("ecg_datetime"))
            birth_dt = parse_datetime(r.get("birthdate"))
            if z06 is None or ecg_dt is None or birth_dt is None:
                continue
            age = (ecg_dt - birth_dt).total_seconds() / (DAYS_PER_YEAR * 86400.0)
            by_subject[s].append(
                {
                    "analysis_subject_key": s,
                    "SHaRe_PID": s2p[s],
                    "ecg_id": str(r.get("ecg_id", "")),
                    "ecg_time": ecg_dt,
                    "ecg_age": age,
                    "z06": z06,
                }
            )
    for s in by_subject:
        by_subject[s].sort(key=lambda q: q["ecg_time"])
    return by_subject


def keyed_rows(rows, pid_col):
    out = {}
    for r in rows:
        p = normalize_pid(r.get(pid_col))
        if p:
            out[p] = r
    return out


def event_value(spec, pid, pko_by_pid, events_by_pid):
    src = pko_by_pid if spec["source"] == "HCM_PatientKeyOutcomes" else events_by_pid
    r = src.get(pid, {})
    age = safe_float(r.get(spec["age"]))
    # Treat a recorded event age as event evidence even if a binary flag is missing.
    present = yes(r.get(spec["flag"])) or age is not None
    return present, age


def srt_info(pid, events_by_pid):
    r = events_by_pid.get(pid, {})
    candidates = []
    for flag_col, age_col, label in (
        ("Myectomy", "Myectomy_Age", "myectomy"),
        ("EtOH_Ablation", "EtOH_Ablation_Age", "etoh_ablation"),
    ):
        a = safe_float(r.get(age_col))
        pos = yes(r.get(flag_col)) or a is not None
        if pos and a is not None:
            candidates.append((a, label))
    if not candidates:
        return False, None, ""
    age, label = min(candidates, key=lambda x: x[0])
    return True, age, label


def followup_age(pid, pko_by_pid, events_by_pid, last_ecg_age):
    """Prefer canonical Last_FollowUp_Age; then LastEncounter_Age; then last ECG."""
    e = events_by_pid.get(pid, {})
    p = pko_by_pid.get(pid, {})
    for value, source in (
        (safe_float(e.get("Last_FollowUp_Age")), "HCM_Events.Last_FollowUp_Age"),
        (safe_float(p.get("LastEncounter_Age")), "HCM_PatientKeyOutcomes.LastEncounter_Age"),
        (safe_float(p.get("LastVisit_Age")), "HCM_PatientKeyOutcomes.LastVisit_Age"),
        (last_ecg_age, "last_valid_z06_ECG"),
    ):
        if value is not None:
            return value, source
    return None, ""


def classify_event(event_present, event_age, landmark_age, censor_age):
    """
    Return classification relative to a landmark.

    prevalent: event known before/at landmark (1-day tolerance)
    incident: event after landmark and on/before censor age
    post_censor: event age lies after available follow-up
    positive_missing_age: positive flag but missing timing
    none: no event recorded
    """
    if not event_present:
        return "none"
    if event_age is None:
        return "positive_missing_age"
    if event_age <= landmark_age + ONE_DAY_YEARS:
        return "prevalent_or_same_day"
    if censor_age is not None and event_age > censor_age + ONE_DAY_YEARS:
        return "post_censor"
    return "incident"


def choose_srt_landmarks(visits, srt_age):
    if srt_age is None:
        return None, None

    pre = []
    post = []
    for v in visits:
        delta_days = (v["ecg_age"] - srt_age) * DAYS_PER_YEAR
        if -PRE_SRT_MAX_DAYS <= delta_days < -1.0:
            pre.append((abs(delta_days), v))
        if POST_SRT_MIN_DAYS <= delta_days <= POST_SRT_MAX_DAYS:
            post.append((delta_days, v))

    pre_visit = min(pre, key=lambda x: x[0])[1] if pre else None
    # First stable post-SRT ECG in the prespecified window.
    post_visit = min(post, key=lambda x: x[0])[1] if post else None
    return pre_visit, post_visit


def main():
    print("Project root :", PROJECT_ROOT)
    print("SHaRe source :", SHARE_FILE)
    print("Crosswalk    :", CROSS)
    print("Longitudinal :", LONG)
    print("Output dir   :", OUT)
    print()

    for p in (SHARE_FILE, CROSS, LONG):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    s2p, _ = load_exact_crosswalk()
    visits_by_subject = load_longitudinal_z06(s2p)

    pko_rows = read_excel_sheet_rows(SHARE_FILE, "HCM_PatientKeyOutcomes")
    event_rows = read_excel_sheet_rows(SHARE_FILE, "HCM_Events")
    pko_by_pid = keyed_rows(pko_rows, "HCM_PatientKeyOutcomes_PID")
    events_by_pid = keyed_rows(event_rows, "HCM_Events_PID")

    subject_rows = []
    for s, visits in sorted(visits_by_subject.items()):
        if not visits:
            continue
        pid = s2p[s]
        baseline = visits[0]
        last_ecg = visits[-1]
        censor_age, censor_source = followup_age(
            pid, pko_by_pid, events_by_pid, last_ecg["ecg_age"]
        )
        # Ensure censoring cannot precede the ECG landmark because of inconsistent
        # source data; such records remain auditable rather than silently dropped.
        followup_inconsistent = (
            censor_age is not None and censor_age + ONE_DAY_YEARS < baseline["ecg_age"]
        )

        has_srt, srt_age, srt_type = srt_info(pid, events_by_pid)
        pre_srt, post_srt = choose_srt_landmarks(visits, srt_age)

        row = {
            "analysis_subject_key": s,
            "SHaRe_PID": pid,
            "n_valid_z06_ecgs": len(visits),
            "baseline_ecg_id": baseline["ecg_id"],
            "baseline_ecg_date": baseline["ecg_time"].isoformat(),
            "baseline_age": baseline["ecg_age"],
            "baseline_z06": baseline["z06"],
            "last_valid_z06_ecg_age": last_ecg["ecg_age"],
            "censor_age": censor_age if censor_age is not None else "",
            "censor_source": censor_source,
            "baseline_to_censor_years": (
                max(0.0, censor_age - baseline["ecg_age"])
                if censor_age is not None and not followup_inconsistent
                else ""
            ),
            "followup_age_inconsistent": followup_inconsistent,
            "has_srt": has_srt,
            "srt_age": srt_age if srt_age is not None else "",
            "srt_type": srt_type,
            "has_pre_srt_ecg_365d": pre_srt is not None,
            "pre_srt_ecg_age": pre_srt["ecg_age"] if pre_srt else "",
            "pre_srt_z06": pre_srt["z06"] if pre_srt else "",
            "has_post_srt_landmark_30_180d": post_srt is not None,
            "post_srt_ecg_age": post_srt["ecg_age"] if post_srt else "",
            "post_srt_z06": post_srt["z06"] if post_srt else "",
            "delta_z06_srt": (
                post_srt["z06"] - pre_srt["z06"] if pre_srt and post_srt else ""
            ),
        }

        for spec in ENDPOINTS:
            present, age = event_value(spec, pid, pko_by_pid, events_by_pid)
            cls = classify_event(present, age, baseline["ecg_age"], censor_age)
            prefix = spec["name"]
            row[f"{prefix}__present"] = present
            row[f"{prefix}__age"] = age if age is not None else ""
            row[f"{prefix}__baseline_class"] = cls
            row[f"{prefix}__incident_from_baseline"] = cls == "incident"
            row[f"{prefix}__time_from_baseline_years"] = (
                age - baseline["ecg_age"] if cls == "incident" else ""
            )

            if post_srt is not None:
                post_cls = classify_event(present, age, post_srt["ecg_age"], censor_age)
            else:
                post_cls = "no_post_srt_landmark"
            row[f"{prefix}__post_srt_class"] = post_cls
            row[f"{prefix}__incident_post_srt"] = post_cls == "incident"
            row[f"{prefix}__time_from_post_srt_years"] = (
                age - post_srt["ecg_age"]
                if post_srt is not None and post_cls == "incident"
                else ""
            )

        subject_rows.append(row)

    # ------------------------------------------------------------------
    # Baseline feasibility summary.
    # ------------------------------------------------------------------
    baseline_summary = []
    for spec in ENDPOINTS:
        name = spec["name"]
        at_risk = []
        incident = []
        prevalent = 0
        missing_age = 0
        post_censor = 0
        no_event = 0
        person_years = []

        for r in subject_rows:
            cls = r[f"{name}__baseline_class"]
            if cls == "prevalent_or_same_day":
                prevalent += 1
                continue
            if cls == "positive_missing_age":
                missing_age += 1
                continue
            if cls == "post_censor":
                post_censor += 1
                # Timing conflicts with the recorded censor date: exclude from the
                # clean at-risk denominator but report explicitly.
                continue

            censor_age = safe_float(r["censor_age"])
            if censor_age is None or r["followup_age_inconsistent"]:
                continue

            at_risk.append(r)
            if cls == "incident":
                incident.append(r)
                end_age = safe_float(r[f"{name}__age"])
            else:
                no_event += 1
                end_age = censor_age
            if end_age is not None:
                person_years.append(max(0.0, end_age - float(r["baseline_age"])))

        n = len(at_risk)
        ne = len(incident)
        py = sum(person_years)
        followups = [
            min(
                safe_float(r[f"{name}__age"]) if r[f"{name}__baseline_class"] == "incident" else safe_float(r["censor_age"]),
                safe_float(r["censor_age"]),
            ) - float(r["baseline_age"])
            for r in at_risk
            if safe_float(r["censor_age"]) is not None
            and (
                r[f"{name}__baseline_class"] != "incident"
                or safe_float(r[f"{name}__age"]) is not None
            )
        ]
        baseline_summary.append(
            {
                "endpoint": name,
                "category": spec["category"],
                "source": spec["source"],
                "n_z06_baseline_subjects": len(subject_rows),
                "n_prevalent_or_same_day_excluded": prevalent,
                "n_positive_missing_event_age_excluded": missing_age,
                "n_event_after_recorded_censor_excluded": post_censor,
                "n_at_risk": n,
                "n_incident_events": ne,
                "incident_event_percent": pct(ne, n),
                "total_person_years": py,
                "events_per_100_person_years": (100.0 * ne / py) if py > 0 else "",
                "median_observed_time_years": median_or_blank(followups),
                "n_no_event": no_event,
            }
        )

    # ------------------------------------------------------------------
    # Post-SRT landmark feasibility summary.
    # ------------------------------------------------------------------
    post_srt_subjects = [r for r in subject_rows if r["has_post_srt_landmark_30_180d"]]
    post_srt_with_pre = [r for r in post_srt_subjects if r["has_pre_srt_ecg_365d"]]
    post_srt_summary = []
    for spec in ENDPOINTS:
        name = spec["name"]
        clean = []
        incident = []
        pre_landmark = 0
        missing_age = 0
        post_censor = 0
        person_years = []

        for r in post_srt_subjects:
            cls = r[f"{name}__post_srt_class"]
            if cls == "prevalent_or_same_day":
                pre_landmark += 1
                continue
            if cls == "positive_missing_age":
                missing_age += 1
                continue
            if cls == "post_censor":
                post_censor += 1
                continue
            censor_age = safe_float(r["censor_age"])
            landmark_age = safe_float(r["post_srt_ecg_age"])
            if censor_age is None or landmark_age is None or censor_age + ONE_DAY_YEARS < landmark_age:
                continue
            clean.append(r)
            if cls == "incident":
                incident.append(r)
                end_age = safe_float(r[f"{name}__age"])
            else:
                end_age = censor_age
            if end_age is not None:
                person_years.append(max(0.0, end_age - landmark_age))

        ne = len(incident)
        n = len(clean)
        py = sum(person_years)
        post_srt_summary.append(
            {
                "endpoint": name,
                "category": spec["category"],
                "n_subjects_with_post_srt_landmark_30_180d": len(post_srt_subjects),
                "n_subjects_with_both_pre_and_post_srt_ecg": len(post_srt_with_pre),
                "n_event_before_or_at_post_srt_landmark_excluded": pre_landmark,
                "n_positive_missing_event_age_excluded": missing_age,
                "n_event_after_recorded_censor_excluded": post_censor,
                "n_post_srt_at_risk": n,
                "n_incident_events_after_post_srt_landmark": ne,
                "post_srt_event_percent": pct(ne, n),
                "post_srt_total_person_years": py,
                "post_srt_events_per_100_person_years": (100.0 * ne / py) if py > 0 else "",
                "median_post_srt_observed_time_years": median_or_blank(person_years),
            }
        )

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    subject_path = OUT / "z06_event_feasibility_subject_level.csv"
    if subject_rows:
        with subject_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(subject_rows[0].keys()))
            w.writeheader()
            w.writerows(subject_rows)

    base_path = OUT / "z06_event_feasibility_summary.csv"
    with base_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baseline_summary[0].keys()))
        w.writeheader()
        w.writerows(baseline_summary)

    srt_path = OUT / "z06_post_srt_feasibility_summary.csv"
    with srt_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(post_srt_summary[0].keys()))
        w.writeheader()
        w.writerows(post_srt_summary)

    report_path = OUT / "z06_event_feasibility_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("SMASH-HCM: z06 future-event feasibility\n")
        f.write("=======================================\n\n")
        f.write(f"Exact-mapped subjects with valid baseline {Z06_COL}: {len(subject_rows)}\n")
        f.write(
            f"Subjects with stable post-SRT z06 ECG ({POST_SRT_MIN_DAYS:.0f}-{POST_SRT_MAX_DAYS:.0f} d): "
            f"{len(post_srt_subjects)}\n"
        )
        f.write(
            f"Subjects with both pre-SRT ECG (<= {PRE_SRT_MAX_DAYS:.0f} d) and stable post-SRT ECG: "
            f"{len(post_srt_with_pre)}\n\n"
        )
        f.write("Baseline landmark feasibility\n")
        f.write("-----------------------------\n")
        for r in baseline_summary:
            pctv = r["incident_event_percent"]
            pct_txt = f"{pctv:.1f}%" if isinstance(pctv, (int, float)) else "NA"
            f.write(
                f"{r['endpoint']}: at risk={r['n_at_risk']}, incident={r['n_incident_events']} "
                f"({pct_txt}), median observed time={r['median_observed_time_years']} y\n"
            )
        f.write("\nPost-SRT landmark feasibility\n")
        f.write("-----------------------------\n")
        for r in post_srt_summary:
            pctv = r["post_srt_event_percent"]
            pct_txt = f"{pctv:.1f}%" if isinstance(pctv, (int, float)) else "NA"
            f.write(
                f"{r['endpoint']}: at risk={r['n_post_srt_at_risk']}, "
                f"incident after landmark={r['n_incident_events_after_post_srt_landmark']} ({pct_txt})\n"
            )
        f.write("\nInterpretation rule for the next step\n")
        f.write("-------------------------------------\n")
        f.write(
            "Use these counts to choose endpoints before fitting Cox models. Sparse endpoints should not be "
            "rescued by adding many covariates. Baseline Cox should precede time-updated and post-SRT landmark "
            "models. Do not use full-history BLUP slopes to predict an earlier event.\n"
        )

    print(f"Exact-mapped subjects with valid baseline {Z06_COL}: {len(subject_rows)}")
    print(f"Post-SRT landmarks ({POST_SRT_MIN_DAYS:.0f}-{POST_SRT_MAX_DAYS:.0f} d): {len(post_srt_subjects)}")
    print(f"Pre+post SRT pairs: {len(post_srt_with_pre)}")
    print("\nBASELINE FEASIBILITY")
    for r in baseline_summary:
        print(
            f"{r['endpoint']:<30} at-risk={r['n_at_risk']:>3}  "
            f"events={r['n_incident_events']:>3}  "
            f"event%={r['incident_event_percent'] if r['incident_event_percent'] != '' else 'NA'}"
        )
    print("\nPOST-SRT FEASIBILITY")
    for r in post_srt_summary:
        print(
            f"{r['endpoint']:<30} at-risk={r['n_post_srt_at_risk']:>3}  "
            f"events={r['n_incident_events_after_post_srt_landmark']:>3}"
        )
    print(f"\nOutputs written to: {OUT}")


if __name__ == "__main__":
    main()
