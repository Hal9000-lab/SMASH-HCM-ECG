#!/usr/bin/env python
"""
01_validate_ptbxl_normal.py

Validate the PTB-XL Normal Reference Cohort downloaded for the SMASH-HCM project.

Expected project structure
--------------------------
_SMASH-HCM/
├── Data/
│   ├── PTBXL_normal_reference_manifest.csv
│   └── PTBXL_Normal/
│       └── records500/
├── Results/
├── Scripts/
│   └── 01_validate_ptbxl_normal.py
└── venv/

Checks performed
----------------
- manifest consistency
- presence of .hea and .dat files
- WFDB readability
- 12 leads
- sampling frequency = 500 Hz
- 5000 samples
- expected 12-lead order
- NaN / Inf values
- constant / zero-variance leads
- amplitude range and simple outlier flags
- duplicated waveform hashes across different ECG IDs

Outputs
-------
Results/01_PTBXL_Normal_Validation/PTBXL_normal_validation_report.csv
Results/01_PTBXL_Normal_Validation/PTBXL_normal_validation_summary.txt
Results/01_PTBXL_Normal_Validation/PTBXL_normal_duplicate_waveforms.csv

Requirements
------------
pip install numpy pandas wfdb tqdm
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


EXPECTED_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
EXPECTED_FS = 500.0
EXPECTED_SAMPLES = 5000
MAX_ABS_MV_FLAG = 20.0
MIN_STD_MV_FLAG = 1e-6


def infer_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidate = script_path.parent.parent
    if (candidate / "Data").exists():
        return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "Data").exists():
        return cwd
    return candidate


def normalize_record_path(filename_hr: str) -> str:
    p = str(filename_hr).strip().replace("\\", "/")
    if p.endswith(".hea") or p.endswith(".dat"):
        p = p[:-4]
    return p


def waveform_sha256(signal: np.ndarray) -> str:
    arr = np.ascontiguousarray(signal.astype(np.float32, copy=False))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def validate_record(row: pd.Series, data_root: Path) -> dict:
    result = {
        "ecg_id": row.get("ecg_id", np.nan),
        "patient_id": row.get("patient_id", np.nan),
        "filename_hr": row.get("filename_hr", ""),
        "hea_exists": False,
        "dat_exists": False,
        "wfdb_readable": False,
        "n_sig": np.nan,
        "n_samples": np.nan,
        "fs_hz": np.nan,
        "lead_names": "",
        "lead_names_normalized": "",
        "lead_order_ok": False,
        "has_nan": False,
        "has_inf": False,
        "flat_lead_count": np.nan,
        "max_abs_mv": np.nan,
        "mean_abs_mv": np.nan,
        "amplitude_flag": False,
        "flat_signal_flag": False,
        "shape_ok": False,
        "fs_ok": False,
        "lead_count_ok": False,
        "all_core_checks_ok": False,
        "waveform_sha256": "",
        "error": "",
    }

    try:
        rel = normalize_record_path(row["filename_hr"])
        record_base = data_root / Path(rel)
        hea = record_base.with_suffix(".hea")
        dat = record_base.with_suffix(".dat")

        result["hea_exists"] = hea.exists()
        result["dat_exists"] = dat.exists()

        if not result["hea_exists"] or not result["dat_exists"]:
            missing = []
            if not result["hea_exists"]:
                missing.append(".hea")
            if not result["dat_exists"]:
                missing.append(".dat")
            result["error"] = "missing file(s): " + ", ".join(missing)
            return result

        record = wfdb.rdrecord(str(record_base), physical=True)
        result["wfdb_readable"] = True
        result["n_sig"] = int(record.n_sig)
        result["fs_hz"] = float(record.fs)
        result["lead_names"] = "|".join(record.sig_name)
        normalized_leads = [str(x).strip().upper() for x in record.sig_name]
        result["lead_names_normalized"] = "|".join(normalized_leads)

        signal = record.p_signal
        if signal is None:
            result["error"] = "wfdb returned no physical signal"
            return result

        signal = np.asarray(signal)
        result["n_samples"] = int(signal.shape[0])
        result["lead_count_ok"] = (signal.ndim == 2 and signal.shape[1] == 12)
        result["shape_ok"] = (
            signal.ndim == 2 and
            signal.shape[0] == EXPECTED_SAMPLES and
            signal.shape[1] == 12
        )
        result["fs_ok"] = np.isclose(result["fs_hz"], EXPECTED_FS)
        result["lead_order_ok"] = normalized_leads == EXPECTED_LEADS
        result["has_nan"] = bool(np.isnan(signal).any())
        result["has_inf"] = bool(np.isinf(signal).any())

        if signal.ndim == 2:
            std_by_lead = np.nanstd(signal, axis=0)
            result["flat_lead_count"] = int(np.sum(std_by_lead < MIN_STD_MV_FLAG))
            result["flat_signal_flag"] = result["flat_lead_count"] > 0

        finite = signal[np.isfinite(signal)]
        if finite.size:
            result["max_abs_mv"] = float(np.max(np.abs(finite)))
            result["mean_abs_mv"] = float(np.mean(np.abs(finite)))
            result["amplitude_flag"] = result["max_abs_mv"] > MAX_ABS_MV_FLAG

        result["waveform_sha256"] = waveform_sha256(signal)
        result["all_core_checks_ok"] = bool(
            result["wfdb_readable"] and
            result["shape_ok"] and
            result["fs_ok"] and
            result["lead_order_ok"] and
            not result["has_nan"] and
            not result["has_inf"] and
            not result["flat_signal_flag"]
        )
        return result

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def make_duplicate_report(report: pd.DataFrame) -> pd.DataFrame:
    valid_hash = report[
        report["wfdb_readable"] & report["waveform_sha256"].astype(bool)
    ].copy()

    dup = valid_hash[
        valid_hash.duplicated("waveform_sha256", keep=False)
    ].copy()

    if dup.empty:
        return pd.DataFrame(
            columns=["waveform_sha256", "ecg_id", "patient_id", "filename_hr"]
        )

    return dup[
        ["waveform_sha256", "ecg_id", "patient_id", "filename_hr"]
    ].sort_values(["waveform_sha256", "ecg_id"])


def build_summary(report, manifest, duplicate_report, manifest_path, data_root) -> str:
    n = len(report)
    n_patients = manifest["patient_id"].nunique() if "patient_id" in manifest else np.nan
    duplicate_hash_groups = (
        duplicate_report["waveform_sha256"].nunique()
        if not duplicate_report.empty else 0
    )
    duplicate_records = len(duplicate_report)

    lines = [
        "PTB-XL Normal Reference Cohort - Validation Summary",
        "=" * 58,
        f"Manifest: {manifest_path}",
        f"Data root: {data_root}",
        "",
        f"Manifest ECG records: {len(manifest):,}",
        f"Unique patients in manifest: {n_patients:,}" if pd.notna(n_patients) else "Unique patients in manifest: N/A",
        "",
        f"Records checked: {n:,}",
        f".hea present: {int(report['hea_exists'].sum()):,}/{n:,}",
        f".dat present: {int(report['dat_exists'].sum()):,}/{n:,}",
        f"WFDB-readable: {int(report['wfdb_readable'].sum()):,}/{n:,}",
        f"12 leads: {int(report['lead_count_ok'].sum()):,}/{n:,}",
        f"5000 x 12 shape: {int(report['shape_ok'].sum()):,}/{n:,}",
        f"500 Hz: {int(report['fs_ok'].sum()):,}/{n:,}",
        f"Expected lead order: {int(report['lead_order_ok'].sum()):,}/{n:,}",
        f"No NaN: {int((~report['has_nan']).sum()):,}/{n:,}",
        f"No Inf: {int((~report['has_inf']).sum()):,}/{n:,}",
        f"No flat leads: {int((~report['flat_signal_flag']).sum()):,}/{n:,}",
        "",
        f"Core-valid records: {int(report['all_core_checks_ok'].sum()):,}/{n:,}",
        f"Core-invalid records: {int((~report['all_core_checks_ok']).sum()):,}/{n:,}",
        f"Amplitude flags (> {MAX_ABS_MV_FLAG:g} mV absolute): {int(report['amplitude_flag'].sum()):,}",
        "",
        f"Exact waveform duplicate hash groups: {duplicate_hash_groups:,}",
        f"Records belonging to duplicate waveform groups: {duplicate_records:,}",
    ]

    readable = report[report["wfdb_readable"]].copy()
    if not readable.empty:
        lines += [
            "",
            "Observed dimensions / signal characteristics",
            "-" * 42,
            f"Sampling rates: {sorted(readable['fs_hz'].dropna().unique().tolist())}",
            f"Sample counts: {sorted(readable['n_samples'].dropna().unique().astype(int).tolist())}",
            f"Signal counts: {sorted(readable['n_sig'].dropna().unique().astype(int).tolist())}",
            f"Median max |amplitude|: {readable['max_abs_mv'].median():.4f} mV",
            f"95th percentile max |amplitude|: {readable['max_abs_mv'].quantile(0.95):.4f} mV",
            f"Maximum observed |amplitude|: {readable['max_abs_mv'].max():.4f} mV",
        ]

    errors = report[report["error"].astype(str).str.len() > 0]
    if not errors.empty:
        lines += [
            "",
            "Records with errors",
            "-" * 42,
            f"{len(errors):,} records have a non-empty error field.",
            "See PTBXL_normal_validation_report.csv for details.",
        ]

    lines += [
        "",
        "Interpretation",
        "-" * 42,
        "A record is 'core-valid' only if it is WFDB-readable, has exactly",
        "5000 samples x 12 leads, is sampled at 500 Hz, has the expected",
        "lead order, contains no NaN/Inf, and contains no flat lead.",
        "",
        "The amplitude threshold is only a QC flag and does NOT automatically",
        "exclude a record.",
    ]
    return "\n".join(lines)


def main() -> int:
    root = infer_project_root()

    parser = argparse.ArgumentParser(
        description="Validate downloaded PTB-XL Normal Reference Cohort."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "Data" / "PTBXL_normal_reference_manifest.csv",
        help="Manifest CSV path."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=root / "Data" / "PTBXL_Normal",
        help="Root containing records500/."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "Results" / "01_PTBXL_Normal_Validation",
        help="Directory for validation reports."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally validate only the first N records."
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        return 2

    manifest = pd.read_csv(manifest_path)

    required = {"ecg_id", "patient_id", "filename_hr"}
    missing_cols = required.difference(manifest.columns)
    if missing_cols:
        print(f"ERROR: manifest missing columns: {sorted(missing_cols)}", file=sys.stderr)
        return 2

    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    print(f"Project root : {root}")
    print(f"Manifest     : {manifest_path}")
    print(f"Data root    : {data_root}")
    print(f"Records      : {len(manifest):,}")
    print()

    rows = []
    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Validating PTB-XL",
        unit="ECG"
    ):
        rows.append(validate_record(row, data_root))

    report = pd.DataFrame(rows)

    report_path = results_dir / "PTBXL_normal_validation_report.csv"
    report.to_csv(report_path, index=False)

    duplicate_report = make_duplicate_report(report)
    dup_path = results_dir / "PTBXL_normal_duplicate_waveforms.csv"
    duplicate_report.to_csv(dup_path, index=False)

    summary = build_summary(
        report=report,
        manifest=manifest,
        duplicate_report=duplicate_report,
        manifest_path=manifest_path,
        data_root=data_root,
    )
    summary_path = results_dir / "PTBXL_normal_validation_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print(f"Detailed report : {report_path}")
    print(f"Duplicate report: {dup_path}")
    print(f"Summary          : {summary_path}")

    return 0 if report["all_core_checks_ok"].all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
