#!/usr/bin/env python
"""
build_smash_hcm_harmonized_4s.py

Build harmonized 12-lead ECG datasets for SMASH-HCM under the working
assumption:

    sampling frequency = 500 Hz
    common window       = first 2000 samples = 4.0 s

The script creates two HDF5 files:

1) Phase1_Normal_vs_HCM_4s500Hz.h5
   - one ECG per PTB-XL normal subject
   - one baseline ECG per HCM subject
   - intended for cross-sectional Normal vs HCM latent-space analysis

2) Phase2_HCM_longitudinal_4s500Hz.h5
   - all unique HCM ECG visits
   - intended for longitudinal latent-space analysis

IMPORTANT
---------
This is a preliminary harmonization based on the explicit working assumption
that the HCM records correspond to 500 Hz. No filtering or amplitude
normalization is applied here.

Expected project structure:

_SMASH-HCM/
├── Data/
│   ├── HCM_ECG_VAE_ready.h5
│   ├── PTBXL_normal_reference_manifest.csv
│   └── PTBXL_Normal/
│       └── records500/
├── Results/
│   └── 02_SMASH_HCM_Harmonized_4s/
├── Scripts/
│   └── build_smash_hcm_harmonized_4s.py
└── venv/

Requirements:
    pip install numpy pandas h5py wfdb tqdm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


FS_HZ = 500.0
N_SAMPLES = 2000
WINDOW_SECONDS = N_SAMPLES / FS_HZ

CANONICAL_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


def infer_project_root() -> Path:
    p = Path(__file__).resolve().parent.parent
    if (p / "Data").exists():
        return p
    cwd = Path.cwd().resolve()
    if (cwd / "Data").exists():
        return cwd
    return p


def decode_array(arr):
    a = np.asarray(arr)
    if a.dtype.kind in {"S", "O", "U"}:
        out = []
        for x in a:
            if isinstance(x, bytes):
                out.append(x.decode("utf-8", errors="replace"))
            else:
                out.append(str(x))
        return np.asarray(out, dtype=object)
    return a


def h5_read_required(h5: h5py.File, path: str):
    if path not in h5:
        raise KeyError(f"Required HDF5 dataset not found: {path}")
    return h5[path][:]


def normalize_lead_name(name: str) -> str:
    return str(name).strip().upper()


def reorder_ptb_signal(signal: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Input WFDB shape: [samples, leads].
    Output shape: [12, samples] in canonical lead order.
    """
    names_norm = [normalize_lead_name(x) for x in names]
    missing = [lead for lead in CANONICAL_LEADS if lead not in names_norm]
    if missing:
        raise ValueError(f"Missing PTB-XL leads: {missing}; observed={names}")

    idx = [names_norm.index(lead) for lead in CANONICAL_LEADS]
    return signal[:, idx].T


def safe_datetime(values):
    return pd.to_datetime(pd.Series(values), errors="coerce")


def choose_first_per_subject(df: pd.DataFrame, subject_col: str,
                             time_candidates: list[str]) -> pd.DataFrame:
    tmp = df.copy()

    chosen_time = None
    for col in time_candidates:
        if col in tmp.columns:
            dt = pd.to_datetime(tmp[col], errors="coerce")
            if dt.notna().any():
                tmp["_sort_time"] = dt
                chosen_time = col
                break

    if chosen_time is None:
        tmp["_sort_time"] = pd.Timestamp.max

    if "ecg_id" in tmp.columns:
        tmp["_sort_ecg_id"] = pd.to_numeric(tmp["ecg_id"], errors="coerce")
    else:
        tmp["_sort_ecg_id"] = np.arange(len(tmp))

    tmp = tmp.sort_values(
        [subject_col, "_sort_time", "_sort_ecg_id"],
        kind="stable"
    )
    out = tmp.drop_duplicates(subject_col, keep="first").copy()
    return out.drop(columns=["_sort_time", "_sort_ecg_id"], errors="ignore")


def load_hcm(hcm_path: Path):
    with h5py.File(hcm_path, "r") as h5:
        signals = h5_read_required(h5, "signals/ecg").astype(np.float32)
        ecg_id = decode_array(h5_read_required(h5, "metadata/ecg_id"))
        patient_id = decode_array(h5_read_required(h5, "metadata/patient_id"))
        subject_index = np.asarray(
            h5_read_required(h5, "metadata/subject_index")
        ).astype(np.int64)

        if "metadata/subject_key" in h5:
            subject_key = decode_array(h5["metadata/subject_key"][:])
        else:
            subject_key = np.asarray(
                [f"HCM_{x}" for x in subject_index], dtype=object
            )

        if "metadata/ecg_time" in h5:
            ecg_time = decode_array(h5["metadata/ecg_time"][:])
        else:
            ecg_time = np.asarray([""] * len(ecg_id), dtype=object)

        birthdate = (
            decode_array(h5["metadata/birthdate"][:])
            if "metadata/birthdate" in h5
            else np.asarray([""] * len(ecg_id), dtype=object)
        )

        ecg_number = (
            np.asarray(h5["metadata/ecg_number_per_patient"][:])
            if "metadata/ecg_number_per_patient" in h5
            else np.full(len(ecg_id), np.nan)
        )

        lengths = (
            np.asarray(h5["signals/lengths"][:])
            if "signals/lengths" in h5
            else np.sum(np.isfinite(signals), axis=2)
        )

    if signals.ndim != 3 or signals.shape[1] != 12:
        raise ValueError(
            f"Unexpected HCM signal shape {signals.shape}; expected [N,12,L]."
        )
    if signals.shape[2] < N_SAMPLES:
        raise ValueError(
            f"HCM array contains only {signals.shape[2]} samples; "
            f"{N_SAMPLES} required."
        )

    x4 = signals[:, :, :N_SAMPLES].copy()

    # Strong QC: every selected point must be finite.
    finite_mask = np.isfinite(x4).all(axis=(1, 2))
    lengths_ok = np.all(lengths >= N_SAMPLES, axis=1)
    keep = finite_mask & lengths_ok

    meta = pd.DataFrame({
        "row_index": np.arange(len(ecg_id)),
        "ecg_id": ecg_id,
        "patient_id": patient_id,
        "subject_index": subject_index,
        "subject_key": subject_key,
        "ecg_time": ecg_time,
        "birthdate": birthdate,
        "ecg_number_per_patient": ecg_number,
        "window_all_finite": finite_mask,
        "all_leads_length_ge_2000": lengths_ok,
    })

    return x4, meta, keep


def read_ptb_record(record_base: Path):
    rec = wfdb.rdrecord(str(record_base), physical=True)
    if float(rec.fs) != FS_HZ:
        raise ValueError(f"Expected 500 Hz, observed {rec.fs}")
    if rec.p_signal is None:
        raise ValueError("WFDB returned no physical signal")

    sig = np.asarray(rec.p_signal, dtype=np.float32)
    if sig.shape[0] < N_SAMPLES:
        raise ValueError(
            f"Only {sig.shape[0]} samples available; need {N_SAMPLES}"
        )

    x = reorder_ptb_signal(sig[:N_SAMPLES, :], list(rec.sig_name))
    if x.shape != (12, N_SAMPLES):
        raise ValueError(f"Unexpected reordered PTB shape: {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("NaN/Inf detected in selected PTB window")
    return x


def write_str_dataset(group, name, values):
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(
        name,
        data=np.asarray([str(x) for x in values], dtype=object),
        dtype=dt,
        compression="gzip",
        compression_opts=4,
    )


def write_phase1(out_path: Path, x_normal: np.ndarray,
                 normal_meta: pd.DataFrame,
                 x_hcm: np.ndarray, hcm_meta: pd.DataFrame):

    X = np.concatenate([x_normal, x_hcm], axis=0).astype(np.float32)
    y = np.concatenate([
        np.zeros(len(x_normal), dtype=np.uint8),
        np.ones(len(x_hcm), dtype=np.uint8)
    ])

    source = np.asarray(
        ["PTBXL_NORMAL"] * len(x_normal) + ["HCM"] * len(x_hcm),
        dtype=object
    )

    subject_key = np.concatenate([
        np.asarray(
            [f"PTBXL_{x}" for x in normal_meta["patient_id"]],
            dtype=object
        ),
        hcm_meta["subject_key"].astype(str).to_numpy(dtype=object)
    ])

    ecg_id = np.concatenate([
        normal_meta["ecg_id"].astype(str).to_numpy(dtype=object),
        hcm_meta["ecg_id"].astype(str).to_numpy(dtype=object)
    ])

    patient_id = np.concatenate([
        normal_meta["patient_id"].astype(str).to_numpy(dtype=object),
        hcm_meta["patient_id"].astype(str).to_numpy(dtype=object)
    ])

    with h5py.File(out_path, "w") as h5:
        h5.attrs["description"] = (
            "SMASH-HCM Phase I raw-harmonized cross-sectional dataset"
        )
        h5.attrs["working_sampling_rate_hz"] = FS_HZ
        h5.attrs["n_samples"] = N_SAMPLES
        h5.attrs["window_seconds"] = WINDOW_SECONDS
        h5.attrs["window_definition"] = "first 2000 samples"
        h5.attrs["normalization_applied"] = False
        h5.attrs["filtering_applied"] = False
        h5.attrs["hcm_500hz_status"] = (
            "working assumption; not yet verified from original raw HCM source"
        )

        sg = h5.create_group("signals")
        sg.create_dataset(
            "ecg",
            data=X,
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            chunks=(1, 12, N_SAMPLES),
        )
        write_str_dataset(
            sg, "lead_names",
            ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]
        )

        lg = h5.create_group("labels")
        lg.create_dataset("hcm", data=y, dtype="uint8")

        mg = h5.create_group("metadata")
        write_str_dataset(mg, "source", source)
        write_str_dataset(mg, "subject_key", subject_key)
        write_str_dataset(mg, "ecg_id", ecg_id)
        write_str_dataset(mg, "patient_id", patient_id)

        # Optional PTB-XL covariates, padded as strings for HCM if unavailable.
        age = np.concatenate([
            pd.to_numeric(
                normal_meta.get("age", pd.Series([np.nan]*len(normal_meta))),
                errors="coerce"
            ).to_numpy(dtype=np.float32),
            np.full(len(hcm_meta), np.nan, dtype=np.float32),
        ])
        mg.create_dataset("age_ptbxl_only", data=age, dtype="float32")

        if "sex" in normal_meta.columns:
            ptb_sex = normal_meta["sex"].astype(str).to_numpy(dtype=object)
        else:
            ptb_sex = np.asarray([""] * len(normal_meta), dtype=object)
        sex = np.concatenate([
            ptb_sex,
            np.asarray([""] * len(hcm_meta), dtype=object)
        ])
        write_str_dataset(mg, "sex_ptbxl_only", sex)


def write_phase2(out_path: Path, x_hcm_all: np.ndarray,
                 hcm_meta_all: pd.DataFrame):
    with h5py.File(out_path, "w") as h5:
        h5.attrs["description"] = (
            "SMASH-HCM Phase II raw-harmonized longitudinal HCM dataset"
        )
        h5.attrs["working_sampling_rate_hz"] = FS_HZ
        h5.attrs["n_samples"] = N_SAMPLES
        h5.attrs["window_seconds"] = WINDOW_SECONDS
        h5.attrs["window_definition"] = "first 2000 samples"
        h5.attrs["normalization_applied"] = False
        h5.attrs["filtering_applied"] = False
        h5.attrs["hcm_500hz_status"] = (
            "working assumption; not yet verified from original raw HCM source"
        )

        sg = h5.create_group("signals")
        sg.create_dataset(
            "ecg",
            data=x_hcm_all.astype(np.float32),
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            chunks=(1, 12, N_SAMPLES),
        )
        write_str_dataset(
            sg, "lead_names",
            ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]
        )

        mg = h5.create_group("metadata")
        for col in [
            "ecg_id", "patient_id", "subject_key",
            "ecg_time", "birthdate"
        ]:
            write_str_dataset(mg, col, hcm_meta_all[col].astype(str).values)

        mg.create_dataset(
            "subject_index",
            data=hcm_meta_all["subject_index"].to_numpy(dtype=np.int64),
            dtype="int64"
        )

        vals = pd.to_numeric(
            hcm_meta_all["ecg_number_per_patient"], errors="coerce"
        ).fillna(-1).to_numpy(dtype=np.int32)
        mg.create_dataset(
            "ecg_number_per_patient", data=vals, dtype="int32"
        )


def main():
    root = infer_project_root()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hcm-h5",
        type=Path,
        default=root / "Data" / "HCM_ECG_VAE_ready.h5"
    )
    ap.add_argument(
        "--ptb-manifest",
        type=Path,
        default=root / "Data" / "PTBXL_normal_reference_manifest.csv"
    )
    ap.add_argument(
        "--ptb-root",
        type=Path,
        default=root / "Data" / "PTBXL_Normal"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "Results" / "02_SMASH_HCM_Harmonized_4s"
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HCM dataset...")
    x_hcm, hcm_meta, hcm_keep = load_hcm(args.hcm_h5)

    n_bad_hcm = int((~hcm_keep).sum())
    print(f"HCM records available: {len(hcm_meta):,}")
    print(f"HCM records valid for first 2000 samples: {hcm_keep.sum():,}")
    print(f"HCM records excluded from 4-s window: {n_bad_hcm:,}")

    x_hcm_valid = x_hcm[hcm_keep]
    hcm_valid_meta = hcm_meta.loc[hcm_keep].reset_index(drop=True)

    # Baseline = earliest ECG per HCM subject.
    hcm_baseline_meta = choose_first_per_subject(
        hcm_valid_meta,
        subject_col="subject_index",
        time_candidates=["ecg_time"]
    )
    hcm_baseline_indices = hcm_baseline_meta.index.to_numpy()
    # choose_first_per_subject preserved original/reset indexes of hcm_valid_meta
    x_hcm_baseline = x_hcm_valid[hcm_baseline_indices]
    hcm_baseline_meta = hcm_valid_meta.loc[
        hcm_baseline_indices
    ].reset_index(drop=True)

    print(f"HCM baseline subjects: {len(hcm_baseline_meta):,}")

    print("\nLoading PTB-XL manifest...")
    ptb = pd.read_csv(args.ptb_manifest)

    if "patient_id" not in ptb or "filename_hr" not in ptb or "ecg_id" not in ptb:
        raise KeyError(
            "PTB manifest must contain patient_id, ecg_id, filename_hr"
        )

    # One ECG per normal subject.
    normal_baseline = choose_first_per_subject(
        ptb,
        subject_col="patient_id",
        time_candidates=["recording_date", "ecg_time", "date"]
    ).reset_index(drop=True)

    print(f"PTB-XL normal subjects selected: {len(normal_baseline):,}")

    x_normal = np.empty(
        (len(normal_baseline), 12, N_SAMPLES), dtype=np.float32
    )
    failures = []

    print("\nReading PTB-XL waveforms...")
    for i, row in tqdm(
        normal_baseline.iterrows(),
        total=len(normal_baseline),
        desc="PTB-XL baseline",
        unit="ECG"
    ):
        rel = str(row["filename_hr"]).replace("\\", "/")
        record_base = args.ptb_root / Path(rel)
        try:
            x_normal[i] = read_ptb_record(record_base)
        except Exception as exc:
            failures.append({
                "row": i,
                "ecg_id": row["ecg_id"],
                "patient_id": row["patient_id"],
                "filename_hr": row["filename_hr"],
                "error": f"{type(exc).__name__}: {exc}",
            })
            x_normal[i] = np.nan

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_path = args.out_dir / "PTBXL_phase1_read_failures.csv"
        fail_df.to_csv(fail_path, index=False)
        print(f"WARNING: {len(failures)} PTB records failed; see {fail_path}")

    good_normal = np.isfinite(x_normal).all(axis=(1, 2))
    x_normal = x_normal[good_normal]
    normal_baseline = normal_baseline.loc[good_normal].reset_index(drop=True)

    phase1 = args.out_dir / "Phase1_Normal_vs_HCM_4s500Hz.h5"
    phase2 = args.out_dir / "Phase2_HCM_longitudinal_4s500Hz.h5"

    print("\nWriting Phase I dataset...")
    write_phase1(
        phase1,
        x_normal,
        normal_baseline,
        x_hcm_baseline,
        hcm_baseline_meta
    )

    print("Writing Phase II dataset...")
    write_phase2(
        phase2,
        x_hcm_valid,
        hcm_valid_meta
    )

    summary = {
        "working_sampling_rate_hz": FS_HZ,
        "samples_per_lead": N_SAMPLES,
        "window_seconds": WINDOW_SECONDS,
        "window": "first 2000 samples",
        "phase1_normal_ecg": int(len(x_normal)),
        "phase1_hcm_ecg": int(len(x_hcm_baseline)),
        "phase1_total_ecg": int(len(x_normal) + len(x_hcm_baseline)),
        "phase2_hcm_ecg": int(len(x_hcm_valid)),
        "phase2_hcm_subjects": int(hcm_valid_meta["subject_index"].nunique()),
        "hcm_records_excluded_for_insufficient_or_nonfinite_first_2000": n_bad_hcm,
        "filtering_applied": False,
        "normalization_applied": False,
        "hcm_500hz_status": "working assumption",
    }

    summary_path = args.out_dir / "harmonization_4s500Hz_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\nDone.")
    print(json.dumps(summary, indent=2))
    print(f"\nPhase I : {phase1}")
    print(f"Phase II: {phase2}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
