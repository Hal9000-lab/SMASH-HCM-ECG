#!/usr/bin/env python
"""
prepare_phase1_normal_anchored.py

SMASH-HCM -- Phase I preparation for a normal-anchored latent-space analysis.

Working dataset:
    12 leads x 2000 samples x 500 Hz = first 4 s

Core design
-----------
1. Perform QC on raw Normal and HCM baseline ECGs.
2. Split NORMAL subjects only into train / validation / test (70/15/15).
3. HCM ECGs are NEVER used for VAE training or hyperparameter selection.
4. Apply the same deterministic preprocessing to both cohorts:
       Butterworth band-pass 0.5--40 Hz, zero-phase.
5. Estimate ONE global mean and ONE global standard deviation using only the
   preprocessed NORMAL TRAINING SET.
   This intentionally avoids per-ECG normalization and preserves relative
   voltage information and cross-lead amplitude relationships.
6. Save a VAE-ready HDF5 dataset with split labels and preprocessing metadata.

Expected project structure
--------------------------
_SMASH-HCM/
├── Data/
├── Results/
│   └── 02_SMASH_HCM_Harmonized_4s/
│       └── Phase1_Normal_vs_HCM_4s500Hz.h5
├── Scripts/
│   └── 03_prepare_phase1_normal_anchored.py
└── venv/

Outputs
-------
Results/03_Phase1_Normal_Anchored/
├── Phase1_NormalAnchored_preprocessed.h5
├── phase1_record_qc.csv
├── phase1_lead_qc_summary.csv
├── phase1_split_summary.json
├── phase1_preprocessing_summary.json
├── qc_global_rms.png
└── qc_global_maxabs.png

Requirements
------------
pip install numpy pandas scipy h5py matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt


FS_HZ = 500.0
LOW_HZ = 0.5
HIGH_HZ = 40.0
FILTER_ORDER = 4
SEED = 20261004

SPLIT_NORMAL_TRAIN = 0
SPLIT_NORMAL_VAL = 1
SPLIT_NORMAL_TEST = 2
SPLIT_HCM_EVAL = 3

SPLIT_NAMES = {
    SPLIT_NORMAL_TRAIN: "normal_train",
    SPLIT_NORMAL_VAL: "normal_val",
    SPLIT_NORMAL_TEST: "normal_test",
    SPLIT_HCM_EVAL: "hcm_eval",
}


def infer_project_root() -> Path:
    script_root = Path(__file__).resolve().parent.parent
    if (script_root / "Data").exists():
        return script_root

    cwd = Path.cwd().resolve()
    if (cwd / "Data").exists():
        return cwd

    return script_root


def decode_str_array(arr) -> np.ndarray:
    out = []
    for x in np.asarray(arr):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def write_str_dataset(group, name, values):
    dt = h5py.string_dtype("utf-8")
    group.create_dataset(
        name,
        data=np.asarray([str(x) for x in values], dtype=object),
        dtype=dt,
        compression="gzip",
        compression_opts=4,
    )


def make_filter():
    return butter(
        FILTER_ORDER,
        [LOW_HZ, HIGH_HZ],
        btype="bandpass",
        fs=FS_HZ,
        output="sos",
    )


def filter_batch(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """
    x shape: [N, 12, samples]
    Filtering is performed on the last axis using zero-phase SOS filtering.
    """
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        out[i] = sosfiltfilt(sos, x[i], axis=-1).astype(np.float32)
    return out


def split_normal_subjects(
    subject_keys: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Assign split codes.

    Normal subjects:
        70% train, 15% validation, 15% test

    HCM:
        all records -> hcm_eval

    The implementation is subject-level even though Phase I currently
    contains one ECG per subject.
    """
    rng = np.random.default_rng(seed)
    split = np.full(len(labels), SPLIT_HCM_EVAL, dtype=np.uint8)

    normal_idx = np.where(labels == 0)[0]
    normal_subjects = np.unique(subject_keys[normal_idx]).copy()
    rng.shuffle(normal_subjects)

    n = len(normal_subjects)
    n_train = int(np.floor(0.70 * n))
    n_val = int(np.floor(0.15 * n))

    train_subjects = set(normal_subjects[:n_train])
    val_subjects = set(normal_subjects[n_train:n_train + n_val])
    test_subjects = set(normal_subjects[n_train + n_val:])

    for i in normal_idx:
        s = subject_keys[i]
        if s in train_subjects:
            split[i] = SPLIT_NORMAL_TRAIN
        elif s in val_subjects:
            split[i] = SPLIT_NORMAL_VAL
        elif s in test_subjects:
            split[i] = SPLIT_NORMAL_TEST
        else:
            raise RuntimeError(f"Normal subject not assigned: {s}")

    # Leakage assertions.
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)

    return split


def compute_record_qc(
    raw: np.ndarray,
    filtered: np.ndarray,
    labels: np.ndarray,
    subject_key: np.ndarray,
    ecg_id: np.ndarray,
    source: np.ndarray,
    split: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for i in range(raw.shape[0]):
        raw_i = raw[i]
        fil_i = filtered[i]

        rows.append({
            "record_index": i,
            "subject_key": subject_key[i],
            "ecg_id": ecg_id[i],
            "source": source[i],
            "class": "HCM" if labels[i] == 1 else "Normal",
            "split": SPLIT_NAMES[int(split[i])],

            "raw_mean_mv": float(np.mean(raw_i)),
            "raw_std_mv": float(np.std(raw_i)),
            "raw_rms_mv": float(np.sqrt(np.mean(raw_i ** 2))),
            "raw_max_abs_mv": float(np.max(np.abs(raw_i))),
            "raw_peak_to_peak_mv": float(np.max(raw_i) - np.min(raw_i)),

            "filtered_mean_mv": float(np.mean(fil_i)),
            "filtered_std_mv": float(np.std(fil_i)),
            "filtered_rms_mv": float(np.sqrt(np.mean(fil_i ** 2))),
            "filtered_max_abs_mv": float(np.max(np.abs(fil_i))),
            "filtered_peak_to_peak_mv": float(np.max(fil_i) - np.min(fil_i)),

            "has_nan_raw": bool(np.isnan(raw_i).any()),
            "has_inf_raw": bool(np.isinf(raw_i).any()),
            "has_nan_filtered": bool(np.isnan(fil_i).any()),
            "has_inf_filtered": bool(np.isinf(fil_i).any()),
        })

    return pd.DataFrame(rows)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(
        ((len(a) - 1) * va + (len(b) - 1) * vb)
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return 0.0

    return float((np.mean(b) - np.mean(a)) / pooled)


def compute_lead_qc(
    raw: np.ndarray,
    filtered: np.ndarray,
    labels: np.ndarray,
    lead_names: list[str],
) -> pd.DataFrame:
    rows = []

    for lead_idx, lead in enumerate(lead_names):
        for version_name, data in [("raw", raw), ("filtered", filtered)]:
            x = data[:, lead_idx, :]
            per_record_rms = np.sqrt(np.mean(x ** 2, axis=1))
            per_record_maxabs = np.max(np.abs(x), axis=1)
            per_record_mean = np.mean(x, axis=1)

            normal = labels == 0
            hcm = labels == 1

            for metric_name, metric in [
                ("rms_mv", per_record_rms),
                ("max_abs_mv", per_record_maxabs),
                ("mean_mv", per_record_mean),
            ]:
                rows.append({
                    "lead": lead,
                    "version": version_name,
                    "metric": metric_name,
                    "normal_mean": float(np.mean(metric[normal])),
                    "normal_median": float(np.median(metric[normal])),
                    "hcm_mean": float(np.mean(metric[hcm])),
                    "hcm_median": float(np.median(metric[hcm])),
                    "cohen_d_hcm_minus_normal": cohen_d(
                        metric[normal], metric[hcm]
                    ),
                })

    return pd.DataFrame(rows)


def plot_qc(record_qc: pd.DataFrame, results_dir: Path):
    normal = record_qc["class"] == "Normal"
    hcm = record_qc["class"] == "HCM"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        record_qc.loc[normal, "filtered_rms_mv"],
        bins=60,
        alpha=0.55,
        label="Normal",
        density=True,
    )
    ax.hist(
        record_qc.loc[hcm, "filtered_rms_mv"],
        bins=60,
        alpha=0.55,
        label="HCM",
        density=True,
    )
    ax.set_xlabel("Filtered global RMS (mV)")
    ax.set_ylabel("Density")
    ax.set_title("Phase I QC: global ECG RMS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "qc_global_rms.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        record_qc.loc[normal, "filtered_max_abs_mv"],
        bins=60,
        alpha=0.55,
        label="Normal",
        density=True,
    )
    ax.hist(
        record_qc.loc[hcm, "filtered_max_abs_mv"],
        bins=60,
        alpha=0.55,
        label="HCM",
        density=True,
    )
    ax.set_xlabel("Filtered maximum absolute amplitude (mV)")
    ax.set_ylabel("Density")
    ax.set_title("Phase I QC: maximum ECG amplitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "qc_global_maxabs.png", dpi=160)
    plt.close(fig)


def main():
    root = infer_project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "Results" / "02_SMASH_HCM_Harmonized_4s"
        / "Phase1_Normal_vs_HCM_4s500Hz.h5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Results" / "03_Phase1_Normal_Anchored"
        / "Phase1_NormalAnchored_preprocessed.h5",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "Results" / "03_Phase1_Normal_Anchored",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print()

    with h5py.File(args.input, "r") as h5:
        x = h5["signals/ecg"][:].astype(np.float32)
        labels = h5["labels/hcm"][:].astype(np.uint8)

        lead_names = decode_str_array(h5["signals/lead_names"][:]).tolist()
        source = decode_str_array(h5["metadata/source"][:])
        subject_key = decode_str_array(h5["metadata/subject_key"][:])
        ecg_id = decode_str_array(h5["metadata/ecg_id"][:])
        patient_id = decode_str_array(h5["metadata/patient_id"][:])

        age = (
            h5["metadata/age_ptbxl_only"][:].astype(np.float32)
            if "metadata/age_ptbxl_only" in h5
            else np.full(len(labels), np.nan, dtype=np.float32)
        )
        sex = (
            decode_str_array(h5["metadata/sex_ptbxl_only"][:])
            if "metadata/sex_ptbxl_only" in h5
            else np.asarray([""] * len(labels), dtype=object)
        )

    if x.shape[1:] != (12, 2000):
        raise ValueError(
            f"Unexpected signal shape: {x.shape}; expected [N,12,2000]."
        )

    if not np.isfinite(x).all():
        raise ValueError("Input contains NaN or Inf values.")

    print(f"Records: {len(labels):,}")
    print(f"Normal : {(labels == 0).sum():,}")
    print(f"HCM    : {(labels == 1).sum():,}")

    split = split_normal_subjects(subject_key, labels, args.seed)

    split_counts = {
        SPLIT_NAMES[k]: int(np.sum(split == k))
        for k in sorted(SPLIT_NAMES)
    }

    print("\nSplit:")
    for k, v in split_counts.items():
        print(f"  {k:14s}: {v:,}")

    print("\nApplying common 0.5-40 Hz zero-phase band-pass...")
    sos = make_filter()
    x_filtered = filter_batch(x, sos)

    if not np.isfinite(x_filtered).all():
        raise ValueError("Filtered signals contain NaN or Inf values.")

    train_mask = split == SPLIT_NORMAL_TRAIN
    train_values = x_filtered[train_mask].astype(np.float64)

    # ONE global scaling transformation from Normal train only.
    train_mean = float(np.mean(train_values))
    train_std = float(np.std(train_values))

    if train_std <= 0 or not np.isfinite(train_std):
        raise ValueError(f"Invalid normal-train scaling std: {train_std}")

    print("\nNormal-train global scaling:")
    print(f"  mean = {train_mean:.8f} mV")
    print(f"  std  = {train_std:.8f} mV")

    x_scaled = ((x_filtered - train_mean) / train_std).astype(np.float32)

    print("\nComputing QC tables...")
    record_qc = compute_record_qc(
        x, x_filtered, labels, subject_key, ecg_id, source, split
    )
    record_qc_path = args.results_dir / "phase1_record_qc.csv"
    record_qc.to_csv(record_qc_path, index=False)

    lead_qc = compute_lead_qc(
        x, x_filtered, labels, lead_names
    )
    lead_qc_path = args.results_dir / "phase1_lead_qc_summary.csv"
    lead_qc.to_csv(lead_qc_path, index=False)

    plot_qc(record_qc, args.results_dir)

    split_summary = {
        "seed": args.seed,
        "normal_total": int(np.sum(labels == 0)),
        "hcm_total": int(np.sum(labels == 1)),
        "normal_train": split_counts["normal_train"],
        "normal_val": split_counts["normal_val"],
        "normal_test": split_counts["normal_test"],
        "hcm_eval": split_counts["hcm_eval"],
        "vae_training_uses_hcm": False,
        "vae_validation_uses_hcm": False,
        "primary_normal_reference_for_hcm_comparison": "normal_test",
    }
    (
        args.results_dir / "phase1_split_summary.json"
    ).write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    preprocessing_summary = {
        "sampling_rate_hz": FS_HZ,
        "samples": 2000,
        "window_seconds": 4.0,
        "filter": {
            "type": "Butterworth band-pass",
            "order": FILTER_ORDER,
            "low_hz": LOW_HZ,
            "high_hz": HIGH_HZ,
            "zero_phase": True,
        },
        "scaling": {
            "type": "single global z-scaling",
            "estimated_from": "normal_train only",
            "mean_mv": train_mean,
            "std_mv": train_std,
            "per_ecg_normalization": False,
            "per_lead_normalization": False,
            "rationale": (
                "preserve relative voltage information and cross-lead "
                "amplitude relationships"
            ),
        },
    }
    (
        args.results_dir / "phase1_preprocessing_summary.json"
    ).write_text(
        json.dumps(preprocessing_summary, indent=2),
        encoding="utf-8"
    )

    print("\nWriting VAE-ready HDF5...")
    with h5py.File(args.output, "w") as h5:
        h5.attrs["description"] = (
            "SMASH-HCM Phase I normal-anchored preprocessed dataset"
        )
        h5.attrs["sampling_rate_hz"] = FS_HZ
        h5.attrs["samples_per_lead"] = 2000
        h5.attrs["window_seconds"] = 4.0
        h5.attrs["vae_training_population"] = "Normal only"
        h5.attrs["hcm_role"] = "Evaluation only"
        h5.attrs["normal_train_scaling_mean_mv"] = train_mean
        h5.attrs["normal_train_scaling_std_mv"] = train_std
        h5.attrs["bandpass_low_hz"] = LOW_HZ
        h5.attrs["bandpass_high_hz"] = HIGH_HZ
        h5.attrs["bandpass_order"] = FILTER_ORDER
        h5.attrs["random_seed"] = args.seed

        sg = h5.create_group("signals")
        sg.create_dataset(
            "ecg",
            data=x_scaled,
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            chunks=(1, 12, 2000),
        )
        write_str_dataset(sg, "lead_names", lead_names)

        lg = h5.create_group("labels")
        lg.create_dataset("hcm", data=labels, dtype="uint8")
        lg.create_dataset("split_code", data=split, dtype="uint8")

        mg = h5.create_group("metadata")
        write_str_dataset(mg, "source", source)
        write_str_dataset(mg, "subject_key", subject_key)
        write_str_dataset(mg, "ecg_id", ecg_id)
        write_str_dataset(mg, "patient_id", patient_id)
        mg.create_dataset("age_ptbxl_only", data=age, dtype="float32")
        write_str_dataset(mg, "sex_ptbxl_only", sex)

        pg = h5.create_group("preprocessing")
        pg.create_dataset(
            "global_mean_mv",
            data=np.asarray(train_mean, dtype=np.float64)
        )
        pg.create_dataset(
            "global_std_mv",
            data=np.asarray(train_std, dtype=np.float64)
        )

        split_name_values = [SPLIT_NAMES[int(v)] for v in split]
        write_str_dataset(lg, "split_name", split_name_values)

    print("\nDone.")
    print(f"Prepared HDF5 : {args.output}")
    print(f"Record QC     : {record_qc_path}")
    print(f"Lead QC       : {lead_qc_path}")
    print(f"QC figures    : {args.results_dir}")
    print()
    print("Normal-anchored design:")
    print("  VAE train      = normal_train only")
    print("  Model selection= normal_val only")
    print("  Normal reference evaluation = normal_test")
    print("  HCM            = hcm_eval only")


if __name__ == "__main__":
    main()
