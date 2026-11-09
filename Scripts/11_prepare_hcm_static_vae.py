#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_prepare_hcm_static_vae.py  [v1.1]

SMASH-HCM -- Model 2
Prepare the HCM-static dataset for a disease-specific beta-VAE.

SCIENTIFIC ROLE
---------------
Model 1 defined a physiological ECG manifold using Normal-only training.

Model 2 asks a different question:

    What are the major continuous axes of electrical heterogeneity WITHIN HCM?

To avoid weighting subjects unequally, this static disease model is trained on exactly
one baseline ECG per HCM patient. In the current Phase-1 dataset, the HCM evaluation
cohort already contains one baseline ECG per retained HCM subject.

Design:
    - HCM only for train / validation / test
    - subject-level split: 70 / 15 / 15
    - global scaling estimated ONLY from HCM-train
    - no per-record or per-lead normalization
    - Normal-test retained only as an optional out-of-disease evaluation set
    - physical median-beat amplitudes are preserved in mV

Expected input
--------------
Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5

Output
------
Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5
Results/11_HCM_Static_VAE_Preparation/hcm_static_split_summary.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np


SEED = 20261004

SPLIT_HCM_TRAIN = 0
SPLIT_HCM_VAL = 1
SPLIT_HCM_TEST = 2
SPLIT_NORMAL_EVAL = 3

SPLIT_NAMES = {
    SPLIT_HCM_TRAIN: "hcm_train",
    SPLIT_HCM_VAL: "hcm_val",
    SPLIT_HCM_TEST: "hcm_test",
    SPLIT_NORMAL_EVAL: "normal_eval",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5"),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Results/11_HCM_Static_VAE_Preparation"),
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument(
        "--exclude-normal-eval",
        action="store_true",
        help="Do not append Phase-1 Normal-test records to the prepared HDF5.",
    )
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def decode_array(arr):
    a = np.asarray(arr)
    if a.dtype.kind in ("S", "O"):
        out = []
        for x in a.ravel():
            if isinstance(x, bytes):
                out.append(x.decode("utf-8", errors="replace"))
            else:
                out.append(str(x))
        return np.asarray(out, dtype=object).reshape(a.shape)
    return a


def inventory(h5: h5py.File) -> Dict[str, h5py.Dataset]:
    out = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj
    h5.visititems(visitor)
    return out


def find_dataset(inv, candidates: Sequence[str]) -> Optional[h5py.Dataset]:
    for c in candidates:
        if c in inv:
            return inv[c]

    by_base = {Path(k).name.lower(): v for k, v in inv.items()}
    for c in candidates:
        cc = Path(c).name.lower()
        if cc in by_base:
            return by_base[cc]

    for c in candidates:
        cl = c.lower()
        for name, ds in inv.items():
            if cl in name.lower():
                return ds
    return None


def standardize_waveform_shape(a):
    x = np.asarray(a, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3-D waveform array, got {x.shape}")
    if x.shape[1] == 12:
        return x
    if x.shape[2] == 12:
        return np.transpose(x, (0, 2, 1))
    raise ValueError(f"Cannot identify 12-lead axis in {x.shape}")


def per_record_metadata(inv, n):
    meta = {}
    for name, ds in inv.items():
        try:
            arr = decode_array(np.asarray(ds))
        except Exception:
            continue
        if arr.ndim == 1 and len(arr) == n:
            meta[Path(name).name] = arr
    return meta


def numeric_mask(arr, value):
    x = np.asarray(arr)
    try:
        num = x.astype(float)
    except Exception:
        return None
    return np.isfinite(num) & (num == float(value))


def derive_phase1_masks(meta, n):
    """
    Preferred Phase-1 convention:
        0 = normal_train
        1 = normal_val
        2 = normal_test
        3 = hcm_eval
    """

    # Exact numeric split is preferred.
    for c in ["split_code", "split", "partition_code"]:
        if c in meta:
            arr = np.asarray(meta[c])
            try:
                num = arr.astype(float)
            except Exception:
                continue
            uniq = set(np.unique(num[np.isfinite(num)]).tolist())
            if {0.0, 1.0, 2.0, 3.0}.issubset(uniq):
                return {
                    "normal_train": num == 0,
                    "normal_val": num == 1,
                    "normal_test": num == 2,
                    "hcm": num == 3,
                }

    # Explicit binary masks.
    def mask_from(names):
        for c in names:
            if c in meta:
                a = np.asarray(meta[c])
                try:
                    v = a.astype(float)
                    finite = v[np.isfinite(v)]
                    if set(np.unique(finite).tolist()).issubset({0.0, 1.0}):
                        return v.astype(bool)
                except Exception:
                    pass
        return None

    hcm = mask_from(["hcm_eval_mask", "hcm_mask", "is_hcm"])
    nt = mask_from(["normal_test_mask", "normal_test", "is_normal_test"])

    # Textual fallback.
    text_cols = [
        c for c in ["split", "set", "partition", "cohort", "source", "group"]
        if c in meta
    ]
    if text_cols:
        txt = np.asarray([""] * n, dtype=object)
        for c in text_cols:
            vals = np.asarray(meta[c]).astype(str)
            txt = np.char.add(np.char.add(txt.astype(str), " "), np.char.lower(vals))

        if hcm is None:
            hcm = np.array(["hcm" in s for s in txt], dtype=bool)
        if nt is None:
            nt = np.array([
                ("normal_test" in s) or ("normal test" in s) or
                ("test_normal" in s) or ("test normal" in s)
                for s in txt
            ], dtype=bool)

    if hcm is None:
        raise RuntimeError(
            "Could not identify the Phase-1 HCM baseline cohort. "
            "Expected split_code==3, hcm mask, or textual cohort metadata."
        )

    if nt is None:
        warnings.warn(
            "Could not identify Phase-1 Normal-test records. "
            "Normal evaluation will be omitted."
        )
        nt = np.zeros(n, dtype=bool)

    return {
        "normal_train": np.zeros(n, dtype=bool),
        "normal_val": np.zeros(n, dtype=bool),
        "normal_test": np.asarray(nt, dtype=bool),
        "hcm": np.asarray(hcm, dtype=bool),
    }


def choose_unique_subject_column(meta, hcm_indices):
    """
    Select a subject identifier only if it is unique within the already-defined
    HCM baseline cohort.

    Important SMASH-HCM constraint:
    `patient_id` alone is known to be unreliable in the source cohort. Therefore
    it is accepted only if it is demonstrably unique within the retained HCM
    baseline set. Prefer coherent/compound identifiers when available.
    """
    preferred = [
        "coherent_subject_id",
        "subject_key",
        "patient_key",
        "subject_id",
        "person_id",
        "compound_subject_id",
        "identity_key",
        "patient_id",
    ]

    diagnostics = []

    for c in preferred:
        if c not in meta:
            continue

        vals = np.asarray(meta[c])[hcm_indices].astype(str)
        unique, counts = np.unique(vals, return_counts=True)
        n_dup_ids = int(np.sum(counts > 1))
        n_duplicate_rows = int(np.sum(counts[counts > 1] - 1))

        diagnostics.append({
            "column": c,
            "n_unique": int(len(unique)),
            "n_duplicate_ids": n_dup_ids,
            "n_duplicate_rows": n_duplicate_rows,
        })

        if n_dup_ids == 0:
            return c, diagnostics

    return None, diagnostics


def subject_level_split(hcm_indices, meta, seed, train_frac, val_frac):
    """
    Split the HCM baseline cohort.

    The input cohort produced by Model 1 contains one retained baseline ECG per
    coherent HCM subject. If a unique coherent subject identifier is available,
    use it explicitly. If no such identifier is persisted in the HDF5, split the
    already-curated baseline rows directly rather than incorrectly collapsing on
    a non-unique `patient_id`.
    """
    subj_col, diagnostics = choose_unique_subject_column(meta, hcm_indices)

    if subj_col is not None:
        subjects = np.asarray(meta[subj_col])[hcm_indices].astype(str)
        split_basis = f"unique subject identifier: {subj_col}"
    else:
        # The Phase-1 HCM cohort is already baseline-only / one-record-per-coherent-subject.
        # Using row IDs here preserves all 649 retained HCM subjects and avoids the known
        # error of treating patient_id as a reliable identity key.
        subjects = np.asarray(
            [f"hcm_baseline_row_{i}" for i in hcm_indices],
            dtype=object,
        )
        split_basis = "curated HCM baseline rows (one row per coherent subject)"

        warnings.warn(
            "No persisted subject identifier is unique across the 649 HCM baseline "
            "records. The script will therefore split the already-curated baseline "
            "rows directly. Non-unique patient_id values are retained as metadata "
            "but are NOT used to define subject identity."
        )

    unique_subjects = np.unique(subjects)
    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)

    nsub = len(shuffled)
    n_train = int(round(train_frac * nsub))
    n_val = int(round(val_frac * nsub))
    n_train = min(n_train, nsub)
    n_val = min(n_val, nsub - n_train)

    train_subjects = set(shuffled[:n_train])
    val_subjects = set(shuffled[n_train:n_train+n_val])
    test_subjects = set(shuffled[n_train+n_val:])

    split = np.empty(len(hcm_indices), dtype=np.int8)
    for j, s in enumerate(subjects):
        if s in train_subjects:
            split[j] = SPLIT_HCM_TRAIN
        elif s in val_subjects:
            split[j] = SPLIT_HCM_VAL
        else:
            split[j] = SPLIT_HCM_TEST

    return split, subj_col, split_basis, diagnostics


def to_h5_compatible(a):
    arr = np.asarray(a)
    if arr.dtype.kind in ("U", "O"):
        return arr.astype(h5py.string_dtype(encoding="utf-8"))
    return arr


def main():
    args = parse_args()
    ensure_dir(args.output.parent)
    ensure_dir(args.results_dir)

    if args.train_frac <= 0 or args.val_frac < 0:
        raise ValueError("Invalid train/validation fractions.")
    if args.train_frac + args.val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1.")

    print("=" * 78)
    print("11 - Prepare HCM-static beta-VAE dataset")
    print("=" * 78)
    print("Input :", args.input)
    print("Output:", args.output)

    with h5py.File(args.input, "r") as h5:
        inv = inventory(h5)

        wave_ds = find_dataset(inv, [
            "median_all_mv",
            "median_beat_all_mv",
            "median_beats_all_mv",
            "median_all",
        ])
        if wave_ds is None:
            raise KeyError("Could not find median_all_mv-like physical waveform dataset.")

        x_mv_all = standardize_waveform_shape(wave_ds)
        n = len(x_mv_all)
        meta = per_record_metadata(inv, n)

    masks = derive_phase1_masks(meta, n)
    hcm_idx = np.where(masks["hcm"])[0]
    normal_test_idx = np.where(masks["normal_test"])[0]

    print(f"Phase-1 HCM baseline records : {len(hcm_idx)}")
    print(f"Phase-1 Normal-test records  : {len(normal_test_idx)}")

    if len(hcm_idx) < 100:
        raise RuntimeError("Unexpectedly small HCM baseline cohort.")

    if len(hcm_idx) != 649:
        warnings.warn(
            f"Expected approximately 649 retained HCM baseline ECGs, found {len(hcm_idx)}."
        )

    hcm_split, subj_col, split_basis, subject_id_diagnostics = subject_level_split(
        hcm_indices=hcm_idx,
        meta=meta,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    print("[subject identity]")
    if subject_id_diagnostics:
        for d in subject_id_diagnostics:
            print(
                f"  {d['column']}: unique={d['n_unique']}, "
                f"duplicate_ids={d['n_duplicate_ids']}, "
                f"duplicate_rows={d['n_duplicate_rows']}"
            )
    print("  split basis:", split_basis)

    train_local = np.where(hcm_split == SPLIT_HCM_TRAIN)[0]
    val_local = np.where(hcm_split == SPLIT_HCM_VAL)[0]
    test_local = np.where(hcm_split == SPLIT_HCM_TEST)[0]

    print("[HCM split]")
    print("  train:", len(train_local))
    print("  val  :", len(val_local))
    print("  test :", len(test_local))

    x_hcm_mv = x_mv_all[hcm_idx]

    # Global scaling from HCM-train only.
    train_values = x_hcm_mv[train_local]
    scale_mean = float(np.mean(train_values))
    scale_std = float(np.std(train_values))

    if not np.isfinite(scale_mean) or not np.isfinite(scale_std) or scale_std <= 0:
        raise RuntimeError("Invalid HCM-train global scaling statistics.")

    print(f"HCM-train global mean: {scale_mean:.12g} mV")
    print(f"HCM-train global std : {scale_std:.12g} mV")

    selected_idx = list(hcm_idx)
    split_codes = list(hcm_split.astype(np.int8))
    source_group = ["HCM"] * len(hcm_idx)

    if not args.exclude_normal_eval and len(normal_test_idx):
        selected_idx.extend(normal_test_idx.tolist())
        split_codes.extend([SPLIT_NORMAL_EVAL] * len(normal_test_idx))
        source_group.extend(["Normal"] * len(normal_test_idx))

    selected_idx = np.asarray(selected_idx, dtype=int)
    split_codes = np.asarray(split_codes, dtype=np.int8)
    source_group = np.asarray(source_group, dtype=object)

    x_mv = x_mv_all[selected_idx].astype(np.float32)
    x_scaled = ((x_mv - scale_mean) / scale_std).astype(np.float32)

    # HCM masks in prepared output.
    is_hcm = split_codes != SPLIT_NORMAL_EVAL

    with h5py.File(args.output, "w") as h5:
        h5.create_dataset(
            "median_all_mv",
            data=x_mv,
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset(
            "median_all_scaled",
            data=x_scaled,
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset("split_code", data=split_codes)
        h5.create_dataset(
            "split_name",
            data=np.asarray(
                [SPLIT_NAMES[int(s)] for s in split_codes],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        h5.create_dataset("is_hcm", data=is_hcm.astype(np.int8))
        h5.create_dataset("phase1_row_index", data=selected_idx.astype(np.int64))
        h5.create_dataset(
            "source_group",
            data=np.asarray(source_group, dtype=h5py.string_dtype("utf-8")),
        )

        # Copy per-record metadata for selected records.
        reserved = {
            "split_code", "split_name", "is_hcm",
            "phase1_row_index", "source_group",
            "median_all_mv", "median_all_scaled",
        }
        for key, arr in meta.items():
            if key in reserved:
                continue
            a = np.asarray(arr)
            if a.ndim != 1 or len(a) != n:
                continue
            try:
                h5.create_dataset(key, data=to_h5_compatible(a[selected_idx]))
            except Exception:
                # Non-essential metadata should never stop preparation.
                pass

        h5.attrs["description"] = (
            "SMASH-HCM Model 2 HCM-static beta-VAE dataset; one baseline ECG per HCM subject"
        )
        h5.attrs["source_file"] = str(args.input.resolve())
        h5.attrs["seed"] = int(args.seed)
        h5.attrs["sampling_rate_hz"] = 500.0
        h5.attrs["samples_per_lead"] = int(x_mv.shape[-1])
        h5.attrs["primary_representation"] = "median_all_mv"
        h5.attrs["hcm_train_scaling_mean_mv"] = scale_mean
        h5.attrs["hcm_train_scaling_std_mv"] = scale_std
        h5.attrs["scaling_population"] = "HCM-train only"
        h5.attrs["per_record_normalization"] = False
        h5.attrs["per_lead_normalization"] = False
        h5.attrs["normal_eval_role"] = "evaluation only; never used in training or model selection"
        h5.attrs["subject_id_column"] = "" if subj_col is None else subj_col
        h5.attrs["hcm_split_basis"] = split_basis
        h5.attrs["patient_id_used_as_identity_key"] = bool(subj_col == "patient_id")

    summary = {
        "script": "11_prepare_hcm_static_vae.py",
        "version": "1.1",
        "design": "HCM-static disease-manifold preparation",
        "source_file": str(args.input),
        "output_file": str(args.output),
        "seed": int(args.seed),
        "phase1_hcm_baseline_records": int(len(hcm_idx)),
        "phase1_normal_test_records": int(len(normal_test_idx)),
        "subject_identifier_used": subj_col,
        "split_basis": split_basis,
        "subject_identifier_diagnostics": subject_id_diagnostics,
        "patient_id_used_as_identity_key": bool(subj_col == "patient_id"),
        "split": {
            "hcm_train": int(np.sum(split_codes == SPLIT_HCM_TRAIN)),
            "hcm_val": int(np.sum(split_codes == SPLIT_HCM_VAL)),
            "hcm_test": int(np.sum(split_codes == SPLIT_HCM_TEST)),
            "normal_eval": int(np.sum(split_codes == SPLIT_NORMAL_EVAL)),
        },
        "train_fraction_requested": float(args.train_frac),
        "val_fraction_requested": float(args.val_frac),
        "hcm_train_scaling_mean_mv": scale_mean,
        "hcm_train_scaling_std_mv": scale_std,
        "normalization": (
            "one global affine transform estimated only from HCM-train; "
            "no per-record or per-lead amplitude normalization"
        ),
        "scientific_constraint": (
            "Only one baseline ECG per HCM patient is permitted in Model 2 training."
        ),
        "normal_eval_used_for_training": False,
        "normal_eval_used_for_model_selection": False,
    }

    summary_path = args.results_dir / "hcm_static_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("DONE")
    print("Prepared HDF5:", args.output)
    print("Summary      :", summary_path)


if __name__ == "__main__":
    main()
