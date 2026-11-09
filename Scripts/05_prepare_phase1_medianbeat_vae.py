#!/usr/bin/env python
"""
05_prepare_phase1_medianbeat_vae.py

SMASH-HCM -- Phase I
Prepare the median-beat dataset for a normal-anchored VAE.

Input
-----
Results/04_Phase1_Median_Beats/Phase1_MedianBeats_v1.h5

Primary morphology representation
---------------------------------
/signals/median_all_mv
    sample-wise median across all complete finite aligned beats.

Inclusion criterion
-------------------
n_usable_beats >= 2

This is a general QC criterion, not an ECG-ID-specific exclusion.
With the current dataset it excludes the single limited record
(ecg_id = 6246), leaving:

    5194 Normal
     649 HCM
    ----
    5843 total ECGs

Core design
-----------
1. Use median_all_mv as the primary VAE morphology input.
2. Do NOT apply a second band-pass filter:
   the median beats were constructed from the already filtered
   Phase-I 0.5--40 Hz signals.
3. Keep only records with >= 2 usable beats.
4. Preserve the original Normal train / validation / test split and
   HCM evaluation split.
5. Estimate ONE global mean and ONE global standard deviation using
   ONLY valid normal_train median beats.
6. Apply that same global scaling to every retained record.
7. Preserve relative amplitudes across leads:
   no per-ECG normalization and no per-lead normalization.
8. Preserve morphology-variability and rhythm descriptors for later
   phenotype analysis; these are NOT inputs to the morphology VAE.

Output
------
Results/05_Phase1_MedianBeat_VAE/Phase1_MedianBeat_NormalAnchored_preprocessed.h5

Results/05_Phase1_MedianBeat_VAE/
    medianbeat_record_qc.csv
    medianbeat_split_summary.json
    medianbeat_preprocessing_summary.json
    medianbeat_excluded_records.csv
    qc_medianbeat_global_rms.png
    qc_medianbeat_global_maxabs.png

VAE-ready signal
----------------
/signals/ecg
    shape [N, 12, 500]
    globally scaled median_all waveform.

Also preserved
--------------
/signals/median_all_mv
/signals/median_dominant_mv
/quality/*
/rhythm/*
/metadata/*
/labels/*

Requirements
------------
pip install numpy pandas h5py matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20261004
FS_HZ = 500.0

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

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


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
            out.append(
                x.decode(
                    "utf-8",
                    errors="replace",
                )
            )
        else:
            out.append(str(x))

    return np.asarray(
        out,
        dtype=object,
    )


def write_str_dataset(
    group: h5py.Group,
    name: str,
    values,
):
    dt = h5py.string_dtype(
        encoding="utf-8"
    )

    group.create_dataset(
        name,
        data=np.asarray(
            [str(x) for x in values],
            dtype=object,
        ),
        dtype=dt,
        compression="gzip",
        compression_opts=4,
    )


def copy_attrs(
    src_obj,
    dst_obj,
):
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def copy_group_subset(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    keep_mask: np.ndarray,
    n_records: int,
):
    """
    Recursively copy an HDF5 group.

    Any dataset whose first dimension equals the original record count is
    subset using keep_mask. Other datasets are copied unchanged.
    """
    copy_attrs(
        src_group,
        dst_group,
    )

    for name, obj in src_group.items():

        if isinstance(obj, h5py.Group):
            child = dst_group.create_group(name)

            copy_group_subset(
                obj,
                child,
                keep_mask,
                n_records,
            )

            continue

        if not isinstance(
            obj,
            h5py.Dataset,
        ):
            continue

        if obj.ndim >= 1 and obj.shape[0] == n_records:
            data = obj[:][keep_mask]
        else:
            data = obj[:]

        kwargs = {}

        if obj.ndim >= 1:
            kwargs["compression"] = "gzip"
            kwargs["compression_opts"] = 4

        if h5py.check_string_dtype(obj.dtype) is not None:
            dt = h5py.string_dtype(
                encoding="utf-8"
            )

            decoded = decode_str_array(
                data
            )

            dst_ds = dst_group.create_dataset(
                name,
                data=np.asarray(
                    decoded,
                    dtype=object,
                ),
                dtype=dt,
                **kwargs,
            )
        else:
            dst_ds = dst_group.create_dataset(
                name,
                data=data,
                dtype=obj.dtype,
                **kwargs,
            )

        copy_attrs(
            obj,
            dst_ds,
        )


def cohen_d(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    a = a[
        np.isfinite(a)
    ]

    b = b[
        np.isfinite(b)
    ]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    va = np.var(
        a,
        ddof=1,
    )

    vb = np.var(
        b,
        ddof=1,
    )

    denom = (
        len(a)
        + len(b)
        - 2
    )

    if denom <= 0:
        return np.nan

    pooled = np.sqrt(
        (
            (len(a) - 1) * va
            + (len(b) - 1) * vb
        )
        / denom
    )

    if pooled <= 0:
        return 0.0

    return float(
        (
            np.mean(b)
            - np.mean(a)
        )
        / pooled
    )


def get_required(
    h5: h5py.File,
    path: str,
):
    if path not in h5:
        raise KeyError(
            f"Required HDF5 dataset not found: {path}"
        )

    return h5[path][:]


def compute_record_qc(
    median_mv: np.ndarray,
    scaled: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    ecg_id: np.ndarray,
    patient_id: np.ndarray,
    subject_key: np.ndarray,
    source: np.ndarray,
    original_index: np.ndarray,
    n_usable: np.ndarray,
):
    rows = []

    for i in range(
        len(median_mv)
    ):
        x = median_mv[i]
        z = scaled[i]

        rows.append({
            "prepared_index":
                i,
            "original_record_index":
                int(
                    original_index[i]
                ),
            "ecg_id":
                ecg_id[i],
            "patient_id":
                patient_id[i],
            "subject_key":
                subject_key[i],
            "source":
                source[i],
            "class":
                (
                    "HCM"
                    if labels[i] == 1
                    else "Normal"
                ),
            "split":
                SPLIT_NAMES[
                    int(
                        split[i]
                    )
                ],
            "n_usable_beats":
                int(
                    n_usable[i]
                ),
            "medianbeat_mean_mv":
                float(
                    np.mean(x)
                ),
            "medianbeat_std_mv":
                float(
                    np.std(x)
                ),
            "medianbeat_rms_mv":
                float(
                    np.sqrt(
                        np.mean(
                            x ** 2
                        )
                    )
                ),
            "medianbeat_max_abs_mv":
                float(
                    np.max(
                        np.abs(x)
                    )
                ),
            "medianbeat_peak_to_peak_mv":
                float(
                    np.max(x)
                    - np.min(x)
                ),
            "scaled_mean":
                float(
                    np.mean(z)
                ),
            "scaled_std":
                float(
                    np.std(z)
                ),
            "scaled_rms":
                float(
                    np.sqrt(
                        np.mean(
                            z ** 2
                        )
                    )
                ),
            "has_nan":
                bool(
                    np.isnan(x).any()
                ),
            "has_inf":
                bool(
                    np.isinf(x).any()
                ),
        })

    return pd.DataFrame(
        rows
    )


def compute_lead_summary(
    median_mv: np.ndarray,
    labels: np.ndarray,
):
    rows = []

    normal = labels == 0
    hcm = labels == 1

    for lead_idx, lead in enumerate(
        LEAD_NAMES
    ):
        x = median_mv[
            :,
            lead_idx,
            :,
        ]

        rms = np.sqrt(
            np.mean(
                x ** 2,
                axis=1,
            )
        )

        maxabs = np.max(
            np.abs(x),
            axis=1,
        )

        p2p = np.ptp(
            x,
            axis=1,
        )

        for metric_name, metric in [
            ("rms_mv", rms),
            ("max_abs_mv", maxabs),
            ("peak_to_peak_mv", p2p),
        ]:
            rows.append({
                "lead":
                    lead,
                "metric":
                    metric_name,
                "normal_mean":
                    float(
                        np.mean(
                            metric[normal]
                        )
                    ),
                "normal_median":
                    float(
                        np.median(
                            metric[normal]
                        )
                    ),
                "hcm_mean":
                    (
                        float(
                            np.mean(
                                metric[hcm]
                            )
                        )
                        if np.any(hcm)
                        else np.nan
                    ),
                "hcm_median":
                    (
                        float(
                            np.median(
                                metric[hcm]
                            )
                        )
                        if np.any(hcm)
                        else np.nan
                    ),
                "cohen_d_hcm_minus_normal":
                    (
                        cohen_d(
                            metric[normal],
                            metric[hcm],
                        )
                        if np.any(hcm)
                        else np.nan
                    ),
            })

    return pd.DataFrame(
        rows
    )


def plot_qc(
    record_qc: pd.DataFrame,
    results_dir: Path,
):
    normal = (
        record_qc["class"]
        == "Normal"
    )

    hcm = (
        record_qc["class"]
        == "HCM"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        record_qc.loc[
            normal,
            "medianbeat_rms_mv",
        ],
        bins=60,
        density=True,
        alpha=0.55,
        label="Normal",
    )

    ax.hist(
        record_qc.loc[
            hcm,
            "medianbeat_rms_mv",
        ],
        bins=60,
        density=True,
        alpha=0.55,
        label="HCM",
    )

    ax.set_xlabel(
        "Median-beat global RMS (mV)"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Median-beat Phase I QC: global RMS"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        results_dir
        / "qc_medianbeat_global_rms.png",
        dpi=170,
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        record_qc.loc[
            normal,
            "medianbeat_max_abs_mv",
        ],
        bins=60,
        density=True,
        alpha=0.55,
        label="Normal",
    )

    ax.hist(
        record_qc.loc[
            hcm,
            "medianbeat_max_abs_mv",
        ],
        bins=60,
        density=True,
        alpha=0.55,
        label="HCM",
    )

    ax.set_xlabel(
        "Median-beat maximum absolute amplitude (mV)"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Median-beat Phase I QC: maximum amplitude"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        results_dir
        / "qc_medianbeat_global_maxabs.png",
        dpi=170,
    )

    plt.close(fig)


def main():
    root = infer_project_root()

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        type=Path,
        default=(
            root
            / "Results"
            / "04_Phase1_Median_Beats"
            / "Phase1_MedianBeats_v1.h5"
        ),
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "Results"
            / "05_Phase1_MedianBeat_VAE"
            / "Phase1_MedianBeat_NormalAnchored_preprocessed.h5"
        ),
    )

    ap.add_argument(
        "--results-dir",
        type=Path,
        default=(
            root
            / "Results"
            / "05_Phase1_MedianBeat_VAE"
        ),
    )

    ap.add_argument(
        "--min-usable-beats",
        type=int,
        default=2,
    )

    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input not found: {args.input}"
        )

    if args.min_usable_beats < 1:
        raise ValueError(
            "--min-usable-beats must be >= 1."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Input : {args.input}"
    )

    print(
        f"Output: {args.output}"
    )

    print(
        f"QC rule: n_usable_beats >= "
        f"{args.min_usable_beats}"
    )

    print()

    with h5py.File(
        args.input,
        "r",
    ) as h5:

        median_all = get_required(
            h5,
            "signals/median_all_mv",
        ).astype(
            np.float32
        )

        median_dominant = get_required(
            h5,
            "signals/median_dominant_mv",
        ).astype(
            np.float32
        )

        n_usable = get_required(
            h5,
            "quality/n_usable_beats",
        ).astype(
            np.int32
        )

        labels = get_required(
            h5,
            "labels/hcm",
        ).astype(
            np.uint8
        )

        split = get_required(
            h5,
            "labels/split_code",
        ).astype(
            np.uint8
        )

        ecg_id = decode_str_array(
            get_required(
                h5,
                "metadata/ecg_id",
            )
        )

        patient_id = decode_str_array(
            get_required(
                h5,
                "metadata/patient_id",
            )
        )

        subject_key = decode_str_array(
            get_required(
                h5,
                "metadata/subject_key",
            )
        )

        source = decode_str_array(
            get_required(
                h5,
                "metadata/source",
            )
        )

        original_n = len(
            median_all
        )

        if median_all.ndim != 3:
            raise ValueError(
                f"Expected median_all_mv to be 3-D; "
                f"found {median_all.shape}."
            )

        if median_all.shape[1] != 12:
            raise ValueError(
                f"Expected 12 leads; "
                f"found {median_all.shape}."
            )

        beat_samples = int(
            median_all.shape[2]
        )

        if beat_samples != 500:
            print(
                "WARNING: expected 500 samples per median beat, "
                f"found {beat_samples}."
            )

        if median_dominant.shape != median_all.shape:
            raise ValueError(
                "median_dominant_mv shape does not match median_all_mv."
            )

        for name, arr in [
            ("n_usable_beats", n_usable),
            ("labels/hcm", labels),
            ("labels/split_code", split),
            ("metadata/ecg_id", ecg_id),
            ("metadata/patient_id", patient_id),
            ("metadata/subject_key", subject_key),
            ("metadata/source", source),
        ]:
            if len(arr) != original_n:
                raise ValueError(
                    f"Length mismatch for {name}: "
                    f"{len(arr)} vs {original_n}"
                )

        finite_primary = np.isfinite(
            median_all
        ).all(
            axis=(1, 2)
        )

        finite_dominant = np.isfinite(
            median_dominant
        ).all(
            axis=(1, 2)
        )

        beat_count_ok = (
            n_usable
            >= args.min_usable_beats
        )

        keep = (
            beat_count_ok
            & finite_primary
            & finite_dominant
        )

        excluded = ~keep

        original_index = np.arange(
            original_n,
            dtype=np.int64,
        )

        excluded_rows = []

        for i in np.where(
            excluded
        )[0]:

            reasons = []

            if not beat_count_ok[i]:
                reasons.append(
                    "n_usable_beats_below_threshold"
                )

            if not finite_primary[i]:
                reasons.append(
                    "nonfinite_median_all"
                )

            if not finite_dominant[i]:
                reasons.append(
                    "nonfinite_median_dominant"
                )

            excluded_rows.append({
                "original_record_index":
                    int(i),
                "ecg_id":
                    ecg_id[i],
                "patient_id":
                    patient_id[i],
                "subject_key":
                    subject_key[i],
                "source":
                    source[i],
                "hcm":
                    int(
                        labels[i]
                    ),
                "split_code":
                    int(
                        split[i]
                    ),
                "split_name":
                    SPLIT_NAMES[
                        int(
                            split[i]
                        )
                    ],
                "n_usable_beats":
                    int(
                        n_usable[i]
                    ),
                "exclusion_reason":
                    ";".join(
                        reasons
                    ),
            })

        excluded_df = pd.DataFrame(
            excluded_rows
        )

        excluded_path = (
            args.results_dir
            / "medianbeat_excluded_records.csv"
        )

        excluded_df.to_csv(
            excluded_path,
            index=False,
        )

        print(
            f"Original records: "
            f"{original_n:,}"
        )

        print(
            f"Retained records: "
            f"{int(keep.sum()):,}"
        )

        print(
            f"Excluded records: "
            f"{int(excluded.sum()):,}"
        )

        if len(excluded_df) > 0:
            print()
            print(
                "Excluded ECG IDs:"
            )

            for _, row in excluded_df.iterrows():
                print(
                    f"  ecg_id={row['ecg_id']} | "
                    f"class="
                    f"{'HCM' if row['hcm'] == 1 else 'Normal'} | "
                    f"usable_beats={row['n_usable_beats']} | "
                    f"reason={row['exclusion_reason']}"
                )

        # Subset.
        median_all_kept = (
            median_all[keep]
        )

        median_dominant_kept = (
            median_dominant[keep]
        )

        labels_kept = (
            labels[keep]
        )

        split_kept = (
            split[keep]
        )

        n_usable_kept = (
            n_usable[keep]
        )

        ecg_id_kept = (
            ecg_id[keep]
        )

        patient_id_kept = (
            patient_id[keep]
        )

        subject_key_kept = (
            subject_key[keep]
        )

        source_kept = (
            source[keep]
        )

        original_index_kept = (
            original_index[keep]
        )

        # Strong split consistency.
        train_idx = np.where(
            split_kept
            == SPLIT_NORMAL_TRAIN
        )[0]

        val_idx = np.where(
            split_kept
            == SPLIT_NORMAL_VAL
        )[0]

        test_idx = np.where(
            split_kept
            == SPLIT_NORMAL_TEST
        )[0]

        hcm_idx = np.where(
            split_kept
            == SPLIT_HCM_EVAL
        )[0]

        if not np.all(
            labels_kept[
                train_idx
            ]
            == 0
        ):
            raise AssertionError(
                "normal_train contains HCM records."
            )

        if not np.all(
            labels_kept[
                val_idx
            ]
            == 0
        ):
            raise AssertionError(
                "normal_val contains HCM records."
            )

        if not np.all(
            labels_kept[
                test_idx
            ]
            == 0
        ):
            raise AssertionError(
                "normal_test contains HCM records."
            )

        if not np.all(
            labels_kept[
                hcm_idx
            ]
            == 1
        ):
            raise AssertionError(
                "hcm_eval contains non-HCM records."
            )

        print()
        print(
            "Retained split:"
        )

        print(
            f"  normal_train : "
            f"{len(train_idx):,}"
        )

        print(
            f"  normal_val   : "
            f"{len(val_idx):,}"
        )

        print(
            f"  normal_test  : "
            f"{len(test_idx):,}"
        )

        print(
            f"  hcm_eval     : "
            f"{len(hcm_idx):,}"
        )

        # New scaling: ONLY the retained median beats of normal_train.
        train_values = (
            median_all_kept[
                train_idx
            ]
            .astype(
                np.float64
            )
        )

        train_mean = float(
            np.mean(
                train_values
            )
        )

        train_std = float(
            np.std(
                train_values
            )
        )

        if (
            not np.isfinite(
                train_mean
            )
            or not np.isfinite(
                train_std
            )
            or train_std <= 0
        ):
            raise ValueError(
                "Invalid median-beat normal-train scaling."
            )

        print()
        print(
            "Median-beat Normal-train global scaling:"
        )

        print(
            f"  mean = "
            f"{train_mean:.10f} mV"
        )

        print(
            f"  std  = "
            f"{train_std:.10f} mV"
        )

        scaled = (
            (
                median_all_kept
                - train_mean
            )
            / train_std
        ).astype(
            np.float32
        )

        if not np.isfinite(
            scaled
        ).all():
            raise ValueError(
                "Scaled median-beat dataset contains NaN/Inf."
            )

        # QC tables.
        record_qc = compute_record_qc(
            median_all_kept,
            scaled,
            labels_kept,
            split_kept,
            ecg_id_kept,
            patient_id_kept,
            subject_key_kept,
            source_kept,
            original_index_kept,
            n_usable_kept,
        )

        record_qc_path = (
            args.results_dir
            / "medianbeat_record_qc.csv"
        )

        record_qc.to_csv(
            record_qc_path,
            index=False,
        )

        lead_summary = compute_lead_summary(
            median_all_kept,
            labels_kept,
        )

        lead_summary_path = (
            args.results_dir
            / "medianbeat_lead_qc_summary.csv"
        )

        lead_summary.to_csv(
            lead_summary_path,
            index=False,
        )

        plot_qc(
            record_qc,
            args.results_dir,
        )

        split_counts = {
            SPLIT_NAMES[k]:
                int(
                    np.sum(
                        split_kept
                        == k
                    )
                )
            for k in sorted(
                SPLIT_NAMES
            )
        }

        split_summary = {
            "source_file":
                str(
                    args.input
                ),
            "original_records":
                int(
                    original_n
                ),
            "retained_records":
                int(
                    keep.sum()
                ),
            "excluded_records":
                int(
                    excluded.sum()
                ),
            "inclusion_rule":
                (
                    "n_usable_beats >= "
                    f"{args.min_usable_beats} "
                    "and finite median_all/dominant waveforms"
                ),
            "normal_total_retained":
                int(
                    np.sum(
                        labels_kept
                        == 0
                    )
                ),
            "hcm_total_retained":
                int(
                    np.sum(
                        labels_kept
                        == 1
                    )
                ),
            "normal_train":
                split_counts[
                    "normal_train"
                ],
            "normal_val":
                split_counts[
                    "normal_val"
                ],
            "normal_test":
                split_counts[
                    "normal_test"
                ],
            "hcm_eval":
                split_counts[
                    "hcm_eval"
                ],
            "vae_training_uses_hcm":
                False,
            "vae_validation_uses_hcm":
                False,
            "primary_normal_reference_for_hcm_comparison":
                "normal_test",
            "excluded_ecg_ids":
                excluded_df[
                    "ecg_id"
                ].astype(
                    str
                ).tolist()
                if len(
                    excluded_df
                ) > 0
                else [],
        }

        split_summary_path = (
            args.results_dir
            / "medianbeat_split_summary.json"
        )

        split_summary_path.write_text(
            json.dumps(
                split_summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        preprocessing_summary = {
            "sampling_rate_hz":
                FS_HZ,
            "samples_per_lead":
                beat_samples,
            "window_seconds":
                float(
                    beat_samples
                    / FS_HZ
                ),
            "beat_representation":
                "median_all_mv",
            "beat_window_definition":
                (
                    "inherited from Phase1_MedianBeats_v1.h5; "
                    "nominally -350 ms to +650 ms around common "
                    "multilead R/QRS fiducial"
                ),
            "filter": {
                "applied_in_this_step":
                    False,
                "inherited":
                    True,
                "description":
                    (
                        "Median beats were constructed from the "
                        "already filtered 0.5-40 Hz zero-phase "
                        "Phase-I ECG signals."
                    ),
            },
            "inclusion": {
                "minimum_usable_beats":
                    int(
                        args.min_usable_beats
                    ),
                "morphologically_different_beats_removed":
                    False,
            },
            "scaling": {
                "type":
                    "single global z-scaling",
                "estimated_from":
                    (
                        "retained normal_train median_all_mv "
                        "only"
                    ),
                "mean_mv":
                    train_mean,
                "std_mv":
                    train_std,
                "per_ecg_normalization":
                    False,
                "per_lead_normalization":
                    False,
                "rationale":
                    (
                        "preserve absolute and cross-lead "
                        "voltage relationships while defining "
                        "normality exclusively from the "
                        "normal training population"
                    ),
            },
            "auxiliary_information_not_used_as_vae_input": [
                "median_dominant_mv",
                "beat-to-beat morphology variability metrics",
                "dominant morphology fraction",
                "RR metrics",
                "heart-rate metrics",
            ],
        }

        preprocessing_summary_path = (
            args.results_dir
            / "medianbeat_preprocessing_summary.json"
        )

        preprocessing_summary_path.write_text(
            json.dumps(
                preprocessing_summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Write output.
        with h5py.File(
            args.output,
            "w",
        ) as out:

            out.attrs[
                "description"
            ] = (
                "SMASH-HCM Phase I median-beat "
                "normal-anchored VAE-ready dataset"
            )

            out.attrs[
                "source_file"
            ] = str(
                args.input
            )

            out.attrs[
                "sampling_rate_hz"
            ] = FS_HZ

            out.attrs[
                "samples_per_lead"
            ] = beat_samples

            out.attrs[
                "window_seconds"
            ] = float(
                beat_samples
                / FS_HZ
            )

            out.attrs[
                "primary_morphology_representation"
            ] = "median_all_mv"

            out.attrs[
                "minimum_usable_beats"
            ] = int(
                args.min_usable_beats
            )

            out.attrs[
                "vae_training_population"
            ] = "Normal only"

            out.attrs[
                "hcm_role"
            ] = "Evaluation only"

            out.attrs[
                "normal_train_scaling_mean_mv"
            ] = train_mean

            out.attrs[
                "normal_train_scaling_std_mv"
            ] = train_std

            out.attrs[
                "filtering_applied_in_this_step"
            ] = False

            out.attrs[
                "filtering_inherited_from_phase1"
            ] = True

            out.attrs[
                "morphologically_different_beats_removed"
            ] = False

            # Primary signals.
            sg = out.create_group(
                "signals"
            )

            sg.create_dataset(
                "ecg",
                data=scaled,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
                chunks=(
                    1,
                    12,
                    beat_samples,
                ),
            )

            sg.create_dataset(
                "median_all_mv",
                data=median_all_kept,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
                chunks=(
                    1,
                    12,
                    beat_samples,
                ),
            )

            sg.create_dataset(
                "median_dominant_mv",
                data=median_dominant_kept,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
                chunks=(
                    1,
                    12,
                    beat_samples,
                ),
            )

            write_str_dataset(
                sg,
                "lead_names",
                LEAD_NAMES,
            )

            # Preserve metadata, labels, quality and rhythm for retained rows.
            for group_name in [
                "metadata",
                "labels",
                "quality",
                "rhythm",
            ]:
                if group_name in h5:
                    dst_group = out.create_group(
                        group_name
                    )

                    copy_group_subset(
                        h5[
                            group_name
                        ],
                        dst_group,
                        keep,
                        original_n,
                    )

            # Add original record mapping explicitly.
            if "metadata" not in out:
                mg = out.create_group(
                    "metadata"
                )
            else:
                mg = out[
                    "metadata"
                ]

            if "original_record_index" in mg:
                del mg[
                    "original_record_index"
                ]

            mg.create_dataset(
                "original_record_index",
                data=original_index_kept,
                dtype="int64",
            )

            # Add preprocessing information.
            pg = out.create_group(
                "preprocessing"
            )

            pg.create_dataset(
                "global_mean_mv",
                data=np.asarray(
                    train_mean,
                    dtype=np.float64,
                ),
            )

            pg.create_dataset(
                "global_std_mv",
                data=np.asarray(
                    train_std,
                    dtype=np.float64,
                ),
            )

            pg.attrs[
                "estimated_from"
            ] = (
                "retained normal_train "
                "median_all_mv only"
            )

            pg.attrs[
                "per_ecg_normalization"
            ] = False

            pg.attrs[
                "per_lead_normalization"
            ] = False

            pg.attrs[
                "second_bandpass_applied"
            ] = False

    print()
    print(
        "Median-beat VAE preparation complete."
    )

    print(
        f"Prepared HDF5 : "
        f"{args.output}"
    )

    print(
        f"Record QC     : "
        f"{record_qc_path}"
    )

    print(
        f"Lead QC       : "
        f"{lead_summary_path}"
    )

    print(
        f"Excluded      : "
        f"{excluded_path}"
    )

    print(
        f"Split summary : "
        f"{split_summary_path}"
    )

    print(
        f"Preprocessing : "
        f"{preprocessing_summary_path}"
    )

    print()
    print(
        "Normal-anchored design:"
    )

    print(
        "  VAE input       = median_all waveform"
    )

    print(
        "  VAE train       = normal_train only"
    )

    print(
        "  Model selection = normal_val only"
    )

    print(
        "  Normal reference= normal_test"
    )

    print(
        "  HCM             = hcm_eval only"
    )

    print(
        "  Rhythm/morph variability = preserved as auxiliary descriptors"
    )


if __name__ == "__main__":
    main()
