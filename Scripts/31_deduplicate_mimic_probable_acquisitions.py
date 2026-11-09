#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31_deduplicate_mimic_probable_acquisitions.py

SMASH-HCM -- Step 31
Conservative, anchor-star, non-transitive deduplication of probable duplicate MIMIC-IV ECG
acquisitions identified in Step 30.

SCIENTIFIC DESIGN
-----------------
Step 30 defines:

    waveform near-identical:
        symmetric NRMSE <= 0.10 AND Pearson r >= 0.995

    probable duplicate acquisition:
        waveform near-identical AND acquisition-time separation <= 1 hour

This step removes records ONLY from the probable-duplicate acquisition set.
Waveform-near-identical ECGs separated by >1 hour are retained.

To avoid unsafe transitive chaining in non-clique similarity components, probable-
duplicate records are collapsed using an anchor-star rule. Within each connected
component, one representative anchor is selected deterministically. Only records
that are DIRECTLY linked to that anchor by a qualifying <=1-hour near-identical
edge are removed. Indirect similarity is never propagated. Thus A~B and B~C do
not imply A~C; C is removed with A only when A~C is itself a qualifying edge.

The retained anchor is selected deterministically using, in order:
    1. valid_for_projection=True
    2. QC quality: good > acceptable > limited > failed
    3. greater number of usable beats
    4. greater number of detected QRS complexes
    5. earlier acquisition time
    6. smaller record_id as a final deterministic tie-break

No averaging or waveform modification is performed. The retained ECG is an observed
record. All non-selected records, including invalid records and near-identical records
outside the <=1-hour duplicate window, are preserved.

DEFAULT INPUTS
--------------
Results/28_MIMIC_SMASH_MedianBeats/mimic_smash_median_beats.h5
Results/28_MIMIC_SMASH_MedianBeats/mimic_smash_median_beat_qc.csv
Results/30_MIMIC_NearDuplicateThresholdReview/mimic_near_duplicate_primary_pairs.csv

DEFAULT OUTPUTS
---------------
Results/31_MIMIC_ConservativeDedup/
    mimic_smash_median_beats_deduplicated.h5
    mimic_smash_median_beat_qc_deduplicated.csv
    mimic_smash_longitudinal_manifest_deduplicated.csv
    mimic_probable_duplicate_decisions.csv
    mimic_conservative_dedup_summary.json
    mimic_conservative_dedup_report.txt

The deduplicated HDF5 preserves Step-28 preprocessing and waveform values. Datasets
whose first dimension is the original record dimension are subset to retained rows;
longitudinal metadata fields are recomputed after record removal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


STEP = 31
STEP_NAME = "MIMIC_ConservativeDedup"
DAYS_PER_YEAR = 365.2425

QC_RANK = {
    "failed": 0,
    "limited": 1,
    "acceptable": 2,
    "good": 3,
}

LONGITUDINAL_COLS = [
    "record_index_sorted",
    "visit_number_all_records",
    "n_records_subject",
    "valid_visit_number",
    "n_valid_records_subject",
    "time_from_first_ecg_days",
    "time_from_first_ecg_years",
    "delta_from_previous_ecg_days",
    "time_from_first_valid_ecg_days",
    "time_from_first_valid_ecg_years",
    "delta_from_previous_valid_ecg_days",
    "has_acquisition_datetime",
]

COMPACT_LONGITUDINAL_COLS = [
    "subject_id",
    "record_id",
    "study_id_folder",
    "acquisition_datetime",
    "visit_number_all_records",
    "n_records_subject",
    "valid_for_projection",
    "valid_visit_number",
    "n_valid_records_subject",
    "time_from_first_ecg_days",
    "time_from_first_ecg_years",
    "delta_from_previous_ecg_days",
    "time_from_first_valid_ecg_days",
    "time_from_first_valid_ecg_years",
    "delta_from_previous_valid_ecg_days",
    "n_detected_qrs",
    "n_usable_beats",
    "qc_label",
    "source_relpath",
    "error",
]


def infer_project_root() -> Path:
    """Locate project root containing Data/ and Results/."""
    candidates: list[Path] = []

    try:
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    except NameError:
        pass

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if (p / "Data").exists() and (p / "Results").exists():
            return p

    for p in candidates:
        if (p / "Results").exists():
            return p

    raise RuntimeError("Cannot locate SMASH-HCM project root.")


def parse_args():
    root = infer_project_root()

    p = argparse.ArgumentParser(
        description="Conservatively deduplicate probable duplicate MIMIC ECG acquisitions."
    )
    p.add_argument(
        "--input-h5",
        type=Path,
        default=root / "Results" / "28_MIMIC_SMASH_MedianBeats" / "mimic_smash_median_beats.h5",
    )
    p.add_argument(
        "--input-qc",
        type=Path,
        default=root / "Results" / "28_MIMIC_SMASH_MedianBeats" / "mimic_smash_median_beat_qc.csv",
    )
    p.add_argument(
        "--pair-table",
        type=Path,
        default=root / "Results" / "30_MIMIC_NearDuplicateThresholdReview" / "mimic_near_duplicate_primary_pairs.csv",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=root / "Results" / f"{STEP:02d}_{STEP_NAME}",
    )
    return p.parse_args()


def to_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0", "yes"])
    )


def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["subject_id"] = d["subject_id"].astype(str).str.strip()
    d["record_id"] = d["record_id"].astype(str).str.strip()
    return d


def representative_score(row: pd.Series) -> tuple:
    valid = int(bool(row["valid_for_projection"]))
    qc = QC_RANK.get(str(row.get("qc_label", "")).strip().lower(), -1)

    usable = pd.to_numeric(pd.Series([row.get("n_usable_beats")]), errors="coerce").iloc[0]
    detected = pd.to_numeric(pd.Series([row.get("n_detected_qrs")]), errors="coerce").iloc[0]
    usable = int(usable) if np.isfinite(usable) else -1
    detected = int(detected) if np.isfinite(detected) else -1

    dt = pd.to_datetime(row.get("acquisition_datetime"), errors="coerce")
    # Larger score is better; negate timestamp so earlier is preferred.
    if pd.isna(dt):
        time_score = -10**30
    else:
        time_score = -int(dt.value)

    rid = str(row.get("record_id", ""))
    try:
        rid_score = -int(float(rid))
    except Exception:
        # Deterministic fallback for non-numeric record IDs.
        rid_score = 0

    return (valid, qc, usable, detected, time_score, rid_score)


def build_pair_decisions(qc: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Anchor-star, non-transitive decisions. Returns audit table and removal identities."""
    q = normalize_identity_columns(qc)
    q["valid_for_projection"] = to_bool(q["valid_for_projection"])

    key_cols = ["subject_id", "record_id"]
    if q.duplicated(key_cols).any():
        dup = q.loc[q.duplicated(key_cols, keep=False), key_cols].drop_duplicates()
        raise RuntimeError(
            "QC table contains duplicate subject_id/record_id identities:\n"
            + dup.head(20).to_string(index=False)
        )

    qidx = q.set_index(key_cols, drop=False)

    p = pairs.copy()
    if "probable_duplicate_acquisition" not in p.columns:
        raise KeyError("Pair table lacks probable_duplicate_acquisition.")
    p = p[to_bool(p["probable_duplicate_acquisition"])].copy()

    if p.empty:
        return pd.DataFrame(), set()

    for c in ["subject_id", "record_id_1", "record_id_2"]:
        p[c] = p[c].astype(str).str.strip()

    p["symmetric_nrmse"] = pd.to_numeric(p["symmetric_nrmse"], errors="coerce")
    p["delta_seconds"] = pd.to_numeric(p["delta_seconds"], errors="coerce")

    # Build one undirected graph per subject using ONLY qualifying probable-duplicate edges.
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = {}
    edge_rows: dict[frozenset, pd.Series] = {}
    for _, r in p.iterrows():
        k1 = (str(r["subject_id"]), str(r["record_id_1"]))
        k2 = (str(r["subject_id"]), str(r["record_id_2"]))
        if k1 not in qidx.index or k2 not in qidx.index:
            missing = [k for k in [k1, k2] if k not in qidx.index]
            raise KeyError(f"Pair-table record(s) missing from QC table: {missing}")
        adjacency.setdefault(k1, set()).add(k2)
        adjacency.setdefault(k2, set()).add(k1)
        edge_rows[frozenset((k1, k2))] = r

    group_of: dict[tuple[str, str], int] = {}
    anchor_of_group: dict[int, tuple[str, str]] = {}
    removed: set[tuple[str, str]] = set()
    group_id = 0

    # Process each subject separately. Within each remaining connected component,
    # select the best-quality representative and remove only its DIRECT neighbors.
    subjects = sorted({k[0] for k in adjacency})
    for sid in subjects:
        remaining = {k for k in adjacency if k[0] == sid}

        while True:
            active = {k for k in remaining if any(n in remaining for n in adjacency.get(k, set()))}
            if not active:
                break

            # Deterministic connected component starting from the smallest record id.
            seed = sorted(active, key=lambda x: x[1])[0]
            component = set()
            stack = [seed]
            while stack:
                cur = stack.pop()
                if cur in component or cur not in remaining:
                    continue
                component.add(cur)
                for nb in adjacency.get(cur, set()):
                    if nb in remaining and nb not in component:
                        stack.append(nb)

            # Primary criterion is representative quality. If tied, prefer the node
            # with more direct qualifying neighbors, then deterministic record id.
            def anchor_key(k):
                direct_degree = sum(nb in component for nb in adjacency.get(k, set()))
                return (representative_score(qidx.loc[k]), direct_degree, str(k[1]))

            anchor = max(component, key=anchor_key)
            direct_neighbors = sorted(
                [n for n in adjacency.get(anchor, set()) if n in component],
                key=lambda x: x[1],
            )

            group_id += 1
            anchor_of_group[group_id] = anchor
            group_members = [anchor] + direct_neighbors
            for k in group_members:
                group_of[k] = group_id
            for k in direct_neighbors:
                removed.add(k)

            # Remove this anchor-star group from further grouping. Any residual nodes
            # in the original component can still form separate direct-edge groups.
            remaining.difference_update(group_members)

    decisions: list[dict] = []
    for order, r in p.reset_index(drop=True).iterrows():
        k1 = (str(r["subject_id"]), str(r["record_id_1"]))
        k2 = (str(r["subject_id"]), str(r["record_id_2"]))
        g1 = group_of.get(k1)
        g2 = group_of.get(k2)

        base = {
            "pair_order": int(order + 1),
            "subject_id": str(r["subject_id"]),
            "record_id_1": str(r["record_id_1"]),
            "record_id_2": str(r["record_id_2"]),
            "acquisition_datetime_1": r.get("acquisition_datetime_1", ""),
            "acquisition_datetime_2": r.get("acquisition_datetime_2", ""),
            "delta_seconds": r.get("delta_seconds", np.nan),
            "symmetric_nrmse": r.get("symmetric_nrmse", np.nan),
            "pearson_r": r.get("pearson_r", np.nan),
            "max_lead_nrmse_audit": r.get("max_lead_nrmse", np.nan),
        }

        if g1 is not None and g1 == g2:
            anchor = anchor_of_group[g1]
            if k1 == anchor and k2 in removed:
                member = k2
            elif k2 == anchor and k1 in removed:
                member = k1
            else:
                decisions.append({
                    **base,
                    "group_id": int(g1),
                    "decision": "redundant_edge_within_anchor_group",
                    "retained_record_id": anchor[1],
                    "removed_record_id": "",
                    "anchor_record_id": anchor[1],
                    "selection_reason": "edge is internal to an already defined direct anchor-star group",
                })
                continue

            keep_row = qidx.loc[anchor]
            remove_row = qidx.loc[member]
            decisions.append({
                **base,
                "group_id": int(g1),
                "decision": "accepted_anchor_star_member",
                "retained_record_id": anchor[1],
                "removed_record_id": member[1],
                "anchor_record_id": anchor[1],
                "retained_qc_label": keep_row.get("qc_label", ""),
                "removed_qc_label": remove_row.get("qc_label", ""),
                "retained_n_usable_beats": keep_row.get("n_usable_beats", np.nan),
                "removed_n_usable_beats": remove_row.get("n_usable_beats", np.nan),
                "retained_n_detected_qrs": keep_row.get("n_detected_qrs", np.nan),
                "removed_n_detected_qrs": remove_row.get("n_detected_qrs", np.nan),
                "retained_acquisition_datetime": keep_row.get("acquisition_datetime", ""),
                "removed_acquisition_datetime": remove_row.get("acquisition_datetime", ""),
                "selection_reason": "direct qualifying edge to deterministic representative anchor",
            })
        else:
            decisions.append({
                **base,
                "group_id": np.nan,
                "decision": "not_collapsed_nontransitive_cross_group",
                "retained_record_id": "",
                "removed_record_id": "",
                "anchor_record_id": "",
                "selection_reason": "edge does not directly connect a retained anchor to one of its removed members",
            })

    return pd.DataFrame(decisions), removed


def recompute_longitudinal_fields(df: pd.DataFrame) -> pd.DataFrame:
    d = normalize_identity_columns(df)
    d["acquisition_datetime"] = pd.to_datetime(d["acquisition_datetime"], errors="coerce")
    d["valid_for_projection"] = to_bool(d["valid_for_projection"])

    # Preserve original HDF5 row index before re-sorting/resetting.
    if "record_index_sorted" not in d.columns:
        raise KeyError("QC table lacks record_index_sorted required to subset the HDF5.")
    d["_source_h5_index"] = pd.to_numeric(d["record_index_sorted"], errors="raise").astype(np.int64)

    d = d.sort_values(
        ["subject_id", "acquisition_datetime", "record_id"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)

    d["record_index_sorted"] = np.arange(len(d), dtype=np.int64)
    d["visit_number_all_records"] = (
        d.groupby("subject_id", dropna=False).cumcount().add(1).astype(np.int32)
    )
    d["n_records_subject"] = (
        d.groupby("subject_id", dropna=False)["record_id"].transform("count").astype(np.int32)
    )

    first_dt = d.groupby("subject_id", dropna=False)["acquisition_datetime"].transform("min")
    d["time_from_first_ecg_days"] = (
        d["acquisition_datetime"] - first_dt
    ).dt.total_seconds() / 86400.0
    d["time_from_first_ecg_years"] = d["time_from_first_ecg_days"] / DAYS_PER_YEAR
    d["delta_from_previous_ecg_days"] = (
        d.groupby("subject_id", dropna=False)["acquisition_datetime"]
        .diff()
        .dt.total_seconds()
        / 86400.0
    )

    d["n_valid_records_subject"] = (
        d["valid_for_projection"]
        .astype(int)
        .groupby(d["subject_id"], dropna=False)
        .transform("sum")
        .astype(np.int32)
    )

    d["valid_visit_number"] = pd.Series(pd.NA, index=d.index, dtype="Int64")
    valid = d["valid_for_projection"].astype(bool)
    d.loc[valid, "valid_visit_number"] = (
        d.loc[valid].groupby("subject_id", dropna=False).cumcount().add(1).astype("Int64")
    )

    d["time_from_first_valid_ecg_days"] = np.nan
    d["time_from_first_valid_ecg_years"] = np.nan
    d["delta_from_previous_valid_ecg_days"] = np.nan

    if valid.any():
        vd = d.loc[valid, ["subject_id", "acquisition_datetime"]].copy()
        first_valid = vd.groupby("subject_id", dropna=False)["acquisition_datetime"].transform("min")
        vd["time_from_first_valid_ecg_days"] = (
            vd["acquisition_datetime"] - first_valid
        ).dt.total_seconds() / 86400.0
        vd["time_from_first_valid_ecg_years"] = vd["time_from_first_valid_ecg_days"] / DAYS_PER_YEAR
        vd["delta_from_previous_valid_ecg_days"] = (
            vd.groupby("subject_id", dropna=False)["acquisition_datetime"]
            .diff()
            .dt.total_seconds()
            / 86400.0
        )
        for c in [
            "time_from_first_valid_ecg_days",
            "time_from_first_valid_ecg_years",
            "delta_from_previous_valid_ecg_days",
        ]:
            d.loc[vd.index, c] = vd[c]

    d["has_acquisition_datetime"] = d["acquisition_datetime"].notna()
    return d


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _series_for_h5(df: pd.DataFrame, name: str, src_dtype) -> np.ndarray:
    s = df[name]

    if name == "acquisition_datetime":
        return s.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("").astype(str).to_numpy(dtype=object)

    if pd.api.types.is_bool_dtype(s) or name == "has_acquisition_datetime":
        return s.fillna(False).astype(bool).to_numpy()

    if name == "valid_visit_number":
        return pd.to_numeric(s, errors="coerce").fillna(-1).to_numpy(dtype=np.float64)

    if h5py.check_string_dtype(src_dtype) is not None or src_dtype.kind in ("O", "S", "U"):
        return s.fillna("").astype(str).to_numpy(dtype=object)

    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    return arr


def export_deduplicated_h5(src_path: Path, dst_path: Path, df: pd.DataFrame) -> None:
    source_indices = df["_source_h5_index"].to_numpy(dtype=np.int64)

    with h5py.File(src_path, "r") as src:
        # Determine original record count from the primary signal dataset.
        if "signals/median_all_mv" not in src:
            raise KeyError("Input HDF5 lacks signals/median_all_mv.")
        n_original = int(src["signals/median_all_mv"].shape[0])

        if len(source_indices) and (source_indices.min() < 0 or source_indices.max() >= n_original):
            raise IndexError("Source HDF5 indices are outside the input record dimension.")

        with h5py.File(dst_path, "w") as dst:
            _copy_attrs(src, dst)
            dst.attrs["deduplication_step"] = STEP
            dst.attrs["deduplication_rule"] = (
                "probable duplicate acquisition = NRMSE<=0.10 AND r>=0.995 AND delta_t<=1h; "
                "anchor-star collapse: remove only direct qualifying neighbors of the retained anchor; no transitive chaining"
            )
            dst.attrs["n_records_before_dedup"] = n_original
            dst.attrs["n_records_after_dedup"] = int(len(df))
            dst.attrs["n_records_removed"] = int(n_original - len(df))

            def copy_group(src_group: h5py.Group, dst_group: h5py.Group, prefix: str = "") -> None:
                _copy_attrs(src_group, dst_group)
                for name, obj in src_group.items():
                    path = f"{prefix}/{name}" if prefix else name
                    if isinstance(obj, h5py.Group):
                        g = dst_group.create_group(name)
                        copy_group(obj, g, path)
                        continue

                    if not isinstance(obj, h5py.Dataset):
                        continue

                    # Recompute longitudinal metadata when represented in the Step-28 HDF5.
                    if path.startswith("metadata/"):
                        col = path.split("/", 1)[1]
                        if col in LONGITUDINAL_COLS and col in df.columns:
                            data = _series_for_h5(df, col, obj.dtype)
                            if h5py.check_string_dtype(obj.dtype) is not None:
                                dt = h5py.string_dtype("utf-8")
                                ds = dst_group.create_dataset(
                                    name,
                                    data=np.asarray(data, dtype=object),
                                    dtype=dt,
                                    compression="gzip",
                                    compression_opts=4,
                                )
                            else:
                                # Step 28 stores numeric longitudinal metadata as float64.
                                ds = dst_group.create_dataset(
                                    name,
                                    data=np.asarray(data),
                                    compression="gzip",
                                    compression_opts=4,
                                )
                            _copy_attrs(obj, ds)
                            continue

                    if obj.ndim >= 1 and obj.shape[0] == n_original:
                        data = obj[source_indices]
                    else:
                        data = obj[()]

                    kwargs = {}
                    if obj.ndim >= 1 and np.asarray(data).ndim >= 1:
                        # Preserve compression where possible; chunks are recalculated automatically.
                        if obj.compression is not None:
                            kwargs["compression"] = obj.compression
                            if obj.compression_opts is not None:
                                kwargs["compression_opts"] = obj.compression_opts
                        if obj.shuffle:
                            kwargs["shuffle"] = True

                    if h5py.check_string_dtype(obj.dtype) is not None:
                        dt = h5py.string_dtype("utf-8")
                        if np.asarray(data).ndim == 0:
                            val = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
                            ds = dst_group.create_dataset(name, data=val, dtype=dt)
                        else:
                            vals = [
                                x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)
                                for x in np.asarray(data)
                            ]
                            ds = dst_group.create_dataset(name, data=np.asarray(vals, dtype=object), dtype=dt, **kwargs)
                    else:
                        ds = dst_group.create_dataset(name, data=data, **kwargs)
                    _copy_attrs(obj, ds)

            copy_group(src, dst)


def main() -> None:
    args = parse_args()
    input_h5 = args.input_h5.resolve()
    input_qc = args.input_qc.resolve()
    pair_table = args.pair_table.resolve()
    out = args.results_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    for label, path in [
        ("Step-28 HDF5", input_h5),
        ("Step-28 QC CSV", input_qc),
        ("Step-30 primary pair table", pair_table),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    qc = pd.read_csv(input_qc)
    pairs = pd.read_csv(pair_table)

    required_qc = {
        "subject_id", "record_id", "record_index_sorted", "acquisition_datetime",
        "valid_for_projection", "qc_label", "n_usable_beats", "n_detected_qrs",
    }
    missing_qc = sorted(required_qc - set(qc.columns))
    if missing_qc:
        raise KeyError(f"QC table missing required columns: {missing_qc}")

    decisions, removed = build_pair_decisions(qc, pairs)

    qn = normalize_identity_columns(qc)
    remove_mask = pd.Series(
        [(sid, rid) in removed for sid, rid in zip(qn["subject_id"], qn["record_id"])],
        index=qn.index,
        dtype=bool,
    )

    if int(remove_mask.sum()) != len(removed):
        raise RuntimeError("Removal identity count does not match QC rows selected for removal.")

    dedup = qn.loc[~remove_mask].copy()
    dedup = recompute_longitudinal_fields(dedup)

    # Audit invariant: no accepted removed identity remains.
    retained_keys = set(zip(dedup["subject_id"].astype(str), dedup["record_id"].astype(str)))
    if removed & retained_keys:
        raise RuntimeError("At least one record marked for removal remains in deduplicated table.")

    decision_path = out / "mimic_probable_duplicate_decisions.csv"
    qc_path = out / "mimic_smash_median_beat_qc_deduplicated.csv"
    longitudinal_path = out / "mimic_smash_longitudinal_manifest_deduplicated.csv"
    h5_path = out / "mimic_smash_median_beats_deduplicated.h5"
    summary_path = out / "mimic_conservative_dedup_summary.json"
    report_path = out / "mimic_conservative_dedup_report.txt"

    decisions.to_csv(decision_path, index=False)

    qc_export = dedup.drop(columns=["_source_h5_index"], errors="ignore").copy()
    qc_export.to_csv(qc_path, index=False)

    qc_export[
        [c for c in COMPACT_LONGITUDINAL_COLS if c in qc_export.columns]
    ].to_csv(longitudinal_path, index=False)

    export_deduplicated_h5(input_h5, h5_path, dedup)

    accepted = decisions[decisions["decision"].eq("accepted_anchor_star_member")].copy() if len(decisions) else decisions
    noncollapsed = decisions[decisions["decision"].eq("not_collapsed_nontransitive_cross_group")].copy() if len(decisions) else decisions
    redundant = decisions[decisions["decision"].eq("redundant_edge_within_anchor_group")].copy() if len(decisions) else decisions

    valid_before = int(to_bool(qc["valid_for_projection"]).sum())
    valid_after = int(to_bool(qc_export["valid_for_projection"]).sum())

    subj_valid_counts = (
        qc_export.loc[to_bool(qc_export["valid_for_projection"])]
        .groupby("subject_id")["record_id"]
        .count()
    )

    summary = {
        "script": Path(__file__).name,
        "step": STEP,
        "input_h5": str(input_h5),
        "input_qc": str(input_qc),
        "pair_table": str(pair_table),
        "results_dir": str(out),
        "rule": {
            "waveform_near_identical": "symmetric_nrmse <= 0.10 and pearson_r >= 0.995",
            "probable_duplicate_acquisition": "waveform near-identical and delta_t <= 1 hour",
            "pair_processing": "anchor-star: within each component choose one representative; remove only records with a direct qualifying edge to that anchor",
            "transitive_chaining": False,
        },
        "representative_selection": [
            "valid_for_projection",
            "QC good > acceptable > limited > failed",
            "greater n_usable_beats",
            "greater n_detected_qrs",
            "earlier acquisition_datetime",
            "smaller record_id",
        ],
        "n_records_before": int(len(qc)),
        "n_records_after": int(len(qc_export)),
        "n_records_removed": int(len(qc) - len(qc_export)),
        "n_valid_records_before": valid_before,
        "n_valid_records_after": valid_after,
        "n_probable_duplicate_candidate_pairs": int(to_bool(pairs["probable_duplicate_acquisition"]).sum()),
        "n_anchor_star_removed_members": int(len(accepted)),
        "n_anchor_star_groups": int(accepted[["subject_id", "group_id"]].drop_duplicates().shape[0]) if len(accepted) else 0,
        "n_noncollapsed_cross_group_edges": int(len(noncollapsed)),
        "n_redundant_internal_edges": int(len(redundant)),
        "n_subjects_with_removed_record": int(accepted["subject_id"].nunique()) if len(accepted) else 0,
        "n_unique_subjects_after": int(qc_export["subject_id"].nunique()),
        "n_subjects_with_at_least_2_valid_records_after": int((subj_valid_counts >= 2).sum()),
        "n_subjects_with_at_least_3_valid_records_after": int((subj_valid_counts >= 3).sum()),
        "outputs": {
            "hdf5": str(h5_path),
            "qc_csv": str(qc_path),
            "longitudinal_csv": str(longitudinal_path),
            "decisions_csv": str(decision_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "SMASH-HCM Step 31 -- Conservative MIMIC probable-duplicate deduplication\n",
        f"Records before: {summary['n_records_before']:,}\n",
        f"Records after : {summary['n_records_after']:,}\n",
        f"Records removed: {summary['n_records_removed']:,}\n",
        f"Valid records before/after: {valid_before:,}/{valid_after:,}\n",
        f"Probable-duplicate candidate pairs: {summary['n_probable_duplicate_candidate_pairs']:,}\n",
        f"Anchor-star groups: {summary['n_anchor_star_groups']:,}\n",
        f"Records removed as direct anchor neighbors: {summary['n_anchor_star_removed_members']:,}\n",
        f"Non-collapsed cross-group edges: {summary['n_noncollapsed_cross_group_edges']:,}\n",
        f"Redundant internal edges: {summary['n_redundant_internal_edges']:,}\n",
        f"Subjects with at least one removed record: {summary['n_subjects_with_removed_record']:,}\n",
        f"Unique subjects after deduplication: {summary['n_unique_subjects_after']:,}\n",
        f"Subjects >=2 valid records after: {summary['n_subjects_with_at_least_2_valid_records_after']:,}\n",
        f"Subjects >=3 valid records after: {summary['n_subjects_with_at_least_3_valid_records_after']:,}\n",
        "\nRule: NRMSE<=0.10, Pearson r>=0.995, and <=1 hour.\n",
        "No transitive chaining is used: every removed record must itself have a direct qualifying edge to the retained anchor. Waveform-near-identical records outside the 1-hour window are retained.\n",
    ]

    if len(accepted):
        lines.append("\nAccepted anchor-star removals:\n")
        for _, r in accepted.iterrows():
            lines.append(
                f"  subject {r['subject_id']}: keep {r['retained_record_id']}, "
                f"remove {r['removed_record_id']}; NRMSE={float(r['symmetric_nrmse']):.6f}, "
                f"r={float(r['pearson_r']):.6f}, dt={float(r['delta_seconds']):.0f}s\n"
            )

    if len(noncollapsed):
        lines.append("\nNon-collapsed candidate edges retained by the non-transitive safeguard:\n")
        for _, r in noncollapsed.iterrows():
            lines.append(
                f"  subject {r['subject_id']}: {r['record_id_1']}--{r['record_id_2']}; "
                f"NRMSE={float(r['symmetric_nrmse']):.6f}\n"
            )

    if len(redundant):
        lines.append("\nRedundant internal edges within accepted anchor-star groups:\n")
        for _, r in redundant.iterrows():
            lines.append(
                f"  subject {r['subject_id']}: {r['record_id_1']}--{r['record_id_2']}; "
                f"NRMSE={float(r['symmetric_nrmse']):.6f}\n"
            )

    report_path.write_text("".join(lines), encoding="utf-8")

    print("".join(lines))
    print(f"HDF5       : {h5_path}")
    print(f"QC CSV     : {qc_path}")
    print(f"Longitudinal: {longitudinal_path}")
    print(f"Decisions  : {decision_path}")
    print(f"Summary    : {summary_path}")
    print(f"Report     : {report_path}")


if __name__ == "__main__":
    main()
