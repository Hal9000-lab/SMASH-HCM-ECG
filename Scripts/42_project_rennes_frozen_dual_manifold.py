#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
42_project_rennes_frozen_dual_manifold.py

SMASH-HCM -- Step 42
Project the Rennes HCM ECGs (Data/Rennes/**/*.mat) through the two frozen VAEs:
Normal-anchored (16 latents) and HCM-oriented (8 latents).

No model is retrained, recalibrated or normalised on Rennes.

SIGNAL PATH (same functions as Step 28, imported, not rewritten)
----------------------------------------------------------------
1. read the 12-lead matrix; order assumed I, II, III, aVR, aVL, aVF, V1-V6
2. subtract each lead's median (the files carry a display offset per lead)
3. keep the first 4 s (4000 samples at the Rennes 1000 Hz) and resample to 500 Hz
4. 4th-order Butterworth 0.5-40 Hz, zero phase
5. multilead QRS detection on those 4 s
6. complete windows from -350 to +650 ms
7. pointwise median across usable beats; at least 2 beats are required

The VAE input is therefore one 12 x 500 median beat (1 s) built from a 4-s segment.
No per-record or per-lead amplitude normalisation is applied; amplitudes are taken
as mV. Scaling, encoders, reference covariances and metrics come from Step 33.

QC COLUMNS
----------
einthoven_r2_min     lead-relation check; low values question the lead order
median_qrs_p2p_max   largest peak-to-peak of the median beat (unit check)
repeated_sample_frac share of identical consecutive samples (digitised signal)
hr_full_record_bpm   rate from the whole record, a check on the sampling rate

Rennes files are 10-s records at 1000 Hz, as confirmed by the data owner; the
.mat files carry no sampling metadata, so the value is set here.

OUTPUT: Results/42_Rennes_FrozenDualManifold/
    rennes_latents_normal_vae.csv   one row per projected record, N_z_00..15
    rennes_latents_hcm_vae.csv      one row per projected record, H_z_00..07
plus the full table, median beats and report (any --fs other than 1000 goes to a
subfolder fs_<value>, so it cannot overwrite the reference results).

Usage:  python Scripts\\42_project_rennes_frozen_dual_manifold.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

N_LEADS = 12
RENNES_FS_HZ = 1000.0
HR_LOW, HR_HIGH = 45.0, 120.0
EINTHOVEN_MIN = 0.90
ID_COLUMNS = ["record_id", "subject_id", "source_relpath", "n_usable_beats", "qc_label"]


def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / "Data").exists() and (p / "Results").exists():
            return p
    raise RuntimeError("Cannot locate the project root (expected Data/ and Results/).")


def parse_args() -> argparse.Namespace:
    root = infer_project_root()
    p = argparse.ArgumentParser(description="Project Rennes ECGs through the frozen SMASH-HCM VAEs.")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--input-root", type=Path, default=root / "Data" / "Rennes")
    p.add_argument("--fs", type=float, default=RENNES_FS_HZ, help="Sampling rate of the .mat files.")
    p.add_argument("--variable", default="val", help="MATLAB variable holding the ECG.")
    p.add_argument("--step28", type=Path, default=root / "Scripts" / "28_prepare_mimic_smash_median_beats.py")
    p.add_argument("--step33", type=Path, default=root / "Scripts" / "33_project_mimic_frozen_dual_manifold.py")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if args.results_dir is None:
        base = root / "Results" / "42_Rennes_FrozenDualManifold"
        args.results_dir = base if args.fs == RENNES_FS_HZ else base / f"fs_{args.fs:g}"
    return args


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def to_leads_by_samples(a: np.ndarray) -> np.ndarray:
    a = np.squeeze(np.asarray(a, dtype=np.float64))
    if a.ndim != 2 or N_LEADS not in a.shape:
        raise ValueError(f"Expected a 2-D matrix with 12 leads; got shape {a.shape}.")
    if a.shape[0] == N_LEADS and a.shape[1] != N_LEADS:
        return a
    if a.shape[1] == N_LEADS:
        return a.T
    raise ValueError(f"Ambiguous matrix shape {a.shape}.")


def read_mat(path: Path, variable: str) -> np.ndarray:
    """Return [12, samples]. MATLAB v7.3 files are HDF5 and are read with h5py."""
    try:
        data = {k: v for k, v in sio.loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError:
        with h5py.File(path, "r") as h5:
            data = {k: np.asarray(h5[k]).T for k in h5.keys() if isinstance(h5[k], h5py.Dataset)}
    if variable in data:
        return to_leads_by_samples(data[variable])
    numeric = [k for k, v in data.items() if isinstance(v, np.ndarray) and v.dtype.kind in "fiu" and v.ndim == 2]
    if len(numeric) == 1:
        return to_leads_by_samples(data[numeric[0]])
    raise KeyError(f"Variable {variable!r} not found; numeric variables: {numeric}")


def einthoven_r2(x: np.ndarray) -> float:
    """Worst R2 of III = II - I, aVR = -(I+II)/2, aVL = I - II/2, aVF = II - I/2."""
    i, ii = x[0], x[1]
    worst = 1.0
    for lead, expected in ((x[2], ii - i), (x[3], -(i + ii) / 2), (x[4], i - ii / 2), (x[5], ii - i / 2)):
        var = float(np.var(lead))
        worst = min(worst, 1.0 - float(np.var(lead - expected)) / var if var > 0 else 0.0)
    return worst


def full_record_rate(x: np.ndarray, fs: float, s28) -> float:
    """Rate from the whole record at the assumed fs; rhythm check only."""
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    peaks = s28.detect_multilead_qrs(sosfiltfilt(sos, x, axis=-1), fs)[0]
    if len(peaks) < 2:
        return np.nan
    return 60.0 * fs / float(np.median(np.diff(peaks)))


def process_file(path: Path, root: Path, fs: float, variable: str, s28, sos) -> tuple[dict, np.ndarray]:
    beat = np.full((N_LEADS, s28.BEAT_SAMPLES), np.nan, dtype=np.float32)
    row = {"record_id": path.stem, "subject_id": path.stem, "source_relpath": str(path.relative_to(root)), "assumed_fs_hz": fs,
           "original_n_samples": np.nan, "record_seconds": np.nan, "n_detected_qrs": 0, "n_usable_beats": 0, "n_excluded_edge": 0,
           "qc_label": "failed", "qc_code": 0, "valid_for_projection": False, "einthoven_r2_min": np.nan, "median_qrs_p2p_max": np.nan,
           "repeated_sample_frac": np.nan, "hr_full_record_bpm": np.nan, "r_peaks_used_samples": "", "error": ""}
    try:
        x = read_mat(path, variable)
        if not np.isfinite(x).all():
            raise ValueError("Signal contains NaN/Inf.")
        x = x - np.median(x, axis=1, keepdims=True)
        row.update(original_n_samples=x.shape[1], record_seconds=x.shape[1] / fs,
                   repeated_sample_frac=float((np.diff(x, axis=1) == 0).mean()), hr_full_record_bpm=full_record_rate(x, fs, s28))
        x4, _ = s28.resample_first_4s(x, fs)
        filtered = s28.filter_ecg(x4, sos)
        row["einthoven_r2_min"] = einthoven_r2(filtered)
        detected = s28.detect_multilead_qrs(filtered, s28.TARGET_FS_HZ)[0]
        beats, used, edge, nonfinite = s28.extract_complete_beats(filtered, detected, s28.PRE_SAMPLES, s28.POST_SAMPLES)
        label, code = s28.qc_label_from_beats(len(beats))
        row.update(n_detected_qrs=len(detected), n_usable_beats=len(beats), n_excluded_edge=int(edge), qc_label=label, qc_code=code,
                   valid_for_projection=len(beats) >= s28.MIN_BEATS_FOR_PROJECTION,
                   r_peaks_used_samples=";".join(str(int(p)) for p in used))
        row.update(s28.rr_metrics(detected, s28.TARGET_FS_HZ))
        if len(beats):
            beat = np.median(beats, axis=0).astype(np.float32)
            row["median_qrs_p2p_max"] = float(np.ptp(beat, axis=1).max())
    except Exception as exc:  # one bad file is recorded and must not stop the cohort
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row, beat


def project(waves: np.ndarray, meta: pd.DataFrame, s33, root: Path, args) -> pd.DataFrame:
    import torch

    paths = s33.resolve_default_paths(root)
    frozen = argparse.Namespace(**paths)
    device = s33.choose_device(args.device)
    models = s33.load_frozen_models(frozen, device)
    lw_n, lw_h, _ = s33.fit_reference_gaussians(paths["normal_latents"], paths["hcm_latents"])
    valid = np.flatnonzero(meta["valid_for_projection"].to_numpy(bool) & np.isfinite(waves).all(axis=(1, 2)))
    if len(valid) == 0:
        raise RuntimeError("No record is eligible for projection.")
    x = waves[valid]
    with torch.inference_mode():
        n_mu, n_lv, n_rec = s33.encode_reconstruct(models["normal_model"], x, float(models["normal_mean_mv"]), float(models["normal_std_mv"]), args.batch_size, device)
        h_mu, h_lv, h_rec = s33.encode_reconstruct(models["hcm_model"], x, float(models["hcm_mean_mv"]), float(models["hcm_std_mv"]), args.batch_size, device)
    d_n, d_n2 = s33.mahalanobis_distance(n_mu, lw_n)
    d_h, d_h2 = s33.mahalanobis_distance(h_mu, lw_h)
    return s33.add_projection_columns(meta, valid, n_mu, n_lv, h_mu, h_lv, d_n, d_n2, d_h, d_h2,
                                      s33.compute_reconstruction_metrics(x, n_rec), s33.compute_reconstruction_metrics(x, h_rec))


def latent_table(table: pd.DataFrame, prefix: str, n_latents: int, extras: list[str]) -> pd.DataFrame:
    """Latents of one VAE for the projected records, with the identifying columns."""
    ok = table[table["projected_frozen_dual_manifold"].astype(bool)]
    return ok[[*ID_COLUMNS, *[f"{prefix}_z_{k:02d}" for k in range(n_latents)], *extras]].copy()


def build_report(table: pd.DataFrame, fs: float) -> str:
    ok = table[table["projected_frozen_dual_manifold"].astype(bool)]
    hr = table["hr_full_record_bpm"].median()
    lines = ["SMASH-HCM Step 42 -- Rennes frozen projection", "", f"Assumed sampling rate : {fs:g} Hz",
             f"Files                 : {len(table)}", f"Projected             : {len(ok)}",
             f"Read or signal errors : {(table['error'].astype(str) != '').sum()}",
             f"Usable beats          : {table['n_usable_beats'].value_counts().sort_index().to_dict()}",
             f"Record duration, s    : median {table['record_seconds'].median():.2f}",
             f"Rate, full record     : median {hr:.1f} bpm (p5 {table['hr_full_record_bpm'].quantile(.05):.1f}, p95 {table['hr_full_record_bpm'].quantile(.95):.1f})",
             f"QRS p2p max of beat   : median {table['median_qrs_p2p_max'].median():.2f}",
             f"Einthoven worst R2    : median {table['einthoven_r2_min'].median():.3f}; below {EINTHOVEN_MIN}: {(table['einthoven_r2_min'] < EINTHOVEN_MIN).sum()}",
             f"Repeated samples      : median {table['repeated_sample_frac'].median():.3f}"]
    if not HR_LOW <= hr <= HR_HIGH:
        lines += ["", f"WARNING: median rate {hr:.1f} bpm is outside {HR_LOW:g}-{HR_HIGH:g}: check --fs before using the latents."]
    if len(ok):
        lines += ["", "Reconstruction and distance (projected records)"]
        for col in ("RE_N_mse", "RE_H_mse", "d_N", "d_HCM"):
            lines.append(f"  {col:9s} median {ok[col].median():.4f}")
        lines += ["", "Latents: mean (SD)"]
        lines += [f"  {c}: {ok[c].mean():+.3f} ({ok[c].std():.3f})" for c in [f"H_z_{k:02d}" for k in range(8)]]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    files = sorted(p for p in args.input_root.rglob("*.mat"))[: args.limit]
    if not files:
        raise FileNotFoundError(f"No .mat files under {args.input_root}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Files: {len(files)}; assumed fs: {args.fs:g} Hz")

    s28 = load_module(args.step28.resolve(), "smash_step28")
    s33 = load_module(args.step33.resolve(), "smash_step33")
    sos = s28.make_filter()
    rows, beats = [], []
    for path in files:
        row, beat = process_file(path, args.input_root, args.fs, args.variable, s28, sos)
        rows.append(row)
        beats.append(beat)
    meta, waves = pd.DataFrame(rows), np.stack(beats)

    table = project(waves, meta, s33, root, args)
    table.to_csv(args.results_dir / "rennes_frozen_dual_manifold.csv", index=False)
    table[table["projected_frozen_dual_manifold"].astype(bool)].to_csv(args.results_dir / "rennes_frozen_dual_manifold_valid.csv", index=False)
    latent_table(table, "N", 16, ["d_N", "RE_N_mse"]).to_csv(args.results_dir / "rennes_latents_normal_vae.csv", index=False)
    latent_table(table, "H", 8, ["d_HCM", "RE_H_mse"]).to_csv(args.results_dir / "rennes_latents_hcm_vae.csv", index=False)
    with h5py.File(args.results_dir / "rennes_median_beats.h5", "w") as h5:
        h5.create_dataset("signals/median_all_mv", data=waves, compression="gzip", compression_opts=4)
        h5.create_dataset("metadata/record_id", data=np.asarray(meta["record_id"].astype(str), dtype=h5py.string_dtype("utf-8")))
        h5.attrs["assumed_fs_hz"] = args.fs
        h5.attrs["lead_order"] = ";".join(s28.LEAD_NAMES)
    report = build_report(table, args.fs)
    (args.results_dir / "rennes_projection_report.txt").write_text(report, encoding="utf-8")
    (args.results_dir / "run_summary.json").write_text(json.dumps({"assumed_fs_hz": args.fs, "n_files": len(files), "input_root": str(args.input_root)}, indent=2), encoding="utf-8")
    print(report)
    print(f"Saved to {args.results_dir}")


if __name__ == "__main__":
    main()
