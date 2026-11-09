#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20b_interpret_longitudinal_rate_vae.py

SMASH-HCM -- Step 20b
Physiological interpretation of the longitudinal RATE-VAE sensitivity modes.

SCIENTIFIC QUESTION
-------------------
What annualized ECG morphological change patterns are represented by each longitudinal latent
coordinate D_z_00 ... D_z_K?

CRITICAL POINT
--------------
The RATE-VAE decoder represents an annualized CHANGE-RATE waveform (Delta X / Delta t), not an ECG.
For interpretation, decoded rates are converted to a change over a specified interval and added to real HCM reference beats; classical ECG features are never measured directly on the rate/difference waveform.

This script uses two complementary analyses:

1) REAL HCM PAIRS
   - extract classical ECG features separately from X(t1) and X(t2)
   - compute Delta feature = feature(t2) - feature(t1)
   - associate D_z_k with Delta feature on train/validation/test pairs

2) DECODER TRAVERSALS
   - hold Delta t fixed
   - vary one D_z_k across -3...+3 train SD
   - decode Delta X
   - add the decoded change to real HCM reference beats
   - measure feature changes relative to the z=0 decoded trajectory
   - repeat across multiple reference HCM beats and aggregate robustly

A third analysis explicitly quantifies the decoder dependence on Delta t by
decoding the same latent state at several time intervals.

The script reuses the exact ECG feature extractor from Step 14.

Default inputs
--------------
Results/16_Longitudinal_DualManifold/Phase3_HCM_Longitudinal_DualManifold.h5
Results/19_Longitudinal_DeltaVAE_rate/pair_metadata.csv
Results/19_Longitudinal_DeltaVAE_rate/delta_vae_latents.csv
Results/19_Longitudinal_DeltaVAE_rate/best_model.pt
Scripts/14_quantify_hcm_static_latent_ecg_features.py
Scripts/19_train_longitudinal_delta_vae.py

Outputs
-------
Results/20b_Longitudinal_RateVAE_Interpretation/
    visit_ecg_features.csv
    pair_delta_ecg_features.csv
    latent_delta_feature_associations_{train,val,test,all}.csv
    decoder_traversal_feature_changes.csv
    decoder_traversal_feature_slopes.csv
    test_real_vs_traversal_concordance.csv
    latent_interpretation_priority_summary.csv
    latent_time_conditioning.csv
    run_summary.json
    figures/
        test_latent_delta_feature_heatmap.png
        real_vs_traversal_concordance.png
        latent_time_conditioning.png

Run
---
python Scripts\\20b_interpret_longitudinal_rate_vae.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch


SEED = 20261005
FS_HZ = 500.0
BEAT_START_MS = -350.0
N_LEADS = 12
N_SAMPLES = 500

LEVELS_SD = np.asarray([-3, -2, -1, 0, 1, 2, 3], dtype=float)

EXCLUDE_FEATURES = {
    "qrs_on_ms",
    "qrs_off_ms",
    "QRS_axis_deg",
    "RR_ms",
    "HR_bpm",
    "QTpeakc_Bazett_ms",
    "QTpeakc_Fridericia_ms",
    "QTc_Bazett_ms_est",
    "QTc_Fridericia_ms_est",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Interpret HCM-only longitudinal RATE-VAE latent modes."
    )
    p.add_argument(
        "--longitudinal-h5",
        type=Path,
        default=Path(
            "Results/16_Longitudinal_DualManifold/Phase3_HCM_Longitudinal_DualManifold.h5"
        ),
    )
    p.add_argument(
        "--pair-metadata",
        type=Path,
        default=Path(
            "Results/19_Longitudinal_DeltaVAE_rate/pair_metadata.csv"
        ),
    )
    p.add_argument(
        "--latents",
        type=Path,
        default=Path(
            "Results/19_Longitudinal_DeltaVAE_rate/delta_vae_latents.csv"
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "Results/19_Longitudinal_DeltaVAE_rate/best_model.pt"
        ),
    )
    p.add_argument(
        "--feature-module",
        type=Path,
        default=Path(
            "Scripts/14_quantify_hcm_static_latent_ecg_features.py"
        ),
    )
    p.add_argument(
        "--delta-vae-module",
        type=Path,
        default=Path(
            "Scripts/19_train_longitudinal_delta_vae.py"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Results/20b_Longitudinal_RateVAE_Interpretation"
        ),
    )
    p.add_argument(
        "--n-reference-beats",
        type=int,
        default=100,
        help="Number of HCM train t1 beats used for decoder traversal interpretation.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
    )
    p.add_argument(
        "--time-grid-years",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 5.0],
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    m = np.isfinite(p)
    if not np.any(m):
        return q
    pf = p[m]
    order = np.argsort(pf)
    ranked = pf[order]
    n = len(ranked)
    qr = ranked * n / np.arange(1, n + 1)
    qr = np.minimum.accumulate(qr[::-1])[::-1]
    qr = np.clip(qr, 0.0, 1.0)
    tmp = np.empty(n, dtype=float)
    tmp[order] = qr
    q[m] = tmp
    return q


def robust_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(m))
    if n < 10:
        return np.nan, np.nan, n
    xx = x[m]
    yy = y[m]
    if np.nanstd(xx) <= 1e-12 or np.nanstd(yy) <= 1e-12:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(xx, yy)
    return float(rho), float(p), n


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def z_columns(df):
    cols = [
        c for c in df.columns
        if c.startswith("D_z_") and c[4:].isdigit()
    ]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def load_inputs(args):
    for p in [
        args.longitudinal_h5,
        args.pair_metadata,
        args.latents,
        args.checkpoint,
        args.feature_module,
        args.delta_vae_module,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    pair = pd.read_csv(args.pair_metadata)
    lat = pd.read_csv(args.latents)

    if "pair_id" not in pair.columns or "pair_id" not in lat.columns:
        raise KeyError("pair_metadata and latent CSV must contain pair_id.")

    merged = pair.merge(
        lat,
        on="pair_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_latent"),
    )

    if len(merged) != len(pair):
        raise RuntimeError(
            f"Pair/latent merge lost rows: pair={len(pair)}, merged={len(merged)}."
        )

    zcols = z_columns(merged)
    if not zcols:
        raise RuntimeError("No D_z_XX columns found.")

    with h5py.File(args.longitudinal_h5, "r") as h5:
        if "signals/median_all_mv" not in h5:
            raise KeyError("Missing signals/median_all_mv.")
        beats = np.asarray(
            h5["signals/median_all_mv"][:],
            dtype=np.float32,
        )

    if beats.ndim != 3 or beats.shape[1:] != (N_LEADS, N_SAMPLES):
        raise ValueError(f"Unexpected mean-beat shape: {beats.shape}")

    return merged, zcols, beats


def extract_visit_features(
    pair_df,
    beats,
    feature_mod,
    fs,
    beat_start_ms,
):
    unique_rows = np.unique(
        np.concatenate([
            pd.to_numeric(pair_df["row_index_t1"], errors="raise").to_numpy(int),
            pd.to_numeric(pair_df["row_index_t2"], errors="raise").to_numpy(int),
        ])
    )

    rows = []
    qc_rows = []

    for k, idx in enumerate(unique_rows):
        beat = beats[idx]
        try:
            feats, qc = feature_mod.extract_ecg_features(
                beat,
                fs=fs,
                beat_start_ms=beat_start_ms,
                rr_ms=np.nan,
            )
            success = True
        except Exception as exc:
            feats = {}
            qc = {"error": repr(exc)}
            success = False

        row = {
            "row_index": int(idx),
            "feature_extraction_success": bool(success),
        }
        row.update(feats)
        rows.append(row)

        qrow = {
            "row_index": int(idx),
            "feature_extraction_success": bool(success),
        }
        qrow.update(qc)
        qc_rows.append(qrow)

        if (k + 1) % 250 == 0 or k == len(unique_rows) - 1:
            print(f"[visit features] {k+1}/{len(unique_rows)}")

    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def select_feature_columns(visit_features):
    cols = []
    non_features = {"row_index", "feature_extraction_success"}

    for c in visit_features.columns:
        if c in non_features or c in EXCLUDE_FEATURES:
            continue
        vals = pd.to_numeric(visit_features[c], errors="coerce").to_numpy(float)
        if np.sum(np.isfinite(vals)) < 30:
            continue
        if np.nanstd(vals) <= 1e-10:
            continue
        cols.append(c)

    return cols


def build_pair_delta_feature_table(pair_df, visit_features, feature_cols):
    f1 = visit_features[["row_index"] + feature_cols].copy()
    f1 = f1.rename(columns={
        c: f"{c}__t1" for c in feature_cols
    })
    f1 = f1.rename(columns={"row_index": "row_index_t1"})

    f2 = visit_features[["row_index"] + feature_cols].copy()
    f2 = f2.rename(columns={
        c: f"{c}__t2" for c in feature_cols
    })
    f2 = f2.rename(columns={"row_index": "row_index_t2"})

    out = pair_df.merge(
        f1,
        on="row_index_t1",
        how="left",
        validate="many_to_one",
    ).merge(
        f2,
        on="row_index_t2",
        how="left",
        validate="many_to_one",
    )

    dt = pd.to_numeric(out["delta_t_years"], errors="coerce")

    for c in feature_cols:
        a = pd.to_numeric(out[f"{c}__t1"], errors="coerce")
        b = pd.to_numeric(out[f"{c}__t2"], errors="coerce")
        out[f"delta__{c}"] = b - a
        out[f"rate__{c}"] = (b - a) / dt

    keep = [
        "pair_id",
        "analysis_subject_key",
        "split",
        "row_index_t1",
        "row_index_t2",
        "delta_t_years",
    ]

    zcols = z_columns(out)
    keep += zcols
    keep += [f"delta__{c}" for c in feature_cols]
    keep += [f"rate__{c}" for c in feature_cols]

    return out[keep].copy()


def association_table(pair_delta_df, zcols, feature_cols, split_name, change_mode="rate"):
    if split_name == "all":
        d = pair_delta_df.copy()
    else:
        d = pair_delta_df[pair_delta_df["split"] == split_name].copy()

    rows = []

    for zc in zcols:
        z = pd.to_numeric(d[zc], errors="coerce").to_numpy(float)

        for feat in feature_cols:
            prefix = "rate__" if change_mode == "rate" else "delta__"
            dc = f"{prefix}{feat}"
            y = pd.to_numeric(d[dc], errors="coerce").to_numpy(float)
            rho, p, n = robust_corr(z, y)

            rows.append({
                "split": split_name,
                "latent": zc,
                "feature": feat,
                "n": n,
                "spearman_rho": rho,
                "spearman_p": p,
                "abs_spearman_rho": abs(rho) if np.isfinite(rho) else np.nan,
            })

    out = pd.DataFrame(rows)
    out["spearman_q_fdr_global"] = bh_fdr(
        out["spearman_p"].to_numpy(float)
    )
    return out


def instantiate_delta_vae(delta_mod, checkpoint, device):
    latent_dim = int(checkpoint.get("latent_dim", 4))
    condition_on_dt = bool(checkpoint.get("condition_on_dt", True))

    if not hasattr(delta_mod, "ConditionalDeltaVAE"):
        raise AttributeError(
            "Step-19 module has no ConditionalDeltaVAE class."
        )

    model = delta_mod.ConditionalDeltaVAE(
        latent_dim=latent_dim,
        condition_on_dt=condition_on_dt,
    ).to(device)

    state = checkpoint.get("model_state_dict")
    if state is None:
        state = checkpoint.get("state_dict")
    if state is None:
        raise KeyError(
            "Checkpoint contains neither model_state_dict nor state_dict."
        )

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def checkpoint_scaling(checkpoint, pair_df):
    aliases = {
        "x_mean": ["x_train_mean", "x_mean", "input_mean"],
        "x_std": ["x_train_std", "x_std", "input_std"],
        "log_dt_mean": [
            "log_dt_train_mean",
            "log_dt_mean",
            "dt_mean",
        ],
        "log_dt_std": [
            "log_dt_train_std",
            "log_dt_std",
            "dt_std",
        ],
    }

    out = {}

    for canonical, keys in aliases.items():
        value = None
        for k in keys:
            if k in checkpoint:
                value = checkpoint[k]
                break
        if value is not None:
            out[canonical] = float(value)

    # Time scaling can be reconstructed exactly from train-pair metadata
    # if it was not persisted.
    train_dt = pd.to_numeric(
        pair_df.loc[pair_df["split"] == "train", "delta_t_years"],
        errors="coerce",
    ).to_numpy(float)
    train_dt = train_dt[np.isfinite(train_dt) & (train_dt > 0)]

    log_dt = np.log1p(train_dt)

    if "log_dt_mean" not in out:
        out["log_dt_mean"] = float(np.mean(log_dt))
    if "log_dt_std" not in out:
        out["log_dt_std"] = float(np.std(log_dt, ddof=0))

    if "x_mean" not in out or "x_std" not in out:
        raise KeyError(
            "Checkpoint must contain Delta-ECG training input mean/std "
            "(x_train_mean and x_train_std)."
        )

    return out


def tau_for_dt(dt_years, scaling):
    return (
        np.log1p(float(dt_years))
        - scaling["log_dt_mean"]
    ) / scaling["log_dt_std"]


def decode_delta(
    model,
    z_np,
    dt_years,
    scaling,
    input_mode,
    device,
):
    z_np = np.asarray(z_np, dtype=np.float32)
    if z_np.ndim == 1:
        z_np = z_np[None, :]

    tau = np.full(
        (len(z_np), 1),
        tau_for_dt(dt_years, scaling),
        dtype=np.float32,
    )

    z = torch.from_numpy(z_np).to(device)
    t = torch.from_numpy(tau).to(device)

    with torch.no_grad():
        y_scaled = model.decode(z, t).cpu().numpy()

    y_phys = (
        y_scaled * scaling["x_std"]
        + scaling["x_mean"]
    )

    if input_mode == "rate":
        y_phys = y_phys * float(dt_years)

    return y_phys.astype(np.float32)


def choose_reference_beats(pair_df, beats, n_reference, seed):
    train = pair_df[pair_df["split"] == "train"].copy()
    idx = pd.to_numeric(
        train["row_index_t1"],
        errors="coerce",
    ).dropna().astype(int).unique()

    idx = np.asarray(idx, dtype=int)

    valid = np.isfinite(beats[idx]).all(axis=(1, 2))
    idx = idx[valid]

    if len(idx) == 0:
        raise RuntimeError("No valid HCM train reference t1 beats.")

    rng = np.random.default_rng(seed)
    if len(idx) > n_reference:
        idx = rng.choice(idx, size=n_reference, replace=False)

    return np.sort(idx)


def reference_feature_cache(reference_indices, beats, feature_mod):
    cache = {}
    good = []

    for idx in reference_indices:
        try:
            feats, qc = feature_mod.extract_ecg_features(
                beats[idx],
                fs=FS_HZ,
                beat_start_ms=BEAT_START_MS,
                rr_ms=np.nan,
            )
            if bool(qc.get("qrs_boundary_ok", True)):
                cache[int(idx)] = feats
                good.append(int(idx))
        except Exception:
            pass

    if len(good) < 10:
        raise RuntimeError(
            f"Only {len(good)} reference HCM beats passed feature extraction."
        )

    return np.asarray(good, dtype=int), cache


def traversal_feature_changes(
    model,
    mu_train,
    pair_df,
    beats,
    reference_indices,
    reference_baseline_features,
    feature_cols,
    feature_mod,
    scaling,
    input_mode,
    device,
):
    latent_dim = mu_train.shape[1]
    latent_sd = np.std(mu_train, axis=0, ddof=1)

    train_dt = pd.to_numeric(
        pair_df.loc[pair_df["split"] == "train", "delta_t_years"],
        errors="coerce",
    ).to_numpy(float)
    train_dt = train_dt[np.isfinite(train_dt) & (train_dt > 0)]
    dt_ref = 1.0 if input_mode == "rate" else float(np.median(train_dt))

    rows = []

    for k in range(latent_dim):
        # z=0 decoded trajectory at exactly the same Delta t.
        z0 = np.zeros((1, latent_dim), dtype=np.float32)
        delta0 = decode_delta(
            model,
            z0,
            dt_ref,
            scaling,
            input_mode,
            device,
        )[0]

        for level in LEVELS_SD:
            z = np.zeros((1, latent_dim), dtype=np.float32)
            z[0, k] = float(level * latent_sd[k])

            decoded = decode_delta(
                model,
                z,
                dt_ref,
                scaling,
                input_mode,
                device,
            )[0]

            per_reference = {f: [] for f in feature_cols}
            n_success = 0

            for idx in reference_indices:
                base = beats[idx]

                # Compare morphology after decoded change at z-level with
                # morphology after the z=0 decoded change, holding both the
                # initial HCM beat and Delta t fixed.
                synth_level = base + decoded
                synth_zero = base + delta0

                try:
                    f_level, _ = feature_mod.extract_ecg_features(
                        synth_level,
                        fs=FS_HZ,
                        beat_start_ms=BEAT_START_MS,
                        rr_ms=np.nan,
                    )
                    f_zero, _ = feature_mod.extract_ecg_features(
                        synth_zero,
                        fs=FS_HZ,
                        beat_start_ms=BEAT_START_MS,
                        rr_ms=np.nan,
                    )
                except Exception:
                    continue

                n_success += 1

                for feat in feature_cols:
                    a = f_zero.get(feat, np.nan)
                    b = f_level.get(feat, np.nan)
                    if np.isfinite(a) and np.isfinite(b):
                        effect = float(b - a)
                        if input_mode == "rate":
                            effect = effect / float(dt_ref)
                        per_reference[feat].append(effect)

            for feat in feature_cols:
                vals = np.asarray(per_reference[feat], dtype=float)
                finite = vals[np.isfinite(vals)]
                rows.append({
                    "latent": f"D_z_{k:02d}",
                    "latent_index": k,
                    "level_sd": float(level),
                    "delta_t_reference_years": dt_ref,
                    "feature": feat,
                    "n_reference_success": int(len(finite)),
                    "median_delta_feature_vs_z0": (
                        float(np.median(finite))
                        if len(finite) else np.nan
                    ),
                    "mean_delta_feature_vs_z0": (
                        float(np.mean(finite))
                        if len(finite) else np.nan
                    ),
                    "iqr_delta_feature_vs_z0": (
                        float(
                            np.percentile(finite, 75)
                            - np.percentile(finite, 25)
                        )
                        if len(finite) else np.nan
                    ),
                })

        print(f"[traversal] D_z_{k:02d} complete")

    return pd.DataFrame(rows), latent_sd, dt_ref


def traversal_slope_table(trav_df):
    rows = []

    for (latent, feature), g in trav_df.groupby(
        ["latent", "feature"],
        sort=True,
    ):
        x = pd.to_numeric(g["level_sd"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(
            g["median_delta_feature_vs_z0"],
            errors="coerce",
        ).to_numpy(float)

        m = np.isfinite(x) & np.isfinite(y)

        if np.sum(m) < 4 or np.nanstd(y[m]) <= 1e-12:
            continue

        lr = stats.linregress(x[m], y[m])
        sr, sp = stats.spearmanr(x[m], y[m])

        rows.append({
            "latent": latent,
            "feature": feature,
            "n_levels": int(np.sum(m)),
            "slope_delta_feature_per_latent_sd": float(lr.slope),
            "r_squared": float(lr.rvalue ** 2),
            "linear_p": float(lr.pvalue),
            "spearman_level_rho": float(sr),
            "spearman_level_p": float(sp),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["linear_q_fdr_global"] = bh_fdr(
            out["linear_p"].to_numpy(float)
        )
    return out


def concordance_table(test_assoc, trav_slopes):
    a = test_assoc[
        [
            "latent",
            "feature",
            "n",
            "spearman_rho",
            "spearman_p",
            "spearman_q_fdr_global",
            "abs_spearman_rho",
        ]
    ].copy()

    b = trav_slopes[
        [
            "latent",
            "feature",
            "slope_delta_feature_per_latent_sd",
            "r_squared",
            "linear_q_fdr_global",
            "spearman_level_rho",
        ]
    ].copy()

    out = a.merge(
        b,
        on=["latent", "feature"],
        how="inner",
        validate="one_to_one",
    )

    out["sign_concordant"] = (
        np.sign(out["spearman_rho"])
        == np.sign(out["slope_delta_feature_per_latent_sd"])
    )

    out["strong_internal_validation"] = (
        (out["abs_spearman_rho"] >= 0.20)
        & (out["spearman_q_fdr_global"] < 0.05)
        & (out["r_squared"] >= 0.80)
        & out["sign_concordant"]
    )

    out["priority_score"] = (
        out["abs_spearman_rho"]
        * np.sqrt(np.clip(out["r_squared"], 0, 1))
    )

    return out.sort_values(
        ["latent", "strong_internal_validation", "priority_score"],
        ascending=[True, False, False],
    )


def priority_summary(test_assoc, trav_slopes, concordance, zcols, top_n):
    rows = []

    for zc in zcols:
        a = test_assoc[test_assoc["latent"] == zc].sort_values(
            ["spearman_q_fdr_global", "abs_spearman_rho"],
            ascending=[True, False],
        )

        c = concordance[concordance["latent"] == zc].sort_values(
            ["strong_internal_validation", "priority_score"],
            ascending=[False, False],
        )

        t = trav_slopes[trav_slopes["latent"] == zc].copy()
        t["trav_score"] = (
            np.abs(t["spearman_level_rho"])
            * np.sqrt(np.clip(t["r_squared"], 0, 1))
        )
        t = t.sort_values("trav_score", ascending=False)

        row = {"latent": zc}

        for r in range(top_n):
            if r < len(a):
                x = a.iloc[r]
                row[f"test_feature_{r+1}"] = x["feature"]
                row[f"test_rho_{r+1}"] = x["spearman_rho"]
                row[f"test_q_{r+1}"] = x["spearman_q_fdr_global"]

            if r < len(c):
                x = c.iloc[r]
                row[f"concordant_feature_{r+1}"] = x["feature"]
                row[f"concordant_test_rho_{r+1}"] = x["spearman_rho"]
                row[f"concordant_traversal_slope_{r+1}"] = (
                    x["slope_delta_feature_per_latent_sd"]
                )
                row[f"concordant_traversal_r2_{r+1}"] = x["r_squared"]
                row[f"concordant_sign_{r+1}"] = x["sign_concordant"]
                row[f"concordant_strong_{r+1}"] = (
                    x["strong_internal_validation"]
                )

            if r < len(t):
                x = t.iloc[r]
                row[f"traversal_feature_{r+1}"] = x["feature"]
                row[f"traversal_slope_{r+1}"] = (
                    x["slope_delta_feature_per_latent_sd"]
                )
                row[f"traversal_r2_{r+1}"] = x["r_squared"]

        rows.append(row)

    return pd.DataFrame(rows)


def time_conditioning_analysis(
    model,
    latent_sd,
    time_grid,
    scaling,
    input_mode,
    device,
):
    latent_dim = len(latent_sd)
    rows = []

    for dt in time_grid:
        if dt <= 0:
            continue

        z0 = np.zeros((1, latent_dim), dtype=np.float32)
        d0 = decode_delta(
            model, z0, dt, scaling, input_mode, device
        )[0]

        for k in range(latent_dim):
            zm = np.zeros((1, latent_dim), dtype=np.float32)
            zp = np.zeros((1, latent_dim), dtype=np.float32)

            zm[0, k] = -2.0 * latent_sd[k]
            zp[0, k] = +2.0 * latent_sd[k]

            dm = decode_delta(
                model, zm, dt, scaling, input_mode, device
            )[0]
            dp = decode_delta(
                model, zp, dt, scaling, input_mode, device
            )[0]

            effect = dp - dm

            rows.append({
                "latent": f"D_z_{k:02d}",
                "delta_t_years": float(dt),
                "tau_standardized": float(tau_for_dt(dt, scaling)),
                "z0_predicted_change_rms_mv": float(np.sqrt(np.mean(d0 ** 2))),
                "plus2_minus_minus2_effect_rms_mv": float(
                    np.sqrt(np.mean(effect ** 2))
                ),
                "plus2_predicted_change_rms_mv": float(
                    np.sqrt(np.mean(dp ** 2))
                ),
                "minus2_predicted_change_rms_mv": float(
                    np.sqrt(np.mean(dm ** 2))
                ),
            })

    return pd.DataFrame(rows)


def plot_heatmap(test_assoc, zcols, path, max_features=30):
    piv = test_assoc.pivot_table(
        index="feature",
        columns="latent",
        values="spearman_rho",
        aggfunc="first",
    )

    cols = [c for c in zcols if c in piv.columns]
    piv = piv[cols]

    rank = piv.abs().max(axis=1).sort_values(ascending=False)
    feats = rank.head(max_features).index.tolist()
    mat = piv.loc[feats, cols].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8, max(7, 0.22 * len(feats))))
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels(feats, fontsize=7)
    ax.set_title("Test HCM pairs: longitudinal latent vs ECG feature change")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_concordance(concordance, path):
    d = concordance[
        np.isfinite(concordance["spearman_rho"])
        & np.isfinite(concordance["spearman_level_rho"])
    ].copy()

    if len(d) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    for latent, g in d.groupby("latent"):
        ax.scatter(
            g["spearman_rho"],
            g["spearman_level_rho"],
            s=18,
            alpha=0.55,
            label=latent,
        )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Real test pairs: latent vs Delta-feature Spearman rho")
    ax.set_ylabel("Decoder traversal: level vs feature-change Spearman rho")
    ax.set_title("Real longitudinal changes vs decoder traversal direction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_time_conditioning(time_df, path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for latent, g in time_df.groupby("latent"):
        g = g.sort_values("delta_t_years")
        ax.plot(
            g["delta_t_years"],
            g["plus2_minus_minus2_effect_rms_mv"],
            marker="o",
            label=latent,
        )

    ax.set_xlabel("Delta t (years)")
    ax.set_ylabel("RMS latent effect: +2 SD minus -2 SD (mV)")
    ax.set_title("Explicit decoder dependence on the inter-visit time interval")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("20b - Interpret HCM longitudinal RATE-VAE sensitivity modes")
    print("=" * 84)
    print("Device:", device)

    feature_mod = load_module(args.feature_module, "smash_step14_features")
    delta_mod = load_module(args.delta_vae_module, "smash_step19_delta_vae")

    pair_df, zcols, beats = load_inputs(args)

    print("Pairs:", len(pair_df))
    print("RATE-VAE latents:", zcols)

    # ------------------------------------------------------------------
    # Real-pair feature changes: feature(t2) - feature(t1)
    # ------------------------------------------------------------------
    visit_features, visit_qc = extract_visit_features(
        pair_df,
        beats,
        feature_mod,
        FS_HZ,
        BEAT_START_MS,
    )

    visit_features.to_csv(
        args.output_dir / "visit_ecg_features.csv",
        index=False,
    )
    visit_qc.to_csv(
        args.output_dir / "visit_ecg_feature_qc.csv",
        index=False,
    )

    feature_cols = select_feature_columns(visit_features)

    pair_delta = build_pair_delta_feature_table(
        pair_df,
        visit_features,
        feature_cols,
    )

    pair_delta.to_csv(
        args.output_dir / "pair_rate_ecg_features.csv",
        index=False,
    )

    # Load checkpoint now so the physiological association scale matches the model input.
    checkpoint = load_checkpoint(args.checkpoint, device)
    input_mode = str(checkpoint.get("input_mode", "delta"))
    if input_mode != "rate":
        raise RuntimeError(
            f"This sensitivity script requires a rate-model checkpoint; found input_mode={input_mode!r}."
        )

    associations = {}

    for split_name in ["train", "val", "test", "all"]:
        a = association_table(
            pair_delta,
            zcols,
            feature_cols,
            split_name,
            change_mode="rate",
        )
        associations[split_name] = a
        a.to_csv(
            args.output_dir
            / f"latent_rate_feature_associations_{split_name}.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # Load frozen VAE-3
    # ------------------------------------------------------------------
    model = instantiate_delta_vae(delta_mod, checkpoint, device)
    scaling = checkpoint_scaling(checkpoint, pair_df)

    # ------------------------------------------------------------------
    # Decoder traversals on real HCM reference beats
    # ------------------------------------------------------------------
    train_mu = pair_df.loc[
        pair_df["split"] == "train",
        zcols,
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    if not np.isfinite(train_mu).all():
        raise RuntimeError("Non-finite Delta-VAE latent values in train set.")

    refs = choose_reference_beats(
        pair_df,
        beats,
        args.n_reference_beats,
        args.seed,
    )

    refs, ref_cache = reference_feature_cache(
        refs,
        beats,
        feature_mod,
    )

    trav_df, latent_sd, dt_ref = traversal_feature_changes(
        model=model,
        mu_train=train_mu,
        pair_df=pair_df,
        beats=beats,
        reference_indices=refs,
        reference_baseline_features=ref_cache,
        feature_cols=feature_cols,
        feature_mod=feature_mod,
        scaling=scaling,
        input_mode=input_mode,
        device=device,
    )

    trav_df.to_csv(
        args.output_dir / "decoder_traversal_feature_changes.csv",
        index=False,
    )

    trav_slopes = traversal_slope_table(trav_df)
    trav_slopes.to_csv(
        args.output_dir / "decoder_traversal_feature_slopes.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Held-out test concordance is primary
    # ------------------------------------------------------------------
    concordance = concordance_table(
        associations["test"],
        trav_slopes,
    )

    concordance.to_csv(
        args.output_dir / "test_rate_real_vs_traversal_concordance.csv",
        index=False,
    )

    priority = priority_summary(
        associations["test"],
        trav_slopes,
        concordance,
        zcols,
        args.top_n,
    )

    priority.to_csv(
        args.output_dir / "latent_rate_interpretation_priority_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Rate model: no learned Delta-t conditioning.
    # Any absolute decoded Delta-X scales mechanically with Delta-t because
    # rate waveforms are multiplied by the requested interval. Therefore the
    # time-grid output is retained only as a dimensional sanity check and is
    # not interpreted as biological or learned temporal modulation.
    # ------------------------------------------------------------------
    time_df = time_conditioning_analysis(
        model,
        latent_sd,
        args.time_grid_years,
        scaling,
        input_mode,
        device,
    )

    time_df.to_csv(
        args.output_dir / "rate_model_time_scaling_sanity_check.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_heatmap(
        associations["test"],
        zcols,
        fig_dir / "test_latent_rate_feature_heatmap.png",
    )

    plot_concordance(
        concordance,
        fig_dir / "real_vs_traversal_concordance.png",
    )

    plot_time_conditioning(
        time_df,
        fig_dir / "rate_model_time_scaling_sanity_check.png",
    )

    # ------------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------------
    strong = concordance[
        concordance["strong_internal_validation"]
    ].copy()

    strongest_by_latent = []

    for zc in zcols:
        d = concordance[concordance["latent"] == zc].copy()
        d = d.sort_values(
            ["strong_internal_validation", "priority_score"],
            ascending=[False, False],
        )

        if len(d):
            x = d.iloc[0]
            strongest_by_latent.append({
                "latent": zc,
                "feature": str(x["feature"]),
                "test_spearman_rho": float(x["spearman_rho"]),
                "test_q_fdr_global": float(x["spearman_q_fdr_global"]),
                "traversal_r_squared": float(x["r_squared"]),
                "traversal_slope_feature_rate_per_sd": float(
                    x["slope_delta_feature_per_latent_sd"]
                ),
                "sign_concordant": bool(x["sign_concordant"]),
                "strong_internal_validation": bool(
                    x["strong_internal_validation"]
                ),
            })

    summary = {
        "script": "20b_interpret_longitudinal_rate_vae.py",
        "version": "1.1-rate-sensitivity",
        "scientific_role": (
            "Sensitivity analysis of HCM-only longitudinal RATE-VAE modes using "
            "real ECG feature-change rates and one-year-equivalent decoder traversals"
        ),
        "input_mode": "rate",
        "feature_association_scale": "(feature_t2 - feature_t1) / delta_t_years",
        "n_pairs": int(len(pair_df)),
        "n_train_pairs": int(np.sum(pair_df["split"] == "train")),
        "n_val_pairs": int(np.sum(pair_df["split"] == "val")),
        "n_test_pairs": int(np.sum(pair_df["split"] == "test")),
        "latent_dim": int(len(zcols)),
        "latent_columns": zcols,
        "n_ecg_features": int(len(feature_cols)),
        "n_reference_hcm_beats_for_traversals": int(len(refs)),
        "traversal_delta_t_reference_years": float(dt_ref),
        "traversal_interpretation": "one-year-equivalent feature-change rate",
        "traversal_levels_sd": LEVELS_SD.tolist(),
        "time_conditioning_grid_years": [
            float(x) for x in args.time_grid_years
        ],
        "n_strong_test_rate_vs_traversal_pairs": int(len(strong)),
        "strongest_interpretation_per_latent": strongest_by_latent,
        "methodological_notes": [
            (
                "Classical ECG features are never extracted directly from "
                "Delta X because a difference waveform is not a physiological ECG."
            ),
            (
                "Real longitudinal feature-change rates are computed as "
                "[feature(X_t2) - feature(X_t1)] / Delta t."
            ),
            (
                "Decoder traversal feature effects are measured after adding "
                "decoded changes to real HCM reference beats."
            ),
            (
                "Traversal effects are centered against the z=0 decoded "
                "trajectory at the same Delta t, isolating the latent-mode effect."
            ),
            (
                "Held-out test-pair associations are primary for interpretation."
            ),
            (
                "Normal ECG subjects are not used in VAE-3 training or in this "
                "VAE-3 physiological interpretation step."
            ),
            (
                "This sensitivity model is not conditioned on Delta t. Any "
                "absolute Delta-X scaling across the time grid is therefore mechanical "
                "rate-times-interval scaling and is not interpreted as learned kinetics."
            ),
            (
                "This RATE-VAE is a sensitivity analysis and does not replace the "
                "primary Delta-VAE conditioned on elapsed inter-visit time."
            ),
            (
                "These are model-dependent electrical change phenotypes, "
                "not direct biological mechanisms or clinical progression scores."
            ),
        ],
    }

    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 84)
    print("STEP 20 COMPLETE")
    print("=" * 84)
    print("Features used              :", len(feature_cols))
    print("Traversal reference beats  :", len(refs))
    print("Reference Delta t           :", f"{dt_ref:.3f} years")
    print("Strong test concordances    :", len(strong))
    print("Results                     :", args.output_dir)


if __name__ == "__main__":
    main()
