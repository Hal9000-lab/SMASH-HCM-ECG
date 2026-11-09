#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
43_ptbxl_normal_longitudinal_control.py

SMASH-HCM -- Step 43
Longitudinal control for the H_z_06 drift: how fast does z06 change in the SAME
people over time when they do not have HCM?

WHY THIS STEP EXISTS
--------------------
Steps 37-38 compared the HCM drift with a CROSS-SECTIONAL age gradient (PTB-XL
people of different ages). That gradient may be biased by cohort effects and by
the selection of normal ECGs. PTB-XL also contains patients recorded more than
once. Their repeated normal ECGs give a within-person drift without HCM.

Step 02 kept only the FIRST ECG per PTB-XL patient, so the later ECGs were never
projected. This step builds and projects all ECGs of patients with repeats.

DATA
----
Data/PTBXL_normal_reference_manifest.csv   records already selected as normal
Data/ptbxl_database.csv (optional)         PhysioNet metadata: dates, ages, codes,
                                           and further normal repeats
Data/PTBXL_Normal/records500/...            waveforms (WFDB, 500 Hz)

An ECG from ptbxl_database.csv counts as normal only if its SCP codes are NORM
(likelihood 100) with at most SR. Records that are needed but not on disk are
listed in ptbxl_repeat_download_manifest.csv, in the format of Step 00:
    python Scripts\\00_download_ptbxl_normal_reference.py --manifest <that file>
then rerun this step.

SIGNAL PATH: Step-28 functions (first 4 s, 0.5-40 Hz, median beat -350/+650 ms),
then the frozen VAEs through Step-33 functions. Nothing is retrained.
Check: first ECGs that are already in Step 37 must reproduce their H_z_06.
The HCM VAE never saw PTB-XL, so H_z values are not affected by training reuse.

PAIRS: per patient, ECGs at least 30 days apart; age > 89 (coded 300) set missing.

ESTIMATOR: within-subject slope (subject fixed effects), cluster-robust SE.
The same estimator is applied to the Step-38 HCM cohorts, also truncated at the
normal follow-up horizon (95th percentile), so the comparison is like for like.

READING RULE (fixed before seeing the results)
----------------------------------------------
R = normal within-subject slope / HCM within-subject slope, pre_core_clean cohort,
horizon-matched; 95% CI by Monte Carlo.
compatible_with_ageing      CI lower bound >= 0.75
ageing_explains_little      CI entirely within -0.25 to 0.25
indeterminate               otherwise (MIMIC non-HCM control then required)

Supportive: longitudinal versus cross-sectional normal gradient (cohort effect),
and in HCM the t x baseline-age interaction versus the curvature of the normal
age curve. Repeated PTB-XL recordings may come from people under clinical
follow-up, so they are not a random normal sample.

OUTPUT: Results/43_PTBXL_Normal_Longitudinal_Control/
Usage:  python Scripts\\43_ptbxl_normal_longitudinal_control.py
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.2425
MIN_GAP_DAYS = 30
MAX_VALID_AGE = 89
MIN_SUBJECTS = 20
HORIZON_QUANTILE = 0.95
MC_DRAWS = 100_000
SEED = 20261108
OUTCOMES = ["H_z_06", "H_z_02", "H_z_01", "H_z_05", "d_N"]
PRIMARY = "H_z_06"
HCM_COHORTS = ["SHaRe_pre_core_clean", "SHaRe_all"]
PRIMARY_COHORT = "SHaRe_pre_core_clean"
PHYSIONET = "https://physionet.org/files/ptb-xl/1.0.3/"
REPRO_TOLERANCE = 0.02


def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [Path.cwd().resolve(), here.parent, *here.parents]:
        if (p / "Data").exists() and (p / "Results").exists():
            return p
    raise RuntimeError("Cannot locate the project root (expected Data/ and Results/).")


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--root", type=Path, default=None)
    known, _ = pre.parse_known_args()
    root = known.root.resolve() if known.root else infer_project_root()
    d, r, s = root / "Data", root / "Results", root / "Scripts"
    p = argparse.ArgumentParser(description="PTB-XL repeated-ECG longitudinal control for H_z_06.")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--manifest", type=Path, default=d / "PTBXL_normal_reference_manifest.csv")
    p.add_argument("--ptbxl-database", type=Path, default=d / "ptbxl_database.csv")
    p.add_argument("--waveform-root", type=Path, default=d / "PTBXL_Normal")
    p.add_argument("--normal-reference", type=Path, default=r / "37_Z06_Age_Sensitivity" / "normal_reference_values.csv")
    p.add_argument("--hcm-models", type=Path, default=r / "38_Z06_Piecewise_Trajectory" / "piecewise_models.csv")
    p.add_argument("--longitudinal", type=Path, default=r / "16_Longitudinal_DualManifold" / "longitudinal_dual_manifold.csv")
    p.add_argument("--step23", type=Path, default=s / "23_sensitivity_major_events_longitudinal.py")
    p.add_argument("--step28", type=Path, default=s / "28_prepare_mimic_smash_median_beats.py")
    p.add_argument("--step33", type=Path, default=s / "33_project_mimic_frozen_dual_manifold.py")
    p.add_argument("--step37", type=Path, default=s / "37_z06_age_sensitivity.py")
    p.add_argument("--step38", type=Path, default=s / "38_z06_piecewise_trajectory.py")
    p.add_argument("--results-dir", type=Path, default=r / "43_PTBXL_Normal_Longitudinal_Control")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Inventory of repeated normal ECGs
# =============================================================================

def is_strict_normal(scp_codes) -> bool:
    """NORM with likelihood 100 and no code other than SR."""
    try:
        codes = ast.literal_eval(str(scp_codes))
    except (ValueError, SyntaxError):
        return False
    return isinstance(codes, dict) and float(codes.get("NORM", 0)) == 100.0 and set(codes) <= {"NORM", "SR"}


def record_path(waveform_root: Path, filename_hr: str) -> Path:
    rel = str(filename_hr).strip().replace("\\", "/")
    rel = rel[:-4] if rel.endswith((".hea", ".dat")) else rel
    return waveform_root / f"{rel}.hea"


def build_inventory(manifest: pd.DataFrame, database: pd.DataFrame | None, waveform_root: Path) -> pd.DataFrame:
    m = manifest.copy()
    m["ecg_id"] = pd.to_numeric(m["ecg_id"], errors="coerce").astype("Int64")
    m["source"] = "manifest"
    if database is not None:
        db = database.copy()
        db["ecg_id"] = pd.to_numeric(db["ecg_id"], errors="coerce").astype("Int64")
        fill = [c for c in ("recording_date", "age", "sex", "patient_id", "filename_hr") if c in db]
        lookup = db.drop_duplicates("ecg_id").set_index("ecg_id")
        for c in fill:
            from_db = m["ecg_id"].map(lookup[c])
            m[c] = from_db if c not in m else m[c].where(m[c].notna(), from_db)
        extra = db[db["scp_codes"].map(is_strict_normal) & ~db["ecg_id"].isin(m["ecg_id"])].copy()
        extra["source"] = "ptbxl_database"
        m = pd.concat([m, extra[[c for c in m.columns if c in extra.columns]]], ignore_index=True)
    missing = {"recording_date", "patient_id", "filename_hr"} - set(m.columns)
    if missing:
        raise KeyError(f"Missing columns {sorted(missing)}: add Data/ptbxl_database.csv or pass --ptbxl-database.")
    m["patient_id"] = pd.to_numeric(m["patient_id"], errors="coerce").astype("Int64")
    m["recording_date"] = pd.to_datetime(m["recording_date"], errors="coerce")
    m["age"] = pd.to_numeric(m.get("age"), errors="coerce").where(lambda a: a <= MAX_VALID_AGE)
    m = m.dropna(subset=["patient_id", "recording_date"]).sort_values(["patient_id", "recording_date", "ecg_id"])
    m["hea_path"] = [str(record_path(waveform_root, f)) for f in m["filename_hr"]]
    m["on_disk"] = [Path(p).is_file() and Path(p).with_suffix(".dat").is_file() for p in m["hea_path"]]
    return select_spaced(m)


def select_spaced(m: pd.DataFrame) -> pd.DataFrame:
    """Per patient keep the first ECG and each ECG >= MIN_GAP_DAYS after the last kept one."""
    keep = []
    for _, g in m.groupby("patient_id", sort=False):
        last = None
        for idx, date in zip(g.index, g["recording_date"]):
            if last is None or (date - last).days >= MIN_GAP_DAYS:
                keep.append(idx)
                last = date
    out = m.loc[keep].copy()
    out["n_spaced_ecgs"] = out.groupby("patient_id")["ecg_id"].transform("size")
    return out[out["n_spaced_ecgs"] >= 2].copy()


def download_manifest(inv: pd.DataFrame) -> pd.DataFrame:
    need = inv[~inv["on_disk"]]
    rel = need["filename_hr"].astype(str).str.replace("\\", "/", regex=False)
    return pd.DataFrame({"ecg_id": need["ecg_id"], "patient_id": need["patient_id"], "filename_hr": rel,
                         "header_url": PHYSIONET + rel + ".hea", "signal_url": PHYSIONET + rel + ".dat"})


# =============================================================================
# Median beats and frozen projection
# =============================================================================

def build_beats(inv: pd.DataFrame, waveform_root: Path, s28) -> tuple[pd.DataFrame, np.ndarray]:
    sos = s28.make_filter()
    rows, beats = [], []
    for rec in inv.itertuples(index=False):
        row, beat = s28.process_record(Path(rec.hea_path), waveform_root, sos)
        rows.append({"ecg_id": int(rec.ecg_id), "valid_for_projection": bool(row["valid_for_projection"]),
                     "n_usable_beats": row["n_usable_beats"], "error": row["error"]})
        beats.append(beat)
    return pd.DataFrame(rows), np.stack(beats).astype(np.float32)


def project(waves: np.ndarray, valid: np.ndarray, root: Path, s33, args) -> pd.DataFrame:
    import torch

    paths = s33.resolve_default_paths(root)
    device = s33.choose_device(args.device)
    models = s33.load_frozen_models(argparse.Namespace(**paths), device)
    lw_n, _, _ = s33.fit_reference_gaussians(paths["normal_latents"], paths["hcm_latents"])
    x = waves[valid]
    with torch.inference_mode():
        h_mu, _, _ = s33.encode_reconstruct(models["hcm_model"], x, float(models["hcm_mean_mv"]), float(models["hcm_std_mv"]), args.batch_size, device)
        n_mu, _, _ = s33.encode_reconstruct(models["normal_model"], x, float(models["normal_mean_mv"]), float(models["normal_std_mv"]), args.batch_size, device)
    out = pd.DataFrame(np.asarray(h_mu), columns=[f"H_z_{k:02d}" for k in range(8)])
    out["d_N"] = np.asarray(s33.mahalanobis_distance(n_mu, lw_n)[0], float)
    return out


# =============================================================================
# Within-subject estimator
# =============================================================================

def within_fit(df: pd.DataFrame, outcome: str, extra: list[str] | None = None) -> dict:
    """Subject fixed effects; cluster-robust (CR1) covariance. Regressors: t (+ extra)."""
    cols = ["t", *(extra or [])]
    d = df[["subject", outcome, *cols]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d.groupby("subject")["t"].transform("nunique") >= 2]
    g = d.groupby("subject")
    X = (d[cols] - g[cols].transform("mean")).to_numpy(float)
    y = (d[outcome] - g[outcome].transform("mean")).to_numpy(float)
    n_sub = d["subject"].nunique()
    if n_sub < 3:
        return {"n_subjects": n_sub, "n_obs": len(d), "beta": np.full(len(cols), np.nan), "cov": np.full((len(cols),) * 2, np.nan), "cols": cols}
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    e = y - X @ beta
    scores = pd.DataFrame(X * e[:, None]).groupby(d["subject"].to_numpy()).sum().to_numpy()
    cov = bread @ (scores.T @ scores) @ bread * n_sub / (n_sub - 1)
    return {"n_subjects": n_sub, "n_obs": len(d), "beta": beta, "cov": cov, "cols": cols,
            "median_followup": float(g["t"].max().median()), "p95_followup": float(g["t"].max().quantile(HORIZON_QUANTILE))}


def term(fit: dict, name: str = "t") -> tuple[float, float]:
    i = fit["cols"].index(name)
    return float(fit["beta"][i]), float(np.sqrt(max(fit["cov"][i, i], 0.0)))


def truncate(df: pd.DataFrame, horizon: float) -> pd.DataFrame:
    d = df[df["t"] <= horizon]
    return d[d.groupby("subject")["t"].transform("nunique") >= 2].copy()


def ratio_interval(num: tuple[float, float], den: tuple[float, float], rng) -> tuple[float, float, float]:
    a = rng.normal(num[0], num[1], MC_DRAWS)
    b = rng.normal(den[0], den[1], MC_DRAWS)
    low, high = np.percentile(a / b, [2.5, 97.5])
    return num[0] / den[0], float(low), float(high)


def classify(low: float, high: float) -> str:
    if low >= 0.75:
        return "compatible_with_ageing"
    if low > -0.25 and high < 0.25:
        return "ageing_explains_little"
    return "indeterminate"


def curvature(ref: dict, ages: np.ndarray, s37) -> tuple[float, float]:
    """Average second difference of the normal age curve (per year^2)."""
    ages = ages[np.isfinite(ages)]
    male = np.zeros(len(ages)) if ref["uses_sex"] else None
    d = lambda a: s37.reference_design(a, ref["knots"], male)
    c = np.mean(d(ages + 1.0) - 2.0 * d(ages) + d(ages - 1.0), axis=0)
    return float(c @ ref["beta"]), float(np.sqrt(max(c @ ref["cov"] @ c, 0.0)))


def row(label, cohort, outcome, fit, name="t", **extra) -> dict:
    est, se = term(fit, name) if np.isfinite(fit["beta"]).all() else (np.nan, np.nan)
    return {"analysis": label, "cohort": cohort, "outcome": outcome, "term": name, "n_subjects": fit["n_subjects"], "n_obs": fit["n_obs"],
            "estimate": est, "se": se, "ci95_low": est - 1.96 * se, "ci95_high": est + 1.96 * se,
            "median_followup_years": fit.get("median_followup", np.nan), **extra}


def analyse(normal: pd.DataFrame, cohorts: dict, reference: pd.DataFrame, s37, step38_models: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    fits = {o: within_fit(normal, o) for o in OUTCOMES if o in normal}
    for o, f in fits.items():
        rows.append(row("normal_within", "PTBXL_repeats", o, f))
    horizon = fits[PRIMARY]["p95_followup"]
    summary = {"horizon_years": horizon}

    hcm = {}
    for name in HCM_COHORTS:
        if name not in cohorts:
            continue
        for scope, data in (("full", cohorts[name]), ("horizon", truncate(cohorts[name], horizon))):
            for o in OUTCOMES:
                if o in data:
                    f = within_fit(data, o)
                    hcm[(name, scope, o)] = f
                    rows.append(row(f"hcm_within_{scope}", name, o, f))

    for o in OUTCOMES:
        key = (PRIMARY_COHORT, "horizon", o)
        if o not in fits or key not in hcm:
            continue
        num, den = term(fits[o]), term(hcm[key])
        r, low, high = ratio_interval(num, den, rng)
        rows.append({"analysis": "ratio_normal_over_hcm", "cohort": f"{PRIMARY_COHORT}_horizon", "outcome": o, "term": "R",
                     "estimate": r, "ci95_low": low, "ci95_high": high, "reading": classify(low, high)})
        if o == PRIMARY:
            summary.update(ratio=r, ratio_low=low, ratio_high=high, reading=classify(low, high))
            for scope in ("full",):
                r2, l2, h2 = ratio_interval(num, term(hcm[(PRIMARY_COHORT, scope, o)]), rng)
                rows.append({"analysis": "ratio_normal_over_hcm", "cohort": f"{PRIMARY_COHORT}_{scope}", "outcome": o, "term": "R",
                             "estimate": r2, "ci95_low": l2, "ci95_high": h2, "reading": classify(l2, h2)})
            if step38_models is not None:
                m = step38_models[(step38_models["cohort"] == PRIMARY_COHORT) & (step38_models["outcome"] == o) & (step38_models["model"] == "linear")]
                if len(m):
                    r3, l3, h3 = ratio_interval(num, (float(m.iloc[0]["slope"]), float(m.iloc[0]["slope_se"])), rng)
                    rows.append({"analysis": "ratio_normal_over_step38_lmm", "cohort": PRIMARY_COHORT, "outcome": o, "term": "R",
                                 "estimate": r3, "ci95_low": l3, "ci95_high": h3, "reading": classify(l3, h3)})

    # Cohort effect: longitudinal versus cross-sectional normal gradient at the same ages.
    for o in [x for x in OUTCOMES if x in reference and x in fits]:
        ref = s37.fit_age_reference(reference["age"].to_numpy(float), reference[o].to_numpy(float))
        ages = normal.loc[normal["t"].eq(0), "age"].to_numpy(float)
        g, gse = s37.average_age_gradient(ref, ages[np.isfinite(ages)])
        b, bse = term(fits[o])
        diff, dse = b - g, float(np.hypot(bse, gse))
        rows.append({"analysis": "normal_long_minus_cross", "cohort": "PTBXL_repeats", "outcome": o, "term": "difference",
                     "estimate": diff, "se": dse, "ci95_low": diff - 1.96 * dse, "ci95_high": diff + 1.96 * dse, "cross_sectional_gradient": g})
        if o == PRIMARY:
            summary.update(cross_gradient=g, long_minus_cross=diff, long_minus_cross_low=diff - 1.96 * dse, long_minus_cross_high=diff + 1.96 * dse)

        # Curvature check in HCM: if the drift were ageing, the slope would change with
        # baseline age as the normal curve bends.
        if o == PRIMARY and PRIMARY_COHORT in cohorts:
            d = cohorts[PRIMARY_COHORT].copy()
            d["age0"] = d.groupby("subject")["age"].transform("min")
            d["age0_c"] = d["age0"] - d.groupby("subject")["age0"].first().mean()
            d["t_x_age0"] = d["t"] * d["age0_c"]
            f = within_fit(d, o, ["t_x_age0"])
            k, kse = curvature(ref, d["age"].to_numpy(float), s37)
            rows.append(row("hcm_slope_by_baseline_age", PRIMARY_COHORT, o, f, "t_x_age0", expected_if_ageing=k, expected_se=kse))
            summary.update(hcm_interaction=term(f, "t_x_age0"), normal_curvature=(k, kse))

    summary["normal"] = term(fits[PRIMARY]) + (fits[PRIMARY]["n_subjects"], fits[PRIMARY]["median_followup"])
    for scope in ("full", "horizon"):
        key = (PRIMARY_COHORT, scope, PRIMARY)
        if key in hcm:
            summary[f"hcm_{scope}"] = term(hcm[key]) + (hcm[key]["n_subjects"],)
    return pd.DataFrame(rows), summary


# =============================================================================
# Report
# =============================================================================

def build_report(inv: pd.DataFrame, beats: pd.DataFrame, repro: dict, models: pd.DataFrame, s: dict) -> str:
    L = ["SMASH-HCM Step 43 -- PTB-XL repeated normal ECGs as a longitudinal control", "",
         f"Patients with >=2 normal ECGs >= {MIN_GAP_DAYS} days apart: {inv['patient_id'].nunique()} ({len(inv)} ECGs)",
         f"  from manifest {(inv['source'] == 'manifest').sum()}, from ptbxl_database {(inv['source'] != 'manifest').sum()}",
         f"  on disk {inv['on_disk'].sum()}, projected {int(beats['valid_for_projection'].sum())}",
         f"Reproducibility vs Step 37 (first ECGs): {repro.get('text', 'not checked')}", ""]
    n_b, n_se, n_n, n_fu = s["normal"]
    L += ["H_z_06 within-subject slope, units/year",
          f"  PTB-XL normals          : {n_b:+.4f} ({n_b - 1.96 * n_se:+.4f} to {n_b + 1.96 * n_se:+.4f}), {n_n} subjects, median follow-up {n_fu:.2f} y"]
    for scope, label in (("hcm_full", "HCM clean, full       "), ("hcm_horizon", f"HCM clean, <= {s['horizon_years']:.1f} y   ")):
        if scope in s:
            b, se, n = s[scope]
            L.append(f"  {label}: {b:+.4f} ({b - 1.96 * se:+.4f} to {b + 1.96 * se:+.4f}), {n} subjects")
    if "ratio" in s:
        L += ["", f"Primary ratio (normal / HCM clean, horizon-matched): R = {s['ratio']:.2f} ({s['ratio_low']:.2f} to {s['ratio_high']:.2f})",
              f"Reading: {s['reading']}"]
    if "cross_gradient" in s:
        L += ["", f"Cross-sectional PTB-XL gradient at the same ages: {s['cross_gradient']:+.4f}",
              f"Longitudinal minus cross-sectional: {s['long_minus_cross']:+.4f} ({s['long_minus_cross_low']:+.4f} to {s['long_minus_cross_high']:+.4f})"]
    if "hcm_interaction" in s:
        (b, se), (k, kse) = s["hcm_interaction"], s["normal_curvature"]
        L += [f"HCM slope change per year of baseline age: {b:+.5f} ({b - 1.96 * se:+.5f} to {b + 1.96 * se:+.5f});",
              f"  expected from normal curve curvature: {k:+.5f} ({k - 1.96 * kse:+.5f} to {k + 1.96 * kse:+.5f})"]
    L += ["", "All outcomes"]
    for _, r in models.iterrows():
        extra = f"  {r['reading']}" if isinstance(r.get("reading"), str) else ""
        n = f" n={int(r['n_subjects'])}" if pd.notna(r.get("n_subjects")) else ""
        L.append(f"  {r['analysis']:30s} {r['cohort']:30s} {r['outcome']:7s} {r['term']:10s} {r['estimate']:+.4f} "
                 f"({r['ci95_low']:+.4f} to {r['ci95_high']:+.4f}){n}{extra}")
    L += ["", "Reading rule: compatible_with_ageing if CI low >= 0.75; ageing_explains_little if CI within +/-0.25; else indeterminate.",
          "", "Guardrails:", "- Repeated PTB-XL recordings may come from people under follow-up; not a random normal sample.",
          "- PTB-XL follow-up is short; the comparison uses the HCM drift over the same horizon.",
          "- Within-subject estimator with cluster-robust SE; no multiplicity correction. Exploratory."]
    return "\n".join(L) + "\n"


def reproducibility(latents: pd.DataFrame, reference: pd.DataFrame) -> dict:
    if "ecg_id" not in reference:
        return {"text": "Step-37 file has no ecg_id"}
    ref = reference[["ecg_id", PRIMARY]].dropna().assign(ecg_id=lambda d: pd.to_numeric(d["ecg_id"], errors="coerce"))
    m = latents[["ecg_id", PRIMARY]].merge(ref, on="ecg_id", suffixes=("", "_step37"))
    if m.empty:
        return {"text": "no overlapping ECG"}
    diff = (m[PRIMARY] - m[f"{PRIMARY}_step37"]).abs()
    ok = diff.max() < REPRO_TOLERANCE
    return {"n": len(m), "max_abs_diff": float(diff.max()), "ok": bool(ok),
            "text": f"{len(m)} ECGs, max |diff| {diff.max():.4f}, r {np.corrcoef(m[PRIMARY], m[f'{PRIMARY}_step37'])[0, 1]:.4f} "
                    f"-> {'MATCH' if ok else 'MISMATCH: preprocessing differs from Step 04; do not use this run'}"}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Inventory of repeated normal ECGs")
    manifest = pd.read_csv(args.manifest)
    database = pd.read_csv(args.ptbxl_database) if args.ptbxl_database.is_file() else None
    if database is None:
        print(f"  {args.ptbxl_database.name} not found: only manifest records are used.")
    inv = build_inventory(manifest, database, args.waveform_root)
    inv.drop(columns=["hea_path"]).to_csv(out / "ptbxl_repeat_inventory.csv", index=False)
    dl = download_manifest(inv)
    if len(dl):
        dl.to_csv(out / "ptbxl_repeat_download_manifest.csv", index=False)
    on_disk = inv[inv["on_disk"]]
    on_disk = on_disk[on_disk.groupby("patient_id")["ecg_id"].transform("size") >= 2]
    print(f"  patients {inv['patient_id'].nunique()}, ECGs {len(inv)}, missing on disk {len(dl)}")
    if on_disk["patient_id"].nunique() < MIN_SUBJECTS:
        msg = (f"Only {on_disk['patient_id'].nunique()} patients have >=2 spaced ECGs on disk (need {MIN_SUBJECTS}).\n"
               f"Download the missing records, then rerun:\n"
               f"  python Scripts\\00_download_ptbxl_normal_reference.py --manifest \"{out / 'ptbxl_repeat_download_manifest.csv'}\"\n")
        (out / "ptbxl_longitudinal_report.txt").write_text(msg, encoding="utf-8")
        print(msg)
        return

    print("[2/5] Median beats (Step-28 functions)")
    s28 = load_module(args.step28.resolve(), "smash_step28")
    beats_meta, waves = build_beats(on_disk, args.waveform_root, s28)
    valid = beats_meta["valid_for_projection"].to_numpy(bool) & np.isfinite(waves).all(axis=(1, 2))

    print("[3/5] Frozen projection (Step-33 functions)")
    s33 = load_module(args.step33.resolve(), "smash_step33")
    lat = project(waves, valid, root, s33, args)
    lat.insert(0, "ecg_id", beats_meta.loc[valid, "ecg_id"].to_numpy())
    normal = on_disk.merge(lat, on="ecg_id")
    normal["subject"] = normal["patient_id"].astype(str)
    first = normal.groupby("subject")["recording_date"].transform("min")
    normal["t"] = (normal["recording_date"] - first).dt.total_seconds() / (86400.0 * DAYS_PER_YEAR)
    normal.drop(columns=["hea_path"]).to_csv(out / "ptbxl_repeat_latents.csv", index=False)
    beats_meta.to_csv(out / "ptbxl_repeat_beat_qc.csv", index=False)

    print("[4/5] HCM cohorts (Step-38 functions)")
    s37 = load_module(args.step37.resolve(), "smash_step37")
    s38 = load_module(args.step38.resolve(), "smash_step38")
    visits = s38.load_share_visits(args.longitudinal.resolve())
    s2p, events = s38.load_step23_events(args.step23.resolve(), root)
    os.chdir(root)
    cohorts, _ = s38.build_share_cohorts(visits, s2p, events)
    reference = pd.read_csv(args.normal_reference)
    step38 = pd.read_csv(args.hcm_models) if args.hcm_models.is_file() else None

    print("[5/5] Models")
    repro = reproducibility(lat, reference)
    models, summary = analyse(normal, cohorts, reference, s37, step38)
    models.to_csv(out / "longitudinal_control_models.csv", index=False)
    report = build_report(inv, beats_meta, repro, models, summary)
    (out / "ptbxl_longitudinal_report.txt").write_text(report, encoding="utf-8")
    (out / "run_summary.json").write_text(json.dumps({k: v for k, v in summary.items() if not isinstance(v, tuple)} | {"reproducibility": repro}, indent=2, default=float), encoding="utf-8")
    print(report)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
