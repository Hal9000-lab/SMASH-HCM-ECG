#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19_train_longitudinal_delta_vae.py

SMASH-HCM VAE-3: HCM-only longitudinal Delta-VAE.

Training population
-------------------
ONLY HCM subjects with repeated valid ECGs.
Normal subjects are never used to train VAE-3.

For each subject, consecutive valid visits are paired:
    delta_X = X(t2) - X(t1)
where X is the 12-lead mean/median beat from Step 16.

The conditional VAE learns a low-dimensional representation of ECG change:
    z_delta = E(delta_X, delta_t)
    delta_X_hat = D(z_delta, delta_t)

Existing frozen normal-manifold (16D) and HCM-oriented (8D) coordinates are
used only AFTER VAE-3 training to interpret the learned change modes.

Expected input
--------------
Results/16_Longitudinal_DualManifold/Phase3_HCM_Longitudinal_DualManifold.h5

Outputs
-------
Results/19_Longitudinal_DeltaVAE/
    Phase5_HCM_Longitudinal_DeltaVAE_pairs.h5
    best_model.pt
    training_history.csv
    pair_metadata.csv
    delta_vae_latents.csv
    reconstruction_metrics_by_pair.csv
    latent_change_associations.csv
    subject_delta_latent_summary.csv
    run_summary.json
    figures/
        training_history.png
        delta_latent_scatter.png
        test_reconstruction_examples.png

Run from project root:
    python Scripts\\19_train_longitudinal_delta_vae.py
"""

from __future__ import annotations

import argparse
import copy
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEED = 20261005
N_LEADS = 12
N_SAMPLES = 500
NORMAL_DIM = 16
HCM_DIM = 8
FS_HZ = 500.0
PRE_MS = -350.0
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-h5", type=Path, default=Path("Results/16_Longitudinal_DualManifold/Phase3_HCM_Longitudinal_DualManifold.h5"))
    p.add_argument("--output-h5", type=Path, default=Path("Results/19_Longitudinal_DeltaVAE/Phase5_HCM_Longitudinal_DeltaVAE_pairs.h5"))
    p.add_argument("--results-dir", type=Path, default=Path("Results/19_Longitudinal_DeltaVAE"))
    p.add_argument("--pair-mode", choices=["consecutive", "all"], default="consecutive")
    p.add_argument("--input-mode", choices=["delta", "rate"], default="delta")
    p.add_argument("--min-delta-days", type=float, default=1.0)
    p.add_argument("--max-delta-years", type=float, default=25.0)
    p.add_argument("--latent-dim", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--kl-warmup-epochs", type=int, default=30)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-condition-on-dt", dest="condition_on_dt", action="store_false")
    p.set_defaults(condition_on_dt=True)
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
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_str(arr):
    out = []
    for x in arr:
        out.append(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x))
    return np.asarray(out, dtype=object)


def write_str(group, name, values):
    dt = h5py.string_dtype("utf-8")
    group.create_dataset(name, data=np.asarray([str(x) for x in values], dtype=object), dtype=dt)


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pf = p[ok]
    order = np.argsort(pf)
    ranked = pf[order]
    m = len(ranked)
    qr = ranked * m / np.arange(1, m + 1)
    qr = np.minimum.accumulate(qr[::-1])[::-1]
    qr = np.clip(qr, 0, 1)
    tmp = np.empty(m)
    tmp[order] = qr
    q[ok] = tmp
    return q


def spearman_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 5 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[m], y[m])
    return float(rho), float(p), n


def load_input(path):
    required = [
        "signals/median_all_mv",
        "quality/valid_for_latent_projection",
        "metadata/analysis_subject_key",
        "metadata/ecg_id",
        "metadata/ecg_time",
        "metadata/time_from_first_valid_ecg_years",
        "normal_manifold/z_mu",
        "hcm_manifold/z_mu",
        "normal_manifold/d_N",
        "hcm_manifold/d_HCM",
    ]
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        missing = [x for x in required if x not in h5]
        if missing:
            raise KeyError("Missing datasets: " + ", ".join(missing))
        d = {
            "beats": np.asarray(h5["signals/median_all_mv"][:], dtype=np.float32),
            "valid": np.asarray(h5["quality/valid_for_latent_projection"][:], dtype=np.uint8).astype(bool),
            "subject": decode_str(h5["metadata/analysis_subject_key"][:]),
            "ecg_id": decode_str(h5["metadata/ecg_id"][:]),
            "ecg_time": decode_str(h5["metadata/ecg_time"][:]),
            "time": np.asarray(h5["metadata/time_from_first_valid_ecg_years"][:], dtype=float),
            "zN": np.asarray(h5["normal_manifold/z_mu"][:], dtype=np.float32),
            "zH": np.asarray(h5["hcm_manifold/z_mu"][:], dtype=np.float32),
            "dN": np.asarray(h5["normal_manifold/d_N"][:], dtype=np.float32),
            "dH": np.asarray(h5["hcm_manifold/d_HCM"][:], dtype=np.float32),
        }
    n = len(d["beats"])
    if d["beats"].shape != (n, N_LEADS, N_SAMPLES):
        raise ValueError(f"Expected beats [N,12,500], found {d['beats'].shape}")
    if d["zN"].shape != (n, NORMAL_DIM):
        raise ValueError(f"Expected normal z [N,16], found {d['zN'].shape}")
    if d["zH"].shape != (n, HCM_DIM):
        raise ValueError(f"Expected HCM z [N,8], found {d['zH'].shape}")
    return d


def build_pairs(d, pair_mode, min_days, max_years):
    rows = []
    min_years = min_days / 365.25
    pair_id = 0
    for subject in pd.unique(d["subject"]):
        idx = np.where((d["subject"] == subject) & d["valid"] & np.isfinite(d["time"]))[0]
        if len(idx) < 2:
            continue
        finite_beats = np.isfinite(d["beats"][idx]).all(axis=(1, 2))
        idx = idx[finite_beats]
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(d["time"][idx], kind="mergesort")]
        candidates = []
        if pair_mode == "consecutive":
            candidates = [(int(idx[j]), int(idx[j + 1])) for j in range(len(idx) - 1)]
        else:
            for a in range(len(idx) - 1):
                for b in range(a + 1, len(idx)):
                    candidates.append((int(idx[a]), int(idx[b])))
        for i1, i2 in candidates:
            dt = float(d["time"][i2] - d["time"][i1])
            if not np.isfinite(dt) or dt <= min_years or (max_years > 0 and dt > max_years):
                continue
            rows.append({
                "pair_id": pair_id,
                "subject": str(subject),
                "i1": i1,
                "i2": i2,
                "ecg_id_t1": str(d["ecg_id"][i1]),
                "ecg_id_t2": str(d["ecg_id"][i2]),
                "ecg_time_t1": str(d["ecg_time"][i1]),
                "ecg_time_t2": str(d["ecg_time"][i2]),
                "t1": float(d["time"][i1]),
                "t2": float(d["time"][i2]),
                "dt": dt,
                "delta": (d["beats"][i2] - d["beats"][i1]).astype(np.float32),
                "dzN": (d["zN"][i2] - d["zN"][i1]).astype(np.float32),
                "dzH": (d["zH"][i2] - d["zH"][i1]).astype(np.float32),
                "ddN": float(d["dN"][i2] - d["dN"][i1]),
                "ddH": float(d["dH"][i2] - d["dH"][i1]),
            })
            pair_id += 1
    if not rows:
        raise RuntimeError("No valid HCM longitudinal pairs")
    return rows


def subject_split(rows, train_fraction, val_fraction, seed):
    subjects = np.asarray(sorted({r["subject"] for r in rows}), dtype=object)
    if len(subjects) < 10:
        raise RuntimeError("Too few repeated-measure HCM subjects")
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Invalid split fractions")
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = min(max(int(round(train_fraction * n)), 1), n - 2)
    n_val = min(max(int(round(val_fraction * n)), 1), n - n_train - 1)
    mapping = {}
    for s in subjects[:n_train]:
        mapping[str(s)] = "train"
    for s in subjects[n_train:n_train + n_val]:
        mapping[str(s)] = "val"
    for s in subjects[n_train + n_val:]:
        mapping[str(s)] = "test"
    return mapping


def prepare_arrays(rows, split_map, input_mode):
    n = len(rows)
    delta = np.stack([r["delta"] for r in rows]).astype(np.float32)
    dt = np.asarray([r["dt"] for r in rows], dtype=np.float32)
    subjects = np.asarray([r["subject"] for r in rows], dtype=object)
    split = np.asarray([split_map[r["subject"]] for r in rows], dtype=object)
    dzN = np.stack([r["dzN"] for r in rows]).astype(np.float32)
    dzH = np.stack([r["dzH"] for r in rows]).astype(np.float32)
    ddN = np.asarray([r["ddN"] for r in rows], dtype=np.float32)
    ddH = np.asarray([r["ddH"] for r in rows], dtype=np.float32)
    x_phys = delta.copy() if input_mode == "delta" else delta / dt[:, None, None]
    train = split == "train"
    x_mean = float(np.mean(x_phys[train]))
    x_std = float(np.std(x_phys[train]))
    if not np.isfinite(x_std) or x_std <= 0:
        raise RuntimeError("Invalid training input standard deviation")
    logdt = np.log1p(dt.astype(float))
    logdt_mean = float(np.mean(logdt[train]))
    logdt_std = float(np.std(logdt[train]))
    if not np.isfinite(logdt_std) or logdt_std <= 0:
        logdt_std = 1.0
    tau = ((logdt - logdt_mean) / logdt_std).astype(np.float32)[:, None]
    x_scaled = ((x_phys - x_mean) / x_std).astype(np.float32)
    weights = np.ones(n, dtype=np.float32)
    u, c = np.unique(subjects[train], return_counts=True)
    count_map = {str(a): int(b) for a, b in zip(u, c)}
    for i in np.where(train)[0]:
        weights[i] = 1.0 / count_map[str(subjects[i])]
    weights[train] /= np.mean(weights[train])
    return {
        "delta": delta, "dt": dt, "subjects": subjects, "split": split,
        "dzN": dzN, "dzH": dzH, "ddN": ddN, "ddH": ddH,
        "x_phys": x_phys, "x_scaled": x_scaled, "tau": tau,
        "weights": weights,
        "scaling": {"x_mean": x_mean, "x_std": x_std, "logdt_mean": logdt_mean, "logdt_std": logdt_std},
    }


class PairDataset(Dataset):
    def __init__(self, x, tau, weights, indices):
        self.x = x
        self.tau = tau
        self.weights = weights
        self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, k):
        i = int(self.indices[k])
        return (
            torch.from_numpy(self.x[i]),
            torch.from_numpy(self.tau[i]),
            torch.tensor(self.weights[i], dtype=torch.float32),
            torch.tensor(i, dtype=torch.long),
        )


class Block(nn.Module):
    def __init__(self, cin, cout, groups):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 4, 2, 1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)


class ConditionalDeltaVAE(nn.Module):
    def __init__(self, latent_dim=4, condition_on_dt=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_on_dt = condition_on_dt
        self.encoder_conv = nn.Sequential(
            Block(12, 16, 4),
            Block(16, 32, 8),
            Block(32, 48, 8),
            Block(48, 64, 8),
        )
        enc_in = 64 * 31 + (1 if condition_on_dt else 0)
        self.encoder_fc = nn.Sequential(nn.Linear(enc_in, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU())
        self.fc_mu = nn.Linear(48, latent_dim)
        self.fc_logvar = nn.Linear(48, latent_dim)
        dec_in = latent_dim + (1 if condition_on_dt else 0)
        self.decoder_fc = nn.Sequential(nn.Linear(dec_in, 48), nn.SiLU(), nn.Linear(48, 64 * 31), nn.SiLU())
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 48, 4, 2, 1, output_padding=0), nn.GroupNorm(8, 48), nn.SiLU(),
            nn.ConvTranspose1d(48, 32, 4, 2, 1, output_padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose1d(32, 16, 4, 2, 1, output_padding=0), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.ConvTranspose1d(16, 12, 4, 2, 1, output_padding=0),
        )
    def encode(self, x, tau):
        h = self.encoder_conv(x).flatten(1)
        if self.condition_on_dt:
            h = torch.cat([h, tau], dim=1)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    def decode(self, z, tau):
        if self.condition_on_dt:
            z = torch.cat([z, tau], dim=1)
        h = self.decoder_fc(z).view(z.shape[0], 64, 31)
        return self.decoder_conv(h)
    def forward(self, x, tau, sample=True):
        mu, logvar = self.encode(x, tau)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z, tau), mu, logvar


def loss_fn(recon, x, mu, logvar, w, beta):
    recon_per = torch.mean((recon - x) ** 2, dim=(1, 2))
    kl_per = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    w = w.reshape(-1)
    recon_loss = torch.sum(recon_per * w) / torch.sum(w)
    kl_loss = torch.sum(kl_per * w) / torch.sum(w)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def run_epoch(model, loader, device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = np.zeros(3, dtype=float)
    nb = 0
    for x, tau, w, _ in loader:
        x = x.to(device)
        tau = tau.to(device)
        w = w.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            recon, mu, logvar = model(x, tau, sample=training)
            total, rec, kl = loss_fn(recon, x, mu, logvar, w, beta)
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        sums += [float(total.detach().cpu()), float(rec.detach().cpu()), float(kl.detach().cpu())]
        nb += 1
    return {"loss": sums[0] / nb, "recon": sums[1] / nb, "kl": sums[2] / nb}


def train_model(model, train_loader, val_loader, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_val = np.inf
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        beta_train = args.beta * min(1.0, epoch / max(1, args.kl_warmup_epochs))
        tr = run_epoch(model, train_loader, device, beta_train, opt)
        with torch.no_grad():
            va = run_epoch(model, val_loader, device, args.beta, None)
        history.append({
            "epoch": epoch, "beta_train": beta_train,
            "train_loss": tr["loss"], "train_recon": tr["recon"], "train_kl": tr["kl"],
            "val_loss": va["loss"], "val_recon": va["recon"], "val_kl": va["kl"],
        })
        print(f"[epoch {epoch:03d}] beta={beta_train:.5f} train={tr['loss']:.6f} val={va['loss']:.6f} val_rec={va['recon']:.6f} val_KL={va['kl']:.6f}")
        if epoch >= args.kl_warmup_epochs:
            if va["loss"] < best_val - 1e-7:
                best_val = va["loss"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                print(f"[early stopping] epoch={epoch}, best_epoch={best_epoch}")
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)
        best_val = history[-1]["val_loss"]
    model.load_state_dict(best_state)
    return pd.DataFrame(history), best_epoch, best_val


def encode_all(model, x, tau, batch_size, device):
    model.eval()
    n = len(x)
    mu = np.empty((n, model.latent_dim), dtype=np.float32)
    logvar = np.empty_like(mu)
    recon = np.empty_like(x)
    with torch.no_grad():
        for a in range(0, n, batch_size):
            b = min(n, a + batch_size)
            xt = torch.from_numpy(x[a:b]).to(device)
            tt = torch.from_numpy(tau[a:b]).to(device)
            m, lv = model.encode(xt, tt)
            rr = model.decode(m, tt)
            mu[a:b] = m.cpu().numpy()
            logvar[a:b] = lv.cpu().numpy()
            recon[a:b] = rr.cpu().numpy()
    return mu, logvar, recon


def reconstruction_metrics(true_delta, recon_delta):
    err2 = (true_delta - recon_delta) ** 2
    mse = np.mean(err2, axis=(1, 2))
    rmse = np.sqrt(mse)
    energy = np.mean(true_delta ** 2, axis=(1, 2))
    return {
        "mse_mv2": mse,
        "rmse_mv": rmse,
        "true_delta_rms_mv": np.sqrt(energy),
        "relative_mse": mse / (energy + 1e-12),
    }


def association_table(mu, arr):
    rows = []
    targets = []
    for j in range(HCM_DIM):
        targets.append(("delta_hcm", f"delta_H_z_{j:02d}", arr["dzH"][:, j]))
        targets.append(("rate_hcm", f"rate_H_z_{j:02d}", arr["dzH"][:, j] / arr["dt"]))
    for j in range(NORMAL_DIM):
        targets.append(("delta_normal", f"delta_N_z_{j:02d}", arr["dzN"][:, j]))
        targets.append(("rate_normal", f"rate_N_z_{j:02d}", arr["dzN"][:, j] / arr["dt"]))
    targets += [
        ("manifold", "delta_d_N", arr["ddN"]),
        ("manifold", "rate_d_N", arr["ddN"] / arr["dt"]),
        ("manifold", "delta_d_HCM", arr["ddH"]),
        ("manifold", "rate_d_HCM", arr["ddH"] / arr["dt"]),
        ("time", "delta_t_years", arr["dt"]),
    ]
    for split_name in ["train", "val", "test", "all"]:
        mask = np.ones(len(mu), dtype=bool) if split_name == "all" else arr["split"] == split_name
        for z in range(mu.shape[1]):
            start = len(rows)
            for group, name, y in targets:
                rho, p, n = spearman_safe(mu[mask, z], np.asarray(y)[mask])
                rows.append({
                    "split": split_name, "delta_latent": f"D_z_{z:02d}",
                    "target_group": group, "target": name,
                    "n": n, "spearman_rho": rho, "p_value": p,
                })
            q = bh_fdr([rows[k]["p_value"] for k in range(start, len(rows))])
            for k, qq in zip(range(start, len(rows)), q):
                rows[k]["q_fdr_within_delta_latent"] = qq
    return pd.DataFrame(rows)


def plot_history(history, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VAE loss")
    ax.set_title("HCM-only longitudinal Delta-VAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_latent(mu, split, path):
    if mu.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in ["train", "val", "test"]:
        m = split == s
        ax.scatter(mu[m, 0], mu[m, 1], s=12, alpha=0.5, label=s)
    ax.set_xlabel("D_z_00")
    ax.set_ylabel("D_z_01")
    ax.set_title("Longitudinal HCM change latent space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_test_reconstruction(true_delta, recon_delta, split, path, seed):
    idx = np.where(split == "test")[0]
    if len(idx) == 0:
        return
    rng = np.random.default_rng(seed)
    choose = rng.choice(idx, size=min(6, len(idx)), replace=False)
    lead = LEAD_NAMES.index("V5")
    t = np.arange(N_SAMPLES) / FS_HZ * 1000.0 + PRE_MS
    vals = np.concatenate([true_delta[choose, lead].ravel(), recon_delta[choose, lead].ravel()])
    scale = float(np.nanpercentile(np.abs(vals), 95))
    if not np.isfinite(scale) or scale <= 0:
        scale = 0.05
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, i in enumerate(choose):
        offset = k * 3.0 * scale
        ax.plot(t, true_delta[i, lead] + offset, linewidth=1.0, label="true delta V5" if k == 0 else None)
        ax.plot(t, recon_delta[i, lead] + offset, linewidth=1.0, linestyle="--", label="reconstructed delta V5" if k == 0 else None)
    ax.set_xlabel("Time relative to R peak (ms)")
    ax.set_ylabel("Offset pair examples")
    ax.set_title("Test-pair longitudinal ECG change reconstruction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def effective_dim(z):
    v = np.var(z, axis=0, ddof=1)
    d = np.sum(v ** 2)
    return float(np.sum(v) ** 2 / d) if d > 0 else np.nan


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("19 - HCM-only longitudinal Delta-VAE")
    print("=" * 80)
    print("Input        :", args.input_h5)
    print("Pair mode    :", args.pair_mode)
    print("Input mode   :", args.input_mode)
    print("Latent dim   :", args.latent_dim)
    print("Device       :", device)
    print("Training set : HCM repeated ECGs only")
    print()

    d = load_input(args.input_h5)
    rows = build_pairs(d, args.pair_mode, args.min_delta_days, args.max_delta_years)
    split_map = subject_split(rows, args.train_fraction, args.val_fraction, args.seed)
    arr = prepare_arrays(rows, split_map, args.input_mode)

    idx = {s: np.where(arr["split"] == s)[0] for s in ["train", "val", "test"]}
    for s in ["train", "val", "test"]:
        print(f"{s:>5s}: {len(idx[s]):4d} pairs, {len(np.unique(arr['subjects'][idx[s]])):3d} HCM subjects")

    train_ds = PairDataset(arr["x_scaled"], arr["tau"], arr["weights"], idx["train"])
    val_ds = PairDataset(arr["x_scaled"], arr["tau"], np.ones(len(rows), dtype=np.float32), idx["val"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = ConditionalDeltaVAE(args.latent_dim, args.condition_on_dt).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Parameters   :", f"{n_params:,}")

    history, best_epoch, best_val = train_model(model, train_loader, val_loader, args, device)
    history.to_csv(args.results_dir / "training_history.csv", index=False)

    mu, logvar, recon_scaled = encode_all(model, arr["x_scaled"], arr["tau"], args.batch_size, device)
    sc = arr["scaling"]
    recon_phys_model = recon_scaled * sc["x_std"] + sc["x_mean"]
    recon_delta = recon_phys_model if args.input_mode == "delta" else recon_phys_model * arr["dt"][:, None, None]
    rec = reconstruction_metrics(arr["delta"], recon_delta)

    pair_meta = pd.DataFrame({
        "pair_id": [r["pair_id"] for r in rows],
        "analysis_subject_key": [r["subject"] for r in rows],
        "split": arr["split"],
        "row_index_t1": [r["i1"] for r in rows],
        "row_index_t2": [r["i2"] for r in rows],
        "ecg_id_t1": [r["ecg_id_t1"] for r in rows],
        "ecg_id_t2": [r["ecg_id_t2"] for r in rows],
        "ecg_time_t1": [r["ecg_time_t1"] for r in rows],
        "ecg_time_t2": [r["ecg_time_t2"] for r in rows],
        "time_t1_years": [r["t1"] for r in rows],
        "time_t2_years": [r["t2"] for r in rows],
        "delta_t_years": arr["dt"],
    })
    pair_meta.to_csv(args.results_dir / "pair_metadata.csv", index=False)

    latent_df = pair_meta[["pair_id", "analysis_subject_key", "split", "delta_t_years"]].copy()
    for j in range(args.latent_dim):
        latent_df[f"D_z_{j:02d}"] = mu[:, j]
        latent_df[f"D_logvar_{j:02d}"] = logvar[:, j]
    latent_df.to_csv(args.results_dir / "delta_vae_latents.csv", index=False)

    rec_df = pair_meta[["pair_id", "analysis_subject_key", "split", "delta_t_years"]].copy()
    for k, v in rec.items():
        rec_df[k] = v
    rec_df.to_csv(args.results_dir / "reconstruction_metrics_by_pair.csv", index=False)

    assoc = association_table(mu, arr)
    assoc.to_csv(args.results_dir / "latent_change_associations.csv", index=False)

    subject_rows = []
    zcols = [f"D_z_{j:02d}" for j in range(args.latent_dim)]
    merged = pair_meta.merge(latent_df[["pair_id"] + zcols], on="pair_id", validate="one_to_one")
    for subject, g in merged.groupby("analysis_subject_key", sort=False):
        row = {
            "analysis_subject_key": subject,
            "split": g["split"].iloc[0],
            "n_pairs": int(len(g)),
            "median_delta_t_years": float(np.median(g["delta_t_years"])),
        }
        for c in zcols:
            row[f"mean_{c}"] = float(np.mean(g[c]))
            row[f"median_{c}"] = float(np.median(g[c]))
        subject_rows.append(row)
    pd.DataFrame(subject_rows).to_csv(args.results_dir / "subject_delta_latent_summary.csv", index=False)

    torch.save({
        "script": "19_train_longitudinal_delta_vae.py",
        "version": "1.0",
        "model_state_dict": model.state_dict(),
        "latent_dim": args.latent_dim,
        "condition_on_dt": args.condition_on_dt,
        "input_mode": args.input_mode,
        "pair_mode": args.pair_mode,
        "x_train_mean": sc["x_mean"],
        "x_train_std": sc["x_std"],
        "logdt_train_mean": sc["logdt_mean"],
        "logdt_train_std": sc["logdt_std"],
        "beta": args.beta,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "training_population": "HCM subjects with repeated valid ECGs only",
        "normal_subjects_used_for_training": False,
    }, args.results_dir / "best_model.pt")

    with h5py.File(args.output_h5, "w") as h5:
        h5.attrs["description"] = "SMASH-HCM HCM-only longitudinal Delta-VAE pair dataset"
        h5.attrs["training_population"] = "HCM repeated ECGs only"
        h5.attrs["normal_subjects_used_for_training"] = False
        h5.attrs["pair_mode"] = args.pair_mode
        h5.attrs["input_mode"] = args.input_mode
        h5.attrs["latent_dim"] = args.latent_dim
        h5.attrs["condition_on_dt"] = args.condition_on_dt
        pg = h5.create_group("pairs")
        pg.create_dataset("delta_mv", data=arr["delta"], compression="gzip", compression_opts=4, chunks=(1, N_LEADS, N_SAMPLES))
        pg.create_dataset("reconstructed_delta_mv", data=recon_delta.astype(np.float32), compression="gzip", compression_opts=4, chunks=(1, N_LEADS, N_SAMPLES))
        pg.create_dataset("delta_t_years", data=arr["dt"], compression="gzip", compression_opts=4)
        pg.create_dataset("z_delta_mu", data=mu, compression="gzip", compression_opts=4)
        pg.create_dataset("z_delta_logvar", data=logvar, compression="gzip", compression_opts=4)
        pg.create_dataset("delta_z_normal", data=arr["dzN"], compression="gzip", compression_opts=4)
        pg.create_dataset("delta_z_hcm", data=arr["dzH"], compression="gzip", compression_opts=4)
        pg.create_dataset("delta_d_N", data=arr["ddN"], compression="gzip", compression_opts=4)
        pg.create_dataset("delta_d_HCM", data=arr["ddH"], compression="gzip", compression_opts=4)
        mg = h5.create_group("metadata")
        write_str(mg, "analysis_subject_key", arr["subjects"])
        write_str(mg, "split", arr["split"])
        write_str(mg, "ecg_id_t1", pair_meta["ecg_id_t1"].values)
        write_str(mg, "ecg_id_t2", pair_meta["ecg_id_t2"].values)
        mg.create_dataset("pair_id", data=pair_meta["pair_id"].to_numpy(dtype=np.int32), compression="gzip", compression_opts=4)

    plot_history(history, fig_dir / "training_history.png")
    plot_latent(mu, arr["split"], fig_dir / "delta_latent_scatter.png")
    plot_test_reconstruction(arr["delta"], recon_delta, arr["split"], fig_dir / "test_reconstruction_examples.png", args.seed)

    split_summary = {}
    for s in ["train", "val", "test"]:
        ii = idx[s]
        split_summary[s] = {
            "n_pairs": int(len(ii)),
            "n_subjects": int(len(np.unique(arr["subjects"][ii]))),
            "median_delta_t_years": float(np.median(arr["dt"][ii])),
            "median_reconstruction_mse_mv2": float(np.median(rec["mse_mv2"][ii])),
            "median_reconstruction_rmse_mv": float(np.median(rec["rmse_mv"][ii])),
            "median_relative_mse": float(np.median(rec["relative_mse"][ii])),
        }

    train_mu = mu[idx["train"]]
    latent_var = np.var(train_mu, axis=0, ddof=1)
    test_assoc = assoc[(assoc["split"] == "test") & (assoc["target_group"] == "delta_hcm")].copy()
    strongest = []
    for j in range(args.latent_dim):
        zname = f"D_z_{j:02d}"
        x = test_assoc[test_assoc["delta_latent"] == zname].copy()
        if len(x):
            x["abs_rho"] = x["spearman_rho"].abs()
            top = x.sort_values("abs_rho", ascending=False).iloc[0]
            strongest.append({
                "delta_latent": zname,
                "target": top["target"],
                "spearman_rho": float(top["spearman_rho"]),
                "p_value": float(top["p_value"]),
                "n": int(top["n"]),
            })

    summary = {
        "script": "19_train_longitudinal_delta_vae.py",
        "version": "1.0",
        "scientific_role": "HCM-only unsupervised representation of within-subject ECG mean-beat change",
        "training_population": "HCM subjects with repeated valid ECGs only",
        "normal_subjects_used_for_training": False,
        "n_hcm_visits_total": int(len(d["beats"])),
        "n_valid_hcm_visits": int(d["valid"].sum()),
        "n_hcm_subjects_with_pairs": int(len(np.unique(arr["subjects"]))),
        "n_pairs_total": int(len(rows)),
        "pair_mode": args.pair_mode,
        "input_mode": args.input_mode,
        "condition_on_delta_t": bool(args.condition_on_dt),
        "subject_wise_split": True,
        "latent_dim": int(args.latent_dim),
        "model_parameters": int(n_params),
        "beta": float(args.beta),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val),
        "splits": split_summary,
        "latent_train_variance": [float(x) for x in latent_var],
        "n_active_latents_variance_gt_0_01": int(np.sum(latent_var > 0.01)),
        "effective_latent_dimension_pr_train": effective_dim(train_mu),
        "strongest_test_association_with_hcm_latent_change_per_delta_latent": strongest,
        "interpretive_notes": [
            "VAE-3 is trained only on HCM within-subject longitudinal ECG changes.",
            "Normal ECGs are not part of VAE-3 training.",
            "Frozen normal and HCM latent coordinates are used only after training for interpretation.",
            "Consecutive visits are the primary pairing strategy.",
            "Train/validation/test splitting is subject-wise to prevent leakage.",
            "A longitudinal latent is a mode of ECG morphological change, not a clinical progression score.",
        ],
    }
    (args.results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 80)
    print("VAE-3 COMPLETE")
    print("=" * 80)
    print("HCM subjects with pairs :", summary["n_hcm_subjects_with_pairs"])
    print("Pairs                  :", summary["n_pairs_total"])
    print("Best epoch             :", best_epoch)
    print("Active latent dims     :", summary["n_active_latents_variance_gt_0_01"], "/", args.latent_dim)
    print("Effective dim (PR)     :", f"{summary['effective_latent_dimension_pr_train']:.3f}")
    print("Test median MSE        :", f"{split_summary['test']['median_reconstruction_mse_mv2']:.6g}", "mV^2")
    print("Results                :", args.results_dir)
    print("Output H5              :", args.output_h5)


if __name__ == "__main__":
    main()
