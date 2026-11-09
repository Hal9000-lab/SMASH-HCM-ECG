#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12_train_hcm_static_beta_vae.py

SMASH-HCM -- Model 2
Compact HCM-specific static beta-VAE.

SCIENTIFIC QUESTION
-------------------
After Model 1 has quantified deviation from normal physiology, Model 2 asks:

    What are the principal continuous axes of electrical heterogeneity WITHIN HCM?

The model is trained only on one baseline ECG per HCM subject.

Key design choices
------------------
- input: representative median beat [12,500]
- HCM-train only for optimization
- HCM-val only for model selection
- HCM-test held out
- Normal-test is evaluation only
- global scaling estimated only from HCM-train
- latent dimension = 8
- compact architecture (~237k parameters)
- beta = 0.01 with 30-epoch warmup
- KL mean across batch and latent dimensions
- deterministic decoder(mu) for exported reconstruction metrics

Expected input
--------------
Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5

Outputs
-------
Results/12_HCM_Static_Beta_VAE/
├── best_model.pt
├── training_history.csv
├── hcm_static_latents.csv
├── hcm_static_latents.npz
├── model_summary.json
├── group_reconstruction_summary.csv
├── latent_train_statistics.csv
└── training_curves.png
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEED = 20261004

SPLIT_HCM_TRAIN = 0
SPLIT_HCM_VAL = 1
SPLIT_HCM_TEST = 2
SPLIT_NORMAL_EVAL = 3

SPLIT_NAMES = {
    0: "hcm_train",
    1: "hcm_val",
    2: "hcm_test",
    3: "normal_eval",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("Results/11_HCM_Static_VAE_Preparation/Phase2_HCMStatic_preprocessed.h5"),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Results/12_HCM_Static_Beta_VAE"),
    )
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=30)
    p.add_argument("--max-epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--early-stop-patience", type=int, default=30)
    p.add_argument("--scheduler-patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ECGDataset(Dataset):
    def __init__(self, x, indices):
        self.x = x
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return torch.from_numpy(self.x[idx]).float(), idx


class HCMStaticBetaVAE(nn.Module):
    """
    Compact HCM-static architecture.

    Encoder temporal sizes:
        500 -> 250 -> 125 -> 62 -> 31

    Channels:
        12 -> 16 -> 32 -> 48 -> 64

    Latent:
        flatten 64*31 -> FC48 -> mu/logvar (default 8)

    Decoder:
        latent -> FC48 -> 64*31
        64 -> 48 -> 32 -> 16 -> 12
        31 -> 62 -> 125 -> 250 -> 500
    """

    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(12, 16, 4, 2, 1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),

            nn.Conv1d(16, 32, 4, 2, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.Conv1d(32, 48, 4, 2, 1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),

            nn.Conv1d(48, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 31, 48),
            nn.SiLU(),
        )

        self.fc_mu = nn.Linear(48, latent_dim)
        self.fc_logvar = nn.Linear(48, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 64 * 31),
            nn.SiLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 48, 4, 2, 1, output_padding=0),  # 31 -> 62
            nn.GroupNorm(8, 48),
            nn.SiLU(),

            nn.ConvTranspose1d(48, 32, 4, 2, 1, output_padding=1),  # 62 -> 125
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.ConvTranspose1d(32, 16, 4, 2, 1, output_padding=0),  # 125 -> 250
            nn.GroupNorm(4, 16),
            nn.SiLU(),

            nn.ConvTranspose1d(16, 12, 4, 2, 1, output_padding=0),  # 250 -> 500
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(z.shape[0], 64, 31)
        return self.decoder_conv(h)

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = torch.mean((recon - x) ** 2)
    kl_per_latent = -0.5 * (
        1.0 + logvar - mu.pow(2) - logvar.exp()
    )
    kl_loss = torch.mean(kl_per_latent)
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def run_epoch(
    model,
    loader,
    device,
    beta,
    optimizer=None,
    grad_clip=5.0,
    deterministic_eval=True,
):
    training = optimizer is not None
    model.train(training)

    totals = []
    recs = []
    kls = []
    ns = []

    for xb, _ in loader:
        xb = xb.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        recon, mu, logvar = model(
            xb,
            deterministic=(not training and deterministic_eval),
        )
        total, rec, kl = vae_loss(recon, xb, mu, logvar, beta)

        if training:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        b = xb.shape[0]
        totals.append(float(total.detach().cpu()) * b)
        recs.append(float(rec.detach().cpu()) * b)
        kls.append(float(kl.detach().cpu()) * b)
        ns.append(b)

    n = sum(ns)
    return {
        "loss": sum(totals) / n,
        "recon": sum(recs) / n,
        "kl": sum(kls) / n,
    }


@torch.inference_mode()
def encode_all(model, x_scaled, batch_size, device):
    model.eval()
    n = len(x_scaled)

    mus = []
    logvars = []
    recon_scaled = np.empty_like(x_scaled, dtype=np.float32)

    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        xb = torch.from_numpy(x_scaled[start:stop]).to(device).float()
        mu, logvar = model.encode(xb)
        y = model.decode(mu)

        mus.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
        recon_scaled[start:stop] = y.cpu().numpy().astype(np.float32)

    return (
        np.concatenate(mus, axis=0),
        np.concatenate(logvars, axis=0),
        recon_scaled,
    )


def participation_ratio(eigvals):
    e = np.asarray(eigvals, dtype=float)
    e = e[e > 0]
    if len(e) == 0:
        return np.nan
    return float((e.sum() ** 2) / np.sum(e ** 2))


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("12 - Train HCM-static beta-VAE")
    print("=" * 78)
    print("Input      :", args.input)
    print("Results    :", args.results_dir)
    print("Device     :", device)
    print("Latent dim :", args.latent_dim)
    print("Beta       :", args.beta)

    with h5py.File(args.input, "r") as h5:
        x_scaled = np.asarray(h5["median_all_scaled"], dtype=np.float32)
        x_mv = np.asarray(h5["median_all_mv"], dtype=np.float32)
        split = np.asarray(h5["split_code"], dtype=np.int8)
        phase1_row_index = np.asarray(h5["phase1_row_index"], dtype=np.int64)

        attrs = {k: h5.attrs[k] for k in h5.attrs.keys()}

        metadata = {}
        n = len(x_scaled)
        for key, ds in h5.items():
            if key in {
                "median_all_scaled", "median_all_mv",
                "split_code", "phase1_row_index"
            }:
                continue
            arr = np.asarray(ds)
            if arr.ndim == 1 and len(arr) == n:
                if arr.dtype.kind == "S":
                    arr = np.asarray([
                        x.decode("utf-8", errors="replace") for x in arr
                    ], dtype=object)
                metadata[key] = arr

    idx_train = np.where(split == SPLIT_HCM_TRAIN)[0]
    idx_val = np.where(split == SPLIT_HCM_VAL)[0]
    idx_test = np.where(split == SPLIT_HCM_TEST)[0]
    idx_normal = np.where(split == SPLIT_NORMAL_EVAL)[0]

    print("[splits]")
    print("  HCM train  :", len(idx_train))
    print("  HCM val    :", len(idx_val))
    print("  HCM test   :", len(idx_test))
    print("  Normal eval:", len(idx_normal))

    if len(idx_train) < 300 or len(idx_val) < 50:
        raise RuntimeError("HCM train/validation split unexpectedly small.")

    train_loader = DataLoader(
        ECGDataset(x_scaled, idx_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        ECGDataset(x_scaled, idx_val),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        ECGDataset(x_scaled, idx_test),
        batch_size=args.batch_size,
        shuffle=False,
    )
    normal_loader = (
        DataLoader(
            ECGDataset(x_scaled, idx_normal),
            batch_size=args.batch_size,
            shuffle=False,
        )
        if len(idx_normal) else None
    )

    model = HCMStaticBetaVAE(latent_dim=args.latent_dim).to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {nparams:,}")

    if args.latent_dim == 8 and nparams != 236988:
        raise RuntimeError(
            f"Unexpected parameter count for latent_dim=8: {nparams:,}; "
            "expected 236,988."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
    )

    history = []
    best_val = math.inf
    best_epoch = None
    bad_epochs = 0
    ckpt_path = args.results_dir / "best_model.pt"

    for epoch in range(1, args.max_epochs + 1):
        beta_epoch = args.beta * min(1.0, epoch / args.warmup_epochs)

        train_m = run_epoch(
            model, train_loader, device,
            beta=beta_epoch,
            optimizer=optimizer,
        )
        val_m = run_epoch(
            model, val_loader, device,
            beta=args.beta,
            optimizer=None,
            deterministic_eval=True,
        )

        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "beta_train": beta_epoch,
            "lr": lr,
            "train_loss": train_m["loss"],
            "train_recon": train_m["recon"],
            "train_kl": train_m["kl"],
            "val_loss_fixed_beta": val_m["loss"],
            "val_recon": val_m["recon"],
            "val_kl": val_m["kl"],
        }
        history.append(row)

        eligible = epoch >= args.warmup_epochs

        if eligible:
            scheduler.step(val_m["loss"])

            if val_m["loss"] < best_val - 1e-7:
                best_val = val_m["loss"]
                best_epoch = epoch
                bad_epochs = 0

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "latent_dim": args.latent_dim,
                    "beta": args.beta,
                    "architecture": "hcm_static_compact_v1",
                    "model_parameters": nparams,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "seed": args.seed,
                }, ckpt_path)
            else:
                bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or eligible and bad_epochs == 0:
            print(
                f"Epoch {epoch:3d} | beta={beta_epoch:.5f} | "
                f"train={train_m['loss']:.5f} "
                f"(rec={train_m['recon']:.5f}, kl={train_m['kl']:.5f}) | "
                f"val={val_m['loss']:.5f} "
                f"(rec={val_m['recon']:.5f}, kl={val_m['kl']:.5f}) | "
                f"lr={lr:.2e}"
            )

        if eligible and bad_epochs >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, best val={best_val:.6f}"
            )
            break

    if not ckpt_path.exists():
        raise RuntimeError("No best checkpoint was saved.")

    hist = pd.DataFrame(history)
    hist.to_csv(args.results_dir / "training_history.csv", index=False)

    # Reload best checkpoint.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Held-out evaluation with fixed beta and deterministic reconstruction.
    test_m = run_epoch(
        model, test_loader, device,
        beta=args.beta,
        optimizer=None,
        deterministic_eval=True,
    )
    normal_m = (
        run_epoch(
            model, normal_loader, device,
            beta=args.beta,
            optimizer=None,
            deterministic_eval=True,
        )
        if normal_loader is not None else None
    )

    # Encode all prepared records.
    mu, logvar, recon_scaled = encode_all(
        model, x_scaled, args.batch_size, device
    )

    scale_mean = float(attrs["hcm_train_scaling_mean_mv"])
    scale_std = float(attrs["hcm_train_scaling_std_mv"])

    recon_mv = recon_scaled * scale_std + scale_mean
    mse_scaled = np.mean((x_scaled - recon_scaled) ** 2, axis=(1,2))
    mse_mv2 = np.mean((x_mv - recon_mv) ** 2, axis=(1,2))
    rmse_mv = np.sqrt(mse_mv2)

    out = pd.DataFrame({
        "row_index": np.arange(len(x_scaled)),
        "phase1_row_index": phase1_row_index,
        "split_code": split,
        "split_name": [SPLIT_NAMES[int(s)] for s in split],
        "recon_mse_scaled": mse_scaled,
        "recon_mse_mv2": mse_mv2,
        "recon_rmse_mv": rmse_mv,
    })

    # Copy useful metadata.
    for key in [
        "ecg_id", "subject_id", "patient_id",
        "age", "sex", "gender", "source_group"
    ]:
        if key in metadata:
            out[key] = metadata[key]

    for k in range(args.latent_dim):
        out[f"z_{k:02d}"] = mu[:, k]
        out[f"logvar_{k:02d}"] = logvar[:, k]

    out.to_csv(args.results_dir / "hcm_static_latents.csv", index=False)

    np.savez_compressed(
        args.results_dir / "hcm_static_latents.npz",
        mu=mu.astype(np.float32),
        logvar=logvar.astype(np.float32),
        split_code=split,
        phase1_row_index=phase1_row_index,
        recon_mse_scaled=mse_scaled.astype(np.float32),
        recon_mse_mv2=mse_mv2.astype(np.float32),
        recon_rmse_mv=rmse_mv.astype(np.float32),
    )

    # Group reconstruction summary.
    group_rows = []
    for code, name in SPLIT_NAMES.items():
        m = split == code
        if not np.any(m):
            continue
        vals = mse_mv2[m]
        group_rows.append({
            "group": name,
            "n": int(m.sum()),
            "recon_mse_mv2_mean": float(np.mean(vals)),
            "recon_mse_mv2_median": float(np.median(vals)),
            "recon_mse_mv2_q25": float(np.quantile(vals, 0.25)),
            "recon_mse_mv2_q75": float(np.quantile(vals, 0.75)),
            "recon_rmse_mv_median": float(np.median(np.sqrt(vals))),
        })
    pd.DataFrame(group_rows).to_csv(
        args.results_dir / "group_reconstruction_summary.csv",
        index=False,
    )

    # HCM-train latent statistics.
    z_train = mu[idx_train]
    cov = np.cov(z_train, rowvar=False)
    eig = np.linalg.eigvalsh(cov)[::-1]
    eff_dim = participation_ratio(eig)

    stats_rows = []
    for k in range(args.latent_dim):
        vals = z_train[:, k]
        stats_rows.append({
            "latent": f"z_{k:02d}",
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)),
            "variance": float(np.var(vals, ddof=1)),
            "active_var_gt_0p01": bool(np.var(vals, ddof=1) > 0.01),
            "mean_logvar": float(np.mean(logvar[idx_train, k])),
            "mean_posterior_variance": float(
                np.mean(np.exp(logvar[idx_train, k]))
            ),
        })
    pd.DataFrame(stats_rows).to_csv(
        args.results_dir / "latent_train_statistics.csv",
        index=False,
    )

    # Training curves.
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(hist["epoch"], hist["train_loss"], label="HCM train")
    ax.plot(hist["epoch"], hist["val_loss_fixed_beta"], label="HCM val")
    if best_epoch is not None:
        ax.axvline(best_epoch, linestyle="--", linewidth=1, label="best epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VAE loss")
    ax.set_title("HCM-static beta-VAE training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.results_dir / "training_curves.png", dpi=180)
    plt.close(fig)

    summary = {
        "script": "12_train_hcm_static_beta_vae.py",
        "design": "HCM-static disease-specific median-beat beta-VAE",
        "architecture": "hcm_static_compact_v1",
        "scientific_question": (
            "principal continuous axes of electrical heterogeneity within HCM"
        ),
        "input_shape": [12, 500],
        "latent_dim": int(args.latent_dim),
        "beta_target": float(args.beta),
        "kl_definition": "mean KL across batch and latent dimensions",
        "kl_warmup_epochs": int(args.warmup_epochs),
        "selection_starts_at_epoch": int(args.warmup_epochs),
        "trainable_parameters": int(nparams),
        "device": str(device),
        "splits": {
            "hcm_train": int(len(idx_train)),
            "hcm_val": int(len(idx_val)),
            "hcm_test": int(len(idx_test)),
            "normal_eval": int(len(idx_normal)),
        },
        "hcm_used_for_training": True,
        "normal_used_for_training": False,
        "normal_used_for_model_selection": False,
        "best_epoch": int(best_epoch),
        "best_hcm_val_loss": float(best_val),
        "hcm_test_loss": float(test_m["loss"]),
        "hcm_test_recon": float(test_m["recon"]),
        "hcm_test_kl": float(test_m["kl"]),
        "normal_eval_loss": None if normal_m is None else float(normal_m["loss"]),
        "normal_eval_recon": None if normal_m is None else float(normal_m["recon"]),
        "normal_eval_kl": None if normal_m is None else float(normal_m["kl"]),
        "hcm_train_scaling_mean_mv": scale_mean,
        "hcm_train_scaling_std_mv": scale_std,
        "active_latent_dimensions_var_gt_0p01": int(
            sum(r["active_var_gt_0p01"] for r in stats_rows)
        ),
        "latent_effective_dimension_participation_ratio": eff_dim,
        "latent_covariance_eigenvalues": eig.tolist(),
        "checkpoint": str(ckpt_path),
        "interpretive_caution": (
            "This model characterizes within-HCM morphology. "
            "Normal-evaluation results are secondary sanity checks and not "
            "a diagnostic endpoint."
        ),
    }

    (args.results_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("HCM-static beta-VAE complete")
    print("=" * 78)
    print("Best epoch       :", best_epoch)
    print("Best HCM-val loss:", f"{best_val:.6f}")
    print("HCM-test loss    :", f"{test_m['loss']:.6f}")
    print("HCM-test recon   :", f"{test_m['recon']:.6f}")
    print("HCM-test KL      :", f"{test_m['kl']:.6f}")
    if normal_m is not None:
        print("Normal-eval recon:", f"{normal_m['recon']:.6f}")
    print(
        "Active latents    :",
        sum(r["active_var_gt_0p01"] for r in stats_rows),
        "/",
        args.latent_dim,
    )
    print("Effective dim     :", f"{eff_dim:.3f}")
    print("Results           :", args.results_dir)


if __name__ == "__main__":
    main()
