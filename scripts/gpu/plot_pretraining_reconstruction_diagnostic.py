#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.pretrain_reconstruction import CAE

OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "leftwing"

RUN_DIR = (
    ROOT
    / "runs"
    / "metadata_v2_safe_rerun_v1"
    / "logenv_site_main_pre50_v2"
    / "utah_2019_reconst_reconst_noanom"
    / "reconst"
    / "pretrain"
    / "base_utah_2019"
)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def load_checkpoint_model(ckpt_path: Path) -> tuple[CAE, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = CAE(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def load_window(npy_path: str | Path) -> torch.Tensor:
    arr = np.load(resolve_path(npy_path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def forward_reconstruction(model: CAE, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        x_hat = out.get("x_hat", out.get("recon"))
        z = out["z"]
    else:
        a, b = out[0], out[1]
        if tuple(a.shape) == tuple(x.shape):
            x_hat, z = a, b
        else:
            z, x_hat = a, b
    x_np = x.squeeze().cpu().numpy()
    x_hat_np = x_hat.squeeze().cpu().numpy()
    residual = x_np - x_hat_np
    z_np = z.squeeze(0).cpu().numpy()
    return x_np, x_hat_np, residual, z_np


def compute_latent_norms(model: CAE, csv_path: Path, max_samples: int = 128) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)
    norms = []
    with torch.no_grad():
        for npy_path in df["npy_path"].tolist():
            x = load_window(npy_path)
            out = model(x)
            if isinstance(out, dict):
                z = out["z"]
            else:
                a, b = out[0], out[1]
                z = b if tuple(a.shape) == tuple(x.shape) else a
            norms.append(float(torch.linalg.vector_norm(z.squeeze(0)).item()))
    return np.asarray(norms, dtype=np.float32)


def robust_limits(*arrays: np.ndarray, percentile: float = 99.5) -> tuple[float, float]:
    values = np.concatenate([a.ravel() for a in arrays])
    vmax = float(np.nanpercentile(np.abs(values), percentile))
    vmax = max(vmax, 1e-8)
    return -vmax, vmax


def plot_image(ax: plt.Axes, arr: np.ndarray, title: str, vmin: float, vmax: float, cmap: str = "RdBu_r"):
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title, fontsize=10.5, fontweight="normal", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def save_single_loss(history: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.0))
    ax.plot(history["epoch"], history["loss"], color="#0f766e", linewidth=2.6)
    ax.scatter(history["epoch"].iloc[-1], history["loss"].iloc[-1], color="#0f766e", s=30, zorder=3)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("MSE loss", fontsize=10)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9.5)
    ax.text(
        0.97,
        0.90,
        f"final: {history['loss'].iloc[-1]:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color="#374151",
    )
    fig.tight_layout()
    out_base = OUT_DIR / "pretraining_reconstruction_loss_logenv_utah2019"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")


def save_single_window(arr: np.ndarray, stem: str, vmin: float, vmax: float, cmap: str = "RdBu_r") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.05))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.1)
    out_base = OUT_DIR / stem
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(RUN_DIR / "train_history.csv")
    model, cfg = load_checkpoint_model(RUN_DIR / "best.pt")
    pretrain_csv = resolve_path(Path(cfg["data"]["split_dir"]) / cfg["data"].get("pretrain_csv", "pretrain.csv"))
    event_csv = resolve_path(Path(cfg["data"]["split_dir"]) / "test.csv")
    event_df = pd.read_csv(event_csv)
    if "label" in event_df.columns and (event_df["label"] == 1).any():
        sample_row = event_df[event_df["label"] == 1].iloc[0]
    else:
        sample_row = event_df.iloc[0]

    x = load_window(sample_row["npy_path"])
    x_np, x_hat_np, residual, z_np = forward_reconstruction(model, x)
    latent_norms = compute_latent_norms(model, pretrain_csv, max_samples=128)
    sample_mse = float(np.mean((x_np - x_hat_np) ** 2))
    sample_l2 = float(np.linalg.norm(z_np))

    vmin, vmax = robust_limits(x_np, x_hat_np)
    rmin, rmax = robust_limits(residual, percentile=99.0)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(10.8, 5.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1.1, 1.35, 1.35], hspace=0.38, wspace=0.38)

    ax_loss = fig.add_subplot(gs[0, 0:2])
    ax_loss.plot(history["epoch"], history["loss"], color="#0f766e", linewidth=2.4)
    ax_loss.scatter(history["epoch"].iloc[-1], history["loss"].iloc[-1], color="#0f766e", s=26, zorder=3)
    ax_loss.set_title("Reconstruction pretraining loss", fontsize=11, fontweight="normal", pad=8)
    ax_loss.set_xlabel("Epoch", fontsize=9.5)
    ax_loss.set_ylabel("MSE loss", fontsize=9.5)
    ax_loss.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    ax_loss.tick_params(labelsize=9)
    ax_loss.text(
        0.98,
        0.92,
        f"final: {history['loss'].iloc[-1]:.4f}",
        transform=ax_loss.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#374151",
    )

    ax_norm = fig.add_subplot(gs[1, 0:2])
    ax_norm.hist(latent_norms, bins=28, color="#1f2937", alpha=0.86, edgecolor="white", linewidth=0.4)
    ax_norm.axvline(float(latent_norms.mean()), color="#d97706", linewidth=2.2, label=f"mean={latent_norms.mean():.2f}")
    ax_norm.axvline(sample_l2, color="#0f766e", linewidth=2.0, linestyle="--", label=f"sample={sample_l2:.2f}")
    ax_norm.set_title("Pretrained latent L2 norm", fontsize=11, fontweight="normal", pad=8)
    ax_norm.set_xlabel(r"$\|z\|_2$", fontsize=9.5)
    ax_norm.set_ylabel("Count", fontsize=9.5)
    ax_norm.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax_norm.spines["top"].set_visible(False)
    ax_norm.spines["right"].set_visible(False)
    ax_norm.legend(frameon=False, fontsize=8.8, loc="upper right")
    ax_norm.tick_params(labelsize=9)

    ax_input = fig.add_subplot(gs[0, 2])
    ax_recon = fig.add_subplot(gs[0, 3])
    ax_resid = fig.add_subplot(gs[1, 2:4])
    im = plot_image(ax_input, x_np, "Input window", vmin, vmax)
    plot_image(ax_recon, x_hat_np, "Reconstruction", vmin, vmax)
    im_res = plot_image(ax_resid, residual, f"Residual  |  MSE={sample_mse:.4f}", rmin, rmax)

    cbar = fig.colorbar(im, ax=[ax_input, ax_recon], fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar_res = fig.colorbar(im_res, ax=ax_resid, fraction=0.046, pad=0.02)
    cbar_res.ax.tick_params(labelsize=8)

    out_base = OUT_DIR / "pretraining_reconstruction_diagnostic_logenv_utah2019"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    save_single_loss(history)
    save_single_window(
        x_np,
        "pretraining_reconstruction_input_window_logenv_utah2019",
        vmin,
        vmax,
    )
    save_single_window(
        x_hat_np,
        "pretraining_reconstruction_reconstructed_window_logenv_utah2019",
        vmin,
        vmax,
    )
    save_single_window(
        residual,
        "pretraining_reconstruction_residual_window_logenv_utah2019",
        rmin,
        rmax,
    )

    metrics = {
        "run_dir": str(RUN_DIR.relative_to(ROOT)),
        "preprocessing": "log-envelope",
        "site": "Utah 2019",
        "best_epoch": int(torch.load(RUN_DIR / "best.pt", map_location="cpu")["epoch"]),
        "best_loss": float(torch.load(RUN_DIR / "best.pt", map_location="cpu")["best_loss"]),
        "final_history_loss": float(history["loss"].iloc[-1]),
        "latent_norm_mean": float(latent_norms.mean()),
        "latent_norm_std": float(latent_norms.std()),
        "sample_latent_l2_norm": sample_l2,
        "sample_reconstruction_mse": sample_mse,
        "sample_npy_path": str(sample_row["npy_path"]),
    }
    (OUT_DIR / "pretraining_reconstruction_diagnostic_logenv_utah2019_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")
    print(f"[DONE] wrote {OUT_DIR / 'pretraining_reconstruction_diagnostic_logenv_utah2019_metrics.json'}")


if __name__ == "__main__":
    main()
