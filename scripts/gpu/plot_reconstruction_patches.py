#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.detection.dataset.finetune_dataset import FinetuneDataset
from src.detection.utils.config_io import load_config
from src.models.pretrain_reconstruction import CAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize CAE reconstruction patches for selected DAS windows.")
    p.add_argument("--base-cfg", required=True)
    p.add_argument("--stage-cfg", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--split-name", default="diagnostic")
    return p.parse_args()


def load_model(cfg, checkpoint: Path, device: torch.device) -> CAE:
    model = CAE(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def robust_limits(*arrays: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([a.reshape(-1) for a in arrays])
    lo, hi = np.percentile(vals, [1, 99])
    lim = max(abs(float(lo)), abs(float(hi)), 1e-6)
    return -lim, lim


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.base_cfg, args.stage_cfg)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = load_model(cfg, Path(args.checkpoint), device)

    df = pd.read_csv(args.csv)
    if len(df) == 0:
        raise ValueError(f"Empty CSV: {args.csv}")

    # Prefer a balanced diagnostic set when labels exist.
    picks = []
    rng = np.random.default_rng(args.seed)
    if "label" in df.columns:
        for _, g in df.groupby("label", sort=True):
            n = min(max(1, args.num_samples // max(df["label"].nunique(), 1)), len(g))
            picks.extend(rng.choice(g.index.to_numpy(), size=n, replace=False).tolist())
    if len(picks) < args.num_samples:
        remaining = np.array([i for i in df.index if i not in set(picks)])
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(args.num_samples - len(picks), len(remaining)), replace=False)
            picks.extend(extra.tolist())
    picks = picks[: args.num_samples]

    diag_csv = out_dir / "reconstruction_patch_metrics.csv"
    rows = []

    dataset = FinetuneDataset(
        csv_path=args.csv,
        normalize=cfg.data.normalize,
        preprocess=cfg.data.preprocess,
        add_channel_dim=True,
        return_meta=False,
    )

    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 2.2 * len(picks)), squeeze=False)
    for row_i, idx in enumerate(picks):
        x, y = dataset[int(idx)]
        x_in = x.unsqueeze(0).to(device)
        with torch.no_grad():
            x_hat, _, _ = model(x_in)
        orig = x_in.detach().cpu().numpy()[0, 0]
        recon = x_hat.detach().cpu().numpy()[0, 0]
        residual = recon - orig
        vmin, vmax = robust_limits(orig, recon)
        rlim = max(abs(np.percentile(residual, 1)), abs(np.percentile(residual, 99)), 1e-6)

        meta = df.iloc[int(idx)]
        title = f"idx={idx}, label={int(meta['label'])}" if "label" in meta.index else f"idx={idx}"
        panels = [
            ("Input", orig, vmin, vmax),
            ("Recon", recon, vmin, vmax),
            ("Residual", residual, -rlim, rlim),
        ]
        for col, (name, arr, lo, hi) in enumerate(panels):
            ax = axes[row_i, col]
            im = ax.imshow(arr, aspect="auto", cmap="seismic", vmin=lo, vmax=hi)
            ax.set_title(f"{title} | {name}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

        mse = float(np.mean(residual ** 2))
        mae = float(np.mean(np.abs(residual)))
        denom = float(np.mean(orig ** 2) + 1e-12)
        rows.append(
            {
                "row_index": int(idx),
                "label": int(meta["label"]) if "label" in meta.index else "",
                "label_name": str(meta.get("label_name", "")),
                "npy_path": str(meta.get("npy_path", "")),
                "mse": mse,
                "mae": mae,
                "relative_mse": mse / denom,
            }
        )

    fig.tight_layout()
    fig.savefig(out_dir / f"reconstruction_patches_{args.split_name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"reconstruction_patches_{args.split_name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(diag_csv, index=False)
    print(f"[WRITE] {out_dir / f'reconstruction_patches_{args.split_name}.pdf'}")
    print(f"[WRITE] {out_dir / f'reconstruction_patches_{args.split_name}.png'}")
    print(f"[WRITE] {diag_csv}")


if __name__ == "__main__":
    main()
