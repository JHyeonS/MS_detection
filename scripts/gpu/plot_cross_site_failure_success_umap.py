#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml


_original_njit = numba.njit
_original_vectorize = numba.vectorize


def _njit_without_cache(*args, **kwargs):
    kwargs["cache"] = False
    return _original_njit(*args, **kwargs)


numba.njit = _njit_without_cache


def _vectorize_without_cache(*args, **kwargs):
    kwargs["cache"] = False
    return _original_vectorize(*args, **kwargs)


numba.vectorize = _vectorize_without_cache
from umap import UMAP


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cnn_encoder import cnn_encoder
from src.detection.utils.config_io import cfg_get


OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "leftwing"

RUNS = [
    {
        "name": "failure_lowpass_rms",
        "panel_title": "Failure: Low-pass + RMS",
        "run_dir": ROOT
        / "runs"
        / "metadata_v2_safe_rerun_v1"
        / "filter_rms_cross_site_reconst_swd_interval10_v1"
        / "utah_2019_to_utah_2023"
        / "reconst",
    },
    {
        "name": "success_logenv",
        "panel_title": "Recovery: Log-envelope",
        "run_dir": ROOT
        / "runs"
        / "metadata_v2_safe_rerun_v1"
        / "logenv_cross_site_reconst_swd_interval10_v1"
        / "utah_2019_to_utah_2023"
        / "reconst",
    },
]

EXPERIMENT = "base_utah_2023__frac0p25"
FRACTION_LABEL = "25%"


class FinetuneMSDNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = cnn_encoder(cfg)
        self.latent_dim = int(cfg_get(cfg, "model", "encoder", "latent_dim", default=128))
        self.head = nn.Linear(self.latent_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        logit = self.head(z)
        return z, logit


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(run_dir: Path) -> tuple[FinetuneMSDNet, dict]:
    cfg = load_yaml(run_dir / "test" / EXPERIMENT / "merged_config.yaml")
    model = FinetuneMSDNet(cfg)
    ckpt_path = run_dir / "finetune" / EXPERIMENT / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def load_batch(paths: list[str]) -> torch.Tensor:
    arrays = []
    for path in paths:
        arr = np.load(resolve_path(path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        arrays.append(arr)
    x = torch.from_numpy(np.stack(arrays, axis=0)).unsqueeze(1)
    return x


def extract_features(run_spec: dict, batch_size: int = 16) -> pd.DataFrame:
    run_dir = run_spec["run_dir"]
    model, _ = load_model(run_dir)
    pred_path = run_dir / "test" / EXPERIMENT / "test_predictions.csv"
    pred = pd.read_csv(pred_path)

    features = []
    with torch.no_grad():
        paths = pred["meta_npy_path"].tolist()
        for start in range(0, len(paths), batch_size):
            x = load_batch(paths[start : start + batch_size])
            z, _ = model(x)
            features.append(z.cpu().numpy())
    feat = np.concatenate(features, axis=0)

    out = pred.copy()
    out["feat"] = list(feat)
    out["setting"] = run_spec["name"]
    out["panel_title"] = run_spec["panel_title"]
    out["is_correct"] = out["label"].astype(int) == out["pred_fc"].astype(int)
    return out


def compute_umap(df: pd.DataFrame) -> pd.DataFrame:
    x = np.stack(df["feat"].to_numpy()).astype(np.float32)
    reducer = UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.12,
        metric="euclidean",
        random_state=42,
    )
    emb = reducer.fit_transform(x)
    out = df.drop(columns=["feat"]).copy()
    out["umap_x"] = emb[:, 0]
    out["umap_y"] = emb[:, 1]
    return out


def metric_summary(run_dir: Path) -> dict:
    metric_path = run_dir / "test" / EXPERIMENT / "test_metrics_fixed_threshold.json"
    with metric_path.open("r", encoding="utf-8") as f:
        return json.load(f)["fc_metrics_fixed_threshold"]


def style_umap_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    metrics: dict,
    show_legend: bool = False,
    clean: bool = False,
) -> None:
    colors = {0: "#4b5563", 1: "#d97706"}
    labels = {0: "Noise", 1: "Event"}
    for label in (0, 1):
        sub = df[df["label"].astype(int) == label]
        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=11 if clean else 18,
            alpha=0.62 if clean else 0.72,
            color=colors[label],
            label=labels[label],
            linewidths=0,
        )
    wrong = df[~df["is_correct"]]
    if not wrong.empty:
        ax.scatter(
            wrong["umap_x"],
            wrong["umap_y"],
            s=26 if clean else 34,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.65 if clean else 0.8,
            label="Misclassified",
        )
    if clean:
        ax.set_title(title.replace(": ", "\n"), fontsize=10.2, fontweight="normal", pad=6, color="#111827")
        ax.set_xlabel(
            f"BalAcc {metrics['balanced_acc']:.3f}  |  Spec {metrics['specificity']:.3f}",
            fontsize=8.4,
            color="#6b7280",
            labelpad=8,
        )
    else:
        ax.set_title(
            f"{title}\nBalAcc={metrics['balanced_acc']:.3f}, Spec={metrics['specificity']:.3f}",
            fontsize=10.5,
            fontweight="normal",
            pad=8,
        )
    style_umap_axis(ax)
    if show_legend:
        ax.legend(
            frameon=False,
            loc="lower left",
            fontsize=8.5,
            handletextpad=0.3,
            markerscale=1.2,
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    embedded = []
    for run in RUNS:
        features = extract_features(run)
        emb = compute_umap(features)
        emb["preprocessing_panel"] = run["panel_title"]
        embedded.append(emb)
    all_emb = pd.concat(embedded, ignore_index=True)
    all_emb.to_csv(OUT_DIR / "cross_site_failure_success_umap_utah2019_to_utah2023_frac0p25.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, run, emb in zip(axes, RUNS, embedded):
        metrics = metric_summary(run["run_dir"])
        plot_panel(ax, emb, run["panel_title"], metrics, show_legend=(ax is axes[0]))
    fig.tight_layout(w_pad=1.0)
    out_base = OUT_DIR / "cross_site_failure_success_umap_utah2019_to_utah2023_frac0p25"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.15))
    for ax, run, emb in zip(axes, RUNS, embedded):
        metrics = metric_summary(run["run_dir"])
        plot_panel(ax, emb, run["panel_title"], metrics, show_legend=False, clean=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=8.6,
        handletextpad=0.25,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=[0.0, 0.105, 1.0, 1.0], w_pad=0.65)
    out_base = OUT_DIR / "cross_site_failure_success_umap_clean_utah2019_to_utah2023_frac0p25"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    for run, emb in zip(RUNS, embedded):
        fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.25))
        plot_panel(ax, emb, run["panel_title"], metric_summary(run["run_dir"]), show_legend=True)
        fig.tight_layout()
        out_base = OUT_DIR / f"cross_site_{run['name']}_umap_utah2019_to_utah2023_frac0p25"
        for ext in ("pdf", "png"):
            fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"[DONE] wrote {OUT_DIR / 'cross_site_failure_success_umap_utah2019_to_utah2023_frac0p25.pdf'}")
    print(f"[DONE] wrote {OUT_DIR / 'cross_site_failure_success_umap_utah2019_to_utah2023_frac0p25.png'}")


if __name__ == "__main__":
    main()
