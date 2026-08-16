#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.utils.config_io import cfg_get
from src.models.cnn_encoder import cnn_encoder


RUN_BASE = ROOT / "runs" / "metadata_v2_safe_rerun_v1"
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "center" / "cross_site_classwise_swd"
DEVICE = torch.device(os.environ.get("CLASSWISE_SWD_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))

PREPROCESSING = {
    "filter_rms": {
        "label": "Low-pass + RMS",
        "color": "#bf4b3e",
        "run_root": RUN_BASE / "filter_rms_cross_site_reconst_swd_interval10_v1",
        "data_root": ROOT / "data" / "visualbest_filter_rms_fs1000_rms0p15_lp50",
    },
    "logenv": {
        "label": "Log-envelope",
        "color": "#376795",
        "run_root": RUN_BASE / "logenv_cross_site_reconst_swd_interval10_v1",
        "data_root": ROOT / "data" / "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5",
    },
}

SITE_SHORT = {"pohang": "A", "utah_2019": "B", "utah_2023": "C"}
SITE_LABEL = {"pohang": "Pohang", "utah_2019": "Utah 2019", "utah_2023": "Utah 2023"}
FRACTION_ORDER = ["0p05", "0p1", "0p25", "0p5", "1"]
FRACTION_LABEL = {"0p05": "0.05", "0p1": "0.10", "0p25": "0.25", "0p5": "0.50", "1": "1.00"}


class FinetuneMSDNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = cnn_encoder(cfg)
        latent_dim = int(cfg_get(cfg, "model", "encoder", "latent_dim", default=128))
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.head(z)


def parse_fraction(name: str) -> str | None:
    match = re.search(r"__frac([0-9p]+)$", name)
    if not match:
        return None
    tag = match.group(1)
    return tag if tag in FRACTION_LABEL else None


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(run_dir: Path, experiment: str) -> FinetuneMSDNet:
    cfg = load_yaml(run_dir / "test" / experiment / "merged_config.yaml")
    model = FinetuneMSDNet(cfg)
    ckpt = torch.load(run_dir / "finetune" / experiment / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def load_batch(paths: list[str]) -> torch.Tensor:
    arrays = []
    for path in paths:
        arr = np.load(resolve_path(path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        arrays.append(arr)
    return torch.from_numpy(np.stack(arrays, axis=0)).unsqueeze(1)


def load_site_tensor(df: pd.DataFrame) -> torch.Tensor:
    return load_batch(df["npy_path"].astype(str).tolist())


def extract_latents(model: FinetuneMSDNet, x_all: torch.Tensor, batch_size: int = 64) -> np.ndarray:
    features = []
    with torch.no_grad():
        for start in range(0, len(x_all), batch_size):
            x = x_all[start : start + batch_size].to(DEVICE, non_blocking=True)
            z, _ = model(x)
            features.append(z.cpu().numpy())
    return np.concatenate(features, axis=0)


def load_site_test(data_root: Path, site: str) -> pd.DataFrame:
    path = data_root / "metadata" / "experiments" / f"stage1_{site}_only" / "test.csv"
    df = pd.read_csv(path)
    df = df[df["label"].isin([0, 1])].copy()
    df["site"] = site
    return df


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int = 128, seed: int = 42) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    dirs = rng.normal(size=(n_proj, dim)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    x_proj = x @ dirs.T
    y_proj = y @ dirs.T
    distances = [wasserstein_distance(x_proj[:, i], y_proj[:, i]) for i in range(n_proj)]
    return float(np.mean(distances))


def standardize_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    both = np.concatenate([x, y], axis=0)
    mu = both.mean(axis=0, keepdims=True)
    sigma = both.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (x - mu) / sigma, (y - mu) / sigma


def class_swd(source_feat: np.ndarray, source_df: pd.DataFrame, target_feat: np.ndarray, target_df: pd.DataFrame, label: int | None) -> float:
    if label is None:
        sx = source_feat
        tx = target_feat
    else:
        sx = source_feat[source_df["label"].to_numpy().astype(int) == label]
        tx = target_feat[target_df["label"].to_numpy().astype(int) == label]
    sx, tx = standardize_pair(sx, tx)
    return sliced_wasserstein(sx, tx)


def metric_summary(run_dir: Path, experiment: str) -> dict:
    path = run_dir / "test" / experiment / "test_metrics_fixed_threshold.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fc_metrics_fixed_threshold", {})


def collect() -> pd.DataFrame:
    rows = []
    print(f"[INFO] device={DEVICE}", flush=True)
    for preproc, spec in PREPROCESSING.items():
        run_root = spec["run_root"]
        data_root = spec["data_root"]
        if not run_root.exists():
            continue
        site_cache: dict[str, pd.DataFrame] = {}
        tensor_cache: dict[str, torch.Tensor] = {}
        for direction_dir in sorted(run_root.glob("*_to_*")):
            if not direction_dir.is_dir():
                continue
            source, target = direction_dir.name.split("_to_", 1)
            run_dir = direction_dir / "reconst"
            if not run_dir.exists():
                continue
            source_df = site_cache.setdefault(source, load_site_test(data_root, source))
            target_df = site_cache.setdefault(target, load_site_test(data_root, target))
            if source not in tensor_cache:
                tensor_cache[source] = load_site_tensor(source_df)
            if target not in tensor_cache:
                tensor_cache[target] = load_site_tensor(target_df)
            for finetune_dir in sorted((run_dir / "finetune").glob("*__frac*")):
                fraction_tag = parse_fraction(finetune_dir.name)
                if fraction_tag is None:
                    continue
                experiment = finetune_dir.name
                if not (run_dir / "test" / experiment / "merged_config.yaml").exists():
                    continue
                model = load_model(run_dir, experiment)
                source_feat = extract_latents(model, tensor_cache[source])
                target_feat = extract_latents(model, tensor_cache[target])
                metrics = metric_summary(run_dir, experiment)
                rows.append(
                    {
                        "preprocessing": preproc,
                        "preprocessing_label": spec["label"],
                        "source_site": source,
                        "target_site": target,
                        "direction": f"{source}_to_{target}",
                        "direction_short": f"{SITE_SHORT[source]}->{SITE_SHORT[target]}",
                        "fraction_tag": fraction_tag,
                        "fraction": float(FRACTION_LABEL[fraction_tag]),
                        "event_site_swd": class_swd(source_feat, source_df, target_feat, target_df, 1),
                        "noise_site_swd": class_swd(source_feat, source_df, target_feat, target_df, 0),
                        "all_site_swd": class_swd(source_feat, source_df, target_feat, target_df, None),
                        "target_balanced_acc": metrics.get("balanced_acc", np.nan),
                        "target_f1": metrics.get("f1", np.nan),
                        "target_specificity": metrics.get("specificity", np.nan),
                        "source_n_event": int((source_df["label"].astype(int) == 1).sum()),
                        "source_n_noise": int((source_df["label"].astype(int) == 0).sum()),
                        "target_n_event": int((target_df["label"].astype(int) == 1).sum()),
                        "target_n_noise": int((target_df["label"].astype(int) == 0).sum()),
                        "run_dir": str(run_dir.relative_to(ROOT)),
                    }
                )
                print(f"[DONE] {preproc} {source}->{target} frac={FRACTION_LABEL[fraction_tag]}", flush=True)
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d7d2c8", linewidth=0.8, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_summary(df: pd.DataFrame) -> None:
    final = df.copy()
    metric_cols = [
        ("event_site_swd", "Event-domain SWD"),
        ("noise_site_swd", "Noise-domain SWD"),
        ("all_site_swd", "All-sample site SWD"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=False)
    for ax, (metric, title) in zip(axes, metric_cols):
        positions = []
        labels = []
        data = []
        colors = []
        pos = 1
        for preproc in ("filter_rms", "logenv"):
            sub = final[final["preprocessing"] == preproc]
            if sub.empty:
                continue
            positions.append(pos)
            labels.append(PREPROCESSING[preproc]["label"])
            data.append(sub[metric].dropna().to_numpy())
            colors.append(PREPROCESSING[preproc]["color"])
            pos += 1
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True, widths=0.65)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.68)
            body.set_edgecolor("#333333")
        for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(1.0)
        rng = np.random.default_rng(9)
        for p, values, color in zip(positions, data, colors):
            jitter = rng.uniform(-0.045, 0.045, size=len(values))
            ax.scatter(np.full(len(values), p) + jitter, values, s=22, color=color, edgecolor="white", linewidth=0.35, alpha=0.82)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel("Cross-site SWD")
        style_axis(ax)
    fig.suptitle("Cross-site latent-domain mismatch by class", fontsize=14, y=1.03)
    fig.tight_layout()
    save(fig, "cross_site_classwise_swd_by_preprocessing")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    for ax, preproc in zip(axes, ("filter_rms", "logenv")):
        sub = final[final["preprocessing"] == preproc].copy()
        directions = sorted(sub["direction_short"].unique())
        width = 0.28
        x = np.arange(len(directions))
        event_means = [sub[sub["direction_short"] == d]["event_site_swd"].mean() for d in directions]
        noise_means = [sub[sub["direction_short"] == d]["noise_site_swd"].mean() for d in directions]
        ax.bar(x - width / 2, event_means, width=width, color="#e85d04", label="Event")
        ax.bar(x + width / 2, noise_means, width=width, color="#1f5fbf", label="Noise")
        ax.set_xticks(x)
        ax.set_xticklabels(directions)
        ax.set_title(PREPROCESSING[preproc]["label"], fontsize=12, pad=10)
        ax.set_xlabel("Transfer direction")
        ax.set_ylabel("Mean cross-site SWD")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "cross_site_event_noise_swd_by_direction")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    for ax, metric, title in [
        (axes[0], "event_site_swd", "Event-domain mismatch"),
        (axes[1], "noise_site_swd", "Noise-domain mismatch"),
    ]:
        for preproc in ("filter_rms", "logenv"):
            sub = final[final["preprocessing"] == preproc].sort_values("fraction")
            grouped = sub.groupby("fraction", as_index=False)[metric].mean()
            ax.plot(
                grouped["fraction"],
                grouped[metric],
                marker="o",
                linewidth=2.0,
                color=PREPROCESSING[preproc]["color"],
                label=PREPROCESSING[preproc]["label"],
            )
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1.0])
        ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("Target label fraction")
        ax.set_ylabel("Mean cross-site SWD")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "cross_site_classwise_swd_by_fraction")


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_note(df: pd.DataFrame) -> None:
    lines = [
        "# Cross-Site Classwise SWD",
        "",
        "This diagnostic computes SWD between source-site and target-site latent distributions using each finetuned cross-site encoder.",
        "",
        "- `event_site_swd`: SWD between source-site event latents and target-site event latents.",
        "- `noise_site_swd`: SWD between source-site noise latents and target-site noise latents.",
        "- `all_site_swd`: SWD between all source-site and target-site labeled test latents.",
        "- Latents are z-score standardized within each source-target comparison before SWD.",
        "- Lower cross-site classwise SWD indicates stronger latent alignment for that class, not necessarily better detection by itself.",
        "",
        f"Rows: {len(df)}",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect()
    if df.empty:
        raise RuntimeError("No cross-site runs found.")
    df.to_csv(OUT_DIR / "cross_site_classwise_swd.csv", index=False)
    plot_summary(df)
    write_note(df)
    summary = (
        df.groupby("preprocessing_label")[["event_site_swd", "noise_site_swd", "all_site_swd", "target_balanced_acc"]]
        .agg(["mean", "median", "std"])
        .round(4)
    )
    summary.to_csv(OUT_DIR / "cross_site_classwise_swd_summary.csv")
    print(f"[DONE] wrote {OUT_DIR / 'cross_site_classwise_swd.csv'}")
    print(f"[DONE] wrote {OUT_DIR / 'cross_site_classwise_swd_by_preprocessing.pdf'}")


if __name__ == "__main__":
    main()
