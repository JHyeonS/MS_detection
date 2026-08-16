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
from scipy.stats import wasserstein_distance


_original_njit = numba.njit
_original_vectorize = numba.vectorize


def _njit_without_cache(*args, **kwargs):
    kwargs["cache"] = False
    return _original_njit(*args, **kwargs)


def _vectorize_without_cache(*args, **kwargs):
    kwargs["cache"] = False
    return _original_vectorize(*args, **kwargs)


numba.njit = _njit_without_cache
numba.vectorize = _vectorize_without_cache
from umap import UMAP


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.utils.config_io import cfg_get
from src.models.cnn_encoder import cnn_encoder


OUT_DIR = ROOT / "figures" / "current_results_summary" / "latent_diagnosis" / "site-shift"
DATA_ROOT = ROOT / "data" / "visualbest_filter_rms_fs1000_rms0p15_lp50"
EXPERIMENT = "base_utah_2019__frac0p25"
SOURCE_SITE = "utah_2023"
TARGET_SITE = "utah_2019"

RUNS = {
    "scratch": {
        "label": "Scratch",
        "source_site": TARGET_SITE,
        "target_site": TARGET_SITE,
        "comparison": "target_train_vs_target_test",
        "run_dir": ROOT
        / "runs"
        / "metadata_v2_safe_rerun_v1"
        / "filter_rms_site_main_pre50_v2"
        / "utah_2019_scratch"
        / "scratch",
        "color": "#7a8699",
    },
    "in_domain": {
        "label": "In-domain",
        "source_site": TARGET_SITE,
        "target_site": TARGET_SITE,
        "comparison": "target_train_vs_target_test",
        "run_dir": ROOT
        / "runs"
        / "metadata_v2_safe_rerun_v1"
        / "filter_rms_site_main_pre50_v2"
        / "utah_2019_reconst_reconst_noanom"
        / "reconst",
        "color": "#376795",
    },
    "cross_domain": {
        "label": "Cross-domain",
        "source_site": SOURCE_SITE,
        "target_site": TARGET_SITE,
        "comparison": "source_test_vs_target_test",
        "run_dir": ROOT
        / "runs"
        / "metadata_v2_safe_rerun_v1"
        / "filter_rms_cross_site_reconst_swd_interval10_v1"
        / "utah_2023_to_utah_2019"
        / "reconst",
        "color": "#bf4b3e",
    },
}
SITE_LABELS = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
LABEL_COLORS = {0: "#4f8fd8", 1: "#f07c3e"}
LABEL_NAMES = {0: "Noise", 1: "Event"}
FRACTION_TAG = "0p25"


class FinetuneMSDNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = cnn_encoder(cfg)
        latent_dim = int(cfg_get(cfg, "model", "encoder", "latent_dim", default=128))
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.head(z)


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(run_dir: Path) -> FinetuneMSDNet:
    cfg = load_yaml(run_dir / "test" / EXPERIMENT / "merged_config.yaml")
    model = FinetuneMSDNet(cfg)
    ckpt = torch.load(run_dir / "finetune" / EXPERIMENT / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
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


def extract_latents(model: FinetuneMSDNet, paths: list[str], batch_size: int = 32) -> np.ndarray:
    features = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            x = load_batch(paths[start : start + batch_size])
            z, _ = model(x)
            features.append(z.cpu().numpy())
    return np.concatenate(features, axis=0)


def load_site_test(site: str) -> pd.DataFrame:
    path = DATA_ROOT / "metadata" / "experiments" / f"stage1_{site}_only" / "test.csv"
    df = pd.read_csv(path)
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df["site"] = site
    return df


def load_finetune_subset(run_dir: Path, site: str) -> pd.DataFrame:
    cache_dir = run_dir / "_label_fraction_cache" / f"base_{site}"
    path = cache_dir / f"train_frac{FRACTION_TAG}_seed42.csv"
    if not path.exists():
        cache_dir = run_dir / "_label_fraction_cache" / site
        path = cache_dir / f"train_frac{FRACTION_TAG}_seed42.csv"
    df = pd.read_csv(path)
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df["site"] = site
    return df


def read_metrics(run_dir: Path) -> dict:
    path = run_dir / "test" / EXPERIMENT / "test_metrics_fixed_threshold.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("or_metrics_fixed_threshold") or data.get("fc_metrics_fixed_threshold")


def standardize_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    both = np.concatenate([x, y], axis=0)
    mu = both.mean(axis=0, keepdims=True)
    sigma = both.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (x - mu) / sigma, (y - mu) / sigma


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int = 128, seed: int = 42) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_proj, x.shape[1])).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    x_proj = x @ dirs.T
    y_proj = y @ dirs.T
    return float(np.mean([wasserstein_distance(x_proj[:, i], y_proj[:, i]) for i in range(n_proj)]))


def class_swd(source_feat: np.ndarray, source_labels: np.ndarray, target_feat: np.ndarray, target_labels: np.ndarray, label: int | None) -> float:
    if label is None:
        sx, tx = source_feat, target_feat
    else:
        sx = source_feat[source_labels == label]
        tx = target_feat[target_labels == label]
    sx, tx = standardize_pair(sx, tx)
    return sliced_wasserstein(sx, tx)


def balanced_umap_sample(df: pd.DataFrame, max_per_group: int = 150, seed: int = 24) -> pd.DataFrame:
    sampled = []
    for _, group in df.groupby(["setting", "latent_site", "label"], sort=False):
        n = min(len(group), max_per_group)
        sampled.append(group.sample(n=n, random_state=seed) if len(group) > n else group)
    return pd.concat(sampled, ignore_index=True)


def collect_setting(setting: str, spec: dict) -> tuple[pd.DataFrame, dict, dict]:
    model = load_model(spec["run_dir"])
    target_df = load_site_test(spec["target_site"])
    if spec["comparison"] == "target_train_vs_target_test":
        source_df = load_finetune_subset(spec["run_dir"], spec["target_site"])
        source_label = "Fine-tune subset"
    else:
        source_df = load_site_test(spec["source_site"])
        source_label = "Source site"
    pred = pd.read_csv(spec["run_dir"] / "test" / EXPERIMENT / "test_predictions.csv")
    pred["meta_npy_path"] = pred["meta_npy_path"].astype(str)
    pred["pred_or"] = pred["pred_or"].astype(int)

    source_feat = extract_latents(model, source_df["npy_path"].astype(str).tolist())
    target_feat = extract_latents(model, target_df["npy_path"].astype(str).tolist())

    source_latent = source_df[["npy_path", "label", "site"]].copy()
    source_latent["latent_site"] = source_label
    source_latent["is_correct"] = True
    source_latent["pred_or"] = np.nan
    source_latent["feat"] = list(source_feat)

    target_latent = target_df[["npy_path", "label", "site"]].copy()
    target_latent = target_latent.merge(
        pred[["meta_npy_path", "pred_or"]].rename(columns={"meta_npy_path": "npy_path"}),
        on="npy_path",
        how="left",
        validate="one_to_one",
    )
    target_latent["latent_site"] = "Target test"
    target_latent["is_correct"] = target_latent["label"].astype(int).eq(target_latent["pred_or"].astype("Int64"))
    target_latent["feat"] = list(target_feat)

    latent = pd.concat([source_latent, target_latent], ignore_index=True)
    latent["setting"] = setting
    latent["setting_label"] = spec["label"]
    latent["source_site"] = spec["source_site"]
    latent["target_site"] = spec["target_site"]

    source_labels = source_df["label"].to_numpy().astype(int)
    target_labels = target_df["label"].to_numpy().astype(int)
    swd = {
        "event_swd": class_swd(source_feat, source_labels, target_feat, target_labels, 1),
        "noise_swd": class_swd(source_feat, source_labels, target_feat, target_labels, 0),
        "all_swd": class_swd(source_feat, source_labels, target_feat, target_labels, None),
    }
    metrics = read_metrics(spec["run_dir"])
    return latent, swd, metrics


def compute_umap(latent_df: pd.DataFrame) -> pd.DataFrame:
    sampled = balanced_umap_sample(latent_df)
    x = np.stack(sampled["feat"].to_numpy()).astype(np.float32)
    emb = UMAP(n_components=2, n_neighbors=22, min_dist=0.14, metric="euclidean", random_state=42).fit_transform(x)
    out = sampled.drop(columns=["feat"]).copy()
    out["umap_x"] = emb[:, 0]
    out["umap_y"] = emb[:, 1]
    return out


def style_axis(ax: plt.Axes) -> None:
    ax.grid(color="#e8ded2", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.2, colors="#5f6673", length=2.2, width=0.55)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#c7cbd1")


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def draw_umap(ax: plt.Axes, df: pd.DataFrame, title: str, metrics: dict) -> None:
    markers = {"Fine-tune subset": "s", "Source site": "^", "Target test": "o"}
    sizes = {"Fine-tune subset": 13, "Source site": 11, "Target test": 12}
    alphas = {"Fine-tune subset": 0.48, "Source site": 0.40, "Target test": 0.76}
    for latent_site in ("Fine-tune subset", "Source site", "Target test"):
        for label in (0, 1):
            sub = df[df["latent_site"].eq(latent_site) & df["label"].astype(int).eq(label)]
            if sub.empty:
                continue
            ax.scatter(
                sub["umap_x"],
                sub["umap_y"],
                s=sizes[latent_site],
                c=LABEL_COLORS[label],
                marker=markers[latent_site],
                alpha=alphas[latent_site],
                linewidths=0,
                label=f"{latent_site} {LABEL_NAMES[label]}",
                rasterized=True,
            )
    correct = as_bool(df["is_correct"])
    wrong = df[df["latent_site"].eq("Target test") & (~correct)]
    if not wrong.empty:
        ax.scatter(
            wrong["umap_x"],
            wrong["umap_y"],
            s=22,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.5,
            label="Target error",
            rasterized=True,
        )
    ax.set_title(f"{title}\nBalAcc {metrics['balanced_acc']:.3f} | Spec {metrics['specificity']:.3f}", fontsize=9.5, fontweight="normal", pad=7)
    ax.set_xlabel("UMAP 1", fontsize=8.0)
    ax.set_ylabel("UMAP 2", fontsize=8.0)
    style_axis(ax)


def plot_umaps(umap_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.35))
    for ax, setting in zip(axes, ("scratch", "in_domain", "cross_domain")):
        sub = umap_df[umap_df["setting"].eq(setting)]
        metrics = metrics_df[metrics_df["setting"].eq(setting)].iloc[0].to_dict()
        source = SITE_LABELS[RUNS[setting]["source_site"]]
        target = SITE_LABELS[RUNS[setting]["target_site"]]
        draw_umap(ax, sub, f"{RUNS[setting]['label']}: {source} -> {target}", metrics)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=6, frameon=False, fontsize=7.4)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.9)
    save(fig, "case1_fixed_lowpass_utah2019_scratch_vs_indomain_vs_cross_umap")


def plot_swd(swd_df: pd.DataFrame) -> None:
    metrics = [("event_swd", "Event"), ("noise_swd", "Noise"), ("all_swd", "All")]
    x = np.arange(len(metrics))
    settings = ("scratch", "in_domain", "cross_domain")
    width = 0.24
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.25))
    max_value = max(float(swd_df[col].max()) for col, _ in metrics)
    label_offset = max(max_value * 0.025, 0.035)
    min_visible = max(max_value * 0.012, 0.018)
    offsets = np.linspace(-width, width, len(settings))
    for offset, setting in zip(offsets, settings):
        row = swd_df[swd_df["setting"].eq(setting)].iloc[0]
        values = [float(row[col]) for col, _ in metrics]
        visible_values = [value if value > 0 else min_visible for value in values]
        bars = ax.bar(x + offset, visible_values, width=width, color=RUNS[setting]["color"], label=RUNS[setting]["label"], alpha=0.92)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                color="#374151",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Latent SWD", fontsize=8.7)
    ax.set_title("Low-pass fixed: target consistency and site mismatch", fontsize=10.3, fontweight="normal", pad=8)
    ax.set_ylim(0, max_value * 1.22 + label_offset)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    fig.tight_layout()
    save(fig, "case1_fixed_lowpass_utah2019_scratch_vs_indomain_vs_cross_swd")


def plot_transfer_gain(metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.9, 3.25))
    settings = ["scratch", "in_domain", "cross_domain"]
    values = [float(metrics_df[metrics_df["setting"].eq(s)]["balanced_acc"].iloc[0]) for s in settings]
    scratch = float(metrics_df["scratch_target_balanced_acc"].iloc[0])
    x = np.arange(len(settings))
    ax.axhline(scratch, color="#8a8178", linewidth=1.5, linestyle="--", label=f"Target scratch ({scratch:.3f})")
    ax.bar(x, values, color=[RUNS[s]["color"] for s in settings], width=0.58, alpha=0.92)
    for idx, (setting, value) in enumerate(zip(settings, values)):
        gain = value - scratch
        ax.text(idx, value + 0.035, f"{gain:+.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([RUNS[s]["label"] for s in settings], rotation=10, ha="right")
    ax.set_ylabel("Balanced accuracy", fontsize=8.7)
    ax.set_title("Low-pass fixed: scratch vs transfer", fontsize=10.3, fontweight="normal", pad=8)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7.8, loc="lower left")
    fig.tight_layout()
    save(fig, "case1_fixed_lowpass_utah2019_scratch_vs_indomain_vs_cross_transfer_gain")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    latent_rows = []
    swd_rows = []
    metric_rows = []
    for setting, spec in RUNS.items():
        latent, swd, metrics = collect_setting(setting, spec)
        latent_rows.append(latent)
        swd_rows.append({"setting": setting, "setting_label": spec["label"], **swd})
        metric_rows.append({"setting": setting, "setting_label": spec["label"], **metrics})

    latent_df = pd.concat(latent_rows, ignore_index=True)
    umap_df = compute_umap(latent_df)
    swd_df = pd.DataFrame(swd_rows)
    metrics_df = pd.DataFrame(metric_rows)

    # Scratch baseline for the same target site, label fraction, and Low-pass preprocessing.
    gain_source = pd.read_csv(ROOT / "figures" / "current_results_summary" / "cross_site_summary" / "transfer_gain_decomposition_metrics.csv")
    scratch = gain_source[
        gain_source["preprocessing"].eq("filter_rms")
        & gain_source["target_site"].eq(TARGET_SITE)
        & np.isclose(gain_source["fraction"], 0.25)
    ]["scratch_target_balanced_acc"].dropna().iloc[0]
    metrics_df["scratch_target_balanced_acc"] = float(scratch)
    metrics_df["transfer_gain_vs_scratch"] = metrics_df["balanced_acc"].astype(float) - float(scratch)

    umap_df.to_csv(OUT_DIR / "case1_fixed_lowpass_utah2019_scratch_vs_indomain_vs_cross_umap_points.csv", index=False)
    swd_df.to_csv(OUT_DIR / "case1_fixed_lowpass_utah2019_scratch_vs_indomain_vs_cross_swd.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "case1_fixed_lowpass_utah2019_scratch_vs_indomain_vs_cross_metrics.csv", index=False)

    plot_umaps(umap_df, metrics_df)
    plot_swd(swd_df)
    plot_transfer_gain(metrics_df)
    print(f"[DONE] wrote fixed-filter Case 1 diagnosis panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
