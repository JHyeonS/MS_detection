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


RUN_BASE = ROOT / "runs" / "metadata_v2_safe_rerun_v1"
OUT_DIR = ROOT / "figures" / "current_results_summary" / "controlled_latent_diagnosis_axes"
SITE_GAIN_CSV = ROOT / "figures" / "current_results_summary" / "cross_site_summary" / "transfer_gain_decomposition_metrics.csv"

SOURCE_SITE = "utah_2019"
TARGET_SITE = "utah_2023"
EXPERIMENT = "base_utah_2023__frac0p25"
FRACTION = 0.25

SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
PREPROCESSING = {
    "raw": {
        "label": "Raw",
        "run_dir": RUN_BASE / "raw_cross_site_reconst_pre50_v1" / "utah_2019_to_utah_2023" / "reconst",
        "data_root": ROOT / "data" / "visualbest_raw_rms_fs1000_rms0p15_nofilter",
        "color": "#6b7280",
        "marker": "^",
    },
    "filter_rms": {
        "label": "Low-pass",
        "run_dir": RUN_BASE / "filter_rms_cross_site_reconst_swd_interval10_v1" / "utah_2019_to_utah_2023" / "reconst",
        "data_root": ROOT / "data" / "visualbest_filter_rms_fs1000_rms0p15_lp50",
        "color": "#bf4b3e",
        "marker": "o",
    },
    "logenv": {
        "label": "Log-envelope",
        "run_dir": RUN_BASE / "logenv_cross_site_reconst_swd_interval10_v1" / "utah_2019_to_utah_2023" / "reconst",
        "data_root": ROOT / "data" / "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5",
        "color": "#376795",
        "marker": "s",
    },
}
PREPROC_ORDER = ["raw", "filter_rms", "logenv"]
LABEL_COLORS = {0: "#4f8fd8", 1: "#f07c3e"}
LABEL_NAMES = {0: "Noise", 1: "Event"}


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


def load_site_test(data_root: Path, site: str) -> pd.DataFrame:
    path = data_root / "metadata" / "experiments" / f"stage1_{site}_only" / "test.csv"
    df = pd.read_csv(path)
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df["site"] = site
    return df


def read_metrics(run_dir: Path) -> dict:
    path = run_dir / "test" / EXPERIMENT / "test_metrics_fixed_threshold.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fc_metrics_fixed_threshold") or data.get("or_metrics_fixed_threshold")


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


def balanced_umap_sample(df: pd.DataFrame, max_per_group: int = 150, seed: int = 12) -> pd.DataFrame:
    sampled = []
    for _, group in df.groupby(["domain", "label"], sort=False):
        n = min(len(group), max_per_group)
        sampled.append(group.sample(n=n, random_state=seed) if len(group) > n else group)
    return pd.concat(sampled, ignore_index=True)


def collect_preproc(preproc: str, spec: dict) -> tuple[pd.DataFrame, dict, dict]:
    model = load_model(spec["run_dir"])
    source_df = load_site_test(spec["data_root"], SOURCE_SITE)
    target_df = load_site_test(spec["data_root"], TARGET_SITE)
    pred = pd.read_csv(spec["run_dir"] / "test" / EXPERIMENT / "test_predictions.csv")
    pred["meta_npy_path"] = pred["meta_npy_path"].astype(str)
    pred["pred_fc"] = pred["pred_fc"].astype(int)

    source_feat = extract_latents(model, source_df["npy_path"].astype(str).tolist())
    target_feat = extract_latents(model, target_df["npy_path"].astype(str).tolist())

    source_latent = source_df[["npy_path", "label"]].copy()
    source_latent["domain"] = "Source"
    source_latent["is_correct"] = True
    source_latent["pred_fc"] = np.nan
    source_latent["feat"] = list(source_feat)

    target_latent = target_df[["npy_path", "label"]].copy()
    target_latent = target_latent.merge(
        pred[["meta_npy_path", "pred_fc"]].rename(columns={"meta_npy_path": "npy_path"}),
        on="npy_path",
        how="left",
        validate="one_to_one",
    )
    target_latent["domain"] = "Target"
    target_latent["is_correct"] = target_latent["label"].astype(int).eq(target_latent["pred_fc"].astype("Int64"))
    target_latent["feat"] = list(target_feat)

    all_latent = pd.concat([source_latent, target_latent], ignore_index=True)
    plot_latent = balanced_umap_sample(all_latent)
    x = np.stack(plot_latent["feat"].to_numpy()).astype(np.float32)
    emb = UMAP(n_components=2, n_neighbors=22, min_dist=0.14, metric="euclidean", random_state=42).fit_transform(x)
    plot_latent = plot_latent.drop(columns=["feat"]).copy()
    plot_latent["umap_x"] = emb[:, 0]
    plot_latent["umap_y"] = emb[:, 1]
    plot_latent["preprocessing"] = preproc
    plot_latent["preprocessing_label"] = spec["label"]

    source_labels = source_df["label"].to_numpy().astype(int)
    target_labels = target_df["label"].to_numpy().astype(int)
    swd = {
        "event_swd": class_swd(source_feat, source_labels, target_feat, target_labels, 1),
        "noise_swd": class_swd(source_feat, source_labels, target_feat, target_labels, 0),
        "all_swd": class_swd(source_feat, source_labels, target_feat, target_labels, None),
    }
    metrics = read_metrics(spec["run_dir"])
    return plot_latent, swd, metrics


def gain_matrix_for(gain: pd.DataFrame, preproc: str, fraction: float = FRACTION) -> np.ndarray:
    matrix = np.full((len(SITES), len(SITES)), np.nan, dtype=float)
    sub = gain[gain["preprocessing"].eq(preproc) & np.isclose(gain["fraction"], fraction)]
    for _, row in sub.iterrows():
        if row["source_site"] not in SITES or row["target_site"] not in SITES:
            continue
        matrix[SITES.index(row["source_site"]), SITES.index(row["target_site"])] = float(row["transfer_gain_balanced_acc"])
    return matrix


def plot_axis1_site_shift_gain() -> None:
    gain = pd.read_csv(SITE_GAIN_CSV)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#f2eee8")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.65), sharex=True, sharey=True)
    images = []
    vmax = 0.50
    for ax, preproc in zip(axes, PREPROC_ORDER):
        matrix = gain_matrix_for(gain, preproc, FRACTION)
        image = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap=cmap)
        images.append(image)
        ax.set_title(PREPROCESSING[preproc]["label"], fontsize=11.0, fontweight="normal", pad=8)
        ax.set_xticks(range(len(SITES)))
        ax.set_yticks(range(len(SITES)))
        ax.set_xticklabels([SITE_LABELS[s] for s in SITES], rotation=30, ha="right", fontsize=8.2)
        ax.set_yticklabels([SITE_LABELS[s] for s in SITES], fontsize=8.2)
        ax.set_xlabel("Fine-tune / target site", fontsize=8.7)
        if ax is axes[0]:
            ax.set_ylabel("Pretrain / source site", fontsize=8.7)
        for i in range(len(SITES)):
            for j in range(len(SITES)):
                value = matrix[i, j]
                if np.isnan(value):
                    ax.text(j, i, "-", ha="center", va="center", fontsize=9.5, color="#8a8178")
                else:
                    color = "white" if abs(value) >= 0.28 else "#111827"
                    ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9.0, color=color)
        ax.set_xticks(np.arange(-0.5, len(SITES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SITES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.90, bottom=0.18, top=0.80, wspace=0.28)
    cax = fig.add_axes([0.925, 0.23, 0.014, 0.54])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label("Transfer gain in balanced accuracy", fontsize=9.0)
    cbar.ax.tick_params(labelsize=8.0)
    fig.suptitle(
        f"Axis 1: site shift under fixed preprocessing (gain relative to target-site scratch, fraction={FRACTION:g})",
        fontsize=12.0,
        y=1.02,
    )
    save(fig, "axis1_site_shift_fixed_preprocessing_gain_matrix")


def style_panel(ax: plt.Axes) -> None:
    ax.grid(color="#e8ded2", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.2, colors="#5f6673", length=2.2, width=0.55)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#c7cbd1")


def draw_umap(ax: plt.Axes, emb: pd.DataFrame, title: str, metrics: dict, letter: str) -> None:
    markers = {"Source": "^", "Target": "o"}
    sizes = {"Source": 10, "Target": 11}
    alphas = {"Source": 0.38, "Target": 0.74}
    for domain in ("Source", "Target"):
        for label in (0, 1):
            sub = emb[emb["domain"].eq(domain) & emb["label"].astype(int).eq(label)]
            ax.scatter(
                sub["umap_x"],
                sub["umap_y"],
                s=sizes[domain],
                c=LABEL_COLORS[label],
                marker=markers[domain],
                alpha=alphas[domain],
                linewidths=0,
                label=f"{domain} {LABEL_NAMES[label]}",
                rasterized=True,
            )
    wrong = emb[emb["domain"].eq("Target") & (~emb["is_correct"].astype(bool))]
    if not wrong.empty:
        ax.scatter(
            wrong["umap_x"],
            wrong["umap_y"],
            s=18,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.48,
            label="Target error",
            rasterized=True,
        )
    ax.set_title(f"{title}\nBalAcc {metrics['balanced_acc']:.3f} | Spec {metrics['specificity']:.3f}", fontsize=9.3, fontweight="normal", pad=7)
    ax.set_xlabel("UMAP 1", fontsize=8.0)
    ax.set_ylabel("UMAP 2", fontsize=8.0)
    ax.text(
        0.02,
        0.98,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    style_panel(ax)


def draw_gain(ax: plt.Axes, gain_df: pd.DataFrame) -> None:
    x = np.arange(len(PREPROC_ORDER))
    labels = [PREPROCESSING[p]["label"] for p in PREPROC_ORDER]
    cross = [float(gain_df[gain_df["preprocessing"].eq(p)]["balanced_acc"].iloc[0]) for p in PREPROC_ORDER]
    scratch = [float(gain_df[gain_df["preprocessing"].eq(p)]["scratch_target_balanced_acc"].iloc[0]) for p in PREPROC_ORDER]
    gains = [float(gain_df[gain_df["preprocessing"].eq(p)]["transfer_gain_balanced_acc"].iloc[0]) for p in PREPROC_ORDER]
    width = 0.34
    ax.bar(x - width / 2, scratch, width=width, color="#d8d3ca", label="Target scratch")
    ax.bar(x + width / 2, cross, width=width, color=[PREPROCESSING[p]["color"] for p in PREPROC_ORDER], label="Cross-site transfer")
    for idx, gain in enumerate(gains):
        ax.text(idx, max(scratch[idx], cross[idx]) + 0.035, f"{gain:+.2f}", ha="center", va="bottom", fontsize=7.8)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("Balanced accuracy", fontsize=8.5)
    ax.set_title("Transfer gain by preprocessing", fontsize=10.0, fontweight="normal", pad=8)
    style_panel(ax)
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)


def draw_swd(ax: plt.Axes, swd_rows: pd.DataFrame) -> None:
    metrics = [("event_swd", "Event"), ("noise_swd", "Noise"), ("all_swd", "All")]
    x = np.arange(len(metrics))
    width = 0.23
    for offset, preproc in zip([-width, 0, width], PREPROC_ORDER):
        row = swd_rows[swd_rows["preprocessing"].eq(preproc)].iloc[0]
        values = [float(row[col]) for col, _ in metrics]
        ax.bar(x + offset, values, width=width, color=PREPROCESSING[preproc]["color"], label=PREPROCESSING[preproc]["label"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Source-target SWD", fontsize=8.5)
    ax.set_title("Latent mismatch by preprocessing", fontsize=10.0, fontweight="normal", pad=8)
    style_panel(ax)
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")


def plot_axis2_preprocessing_effect() -> None:
    site_gain = pd.read_csv(SITE_GAIN_CSV)
    gain = site_gain[
        site_gain["source_site"].eq(SOURCE_SITE)
        & site_gain["target_site"].eq(TARGET_SITE)
        & np.isclose(site_gain["fraction"], FRACTION)
        & site_gain["preprocessing"].isin(PREPROC_ORDER)
        & site_gain["gain_type"].eq("cross_domain")
    ].copy()

    embedded = []
    swd_rows = []
    metric_rows = []
    for preproc in PREPROC_ORDER:
        emb, swd, metrics = collect_preproc(preproc, PREPROCESSING[preproc])
        embedded.append(emb)
        swd_rows.append({"preprocessing": preproc, "preprocessing_label": PREPROCESSING[preproc]["label"], **swd})
        metric_rows.append({"preprocessing": preproc, "preprocessing_label": PREPROCESSING[preproc]["label"], **metrics})

    emb_df = pd.concat(embedded, ignore_index=True)
    swd_df = pd.DataFrame(swd_rows)
    metric_df = pd.DataFrame(metric_rows)
    emb_df.to_csv(OUT_DIR / "axis2_utah2019_to_utah2023_frac0p25_source_target_umap_points.csv", index=False)
    swd_df.to_csv(OUT_DIR / "axis2_utah2019_to_utah2023_frac0p25_source_target_swd.csv", index=False)
    metric_df.to_csv(OUT_DIR / "axis2_utah2019_to_utah2023_frac0p25_transfer_metrics.csv", index=False)
    gain.to_csv(OUT_DIR / "axis2_utah2019_to_utah2023_frac0p25_transfer_gain.csv", index=False)

    fig = plt.figure(figsize=(12.8, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.34)
    for idx, preproc in enumerate(PREPROC_ORDER):
        ax = fig.add_subplot(gs[0, idx])
        emb = emb_df[emb_df["preprocessing"].eq(preproc)]
        metrics = metric_df[metric_df["preprocessing"].eq(preproc)].iloc[0].to_dict()
        draw_umap(ax, emb, f"{PREPROCESSING[preproc]['label']} latent UMAP", metrics, chr(ord("A") + idx))
    draw_gain(fig.add_subplot(gs[1, 0]), gain)
    draw_swd(fig.add_subplot(gs[1, 1]), swd_df)
    ax_note = fig.add_subplot(gs[1, 2])
    ax_note.axis("off")
    note = (
        "Controlled comparison\n"
        f"Source-target pair fixed: {SITE_LABELS[SOURCE_SITE]} -> {SITE_LABELS[TARGET_SITE]}\n"
        "Only preprocessing changes: Raw / Low-pass / Log-envelope\n\n"
        "This axis tests whether preprocessing changes latent alignment and transfer gain\n"
        "without mixing source-preprocessing and target-preprocessing domains."
    )
    ax_note.text(0.02, 0.92, note, ha="left", va="top", fontsize=9.0, color="#374151", linespacing=1.35)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.28,
        columnspacing=0.8,
    )
    fig.suptitle(
        f"Axis 2: preprocessing effect under a fixed site pair ({SITE_LABELS[SOURCE_SITE]} -> {SITE_LABELS[TARGET_SITE]}, fraction={FRACTION:g})",
        fontsize=12.2,
        y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.11, top=0.88)
    save(fig, "axis2_fixed_site_pair_preprocessing_latent_diagnosis")


def write_axis_note() -> None:
    lines = [
        "# Controlled latent diagnosis axes",
        "",
        "## Axis 1: Same preprocessing, different sites",
        "",
        "- Purpose: diagnose site-dependent distribution shift.",
        "- Control: preprocessing is fixed within each panel.",
        "- Variable: source and target site.",
        "- Figure: `axis1_site_shift_fixed_preprocessing_gain_matrix.pdf`.",
        "",
        "## Axis 2: Same site pair, different preprocessing",
        "",
        f"- Purpose: test whether preprocessing changes latent alignment and transfer gain for a fixed transfer direction.",
        f"- Control: source-target pair is fixed as {SITE_LABELS[SOURCE_SITE]} -> {SITE_LABELS[TARGET_SITE]}.",
        "- Variable: Raw, Low-pass, and Log-envelope preprocessing.",
        "- Figure: `axis2_fixed_site_pair_preprocessing_latent_diagnosis.pdf`.",
        "",
        "We do not use mismatched source-target preprocessing for latent diagnosis because it confounds site-dependent distribution shift with preprocessing-induced input-representation shift.",
    ]
    (OUT_DIR / "controlled_latent_diagnosis_axes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    plot_axis1_site_shift_gain()
    plot_axis2_preprocessing_effect()
    write_axis_note()
    print(f"[DONE] wrote controlled latent diagnosis axis figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
