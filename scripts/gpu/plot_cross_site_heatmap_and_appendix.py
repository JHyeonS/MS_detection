#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "figures" / "current_results_summary" / "pretrain_setting_transfer" / "pretrain_setting_transfer_metrics.csv"
OUT_DIR = ROOT / "figures" / "current_results_summary" / "cross_site_summary"

SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
PREPROCESSING = {
    "raw": {"label": "Raw", "color": "#6b7280", "marker": "^"},
    "filter_rms": {"label": "Low-pass", "color": "#bf4b3e", "marker": "o"},
    "logenv": {"label": "Log-envelope", "color": "#376795", "marker": "s"},
}
PREPROC_ORDER = ["raw", "filter_rms", "logenv"]
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
DIRECTION_ORDER = [
    "pohang_to_utah_2019",
    "pohang_to_utah_2023",
    "utah_2019_to_pohang",
    "utah_2019_to_utah_2023",
    "utah_2023_to_pohang",
    "utah_2023_to_utah_2019",
]


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def setup_curve_axis(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels(["5", "10", "25", "50", "100"], fontsize=7.2)
    ax.grid(axis="y", color="#ded8cf", linewidth=0.78, alpha=0.78)
    ax.grid(axis="x", color="#eee7dd", linewidth=0.45, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.4, colors="#4b5563")
    ax.set_xlabel("Label fraction (%)", fontsize=7.6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.0)


def direction_label(direction: str) -> str:
    source, target = direction.split("_to_", 1)
    return f"{SITE_LABELS[source]} -> {SITE_LABELS[target]}"


def plot_appendix_all_curves(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(15.8, 7.4), sharex=True, sharey=True)
    for row_idx, preproc in enumerate(PREPROC_ORDER):
        spec = PREPROCESSING[preproc]
        for col_idx, direction in enumerate(DIRECTION_ORDER):
            ax = axes[row_idx, col_idx]
            sub = (
                df[
                    df["preprocessing"].eq(preproc)
                    & df["direction"].eq(direction)
                ]
                .sort_values("fraction")
            )
            if not sub.empty:
                ax.plot(
                    sub["fraction"],
                    sub["balanced_acc"],
                    color=spec["color"],
                    marker=spec["marker"],
                    markersize=4.0,
                    linewidth=1.7,
                )
            if row_idx == 0:
                ax.set_title(direction_label(direction), fontsize=8.4, fontweight="normal", pad=7)
            if col_idx == 0:
                ax.text(
                    -0.35,
                    0.5,
                    spec["label"],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9.0,
                    fontweight="bold",
                    color=spec["color"],
                )
            setup_curve_axis(ax, "Balanced accuracy" if col_idx == 0 else None)
    fig.suptitle("Appendix: complete cross-site label-efficiency curves", fontsize=12.5, y=1.01)
    fig.tight_layout(rect=(0.02, 0, 1, 0.98), w_pad=0.65, h_pad=0.75)
    save(fig, "appendix_cross_site_all_fraction_curves_balanced_accuracy")


def matrix_for(df: pd.DataFrame, preproc: str, metric: str, fraction: float) -> np.ndarray:
    matrix = np.full((len(SITES), len(SITES)), np.nan, dtype=float)
    sub = df[df["preprocessing"].eq(preproc) & np.isclose(df["fraction"], fraction)]
    for _, row in sub.iterrows():
        if row["source_site"] not in SITES or row["target_site"] not in SITES:
            continue
        i = SITES.index(row["source_site"])
        j = SITES.index(row["target_site"])
        matrix[i, j] = float(row[metric])
    return matrix


def compute_cross_site_gain(df: pd.DataFrame, metric: str = "balanced_acc") -> pd.DataFrame:
    cross = df[df["setting"].eq("reconst_cross_domain")].copy()
    scratch = df[df["setting"].eq("no_pretrain")].copy()
    baseline = scratch[
        [
            "preprocessing",
            "target_site",
            "fraction",
            metric,
        ]
    ].rename(columns={metric: f"scratch_target_{metric}"})
    gain = cross.merge(
        baseline,
        on=["preprocessing", "target_site", "fraction"],
        how="left",
        validate="many_to_one",
    )
    gain[f"transfer_gain_{metric}"] = gain[metric] - gain[f"scratch_target_{metric}"]
    return gain


def compute_transfer_gain_decomposition(df: pd.DataFrame, metric: str = "balanced_acc") -> pd.DataFrame:
    scratch = df[df["setting"].eq("no_pretrain")].copy()
    scratch_baseline = scratch[
        [
            "preprocessing",
            "target_site",
            "fraction",
            metric,
        ]
    ].rename(columns={metric: f"scratch_target_{metric}"})

    indomain = df[df["setting"].eq("reconst_indomain")].copy()
    indomain = indomain.merge(
        scratch_baseline,
        on=["preprocessing", "target_site", "fraction"],
        how="left",
        validate="many_to_one",
    )
    indomain["source_site"] = indomain["target_site"]
    indomain["direction"] = indomain["target_site"] + "_to_" + indomain["target_site"]
    indomain["gain_type"] = "in_domain"
    indomain[f"transfer_gain_{metric}"] = indomain[metric] - indomain[f"scratch_target_{metric}"]

    cross = compute_cross_site_gain(df, metric=metric).copy()
    cross["gain_type"] = "cross_domain"

    keep = [
        "gain_type",
        "preprocessing",
        "source_site",
        "target_site",
        "direction",
        "fraction_tag",
        "fraction",
        "method",
        "path",
        metric,
        f"scratch_target_{metric}",
        f"transfer_gain_{metric}",
    ]
    return pd.concat([indomain[keep], cross[keep]], ignore_index=True)


def gain_matrix_for(gain_df: pd.DataFrame, preproc: str, metric: str, fraction: float) -> np.ndarray:
    gain_key = f"transfer_gain_{metric}"
    matrix = np.full((len(SITES), len(SITES)), np.nan, dtype=float)
    sub = gain_df[gain_df["preprocessing"].eq(preproc) & np.isclose(gain_df["fraction"], fraction)]
    for _, row in sub.iterrows():
        if row["source_site"] not in SITES or row["target_site"] not in SITES:
            continue
        i = SITES.index(row["source_site"])
        j = SITES.index(row["target_site"])
        matrix[i, j] = float(row[gain_key])
    return matrix


def decomposition_matrix_for(gain_df: pd.DataFrame, preproc: str, metric: str, fraction: float) -> np.ndarray:
    gain_key = f"transfer_gain_{metric}"
    matrix = np.full((len(SITES), len(SITES)), np.nan, dtype=float)
    sub = gain_df[gain_df["preprocessing"].eq(preproc) & np.isclose(gain_df["fraction"], fraction)]
    for _, row in sub.iterrows():
        if row["source_site"] not in SITES or row["target_site"] not in SITES:
            continue
        i = SITES.index(row["source_site"])
        j = SITES.index(row["target_site"])
        matrix[i, j] = float(row[gain_key])
    return matrix


def plot_transfer_gain_decomposition_heatmap(gain_df: pd.DataFrame, fraction: float = 0.25, metric: str = "balanced_acc") -> None:
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#f2eee8")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.95), sharex=True, sharey=True)
    images = []
    vmax = 0.50
    for ax, preproc in zip(axes, PREPROC_ORDER):
        matrix = decomposition_matrix_for(gain_df, preproc, metric, fraction)
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
                    continue
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
        f"Transfer gain decomposition relative to target-site scratch at label fraction {fraction:g}",
        fontsize=12.5,
        y=1.025,
    )
    save(fig, f"representative_transfer_gain_decomposition_frac{str(fraction).replace('.', 'p')}_balanced_accuracy")


def plot_transfer_gain_heatmap(gain_df: pd.DataFrame, fraction: float = 0.25, metric: str = "balanced_acc") -> None:
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#f2eee8")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharex=True, sharey=True)
    images = []
    vmax = 0.50
    for ax, preproc in zip(axes, PREPROC_ORDER):
        matrix = gain_matrix_for(gain_df, preproc, metric, fraction)
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
                    continue
                color = "white" if abs(value) >= 0.28 else "#111827"
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9.0, color=color)
        ax.set_xticks(np.arange(-0.5, len(SITES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SITES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.90, bottom=0.18, top=0.82, wspace=0.28)
    cax = fig.add_axes([0.925, 0.23, 0.014, 0.54])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label("Transfer gain in balanced accuracy", fontsize=9.0)
    cbar.ax.tick_params(labelsize=8.0)
    fig.suptitle(f"Cross-site transfer gain relative to scratch at label fraction {fraction:g}", fontsize=12.5, y=1.03)
    save(fig, f"representative_cross_site_transfer_gain_frac{str(fraction).replace('.', 'p')}_balanced_accuracy")


def plot_appendix_gain_curves(gain_df: pd.DataFrame, metric: str = "balanced_acc") -> None:
    gain_key = f"transfer_gain_{metric}"
    fig, axes = plt.subplots(3, 6, figsize=(15.8, 7.4), sharex=True, sharey=True)
    for row_idx, preproc in enumerate(PREPROC_ORDER):
        spec = PREPROCESSING[preproc]
        for col_idx, direction in enumerate(DIRECTION_ORDER):
            ax = axes[row_idx, col_idx]
            sub = (
                gain_df[
                    gain_df["preprocessing"].eq(preproc)
                    & gain_df["direction"].eq(direction)
                ]
                .sort_values("fraction")
            )
            if not sub.empty:
                ax.axhline(0, color="#111827", linewidth=0.9, alpha=0.65)
                ax.plot(
                    sub["fraction"],
                    sub[gain_key],
                    color=spec["color"],
                    marker=spec["marker"],
                    markersize=4.0,
                    linewidth=1.7,
                )
            if row_idx == 0:
                ax.set_title(direction_label(direction), fontsize=8.4, fontweight="normal", pad=7)
            if col_idx == 0:
                ax.text(
                    -0.35,
                    0.5,
                    spec["label"],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9.0,
                    fontweight="bold",
                    color=spec["color"],
                )
            ax.set_ylim(-0.65, 0.65)
            ax.set_xticks(FRACTIONS)
            ax.set_xticklabels(["5", "10", "25", "50", "100"], fontsize=7.2)
            ax.grid(axis="y", color="#ded8cf", linewidth=0.78, alpha=0.78)
            ax.grid(axis="x", color="#eee7dd", linewidth=0.45, alpha=0.55)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7.4, colors="#4b5563")
            ax.set_xlabel("Label fraction (%)", fontsize=7.6)
            if col_idx == 0:
                ax.set_ylabel("Transfer gain", fontsize=8.0)
    fig.suptitle("Appendix: complete cross-site transfer gain curves", fontsize=12.5, y=1.01)
    fig.tight_layout(rect=(0.02, 0, 1, 0.98), w_pad=0.65, h_pad=0.75)
    save(fig, "appendix_cross_site_transfer_gain_curves_balanced_accuracy")


def plot_representative_heatmap(df: pd.DataFrame, fraction: float = 0.25, metric: str = "balanced_acc") -> None:
    cmap = plt.get_cmap("RdYlBu").copy()
    cmap.set_bad("#f2eee8")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharex=True, sharey=True)
    images = []
    for ax, preproc in zip(axes, PREPROC_ORDER):
        matrix = matrix_for(df, preproc, metric, fraction)
        image = ax.imshow(matrix, vmin=0.45, vmax=1.0, cmap=cmap)
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
                    continue
                color = "white" if value <= 0.58 or value >= 0.88 else "#1f2937"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9.0, color=color)
        ax.set_xticks(np.arange(-0.5, len(SITES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SITES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.90, bottom=0.18, top=0.82, wspace=0.28)
    cax = fig.add_axes([0.925, 0.23, 0.014, 0.54])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label("Balanced accuracy", fontsize=9.0)
    cbar.ax.tick_params(labelsize=8.0)
    fig.suptitle(f"Cross-site transferability heatmap at label fraction {fraction:g}", fontsize=12.5, y=1.03)
    save(fig, f"representative_cross_site_heatmap_frac{str(fraction).replace('.', 'p')}_balanced_accuracy")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    df = pd.read_csv(IN_CSV)
    cross = df[df["setting"].eq("reconst_cross_domain")].copy()
    gain = compute_cross_site_gain(df, metric="balanced_acc")
    gain_decomposition = compute_transfer_gain_decomposition(df, metric="balanced_acc")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cross.to_csv(OUT_DIR / "cross_site_reconst_metrics.csv", index=False)
    gain.to_csv(OUT_DIR / "cross_site_transfer_gain_metrics.csv", index=False)
    gain_decomposition.to_csv(OUT_DIR / "transfer_gain_decomposition_metrics.csv", index=False)
    plot_appendix_all_curves(cross)
    plot_representative_heatmap(cross, fraction=0.25, metric="balanced_acc")
    plot_transfer_gain_heatmap(gain, fraction=0.25, metric="balanced_acc")
    plot_transfer_gain_decomposition_heatmap(gain_decomposition, fraction=0.25, metric="balanced_acc")
    plot_appendix_gain_curves(gain, metric="balanced_acc")
    print(f"[DONE] wrote cross-site summary figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
