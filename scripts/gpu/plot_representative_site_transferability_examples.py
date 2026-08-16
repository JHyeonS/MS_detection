#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_CSV = (
    ROOT
    / "temp"
    / "current_results_summary"
    / "figures_metadata_v2"
    / "center"
    / "cross_site_classwise_swd"
    / "cross_site_classwise_swd.csv"
)
OUT_DIR = (
    ROOT
    / "temp"
    / "current_results_summary"
    / "figures_metadata_v2"
    / "seg"
    / "representative_site_transferability"
)

DIRECTIONS = ["pohang_to_utah_2019", "utah_2019_to_utah_2023", "utah_2023_to_pohang"]
DIRECTION_LABELS = {
    "pohang_to_utah_2019": "A -> B\nPohang -> Utah 2019",
    "utah_2019_to_utah_2023": "B -> C\nUtah 2019 -> Utah 2023",
    "utah_2023_to_pohang": "C -> A\nUtah 2023 -> Pohang",
}
PREPROC_ORDER = ["Low-pass + RMS", "Log-envelope"]
COLORS = {"Low-pass + RMS": "#bf4b3e", "Log-envelope": "#376795"}
MARKERS = {"Low-pass + RMS": "s", "Log-envelope": "o"}
LINESTYLES = {"Low-pass + RMS": "--", "Log-envelope": "-"}
X_OFFSETS = {"Low-pass + RMS": 0.965, "Log-envelope": 1.035}
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
FRACTION_LABELS = ["0.05", "0.10", "0.25", "0.50", "1.00"]


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d7d2c8", linewidth=0.8, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    df = df[df["direction"].isin(DIRECTIONS)].copy()
    df["preprocessing_label"] = pd.Categorical(df["preprocessing_label"], PREPROC_ORDER, ordered=True)
    df["direction"] = pd.Categorical(df["direction"], DIRECTIONS, ordered=True)
    return df.sort_values(["direction", "preprocessing_label", "fraction"])


def plot_metric_curves(df: pd.DataFrame, metric: str, ylabel: str, name: str, ylim: tuple[float, float] = (0.0, 1.05)) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.45), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        sub = df[df["direction"].astype(str).eq(direction)]
        for label in PREPROC_ORDER:
            d = sub[sub["preprocessing_label"].astype(str).eq(label)].sort_values("fraction")
            x = d["fraction"].to_numpy() * X_OFFSETS[label]
            ax.plot(
                x,
                d[metric],
                marker=MARKERS[label],
                markersize=5.8,
                linewidth=2.0,
                linestyle=LINESTYLES[label],
                color=COLORS[label],
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xticks(FRACTIONS)
        ax.set_xticklabels(FRACTION_LABELS)
        ax.set_ylim(*ylim)
        ax.set_title(DIRECTION_LABELS[direction], fontsize=11.2, pad=10)
        ax.set_xlabel("Target label fraction")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.1)
    save(fig, name)


def plot_balacc_specificity_grid(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.2), sharex=True)
    rows = [
        ("target_balanced_acc", "Balanced accuracy"),
        ("target_specificity", "Specificity"),
    ]
    for row_idx, (metric, ylabel) in enumerate(rows):
        for col_idx, direction in enumerate(DIRECTIONS):
            ax = axes[row_idx, col_idx]
            sub = df[df["direction"].astype(str).eq(direction)]
            for label in PREPROC_ORDER:
                d = sub[sub["preprocessing_label"].astype(str).eq(label)].sort_values("fraction")
                x = d["fraction"].to_numpy() * X_OFFSETS[label]
                ax.plot(
                    x,
                    d[metric],
                    marker=MARKERS[label],
                    markersize=5.8,
                    linewidth=2.0,
                    linestyle=LINESTYLES[label],
                    color=COLORS[label],
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_xticks(FRACTIONS)
            ax.set_xticklabels(FRACTION_LABELS)
            ax.set_ylim(0.0, 1.05)
            if row_idx == 0:
                ax.set_title(DIRECTION_LABELS[direction], fontsize=11.2, pad=10)
            ax.set_xlabel("Target label fraction" if row_idx == 1 else "")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.065, 1, 1), h_pad=1.2, w_pad=1.05)
    save(fig, "representative_site_transferability_balacc_specificity_grid")


def plot_lowlabel_grouped_bar(df: pd.DataFrame, metric: str, ylabel: str, name: str) -> None:
    low = df[df["fraction"].isin([0.05, 0.10, 0.25])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), sharey=True)
    width = 0.32
    x = np.arange(3)
    for ax, direction in zip(axes, DIRECTIONS):
        sub = low[low["direction"].astype(str).eq(direction)]
        for offset, label in zip([-width / 2, width / 2], PREPROC_ORDER):
            d = sub[sub["preprocessing_label"].astype(str).eq(label)].set_index("fraction").loc[[0.05, 0.10, 0.25]]
            ax.bar(x + offset, d[metric], width=width, color=COLORS[label], alpha=0.9, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(["0.05", "0.10", "0.25"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(DIRECTION_LABELS[direction], fontsize=11.2, pad=10)
        ax.set_xlabel("Target label fraction")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.1)
    save(fig, name)


def plot_fraction_heatmap(df: pd.DataFrame, metric: str, title: str, name: str) -> None:
    rows = []
    ylabels = []
    for direction in DIRECTIONS:
        for label in PREPROC_ORDER:
            sub = df[
                df["direction"].astype(str).eq(direction)
                & df["preprocessing_label"].astype(str).eq(label)
            ].set_index("fraction")
            rows.append(sub.loc[FRACTIONS, metric].to_numpy())
            ylabels.append(f"{direction.replace('_to_', ' -> ')}\n{label}")
    matrix = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(7.9, 5.0))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(FRACTIONS)))
    ax.set_xticklabels(FRACTION_LABELS)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Target label fraction")
    ax.set_title(title, fontsize=12, pad=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.2, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, name)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    df = load_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "representative_site_transferability_metrics.csv", index=False)
    plot_metric_curves(
        df,
        "target_balanced_acc",
        "Balanced accuracy",
        "representative_site_transferability_balanced_acc_curves",
    )
    plot_metric_curves(
        df,
        "target_specificity",
        "Specificity",
        "representative_site_transferability_specificity_curves",
    )
    plot_balacc_specificity_grid(df)
    plot_lowlabel_grouped_bar(
        df,
        "target_balanced_acc",
        "Balanced accuracy",
        "representative_site_transferability_lowlabel_balacc_bars",
    )
    plot_lowlabel_grouped_bar(
        df,
        "target_specificity",
        "Specificity",
        "representative_site_transferability_lowlabel_specificity_bars",
    )
    plot_fraction_heatmap(
        df,
        "target_balanced_acc",
        "Representative site transferability: balanced accuracy",
        "representative_site_transferability_balacc_heatmap",
    )
    plot_fraction_heatmap(
        df,
        "target_specificity",
        "Representative site transferability: specificity",
        "representative_site_transferability_specificity_heatmap",
    )
    print(f"[DONE] wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
