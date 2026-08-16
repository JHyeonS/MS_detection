#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "temp" / "current_results_summary"
OUT_DIR = SUMMARY_DIR / "figures"

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
SITE_LABELS = {
    "pohang": "Site A\nPohang",
    "utah_2019": "Site B\nUtah 2019",
    "utah_2023": "Site C\nUtah 2023",
}
METHOD_LABELS = {
    "scratch": "Scratch",
    "contrast": "Contrast",
    "reconst": "Reconst",
    "reconst_noanom": "Reconst no-anom",
}
METHOD_COLORS = {
    "scratch": "#4c78a8",
    "contrast": "#f58518",
    "reconst": "#54a24b",
    "reconst_noanom": "#b279a2",
}
METRICS = [
    ("balanced_acc", "Balanced accuracy"),
    ("f1", "F1"),
    ("specificity", "Specificity"),
]
EXPERIMENT_TITLES = {
    "experiment_1_log_env": "Experiment 1: log env",
    "experiment_2_no_log_env": "Experiment 2: no log env",
    "experiment_3_log_env_cross_site_transfer": "Experiment 3: log env cross-site transfer",
    "experiment_4_filter_rms_cross_site_transfer": "Experiment 4: filter RMS cross-site transfer",
}


def clean_axes(ax: plt.Axes) -> None:
    ax.set_ylim(-0.03, 1.04)
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"], rotation=0)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_in_domain(df: pd.DataFrame, experiment_id: str, filename: str) -> None:
    sub = df[df["experiment_id"] == experiment_id].copy()
    sites = ["pohang", "utah_2019", "utah_2023"]
    methods = ["scratch", "contrast", "reconst", "reconst_noanom"]
    fig, axes = plt.subplots(
        len(sites),
        len(METRICS),
        figsize=(13.5, 9.0),
        sharex=True,
        sharey=True,
    )
    for row_idx, site in enumerate(sites):
        site_df = sub[sub["site"] == site]
        for col_idx, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            for method in methods:
                method_df = site_df[site_df["method"] == method].sort_values("fraction")
                if method_df.empty:
                    continue
                ax.plot(
                    method_df["fraction"],
                    method_df[metric],
                    marker="o",
                    linewidth=2.0,
                    markersize=4.5,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
            clean_axes(ax)
            if row_idx == 0:
                ax.set_title(metric_label, fontsize=11, pad=10)
            if col_idx == 0:
                ax.set_ylabel(SITE_LABELS[site], fontsize=10)
            if row_idx == len(sites) - 1:
                ax.set_xlabel("Label fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(EXPERIMENT_TITLES[experiment_id], fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    save(fig, filename)


def pretty_direction(direction: str) -> str:
    source, _, target = direction.partition("_to_")
    return f"{source.replace('_', ' ')} -> {target.replace('_', ' ')}"


def plot_cross_site(df: pd.DataFrame, experiment_id: str, filename: str) -> None:
    sub = df[df["experiment_id"] == experiment_id].copy()
    site_order = ["pohang", "utah_2019", "utah_2023"]
    # Keep columns anchored to the fine-tuning/target site. This makes the
    # cross-site panels comparable by the supervised adaptation domain.
    panel_grid = [
        [
            (source, target)
            for target in site_order
            for source in site_order
            if source != target
        ][row_idx :: 2]
        for row_idx in range(2)
    ]
    metric_styles = {
        "balanced_acc": ("Balanced accuracy", "#4c78a8", "o", "-"),
        "f1": ("F1", "#f58518", "s", "-"),
        "specificity": ("Specificity", "#54a24b", "^", "--"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True, sharey=True)
    for row_idx in range(2):
        for col_idx, (source, target) in enumerate(panel_grid[row_idx]):
            ax = axes[row_idx, col_idx]
            direction = f"{source}_to_{target}"
            direction_df = sub[sub["direction"] == direction].sort_values("fraction")
            if row_idx == 0:
                ax.set_title(f"Fine-tune: {SITE_LABELS[target]}", fontsize=10, pad=8)
            ax.text(
                0.03,
                0.08,
                f"Pretrain: {SITE_LABELS[source].replace(chr(10), ' ')}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#d0d0d0",
                    "alpha": 0.9,
                },
            )
            for metric, (label, color, marker, linestyle) in metric_styles.items():
                if direction_df.empty:
                    continue
                ax.plot(
                    direction_df["fraction"],
                    direction_df[metric],
                    marker=marker,
                    linewidth=2.0,
                    markersize=4.5,
                    linestyle=linestyle,
                    color=color,
                    label=label,
                )
            clean_axes(ax)
            if row_idx == 1:
                ax.set_xlabel("Fine-tuning label fraction")
            if col_idx == 0:
                ax.set_ylabel("Metric")
            if direction_df.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No completed tests",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#777777",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(EXPERIMENT_TITLES[experiment_id], fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.955])
    save(fig, filename)


def main() -> None:
    fc_path = SUMMARY_DIR / "fc_metrics.csv"
    df = pd.read_csv(fc_path)
    for metric, _ in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")

    plot_in_domain(df, "experiment_1_log_env", "experiment_1_log_env")
    plot_in_domain(df, "experiment_2_no_log_env", "experiment_2_no_log_env")
    plot_cross_site(
        df,
        "experiment_3_log_env_cross_site_transfer",
        "experiment_3_log_env_cross_site_transfer",
    )
    plot_cross_site(
        df,
        "experiment_4_filter_rms_cross_site_transfer",
        "experiment_4_filter_rms_cross_site_transfer",
    )
    print(f"[DONE] wrote experiment figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
