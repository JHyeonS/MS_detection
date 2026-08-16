#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "center" / "cross_site_classwise_swd"
OUT_DIR = IN_DIR / "utah2019_to_utah2023"
CSV_PATH = IN_DIR / "cross_site_classwise_swd.csv"

COLORS = {
    "Low-pass + RMS": "#bf4b3e",
    "Log-envelope": "#376795",
}
MARKERS = {
    "Low-pass + RMS": "s",
    "Log-envelope": "o",
}
PREPROC_ORDER = ["Low-pass + RMS", "Log-envelope"]
FRACTION_ORDER = [0.05, 0.10, 0.25, 0.50, 1.00]
LOW_LABEL_FRACTIONS = [0.05, 0.10, 0.25]


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


def load_subset() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    sub = df[df["direction"].eq("utah_2019_to_utah_2023")].copy()
    sub["preprocessing_label"] = pd.Categorical(sub["preprocessing_label"], PREPROC_ORDER, ordered=True)
    sub = sub.sort_values(["preprocessing_label", "fraction"])
    return sub


def plot_low_label_swd_specificity(sub: pd.DataFrame) -> None:
    data = sub[sub["fraction"].isin(LOW_LABEL_FRACTIONS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    panels = [
        ("event_site_swd", "Event-domain SWD", "Cross-site SWD"),
        ("target_specificity", "Target specificity", "Specificity"),
    ]
    for ax, (col, title, ylabel) in zip(axes, panels):
        for label in PREPROC_ORDER:
            d = data[data["preprocessing_label"].astype(str).eq(label)]
            ax.plot(
                d["fraction"],
                d[col],
                marker=MARKERS[label],
                markersize=6,
                linewidth=2.0,
                color=COLORS[label],
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xticks(LOW_LABEL_FRACTIONS)
        ax.set_xticklabels(["0.05", "0.10", "0.25"])
        ax.set_xlabel("Target label fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, pad=10)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "utah2019_to_utah2023_lowlabel_event_swd_specificity")


def plot_low_label_swd_balacc_specificity(sub: pd.DataFrame) -> None:
    data = sub[sub["fraction"].isin(LOW_LABEL_FRACTIONS)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.7))
    panels = [
        ("event_site_swd", "Event-domain SWD", "Cross-site SWD"),
        ("target_balanced_acc", "Balanced accuracy", "Balanced accuracy"),
        ("target_specificity", "Specificity", "Specificity"),
    ]
    for ax, (col, title, ylabel) in zip(axes, panels):
        for label in PREPROC_ORDER:
            d = data[data["preprocessing_label"].astype(str).eq(label)]
            ax.plot(
                d["fraction"],
                d[col],
                marker=MARKERS[label],
                markersize=6,
                linewidth=2.0,
                color=COLORS[label],
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xticks(LOW_LABEL_FRACTIONS)
        ax.set_xticklabels(["0.05", "0.10", "0.25"])
        ax.set_xlabel("Target label fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, pad=10)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "utah2019_to_utah2023_lowlabel_swd_balacc_specificity")


def plot_full_fraction_lines(sub: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    panels = [
        ("event_site_swd", "Event-domain SWD", "Cross-site SWD"),
        ("all_site_swd", "Overall site SWD", "Cross-site SWD"),
        ("target_balanced_acc", "Balanced accuracy", "Balanced accuracy"),
        ("target_specificity", "Specificity", "Specificity"),
    ]
    for ax, (col, title, ylabel) in zip(axes.flat, panels):
        for label in PREPROC_ORDER:
            d = sub[sub["preprocessing_label"].astype(str).eq(label)]
            ax.plot(
                d["fraction"],
                d[col],
                marker=MARKERS[label],
                markersize=5.5,
                linewidth=2.0,
                color=COLORS[label],
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xticks(FRACTION_ORDER)
        ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
        ax.set_xlabel("Target label fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, pad=10)
        style_axis(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "utah2019_to_utah2023_full_fraction_swd_metrics")


def plot_paired_bars_low_label(sub: pd.DataFrame) -> None:
    data = sub[sub["fraction"].isin(LOW_LABEL_FRACTIONS)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    width = 0.32
    x = np.arange(len(LOW_LABEL_FRACTIONS))
    labels_frac = ["0.05", "0.10", "0.25"]
    for ax, metric, title, ylabel in [
        (axes[0], "event_site_swd", "Event-domain mismatch", "Event-site SWD"),
        (axes[1], "target_specificity", "Target-domain recovery", "Specificity"),
    ]:
        for offset, label in zip([-width / 2, width / 2], PREPROC_ORDER):
            d = data[data["preprocessing_label"].astype(str).eq(label)].set_index("fraction").loc[LOW_LABEL_FRACTIONS]
            ax.bar(
                x + offset,
                d[metric].to_numpy(),
                width=width,
                color=COLORS[label],
                label=label,
                alpha=0.88,
            )
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel(ylabel)
        style_axis(ax)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_frac)
    axes[1].set_xlabel("Target label fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "utah2019_to_utah2023_lowlabel_paired_bar_swd_specificity")


def plot_paired_bars_low_label_1x6(sub: pd.DataFrame) -> None:
    data = sub[sub["fraction"].isin(LOW_LABEL_FRACTIONS)].copy()
    fig, axes = plt.subplots(1, 6, figsize=(18.4, 3.8), sharey=False)
    panel_specs = []
    for frac in LOW_LABEL_FRACTIONS:
        panel_specs.append(("event_site_swd", frac, "Event-site SWD", "SWD"))
    for frac in LOW_LABEL_FRACTIONS:
        panel_specs.append(("target_specificity", frac, "Specificity", "Specificity"))

    for ax, (metric, frac, group_title, ylabel) in zip(axes, panel_specs):
        d = data[data["fraction"].eq(frac)].set_index("preprocessing_label")
        values = [float(d.loc[label, metric]) for label in PREPROC_ORDER]
        ax.bar(
            [0, 1],
            values,
            color=[COLORS[label] for label in PREPROC_ORDER],
            width=0.64,
            alpha=0.9,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low-pass\n+ RMS", "Log-\nenvelope"], fontsize=8)
        ax.set_title(f"{group_title}\nfrac={frac:.2f}", fontsize=10.5, pad=9)
        ax.set_ylabel(ylabel if ax in (axes[0], axes[3]) else "")
        style_axis(ax)
        ymax = max(values) * 1.18 if max(values) > 0 else 1.0
        if metric == "target_specificity":
            ymax = 1.05
        ax.set_ylim(0, ymax)
        for xpos, value in zip([0, 1], values):
            ax.text(xpos, value + ymax * 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8.2)

    fig.text(
        0.255,
        0.985,
        "Latent mismatch",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#374151",
    )
    fig.text(
        0.745,
        0.985,
        "Target recovery",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#374151",
    )
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.23, top=0.84, wspace=0.55)
    save(fig, "utah2019_to_utah2023_lowlabel_paired_bar_swd_specificity_1x6")


def plot_heatmap(sub: pd.DataFrame) -> None:
    metrics = [
        ("event_site_swd", "Event SWD"),
        ("noise_site_swd", "Noise SWD"),
        ("all_site_swd", "All SWD"),
        ("target_balanced_acc", "BalAcc"),
        ("target_specificity", "Spec"),
    ]
    rows = []
    ylabels = []
    for label in PREPROC_ORDER:
        d = sub[sub["preprocessing_label"].astype(str).eq(label)].set_index("fraction").loc[FRACTION_ORDER]
        for metric, metric_label in metrics:
            rows.append(d[metric].to_numpy())
            ylabels.append(f"{label}\n{metric_label}")
    matrix = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(FRACTION_ORDER)))
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8.2)
    ax.set_xlabel("Target label fraction")
    ax.set_title("Utah 2019 -> Utah 2023: SWD and target metrics", fontsize=12, pad=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.2, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "utah2019_to_utah2023_swd_metric_heatmap")


def write_subset_csv(sub: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "preprocessing_label",
        "fraction",
        "event_site_swd",
        "noise_site_swd",
        "all_site_swd",
        "target_balanced_acc",
        "target_f1",
        "target_specificity",
    ]
    sub[cols].to_csv(OUT_DIR / "utah2019_to_utah2023_classwise_swd_metrics.csv", index=False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    sub = load_subset()
    write_subset_csv(sub)
    plot_low_label_swd_specificity(sub)
    plot_low_label_swd_balacc_specificity(sub)
    plot_full_fraction_lines(sub)
    plot_paired_bars_low_label(sub)
    plot_paired_bars_low_label_1x6(sub)
    plot_heatmap(sub)
    print(f"[DONE] wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
