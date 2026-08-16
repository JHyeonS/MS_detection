#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "temp" / "current_results_summary"
OUT_DIR = SUMMARY_DIR / "figures"

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
SITE_ORDER = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Site A\nPohang",
    "utah_2019": "Site B\nUtah 2019",
    "utah_2023": "Site C\nUtah 2023",
}
SITE_SHORT = {
    "pohang": "A",
    "utah_2019": "B",
    "utah_2023": "C",
}
DATASETS = [
    {
        "label": "Log-envelope + RMS",
        "scratch_exp": "experiment_1_log_env",
        "transfer_exp": "experiment_3_log_env_cross_site_transfer",
    },
    {
        "label": "Filter + RMS",
        "scratch_exp": "experiment_2_no_log_env",
        "transfer_exp": "experiment_4_filter_rms_cross_site_transfer",
    },
]
COLORS = {
    "scratch": "#2f2f2f",
    "pohang": "#4c78a8",
    "utah_2019": "#f58518",
    "utah_2023": "#54a24b",
}


def clean_axes(ax: plt.Axes) -> None:
    ax.set_ylim(-0.03, 1.04)
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(SUMMARY_DIR / "fc_metrics.csv")
    df = df[df["test_done"].astype(bool)].copy()
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["balanced_acc"] = pd.to_numeric(df["balanced_acc"], errors="coerce")

    fig, axes = plt.subplots(
        len(DATASETS),
        len(SITE_ORDER),
        figsize=(13.5, 6.8),
        sharex=True,
        sharey=True,
    )

    for row_idx, dataset_spec in enumerate(DATASETS):
        scratch_df = df[
            (df["experiment_id"] == dataset_spec["scratch_exp"])
            & (df["method"] == "scratch")
        ].copy()
        transfer_df = df[df["experiment_id"] == dataset_spec["transfer_exp"]].copy()

        for col_idx, target in enumerate(SITE_ORDER):
            ax = axes[row_idx, col_idx]

            target_scratch = scratch_df[scratch_df["site"] == target].sort_values("fraction")
            if not target_scratch.empty:
                ax.plot(
                    target_scratch["fraction"],
                    target_scratch["balanced_acc"],
                    marker="o",
                    linewidth=2.4,
                    markersize=4.8,
                    color=COLORS["scratch"],
                    label="Scratch target-only",
                )

            for source in SITE_ORDER:
                if source == target:
                    continue
                direction = f"{source}_to_{target}"
                direction_df = transfer_df[
                    transfer_df["direction"] == direction
                ].sort_values("fraction")
                if direction_df.empty:
                    continue
                ax.plot(
                    direction_df["fraction"],
                    direction_df["balanced_acc"],
                    marker="s",
                    linewidth=2.0,
                    markersize=4.5,
                    linestyle="--",
                    color=COLORS[source],
                    label=f"Pretrain {SITE_SHORT[source]} -> fine-tune {SITE_SHORT[target]}",
                )

            clean_axes(ax)
            if row_idx == 0:
                ax.set_title(f"Fine-tune target: {SITE_LABELS[target]}", fontsize=10, pad=9)
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_spec['label']}\nBalanced accuracy", fontsize=10)
            if row_idx == len(DATASETS) - 1:
                ax.set_xlabel("Fine-tuning label fraction")

    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=9,
    )
    fig.suptitle(
        "Cross-site transfer learning vs target-site scratch baseline",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    save(fig, "cross_site_transfer_vs_scratch_balanced_accuracy")
    print(f"[DONE] wrote {OUT_DIR / 'cross_site_transfer_vs_scratch_balanced_accuracy.pdf'}")


if __name__ == "__main__":
    main()
