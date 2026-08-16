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
    / "writing_followup"
    / "classwise_swd_metric_scatter.csv"
)
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "writing_followup"

COLORS = {
    "Low-pass + RMS": "#bf4b3e",
    "Log-envelope": "#376795",
}
DIRECTION_LABELS = {
    "pohang_to_utah_2019": "Pohang -> Utah 2019",
    "pohang_to_utah_2023": "Pohang -> Utah 2023",
    "utah_2019_to_pohang": "Utah 2019 -> Pohang",
    "utah_2019_to_utah_2023": "Utah 2019 -> Utah 2023",
    "utah_2023_to_pohang": "Utah 2023 -> Pohang",
    "utah_2023_to_utah_2019": "Utah 2023 -> Utah 2019",
}
DIRECTION_ORDER = [
    "pohang_to_utah_2019",
    "utah_2019_to_pohang",
    "pohang_to_utah_2023",
    "utah_2023_to_pohang",
    "utah_2019_to_utah_2023",
    "utah_2023_to_utah_2019",
]


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def corr(x: pd.Series, y: pd.Series) -> float:
    xx = x.to_numpy(float)
    yy = y.to_numpy(float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    if ok.sum() < 3 or np.std(xx[ok]) == 0 or np.std(yy[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(xx[ok], yy[ok])[0, 1])


def style_axis(ax: plt.Axes) -> None:
    ax.grid(color="#ddd8ce", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.6)


def plot_grid(df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, name: str) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(18.6, 3.25), sharey=True)
    for ax, direction in zip(axes, DIRECTION_ORDER):
        sub_dir = df[df["direction"].eq(direction)]
        for preproc, color in COLORS.items():
            sub = sub_dir[sub_dir["preprocessing_label"].eq(preproc)].sort_values("fraction")
            ax.plot(
                sub[x_col],
                sub[y_col],
                marker="o",
                markersize=4.2,
                linewidth=1.15,
                color=color,
                alpha=0.9,
                label=preproc,
            )
        r = corr(sub_dir[x_col], sub_dir[y_col])
        r_text = "r=n/a" if np.isnan(r) else f"r={r:.2f}"
        ax.text(
            0.05,
            0.07,
            r_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
        )
        ax.set_title(DIRECTION_LABELS[direction], fontsize=8.8, pad=8)
        ax.set_xlabel(x_label, fontsize=8.4)
        ax.set_ylim(-0.04, 1.05)
        if ax is axes[0]:
            ax.set_ylabel(y_label, fontsize=8.8)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.9)
    save(fig, name)


def plot_combined(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(18.6, 6.2), sharey="row")
    rows = [
        ("event_site_swd", "target_balanced_acc", "Event-site SWD", "Balanced accuracy"),
        ("noise_site_swd", "target_specificity", "Noise-site SWD", "Specificity"),
    ]
    for row_idx, (x_col, y_col, x_label, y_label) in enumerate(rows):
        for ax, direction in zip(axes[row_idx], DIRECTION_ORDER):
            sub_dir = df[df["direction"].eq(direction)]
            for preproc, color in COLORS.items():
                sub = sub_dir[sub_dir["preprocessing_label"].eq(preproc)].sort_values("fraction")
                ax.plot(
                    sub[x_col],
                    sub[y_col],
                    marker="o",
                    markersize=4.0,
                    linewidth=1.1,
                    color=color,
                    alpha=0.9,
                    label=preproc,
                )
            r = corr(sub_dir[x_col], sub_dir[y_col])
            r_text = "r=n/a" if np.isnan(r) else f"r={r:.2f}"
            ax.text(
                0.05,
                0.07,
                r_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.4,
                color="#374151",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
            )
            if row_idx == 0:
                ax.set_title(DIRECTION_LABELS[direction], fontsize=8.8, pad=8)
            ax.set_xlabel(x_label, fontsize=8.2)
            ax.set_ylim(-0.04, 1.05)
            if ax is axes[row_idx, 0]:
                ax.set_ylabel(y_label, fontsize=8.8)
            style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.9, h_pad=1.1)
    save(fig, "classwise_swd_metric_scatter_all_directions_combined")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    df = pd.read_csv(IN_CSV)
    plot_grid(
        df,
        "event_site_swd",
        "target_balanced_acc",
        "Event-site SWD",
        "Balanced accuracy",
        "event_site_swd_vs_balanced_acc_all_directions",
    )
    plot_grid(
        df,
        "noise_site_swd",
        "target_specificity",
        "Noise-site SWD",
        "Specificity",
        "noise_site_swd_vs_specificity_all_directions",
    )
    plot_combined(df)

    summary = []
    for direction, sub in df.groupby("direction"):
        summary.append(
            {
                "direction": direction,
                "direction_label": DIRECTION_LABELS.get(direction, direction),
                "n": len(sub),
                "event_site_swd_vs_balanced_acc_r": corr(sub["event_site_swd"], sub["target_balanced_acc"]),
                "noise_site_swd_vs_specificity_r": corr(sub["noise_site_swd"], sub["target_specificity"]),
            }
        )
    pd.DataFrame(summary).to_csv(OUT_DIR / "classwise_swd_metric_scatter_direction_summary.csv", index=False)
    print(f"[DONE] wrote all-direction classwise SWD scatter plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
