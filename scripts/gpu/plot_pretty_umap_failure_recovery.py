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
    / "leftwing"
    / "cross_site_failure_success_umap_utah2019_to_utah2023_frac0p25.csv"
)
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "leftwing"

COLORS = {
    0: "#5f8fd3",
    1: "#f27d42",
}
LABELS = {
    0: "Noise",
    1: "Event",
}
PANELS = [
    ("failure_lowpass_rms", "Low-pass + RMS", "Failure"),
    ("success_logenv", "Log-envelope", "Recovery"),
]


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("UMAP 1", fontsize=7.8, color="#5f6673")
    ax.set_ylabel("UMAP 2", fontsize=7.8, color="#5f6673")
    ax.tick_params(labelsize=6.9, colors="#6b7280", length=2.2, width=0.55)
    ax.grid(color="#e9dfd2", linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#c7cbd1")


def metric_text(df: pd.DataFrame) -> str:
    y = df["label"].astype(int).to_numpy()
    p = df["pred_fc"].astype(int).to_numpy()
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    bal = 0.5 * (sens + spec)
    return f"BalAcc {bal:.3f}   Spec {spec:.3f}   FP {fp}/{tn + fp}"


def pad_limits(values: np.ndarray, frac: float = 0.08) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    pad = max((hi - lo) * frac, 0.5)
    return lo - pad, hi + pad


def draw_panel(ax: plt.Axes, df: pd.DataFrame, title: str, subtitle: str, letter: str) -> None:
    for lab in (0, 1):
        sub = df[df["label"].astype(int).eq(lab)]
        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=9.5,
            c=COLORS[lab],
            label=LABELS[lab],
            alpha=0.74,
            linewidths=0,
            rasterized=True,
        )

    wrong = df[~df["is_correct"].astype(bool)]
    if not wrong.empty:
        ax.scatter(
            wrong["umap_x"],
            wrong["umap_y"],
            s=16,
            facecolors="none",
            edgecolors="#0f172a",
            linewidths=0.48,
            label="Misclassified",
            rasterized=True,
        )

    ax.text(
        0.0,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.08,
        1.08,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="normal",
        color="#111827",
    )
    ax.text(
        0.08,
        1.015,
        f"{subtitle} | {metric_text(df)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.75,
        color="#6b7280",
    )
    style_axis(ax)


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

    df = pd.read_csv(IN_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(8.25, 3.15))
    for ax, (setting, preproc, state), letter in zip(axes, PANELS, ("A", "B")):
        sub = df[df["setting"].eq(setting)].copy()
        draw_panel(ax, sub, preproc, state, letter)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.085),
        ncol=3,
        frameon=False,
        fontsize=8.8,
        markerscale=1.0,
        handletextpad=0.35,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0.005, 0.115, 0.995, 0.985), w_pad=0.78)
    save(fig, "pretty_umap_failure_recovery_utah2019_to_utah2023_frac0p25")

    for setting, preproc, state in PANELS:
        sub = df[df["setting"].eq(setting)].copy()
        fig, ax = plt.subplots(1, 1, figsize=(4.25, 3.15))
        draw_panel(ax, sub, preproc, state, "A")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.09),
            ncol=3,
            frameon=False,
            fontsize=8.4,
            markerscale=0.9,
            handletextpad=0.25,
            columnspacing=0.75,
        )
        fig.tight_layout(rect=(0.005, 0.13, 0.995, 0.985))
        save(fig, f"pretty_umap_{setting}_utah2019_to_utah2023_frac0p25")

    print(f"[DONE] wrote pretty UMAP plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
