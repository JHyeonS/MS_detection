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
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "writing_followup"

COLORS = {
    "Low-pass + RMS": "#bf4b3e",
    "Log-envelope": "#376795",
}
SITE_LABEL = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(color="#d7d2c8", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def annotate_corr(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return
    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    ax.text(
        0.04,
        0.06,
        f"Pearson r = {r:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )


def load_data() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    df = df[df["preprocessing_label"].isin(COLORS)].copy()
    df["target_site_label"] = df["target_site"].map(SITE_LABEL)
    df["direction_label"] = df["source_site"].map(SITE_LABEL) + " -> " + df["target_site"].map(SITE_LABEL)
    return df


def plot_overall(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for label, color in COLORS.items():
        sub = df[df["preprocessing_label"].eq(label)]
        ax.scatter(
            sub["event_site_swd"],
            sub["target_specificity"],
            s=55,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.86,
            label=label,
        )
    annotate_corr(ax, df["event_site_swd"].to_numpy(float), df["target_specificity"].to_numpy(float))
    ax.set_xlabel("Event-site SWD")
    ax.set_ylabel("Target specificity")
    ax.set_ylim(-0.04, 1.05)
    ax.set_title("Event-Domain Latent Mismatch vs. Specificity", fontsize=12.5, pad=11)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "event_site_swd_vs_specificity")


def plot_by_target(df: pd.DataFrame) -> None:
    targets = ["pohang", "utah_2019", "utah_2023"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)
    for ax, target in zip(axes, targets):
        sub_target = df[df["target_site"].eq(target)]
        for label, color in COLORS.items():
            sub = sub_target[sub_target["preprocessing_label"].eq(label)]
            ax.scatter(
                sub["event_site_swd"],
                sub["target_specificity"],
                s=50,
                color=color,
                edgecolor="white",
                linewidth=0.65,
                alpha=0.86,
                label=label,
            )
        annotate_corr(
            ax,
            sub_target["event_site_swd"].to_numpy(float),
            sub_target["target_specificity"].to_numpy(float),
        )
        ax.set_title(f"Target: {SITE_LABEL[target]}", fontsize=11.2, pad=9)
        ax.set_xlabel("Event-site SWD")
        ax.set_ylabel("Target specificity" if ax is axes[0] else "")
        ax.set_ylim(-0.04, 1.05)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.1)
    save(fig, "event_site_swd_vs_specificity_by_target")


def plot_b_to_c(df: pd.DataFrame) -> None:
    sub = df[df["direction"].eq("utah_2019_to_utah_2023")].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for label, color in COLORS.items():
        d = sub[sub["preprocessing_label"].eq(label)].sort_values("fraction")
        ax.plot(
            d["event_site_swd"],
            d["target_specificity"],
            marker="o",
            linewidth=1.8,
            markersize=6,
            color=color,
            alpha=0.9,
            label=label,
        )
        for _, row in d.iterrows():
            ax.text(
                row["event_site_swd"],
                row["target_specificity"] + 0.025,
                f"{row['fraction']:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color="#374151",
            )
    annotate_corr(sub.axes[0] if False else ax, sub["event_site_swd"].to_numpy(float), sub["target_specificity"].to_numpy(float))
    ax.set_xlabel("Event-site SWD")
    ax.set_ylabel("Target specificity")
    ax.set_ylim(-0.04, 1.05)
    ax.set_title("Utah 2019 -> Utah 2023: Event SWD vs. Specificity", fontsize=12.2, pad=11)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "event_site_swd_vs_specificity_utah2019_to_utah2023")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    df = load_data()
    plot_overall(df)
    plot_by_target(df)
    plot_b_to_c(df)
    out = df[
        [
            "preprocessing_label",
            "direction_label",
            "fraction",
            "event_site_swd",
            "noise_site_swd",
            "target_specificity",
            "target_balanced_acc",
            "target_f1",
        ]
    ].copy()
    out.to_csv(OUT_DIR / "event_site_swd_vs_specificity.csv", index=False)
    print(f"[DONE] wrote event-SWD specificity plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
