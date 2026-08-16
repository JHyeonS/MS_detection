#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = ROOT / "figures" / "current_results_summary" / "controlled_latent_diagnosis_axes"
LATENT_DIR = ROOT / "figures" / "current_results_summary" / "latent_diagnosis"
SITE_DIR = LATENT_DIR / "site-shift"
FILTER_DIR = LATENT_DIR / "filter-various"

UMAP_CSV = CONTROL_DIR / "axis2_utah2019_to_utah2023_frac0p25_source_target_umap_points.csv"
SWD_CSV = CONTROL_DIR / "axis2_utah2019_to_utah2023_frac0p25_source_target_swd.csv"
GAIN_CSV = CONTROL_DIR / "axis2_utah2019_to_utah2023_frac0p25_transfer_gain.csv"
SITE_MATRIX_PDF = CONTROL_DIR / "axis1_site_shift_fixed_preprocessing_gain_matrix.pdf"
SITE_MATRIX_PNG = CONTROL_DIR / "axis1_site_shift_fixed_preprocessing_gain_matrix.png"

PREPROCESSING = ["raw", "filter_rms", "logenv"]
PREPROC_LABELS = {
    "raw": "Raw",
    "filter_rms": "Low-pass",
    "logenv": "Log-envelope",
}
PREPROC_COLORS = {
    "raw": "#6b7280",
    "filter_rms": "#bf4b3e",
    "logenv": "#376795",
}
LABEL_COLORS = {0: "#4f8fd8", 1: "#f07c3e"}
LABEL_NAMES = {0: "Noise", 1: "Event"}


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def style_axis(ax: plt.Axes) -> None:
    ax.grid(color="#e8ded2", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.2, colors="#5f6673", length=2.2, width=0.55)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#c7cbd1")


def draw_umap(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    markers = {"Source": "^", "Target": "o"}
    sizes = {"Source": 12, "Target": 13}
    alphas = {"Source": 0.40, "Target": 0.76}
    for domain in ("Source", "Target"):
        for label in (0, 1):
            sub = df[df["domain"].eq(domain) & df["label"].astype(int).eq(label)]
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
    correct = as_bool(df["is_correct"])
    wrong = df[df["domain"].eq("Target") & (~correct)]
    if not wrong.empty:
        ax.scatter(
            wrong["umap_x"],
            wrong["umap_y"],
            s=24,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.52,
            label="Target error",
            rasterized=True,
        )
    ax.set_title(title, fontsize=10.2, fontweight="normal", pad=8)
    ax.set_xlabel("UMAP 1", fontsize=8.0)
    ax.set_ylabel("UMAP 2", fontsize=8.0)
    style_axis(ax)


def plot_individual_umaps(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    for preproc in PREPROCESSING:
        sub = df[df["preprocessing"].eq(preproc)].copy()
        fig, ax = plt.subplots(1, 1, figsize=(4.25, 3.25))
        draw_umap(ax, sub, f"{PREPROC_LABELS[preproc]} latent UMAP")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.095),
            ncol=3,
            frameon=False,
            fontsize=7.8,
            handletextpad=0.25,
            columnspacing=0.65,
        )
        fig.tight_layout(rect=(0.005, 0.11, 0.995, 0.99))
        save(fig, out_dir, f"{prefix}_umap_{preproc}")


def plot_umap_multipanel(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.35))
    for ax, preproc in zip(axes, PREPROCESSING):
        sub = df[df["preprocessing"].eq(preproc)].copy()
        draw_umap(ax, sub, PREPROC_LABELS[preproc])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=5,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.28,
        columnspacing=0.75,
    )
    fig.tight_layout(rect=(0.005, 0.12, 0.995, 0.99), w_pad=0.9)
    save(fig, out_dir, f"{prefix}_umap_all_preprocessing")


def plot_swd(swd: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    metrics = [("event_swd", "Event"), ("noise_swd", "Noise"), ("all_swd", "All")]
    x = np.arange(len(metrics))
    width = 0.23
    fig, ax = plt.subplots(1, 1, figsize=(5.1, 3.25))
    for offset, preproc in zip([-width, 0, width], PREPROCESSING):
        row = swd[swd["preprocessing"].eq(preproc)].iloc[0]
        values = [float(row[col]) for col, _ in metrics]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=PREPROC_COLORS[preproc],
            label=PREPROC_LABELS[preproc],
            alpha=0.92,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Source-target SWD", fontsize=8.7)
    ax.set_title("Source-target latent distribution mismatch", fontsize=10.3, fontweight="normal", pad=8)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    fig.tight_layout()
    save(fig, out_dir, f"{prefix}_swd_analysis")


def plot_transfer_gain_bar(gain: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    x = np.arange(len(PREPROCESSING))
    scratch = [float(gain[gain["preprocessing"].eq(p)]["scratch_target_balanced_acc"].iloc[0]) for p in PREPROCESSING]
    transfer = [float(gain[gain["preprocessing"].eq(p)]["balanced_acc"].iloc[0]) for p in PREPROCESSING]
    transfer_gain = [float(gain[gain["preprocessing"].eq(p)]["transfer_gain_balanced_acc"].iloc[0]) for p in PREPROCESSING]
    width = 0.34
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.35))
    ax.bar(x - width / 2, scratch, width=width, color="#d8d3ca", label="Target scratch")
    ax.bar(
        x + width / 2,
        transfer,
        width=width,
        color=[PREPROC_COLORS[p] for p in PREPROCESSING],
        label="Cross-site transfer",
    )
    for idx, gain_value in enumerate(transfer_gain):
        ax.text(idx, max(scratch[idx], transfer[idx]) + 0.035, f"{gain_value:+.2f}", ha="center", va="bottom", fontsize=8.0)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([PREPROC_LABELS[p] for p in PREPROCESSING], rotation=15, ha="right")
    ax.set_ylabel("Balanced accuracy", fontsize=8.7)
    ax.set_title("Transfer gain relative to target scratch", fontsize=10.3, fontweight="normal", pad=8)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.0, loc="lower left")
    fig.tight_layout()
    save(fig, out_dir, f"{prefix}_transfer_gain")


def copy_site_shift_matrix() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    if SITE_MATRIX_PDF.exists():
        shutil.copy2(SITE_MATRIX_PDF, SITE_DIR / "case1_transfer_gain_matrix.pdf")
    if SITE_MATRIX_PNG.exists():
        shutil.copy2(SITE_MATRIX_PNG, SITE_DIR / "case1_transfer_gain_matrix.png")


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
    umap = pd.read_csv(UMAP_CSV)
    swd = pd.read_csv(SWD_CSV)
    gain = pd.read_csv(GAIN_CSV)

    copy_site_shift_matrix()
    plot_individual_umaps(umap, SITE_DIR, "case1_site_shift")
    plot_umap_multipanel(umap, SITE_DIR, "case1_site_shift")
    plot_swd(swd, SITE_DIR, "case1_site_shift")
    plot_transfer_gain_bar(gain, SITE_DIR, "case1_site_shift")

    plot_individual_umaps(umap, FILTER_DIR, "case2_filter_various")
    plot_umap_multipanel(umap, FILTER_DIR, "case2_filter_various")
    plot_swd(swd, FILTER_DIR, "case2_filter_various")
    plot_transfer_gain_bar(gain, FILTER_DIR, "case2_filter_various")
    print(f"[DONE] wrote separate latent diagnosis panels to {LATENT_DIR}")


if __name__ == "__main__":
    main()
