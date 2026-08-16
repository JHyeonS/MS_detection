#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1"
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "seg"

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
FRACTION_LABELS = ["5%", "10%", "25%", "50%", "100%"]

PREPROCESSING_ROOTS = {
    "Low-pass + RMS": RUN_ROOT / "filter_rms_cross_site_reconst_swd_interval10_v1",
    "Log-envelope": RUN_ROOT / "logenv_cross_site_reconst_swd_interval10_v1",
}


def frac_tag(fraction: float) -> str:
    return {
        0.05: "0p05",
        0.10: "0p1",
        0.25: "0p25",
        0.50: "0p5",
        1.00: "1",
    }[fraction]


def read_failure_case() -> pd.DataFrame:
    rows = []
    direction = "utah_2019_to_utah_2023"
    target_prefix = "base_utah_2023"
    for preprocessing, root in PREPROCESSING_ROOTS.items():
        for fraction in FRACTIONS:
            metrics_path = (
                root
                / direction
                / "reconst"
                / "test"
                / f"{target_prefix}__frac{frac_tag(fraction)}"
                / "test_metrics_fixed_threshold.json"
            )
            with metrics_path.open() as f:
                metrics = json.load(f)["fc_metrics_fixed_threshold"]
            rows.append(
                {
                    "preprocessing": preprocessing,
                    "source": "Utah 2019",
                    "target": "Utah 2023",
                    "fraction": fraction,
                    "f1": metrics["f1"],
                    "balanced_acc": metrics["balanced_acc"],
                    "specificity": metrics["specificity"],
                    "recall": metrics["recall"],
                }
            )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels(FRACTION_LABELS, fontsize=9)
    ax.set_xlabel("Target label fraction", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.9)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#374151", labelsize=9)


def plot_failure_case(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    colors = {
        "f1": "#2563eb",
        "balanced_acc": "#111827",
        "specificity": "#dc2626",
        "recall": "#16a34a",
    }
    labels = {
        "f1": "F1",
        "balanced_acc": "Balanced acc.",
        "specificity": "Specificity",
        "recall": "Recall",
    }
    markers = {
        "f1": "o",
        "balanced_acc": "s",
        "specificity": "^",
        "recall": "D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.25), sharey=True)
    for ax, preprocessing in zip(axes, PREPROCESSING_ROOTS):
        sub = df[df["preprocessing"] == preprocessing].sort_values("fraction")
        for metric in ("f1", "balanced_acc", "specificity", "recall"):
            ax.plot(
                sub["fraction"],
                sub[metric],
                color=colors[metric],
                marker=markers[metric],
                linewidth=2.0,
                markersize=4.8,
                label=labels[metric],
            )
        ax.set_title(preprocessing, fontsize=12, fontweight="normal", pad=8)
        style_axis(ax, "Score" if ax is axes[0] else None)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0.02, 0.12, 1.0, 1.0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / "seg_cross_site_failure_utah2019_to_utah2023"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    csv_path = OUT_DIR / "seg_cross_site_failure_utah2019_to_utah2023_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")
    print(f"[DONE] wrote {csv_path}")


def plot_balanced_accuracy_only(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    colors = {
        "Log-envelope": "#0f766e",
        "Low-pass + RMS": "#b45309",
    }
    markers = {
        "Log-envelope": "o",
        "Low-pass + RMS": "s",
    }

    fig, ax = plt.subplots(1, 1, figsize=(5.7, 3.35))
    for preprocessing in ("Log-envelope", "Low-pass + RMS"):
        sub = df[df["preprocessing"] == preprocessing].sort_values("fraction")
        ax.plot(
            sub["fraction"],
            sub["balanced_acc"],
            color=colors[preprocessing],
            marker=markers[preprocessing],
            linewidth=2.4,
            markersize=5.4,
            label=preprocessing,
        )

    style_axis(ax, "Balanced accuracy")
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#e5e7eb",
        fontsize=9.5,
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / "seg_cross_site_failure_utah2019_to_utah2023_balanced_acc_only"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")


def main() -> None:
    df = read_failure_case()
    plot_failure_case(df)
    plot_balanced_accuracy_only(df)


if __name__ == "__main__":
    main()
