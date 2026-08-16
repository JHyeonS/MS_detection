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


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(color="#d7d2c8", linewidth=0.8, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def corr_text(x: pd.Series, y: pd.Series) -> str:
    arr_x = x.to_numpy(float)
    arr_y = y.to_numpy(float)
    ok = np.isfinite(arr_x) & np.isfinite(arr_y)
    if ok.sum() < 3 or np.std(arr_x[ok]) == 0 or np.std(arr_y[ok]) == 0:
        return "r = n/a"
    return f"r = {np.corrcoef(arr_x[ok], arr_y[ok])[0, 1]:.2f}"


def annotate_corr(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    lines = [f"All: {corr_text(df[x_col], df[y_col])}"]
    for label in COLORS:
        sub = df[df["preprocessing_label"].eq(label)]
        lines.append(f"{label}: {corr_text(sub[x_col], sub[y_col])}")
    ax.text(
        0.04,
        0.06,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#374151",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "alpha": 0.92,
        },
    )


def draw_pair(df: pd.DataFrame, name: str, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharey=False)
    specs = [
        ("event_site_swd", "target_balanced_acc", "Event-site SWD", "Balanced accuracy"),
        ("noise_site_swd", "target_specificity", "Noise-site SWD", "Specificity"),
    ]
    for ax, (x_col, y_col, x_label, y_label) in zip(axes, specs):
        for label, color in COLORS.items():
            sub = df[df["preprocessing_label"].eq(label)]
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=54,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.87,
                label=label,
            )
        annotate_corr(ax, df, x_col, y_col)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_ylim(-0.04, 1.05)
        style_axis(ax)
    axes[0].set_title("Event mismatch vs. target recovery", fontsize=11.5, pad=9)
    axes[1].set_title("Noise mismatch vs. false-positive control", fontsize=11.5, pad=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
    if title_suffix:
        fig.suptitle(title_suffix, fontsize=12.5, y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, name)


def draw_b_to_c(df: pd.DataFrame) -> None:
    sub = df[df["direction"].eq("utah_2019_to_utah_2023")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharey=False)
    specs = [
        ("event_site_swd", "target_balanced_acc", "Event-site SWD", "Balanced accuracy"),
        ("noise_site_swd", "target_specificity", "Noise-site SWD", "Specificity"),
    ]
    for ax, (x_col, y_col, x_label, y_label) in zip(axes, specs):
        for label, color in COLORS.items():
            d = sub[sub["preprocessing_label"].eq(label)].sort_values("fraction")
            ax.plot(
                d[x_col],
                d[y_col],
                marker="o",
                markersize=5.8,
                linewidth=1.7,
                color=color,
                alpha=0.92,
                label=label,
            )
            for _, row in d.iterrows():
                ax.text(
                    row[x_col],
                    row[y_col] + 0.025,
                    f"{row['fraction']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#374151",
                )
        annotate_corr(ax, sub, x_col, y_col)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_ylim(-0.04, 1.05)
        style_axis(ax)
    axes[0].set_title("Event mismatch vs. balanced accuracy", fontsize=11.5, pad=9)
    axes[1].set_title("Noise mismatch vs. specificity", fontsize=11.5, pad=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
    fig.suptitle("Utah 2019 -> Utah 2023", fontsize=12.5, y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "classwise_swd_metric_scatter_utah2019_to_utah2023")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    df = pd.read_csv(IN_CSV)
    df = df[df["preprocessing_label"].isin(COLORS)].copy()
    draw_pair(df, "classwise_swd_metric_scatter_overall")
    draw_b_to_c(df)
    out = df[
        [
            "preprocessing_label",
            "direction",
            "direction_short",
            "fraction",
            "event_site_swd",
            "noise_site_swd",
            "target_balanced_acc",
            "target_specificity",
            "target_f1",
        ]
    ].copy()
    out.to_csv(OUT_DIR / "classwise_swd_metric_scatter.csv", index=False)
    print(f"[DONE] wrote classwise SWD metric scatter plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
