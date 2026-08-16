#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1" / "preprocessing_cross_reconst_pre50_v1"
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "writing_followup"

SITES = {
    "pohang": ("Pohang", "pohang"),
    "utah_2019": ("Utah 2019", "base_utah_2019"),
    "utah_2023": ("Utah 2023", "base_utah_2023"),
}
PAIRS = {
    "filter_rms_to_logenv": "Low-pass + RMS -> Log-envelope",
    "logenv_to_filter_rms": "Log-envelope -> Low-pass + RMS",
}
FRACTIONS = ["0p05", "0p1", "0p25", "0p5", "1"]
FRACTION_VALUE = {"0p05": 0.05, "0p1": 0.10, "0p25": 0.25, "0p5": 0.50, "1": 1.00}
FRACTION_LABEL = {"0p05": "0.05", "0p1": "0.10", "0p25": "0.25", "0p5": "0.50", "1": "1.00"}
COLORS = {
    "filter_rms_to_logenv": "#bf4b3e",
    "logenv_to_filter_rms": "#376795",
}
MARKERS = {
    "filter_rms_to_logenv": "s",
    "logenv_to_filter_rms": "o",
}


def read_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fc_metrics_fixed_threshold", {})


def collect() -> pd.DataFrame:
    rows = []
    for site, (site_label, exp) in SITES.items():
        for pair, pair_label in PAIRS.items():
            for tag in FRACTIONS:
                path = RUN_ROOT / site / pair / "reconst" / "test" / f"{exp}__frac{tag}" / "test_metrics_fixed_threshold.json"
                if not path.exists():
                    rows.append(
                        {
                            "site": site,
                            "site_label": site_label,
                            "preprocessing_transfer": pair,
                            "preprocessing_transfer_label": pair_label,
                            "fraction": FRACTION_VALUE[tag],
                            "fraction_label": FRACTION_LABEL[tag],
                            "status": "missing",
                        }
                    )
                    continue
                m = read_metrics(path)
                rows.append(
                    {
                        "site": site,
                        "site_label": site_label,
                        "preprocessing_transfer": pair,
                        "preprocessing_transfer_label": pair_label,
                        "fraction": FRACTION_VALUE[tag],
                        "fraction_label": FRACTION_LABEL[tag],
                        "balanced_acc": m.get("balanced_acc"),
                        "f1": m.get("f1"),
                        "specificity": m.get("specificity"),
                        "recall": m.get("recall"),
                        "precision": m.get("precision"),
                        "status": "complete",
                        "metrics_path": str(path.relative_to(ROOT)),
                    }
                )
    return pd.DataFrame(rows)


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


def plot_metric_grid(df: pd.DataFrame, metric: str, ylabel: str, name: str) -> None:
    complete = df[df["status"].eq("complete")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.45), sharey=True)
    for ax, (site, (site_label, _)) in zip(axes, SITES.items()):
        sub = complete[complete["site"].eq(site)]
        for pair, pair_label in PAIRS.items():
            d = sub[sub["preprocessing_transfer"].eq(pair)].sort_values("fraction")
            ax.plot(
                d["fraction"],
                d[metric],
                marker=MARKERS[pair],
                markersize=5.8,
                linewidth=2.0,
                color=COLORS[pair],
                label=pair_label,
            )
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.10, 0.25, 0.50, 1.00])
        ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(site_label, fontsize=11.5, pad=10)
        ax.set_xlabel("Target label fraction")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.1)
    save(fig, name)


def plot_heatmap(df: pd.DataFrame, metric: str, title: str, name: str) -> None:
    complete = df[df["status"].eq("complete")].copy()
    rows = []
    labels = []
    for site, (site_label, _) in SITES.items():
        for pair, pair_label in PAIRS.items():
            d = complete[complete["site"].eq(site) & complete["preprocessing_transfer"].eq(pair)].set_index("fraction")
            rows.append(d.loc[[0.05, 0.10, 0.25, 0.50, 1.00], metric].to_numpy())
            labels.append(f"{site_label}\n{pair_label}")
    matrix = pd.DataFrame(rows).to_numpy()
    fig, ax = plt.subplots(figsize=(8.0, 4.85))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Target label fraction")
    ax.set_title(title, fontsize=12, pad=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.2, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, name)


def write_note(df: pd.DataFrame) -> None:
    complete = df[df["status"].eq("complete")]
    missing = df[df["status"].eq("missing")]
    summary = complete.groupby(["site_label", "preprocessing_transfer_label"])[["balanced_acc", "specificity", "f1"]].mean().round(4)
    summary_reset = summary.reset_index()
    table_lines = [
        "| Site | Preprocessing transfer | Balanced accuracy | Specificity | F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary_reset.iterrows():
        table_lines.append(
            f"| {row['site_label']} | {row['preprocessing_transfer_label']} | "
            f"{row['balanced_acc']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} |"
        )
    lines = [
        "# Writing Follow-Up Results",
        "",
        "## Preprocessing-Cross Transfer",
        "",
        "This analysis fine-tunes each site with a target preprocessing while initializing from a reconstruction-pretrained encoder trained using the other preprocessing.",
        "",
        f"Complete rows: {len(complete)}",
        f"Missing rows: {len(missing)}",
        "",
        "## Mean Metrics",
        "",
        "\n".join(table_lines),
        "",
    ]
    (OUT_DIR / "writing_followup_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT_DIR / "preprocessing_cross_transfer_metrics.csv", index=False)
    plot_metric_grid(df, "balanced_acc", "Balanced accuracy", "preprocessing_cross_transfer_balanced_acc_curves")
    plot_metric_grid(df, "specificity", "Specificity", "preprocessing_cross_transfer_specificity_curves")
    plot_metric_grid(df, "f1", "F1 score", "preprocessing_cross_transfer_f1_curves")
    plot_heatmap(df, "balanced_acc", "Preprocessing-cross transfer: balanced accuracy", "preprocessing_cross_transfer_balanced_acc_heatmap")
    plot_heatmap(df, "specificity", "Preprocessing-cross transfer: specificity", "preprocessing_cross_transfer_specificity_heatmap")
    write_note(df)
    print(f"[DONE] wrote writing follow-up outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
