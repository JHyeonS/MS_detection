#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1"
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2"

SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Site A / Pohang",
    "utah_2019": "Site B / Utah 2019",
    "utah_2023": "Site C / Utah 2023",
}
SITE_EXP = {
    "pohang": "pohang",
    "utah_2019": "base_utah_2019",
    "utah_2023": "base_utah_2023",
}
METHODS = ["scratch", "reconst"]
METHOD_LABELS = {"scratch": "Scratch", "reconst": "Reconst"}
PREPROCESS_LABELS = {
    "raw": "Raw",
    "filter_rms": "Filter + RMS",
    "logenv": "Filter + log-env + RMS",
}
PREPROCESS_COLORS = {
    "raw": "#4b5563",
    "filter_rms": "#c44e52",
    "logenv": "#2f7f75",
}
FRACTIONS = {
    "0p05": 0.05,
    "0p1": 0.10,
    "0p25": 0.25,
    "0p5": 0.50,
    "1": 1.00,
}
FRACTION_ORDER = [0.05, 0.10, 0.25, 0.50, 1.00]


def parse_fraction(name: str) -> float | None:
    match = re.search(r"__frac([0-9p]+)$", name)
    if not match:
        return None
    return FRACTIONS.get(match.group(1))


def read_fc_metrics(path: Path) -> dict[str, float] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = data.get("fc_metrics_fixed_threshold")
    if not isinstance(metrics, dict):
        return None
    return {
        "f1": float(metrics.get("f1", float("nan"))),
        "balanced_acc": float(metrics.get("balanced_acc", float("nan"))),
        "specificity": float(metrics.get("specificity", float("nan"))),
        "recall": float(metrics.get("recall", float("nan"))),
        "precision": float(metrics.get("precision", float("nan"))),
        "acc": float(metrics.get("acc", float("nan"))),
    }


def collect_raw() -> list[dict]:
    root = RUN_ROOT / "raw_site_main_pre50_v1"
    rows: list[dict] = []
    for site in SITES:
        for method in METHODS:
            test_root = root / site / method / "test"
            for metric_path in sorted(test_root.glob("*/test_metrics_fixed_threshold.json")):
                fraction = parse_fraction(metric_path.parent.name)
                metrics = read_fc_metrics(metric_path)
                if fraction is None or metrics is None:
                    continue
                row = {
                    "site": site,
                    "method": method,
                    "preprocessing": "raw",
                    "fraction": fraction,
                    "path": str(metric_path.relative_to(ROOT)),
                }
                row.update(metrics)
                rows.append(row)
    return rows


def collect_filtered(preprocessing: str, root_name: str) -> list[dict]:
    root = RUN_ROOT / root_name
    rows: list[dict] = []
    for site in SITES:
        for method in METHODS:
            for group_dir in sorted(root.glob(f"{site}_*")):
                test_root = group_dir / method / "test"
                if not test_root.exists():
                    continue
                for metric_path in sorted(test_root.glob("*/test_metrics_fixed_threshold.json")):
                    fraction = parse_fraction(metric_path.parent.name)
                    metrics = read_fc_metrics(metric_path)
                    if fraction is None or metrics is None:
                        continue
                    row = {
                        "site": site,
                        "method": method,
                        "preprocessing": preprocessing,
                        "fraction": fraction,
                        "path": str(metric_path.relative_to(ROOT)),
                    }
                    row.update(metrics)
                    rows.append(row)
    return rows


def collect() -> pd.DataFrame:
    rows = []
    rows.extend(collect_raw())
    rows.extend(collect_filtered("filter_rms", "filter_rms_site_main_pre50_v2"))
    rows.extend(collect_filtered("logenv", "logenv_site_main_pre50_v2"))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(["site", "method", "preprocessing", "fraction"])
    return df.sort_values(["site", "method", "preprocessing", "fraction"])


def style_axis(ax: plt.Axes, ylabel: str | None, ylim: tuple[float, float]) -> None:
    ax.set_ylim(*ylim)
    ax.set_xticks(FRACTION_ORDER)
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.grid(axis="y", color="#d8d2c8", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    suffix: str = "",
    ylim: tuple[float, float] = (0.0, 1.03),
) -> None:
    fig, axes = plt.subplots(
        nrows=len(SITES),
        ncols=len(METHODS),
        figsize=(10.8, 9.0),
        sharex=True,
        sharey=True,
    )
    for row_idx, site in enumerate(SITES):
        for col_idx, method in enumerate(METHODS):
            ax = axes[row_idx, col_idx]
            sub = df[(df["site"] == site) & (df["method"] == method)]
            for preprocessing in ["raw", "filter_rms", "logenv"]:
                pre_df = sub[sub["preprocessing"] == preprocessing].sort_values("fraction")
                if pre_df.empty:
                    continue
                ax.plot(
                    pre_df["fraction"],
                    pre_df[metric],
                    marker="o",
                    linewidth=2.1,
                    markersize=5.2,
                    color=PREPROCESS_COLORS[preprocessing],
                    label=PREPROCESS_LABELS[preprocessing],
                )
            if row_idx == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=12, pad=10)
            if col_idx == 0:
                ax.text(
                    -0.25,
                    0.5,
                    SITE_LABELS[site],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )
            if row_idx == len(SITES) - 1:
                ax.set_xlabel("Fine-tuning label fraction")
            style_axis(ax, metric_label if col_idx == 0 else None, ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"In-domain performance before and after filtering ({metric_label})",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.04, 0.05, 1.0, 0.96])
    save_figure(fig, f"raw_vs_filtered_indomain_{metric}{suffix}")


def main() -> None:
    df = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "raw_vs_filtered_indomain_metrics.csv"
    df.to_csv(csv_path, index=False)
    plot_metric(df, "balanced_acc", "Balanced accuracy")
    plot_metric(df, "f1", "F1")
    plot_metric(df, "balanced_acc", "Balanced accuracy", suffix="_zoom", ylim=(0.45, 1.02))
    plot_metric(df, "f1", "F1", suffix="_zoom", ylim=(0.45, 1.02))
    print(f"[DONE] wrote {csv_path}")
    print(f"[DONE] wrote {OUT_DIR / 'raw_vs_filtered_indomain_balanced_acc.pdf'}")
    print(f"[DONE] wrote {OUT_DIR / 'raw_vs_filtered_indomain_f1.pdf'}")
    print(f"[DONE] wrote {OUT_DIR / 'raw_vs_filtered_indomain_balanced_acc_zoom.pdf'}")
    print(f"[DONE] wrote {OUT_DIR / 'raw_vs_filtered_indomain_f1_zoom.pdf'}")


if __name__ == "__main__":
    main()
