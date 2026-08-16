#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1" / "logenv_site_main_pre50_v2"
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures"

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
FRACTION_TAGS = {
    "0p05": 0.05,
    "0p1": 0.10,
    "0p25": 0.25,
    "0p5": 0.50,
    "1": 1.00,
}
SITE_DIRS = {
    "pohang": ("Pohang", "pohang"),
    "utah_2019": ("Utah 2019", "base_utah_2019"),
    "utah_2023": ("Utah 2023", "base_utah_2023"),
}
METHOD_ORDER = ["scratch", "contrast", "reconst", "reconst_noanom"]
METHOD_LABELS = {
    "scratch": "Scratch",
    "contrast": "Contrast",
    "reconst": "Reconst",
    "reconst_noanom": "Reconst no-anom",
}
METHOD_COLORS = {
    "scratch": "#2F4858",
    "contrast": "#7570B3",
    "reconst": "#D95F02",
    "reconst_noanom": "#1B9E77",
}
METHOD_MARKERS = {
    "scratch": "o",
    "contrast": "D",
    "reconst": "s",
    "reconst_noanom": "^",
}
METRIC_LABELS = {
    "balanced_acc": "Balanced accuracy",
    "f1": "F1-score",
    "recall": "Recall",
    "specificity": "Specificity",
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def best_branch_metrics(path: Path) -> dict[str, float | int | str]:
    data = json.loads(path.read_text())
    candidates = []
    branch_keys = [
        ("anomaly_metrics_fixed_threshold", "anomaly"),
        ("fc_metrics_fixed_threshold", "fc"),
        ("or_metrics_fixed_threshold", "or"),
        ("and_metrics_fixed_threshold", "and"),
    ]
    for key, branch in branch_keys:
        if key in data:
            metrics = data[key]
            candidates.append((float(metrics.get("f1", -1.0)), branch, metrics))
    if not candidates:
        raise ValueError(f"No fixed-threshold metric block found in {path}")
    _, branch, metrics = max(candidates, key=lambda item: item[0])
    return {
        "branch": branch,
        "f1": float(metrics["f1"]),
        "balanced_acc": float(metrics["balanced_acc"]),
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
        "tp": int(metrics["tp"]),
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
    }


def load_summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for site_key, (site_label, _) in SITE_DIRS.items():
        for summary_path in sorted(RUN_ROOT.glob(f"{site_key}_*/summary.csv")):
            df = pd.read_csv(summary_path)
            for _, row in df.iterrows():
                rows.append(
                    {
                        "site": site_key,
                        "site_label": site_label,
                        "method": str(row["method"]),
                        "fraction": float(row["fraction"]),
                        "balanced_acc": float(row["test_balanced_acc"]),
                        "f1": float(row["test_f1"]),
                        "recall": float(row["test_recall"]),
                        "specificity": float(row["test_specificity"]),
                        "branch": str(row["test_best_branch"]),
                        "tp": int(row["test_tp"]),
                        "tn": int(row["test_tn"]),
                        "fp": int(row["test_fp"]),
                        "fn": int(row["test_fn"]),
                        "status": "complete",
                    }
                )
    return rows


def load_partial_rows(existing: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for site_key, (site_label, exp_name) in SITE_DIRS.items():
        for method_group in ["scratch", "contrast", "reconst_reconst_noanom"]:
            group_dir = RUN_ROOT / f"{site_key}_{method_group}"
            if not group_dir.exists():
                continue
            for method_dir in group_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                test_dir = method_dir / "test"
                if not test_dir.exists():
                    continue
                for tag, fraction in FRACTION_TAGS.items():
                    if (
                        not existing.empty
                        and (
                            (existing["site"] == site_key)
                            & (existing["method"] == method)
                            & (existing["fraction"] == fraction)
                        ).any()
                    ):
                        continue
                    metric_path = test_dir / f"{exp_name}__frac{tag}" / "test_metrics_fixed_threshold.json"
                    if not metric_path.exists():
                        continue
                    metrics = best_branch_metrics(metric_path)
                    rows.append(
                        {
                            "site": site_key,
                            "site_label": site_label,
                            "method": method,
                            "fraction": fraction,
                            "status": "partial",
                            **metrics,
                        }
                    )
    return rows


def load_results() -> pd.DataFrame:
    rows = load_summary_rows()
    complete_df = pd.DataFrame(rows)
    rows.extend(load_partial_rows(complete_df))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No results found under {RUN_ROOT}")
    return df.sort_values(["site", "method", "fraction"]).reset_index(drop=True)


def clean_axis(ax: plt.Axes, metric: str) -> None:
    ax.set_ylim(-0.03, 1.04)
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.set_xlabel("Labeled fraction")
    ax.set_ylabel(METRIC_LABELS[metric])


def plot_grid(df: pd.DataFrame, metrics: list[str], out_name: str, title: str) -> None:
    sites = ["pohang", "utah_2019", "utah_2023"]
    fig, axes = plt.subplots(
        len(metrics),
        len(sites),
        figsize=(12.4, 3.3 * len(metrics)),
        sharex=True,
        sharey=True,
    )
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)

    for col, site in enumerate(sites):
        site_df = df[df["site"] == site]
        status = "partial" if (site_df["status"] == "partial").any() else "complete"
        site_title = SITE_DIRS[site][0]
        if status == "partial":
            site_title = f"{site_title} (partial)"
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            for method in METHOD_ORDER:
                method_df = site_df[site_df["method"] == method].sort_values("fraction")
                if method_df.empty:
                    continue
                linestyle = "--" if (method_df["status"] == "partial").any() else "-"
                ax.plot(
                    method_df["fraction"],
                    method_df[metric],
                    label=METHOD_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method, "#333333"),
                    marker=METHOD_MARKERS.get(method, "o"),
                    markersize=5.0,
                    linewidth=2.0,
                    linestyle=linestyle,
                )
            clean_axis(ax, metric)
            if row == 0:
                ax.set_title(site_title, pad=9)
            if col != 0:
                ax.set_ylabel("")
            if row != len(metrics) - 1:
                ax.set_xlabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0.055, 1, 0.955])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"{out_name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_matplotlib()
    df = load_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "logenv_current_results_table.csv", index=False)
    plot_grid(
        df,
        ["balanced_acc", "f1"],
        "logenv_label_efficiency_current",
        "Log-Envelope Label Efficiency Results",
    )
    plot_grid(
        df,
        ["recall", "specificity"],
        "logenv_recall_specificity_current",
        "Log-Envelope Detection Trade-off",
    )
    print(f"[DONE] wrote figures and table to {OUT_DIR}")


if __name__ == "__main__":
    main()
