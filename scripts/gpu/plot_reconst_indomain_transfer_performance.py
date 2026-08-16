#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "figures" / "current_results_summary" / "metadata_v2_metrics_current.csv"
OUT_DIR = ROOT / "figures" / "current_results_summary" / "reconst_indomain_transfer"

SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
PREPROC_LABELS = {
    "raw": "Raw",
    "filter_rms": "Low-pass",
    "logenv": "Log-envelope",
}
PREPROC_DIRS = {
    "raw": "raw",
    "filter_rms": "filter",
    "logenv": "log_envelope",
}
COLORS = {
    "raw": "#6b7280",
    "filter_rms": "#bf4b3e",
    "logenv": "#376795",
}
MARKERS = {
    "raw": "^",
    "filter_rms": "o",
    "logenv": "s",
}
RAW_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1" / "raw_site_main_pre50_v1"
FRACTIONS = {
    "0p05": 0.05,
    "0p1": 0.10,
    "0p25": 0.25,
    "0p5": 0.50,
    "1": 1.00,
}


def save(fig: plt.Figure, name: str, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axis(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks([0.05, 0.10, 0.25, 0.50, 1.00])
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.grid(axis="y", color="#ded8cf", linewidth=0.85, alpha=0.82)
    ax.grid(axis="x", color="#eee7dd", linewidth=0.55, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.8, colors="#4b5563")
    ax.set_xlabel("Fine-tuning label fraction", fontsize=9.2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.6)


def load_reconst() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    sub = df[
        (df["study"].eq("site_main"))
        & (df["method"].eq("reconst"))
        & (df["preprocessing"].isin(PREPROC_LABELS))
        & (df["target_site"].isin(SITES))
    ].copy()
    raw_rows = []
    for site in SITES:
        test_root = RAW_ROOT / site / "reconst" / "test"
        if not test_root.exists():
            continue
        for metric_path in sorted(test_root.glob("*/test_metrics_fixed_threshold.json")):
            match = re.search(r"__frac([0-9p]+)$", metric_path.parent.name)
            if not match:
                continue
            tag = match.group(1)
            if tag not in FRACTIONS:
                continue
            try:
                data = json.loads(metric_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metrics = data.get("or_metrics_fixed_threshold") or data.get("fc_metrics_fixed_threshold")
            if not isinstance(metrics, dict):
                continue
            raw_rows.append(
                {
                    "study": "site_main",
                    "preprocessing": "raw",
                    "source_site": site,
                    "target_site": site,
                    "direction": f"{site}_in_domain",
                    "method": "reconst",
                    "fraction_tag": tag,
                    "fraction": FRACTIONS[tag],
                    "path": str(metric_path.relative_to(ROOT)),
                    "f1": float(metrics.get("f1", float("nan"))),
                    "balanced_acc": float(metrics.get("balanced_acc", float("nan"))),
                    "specificity": float(metrics.get("specificity", float("nan"))),
                    "recall": float(metrics.get("recall", float("nan"))),
                    "precision": float(metrics.get("precision", float("nan"))),
                    "acc": float(metrics.get("acc", float("nan"))),
                    "tp": int(metrics.get("tp", 0) or 0),
                    "tn": int(metrics.get("tn", 0) or 0),
                    "fp": int(metrics.get("fp", 0) or 0),
                    "fn": int(metrics.get("fn", 0) or 0),
                }
            )
    if raw_rows:
        sub = pd.concat([pd.DataFrame(raw_rows), sub], ignore_index=True)
    sub = sub.sort_values(["target_site", "preprocessing", "fraction"])
    return sub


def plot_metric_only(df: pd.DataFrame, metric: str, ylabel: str, name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), sharey=True)
    for ax, site in zip(axes, SITES):
        site_df = df[df["target_site"].eq(site)]
        for preproc in ["raw", "filter_rms", "logenv"]:
            p = site_df[site_df["preprocessing"].eq(preproc)].sort_values("fraction")
            if p.empty:
                continue
            ax.plot(
                p["fraction"],
                p[metric],
                marker=MARKERS[preproc],
                markersize=5.2,
                linewidth=2.15,
                color=COLORS[preproc],
                label=PREPROC_LABELS[preproc],
            )
        ax.set_title(SITE_LABELS[site], fontsize=10.8, fontweight="normal", pad=8)
        style_axis(ax, ylabel if ax is axes[0] else None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=3, frameon=False, fontsize=9.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.9)
    save(fig, name)


def plot_metric_single_preproc(df: pd.DataFrame, preproc: str, out_dir: Path, metric: str, ylabel: str, name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), sharey=True)
    for ax, site in zip(axes, SITES):
        p = df[
            df["target_site"].eq(site)
            & df["preprocessing"].eq(preproc)
        ].sort_values("fraction")
        if not p.empty:
            ax.plot(
                p["fraction"],
                p[metric],
                marker=MARKERS[preproc],
                markersize=5.3,
                linewidth=2.15,
                color=COLORS[preproc],
                label=PREPROC_LABELS[preproc],
            )
        ax.set_title(SITE_LABELS[site], fontsize=10.8, fontweight="normal", pad=8)
        style_axis(ax, ylabel if ax is axes[0] else None)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=1, frameon=False, fontsize=9.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.9)
    save(fig, name, out_dir)


def plot_metric_dashboard(df: pd.DataFrame) -> None:
    metrics = [
        ("balanced_acc", "Balanced accuracy"),
        ("specificity", "Specificity"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), sharex=True, sharey=True)
    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, site in enumerate(SITES):
            ax = axes[row_idx, col_idx]
            site_df = df[df["target_site"].eq(site)]
            for preproc in ["raw", "filter_rms", "logenv"]:
                p = site_df[site_df["preprocessing"].eq(preproc)].sort_values("fraction")
                if p.empty:
                    continue
                ax.plot(
                    p["fraction"],
                    p[metric],
                    marker=MARKERS[preproc],
                    markersize=4.8,
                    linewidth=2.0,
                    color=COLORS[preproc],
                    label=PREPROC_LABELS[preproc],
                )
            if row_idx == 0:
                ax.set_title(SITE_LABELS[site], fontsize=10.8, fontweight="normal", pad=8)
            style_axis(ax, ylabel if col_idx == 0 else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=3, frameon=False, fontsize=9.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.9, h_pad=1.0)
    save(fig, "reconst_indomain_balacc_specificity_dashboard")


def plot_metric_dashboard_single_preproc(df: pd.DataFrame, preproc: str, out_dir: Path) -> None:
    metrics = [
        ("balanced_acc", "Balanced accuracy"),
        ("specificity", "Specificity"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.7), sharex=True, sharey=True)
    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, site in enumerate(SITES):
            ax = axes[row_idx, col_idx]
            p = df[
                df["target_site"].eq(site)
                & df["preprocessing"].eq(preproc)
            ].sort_values("fraction")
            if not p.empty:
                ax.plot(
                    p["fraction"],
                    p[metric],
                    marker=MARKERS[preproc],
                    markersize=4.8,
                    linewidth=2.0,
                    color=COLORS[preproc],
                    label=PREPROC_LABELS[preproc],
                )
            if row_idx == 0:
                ax.set_title(SITE_LABELS[site], fontsize=10.8, fontweight="normal", pad=8)
            style_axis(ax, ylabel if col_idx == 0 else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=1, frameon=False, fontsize=9.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.9, h_pad=1.0)
    save(fig, "balanced_accuracy_specificity_dashboard", out_dir)


def write_preprocessing_subfolders(df: pd.DataFrame) -> None:
    for preproc, dirname in PREPROC_DIRS.items():
        out_dir = OUT_DIR / dirname
        sub = df[df["preprocessing"].eq(preproc)].copy()
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_dir / "metrics.csv", index=False)
        if sub.empty:
            continue
        plot_metric_single_preproc(df, preproc, out_dir, "balanced_acc", "Balanced accuracy", "balanced_accuracy")
        plot_metric_single_preproc(df, preproc, out_dir, "f1", "F1 score", "f1_score")
        plot_metric_dashboard_single_preproc(df, preproc, out_dir)


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
    df = load_reconst()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "reconst_indomain_transfer_metrics.csv", index=False)
    plot_metric_only(df, "balanced_acc", "Balanced accuracy", "reconst_indomain_balanced_accuracy")
    plot_metric_only(df, "f1", "F1 score", "reconst_indomain_f1_score")
    plot_metric_dashboard(df)
    write_preprocessing_subfolders(df)
    print(f"[DONE] wrote reconstruction in-domain transfer figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
