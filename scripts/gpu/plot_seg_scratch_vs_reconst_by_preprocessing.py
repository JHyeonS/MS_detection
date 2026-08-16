#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Pretendard", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1"
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "seg"

SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Site A / Pohang",
    "utah_2019": "Site B / Utah 2019",
    "utah_2023": "Site C / Utah 2023",
}
METHODS = ["scratch", "reconst"]
METHOD_LABELS = {
    "scratch": "Scratch",
    "reconst": "Reconst transfer",
}
METHOD_COLORS = {
    "scratch": "#41464b",
    "reconst": "#c4513e",
}
METHOD_MARKERS = {
    "scratch": "o",
    "reconst": "s",
}
PREPROCESSING_COLORS = {
    "raw": "#4b5563",
    "filter_rms": "#c4513e",
    "logenv": "#2f7f75",
}
PREPROCESSING_MARKERS = {
    "raw": "o",
    "filter_rms": "s",
    "logenv": "^",
}
PREPROCESSING = {
    "raw": {
        "label": "Raw",
        "root": RUN_ROOT / "raw_site_main_pre50_v1",
        "layout": "plain_site",
    },
    "filter_rms": {
        "label": "Low-pass filter + RMS normalization",
        "root": RUN_ROOT / "filter_rms_site_main_pre50_v2",
        "layout": "grouped",
    },
    "logenv": {
        "label": "Low-pass filter + log-envelope scaling + RMS normalization",
        "root": RUN_ROOT / "logenv_site_main_pre50_v2",
        "layout": "grouped",
    },
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


def read_metrics(path: Path) -> dict[str, float] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = data.get("fc_metrics_fixed_threshold")
    if not isinstance(metrics, dict):
        return None
    return {
        "balanced_acc": float(metrics.get("balanced_acc", float("nan"))),
        "f1": float(metrics.get("f1", float("nan"))),
        "specificity": float(metrics.get("specificity", float("nan"))),
        "recall": float(metrics.get("recall", float("nan"))),
    }


def collect_plain_site(preprocessing: str, root: Path) -> list[dict]:
    rows: list[dict] = []
    for site in SITES:
        for method in METHODS:
            for metric_path in sorted((root / site / method / "test").glob("*/test_metrics_fixed_threshold.json")):
                fraction = parse_fraction(metric_path.parent.name)
                metrics = read_metrics(metric_path)
                if fraction is None or metrics is None:
                    continue
                row = {
                    "preprocessing": preprocessing,
                    "site": site,
                    "method": method,
                    "fraction": fraction,
                    "path": str(metric_path.relative_to(ROOT)),
                }
                row.update(metrics)
                rows.append(row)
    return rows


def collect_grouped(preprocessing: str, root: Path) -> list[dict]:
    rows: list[dict] = []
    for site in SITES:
        for group_dir in sorted(root.glob(f"{site}_*")):
            for method in METHODS:
                test_root = group_dir / method / "test"
                if not test_root.exists():
                    continue
                for metric_path in sorted(test_root.glob("*/test_metrics_fixed_threshold.json")):
                    fraction = parse_fraction(metric_path.parent.name)
                    metrics = read_metrics(metric_path)
                    if fraction is None or metrics is None:
                        continue
                    row = {
                        "preprocessing": preprocessing,
                        "site": site,
                        "method": method,
                        "fraction": fraction,
                        "path": str(metric_path.relative_to(ROOT)),
                    }
                    row.update(metrics)
                    rows.append(row)
    return rows


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for preprocessing, spec in PREPROCESSING.items():
        if spec["layout"] == "plain_site":
            rows.extend(collect_plain_site(preprocessing, spec["root"]))
        else:
            rows.extend(collect_grouped(preprocessing, spec["root"]))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(["preprocessing", "site", "method", "fraction"])
    return df.sort_values(["preprocessing", "site", "method", "fraction"])


def style_axis(ax: plt.Axes, ylabel: str | None) -> None:
    ax.set_ylim(0.40, 1.03)
    ax.set_xticks(FRACTION_ORDER)
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.grid(axis="y", color="#ded8ce", linewidth=0.8, alpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Fine-tuning label fraction")


def plot_preprocessing(df: pd.DataFrame, preprocessing: str) -> None:
    spec = PREPROCESSING[preprocessing]
    sub = df[df["preprocessing"] == preprocessing]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.85), sharey=True)
    for ax, site in zip(axes, SITES):
        site_df = sub[sub["site"] == site]
        for method in METHODS:
            method_df = site_df[site_df["method"] == method].sort_values("fraction")
            if method_df.empty:
                continue
            ax.plot(
                method_df["fraction"],
                method_df["balanced_acc"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=2.4,
                markersize=5.8,
                label=METHOD_LABELS[method],
            )
        ax.set_title(SITE_LABELS[site], fontsize=11.5, pad=9)
        style_axis(ax, "Balanced accuracy" if ax is axes[0] else None)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        fontsize=10.5,
    )
    fig.suptitle(
        f"{spec['label']}: scratch vs reconstruction-based transfer",
        fontsize=14,
        fontweight="bold",
        y=1.035,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / f"seg_{preprocessing}_scratch_vs_reconst_balanced_acc"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")


def plot_preprocessing_f1_with_balacc_context(df: pd.DataFrame, preprocessing: str) -> None:
    spec = PREPROCESSING[preprocessing]
    sub = df[df["preprocessing"] == preprocessing]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.95), sharey=True)
    for ax, site in zip(axes, SITES):
        site_df = sub[sub["site"] == site]
        for method in METHODS:
            method_df = site_df[site_df["method"] == method].sort_values("fraction")
            if method_df.empty:
                continue
            ax.plot(
                method_df["fraction"],
                method_df["f1"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=2.5,
                markersize=5.8,
                label=f"{METHOD_LABELS[method]} F1",
            )
            ax.plot(
                method_df["fraction"],
                method_df["balanced_acc"],
                color=METHOD_COLORS[method],
                linestyle="--",
                linewidth=1.45,
                alpha=0.42,
                label=f"{METHOD_LABELS[method]} bal. acc.",
            )
        ax.set_title(SITE_LABELS[site], fontsize=11.5, pad=9)
        style_axis(ax, "F1 score / balanced accuracy" if ax is axes[0] else None)

        # Mark cases where F1 remains high while balanced accuracy is notably lower.
        warn_df = site_df[
            (site_df["method"] == "reconst")
            & (site_df["f1"] >= 0.70)
            & ((site_df["f1"] - site_df["balanced_acc"]) >= 0.12)
        ].sort_values("fraction")
        if not warn_df.empty:
            ax.scatter(
                warn_df["fraction"],
                warn_df["f1"],
                marker="x",
                s=70,
                linewidths=1.7,
                color="#7f1d1d",
                zorder=5,
                label="F1-bal. acc. gap",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9.2,
    )
    fig.suptitle(
        f"{spec['label']}: F1-focused scratch vs reconstruction transfer",
        fontsize=14,
        fontweight="bold",
        y=1.035,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / f"seg_{preprocessing}_scratch_vs_reconst_f1_with_balacc_context"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")


def plot_delta_f1_heatmap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.85), sharey=True)
    vmax = 0.45
    vmin = -0.45
    heatmaps = []
    for ax, (preprocessing, spec) in zip(axes, PREPROCESSING.items()):
        sub = df[df["preprocessing"] == preprocessing]
        matrix = np.full((len(SITES), len(FRACTION_ORDER)), np.nan)
        for r, site in enumerate(SITES):
            for c, fraction in enumerate(FRACTION_ORDER):
                point = sub[(sub["site"] == site) & (sub["fraction"] == fraction)]
                scratch = point[point["method"] == "scratch"]["f1"]
                reconst = point[point["method"] == "reconst"]["f1"]
                if not scratch.empty and not reconst.empty:
                    matrix[r, c] = float(reconst.iloc[0] - scratch.iloc[0])
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        heatmaps.append(im)
        ax.set_title(spec["label"], fontsize=11.3, pad=10)
        ax.set_xticks(range(len(FRACTION_ORDER)))
        ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"], rotation=0)
        ax.set_xlabel("Fine-tuning label fraction")
        if ax is axes[0]:
            ax.set_yticks(range(len(SITES)))
            ax.set_yticklabels([SITE_LABELS[s] for s in SITES])
        else:
            ax.set_yticks(range(len(SITES)))
            ax.tick_params(axis="y", length=0)

        for r in range(len(SITES)):
            for c in range(len(FRACTION_ORDER)):
                value = matrix[r, c]
                if np.isnan(value):
                    label = "NA"
                    color = "#1f2933"
                else:
                    label = f"{value:+.2f}"
                    color = "white" if abs(value) > 0.22 else "#1f2933"
                ax.text(c, r, label, ha="center", va="center", fontsize=8.7, color=color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    cbar = fig.colorbar(heatmaps[-1], ax=axes, fraction=0.025, pad=0.025)
    cbar.set_label(r"$\Delta$F1 = Reconst transfer - Scratch", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        "Label-efficiency gain from reconstruction-based transfer",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.025,
        "Positive values indicate that reconstruction-based transfer outperforms training from scratch.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#374151",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / "seg_delta_f1_reconst_minus_scratch_heatmap"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")


def plot_delta_f1_heatmap_compact(df: pd.DataFrame, include_full_fraction: bool = False) -> None:
    compact_fractions = [0.05, 0.10, 0.25, 0.50, 1.00] if include_full_fraction else [0.05, 0.10, 0.25, 0.50]
    fig_width = 11.8 if include_full_fraction else 10.8
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 3.25), sharey=True)
    vmax = 0.35
    vmin = -0.35
    im = None
    short_titles = {
        "raw": "Raw",
        "filter_rms": "Low-pass + RMS",
        "logenv": "Log-envelope",
    }
    for ax, preprocessing in zip(axes, PREPROCESSING):
        sub = df[df["preprocessing"] == preprocessing]
        matrix = np.full((len(SITES), len(compact_fractions)), np.nan)
        for r, site in enumerate(SITES):
            for c, fraction in enumerate(compact_fractions):
                point = sub[(sub["site"] == site) & (sub["fraction"] == fraction)]
                scratch = point[point["method"] == "scratch"]["f1"]
                reconst = point[point["method"] == "reconst"]["f1"]
                if not scratch.empty and not reconst.empty:
                    matrix[r, c] = float(reconst.iloc[0] - scratch.iloc[0])
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(short_titles[preprocessing], fontsize=12, fontweight="normal", pad=9)
        ax.set_xticks(range(len(compact_fractions)))
        xtick_labels = ["5%", "10%", "25%", "50%", "100%"] if include_full_fraction else ["5%", "10%", "25%", "50%"]
        ax.set_xticklabels(xtick_labels, fontsize=9.5)
        ax.set_xlabel("Label fraction", fontsize=10)
        if ax is axes[0]:
            ax.set_yticks(range(len(SITES)))
            ax.set_yticklabels(["Pohang", "Utah 2019", "Utah 2023"], fontsize=10)
        else:
            ax.set_yticks(range(len(SITES)))
            ax.tick_params(axis="y", length=0, labelleft=False)

        for r in range(len(SITES)):
            for c in range(len(compact_fractions)):
                value = matrix[r, c]
                if np.isnan(value):
                    continue
                if abs(value) >= 0.08:
                    label = f"{value:+.2f}"
                    color = "white" if abs(value) > 0.20 else "#111827"
                    ax.text(c, r, label, ha="center", va="center", fontsize=9.2, color=color)

        ax.set_xticks(np.arange(-0.5, len(compact_fractions), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SITES), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    assert im is not None
    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.024)
    cbar.set_label(r"$\Delta$F1", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.text(
        0.965,
        -0.035,
        "Warm: reconst transfer improves F1\nCool: scratch is better",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#b0b7c3",
        linespacing=1.15,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "compact_full_frac" if include_full_fraction else "compact_frac_le_0p5"
    out_base = OUT_DIR / f"seg_delta_f1_reconst_minus_scratch_heatmap_{suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")


def plot_scratch_f1_heatmap(df: pd.DataFrame, include_full_fraction: bool = False) -> None:
    compact_fractions = [0.05, 0.10, 0.25, 0.50, 1.00] if include_full_fraction else [0.05, 0.10, 0.25, 0.50]
    fig_width = 11.5 if include_full_fraction else 10.8
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 3.25), sharey=True)
    im = None
    short_titles = {
        "raw": "Raw",
        "filter_rms": "Low-pass + RMS",
        "logenv": "Log-envelope",
    }
    for ax, preprocessing in zip(axes, PREPROCESSING):
        sub = df[(df["preprocessing"] == preprocessing) & (df["method"] == "scratch")]
        matrix = np.full((len(SITES), len(compact_fractions)), np.nan)
        for r, site in enumerate(SITES):
            for c, fraction in enumerate(compact_fractions):
                point = sub[(sub["site"] == site) & (sub["fraction"] == fraction)]
                if not point.empty:
                    matrix[r, c] = float(point["f1"].iloc[0])
        im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.45, vmax=1.0, aspect="auto")
        ax.set_title(short_titles[preprocessing], fontsize=12, fontweight="normal", pad=9)
        ax.set_xticks(range(len(compact_fractions)))
        labels = ["5%", "10%", "25%", "50%", "100%"] if include_full_fraction else ["5%", "10%", "25%", "50%"]
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_xlabel("Label fraction", fontsize=10)
        if ax is axes[0]:
            ax.set_yticks(range(len(SITES)))
            ax.set_yticklabels(["Pohang", "Utah 2019", "Utah 2023"], fontsize=10)
        else:
            ax.set_yticks(range(len(SITES)))
            ax.tick_params(axis="y", length=0, labelleft=False)

        for r in range(len(SITES)):
            for c in range(len(compact_fractions)):
                value = matrix[r, c]
                if np.isnan(value):
                    continue
                color = "white" if value >= 0.82 else "#111827"
                ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=9.2, color=color)

        ax.set_xticks(np.arange(-0.5, len(compact_fractions), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SITES), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    assert im is not None
    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.024)
    cbar.set_label("Scratch F1", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.text(
        0.965,
        -0.035,
        "Higher values indicate stronger\nscratch baseline performance",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#b0b7c3",
        linespacing=1.15,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "full_frac" if include_full_fraction else "compact_frac_le_0p5"
    out_base = OUT_DIR / f"seg_scratch_f1_heatmap_{suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")


def plot_scratch_f1_lineplot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.65), sharey=True)
    for ax, site in zip(axes, SITES):
        site_df = df[(df["site"] == site) & (df["method"] == "scratch")]
        for preprocessing in PREPROCESSING:
            sub = site_df[site_df["preprocessing"] == preprocessing].sort_values("fraction")
            if sub.empty:
                continue
            label = {
                "raw": "Raw",
                "filter_rms": "Low-pass + RMS",
                "logenv": "Log-envelope",
            }[preprocessing]
            ax.plot(
                sub["fraction"],
                sub["f1"],
                color=PREPROCESSING_COLORS[preprocessing],
                marker=PREPROCESSING_MARKERS[preprocessing],
                linewidth=2.35,
                markersize=5.8,
                label=label,
            )
        ax.set_title(SITE_LABELS[site], fontsize=11.5, fontweight="normal", pad=9)
        ax.set_ylim(0.15, 1.03)
        ax.set_xticks(FRACTION_ORDER)
        ax.set_xticklabels(["5%", "10%", "25%", "50%", "100%"], fontsize=9.5)
        ax.set_xlabel("Label fraction", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Scratch F1", fontsize=10.5)
        ax.grid(axis="y", color="#ded8ce", linewidth=0.8, alpha=0.95)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
        fontsize=9.8,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1.0])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / "seg_scratch_f1_lineplot_by_preprocessing"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")


def main() -> None:
    df = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "seg_scratch_vs_reconst_by_preprocessing_metrics.csv"
    df.to_csv(csv_path, index=False)
    for preprocessing in PREPROCESSING:
        plot_preprocessing(df, preprocessing)
        plot_preprocessing_f1_with_balacc_context(df, preprocessing)
    plot_scratch_f1_lineplot(df)
    plot_scratch_f1_heatmap(df, include_full_fraction=False)
    plot_scratch_f1_heatmap(df, include_full_fraction=True)
    plot_delta_f1_heatmap(df)
    plot_delta_f1_heatmap_compact(df)
    plot_delta_f1_heatmap_compact(df, include_full_fraction=True)
    print(f"[DONE] wrote {csv_path}")


if __name__ == "__main__":
    main()
