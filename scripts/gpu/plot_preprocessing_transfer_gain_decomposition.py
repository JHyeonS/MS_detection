#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1" / "preprocessing_cross_reconst_pre50_v1"
METRICS_CSV = ROOT / "figures" / "current_results_summary" / "pretrain_setting_transfer" / "pretrain_setting_transfer_metrics.csv"
OUT_DIR = ROOT / "figures" / "current_results_summary" / "preprocessing_transfer_summary"

SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
PREPROCESSING = ["raw", "filter_rms", "logenv"]
PREPROC_LABELS = {
    "raw": "Raw",
    "filter_rms": "Low-pass",
    "logenv": "Log-envelope",
}
FRACTIONS = {
    "0p05": 0.05,
    "0p1": 0.10,
    "0p25": 0.25,
    "0p5": 0.50,
    "1": 1.00,
}


def read_metric(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data.get("fc_metrics_fixed_threshold") or data.get("or_metrics_fixed_threshold")


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def collect_preprocessing_cross() -> pd.DataFrame:
    rows = []
    for metric_path in sorted(RUN_ROOT.glob("*/*/reconst/test/*/test_metrics_fixed_threshold.json")):
        parts = metric_path.parts
        site = parts[parts.index("preprocessing_cross_reconst_pre50_v1") + 1]
        direction = parts[parts.index(site) + 1]
        if "_to_" not in direction:
            continue
        source_preproc, target_preproc = direction.split("_to_", 1)
        match = re.search(r"__frac([0-9p]+)$", metric_path.parent.name)
        if not match or match.group(1) not in FRACTIONS:
            continue
        metrics = read_metric(metric_path)
        if metrics is None:
            continue
        rows.append(
            {
                "gain_type": "cross_preprocessing",
                "site": site,
                "source_preprocessing": source_preproc,
                "target_preprocessing": target_preproc,
                "direction": direction,
                "fraction_tag": match.group(1),
                "fraction": FRACTIONS[match.group(1)],
                "balanced_acc": float(metrics.get("balanced_acc", np.nan)),
                "f1": float(metrics.get("f1", np.nan)),
                "specificity": float(metrics.get("specificity", np.nan)),
                "path": str(metric_path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def build_gain_table(metric: str = "balanced_acc") -> pd.DataFrame:
    cross = collect_preprocessing_cross()
    site_main = pd.read_csv(METRICS_CSV)
    scratch = (
        site_main[site_main["setting"].eq("no_pretrain")]
        [["target_site", "preprocessing", "fraction", metric]]
        .rename(
            columns={
                "target_site": "site",
                "preprocessing": "target_preprocessing",
                metric: f"scratch_target_{metric}",
            }
        )
    )
    cross_gain = cross.merge(
        scratch,
        on=["site", "target_preprocessing", "fraction"],
        how="left",
        validate="many_to_one",
    )
    cross_gain[f"transfer_gain_{metric}"] = cross_gain[metric] - cross_gain[f"scratch_target_{metric}"]

    indomain = site_main[site_main["setting"].eq("reconst_indomain")].copy()
    indomain = indomain.merge(
        scratch,
        left_on=["target_site", "preprocessing", "fraction"],
        right_on=["site", "target_preprocessing", "fraction"],
        how="left",
        validate="many_to_one",
    )
    diag = pd.DataFrame(
        {
            "gain_type": "in_domain",
            "site": indomain["target_site"],
            "source_preprocessing": indomain["preprocessing"],
            "target_preprocessing": indomain["preprocessing"],
            "direction": indomain["preprocessing"] + "_to_" + indomain["preprocessing"],
            "fraction_tag": indomain["fraction_tag"],
            "fraction": indomain["fraction"],
            metric: indomain[metric],
            "f1": indomain["f1"],
            "specificity": indomain["specificity"],
            f"scratch_target_{metric}": indomain[f"scratch_target_{metric}"],
            f"transfer_gain_{metric}": indomain[metric] - indomain[f"scratch_target_{metric}"],
            "path": indomain["path"],
        }
    )
    combined = pd.concat([diag, cross_gain], ignore_index=True, sort=False)
    return combined.sort_values(["site", "source_preprocessing", "target_preprocessing", "fraction"])


def matrix_for(df: pd.DataFrame, site: str, fraction: float, metric: str) -> np.ndarray:
    key = f"transfer_gain_{metric}"
    matrix = np.full((len(PREPROCESSING), len(PREPROCESSING)), np.nan, dtype=float)
    sub = df[df["site"].eq(site) & np.isclose(df["fraction"], fraction)]
    for _, row in sub.iterrows():
        if row["source_preprocessing"] not in PREPROCESSING or row["target_preprocessing"] not in PREPROCESSING:
            continue
        i = PREPROCESSING.index(row["source_preprocessing"])
        j = PREPROCESSING.index(row["target_preprocessing"])
        matrix[i, j] = float(row[key])
    return matrix


def plot_gain_heatmap(df: pd.DataFrame, fraction: float = 0.25, metric: str = "balanced_acc") -> None:
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#f2eee8")
    vmax = 0.55
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.85), sharex=True, sharey=True)
    images = []
    labels = [PREPROC_LABELS[p] for p in PREPROCESSING]
    for ax, site in zip(axes, SITES):
        matrix = matrix_for(df, site, fraction, metric)
        image = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap=cmap)
        images.append(image)
        ax.set_title(SITE_LABELS[site], fontsize=11.0, fontweight="normal", pad=8)
        ax.set_xticks(range(len(PREPROCESSING)))
        ax.set_yticks(range(len(PREPROCESSING)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.2)
        ax.set_yticklabels(labels, fontsize=8.2)
        ax.set_xlabel("Fine-tune / target preprocessing", fontsize=8.7)
        if ax is axes[0]:
            ax.set_ylabel("Pretrain / source preprocessing", fontsize=8.7)
        for i in range(len(PREPROCESSING)):
            for j in range(len(PREPROCESSING)):
                value = matrix[i, j]
                if np.isnan(value):
                    ax.text(j, i, "-", ha="center", va="center", fontsize=9.5, color="#8a8178")
                    continue
                color = "white" if abs(value) >= 0.30 else "#111827"
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9.0, color=color)
        ax.set_xticks(np.arange(-0.5, len(PREPROCESSING), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(PREPROCESSING), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.20, top=0.78, wspace=0.32)
    cax = fig.add_axes([0.925, 0.25, 0.014, 0.50])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label("Transfer gain in balanced accuracy", fontsize=9.0)
    cbar.ax.tick_params(labelsize=8.0)
    fig.suptitle(
        f"Preprocessing-domain transfer gain relative to target-preprocessing scratch at label fraction {fraction:g}",
        fontsize=12.2,
        y=1.01,
    )
    save(fig, f"representative_preprocessing_transfer_gain_decomposition_frac{str(fraction).replace('.', 'p')}_balanced_accuracy")


def plot_appendix_curves(df: pd.DataFrame, metric: str = "balanced_acc") -> None:
    key = f"transfer_gain_{metric}"
    directions = [
        "raw_to_filter_rms",
        "raw_to_logenv",
        "filter_rms_to_raw",
        "filter_rms_to_logenv",
        "logenv_to_raw",
        "logenv_to_filter_rms",
    ]
    direction_labels = [
        "Raw -> Low-pass",
        "Raw -> Log-envelope",
        "Low-pass -> Raw",
        "Low-pass -> Log-envelope",
        "Log-envelope -> Raw",
        "Log-envelope -> Low-pass",
    ]
    colors = {"pohang": "#365f91", "utah_2019": "#c45746", "utah_2023": "#2f8f6b"}
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.3), sharex=True, sharey=True)
    for ax, direction, label in zip(axes.ravel(), directions, direction_labels):
        for site in SITES:
            sub = df[df["site"].eq(site) & df["direction"].eq(direction)].sort_values("fraction")
            if sub.empty:
                continue
            ax.plot(
                sub["fraction"],
                sub[key],
                marker="o",
                linewidth=1.8,
                markersize=4.2,
                color=colors[site],
                label=SITE_LABELS[site],
            )
        ax.axhline(0, color="#111827", linewidth=0.85, alpha=0.65)
        ax.set_title(label, fontsize=9.2, fontweight="normal", pad=8)
        ax.set_ylim(-0.65, 0.65)
        ax.set_xticks([0.05, 0.10, 0.25, 0.50, 1.00])
        ax.set_xticklabels(["5", "10", "25", "50", "100"])
        ax.set_xlabel("Label fraction (%)", fontsize=8.0)
        ax.grid(axis="y", color="#ded8cf", linewidth=0.78, alpha=0.78)
        ax.grid(axis="x", color="#eee7dd", linewidth=0.45, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Transfer gain", fontsize=8.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8.8)
    fig.suptitle("Appendix: preprocessing-domain transfer gain curves", fontsize=12.2, y=1.01)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97), w_pad=0.9, h_pad=0.95)
    save(fig, "appendix_preprocessing_transfer_gain_curves_balanced_accuracy")


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_gain_table(metric="balanced_acc")
    df.to_csv(OUT_DIR / "preprocessing_transfer_gain_decomposition_metrics.csv", index=False)
    plot_gain_heatmap(df, fraction=0.25, metric="balanced_acc")
    plot_appendix_curves(df, metric="balanced_acc")
    print(f"[DONE] wrote preprocessing transfer figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
