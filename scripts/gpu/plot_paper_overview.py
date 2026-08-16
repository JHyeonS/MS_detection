#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("runs/paper_overview_figures_v1")

COLORS = {
    "scratch": "#2F4858",
    "reconst": "#D95F02",
    "contrast": "#7570B3",
    "reconst_noanom": "#1B9E77",
    "baseline": "#2F4858",
    "bandpass_agc_none": "#D95F02",
    "bandpass_agc_robust": "#1B9E77",
}

MARKERS = {
    "scratch": "o",
    "reconst": "s",
    "contrast": "D",
    "reconst_noanom": "^",
    "baseline": "o",
    "bandpass_agc_none": "s",
    "bandpass_agc_robust": "^",
}

LABELS = {
    "scratch": "Scratch",
    "reconst": "Reconstruction pretrain",
    "contrast": "Contrastive pretrain",
    "reconst_noanom": "Recon. pretrain, no anomaly",
    "baseline": "Baseline",
    "bandpass_agc_none": "Bandpass + AGC",
    "bandpass_agc_robust": "Bandpass + AGC + robust",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_fraction(x: float) -> str:
    if pd.isna(x):
        return ""
    if abs(float(x) - 1.0) < 1e-9:
        return "1.0"
    return f"{float(x):.2f}".rstrip("0").rstrip(".")


def plot_site_main_studies():
    pohang = pd.read_csv("runs/pohang_main_study/summary.csv").copy()
    utah19 = pd.read_csv("runs/utah_2019_preprocess_study/bandpass_agc/summary.csv").copy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    site_specs = [
        (pohang, "Pohang", 0, ["scratch", "reconst", "contrast", "reconst_noanom"]),
        (utah19, "Utah 2019", 1, ["reconst"]),
    ]
    for df, site, row, method_candidates in site_specs:
        methods = [m for m in method_candidates if m in set(df["method"])]
        for method in methods:
            sub = df[df["method"] == method].sort_values("fraction")
            axes[row, 0].plot(
                sub["fraction"],
                sub["test_f1"],
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=2,
                label=LABELS[method],
            )
            axes[row, 1].plot(
                sub["fraction"],
                sub["test_balanced_acc"],
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=2,
                label=LABELS[method],
            )
        axes[row, 0].set_title(f"{site}: F1")
        axes[row, 1].set_title(f"{site}: Balanced accuracy")
        axes[row, 0].set_ylim(-0.02, 1.02)
        axes[row, 1].set_ylim(-0.02, 1.02)
        xticks = sorted(df["fraction"].dropna().unique())
        axes[row, 0].set_xticks(xticks)
        axes[row, 1].set_xticks(xticks)
        axes[row, 0].set_xticklabels([format_fraction(x) for x in xticks])
        axes[row, 1].set_xticklabels([format_fraction(x) for x in xticks])

    axes[1, 0].set_xlabel("Labeled fraction")
    axes[1, 1].set_xlabel("Labeled fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if "reconst" in set(utah19["method"]):
        handles2, labels2 = axes[1, 0].get_legend_handles_labels()
        seen = set(labels)
        for h, l in zip(handles2, labels2):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Main label-efficiency results on recoverable domains")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, "main_label_efficiency_overview")


def plot_normalization_ablation():
    specs = [
        ("Pohang", "runs/pohang_normalization_ablation_v2/bandpass_agc_none/summary.csv", "runs/pohang_normalization_ablation_v2/bandpass_agc_robust/summary.csv"),
        ("Utah 2019", "runs/utah_2019_normalization_ablation_v2/bandpass_agc_none/summary.csv", "runs/utah_2019_normalization_ablation_v2/bandpass_agc_robust/summary.csv"),
        ("Utah 2023", "runs/utah_2023_normalization_ablation_v2/bandpass_agc_none/summary.csv", "runs/utah_2023_normalization_ablation_v2/bandpass_agc_robust/summary.csv"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=False)
    for row, (site, none_path, robust_path) in enumerate(specs):
        frames = []
        for variant, path in [("bandpass_agc_none", none_path), ("bandpass_agc_robust", robust_path)]:
            df = pd.read_csv(path).copy()
            df["variant"] = variant
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        for variant in ["bandpass_agc_none", "bandpass_agc_robust"]:
            sub = df[df["variant"] == variant].sort_values("fraction")
            axes[row, 0].plot(
                sub["fraction"], sub["test_f1"],
                marker=MARKERS[variant], color=COLORS[variant], linewidth=2, label=LABELS[variant]
            )
            axes[row, 1].plot(
                sub["fraction"], sub["test_balanced_acc"],
                marker=MARKERS[variant], color=COLORS[variant], linewidth=2, label=LABELS[variant]
            )
        axes[row, 0].set_title(f"{site}: F1")
        axes[row, 1].set_title(f"{site}: Balanced accuracy")
        axes[row, 0].set_ylim(-0.02, 1.02)
        axes[row, 1].set_ylim(-0.02, 1.02)
        xticks = sorted(df["fraction"].dropna().unique())
        axes[row, 0].set_xticks(xticks)
        axes[row, 1].set_xticks(xticks)
        axes[row, 0].set_xticklabels([format_fraction(x) for x in xticks])
        axes[row, 1].set_xticklabels([format_fraction(x) for x in xticks])

    axes[2, 0].set_xlabel("Labeled fraction")
    axes[2, 1].set_xlabel("Labeled fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Normalization ablation: AGC vs AGC + robust normalization")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, "normalization_ablation_overview")


def plot_cross_mixed():
    mixed = pd.read_csv("runs/pair_pohang_utah2019_parallel_v3/mixed_pohang_utah_2019/summary.csv").copy()
    p2u = pd.read_csv("runs/pair_pohang_utah2019_parallel_v3/cross_pohang_to_utah_2019/summary.csv").copy()
    u2p = pd.read_csv("runs/pair_pohang_utah2019_parallel_v3/cross_utah_2019_to_pohang/summary.csv").copy()
    mixed["setting"] = "Mixed P+U19"
    p2u["setting"] = "Pohang -> U19"
    u2p["setting"] = "U19 -> Pohang"
    df = pd.concat([mixed, p2u, u2p], ignore_index=True)

    settings = ["Mixed P+U19", "Pohang -> U19", "U19 -> Pohang"]
    methods = ["scratch", "reconst"]
    x = range(len(settings))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, ylabel in [
        (axes[0], "test_f1", "Test F1"),
        (axes[1], "test_balanced_acc", "Balanced accuracy"),
    ]:
        for i, method in enumerate(methods):
            vals = []
            for setting in settings:
                sub = df[(df["setting"] == setting) & (df["method"] == method)]
                vals.append(float(sub.iloc[0][metric]))
            xpos = [j + (i - 0.5) * width for j in x]
            ax.bar(xpos, vals, width=width, color=COLORS[method], label=LABELS[method])
        ax.set_xticks(list(x))
        ax.set_xticklabels(settings)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cross-site and mixed-domain transfer")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, "cross_mixed_overview")


def plot_utah2023_wasserstein():
    df = pd.read_csv("runs/utah_2023_wasserstein_offline_v1/summary.csv").copy()
    # Merge reconst metrics only
    main = pd.read_csv("runs/utah_2023_main_study/summary.csv")
    main = main[main["method"] == "reconst"].copy()
    main["variant"] = "baseline"
    none = pd.read_csv("runs/utah_2023_normalization_ablation_v2/bandpass_agc_none/summary.csv")
    none = none[none["method"] == "reconst"].copy()
    none["variant"] = "bandpass_agc_none"
    robust = pd.read_csv("runs/utah_2023_normalization_ablation_v2/bandpass_agc_robust/summary.csv")
    robust = robust[robust["method"] == "reconst"].copy()
    robust["variant"] = "bandpass_agc_robust"
    metrics = pd.concat([main, none, robust], ignore_index=True).rename(
        columns={"experiment": "run_name", "test_balanced_acc": "test_bal_acc"}
    )
    df = df.merge(
        metrics[["variant", "run_name", "fraction", "test_bal_acc", "test_specificity", "test_f1"]],
        on=["variant", "run_name"],
        how="left",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    for variant in ["baseline", "bandpass_agc_none", "bandpass_agc_robust"]:
        sub = df[df["variant"] == variant].sort_values("fraction")
        axes[0].plot(sub["fraction"], sub["test_event_noise_swd"], marker=MARKERS[variant], color=COLORS[variant], linewidth=2, label=LABELS[variant])
        axes[1].plot(sub["fraction"], sub["test_dist_gap_event_minus_noise"], marker=MARKERS[variant], color=COLORS[variant], linewidth=2, label=LABELS[variant])
        axes[2].plot(sub["fraction"], sub["test_specificity"], marker=MARKERS[variant], color=COLORS[variant], linewidth=2, label=LABELS[variant])

    axes[0].set_title("Test sliced Wasserstein")
    axes[1].set_title("Test center gap")
    axes[2].set_title("Test specificity")
    axes[0].set_ylabel("Distance")
    axes[1].set_ylabel("Gap")
    axes[2].set_ylabel("Specificity")
    for ax in axes:
        ax.set_xlabel("Labeled fraction")
        ax.set_xticks(sorted(df["fraction"].dropna().unique()))
        ax.set_xticklabels([format_fraction(x) for x in sorted(df["fraction"].dropna().unique())])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Utah 2023 failure case: latent separation remains weak")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, "utah2023_wasserstein_overview")

    export_cols = [
        "variant", "run_name", "fraction", "test_event_noise_swd",
        "test_dist_gap_event_minus_noise", "test_specificity",
        "test_bal_acc", "test_f1",
    ]
    df[export_cols].sort_values(["variant", "fraction"]).to_csv(OUT_DIR / "utah2023_wasserstein_plot_table.csv", index=False)


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_site_main_studies()
    plot_normalization_ablation()
    plot_cross_mixed()
    plot_utah2023_wasserstein()


if __name__ == "__main__":
    main()
