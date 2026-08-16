#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1"
OUT_DIR = ROOT / "figures" / "current_results_summary" / "pretrain_setting_transfer"

SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
SITE_EXP_PREFIX = {
    "pohang": "pohang",
    "utah_2019": "base_utah_2019",
    "utah_2023": "base_utah_2023",
}
FRACTIONS = {
    "0p05": 0.05,
    "0p1": 0.10,
    "0p25": 0.25,
    "0p5": 0.50,
    "1": 1.00,
}
PREPROCESSING = {
    "raw": {
        "label": "Raw",
        "color": "#6b7280",
        "marker": "^",
        "site_root": RUN_ROOT / "raw_site_main_pre50_v1",
        "cross_root": RUN_ROOT / "raw_cross_site_reconst_pre50_v1",
    },
    "filter_rms": {
        "label": "Low-pass",
        "color": "#bf4b3e",
        "marker": "o",
        "site_root": RUN_ROOT / "filter_rms_site_main_pre50_v2",
        "cross_root": RUN_ROOT / "filter_rms_cross_site_reconst_swd_interval10_v1",
    },
    "logenv": {
        "label": "Log-envelope",
        "color": "#376795",
        "marker": "s",
        "site_root": RUN_ROOT / "logenv_site_main_pre50_v2",
        "cross_root": RUN_ROOT / "logenv_cross_site_reconst_swd_interval10_v1",
    },
}
PREPROC_DIRS = {
    "raw": "raw",
    "filter_rms": "filter",
    "logenv": "log_envelope",
}
SETTING_DIRS = {
    "no_pretrain": "scratch",
    "reconst_indomain": "in_domain",
    "reconst_cross_domain": "cross_site",
}
DIRECTION_ORDER = [
    "pohang_to_utah_2019",
    "pohang_to_utah_2023",
    "utah_2019_to_pohang",
    "utah_2019_to_utah_2023",
    "utah_2023_to_pohang",
    "utah_2023_to_utah_2019",
]


def fraction_from_name(name: str) -> tuple[str, float] | None:
    match = re.search(r"__frac([0-9p]+)$", name)
    if not match:
        return None
    tag = match.group(1)
    if tag not in FRACTIONS:
        return None
    return tag, FRACTIONS[tag]


def read_metrics(path: Path) -> dict[str, float] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = data.get("or_metrics_fixed_threshold") or data.get("fc_metrics_fixed_threshold")
    if not isinstance(metrics, dict):
        return None
    out = {}
    for key in ("balanced_acc", "f1", "specificity", "recall", "precision", "acc"):
        out[key] = float(metrics.get(key, float("nan")))
    for key in ("tp", "tn", "fp", "fn"):
        out[key] = int(metrics.get(key, 0) or 0)
    return out


def site_group_dir(site_root: Path, site: str, method: str) -> Path | None:
    direct = site_root / site / method
    if direct.exists():
        return direct
    candidates = sorted(site_root.glob(f"{site}_*"))
    for candidate in candidates:
        if (candidate / method).exists():
            return candidate / method
    return None


def collect_site_main(method: str, setting_label: str) -> list[dict]:
    rows = []
    for preproc, spec in PREPROCESSING.items():
        for site in SITES:
            group = site_group_dir(spec["site_root"], site, method)
            if group is None:
                continue
            for metric_path in sorted((group / "test").glob("*/test_metrics_fixed_threshold.json")):
                parsed = fraction_from_name(metric_path.parent.name)
                if parsed is None:
                    continue
                tag, fraction = parsed
                metrics = read_metrics(metric_path)
                if metrics is None:
                    continue
                rows.append(
                    {
                        "setting": setting_label,
                        "preprocessing": preproc,
                        "site": site,
                        "source_site": site,
                        "target_site": site,
                        "direction": f"{site}_in_domain",
                        "method": method,
                        "fraction_tag": tag,
                        "fraction": fraction,
                        "path": str(metric_path.relative_to(ROOT)),
                        **metrics,
                    }
                )
    return rows


def collect_cross_reconst() -> list[dict]:
    rows = []
    for preproc, spec in PREPROCESSING.items():
        cross_root = spec["cross_root"]
        if not cross_root.exists():
            continue
        for direction in DIRECTION_ORDER:
            pair_dir = cross_root / direction / "reconst"
            if not pair_dir.exists():
                continue
            source, target = direction.split("_to_", 1)
            for metric_path in sorted((pair_dir / "test").glob("*/test_metrics_fixed_threshold.json")):
                parsed = fraction_from_name(metric_path.parent.name)
                if parsed is None:
                    continue
                tag, fraction = parsed
                metrics = read_metrics(metric_path)
                if metrics is None:
                    continue
                rows.append(
                    {
                        "setting": "reconst_cross_domain",
                        "preprocessing": preproc,
                        "site": target,
                        "source_site": source,
                        "target_site": target,
                        "direction": direction,
                        "method": "reconst",
                        "fraction_tag": tag,
                        "fraction": fraction,
                        "path": str(metric_path.relative_to(ROOT)),
                        **metrics,
                    }
                )
    return rows


def setup_ax(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks([0.05, 0.10, 0.25, 0.50, 1.00])
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.grid(axis="y", color="#ded8cf", linewidth=0.85, alpha=0.82)
    ax.grid(axis="x", color="#eee7dd", linewidth=0.55, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.2, colors="#4b5563")
    ax.set_xlabel("Label fraction", fontsize=8.8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.2)


def save(fig: plt.Figure, stem: str, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_indomain(df: pd.DataFrame, setting: str, title: str, stem: str, metric: str = "balanced_acc", ylabel: str = "Balanced accuracy") -> None:
    sub = df[df["setting"].eq(setting)]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), sharey=True)
    for ax, site in zip(axes, SITES):
        site_df = sub[sub["target_site"].eq(site)]
        for preproc, spec in PREPROCESSING.items():
            p = site_df[site_df["preprocessing"].eq(preproc)].sort_values("fraction")
            if p.empty:
                continue
            ax.plot(
                p["fraction"],
                p[metric],
                color=spec["color"],
                marker=spec["marker"],
                markersize=5.1,
                linewidth=2.05,
                label=spec["label"],
            )
        ax.set_title(SITE_LABELS[site], fontsize=10.6, fontweight="normal", pad=8)
        setup_ax(ax, ylabel if ax is axes[0] else None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=3, frameon=False, fontsize=9.0)
    fig.suptitle(title, fontsize=12.0, y=1.035)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.9)
    save(fig, stem)


def plot_indomain_single_preproc(
    df: pd.DataFrame,
    setting: str,
    preproc: str,
    title: str,
    stem: str,
    out_dir: Path,
    metric: str = "balanced_acc",
    ylabel: str = "Balanced accuracy",
) -> None:
    sub = df[df["setting"].eq(setting) & df["preprocessing"].eq(preproc)]
    spec = PREPROCESSING[preproc]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), sharey=True)
    for ax, site in zip(axes, SITES):
        site_df = sub[sub["target_site"].eq(site)].sort_values("fraction")
        if not site_df.empty:
            ax.plot(
                site_df["fraction"],
                site_df[metric],
                color=spec["color"],
                marker=spec["marker"],
                markersize=5.3,
                linewidth=2.15,
                label=spec["label"],
            )
        ax.set_title(SITE_LABELS[site], fontsize=10.6, fontweight="normal", pad=8)
        setup_ax(ax, ylabel if ax is axes[0] else None)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=1, frameon=False, fontsize=9.0)
    fig.suptitle(title, fontsize=12.0, y=1.035)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.9)
    save(fig, stem, out_dir)


def direction_label(direction: str) -> str:
    source, target = direction.split("_to_", 1)
    return f"{SITE_LABELS[source]} -> {SITE_LABELS[target]}"


def plot_cross(df: pd.DataFrame, metric: str = "balanced_acc", ylabel: str = "Balanced accuracy", stem: str = "reconst_cross_domain_balanced_accuracy") -> None:
    sub = df[df["setting"].eq("reconst_cross_domain")]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.0), sharey=True)
    for ax, direction in zip(axes.ravel(), DIRECTION_ORDER):
        ddf = sub[sub["direction"].eq(direction)]
        for preproc, spec in PREPROCESSING.items():
            p = ddf[ddf["preprocessing"].eq(preproc)].sort_values("fraction")
            if p.empty:
                continue
            ax.plot(
                p["fraction"],
                p[metric],
                color=spec["color"],
                marker=spec["marker"],
                markersize=4.7,
                linewidth=1.9,
                label=spec["label"],
            )
        ax.set_title(direction_label(direction), fontsize=9.2, fontweight="normal", pad=8)
        setup_ax(ax, ylabel if ax in axes[:, 0] else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=3, frameon=False, fontsize=9.0)
    fig.suptitle("Reconstruction-pretrained cross-domain transfer", fontsize=12.0, y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.8, h_pad=1.0)
    save(fig, stem)


def plot_cross_single_preproc(
    df: pd.DataFrame,
    preproc: str,
    out_dir: Path,
    metric: str = "balanced_acc",
    ylabel: str = "Balanced accuracy",
    stem: str = "balanced_accuracy",
) -> None:
    sub = df[df["setting"].eq("reconst_cross_domain") & df["preprocessing"].eq(preproc)]
    spec = PREPROCESSING[preproc]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.0), sharey=True)
    for ax, direction in zip(axes.ravel(), DIRECTION_ORDER):
        ddf = sub[sub["direction"].eq(direction)].sort_values("fraction")
        if not ddf.empty:
            ax.plot(
                ddf["fraction"],
                ddf[metric],
                color=spec["color"],
                marker=spec["marker"],
                markersize=4.8,
                linewidth=2.0,
                label=spec["label"],
            )
        ax.set_title(direction_label(direction), fontsize=9.2, fontweight="normal", pad=8)
        setup_ax(ax, ylabel if ax in axes[:, 0] else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=1, frameon=False, fontsize=9.0)
    fig.suptitle(f"Reconstruction-pretrained cross-domain transfer: {spec['label']}", fontsize=12.0, y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.8, h_pad=1.0)
    save(fig, stem, out_dir)


def write_organized_outputs(df: pd.DataFrame) -> None:
    for setting, setting_dir_name in SETTING_DIRS.items():
        for preproc, preproc_dir_name in PREPROC_DIRS.items():
            out_dir = OUT_DIR / setting_dir_name / preproc_dir_name
            sub = df[df["setting"].eq(setting) & df["preprocessing"].eq(preproc)].copy()
            out_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(out_dir / "metrics.csv", index=False)
            if sub.empty:
                continue
            if setting == "no_pretrain":
                plot_indomain_single_preproc(
                    df,
                    setting,
                    preproc,
                    f"No-pretrain in-domain baseline: {PREPROCESSING[preproc]['label']}",
                    "balanced_accuracy",
                    out_dir,
                )
                plot_indomain_single_preproc(
                    df,
                    setting,
                    preproc,
                    f"No-pretrain in-domain baseline: {PREPROCESSING[preproc]['label']}",
                    "f1_score",
                    out_dir,
                    metric="f1",
                    ylabel="F1 score",
                )
            elif setting == "reconst_indomain":
                plot_indomain_single_preproc(
                    df,
                    setting,
                    preproc,
                    f"Reconstruction-pretrained in-domain transfer: {PREPROCESSING[preproc]['label']}",
                    "balanced_accuracy",
                    out_dir,
                )
                plot_indomain_single_preproc(
                    df,
                    setting,
                    preproc,
                    f"Reconstruction-pretrained in-domain transfer: {PREPROCESSING[preproc]['label']}",
                    "f1_score",
                    out_dir,
                    metric="f1",
                    ylabel="F1 score",
                )
            elif setting == "reconst_cross_domain":
                plot_cross_single_preproc(df, preproc, out_dir)
                plot_cross_single_preproc(df, preproc, out_dir, metric="f1", ylabel="F1 score", stem="f1_score")


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
    rows = []
    rows.extend(collect_site_main("scratch", "no_pretrain"))
    rows.extend(collect_site_main("reconst", "reconst_indomain"))
    rows.extend(collect_cross_reconst())
    df = pd.DataFrame(rows).sort_values(["setting", "preprocessing", "direction", "fraction"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "pretrain_setting_transfer_metrics.csv", index=False)

    plot_indomain(df, "no_pretrain", "No-pretrain in-domain baseline", "no_pretrain_indomain_balanced_accuracy")
    plot_indomain(df, "no_pretrain", "No-pretrain in-domain baseline", "no_pretrain_indomain_f1_score", metric="f1", ylabel="F1 score")
    plot_indomain(
        df,
        "reconst_indomain",
        "Reconstruction-pretrained in-domain transfer",
        "reconst_indomain_balanced_accuracy",
    )
    plot_indomain(
        df,
        "reconst_indomain",
        "Reconstruction-pretrained in-domain transfer",
        "reconst_indomain_f1_score",
        metric="f1",
        ylabel="F1 score",
    )
    plot_cross(df)
    plot_cross(df, metric="f1", ylabel="F1 score", stem="reconst_cross_domain_f1_score")
    write_organized_outputs(df)
    print(f"[DONE] wrote pretrain-setting figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
