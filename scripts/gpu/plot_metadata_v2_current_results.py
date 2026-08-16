#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1"
OUT_ROOT = ROOT / "temp" / "current_results_summary"
FIG_DIR = OUT_ROOT / "figures_metadata_v2"

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
FRACTION_TAGS = ["0p05", "0p1", "0p25", "0p5", "1"]
FRACTIONS = {
    "0p05": 0.05,
    "0p1": 0.10,
    "0p25": 0.25,
    "0p5": 0.50,
    "1": 1.00,
}
METHODS = ["scratch", "contrast", "reconst", "reconst_noanom"]
METHOD_LABELS = {
    "scratch": "Scratch",
    "contrast": "Contrast",
    "reconst": "Reconst",
    "reconst_noanom": "Reconst no-anom",
}
COLORS = {
    "scratch": "#376795",
    "contrast": "#d17b0f",
    "reconst": "#3f8f4f",
    "reconst_noanom": "#8f5b9a",
    "logenv": "#376795",
    "filter_rms": "#bf4b3e",
}
METRIC_BRANCH = "or_metrics_fixed_threshold"


def load_metrics(path: Path) -> dict[str, float] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = data.get(METRIC_BRANCH)
    if not isinstance(metrics, dict):
        metrics = data.get("fc_metrics_fixed_threshold")
    if not isinstance(metrics, dict):
        return None
    out: dict[str, float] = {}
    for key in ("f1", "balanced_acc", "specificity", "recall", "precision", "acc"):
        value = metrics.get(key)
        out[key] = float(value) if value is not None else np.nan
    for key in ("tp", "tn", "fp", "fn"):
        value = metrics.get(key)
        out[key] = int(value) if value is not None else 0
    return out


def parse_fraction(exp_dir_name: str) -> tuple[str, float] | None:
    match = re.search(r"__frac([0-9p]+)$", exp_dir_name)
    if not match:
        return None
    tag = match.group(1)
    if tag not in FRACTIONS:
        return None
    return tag, FRACTIONS[tag]


def collect_site_main(preproc: str, root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows
    for site in SITES:
        for group_dir in root.glob(f"{site}_*"):
            if not group_dir.is_dir():
                continue
            for method in METHODS:
                test_root = group_dir / method / "test"
                if not test_root.exists():
                    continue
                for metric_path in sorted(test_root.glob("*/test_metrics_fixed_threshold.json")):
                    parsed = parse_fraction(metric_path.parent.name)
                    if parsed is None:
                        continue
                    tag, fraction = parsed
                    metrics = load_metrics(metric_path)
                    if metrics is None:
                        continue
                    row = {
                        "study": "site_main",
                        "preprocessing": preproc,
                        "source_site": site,
                        "target_site": site,
                        "direction": f"{site}_in_domain",
                        "method": method,
                        "fraction_tag": tag,
                        "fraction": fraction,
                        "path": str(metric_path.relative_to(ROOT)),
                    }
                    row.update(metrics)
                    rows.append(row)
    return rows


def collect_cross(preproc: str, root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows
    for pair_dir in sorted(root.glob("*_to_*")):
        if not pair_dir.is_dir():
            continue
        source, target = pair_dir.name.split("_to_", 1)
        test_root = pair_dir / "reconst" / "test"
        if not test_root.exists():
            continue
        for metric_path in sorted(test_root.glob("*/test_metrics_fixed_threshold.json")):
            parsed = parse_fraction(metric_path.parent.name)
            if parsed is None:
                continue
            tag, fraction = parsed
            metrics = load_metrics(metric_path)
            if metrics is None:
                continue
            row = {
                "study": "cross_site",
                "preprocessing": preproc,
                "source_site": source,
                "target_site": target,
                "direction": pair_dir.name,
                "method": "reconst",
                "fraction_tag": tag,
                "fraction": fraction,
                "path": str(metric_path.relative_to(ROOT)),
            }
            row.update(metrics)
            rows.append(row)
    return rows


def collect_all() -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(collect_site_main("logenv", RUN_ROOT / "logenv_site_main_pre50_v2"))
    rows.extend(collect_site_main("filter_rms", RUN_ROOT / "filter_rms_site_main_pre50_v2"))
    rows.extend(
        collect_cross(
            "logenv",
            RUN_ROOT / "logenv_cross_site_reconst_swd_interval10_v1",
        )
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["study", "preprocessing", "target_site", "source_site", "method", "fraction"]
        )
    return df


def setup_ax(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([FRACTIONS[tag] for tag in FRACTION_TAGS])
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.grid(axis="y", color="#d7d2c8", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_site_main(df: pd.DataFrame, metric: str = "balanced_acc") -> None:
    metric_label = "Balanced accuracy" if metric == "balanced_acc" else metric
    for preproc in ("logenv", "filter_rms"):
        sub = df[(df["study"] == "site_main") & (df["preprocessing"] == preproc)]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
        for ax, site in zip(axes, SITES):
            site_df = sub[sub["target_site"] == site]
            for method in METHODS:
                mdf = site_df[site_df["method"] == method].sort_values("fraction")
                if mdf.empty:
                    continue
                ax.plot(
                    mdf["fraction"],
                    mdf[metric],
                    marker="o",
                    linewidth=2.2,
                    markersize=5,
                    color=COLORS[method],
                    label=METHOD_LABELS[method],
                )
            ax.set_title(SITE_LABELS[site], fontsize=11, pad=10)
            ax.set_xlabel("Fine-tuning label fraction")
            setup_ax(ax, metric_label if ax is axes[0] else None)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.06),
        )
        fig.suptitle(f"{preproc}: in-domain label efficiency", y=1.04, fontsize=14)
        fig.tight_layout(rect=[0, 0.10, 1, 0.96])
        save(fig, f"{preproc}_site_main_{metric}")


def plot_site_metric_dashboard(df: pd.DataFrame) -> None:
    metrics = [
        ("balanced_acc", "Balanced accuracy"),
        ("f1", "F1"),
        ("specificity", "Specificity"),
        ("recall", "Recall"),
    ]
    for preproc in ("logenv", "filter_rms"):
        sub = df[(df["study"] == "site_main") & (df["preprocessing"] == preproc)]
        if sub.empty:
            continue
        fig, axes = plt.subplots(3, 4, figsize=(15.5, 9.2), sharex=True, sharey=True)
        for r, site in enumerate(SITES):
            site_df = sub[sub["target_site"] == site]
            for c, (metric, label) in enumerate(metrics):
                ax = axes[r, c]
                for method in METHODS:
                    mdf = site_df[site_df["method"] == method].sort_values("fraction")
                    if mdf.empty:
                        continue
                    ax.plot(
                        mdf["fraction"],
                        mdf[metric],
                        marker="o",
                        linewidth=1.9,
                        markersize=4.5,
                        color=COLORS[method],
                        label=METHOD_LABELS[method],
                    )
                setup_ax(ax)
                if r == 0:
                    ax.set_title(label, fontsize=11, pad=10)
                if c == 0:
                    ax.set_ylabel(SITE_LABELS[site])
                if r == 2:
                    ax.set_xlabel("Label fraction")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.005),
        )
        fig.suptitle(f"{preproc}: site-wise metrics", y=1.0, fontsize=14)
        fig.tight_layout(rect=[0, 0.045, 1, 0.965])
        save(fig, f"{preproc}_site_main_metric_dashboard")


def plot_preprocessing_comparison(df: pd.DataFrame) -> None:
    sub = df[(df["study"] == "site_main") & (df["method"] == "reconst")]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    for ax, site in zip(axes, SITES):
        site_df = sub[sub["target_site"] == site]
        for preproc in ("logenv", "filter_rms"):
            pdf = site_df[site_df["preprocessing"] == preproc].sort_values("fraction")
            if pdf.empty:
                continue
            ax.plot(
                pdf["fraction"],
                pdf["balanced_acc"],
                marker="o",
                linewidth=2.3,
                markersize=5,
                color=COLORS[preproc],
                label="Log envelope" if preproc == "logenv" else "Filter + RMS",
            )
        ax.set_title(SITE_LABELS[site], fontsize=11, pad=10)
        ax.set_xlabel("Fine-tuning label fraction")
        setup_ax(ax, "Balanced accuracy" if ax is axes[0] else None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.045),
    )
    fig.suptitle("Reconst transfer: preprocessing sensitivity", y=1.04, fontsize=14)
    fig.tight_layout(rect=[0, 0.095, 1, 0.96])
    save(fig, "reconst_preprocessing_comparison_balanced_acc")


def plot_utah2023_failure(df: pd.DataFrame) -> None:
    sub = df[
        (df["study"] == "site_main")
        & (df["target_site"] == "utah_2023")
        & (df["method"].isin(["scratch", "reconst"]))
    ]
    if sub.empty:
        return
    metrics = [
        ("f1", "F1"),
        ("balanced_acc", "Balanced accuracy"),
        ("specificity", "Specificity"),
        ("recall", "Recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for preproc in ("logenv", "filter_rms"):
            for method in ("scratch", "reconst"):
                mdf = sub[
                    (sub["preprocessing"] == preproc) & (sub["method"] == method)
                ].sort_values("fraction")
                if mdf.empty:
                    continue
                linestyle = "-" if method == "reconst" else "--"
                marker = "o" if method == "reconst" else "s"
                ax.plot(
                    mdf["fraction"],
                    mdf[metric],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.0,
                    markersize=4.5,
                    color=COLORS[preproc],
                    label=f"{'Log env' if preproc == 'logenv' else 'Filter RMS'} / {METHOD_LABELS[method]}",
                )
        ax.set_title(label, fontsize=11, pad=10)
        setup_ax(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Metric value")
    for ax in axes[-1, :]:
        ax.set_xlabel("Fine-tuning label fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle("Site C / Utah 2023: F1 can hide specificity collapse", y=1.0, fontsize=14)
    fig.tight_layout(rect=[0, 0.095, 1, 0.955])
    save(fig, "utah2023_failure_metric_dashboard")


def plot_cross_site(df: pd.DataFrame) -> None:
    sub = df[(df["study"] == "cross_site") & (df["preprocessing"] == "logenv")]
    if sub.empty:
        return
    directions = [
        ("pohang", "utah_2019"),
        ("utah_2019", "pohang"),
        ("pohang", "utah_2023"),
        ("utah_2023", "pohang"),
        ("utah_2019", "utah_2023"),
        ("utah_2023", "utah_2019"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2), sharex=True, sharey=True)
    for ax, (source, target) in zip(axes.ravel(), directions):
        ddf = sub[
            (sub["source_site"] == source) & (sub["target_site"] == target)
        ].sort_values("fraction")
        for metric, label, color, marker in [
            ("balanced_acc", "Balanced accuracy", "#376795", "o"),
            ("f1", "F1", "#d17b0f", "s"),
            ("specificity", "Specificity", "#3f8f4f", "^"),
        ]:
            if ddf.empty:
                continue
            ax.plot(
                ddf["fraction"],
                ddf[metric],
                marker=marker,
                linewidth=2.0,
                markersize=4.5,
                color=color,
                label=label,
            )
        ax.set_title(
            f"{SITE_LABELS[source]} -> {SITE_LABELS[target]}",
            fontsize=10,
            pad=10,
        )
        ax.set_xlabel("Target label fraction")
        setup_ax(ax, "Metric value" if ax in axes[:, 0] else None)
        if ddf.empty:
            ax.text(0.5, 0.5, "No completed test", ha="center", va="center", transform=ax.transAxes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle("Log-envelope cross-site reconst transfer", y=1.0, fontsize=14)
    fig.tight_layout(rect=[0, 0.065, 1, 0.955])
    save(fig, "logenv_cross_site_reconst_metrics")


def plot_cross_matrix(df: pd.DataFrame) -> None:
    sub = df[(df["study"] == "cross_site") & (df["preprocessing"] == "logenv")]
    if sub.empty:
        return
    available_tags = [tag for tag in FRACTION_TAGS if (sub["fraction_tag"] == tag).any()]
    if not available_tags:
        return
    fig, axes = plt.subplots(1, len(available_tags), figsize=(3.35 * len(available_tags), 3.8), squeeze=False)
    for ax, tag in zip(axes.ravel(), available_tags):
        mat = np.full((len(SITES), len(SITES)), np.nan)
        for r, source in enumerate(SITES):
            for c, target in enumerate(SITES):
                if source == target:
                    continue
                row = sub[
                    (sub["fraction_tag"] == tag)
                    & (sub["source_site"] == source)
                    & (sub["target_site"] == target)
                ]
                if not row.empty:
                    mat[r, c] = float(row.iloc[0]["balanced_acc"])
        image = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"frac={FRACTIONS[tag]:.2f}", fontsize=10, pad=8)
        ax.set_xticks(range(len(SITES)))
        ax.set_yticks(range(len(SITES)))
        ax.set_xticklabels(["A", "B", "C"])
        ax.set_yticklabels(["A", "B", "C"])
        ax.set_xlabel("Target")
        if ax is axes.ravel()[0]:
            ax.set_ylabel("Source")
        for r in range(len(SITES)):
            for c in range(len(SITES)):
                if np.isfinite(mat[r, c]):
                    ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", color="white", fontsize=8)
                elif r == c:
                    ax.text(c, r, "-", ha="center", va="center", color="#444444", fontsize=10)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.024, pad=0.015)
    cbar.set_label("Balanced accuracy")
    fig.suptitle("Cross-site transfer matrix: log-envelope reconst", y=1.04, fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])
    save(fig, "logenv_cross_site_transfer_matrix_balanced_acc")


def write_completion(df: pd.DataFrame) -> None:
    expected = []
    for preproc in ("logenv", "filter_rms"):
        for site in SITES:
            for method in METHODS:
                for tag in FRACTION_TAGS:
                    expected.append((preproc, "site_main", site, site, method, tag))
    for source in SITES:
        for target in SITES:
            if source == target:
                continue
            for tag in FRACTION_TAGS:
                expected.append(("logenv", "cross_site", source, target, "reconst", tag))

    rows = []
    keys = set(
        zip(
            df["preprocessing"],
            df["study"],
            df["source_site"],
            df["target_site"],
            df["method"],
            df["fraction_tag"],
        )
    ) if not df.empty else set()
    for preproc, study, source, target, method, tag in expected:
        rows.append(
            {
                "preprocessing": preproc,
                "study": study,
                "source_site": source,
                "target_site": target,
                "method": method,
                "fraction_tag": tag,
                "fraction": FRACTIONS[tag],
                "test_done": (preproc, study, source, target, method, tag) in keys,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_ROOT / "metadata_v2_completion_current.csv", index=False)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = collect_all()
    df.to_csv(OUT_ROOT / "metadata_v2_metrics_current.csv", index=False)
    write_completion(df)
    if df.empty:
        raise SystemExit("No completed metadata_v2 metrics found.")

    plot_site_main(df, "balanced_acc")
    plot_site_metric_dashboard(df)
    plot_preprocessing_comparison(df)
    plot_utah2023_failure(df)
    plot_cross_site(df)
    plot_cross_matrix(df)
    print(f"[DONE] rows={len(df)}")
    print(f"[DONE] wrote CSV: {OUT_ROOT / 'metadata_v2_metrics_current.csv'}")
    print(f"[DONE] wrote figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
