#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_BASE = ROOT / "runs" / "metadata_v2_safe_rerun_v1"
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "center" / "swd_distribution"

ROOTS = {
    "logenv": RUN_BASE / "logenv_cross_site_reconst_swd_interval10_v1",
    "filter_rms": RUN_BASE / "filter_rms_cross_site_reconst_swd_interval10_v1",
}
LABELS = {
    "logenv": "Log envelope",
    "filter_rms": "Filter + RMS",
}
SITE_SHORT = {
    "pohang": "A",
    "utah_2019": "B",
    "utah_2023": "C",
}
FRACTION_LABEL = {
    "0p05": "0.05",
    "0p1": "0.10",
    "0p25": "0.25",
    "0p5": "0.50",
    "1": "1.00",
}
FRACTION_ORDER = ["0p05", "0p1", "0p25", "0p5", "1"]
COLORS = {
    "logenv": "#376795",
    "filter_rms": "#bf4b3e",
}


def parse_fraction(name: str) -> str | None:
    match = re.search(r"__frac([0-9p]+)$", name)
    if not match:
        return None
    tag = match.group(1)
    return tag if tag in FRACTION_LABEL else None


def collect(root: Path, preprocessing: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.glob("*_to_*/reconst/finetune/*/center_history.csv")):
        run_dir = path.parent
        if not (run_dir / "finetune_summary.json").exists():
            continue
        tag = parse_fraction(run_dir.name)
        if tag is None:
            continue
        source, target = path.parts[-5].split("_to_", 1)
        df = pd.read_csv(path)
        if "val_event_noise_swd" not in df.columns:
            continue
        df = df[df["val_event_noise_swd"].notna()].copy()
        if df.empty:
            continue
        df["preprocessing"] = preprocessing
        df["source_site"] = source
        df["target_site"] = target
        df["direction"] = f"{source}_to_{target}"
        df["direction_short"] = f"{SITE_SHORT[source]}->{SITE_SHORT[target]}"
        df["fraction_tag"] = tag
        df["fraction_label"] = FRACTION_LABEL[tag]
        df["center_history_path"] = str(path.relative_to(ROOT))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d7d2c8", linewidth=0.8, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_empty(preprocessing: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.axis("off")
    ax.text(
        0.5,
        0.58,
        f"No SWD history found for {LABELS[preprocessing]}",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.5,
        0.42,
        "The completed runs contain performance summaries,\n"
        "but no center_history.csv with val_event_noise_swd.",
        ha="center",
        va="center",
        fontsize=11,
        color="#555555",
    )
    save(fig, f"{preprocessing}_swd_distribution_not_logged")


def plot_preprocessing_distribution(df: pd.DataFrame, preprocessing: str) -> None:
    if df.empty:
        plot_empty(preprocessing)
        return
    final = df.loc[df.groupby(["direction", "fraction_tag"])["epoch"].idxmax()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    axes[0].hist(
        df["val_event_noise_swd"],
        bins=18,
        color=COLORS[preprocessing],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].set_title("All logged epochs", fontsize=12, pad=10)
    axes[0].set_xlabel("Validation event-noise SWD")
    axes[0].set_ylabel("Count")
    style_axis(axes[0])

    axes[1].hist(
        final["val_event_noise_swd"],
        bins=min(12, max(4, len(final) // 2)),
        color=COLORS[preprocessing],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[1].set_title("Final epoch per run", fontsize=12, pad=10)
    axes[1].set_xlabel("Validation event-noise SWD")
    axes[1].set_ylabel("Count")
    style_axis(axes[1])
    fig.suptitle(f"{LABELS[preprocessing]}: SWD distribution", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, f"{preprocessing}_swd_distribution_hist")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    data = [
        final[final["fraction_tag"] == tag]["val_event_noise_swd"].to_numpy()
        for tag in FRACTION_ORDER
        if not final[final["fraction_tag"] == tag].empty
    ]
    labels = [
        FRACTION_LABEL[tag]
        for tag in FRACTION_ORDER
        if not final[final["fraction_tag"] == tag].empty
    ]
    parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.72)
    for body in parts["bodies"]:
        body.set_facecolor(COLORS[preprocessing])
        body.set_alpha(0.65)
        body.set_edgecolor("#333333")
    for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1.1)
    rng = np.random.default_rng(7)
    for i, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.055, 0.055, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=34,
            color="#222222",
            alpha=0.72,
            zorder=3,
        )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Target label fraction")
    ax.set_ylabel("Final validation event-noise SWD")
    ax.set_title(f"{LABELS[preprocessing]}: final SWD by label fraction", fontsize=13, pad=12)
    style_axis(ax)
    fig.tight_layout()
    save(fig, f"{preprocessing}_swd_final_by_fraction_violin")

    fig, ax = plt.subplots(figsize=(9.5, 5.1))
    directions = sorted(final["direction_short"].unique())
    positions = np.arange(len(directions))
    vals = [final[final["direction_short"] == d]["val_event_noise_swd"].to_numpy() for d in directions]
    ax.boxplot(vals, positions=positions, widths=0.55, patch_artist=True)
    for patch in ax.artists:
        patch.set_facecolor(COLORS[preprocessing])
        patch.set_alpha(0.65)
    for i, direction in enumerate(directions):
        sub = final[final["direction_short"] == direction]
        ax.scatter(
            np.full(len(sub), i),
            sub["val_event_noise_swd"],
            s=42,
            color=COLORS[preprocessing],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(directions)
    ax.set_xlabel("Transfer direction")
    ax.set_ylabel("Final validation event-noise SWD")
    ax.set_title(f"{LABELS[preprocessing]}: final SWD by transfer direction", fontsize=13, pad=12)
    style_axis(ax)
    fig.tight_layout()
    save(fig, f"{preprocessing}_swd_final_by_direction_box")

    final.to_csv(OUT_DIR / f"{preprocessing}_swd_final_epoch.csv", index=False)


def plot_combined(logenv: pd.DataFrame, filter_rms: pd.DataFrame) -> None:
    frames = []
    for df in (logenv, filter_rms):
        if df.empty:
            continue
        frames.append(df.loc[df.groupby(["preprocessing", "direction", "fraction_tag"])["epoch"].idxmax()].copy())
    if not frames:
        return
    final = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = []
    data = []
    colors = []
    for preproc in ("logenv", "filter_rms"):
        sub = final[final["preprocessing"] == preproc]
        if sub.empty:
            continue
        labels.append(LABELS[preproc])
        data.append(sub["val_event_noise_swd"].to_numpy())
        colors.append(COLORS[preproc])
    parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.72)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor("#333333")
    for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1.1)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final validation event-noise SWD")
    ax.set_title("Final SWD distribution by preprocessing", fontsize=13, pad=12)
    style_axis(ax)
    fig.tight_layout()
    save(fig, "combined_swd_final_by_preprocessing_violin")
    final.to_csv(OUT_DIR / "combined_swd_final_epoch.csv", index=False)


def plot_combined_gap_and_swd(logenv: pd.DataFrame, filter_rms: pd.DataFrame) -> None:
    frames = []
    for df in (logenv, filter_rms):
        if df.empty:
            continue
        frames.append(df.loc[df.groupby(["preprocessing", "direction", "fraction_tag"])["epoch"].idxmax()].copy())
    if not frames:
        return
    final = pd.concat(frames, ignore_index=True)
    metrics = [
        ("val_event_noise_swd", "Final validation SWD"),
        ("val_dist_gap_event_minus_noise", "Final validation center gap"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    for ax, (metric, ylabel) in zip(axes, metrics):
        labels = []
        data = []
        colors = []
        for preproc in ("logenv", "filter_rms"):
            sub = final[final["preprocessing"] == preproc]
            if sub.empty or metric not in sub.columns:
                continue
            labels.append(LABELS[preproc])
            data.append(sub[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
            colors.append(COLORS[preproc])
        parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.72)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.65)
            body.set_edgecolor("#333333")
        for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(1.1)
        rng = np.random.default_rng(11)
        for i, values in enumerate(data, start=1):
            jitter = rng.uniform(-0.055, 0.055, size=len(values))
            ax.scatter(np.full(len(values), i) + jitter, values, s=28, color="#222222", alpha=0.55, zorder=3)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        style_axis(ax)
    fig.suptitle("Preprocessing changes final latent geometry", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, "combined_final_swd_and_center_gap_by_preprocessing")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for preproc in ("logenv", "filter_rms"):
        sub = final[final["preprocessing"] == preproc]
        if sub.empty:
            continue
        ax.scatter(
            sub["val_dist_gap_event_minus_noise"],
            sub["val_event_noise_swd"],
            s=58,
            color=COLORS[preproc],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
            label=LABELS[preproc],
        )
    ax.axvline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_xlabel("Final validation center gap")
    ax.set_ylabel("Final validation event-noise SWD")
    ax.set_title("Final center gap vs SWD by preprocessing", fontsize=13, pad=12)
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    save(fig, "combined_final_center_gap_vs_swd_by_preprocessing")


def plot_pairwise_delta(logenv: pd.DataFrame, filter_rms: pd.DataFrame) -> None:
    if logenv.empty or filter_rms.empty:
        return

    def finalize(df: pd.DataFrame) -> pd.DataFrame:
        final = df.loc[df.groupby(["preprocessing", "direction", "fraction_tag"])["epoch"].idxmax()].copy()
        return final[
            [
                "preprocessing",
                "direction",
                "direction_short",
                "fraction_tag",
                "fraction_label",
                "val_event_noise_swd",
                "val_dist_gap_event_minus_noise",
            ]
        ]

    left = finalize(logenv).rename(
        columns={
            "val_event_noise_swd": "swd_logenv",
            "val_dist_gap_event_minus_noise": "gap_logenv",
        }
    )
    right = finalize(filter_rms).rename(
        columns={
            "val_event_noise_swd": "swd_filter_rms",
            "val_dist_gap_event_minus_noise": "gap_filter_rms",
        }
    )
    merged = pd.merge(
        left.drop(columns=["preprocessing"]),
        right.drop(columns=["preprocessing", "direction_short", "fraction_label"]),
        on=["direction", "fraction_tag"],
        how="inner",
    )
    if merged.empty:
        return
    merged["delta_swd_filter_minus_logenv"] = merged["swd_filter_rms"] - merged["swd_logenv"]
    merged["delta_gap_filter_minus_logenv"] = merged["gap_filter_rms"] - merged["gap_logenv"]
    merged = merged.sort_values(["direction", "fraction_tag"])
    merged.to_csv(OUT_DIR / "pairwise_filter_minus_logenv_latent_delta.csv", index=False)

    y_labels = [f"{row.direction_short} / {row.fraction_label}" for row in merged.itertuples()]
    y = np.arange(len(merged))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, max(5.4, 0.29 * len(merged))), sharey=True)
    for ax, metric, title in [
        (axes[0], "delta_swd_filter_minus_logenv", "Delta SWD"),
        (axes[1], "delta_gap_filter_minus_logenv", "Delta center gap"),
    ]:
        vals = merged[metric].to_numpy()
        colors = np.where(vals >= 0, COLORS["filter_rms"], COLORS["logenv"])
        ax.barh(y, vals, color=colors, alpha=0.82)
        ax.axvline(0, color="#333333", linewidth=1.0)
        ax.set_title(f"Filter RMS - Log env: {title}", fontsize=12, pad=10)
        ax.set_xlabel("Difference")
        style_axis(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(y_labels, fontsize=8)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle("Pairwise latent-diagnostic change caused by preprocessing", y=1.01, fontsize=14)
    fig.tight_layout()
    save(fig, "pairwise_filter_minus_logenv_swd_center_gap_delta")


def write_availability(logenv: pd.DataFrame, filter_rms: pd.DataFrame) -> None:
    rows = []
    for preproc, root, df in (
        ("logenv", ROOTS["logenv"], logenv),
        ("filter_rms", ROOTS["filter_rms"], filter_rms),
    ):
        rows.append(
            {
                "preprocessing": preproc,
                "root": str(root.relative_to(ROOT)) if root.exists() else str(root.relative_to(ROOT)),
                "root_exists": root.exists(),
                "center_history_runs": int(df[["direction", "fraction_tag"]].drop_duplicates().shape[0]) if not df.empty else 0,
                "epoch_rows": int(len(df)) if not df.empty else 0,
                "has_swd": bool(not df.empty),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "swd_availability.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logenv = collect(ROOTS["logenv"], "logenv")
    filter_rms = collect(ROOTS["filter_rms"], "filter_rms")
    write_availability(logenv, filter_rms)
    if not logenv.empty:
        logenv.to_csv(OUT_DIR / "logenv_swd_all_epochs.csv", index=False)
    if not filter_rms.empty:
        filter_rms.to_csv(OUT_DIR / "filter_rms_swd_all_epochs.csv", index=False)
    plot_preprocessing_distribution(logenv, "logenv")
    plot_preprocessing_distribution(filter_rms, "filter_rms")
    plot_combined(logenv, filter_rms)
    plot_combined_gap_and_swd(logenv, filter_rms)
    plot_pairwise_delta(logenv, filter_rms)
    print(f"[DONE] logenv SWD rows={len(logenv)}")
    print(f"[DONE] filter_rms SWD rows={len(filter_rms)}")
    print(f"[DONE] wrote SWD distribution figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
