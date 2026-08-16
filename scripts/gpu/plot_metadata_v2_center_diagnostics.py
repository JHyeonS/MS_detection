#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1" / "logenv_cross_site_reconst_swd_interval10_v1"
OUT_ROOT = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "center"

SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_SHORT = {"pohang": "A", "utah_2019": "B", "utah_2023": "C"}
SITE_LABEL = {
    "pohang": "Site A / Pohang",
    "utah_2019": "Site B / Utah 2019",
    "utah_2023": "Site C / Utah 2023",
}
FRACTION_LABEL = {
    "0p05": "0.05",
    "0p1": "0.10",
    "0p25": "0.25",
    "0p5": "0.50",
    "1": "1.00",
}
FRACTION_VALUE = {key: float(value) for key, value in FRACTION_LABEL.items()}
FRACTION_ORDER = ["0p05", "0p1", "0p25", "0p5", "1"]
FRACTION_COLORS = {
    "0p05": "#376795",
    "0p1": "#4c956c",
    "0p25": "#d17b0f",
    "0p5": "#bf4b3e",
    "1": "#6f4e9b",
}


def parse_fraction(run_name: str) -> str | None:
    match = re.search(r"__frac([0-9p]+)$", run_name)
    if not match:
        return None
    tag = match.group(1)
    return tag if tag in FRACTION_LABEL else None


def collect_center_history() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not RUN_ROOT.exists():
        return pd.DataFrame()
    for path in sorted(RUN_ROOT.glob("*_to_*/reconst/finetune/*/center_history.csv")):
        run_dir = path.parent
        if not (run_dir / "finetune_summary.json").exists():
            # Skip interrupted runs; partial epoch-1 diagnostics are not stable
            # enough for poster figures even when a temporary best.pt exists.
            continue
        fraction_tag = parse_fraction(run_dir.name)
        if fraction_tag is None:
            continue
        source, target = path.parts[-5].split("_to_", 1)
        df = pd.read_csv(path)
        df["source_site"] = source
        df["target_site"] = target
        df["direction"] = f"{source}_to_{target}"
        df["fraction_tag"] = fraction_tag
        df["fraction"] = FRACTION_VALUE[fraction_tag]
        df["center_history_path"] = str(path.relative_to(ROOT))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out.sort_values(["direction", "fraction", "epoch"])
    return out


def final_epoch(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["direction", "fraction_tag"])["epoch"].idxmax()
    out = df.loc[idx].copy()
    out = out.sort_values(["target_site", "source_site", "fraction"])
    return out


def setup_epoch_axis(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.grid(axis="y", color="#d7d2c8", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Fine-tuning epoch")
    if ylabel:
        ax.set_ylabel(ylabel)


def setup_fraction_axis(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_xticks([FRACTION_VALUE[tag] for tag in FRACTION_ORDER])
    ax.set_xticklabels([FRACTION_LABEL[tag] for tag in FRACTION_ORDER])
    ax.grid(axis="y", color="#d7d2c8", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Target label fraction")
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig: plt.Figure, name: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_ROOT / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def direction_title(source: str, target: str) -> str:
    return f"{SITE_SHORT[source]}->{SITE_SHORT[target]}  ({SITE_LABEL[source]} -> {SITE_LABEL[target]})"


def plot_epoch_trajectories(df: pd.DataFrame) -> None:
    directions = [
        ("pohang", "utah_2019"),
        ("utah_2019", "pohang"),
        ("pohang", "utah_2023"),
        ("utah_2023", "pohang"),
        ("utah_2019", "utah_2023"),
        ("utah_2023", "utah_2019"),
    ]
    metrics = [
        ("val_dist_gap_event_minus_noise", "Validation center gap"),
        ("val_event_noise_swd", "Validation latent SWD"),
    ]
    for metric, label in metrics:
        if metric not in df.columns:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.2), sharex=False, sharey=False)
        for ax, (source, target) in zip(axes.ravel(), directions):
            ddf = df[(df["source_site"] == source) & (df["target_site"] == target)]
            for tag in FRACTION_ORDER:
                sub = ddf[ddf["fraction_tag"] == tag].sort_values("epoch")
                if sub.empty:
                    continue
                ax.plot(
                    sub["epoch"],
                    sub[metric],
                    marker="o",
                    markersize=4,
                    linewidth=2.0,
                    color=FRACTION_COLORS[tag],
                    label=FRACTION_LABEL[tag],
                )
            ax.set_title(direction_title(source, target), fontsize=9.5, pad=8)
            setup_epoch_axis(ax, label if ax in axes[:, 0] else None)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=5,
            frameon=False,
            title="Label fraction",
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(f"Cross-site latent diagnostics by epoch: {label}", y=1.0, fontsize=14)
        fig.tight_layout(rect=[0, 0.08, 1, 0.955])
        save(fig, f"cross_site_epoch_trajectory_{metric}")


def plot_final_by_fraction(final: pd.DataFrame) -> None:
    metrics = [
        ("val_dist_gap_event_minus_noise", "Final validation center gap"),
        ("val_event_noise_swd", "Final validation latent SWD"),
        ("val_dist_ratio_event_over_noise", "Final event/noise center-distance ratio"),
    ]
    directions = [
        ("pohang", "utah_2019"),
        ("utah_2019", "pohang"),
        ("pohang", "utah_2023"),
        ("utah_2023", "pohang"),
        ("utah_2019", "utah_2023"),
        ("utah_2023", "utah_2019"),
    ]
    colors = {
        "pohang_to_utah_2019": "#376795",
        "utah_2019_to_pohang": "#78a5c9",
        "pohang_to_utah_2023": "#bf4b3e",
        "utah_2023_to_pohang": "#e3a09a",
        "utah_2019_to_utah_2023": "#d17b0f",
        "utah_2023_to_utah_2019": "#4c956c",
    }
    for metric, label in metrics:
        if metric not in final.columns:
            continue
        fig, ax = plt.subplots(figsize=(9.2, 5.0))
        for source, target in directions:
            direction = f"{source}_to_{target}"
            sub = final[final["direction"] == direction].sort_values("fraction")
            if sub.empty:
                continue
            ax.plot(
                sub["fraction"],
                sub[metric],
                marker="o",
                linewidth=2.2,
                markersize=5,
                color=colors[direction],
                label=f"{SITE_SHORT[source]}->{SITE_SHORT[target]}",
            )
        setup_fraction_axis(ax, label)
        ax.set_title(label, fontsize=13, pad=12)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Direction")
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        save(fig, f"cross_site_final_by_fraction_{metric}")


def plot_final_matrix(final: pd.DataFrame) -> None:
    metrics = [
        ("val_dist_gap_event_minus_noise", "Center gap"),
        ("val_event_noise_swd", "Latent SWD"),
    ]
    available_tags = [
        tag
        for tag in FRACTION_ORDER
        if tag in set(final["fraction_tag"].dropna().astype(str))
    ]
    for metric, label in metrics:
        if metric not in final.columns or not available_tags:
            continue
        fig, axes = plt.subplots(
            1,
            len(available_tags),
            figsize=(3.25 * len(available_tags), 3.85),
            squeeze=False,
        )
        values = final[metric].replace([np.inf, -np.inf], np.nan).dropna()
        vmax = float(values.quantile(0.95)) if not values.empty else 1.0
        vmin = float(values.quantile(0.05)) if not values.empty else 0.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        image = None
        for ax, tag in zip(axes.ravel(), available_tags):
            mat = np.full((len(SITES), len(SITES)), np.nan)
            for r, source in enumerate(SITES):
                for c, target in enumerate(SITES):
                    if source == target:
                        continue
                    row = final[
                        (final["fraction_tag"] == tag)
                        & (final["source_site"] == source)
                        & (final["target_site"] == target)
                    ]
                    if not row.empty:
                        mat[r, c] = float(row.iloc[0][metric])
            image = ax.imshow(mat, cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"frac={FRACTION_LABEL[tag]}", fontsize=10, pad=8)
            ax.set_xticks(range(len(SITES)))
            ax.set_yticks(range(len(SITES)))
            ax.set_xticklabels([SITE_SHORT[s] for s in SITES])
            ax.set_yticklabels([SITE_SHORT[s] for s in SITES])
            ax.set_xlabel("Target")
            if ax is axes.ravel()[0]:
                ax.set_ylabel("Source")
            for r in range(len(SITES)):
                for c in range(len(SITES)):
                    if np.isfinite(mat[r, c]):
                        ax.text(c, r, f"{mat[r, c]:.2g}", ha="center", va="center", color="white", fontsize=8)
                    elif r == c:
                        ax.text(c, r, "-", ha="center", va="center", color="#333333", fontsize=10)
        if image is not None:
            cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.026, pad=0.014)
            cbar.set_label(label)
        fig.suptitle(f"Cross-site final latent diagnostic matrix: {label}", y=1.04, fontsize=13)
        fig.tight_layout(rect=[0, 0, 0.98, 0.94])
        save(fig, f"cross_site_final_matrix_{metric}")


def plot_gap_vs_swd(final: pd.DataFrame) -> None:
    required = {"val_dist_gap_event_minus_noise", "val_event_noise_swd"}
    if not required.issubset(final.columns):
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for tag in FRACTION_ORDER:
        sub = final[final["fraction_tag"] == tag]
        if sub.empty:
            continue
        ax.scatter(
            sub["val_dist_gap_event_minus_noise"],
            sub["val_event_noise_swd"],
            s=58,
            color=FRACTION_COLORS[tag],
            label=FRACTION_LABEL[tag],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{SITE_SHORT[row['source_site']]}->{SITE_SHORT[row['target_site']]}",
                (row["val_dist_gap_event_minus_noise"], row["val_event_noise_swd"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7.5,
                color="#333333",
            )
    ax.grid(color="#d7d2c8", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Final validation center gap")
    ax.set_ylabel("Final validation latent SWD")
    ax.set_title("Final center gap vs latent SWD", fontsize=13, pad=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Label fraction")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    save(fig, "cross_site_final_gap_vs_swd")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = collect_center_history()
    if df.empty:
        raise SystemExit("No completed center_history.csv files found.")
    final = final_epoch(df)
    df.to_csv(OUT_ROOT / "center_history_all_epochs.csv", index=False)
    final.to_csv(OUT_ROOT / "center_history_final_epoch.csv", index=False)
    plot_epoch_trajectories(df)
    plot_final_by_fraction(final)
    plot_final_matrix(final)
    plot_gap_vs_swd(final)
    print(f"[DONE] center histories: {df[['direction','fraction_tag']].drop_duplicates().shape[0]}")
    print(f"[DONE] rows: {len(df)}")
    print(f"[DONE] wrote center figures to {OUT_ROOT}")


if __name__ == "__main__":
    main()
