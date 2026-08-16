#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "fixed_center": {"label": "Fixed center", "color": "#D95F02", "linestyle": "-"},
    "dynamic_center": {"label": "Dynamic center", "color": "#1B9E77", "linestyle": "-"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot center movement and distance diagnostics from finetune center_history.csv files."
    )
    parser.add_argument(
        "--fixed-root",
        default=None,
        help="Root containing fixed-center finetune runs, e.g. runs/pohang_center_diagnostics/fixed_center.",
    )
    parser.add_argument(
        "--dynamic-root",
        default=None,
        help="Root containing dynamic-center finetune runs, e.g. runs/pohang_center_diagnostics/dynamic_center.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for figures.",
    )
    parser.add_argument("--site-title", default="Pohang")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fraction_from_experiment(name: str) -> float | None:
    match = re.search(r"__frac([0-9p]+)$", name)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def read_histories(root: str | None, method: str) -> pd.DataFrame:
    if not root:
        return pd.DataFrame()
    root_path = Path(root)
    rows = []
    for path in sorted(root_path.glob("finetune/*/center_history.csv")):
        experiment = path.parent.name
        fraction = fraction_from_experiment(experiment)
        if fraction is None:
            continue
        df = pd.read_csv(path)
        df["method"] = method
        df["fraction"] = fraction
        df["experiment"] = experiment
        df["center_history_path"] = str(path)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def save_figure(fig, out_base: Path, formats: list[str], dpi: int):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def format_fraction(x: float) -> str:
    if abs(x - 1.0) < 1e-9:
        return "1.0"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def plot_by_fraction(df: pd.DataFrame, y_col: str, y_label: str, title: str, out_base: Path, formats: list[str], dpi: int):
    fractions = sorted(df["fraction"].dropna().unique())
    if not fractions:
        raise ValueError("No fraction values found for plotting.")
    n = len(fractions)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.85), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, fraction in zip(axes, fractions):
        sub_frac = df[df["fraction"] == fraction]
        for method in ["fixed_center", "dynamic_center"]:
            sub = sub_frac[sub_frac["method"] == method].sort_values("epoch")
            if sub.empty or y_col not in sub.columns:
                continue
            style = STYLE[method]
            ax.plot(
                sub["epoch"],
                sub[y_col],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.9,
            )
        ax.set_title(f"Labeled fraction = {format_fraction(fraction)}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    save_figure(fig, out_base, formats, dpi)
    plt.close(fig)


def make_final_scatter(df: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int, site_title: str):
    last_rows = []
    for (_, fraction, method), sub in df.groupby(["experiment", "fraction", "method"]):
        last_rows.append(sub.sort_values("epoch").iloc[-1])
    final = pd.DataFrame(last_rows)
    final.to_csv(out_dir / "center_diagnostics_final_epoch.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    for method in ["fixed_center", "dynamic_center"]:
        sub = final[final["method"] == method].sort_values("fraction")
        if sub.empty:
            continue
        style = STYLE[method]
        axes[0].plot(
            sub["fraction"],
            sub["center_delta_from_initial"],
            marker="o",
            color=style["color"],
            label=style["label"],
            linewidth=2.0,
        )
        axes[1].plot(
            sub["fraction"],
            sub["val_dist_gap_event_minus_noise"],
            marker="o",
            color=style["color"],
            label=style["label"],
            linewidth=2.0,
        )
    for ax in axes:
        ax.set_xlabel("Labeled fraction")
        ax.set_xticks(sorted(final["fraction"].unique()))
        ax.set_xticklabels([format_fraction(x) for x in sorted(final["fraction"].unique())])
    axes[0].set_title("(a) Final center movement")
    axes[0].set_ylabel(r"$||c_T - c_0||_2$")
    axes[1].set_title("(b) Final event-noise distance gap")
    axes[1].set_ylabel("Mean event distance - mean noise distance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Center Diagnostics on {site_title}")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    save_figure(fig, out_dir / "center_diagnostics_final_by_fraction", formats, dpi)
    plt.close(fig)


def make_wasserstein_final(df: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int, site_title: str):
    if "val_event_noise_swd" not in df.columns:
        return
    last_rows = []
    for (_, fraction, method), sub in df.groupby(["experiment", "fraction", "method"]):
        last_rows.append(sub.sort_values("epoch").iloc[-1])
    final = pd.DataFrame(last_rows)
    final = final[final["val_event_noise_swd"].notna()].copy()
    if final.empty:
        return

    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for method in ["fixed_center", "dynamic_center"]:
        sub = final[final["method"] == method].sort_values("fraction")
        if sub.empty:
            continue
        style = STYLE[method]
        ax.plot(
            sub["fraction"],
            sub["val_event_noise_swd"],
            marker="o",
            color=style["color"],
            label=style["label"],
            linewidth=2.0,
        )
    ax.set_xlabel("Labeled fraction")
    ax.set_xticks(sorted(final["fraction"].unique()))
    ax.set_xticklabels([format_fraction(x) for x in sorted(final["fraction"].unique())])
    ax.set_ylabel("Validation event-noise SWD")
    ax.set_title(f"Sliced Wasserstein on {site_title}")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir / "center_wasserstein_final_by_fraction", formats, dpi)
    plt.close(fig)


def main():
    args = parse_args()
    setup_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]

    df = pd.concat(
        [
            read_histories(args.fixed_root, "fixed_center"),
            read_histories(args.dynamic_root, "dynamic_center"),
        ],
        ignore_index=True,
    )
    if df.empty:
        raise ValueError("No center_history.csv files found.")
    df.to_csv(out_dir / "center_diagnostics_all_epochs.csv", index=False)

    plot_by_fraction(
        df,
        y_col="center_delta_from_initial",
        y_label=r"$||c_t - c_0||_2$",
        title=f"Center Movement on {args.site_title}",
        out_base=out_dir / "center_movement_by_epoch",
        formats=formats,
        dpi=args.dpi,
    )
    plot_by_fraction(
        df,
        y_col="val_dist_gap_event_minus_noise",
        y_label="Mean event distance - mean noise distance",
        title=f"Validation Distance Separation on {args.site_title}",
        out_base=out_dir / "center_distance_gap_by_epoch",
        formats=formats,
        dpi=args.dpi,
    )
    plot_by_fraction(
        df,
        y_col="val_dist_ratio_event_over_noise",
        y_label="Mean event distance / mean noise distance",
        title=f"Validation Distance Ratio on {args.site_title}",
        out_base=out_dir / "center_distance_ratio_by_epoch",
        formats=formats,
        dpi=args.dpi,
    )
    if "val_event_noise_swd" in df.columns:
        plot_by_fraction(
            df,
            y_col="val_event_noise_swd",
            y_label="Validation event-noise SWD",
            title=f"Validation Sliced Wasserstein on {args.site_title}",
            out_base=out_dir / "center_wasserstein_by_epoch",
            formats=formats,
            dpi=args.dpi,
        )
    make_final_scatter(df, out_dir, formats, args.dpi, args.site_title)
    make_wasserstein_final(df, out_dir, formats, args.dpi, args.site_title)

    print(f"[DONE] wrote center diagnostic figures to {out_dir}")


if __name__ == "__main__":
    main()
