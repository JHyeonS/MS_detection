#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "metadata_v2_safe_rerun_v1"
SWD_CSV = (
    ROOT
    / "temp"
    / "current_results_summary"
    / "figures_metadata_v2"
    / "center"
    / "cross_site_classwise_swd"
    / "cross_site_classwise_swd.csv"
)
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "writing_followup"

SITE_EXP = {
    "pohang": "pohang",
    "utah_2019": "base_utah_2019",
    "utah_2023": "base_utah_2023",
}
SITE_LABEL = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
PREPROC_ROOT = {
    "filter_rms": RUN_ROOT / "filter_rms_site_main_pre50_v2",
    "logenv": RUN_ROOT / "logenv_site_main_pre50_v2",
}
PREPROC_LABEL = {
    "filter_rms": "Low-pass + RMS",
    "logenv": "Log-envelope",
}
COLORS = {
    "filter_rms": "#bf4b3e",
    "logenv": "#376795",
}


def frac_tag(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def read_balacc(path: Path) -> float:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data["fc_metrics_fixed_threshold"]["balanced_acc"])


def scratch_metric_path(preproc: str, site: str, fraction: float) -> Path:
    exp = SITE_EXP[site]
    tag = frac_tag(fraction)
    return PREPROC_ROOT[preproc] / f"{site}_scratch" / "scratch" / "test" / f"{exp}__frac{tag}" / "test_metrics_fixed_threshold.json"


def build_dataframe() -> pd.DataFrame:
    df = pd.read_csv(SWD_CSV)
    df = df[df["preprocessing"].isin(["filter_rms", "logenv"])].copy()
    rows = []
    for _, row in df.iterrows():
        preproc = str(row["preprocessing"])
        target = str(row["target_site"])
        fraction = float(row["fraction"])
        scratch_path = scratch_metric_path(preproc, target, fraction)
        if not scratch_path.exists():
            continue
        scratch_balacc = read_balacc(scratch_path)
        cross_balacc = float(row["target_balanced_acc"])
        out = row.to_dict()
        out["scratch_balanced_acc"] = scratch_balacc
        out["transfer_gain_balacc"] = cross_balacc - scratch_balacc
        out["preprocessing_label"] = PREPROC_LABEL[preproc]
        out["target_site_label"] = SITE_LABEL[target]
        out["direction_label"] = f"{SITE_LABEL[str(row['source_site'])]} -> {SITE_LABEL[target]}"
        rows.append(out)
    return pd.DataFrame(rows)


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def annotate_corr(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return
    corr = float(np.corrcoef(x[ok], y[ok])[0, 1])
    ax.text(
        0.04,
        0.95,
        f"Pearson r = {corr:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )


def style_axis(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.65)
    ax.grid(color="#d7d2c8", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_scatter(df: pd.DataFrame, x_col: str, x_label: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for preproc in ["filter_rms", "logenv"]:
        sub = df[df["preprocessing"].eq(preproc)]
        ax.scatter(
            sub[x_col],
            sub["transfer_gain_balacc"],
            s=54,
            color=COLORS[preproc],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.86,
            label=PREPROC_LABEL[preproc],
        )
    x = df[x_col].to_numpy(float)
    y = df["transfer_gain_balacc"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 3:
        coef = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(float(np.nanmin(x[ok])), float(np.nanmax(x[ok])), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#111827", linewidth=1.4, alpha=0.7, label="Linear trend")
    annotate_corr(ax, x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Transfer gain in balanced accuracy\n(cross-site reconst - target scratch)")
    ax.set_title("Latent Mismatch vs. Transfer Gain", fontsize=12.5, pad=11)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, name)


def plot_faceted_by_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)
    for ax, target in zip(axes, ["pohang", "utah_2019", "utah_2023"]):
        sub_target = df[df["target_site"].eq(target)]
        for preproc in ["filter_rms", "logenv"]:
            sub = sub_target[sub_target["preprocessing"].eq(preproc)]
            ax.scatter(
                sub["event_site_swd"],
                sub["transfer_gain_balacc"],
                s=48,
                color=COLORS[preproc],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.86,
                label=PREPROC_LABEL[preproc],
            )
        annotate_corr(
            ax,
            sub_target["event_site_swd"].to_numpy(float),
            sub_target["transfer_gain_balacc"].to_numpy(float),
        )
        ax.set_title(f"Target: {SITE_LABEL[target]}", fontsize=11.2, pad=9)
        ax.set_xlabel("Event-site SWD")
        ax.set_ylabel("Transfer gain (BalAcc)" if ax is axes[0] else "")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.1)
    save(fig, "latent_mismatch_event_swd_vs_transfer_gain_by_target")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataframe()
    df.to_csv(OUT_DIR / "latent_mismatch_vs_transfer_gain.csv", index=False)
    plot_scatter(df, "event_site_swd", "Event-site SWD", "latent_mismatch_event_swd_vs_transfer_gain")
    plot_scatter(df, "noise_site_swd", "Noise-site SWD", "latent_mismatch_noise_swd_vs_transfer_gain")
    plot_scatter(df, "all_site_swd", "All-sample site SWD", "latent_mismatch_all_swd_vs_transfer_gain")
    plot_faceted_by_target(df)
    print(f"[DONE] wrote scatter plots to {OUT_DIR}")
    print(df[["preprocessing_label", "direction_label", "fraction", "event_site_swd", "transfer_gain_balacc"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
