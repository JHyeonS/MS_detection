#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2"

DATASETS = {
    "raw": ROOT / "data" / "visualbest_raw_rms_fs1000_rms0p15_nofilter",
    "filter_rms": ROOT / "data" / "visualbest_filter_rms_fs1000_rms0p15_lp50",
    "logenv": ROOT
    / "data"
    / "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5",
}
SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
CHANNEL_SPACING_M = {
    "pohang": 2.0,
    "utah_2019": 3.35,
    "utah_2023": 1.0249,
}


def robust_norm(arr: np.ndarray) -> np.ndarray:
    scale = float(np.nanpercentile(np.abs(arr), 99.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(arr / scale, -1.0, 1.0)


def image_limits(arr: np.ndarray, symmetric: bool = True) -> tuple[float, float]:
    if symmetric:
        lim = float(np.nanpercentile(np.abs(arr), 99.0))
        if not np.isfinite(lim) or lim <= 0:
            lim = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
        return -lim, lim
    vmax = float(np.nanpercentile(arr, 99.0))
    vmin = float(np.nanpercentile(arr, 1.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    return vmin, vmax


def sample_path(dataset: str, site: str, sample_id: str) -> Path:
    return DATASETS[dataset] / site / "1_event" / f"{sample_id}.npy"


def load_event_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(DATASETS["filter_rms"] / "metadata" / "all_samples.csv")
    manifest = manifest[(manifest["site"].isin(SITES)) & (manifest["label"] == 1)].copy()
    manifest["sample_id"] = manifest["npy_path"].map(lambda x: Path(str(x)).stem)
    return manifest


def score_sample(site: str, sample_id: str) -> float | None:
    paths = {name: sample_path(name, site, sample_id) for name in DATASETS}
    if not all(path.exists() for path in paths.values()):
        return None
    raw = np.load(paths["raw"])
    filt = np.load(paths["filter_rms"])
    logenv = np.load(paths["logenv"])
    if raw.shape != filt.shape or raw.shape != logenv.shape:
        return None
    raw_n = robust_norm(raw)
    filt_n = robust_norm(filt)
    log_n = robust_norm(logenv)
    raw_filter_delta = float(np.mean(np.abs(raw_n - filt_n)))
    logenv_sparsity = float(np.nanpercentile(log_n, 99.0) - np.nanpercentile(log_n, 50.0))
    event_strength = float(np.nanpercentile(np.abs(filt_n), 99.5))
    return raw_filter_delta + 0.25 * logenv_sparsity + 0.10 * event_strength


def select_sample(max_per_site: int = 40) -> dict:
    manifest = load_event_manifest()
    scored = []
    for site in SITES:
        site_df = manifest[manifest["site"] == site].head(max_per_site)
        for _, row in site_df.iterrows():
            sample_id = str(row["sample_id"])
            score = score_sample(site, sample_id)
            if score is None:
                continue
            scored.append(
                {
                    "site": site,
                    "sample_id": sample_id,
                    "score": score,
                    "group_id": row.get("group_id", ""),
                    "start_sec": row.get("start_sec", np.nan),
                    "end_sec": row.get("end_sec", np.nan),
                    "ch_start": row.get("ch_start", np.nan),
                    "ch_end": row.get("ch_end", np.nan),
                }
            )
    if not scored:
        raise RuntimeError("No common event sample found across raw/filter/logenv datasets.")
    return max(scored, key=lambda row: row["score"])


def add_arrow(fig: plt.Figure, x0: float, x1: float, y: float, text: str) -> None:
    arrow = FancyArrowPatch(
        (x0, y),
        (x1, y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#2f3437",
        connectionstyle="arc3,rad=0",
    )
    fig.add_artist(arrow)
    fig.text(
        (x0 + x1) / 2,
        y - 0.035,
        text,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#2f3437",
        linespacing=1.18,
    )


def plot_pipeline() -> None:
    selected = select_sample()
    site = selected["site"]
    sample_id = selected["sample_id"]
    arrays = {
        "Raw DAS window": np.load(sample_path("raw", site, sample_id)),
        "Low-pass filtered": np.load(sample_path("filter_rms", site, sample_id)),
        "Log-envelope scaled": np.load(sample_path("logenv", site, sample_id)),
    }
    n_channels, n_samples = next(iter(arrays.values())).shape
    duration_sec = n_samples / 1000.0
    aperture_m = (n_channels - 1) * CHANNEL_SPACING_M[site]

    fig = plt.figure(figsize=(14.5, 7.1), facecolor="white")
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[4.8, 1.05, 1.25],
        hspace=0.36,
        wspace=0.18,
        left=0.055,
        right=0.965,
        top=0.90,
        bottom=0.08,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    shared_abs_lim = max(image_limits(arr, symmetric=True)[1] for arr in arrays.values())
    cmaps = ["RdBu_r", "RdBu_r", "RdBu_r"]

    for ax, (title, arr), cmap in zip(axes, arrays.items(), cmaps):
        vmin, vmax = -shared_abs_lim, shared_abs_lim
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0.0, duration_sec, 0.0, aperture_m],
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Time (s)")
        if ax is axes[0]:
            ax.set_ylabel("Relative fiber distance (m)")
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.018)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Amplitude (a.u.)", fontsize=8.5)

    fig.text(
        0.055,
        0.965,
        f"Preprocessing pipeline visualization ({SITE_LABELS[site]} event window)",
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.965,
        0.965,
        f"sample={sample_id}",
        fontsize=9,
        color="#5b6268",
        ha="right",
        va="top",
    )

    pos = [ax.get_position() for ax in axes]
    arrow_y = pos[0].y0 - 0.055
    add_arrow(
        fig,
        pos[0].x1 + 0.012,
        pos[1].x0 - 0.012,
        arrow_y,
        "Low-pass filtering\nsuppresses high-frequency components",
    )
    add_arrow(
        fig,
        pos[1].x1 + 0.012,
        pos[2].x0 - 0.012,
        arrow_y,
        "Envelope + log compression\nemphasizes amplitude structure",
    )

    text_ax = fig.add_subplot(gs[2, :])
    text_ax.axis("off")
    formula = (
        r"$\tilde{x}(t,c)=\mathrm{LPF}_{f_c}\{x(t,c)\}$"
        "\n"
        r"$x_{\mathrm{logenv}}(t,c)=\log\left(1+\left|\mathcal{H}(\tilde{x}(t,c))\right|\right)$"
        "\n"
        r"$x_{\mathrm{model}}(t,c)=\alpha_{\mathrm{site}}\;x_{\mathrm{processed}}(t,c)$"
    )
    text_ax.text(
        0.5,
        0.52,
        formula,
        ha="center",
        va="center",
        fontsize=16,
        bbox={
            "boxstyle": "round,pad=0.55,rounding_size=0.08",
            "facecolor": "#f5f0e8",
            "edgecolor": "#d2c4b0",
            "linewidth": 1.1,
        },
        linespacing=1.65,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / "methodology_preprocessing_pipeline_raw_filter_logenv"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    pd.DataFrame([selected]).to_csv(out_base.with_suffix(".csv"), index=False)
    print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
    print(f"[DONE] wrote {out_base.with_suffix('.png')}")
    print(f"[DONE] wrote {out_base.with_suffix('.csv')}")


def plot_clean_panels() -> None:
    csv_path = OUT_DIR / "methodology_preprocessing_pipeline_raw_filter_logenv.csv"
    if csv_path.exists():
        selected = pd.read_csv(csv_path).iloc[0].to_dict()
    else:
        selected = select_sample()
    site = str(selected["site"])
    sample_id = str(selected["sample_id"]).zfill(7)
    panel_specs = [
        ("raw", "raw"),
        ("filter_rms", "lowpass_filter"),
        ("logenv", "logenv_scaling"),
    ]
    n_channels, n_samples = np.load(sample_path("raw", site, sample_id)).shape
    duration_sec = n_samples / 1000.0
    aperture_m = (n_channels - 1) * CHANNEL_SPACING_M[site]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {dataset_key: np.load(sample_path(dataset_key, site, sample_id)) for dataset_key, _ in panel_specs}
    shared_abs_lim = max(image_limits(arr, symmetric=True)[1] for arr in arrays.values())
    for dataset_key, out_name in panel_specs:
        arr = arrays[dataset_key]
        vmin, vmax = -shared_abs_lim, shared_abs_lim
        fig, ax = plt.subplots(figsize=(4.8, 3.35), facecolor="white")
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[0.0, duration_sec, 0.0, aperture_m],
            interpolation="nearest",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("")
        fig.tight_layout()
        out_base = OUT_DIR / f"methodology_preprocessing_clean_{out_name}"
        for ext in ("pdf", "png"):
            fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"[DONE] wrote {out_base.with_suffix('.pdf')}")
        print(f"[DONE] wrote {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    plot_pipeline()
    plot_clean_panels()
