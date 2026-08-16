#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detection.dataset.preprocessing import agc_filter, bandpass_filter, remove_mean


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "event_filter_examples_v1"
METADATA_CSV = ROOT / "data" / "0406" / "metadata" / "all_samples.csv"

SITES = ["pohang", "utah_2019", "utah_2023"]
N_EXAMPLES = 4
FS = 1000.0
BANDPASS_LOW = 3.0
BANDPASS_HIGH = 50.0
BANDPASS_ORDER = 4
AGC_WINDOW_SEC = 0.2
AGC_CLIP = 10.0


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_event_examples(site: str, n_examples: int = N_EXAMPLES) -> pd.DataFrame:
    df = pd.read_csv(METADATA_CSV)
    sub = df[(df["site"] == site) & (df["label_name"] == "event")].copy().reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No event rows found for site={site}")
    if len(sub) <= n_examples:
        return sub
    idx = np.linspace(0, len(sub) - 1, n_examples, dtype=int)
    return sub.iloc[idx].reset_index(drop=True)


def _prepare_variants(x: np.ndarray) -> dict[str, np.ndarray]:
    x = x.astype(np.float32)
    if x.ndim == 3:
        x = x[0]
    raw = x
    detrended = remove_mean(x)
    band = bandpass_filter(detrended, fs=FS, fmin=BANDPASS_LOW, fmax=BANDPASS_HIGH, order=BANDPASS_ORDER)
    band_agc = agc_filter(band, fs=FS, window_sec=AGC_WINDOW_SEC, clip=AGC_CLIP)
    return {
        "Raw": raw,
        "Bandpass (3–50 Hz)": band,
        "Bandpass + AGC": band_agc,
    }


def _display_limits(arr: np.ndarray) -> tuple[float, float]:
    vmax = float(np.percentile(np.abs(arr), 99.5))
    vmax = max(vmax, 1e-6)
    return -vmax, vmax


def _site_title(site: str) -> str:
    return {
        "pohang": "Pohang",
        "utah_2019": "Utah 2019",
        "utah_2023": "Utah 2023",
    }[site]


def plot_site(site: str) -> None:
    examples = _load_event_examples(site)
    fig, axes = plt.subplots(
        len(examples),
        3,
        figsize=(10.8, 2.45 * len(examples)),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    if len(examples) == 1:
        axes = np.array([axes])

    for row_idx, (_, row) in enumerate(examples.iterrows()):
        x = np.load(row["npy_path"])
        variants = _prepare_variants(x)
        sample_label = f"#{int(row['sample_index_within_class']):04d}"
        time0 = float(row.get("start_sec", 0.0))
        time1 = float(row.get("end_sec", 2.0))
        ch0 = int(row.get("ch_start", 0))
        ch1 = int(row.get("ch_end", x.shape[0]))
        extent = [time0, time1, ch1, ch0]

        for col_idx, (name, arr) in enumerate(variants.items()):
            ax = axes[row_idx, col_idx]
            vmin, vmax = _display_limits(arr)
            im = ax.imshow(
                arr,
                cmap="seismic",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            if row_idx == 0:
                ax.set_title(name, fontsize=11, pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{sample_label}\nChannel", fontsize=10)
            if row_idx == len(examples) - 1:
                ax.set_xlabel("Time (s)", fontsize=10)
            ax.tick_params(labelsize=8)

            cbar = fig.colorbar(im, ax=ax, fraction=0.027, pad=0.015)
            cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"{_site_title(site)} Event Examples", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    out_base = OUT_DIR / f"{site}_event_filter_examples"
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_site_single_examples(site: str) -> None:
    examples = _load_event_examples(site)
    for example_idx, (_, row) in enumerate(examples.iterrows(), start=1):
        x = np.load(row["npy_path"])
        variants = _prepare_variants(x)
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharex=True, sharey=True)

        sample_label = f"sample {example_idx} / idx {int(row['sample_index_within_class']):04d}"
        time0 = float(row.get("start_sec", 0.0))
        time1 = float(row.get("end_sec", 2.0))
        ch0 = int(row.get("ch_start", 0))
        ch1 = int(row.get("ch_end", x.shape[0]))
        extent = [time0, time1, ch1, ch0]

        for col_idx, (name, arr) in enumerate(variants.items()):
            ax = axes[col_idx]
            vmin, vmax = _display_limits(arr)
            im = ax.imshow(
                arr,
                cmap="seismic",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            ax.set_title(name, fontsize=11, pad=8)
            ax.set_xlabel("Time (s)", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("Channel", fontsize=10)
            ax.tick_params(labelsize=8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)

        fig.suptitle(f"{_site_title(site)} Event {sample_label}", fontsize=14, y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        out_base = OUT_DIR / f"{site}_event_filter_example_{example_idx:02d}"
        fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_combined_sheet() -> None:
    rows = []
    for site in SITES:
        rows.append(_load_event_examples(site, n_examples=1).iloc[0])

    fig, axes = plt.subplots(len(rows), 3, figsize=(10.8, 6.8), sharex=True, sharey=False)
    for row_idx, row in enumerate(rows):
        site = str(row["site"])
        x = np.load(row["npy_path"])
        variants = _prepare_variants(x)
        time0 = float(row.get("start_sec", 0.0))
        time1 = float(row.get("end_sec", 2.0))
        ch0 = int(row.get("ch_start", 0))
        ch1 = int(row.get("ch_end", x.shape[0]))
        extent = [time0, time1, ch1, ch0]
        for col_idx, (name, arr) in enumerate(variants.items()):
            ax = axes[row_idx, col_idx]
            vmin, vmax = _display_limits(arr)
            im = ax.imshow(
                arr,
                cmap="seismic",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            if row_idx == 0:
                ax.set_title(name, fontsize=11, pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{_site_title(site)}\nChannel", fontsize=10)
            if row_idx == len(rows) - 1:
                ax.set_xlabel("Time (s)", fontsize=10)
            ax.tick_params(labelsize=8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
            cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Representative Event Windows Across Sites", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    out_base = OUT_DIR / "combined_event_filter_examples"
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _ensure()
    for site in SITES:
        plot_site(site)
        plot_site_single_examples(site)
    plot_combined_sheet()
    readme = (
        "Event filter examples\n"
        f"Sites: {', '.join(SITES)}\n"
        f"Variants: Raw, Bandpass ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz), Bandpass + AGC\n"
        f"AGC window: {AGC_WINDOW_SEC}s, clip={AGC_CLIP}\n"
    )
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
