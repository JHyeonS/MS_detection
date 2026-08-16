#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


SITE_ORDER = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
SITE_COLORS = {
    "pohang": "#1f77b4",
    "utah_2019": "#ff7f0e",
    "utah_2023": "#2ca02c",
}
CLASS_LABELS = {
    "noise": "Non-event background PSD by site",
    "event": "Event-window PSD by site",
}
CLASS_COLORS = {
    "noise": "#1f77b4",
    "event": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot site-wise PSD summaries for event and noise windows.")
    parser.add_argument(
        "--metadata",
        default="data/0406/metadata/all_samples.csv",
        help="Path to all_samples.csv metadata.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/site_psd_by_class_v1",
        help="Output directory for figures and tables.",
    )
    parser.add_argument(
        "--max-samples-per-site-class",
        type=int,
        default=200,
        help="Maximum number of windows to sample for each site/class combination.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for window sampling.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=1000.0,
        help="Sampling rate in Hz.",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=256,
        help="Welch segment length.",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=100.0,
        help="Maximum frequency to display in plots.",
    )
    return parser.parse_args()


def average_window_psd(arr: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    freq, pxx = welch(arr, fs=fs, axis=-1, nperseg=min(nperseg, arr.shape[-1]), noverlap=min(nperseg, arr.shape[-1]) // 2)
    return freq, pxx.mean(axis=0)


def sample_group(df: pd.DataFrame, site: str, label_name: str, max_samples: int, seed: int) -> pd.DataFrame:
    group = df[(df["site"] == site) & (df["label_name"] == label_name)].copy()
    if len(group) <= max_samples:
        return group.sort_values("global_index").reset_index(drop=True)
    return group.sample(n=max_samples, random_state=seed).sort_values("global_index").reset_index(drop=True)


def collect_psd_curves(df: pd.DataFrame, fs: float, nperseg: int, max_samples: int, seed: int) -> tuple[dict, pd.DataFrame]:
    psd_data: dict[str, dict[str, dict[str, np.ndarray]]] = {c: {} for c in CLASS_LABELS}
    rows = []
    for label_name in CLASS_LABELS:
        for site in SITE_ORDER:
            sampled = sample_group(df, site, label_name, max_samples=max_samples, seed=seed)
            curves = []
            freq = None
            for _, row in sampled.iterrows():
                arr = np.load(row["npy_path"])
                freq, curve = average_window_psd(arr, fs=fs, nperseg=nperseg)
                curves.append(curve)
            curves_arr = np.vstack(curves)
            psd_data[label_name][site] = {
                "freq": freq,
                "curves": curves_arr,
                "mean": curves_arr.mean(axis=0),
                "q25": np.percentile(curves_arr, 25, axis=0),
                "q75": np.percentile(curves_arr, 75, axis=0),
            }
            rows.append(
                {
                    "site": site,
                    "class": label_name,
                    "available_windows": int(((df["site"] == site) & (df["label_name"] == label_name)).sum()),
                    "sampled_windows": int(len(sampled)),
                    "fs_hz": fs,
                    "nperseg": nperseg,
                }
            )
    return psd_data, pd.DataFrame(rows)


def plot_single_class(psd_data: dict, label_name: str, out_dir: Path, max_freq: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for site in SITE_ORDER:
        payload = psd_data[label_name][site]
        mask = payload["freq"] <= max_freq
        freq = payload["freq"][mask]
        mean = payload["mean"][mask]
        q25 = payload["q25"][mask]
        q75 = payload["q75"][mask]
        color = SITE_COLORS[site]
        ax.plot(freq, mean, lw=2.2, color=color, label=SITE_LABELS[site])
    ax.set_yscale("log")
    ax.set_xlim(0, max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(CLASS_LABELS[label_name])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    stem = f"{label_name}_psd_by_site"
    fig.savefig(out_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined(psd_data: dict, out_dir: Path, max_freq: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharex=True)
    for ax, label_name in zip(axes, ["noise", "event"]):
        for site in SITE_ORDER:
            payload = psd_data[label_name][site]
            mask = payload["freq"] <= max_freq
            freq = payload["freq"][mask]
            mean = payload["mean"][mask]
            q25 = payload["q25"][mask]
            q75 = payload["q75"][mask]
            color = SITE_COLORS[site]
            ax.plot(freq, mean, lw=2.2, color=color, label=SITE_LABELS[site])
        ax.set_yscale("log")
        ax.set_xlim(0, max_freq)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(CLASS_LABELS[label_name])
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("PSD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_dir / "site_psd_by_class_combined.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "site_psd_by_class_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sitewise_event_vs_noise(psd_data: dict, out_dir: Path, max_freq: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), sharex=True, sharey=True)
    for ax, site in zip(axes, SITE_ORDER):
        for label_name in ["noise", "event"]:
            payload = psd_data[label_name][site]
            mask = payload["freq"] <= max_freq
            freq = payload["freq"][mask]
            mean = payload["mean"][mask]
            q25 = payload["q25"][mask]
            q75 = payload["q75"][mask]
            color = CLASS_COLORS[label_name]
            label = "Non-event background" if label_name == "noise" else "Event window"
            ax.plot(freq, mean, lw=2.2, color=color, label=label)
        ax.set_yscale("log")
        ax.set_xlim(0, max_freq)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(SITE_LABELS[site])
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("PSD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_dir / "sitewise_event_vs_background_psd.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "sitewise_event_vs_background_psd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    df = df[df["label_name"].isin(["noise", "event"])].copy()

    psd_data, summary_df = collect_psd_curves(
        df=df,
        fs=args.fs,
        nperseg=args.nperseg,
        max_samples=args.max_samples_per_site_class,
        seed=args.seed,
    )
    summary_df.to_csv(out_dir / "site_psd_sampling_summary.csv", index=False)

    for label_name in ["noise", "event"]:
        plot_single_class(psd_data, label_name, out_dir, args.max_freq)
    plot_combined(psd_data, out_dir, args.max_freq)
    plot_sitewise_event_vs_noise(psd_data, out_dir, args.max_freq)

    readme = out_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Site-wise PSD summaries",
                f"- metadata: {args.metadata}",
                f"- sampled windows per site/class: up to {args.max_samples_per_site_class}",
                f"- sampling seed: {args.seed}",
                f"- Welch fs: {args.fs} Hz",
                f"- Welch nperseg: {args.nperseg}",
                f"- plotted frequency range: 0-{args.max_freq} Hz",
                "- per-window PSD was computed by averaging Welch PSDs across the 405 DAS channels.",
                "- curves show the mean PSD across sampled windows; no variability shading is drawn.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
