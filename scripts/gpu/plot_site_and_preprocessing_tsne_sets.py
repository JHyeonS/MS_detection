#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "site_preprocessing_tsne_sets_v1"


def _load(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(OUT_DIR / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _scatter(ax, df: pd.DataFrame, title: str) -> None:
    colors = {0: "#4c72b0", 1: "#dd8452"}
    labels = {0: "Noise", 1: "Event"}
    for lab in [0, 1]:
        sub = df[df["label"] == lab]
        ax.scatter(sub["tsne_x"], sub["tsne_y"], s=7, alpha=0.55, c=colors[lab], label=labels[lab], linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def plot_site_fraction_sets() -> None:
    panels = [
        (
            "runs/pohang_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/pohang__frac0p05/tsne/tsne_encoder_events_noises.csv",
            "Site A (Pohang), 5%",
        ),
        (
            "runs/pohang_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/pohang__frac0p1/tsne/tsne_encoder_events_noises.csv",
            "Site A (Pohang), 10%",
        ),
        (
            "runs/pohang_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/pohang__frac0p25/tsne/tsne_encoder_events_noises.csv",
            "Site A (Pohang), 25%",
        ),
        (
            "runs/utah_2019_preprocess_study/bandpass_agc/reconst/finetune/base_utah_2019__frac0p05/tsne/tsne_encoder_events_noises.csv",
            "Site B (Utah 2019), 5%",
        ),
        (
            "runs/utah_2019_preprocess_study/bandpass_agc/reconst/finetune/base_utah_2019__frac0p1/tsne/tsne_encoder_events_noises.csv",
            "Site B (Utah 2019), 10%",
        ),
        (
            "runs/utah_2019_preprocess_study/bandpass_agc/reconst/finetune/base_utah_2019__frac1/tsne/tsne_encoder_events_noises.csv",
            "Site B (Utah 2019), 100%",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    for ax, (path, title) in zip(axes.flat, panels):
        _scatter(ax, _load(path), title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    _save(fig, "site_fraction_tsne")


def plot_preprocessing_sets() -> None:
    panels = [
        (
            "runs/pohang_main_study/reconst/finetune/pohang__frac0p1/tsne/tsne_encoder_events_noises.csv",
            "Pohang baseline, 10%",
        ),
        (
            "runs/pohang_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/pohang__frac0p1/tsne/tsne_encoder_events_noises.csv",
            "Pohang AGC-no-norm, 10%",
        ),
        (
            "runs/utah_2019_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/base_utah_2019__frac1/tsne/tsne_encoder_events_noises.csv",
            "Utah 2019 AGC-no-norm, 100%",
        ),
        (
            "runs/utah_2019_normalization_ablation_v2/bandpass_agc_robust/reconst/finetune/base_utah_2019__frac1/tsne/tsne_encoder_events_noises.csv",
            "Utah 2019 AGC+robust, 100%",
        ),
        (
            "runs/utah_2023_main_study/reconst/finetune/base_utah_2023__frac0p25/tsne/tsne_encoder_events_noises.csv",
            "Utah 2023 baseline, 25%",
        ),
        (
            "runs/utah_2023_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/base_utah_2023__frac0p5/tsne/tsne_encoder_events_noises.csv",
            "Utah 2023 AGC-no-norm, 50%",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    for ax, (path, title) in zip(axes.flat, panels):
        _scatter(ax, _load(path), title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    _save(fig, "preprocessing_tsne")


def write_note() -> None:
    note = """Figures produced:

Important note:
- Each panel uses an independently fitted t-SNE embedding.
- The axis values are therefore not directly comparable across panels.
- Use these figures to compare cluster overlap, separation, and geometry.

1. site_fraction_tsne
- Site A (Pohang): 5%, 10%, 25%
- Site B (Utah 2019): 5%, 10%, 100%

2. preprocessing_tsne
- Pohang baseline vs AGC-no-norm at 10%
- Utah 2019 AGC-no-norm vs AGC+robust at 100%
- Utah 2023 baseline vs AGC-no-norm as failure-case reference
"""
    (OUT_DIR / "README.txt").write_text(note, encoding="utf-8")


def main() -> None:
    _ensure()
    plot_site_fraction_sets()
    plot_preprocessing_sets()
    write_note()


if __name__ == "__main__":
    main()
