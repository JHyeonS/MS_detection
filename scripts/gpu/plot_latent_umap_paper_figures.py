#!/home/ted1204/.conda/envs/ms_detection/bin/python
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "latent_umap_paper_figures_v1"


def _load_features(rel: str) -> pd.DataFrame:
    with open(ROOT / rel, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {rel}, got {type(df)}")
    return df


def _compute_umap(df: pd.DataFrame) -> pd.DataFrame:
    feats = np.stack(df["feat"].to_numpy())
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(30, max(5, len(df) - 1)),
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    emb = reducer.fit_transform(feats)
    out = df.copy()
    out["umap_x"] = emb[:, 0]
    out["umap_y"] = emb[:, 1]
    return out


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(OUT_DIR / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_umap(ax, df: pd.DataFrame, title: str) -> None:
    colors = {0: "#4c72b0", 1: "#dd8452"}
    labels = {0: "Noise", 1: "Event"}
    for lab in [0, 1]:
        sub = df[df["label"] == lab]
        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=8,
            alpha=0.55,
            c=colors[lab],
            label=labels[lab],
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.18)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def build_recoverable_domains() -> None:
    panels = [
        (
            "runs/pohang_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/pohang__frac0p1/tsne/encoder_features.pkl",
            "Site A (Pohang), AGC-no-norm, 10% labels",
        ),
        (
            "runs/utah_2019_preprocess_study/bandpass_agc/reconst/finetune/base_utah_2019__frac0p1/tsne/encoder_features.pkl",
            "Site B (Utah 2019), AGC, 10% labels",
        ),
        (
            "runs/utah_2019_preprocess_study/bandpass_agc/reconst/finetune/base_utah_2019__frac1/tsne/encoder_features.pkl",
            "Site B (Utah 2019), AGC, 100% labels",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax, (path, title) in zip(axes, panels):
        df = _compute_umap(_load_features(path))
        _plot_umap(ax, df, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    _save(fig, "recoverable_domains_umap")


def build_site_c_failure() -> None:
    panels = [
        (
            "runs/utah_2023_main_study/reconst/finetune/base_utah_2023__frac0p25/tsne/encoder_features.pkl",
            "Site C (Utah 2023), baseline, 25% labels",
        ),
        (
            "runs/utah_2023_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/base_utah_2023__frac0p5/tsne/encoder_features.pkl",
            "Site C (Utah 2023), AGC-no-norm, 50% labels",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, (path, title) in zip(axes, panels):
        df = _compute_umap(_load_features(path))
        _plot_umap(ax, df, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    _save(fig, "site_c_failure_umap")


def write_note() -> None:
    text = """These UMAP figures are intended as qualitative support for the latent-dynamics analysis.

Important note:
- Each panel is generated from a separately fitted UMAP embedding.
- Absolute axis coordinates are therefore not comparable across panels.
- Interpret cluster separation, overlap, and geometry rather than numeric axis values.

recoverable_domains_umap:
- Site A (Pohang) at 10% labels under AGC-no-norm
- Site B (Utah 2019) at 10% labels under AGC
- Site B (Utah 2019) at 100% labels under AGC

site_c_failure_umap:
- Site C (Utah 2023) baseline at 25% labels
- Site C (Utah 2023) AGC-no-norm at 50% labels
"""
    (OUT_DIR / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    _ensure()
    build_recoverable_domains()
    build_site_c_failure()
    write_note()


if __name__ == "__main__":
    main()
