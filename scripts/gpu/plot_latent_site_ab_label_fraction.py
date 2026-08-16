#!/home/ted1204/.conda/envs/ms_detection/bin/python
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "latent_site_ab_label_fraction_v1"


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(OUT_DIR / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


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


def _scatter(ax, df: pd.DataFrame, title: str) -> None:
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


def build_latent_umap() -> None:
    panels = [
        (
            "runs/pohang_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/pohang__frac0p1/tsne/encoder_features.pkl",
            "Site A (Pohang), 10%",
        ),
        (
            "runs/pohang_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/pohang__frac0p5/tsne/encoder_features.pkl",
            "Site A (Pohang), 50%",
        ),
        (
            "runs/pohang_normalization_ablation_v2/bandpass_agc_none/reconst/finetune/pohang__frac1/tsne/encoder_features.pkl",
            "Site A (Pohang), 100%",
        ),
        (
            "runs/utah_2019_preprocess_study/bandpass_agc/reconst/finetune/base_utah_2019__frac0p1/tsne/encoder_features.pkl",
            "Site B (Utah 2019), 10% (AGC+robust)",
        ),
        (
            "runs/utah_2019_preprocess_study/bandpass_agc/reconst/finetune/base_utah_2019__frac0p5/tsne/encoder_features.pkl",
            "Site B (Utah 2019), 50% (AGC+robust)",
        ),
        (
            "runs/utah_2019_preprocess_study/bandpass_agc/reconst/finetune/base_utah_2019__frac1/tsne/encoder_features.pkl",
            "Site B (Utah 2019), 100% (AGC+robust)",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    for ax, (path, title) in zip(axes.flat, panels):
        _scatter(ax, _compute_umap(_load_features(path)), title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    _save(fig, "latent_umap_site_ab_label_fraction")


def build_latent_diagnostics() -> None:
    diag = _load_csv("runs/in_domain_center_dynamics_v1/in_domain_center_dynamics_summary_table.csv")
    diag = diag.copy()
    diag["fraction_pct"] = diag["fraction"] * 100

    colors = {"Pohang": "#4c72b0", "Utah 2019": "#dd8452"}
    markers = {"Pohang": "o", "Utah 2019": "s"}
    metrics = [
        ("val_dist_gap_event_minus_noise", "Final validation center gap"),
        ("val_event_noise_swd", "Final validation latent SWD"),
        ("test_balanced_acc", "Test balanced accuracy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, (col, title) in zip(axes, metrics):
        for site in ["Pohang", "Utah 2019"]:
            sub = diag[diag["site"] == site].sort_values("fraction_pct")
            ax.plot(
                sub["fraction_pct"],
                sub[col],
                marker=markers[site],
                linewidth=2.2,
                markersize=6,
                color=colors[site],
                label=site,
            )
        ax.set_title(title)
        ax.set_xlabel("Label fraction (%)")
        ax.set_xticks([10, 50, 100])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    _save(fig, "latent_diagnostics_site_ab_label_fraction")

    diag.to_csv(OUT_DIR / "latent_diagnostics_site_ab_label_fraction_table.csv", index=False)


def write_note() -> None:
    note = """Figures produced:

1. latent_umap_site_ab_label_fraction
- Representative in-domain latent UMAP panels for Site A (Pohang) and Site B (Utah 2019)
- Fractions shown: 10%, 50%, 100%
- Important note: each panel uses an independently fitted UMAP embedding; compare cluster overlap and separation rather than axis values

2. latent_diagnostics_site_ab_label_fraction
- Final-epoch validation center gap versus label fraction
- Final-epoch validation latent SWD versus label fraction
- Test balanced accuracy versus label fraction
- Representative settings:
  * Site A: AGC-no-norm
  * Site B: AGC+robust
"""
    (OUT_DIR / "README.txt").write_text(note, encoding="utf-8")


def main() -> None:
    _ensure()
    build_latent_umap()
    build_latent_diagnostics()
    write_note()


if __name__ == "__main__":
    main()
