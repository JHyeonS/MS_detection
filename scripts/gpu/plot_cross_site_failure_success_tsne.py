#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.gpu.plot_cross_site_failure_success_umap as umap_script


OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "leftwing"


def compute_tsne(df: pd.DataFrame) -> pd.DataFrame:
    x = np.stack(df["feat"].to_numpy()).astype(np.float32)
    perplexity = min(35, max(5, (len(df) - 1) // 5))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
        metric="euclidean",
    )
    emb = reducer.fit_transform(x)
    out = df.drop(columns=["feat"]).copy()
    out["tsne_x"] = emb[:, 0]
    out["tsne_y"] = emb[:, 1]
    return out


def style_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_panel(ax: plt.Axes, df: pd.DataFrame, title: str, metrics: dict, show_legend: bool = False) -> None:
    colors = {0: "#5f6875", 1: "#d97706"}
    labels = {0: "Noise", 1: "Event"}
    for label in (0, 1):
        sub = df[df["label"].astype(int) == label]
        ax.scatter(
            sub["tsne_x"],
            sub["tsne_y"],
            s=13,
            alpha=0.66,
            color=colors[label],
            label=labels[label],
            linewidths=0,
        )
    wrong = df[~df["is_correct"]]
    if not wrong.empty:
        ax.scatter(
            wrong["tsne_x"],
            wrong["tsne_y"],
            s=28,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.7,
            label="Misclassified",
        )
    ax.set_title(title.replace(": ", "\n"), fontsize=10.2, fontweight="normal", pad=6, color="#111827")
    ax.set_xlabel(
        f"BalAcc {metrics['balanced_acc']:.3f}  |  Spec {metrics['specificity']:.3f}",
        fontsize=8.4,
        color="#6b7280",
        labelpad=8,
    )
    style_axis(ax)
    if show_legend:
        ax.legend(frameon=False, loc="lower left", fontsize=8.4, handletextpad=0.25)


def plot_panel_classic(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    colors = {0: "#1f5fbf", 1: "#e85d04"}
    labels = {0: "Noise", 1: "Event"}
    for lab in [0, 1]:
        sub = df[df["label"].astype(int) == lab]
        ax.scatter(
            sub["tsne_x"],
            sub["tsne_y"],
            s=8,
            alpha=0.55,
            c=colors[lab],
            label=labels[lab],
            linewidths=0,
        )
    ax.set_title(title, fontsize=11, fontweight="normal")
    ax.set_xlabel("t-SNE 1", fontsize=9.5)
    ax.set_ylabel("t-SNE 2", fontsize=9.5)
    ax.grid(alpha=0.18)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    embedded = []
    for run in umap_script.RUNS:
        features = umap_script.extract_features(run)
        emb = compute_tsne(features)
        emb["preprocessing_panel"] = run["panel_title"]
        embedded.append(emb)
    all_emb = pd.concat(embedded, ignore_index=True)
    all_emb.to_csv(OUT_DIR / "cross_site_failure_success_tsne_utah2019_to_utah2023_frac0p25.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.15))
    for ax, run, emb in zip(axes, umap_script.RUNS, embedded):
        metrics = umap_script.metric_summary(run["run_dir"])
        plot_panel(ax, emb, run["panel_title"], metrics)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=8.6,
        handletextpad=0.25,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=[0.0, 0.105, 1.0, 1.0], w_pad=0.65)
    out_base = OUT_DIR / "cross_site_failure_success_tsne_utah2019_to_utah2023_frac0p25"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    classic_titles = [
        "Failure: Low-pass + RMS",
        "Recovery: Log-envelope",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, emb, title in zip(axes, embedded, classic_titles):
        plot_panel_classic(ax, emb, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    out_base = OUT_DIR / "cross_site_failure_success_tsne_classic_utah2019_to_utah2023_frac0p25"
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=240)
    plt.close(fig)

    for emb, title, run in zip(embedded, classic_titles, umap_script.RUNS):
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.25))
        plot_panel_classic(ax, emb, title)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
        out_base = OUT_DIR / f"cross_site_{run['name']}_tsne_classic_utah2019_to_utah2023_frac0p25"
        for ext in ("pdf", "png"):
            fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=240)
        plt.close(fig)

    for run, emb in zip(umap_script.RUNS, embedded):
        fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.25))
        plot_panel(ax, emb, run["panel_title"], umap_script.metric_summary(run["run_dir"]), show_legend=True)
        fig.tight_layout()
        out_base = OUT_DIR / f"cross_site_{run['name']}_tsne_utah2019_to_utah2023_frac0p25"
        for ext in ("pdf", "png"):
            fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"[DONE] wrote {OUT_DIR / 'cross_site_failure_success_tsne_utah2019_to_utah2023_frac0p25.pdf'}")
    print(f"[DONE] wrote {OUT_DIR / 'cross_site_failure_success_tsne_utah2019_to_utah2023_frac0p25.png'}")


if __name__ == "__main__":
    main()
