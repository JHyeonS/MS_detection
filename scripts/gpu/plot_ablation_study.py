#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "ablation_study_assets_v1"


def _load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def _ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison() -> pd.DataFrame:
    pohang = _load_csv("runs/pohang_main_study/summary.csv").copy()
    utah = _load_csv("runs/utah_2019_main_study/summary.csv").copy()

    pohang = pohang[pohang["method"].isin(["scratch", "reconst", "contrast", "reconst_noanom"])]
    utah = utah[utah["method"].isin(["scratch", "reconst", "contrast", "reconst_noanom"])]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    style = {
        "scratch": ("#4c72b0", "o"),
        "reconst": ("#dd8452", "s"),
        "reconst_noanom": ("#55a868", "^"),
        "contrast": ("#c44e52", "D"),
    }

    for ax, df, title, metric in [
        (axes[0, 0], pohang, "Pohang", "test_balanced_acc"),
        (axes[0, 1], pohang, "Pohang", "test_f1"),
        (axes[1, 0], utah, "Utah 2019", "test_balanced_acc"),
        (axes[1, 1], utah, "Utah 2019", "test_f1"),
    ]:
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method].sort_values("fraction")
            color, marker = style[method]
            ax.plot(
                sub["fraction"],
                sub[metric],
                marker=marker,
                color=color,
                linewidth=2,
                markersize=6,
                label=method,
            )
        ax.set_title(f"{title} - {'Balanced Accuracy' if metric == 'test_balanced_acc' else 'F1'}")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.25)
        ax.set_ylim(0.0, 1.05)
        fracs = sorted(df["fraction"].unique())
        ax.set_xticks(fracs)
        ax.set_xticklabels([f"{x:.2f}" for x in fracs])

    for ax in axes[-1]:
        ax.set_xlabel("Label fraction")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if "reconst_noanom" not in set(utah["method"].unique()):
        handles.append(Line2D([0], [0], color=style["reconst_noanom"][0], marker=style["reconst_noanom"][1], linestyle="--"))
        labels.append("reconst_noanom (N/A in Utah 2019)")
        note = "reconst_noanom not run"
        axes[1, 0].text(0.98, 0.10, note, transform=axes[1, 0].transAxes, ha="right", va="bottom", fontsize=8, color="#666666")
        axes[1, 1].text(0.98, 0.10, note, transform=axes[1, 1].transAxes, ha="right", va="bottom", fontsize=8, color="#666666")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    _save(fig, "ablation_model_comparison")

    combined = pd.concat(
        [
            pohang.assign(site="Pohang")[["site", "method", "fraction", "test_f1", "test_balanced_acc", "test_specificity"]],
            utah.assign(site="Utah 2019")[["site", "method", "fraction", "test_f1", "test_balanced_acc", "test_specificity"]],
        ],
        ignore_index=True,
    ).sort_values(["site", "method", "fraction"])
    combined.to_csv(OUT_DIR / "ablation_model_comparison_table.csv", index=False)
    return combined


def plot_preprocessing_ablation() -> pd.DataFrame:
    pohang_base = _load_csv("runs/pohang_main_study/summary.csv")
    pohang_base = pohang_base[pohang_base["method"] == "reconst"].copy()
    pohang_base["variant"] = "baseline"
    pohang_none = _load_csv("runs/pohang_normalization_ablation_v2/bandpass_agc_none/summary.csv").copy()
    pohang_none["variant"] = "bandpass_agc_none"
    pohang_robust = _load_csv("runs/pohang_normalization_ablation_v2/bandpass_agc_robust/summary.csv").copy()
    pohang_robust["variant"] = "bandpass_agc_robust"

    u19_base = _load_csv("runs/utah_2019_main_study/summary.csv")
    u19_base = u19_base[u19_base["method"] == "reconst"].copy()
    u19_base["variant"] = "baseline"
    u19_none = _load_csv("runs/utah_2019_normalization_ablation_v2/bandpass_agc_none/summary.csv").copy()
    u19_none["variant"] = "bandpass_agc_none"
    u19_robust = _load_csv("runs/utah_2019_normalization_ablation_v2/bandpass_agc_robust/summary.csv").copy()
    u19_robust["variant"] = "bandpass_agc_robust"

    u23_base = _load_csv("runs/utah_2023_main_study/summary.csv").copy()
    u23_base["variant"] = "baseline"
    u23_none = _load_csv("runs/utah_2023_normalization_ablation_v2/bandpass_agc_none/summary.csv").copy()
    u23_none["variant"] = "bandpass_agc_none"
    u23_robust = _load_csv("runs/utah_2023_normalization_ablation_v2/bandpass_agc_robust/summary.csv").copy()
    u23_robust["variant"] = "bandpass_agc_robust"

    rows = []
    for site, dfs in [
        ("Pohang", [pohang_base, pohang_none, pohang_robust]),
        ("Utah 2019", [u19_base, u19_none, u19_robust]),
        ("Utah 2023", [u23_base, u23_none, u23_robust]),
    ]:
        for df in dfs:
            keep = ["fraction", "variant", "test_f1", "test_balanced_acc", "test_specificity"]
            tmp = df[keep].copy()
            tmp["site"] = site
            rows.append(tmp)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(OUT_DIR / "ablation_preprocessing_table.csv", index=False)

    palette = {
        "baseline": "#4c72b0",
        "bandpass_agc_none": "#55a868",
        "bandpass_agc_robust": "#c44e52",
    }
    markers = {"baseline": "o", "bandpass_agc_none": "s", "bandpass_agc_robust": "D"}
    label_map = {"baseline": "base", "bandpass_agc_none": "AGC", "bandpass_agc_robust": "AGC+R"}
    variant_order = ["baseline", "bandpass_agc_none", "bandpass_agc_robust"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=False)
    site_order = ["Pohang", "Utah 2019", "Utah 2023"]
    for i, site in enumerate(site_order):
        site_df = combined[combined["site"] == site]
        for variant in variant_order:
            if variant not in site_df["variant"].unique():
                continue
            sub = site_df[site_df["variant"] == variant].sort_values("fraction")
            axes[i, 0].plot(
                sub["fraction"],
                sub["test_balanced_acc"],
                marker=markers[variant],
                color=palette[variant],
                linewidth=2.6,
                markersize=7,
                zorder=4 if variant == "baseline" else 3,
                label=variant,
            )
            axes[i, 1].plot(
                sub["fraction"],
                sub["test_specificity"],
                marker=markers[variant],
                color=palette[variant],
                linewidth=2.6,
                markersize=7,
                zorder=4 if variant == "baseline" else 3,
                label=variant,
            )
            last = sub.iloc[-1]
            axes[i, 0].annotate(
                label_map[variant],
                (last["fraction"], last["test_balanced_acc"]),
                xytext=(6, 0),
                textcoords="offset points",
                color=palette[variant],
                fontsize=8,
                va="center",
                fontweight="bold",
            )
            axes[i, 1].annotate(
                label_map[variant],
                (last["fraction"], last["test_specificity"]),
                xytext=(6, 0),
                textcoords="offset points",
                color=palette[variant],
                fontsize=8,
                va="center",
                fontweight="bold",
            )
        axes[i, 0].set_title(f"{site} - Balanced Accuracy")
        axes[i, 1].set_title(f"{site} - Specificity")
        axes[i, 0].set_ylabel("Score")
        axes[i, 1].set_ylabel("Score")
        axes[i, 0].set_ylim(0.0, 1.05)
        axes[i, 1].set_ylim(0.0, 1.05)
        axes[i, 0].grid(alpha=0.25)
        axes[i, 1].grid(alpha=0.25)
        axes[i, 0].set_xlabel("Label fraction")
        axes[i, 1].set_xlabel("Label fraction")
        axes[i, 0].set_xlim(0.03, 1.08)
        axes[i, 1].set_xlim(0.03, 1.08)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    _save(fig, "ablation_preprocessing")
    return combined


def plot_utah2023_failure() -> pd.DataFrame:
    perf = []

    baseline = _load_csv("runs/utah_2023_main_study/summary.csv").copy()
    baseline = baseline[baseline["method"] == "reconst"].copy()
    baseline["variant"] = "baseline"
    none = _load_csv("runs/utah_2023_normalization_ablation_v2/bandpass_agc_none/summary.csv").copy()
    none["variant"] = "bandpass_agc_none"
    robust = _load_csv("runs/utah_2023_normalization_ablation_v2/bandpass_agc_robust/summary.csv").copy()
    robust["variant"] = "bandpass_agc_robust"

    for df in [baseline, none, robust]:
        keep = ["fraction", "variant", "test_f1", "test_balanced_acc", "test_specificity"]
        perf.append(df[keep].copy())
    perf_df = pd.concat(perf, ignore_index=True)

    wdf = _load_csv("runs/utah_2023_wasserstein_offline_v1/summary.csv").copy()
    mapping = {"bandpass_agc_none": "bandpass_agc_none", "bandpass_agc_robust": "bandpass_agc_robust"}
    wdf["variant"] = wdf["variant"].map(mapping).fillna(wdf["variant"])
    frac_map = {}
    for run_name in wdf["run_name"]:
        tag = run_name.split("__frac")[-1]
        frac = tag.replace("p", ".")
        frac_map[run_name] = float(frac)
    wdf["fraction"] = wdf["run_name"].map(frac_map)

    baseline_w = wdf[wdf["variant"] == "baseline"] if "baseline" in wdf["variant"].unique() else pd.DataFrame()
    if baseline_w.empty:
        # Derive baseline offline metrics from known main-study rows by merging the offline summary rows
        baseline_w = wdf[wdf["checkpoint"].str.contains("utah_2023_main_study", na=False)].copy()
        baseline_w["variant"] = "baseline"

    latent = wdf[["variant", "fraction", "test_event_noise_swd", "test_dist_gap_event_minus_noise"]].copy()
    latent.to_csv(OUT_DIR / "ablation_utah2023_latent_table.csv", index=False)
    perf_df.to_csv(OUT_DIR / "ablation_utah2023_performance_table.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    palette = {"baseline": "#4c72b0", "bandpass_agc_none": "#55a868", "bandpass_agc_robust": "#c44e52"}
    markers = {"baseline": "o", "bandpass_agc_none": "s", "bandpass_agc_robust": "D"}

    for variant in perf_df["variant"].unique():
        sub = perf_df[perf_df["variant"] == variant].sort_values("fraction")
        axes[0, 0].plot(sub["fraction"], sub["test_f1"], marker=markers[variant], color=palette[variant], linewidth=2, label=variant)
        axes[0, 1].plot(sub["fraction"], sub["test_balanced_acc"], marker=markers[variant], color=palette[variant], linewidth=2, label=variant)
        axes[1, 0].plot(sub["fraction"], sub["test_specificity"], marker=markers[variant], color=palette[variant], linewidth=2, label=variant)

    for variant in latent["variant"].unique():
        sub = latent[latent["variant"] == variant].sort_values("fraction")
        axes[1, 1].plot(sub["fraction"], sub["test_event_noise_swd"], marker=markers[variant], color=palette[variant], linewidth=2, label=variant)

    axes[0, 0].set_title("Utah 2023 - F1")
    axes[0, 1].set_title("Utah 2023 - Balanced Accuracy")
    axes[1, 0].set_title("Utah 2023 - Specificity")
    axes[1, 1].set_title("Utah 2023 - Latent SWD")
    for ax in axes.flat:
        ax.set_xlabel("Label fraction")
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(perf_df["fraction"].unique()))
        ax.set_xticklabels([f"{x:.2f}" for x in sorted(perf_df["fraction"].unique())])
    axes[0, 0].set_ylabel("Score")
    axes[0, 1].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[1, 1].set_ylabel("Distance")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[1, 0].set_ylim(0.0, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    _save(fig, "ablation_utah2023_failure")

    return pd.merge(perf_df, latent, on=["variant", "fraction"], how="left")


def write_latex_draft(model_df: pd.DataFrame, prep_df: pd.DataFrame, u23_df: pd.DataFrame) -> None:
    best_u23 = u23_df.sort_values(["test_balanced_acc", "test_specificity"], ascending=False).iloc[0]
    latex = rf"""
\subsection{{Ablation on Architecture, Loss, and Preprocessing}}
\label{{sec:ablation}}

We structured the ablation study around two practical questions: which model design choices materially changed detection behavior, and which preprocessing choices governed transfer stability across sites. Because the encoder backbone had already been selected through architecture HPO, the ablation focus shifted to candidate training objectives and preprocessing variants.

\subsubsection{{Candidate model comparison}}
We first compared the candidate training formulations under the same encoder family. In Pohang, the reconstruction-pretrained model remained the strongest overall configuration, especially once the label fraction exceeded 0.25. By contrast, the contrastive pretraining variant consistently underperformed the reconstruction-based models and frequently collapsed toward event-heavy predictions at low label fractions. The anomaly-free variant (reconst\_noanom) improved over contrastive pretraining, but still did not match the full reconstruction-pretrained detector in the moderate- and high-label regimes.

In Utah 2019, the ranking was less stable. Scratch, reconstruction pretraining, and contrastive pretraining all produced modest performance, and no candidate yielded a uniformly dominant curve across all fractions. This instability is itself informative: changing the pretraining objective alone was insufficient to guarantee recovery in the shifted Utah 2019 domain.

\subsubsection{{Preprocessing and normalization ablation}}
We then evaluated preprocessing alignment using three settings: the baseline preprocessing used in the main reconstruction-pretrained pipeline, bandpass filtering with AGC and no additional normalization, and bandpass filtering with AGC followed by robust normalization. The outcome was strongly site-dependent. In Pohang, AGC without additional robust normalization produced the clearest gains in the low-label regime, including a balanced accuracy of 0.7345 at 5\% labels and 0.9758 at 25\% labels. In Utah 2019, however, robust normalization after AGC became beneficial once sufficient labels were available, improving balanced accuracy from 0.6145 to 0.7754 at 50\% labels and from 0.8933 to 0.9173 at full supervision. Utah 2023 did not exhibit the same recovery: both AGC-based variants remained near chance-level balanced accuracy, despite F1 values near 0.73.

\subsubsection{{Failure-aware loss interpretation in Utah 2023}}
Utah 2023 required a separate failure-aware interpretation. The baseline reconstruction-pretrained detector produced F1 scores around 0.73--0.74, but specificity stayed close to zero and balanced accuracy remained near chance level. This indicates that the model was not learning a reliable event--noise boundary, but was instead converging to an event-heavy solution. The strongest Utah 2023 setting in our ablation-related diagnostics still achieved only a balanced accuracy of {best_u23['test_balanced_acc']:.4f} with specificity {best_u23['test_specificity']:.4f}, reinforcing that objective and preprocessing changes alone were insufficient to recover a robust detector in this domain.

\begin{{table}}[t]
\centering
\caption{{Representative preprocessing ablation results used in the main discussion.}}
\label{{tab:ablation_preprocessing_main}}
\begin{{tabular}}{{llcccc}}
\toprule
Site & Setting & Fraction & F1 & Balanced acc. & Specificity \\
\midrule
Pohang & AGC + no normalization & 0.05 & 0.7577 & 0.7345 & 0.5037 \\
Pohang & AGC + robust normalization & 0.05 & 0.7322 & 0.7029 & 0.4667 \\
Pohang & AGC + no normalization & 0.25 & 0.9739 & 0.9758 & 0.9778 \\
Utah 2019 & AGC + no normalization & 0.50 & 0.4314 & 0.6145 & 0.6500 \\
Utah 2019 & AGC + robust normalization & 0.50 & 0.6500 & 0.7754 & 0.8667 \\
Utah 2019 & AGC + no normalization & 1.00 & 0.8348 & 0.8933 & 0.9444 \\
Utah 2019 & AGC + robust normalization & 1.00 & 0.8455 & 0.9173 & 0.9222 \\
Utah 2023 & AGC + no normalization & 1.00 & 0.7323 & 0.5013 & 0.0026 \\
Utah 2023 & AGC + robust normalization & 1.00 & 0.7318 & 0.5000 & 0.0000 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering
\caption{{Representative candidate-model comparison results.}}
\label{{tab:ablation_model_main}}
\begin{{tabular}}{{llcccc}}
\toprule
Site & Method & Fraction & F1 & Balanced acc. & Specificity \\
\midrule
Pohang & contrast & 0.50 & 0.5891 & 0.5596 & 0.4148 \\
Pohang & reconst\_noanom & 0.50 & 0.8476 & 0.8647 & 0.9556 \\
Pohang & reconst & 0.50 & 0.9913 & 0.9919 & 0.9926 \\
Utah 2019 & scratch & 0.50 & 0.5289 & 0.6918 & 0.8222 \\
Utah 2019 & contrast & 0.50 & 0.4769 & 0.6553 & 0.7667 \\
Utah 2019 & reconst & 0.50 & 0.3974 & 0.5830 & 0.6222 \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (ROOT / "ablation_study_draft.tex").write_text(latex.strip() + "\n", encoding="utf-8")


def main() -> None:
    _ensure_out()
    model_df = plot_model_comparison()
    prep_df = plot_preprocessing_ablation()
    u23_df = plot_utah2023_failure()
    write_latex_draft(model_df, prep_df, u23_df)


if __name__ == "__main__":
    main()
