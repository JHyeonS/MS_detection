#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "remaining_results_sections_v1"


def _load(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_failure_recovery_table() -> pd.DataFrame:
    p2u = _load("runs/pair_pohang_utah2019_parallel_v3/cross_pohang_to_utah_2019/summary.csv")
    p2u["setting"] = "A→B"
    u2p = _load("runs/pair_pohang_utah2019_parallel_v3/cross_utah_2019_to_pohang/summary.csv")
    u2p["setting"] = "B→A"
    mixed = _load("runs/pair_pohang_utah2019_parallel_v3/mixed_pohang_utah_2019/summary.csv")
    mixed["setting"] = "Mixed A+B"
    df = pd.concat([p2u, u2p, mixed], ignore_index=True)
    df = df[
        [
            "setting",
            "method",
            "test_f1",
            "test_balanced_acc",
            "test_specificity",
            "test_precision",
            "test_recall",
            "test_tp",
            "test_tn",
            "test_fp",
            "test_fn",
        ]
    ].sort_values(["setting", "method"])
    df.to_csv(OUT_DIR / "failure_recovery_site_ab_table.csv", index=False)
    return df


def plot_failure_recovery(table: pd.DataFrame) -> None:
    settings = ["A→B", "B→A", "Mixed A+B"]
    x = list(range(len(settings)))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    for ax, metric, title in [
        (axes[0], "test_balanced_acc", "Balanced Accuracy gain"),
        (axes[1], "test_f1", "F1 gain"),
        (axes[2], "test_specificity", "Specificity gain"),
    ]:
        gains = []
        labels = []
        for s in settings:
            scratch_row = table[(table["setting"] == s) & (table["method"] == "scratch")].iloc[0]
            reconst_row = table[(table["setting"] == s) & (table["method"] == "reconst")].iloc[0]
            gains.append(reconst_row[metric] - scratch_row[metric])
            labels.append(f"{scratch_row[metric]:.2f}→{reconst_row[metric]:.2f}")
        ax.bar(x, gains, width=0.55, color="#8172b3")
        ax.axhline(0.0, color="#666666", linewidth=1)
        for xpos, gain, lab in zip(x, gains, labels):
            ax.text(xpos, gain + (0.015 if gain >= 0 else -0.03), lab, ha="center", va="bottom" if gain >= 0 else "top", fontsize=8, color="#444444")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(settings)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Reconst - Scratch")
    _save(fig, "failure_recovery_site_ab")


def build_utah2023_failure_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    perf_main = _load("runs/utah_2023_main_study/summary.csv")
    perf_main = perf_main[perf_main["method"] == "reconst"].copy()
    perf_main["variant"] = "baseline"

    none = _load("runs/utah_2023_normalization_ablation_v2/bandpass_agc_none/summary.csv")
    none["variant"] = "bandpass_agc_none"
    robust = _load("runs/utah_2023_normalization_ablation_v2/bandpass_agc_robust/summary.csv")
    robust["variant"] = "bandpass_agc_robust"
    perf = pd.concat([perf_main, none, robust], ignore_index=True)
    perf = perf[
        ["variant", "fraction", "test_f1", "test_balanced_acc", "test_specificity", "test_precision", "test_recall"]
    ].sort_values(["variant", "fraction"])
    perf.to_csv(OUT_DIR / "site_c_failure_performance_table.csv", index=False)

    latent = _load("runs/utah_2023_wasserstein_offline_v1/summary.csv")
    latent = latent[
        ["variant", "run_name", "test_event_noise_swd", "test_dist_gap_event_minus_noise"]
    ].copy()
    latent["fraction"] = latent["run_name"].str.split("__frac").str[-1].str.replace("p", ".", regex=False).astype(float)
    latent = latent[["variant", "fraction", "test_event_noise_swd", "test_dist_gap_event_minus_noise"]].sort_values(
        ["variant", "fraction"]
    )
    latent.to_csv(OUT_DIR / "site_c_failure_latent_table.csv", index=False)
    return perf, latent


def plot_utah2023_failure(perf: pd.DataFrame, latent: pd.DataFrame) -> None:
    colors = {"baseline": "#4c72b0", "bandpass_agc_none": "#55a868", "bandpass_agc_robust": "#c44e52"}
    markers = {"baseline": "o", "bandpass_agc_none": "s", "bandpass_agc_robust": "D"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for variant in perf["variant"].unique():
        sub = perf[perf["variant"] == variant].sort_values("fraction")
        axes[0, 0].plot(sub["fraction"], sub["test_f1"], color=colors[variant], marker=markers[variant], linewidth=2, label=variant)
        axes[0, 1].plot(
            sub["fraction"], sub["test_balanced_acc"], color=colors[variant], marker=markers[variant], linewidth=2, label=variant
        )
        axes[1, 0].plot(
            sub["fraction"], sub["test_specificity"], color=colors[variant], marker=markers[variant], linewidth=2, label=variant
        )
    for variant in latent["variant"].unique():
        sub = latent[latent["variant"] == variant].sort_values("fraction")
        axes[1, 1].plot(
            sub["fraction"], sub["test_event_noise_swd"], color=colors[variant], marker=markers[variant], linewidth=2, label=variant
        )
    titles = ["F1", "Balanced Accuracy", "Specificity", "Latent SWD"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Label fraction")
        ax.grid(alpha=0.25)
        ticks = sorted(perf["fraction"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x:.2f}" for x in ticks])
    axes[0, 0].set_ylabel("Score")
    axes[0, 1].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[1, 1].set_ylabel("Distance")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[1, 0].set_ylim(0.0, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    _save(fig, "site_c_failure_case")


def build_preprocessing_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    ph_base = _load("runs/pohang_main_study/summary.csv")
    ph_base = ph_base[ph_base["method"] == "reconst"].copy()
    ph_base["site"] = "Pohang"
    ph_base["variant"] = "baseline"
    ph_none = _load("runs/pohang_normalization_ablation_v2/bandpass_agc_none/summary.csv")
    ph_none["site"] = "Pohang"
    ph_none["variant"] = "bandpass_agc_none"
    ph_rob = _load("runs/pohang_normalization_ablation_v2/bandpass_agc_robust/summary.csv")
    ph_rob["site"] = "Pohang"
    ph_rob["variant"] = "bandpass_agc_robust"

    u19_base = _load("runs/utah_2019_main_study/summary.csv")
    u19_base = u19_base[u19_base["method"] == "reconst"].copy()
    u19_base["site"] = "Utah 2019"
    u19_base["variant"] = "baseline"
    u19_none = _load("runs/utah_2019_normalization_ablation_v2/bandpass_agc_none/summary.csv")
    u19_none["site"] = "Utah 2019"
    u19_none["variant"] = "bandpass_agc_none"
    u19_rob = _load("runs/utah_2019_normalization_ablation_v2/bandpass_agc_robust/summary.csv")
    u19_rob["site"] = "Utah 2019"
    u19_rob["variant"] = "bandpass_agc_robust"

    u23_base = _load("runs/utah_2023_main_study/summary.csv")
    u23_base = u23_base[u23_base["method"] == "reconst"].copy()
    u23_base["site"] = "Utah 2023"
    u23_base["variant"] = "baseline"
    u23_none = _load("runs/utah_2023_normalization_ablation_v2/bandpass_agc_none/summary.csv")
    u23_none["site"] = "Utah 2023"
    u23_none["variant"] = "bandpass_agc_none"
    u23_rob = _load("runs/utah_2023_normalization_ablation_v2/bandpass_agc_robust/summary.csv")
    u23_rob["site"] = "Utah 2023"
    u23_rob["variant"] = "bandpass_agc_robust"

    perf = pd.concat(
        [ph_base, ph_none, ph_rob, u19_base, u19_none, u19_rob, u23_base, u23_none, u23_rob],
        ignore_index=True,
    )
    perf = perf[
        ["site", "variant", "fraction", "test_f1", "test_balanced_acc", "test_specificity"]
    ].sort_values(["site", "variant", "fraction"])
    perf.to_csv(OUT_DIR / "preprocessing_sensitivity_label_efficiency_table.csv", index=False)

    rows = []
    center_paths = [
        ("Pohang", "baseline", 0.10, "runs/preprocessing_center_diagnostics_v2/pohang/baseline/finetune/pohang__frac0p1/center_history.csv"),
        ("Pohang", "bandpass_agc_none", 0.10, "runs/preprocessing_center_diagnostics_v2/pohang/bandpass_agc_none/finetune/pohang__frac0p1/center_history.csv"),
        ("Pohang", "bandpass_agc_robust", 0.10, "runs/preprocessing_center_diagnostics_v2/pohang/bandpass_agc_robust/finetune/pohang__frac0p1/center_history.csv"),
        ("Utah 2019", "bandpass_agc_none", 1.00, "runs/preprocessing_center_diagnostics_v2/utah_2019/bandpass_agc_none/finetune/base_utah_2019__frac1/center_history.csv"),
        ("Utah 2019", "bandpass_agc_robust", 1.00, "runs/preprocessing_center_diagnostics_v2/utah_2019/bandpass_agc_robust/finetune/base_utah_2019__frac1/center_history.csv"),
    ]
    for site, variant, fraction, path in center_paths:
        df = _load(path)
        last = df.iloc[-1]
        rows.append(
            {
                "site": site,
                "variant": variant,
                "fraction": fraction,
                "epoch": int(last["epoch"]),
                "center_delta_from_initial": float(last["center_delta_from_initial"]),
                "val_dist_gap_event_minus_noise": float(last["val_dist_gap_event_minus_noise"]),
                "val_event_noise_swd": float(last["val_event_noise_swd"]),
            }
        )
    latent = pd.DataFrame(rows).sort_values(["site", "variant"])
    latent.to_csv(OUT_DIR / "preprocessing_sensitivity_latent_table.csv", index=False)
    return perf, latent


def plot_preprocessing_label_efficiency(perf: pd.DataFrame) -> None:
    colors = {"baseline": "#4c72b0", "bandpass_agc_none": "#55a868", "bandpass_agc_robust": "#c44e52"}
    markers = {"baseline": "o", "bandpass_agc_none": "s", "bandpass_agc_robust": "D"}
    label_map = {"baseline": "base", "bandpass_agc_none": "AGC", "bandpass_agc_robust": "AGC+R"}
    variant_order = ["baseline", "bandpass_agc_none", "bandpass_agc_robust"]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.5))
    site_order = ["Pohang", "Utah 2019", "Utah 2023"]
    for i, site in enumerate(site_order):
        site_df = perf[perf["site"] == site]
        for variant in variant_order:
            if variant not in site_df["variant"].unique():
                continue
            sub = site_df[site_df["variant"] == variant].sort_values("fraction")
            axes[i, 0].plot(
                sub["fraction"],
                sub["test_balanced_acc"],
                color=colors[variant],
                marker=markers[variant],
                linewidth=2.6,
                markersize=7,
                zorder=4 if variant == "baseline" else 3,
                label=variant,
            )
            axes[i, 1].plot(
                sub["fraction"],
                sub["test_specificity"],
                color=colors[variant],
                marker=markers[variant],
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
                color=colors[variant],
                fontsize=8,
                va="center",
                fontweight="bold",
            )
            axes[i, 1].annotate(
                label_map[variant],
                (last["fraction"], last["test_specificity"]),
                xytext=(6, 0),
                textcoords="offset points",
                color=colors[variant],
                fontsize=8,
                va="center",
                fontweight="bold",
            )
        axes[i, 0].set_title(f"{site} - Balanced Accuracy")
        axes[i, 1].set_title(f"{site} - Specificity")
        axes[i, 0].set_ylim(0.0, 1.05)
        axes[i, 1].set_ylim(0.0, 1.05)
        axes[i, 0].grid(alpha=0.25)
        axes[i, 1].grid(alpha=0.25)
        axes[i, 0].set_xlabel("Label fraction")
        axes[i, 1].set_xlabel("Label fraction")
        axes[i, 0].set_ylabel("Score")
        axes[i, 1].set_ylabel("Score")
        axes[i, 0].set_xlim(0.03, 1.08)
        axes[i, 1].set_xlim(0.03, 1.08)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    _save(fig, "preprocessing_sensitivity_label_efficiency")


def plot_preprocessing_latent_dynamics(latent: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {"baseline": "#4c72b0", "bandpass_agc_none": "#55a868", "bandpass_agc_robust": "#c44e52"}
    markers = {"baseline": "o", "bandpass_agc_none": "s", "bandpass_agc_robust": "D"}
    label_map = {
        ("Pohang", "baseline"): "PH\nbase",
        ("Pohang", "bandpass_agc_none"): "PH\nAGC",
        ("Pohang", "bandpass_agc_robust"): "PH\nAGC+R",
        ("Utah 2019", "bandpass_agc_none"): "U19\nAGC",
        ("Utah 2019", "bandpass_agc_robust"): "U19\nAGC+R",
    }

    xs = []
    xticks = []
    for xpos, (_, row) in enumerate(latent.iterrows()):
        xs.append(xpos)
        xticks.append(label_map[(row["site"], row["variant"])])
        axes[0].scatter(
            xpos,
            row["val_dist_gap_event_minus_noise"],
            color=colors[row["variant"]],
            marker=markers[row["variant"]],
            s=80,
        )
        axes[1].scatter(
            xpos,
            row["val_event_noise_swd"],
            color=colors[row["variant"]],
            marker=markers[row["variant"]],
            s=80,
        )
    axes[0].set_title("Validation center gap")
    axes[1].set_title("Validation latent SWD")
    axes[0].set_ylabel("Value")
    axes[1].set_ylabel("Value")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(xticks)
    _save(fig, "preprocessing_sensitivity_latent_dynamics")


def write_draft() -> None:
    text = r"""
\subsubsection{Failure and recovery patterns between Site A and Site B}

The transfer results between Site A (Pohang) and Site B (Utah 2019) reveal a clear pattern of asymmetric recovery. Reconstruction-pretrained initialization consistently improved over scratch in all directional and mixed-domain settings, but the magnitude of the gain depended strongly on transfer direction. The weakest case was the transfer from Site A to Site B, where balanced accuracy improved from 0.5043 to 0.6202 but absolute performance remained modest. In the reverse direction, from Site B to Site A, the same initialization increased balanced accuracy from 0.5833 to 0.7639 and F1 from 0.5287 to 0.6848, indicating that the learned representation transferred more effectively into Pohang than into Utah 2019. The mixed-domain setting provided the most stable compromise: although it did not exceed the best directional F1, it achieved the highest specificity (0.9667) together with a strong balanced accuracy of 0.7542. These results suggest that recovery under domain shift is not symmetric, and that joint-domain exposure partially regularizes against site-specific collapse.

\subsection{Dataset-Level Failure Case: Site C (Utah 2023)}

\subsubsection{Collapse under label-scarce and shifted conditions}

Site C (Utah 2023) behaved differently from both recoverable domains. Across label fractions, the reconstruction-pretrained detector often converged to an event-heavy solution rather than a meaningful event--noise separator. In the baseline setting, F1 remained between approximately 0.73 and 0.74, but balanced accuracy stayed near 0.50--0.53. AGC-based preprocessing variants did not improve this behavior; instead, they frequently collapsed to balanced accuracy values of 0.50--0.501. This indicates that label scarcity alone does not explain the difficulty of Site C. Rather, the target-domain condition itself appears substantially more shifted and less recoverable than Site A or Site B.

\subsubsection{Misleading F1 and the necessity of balanced metrics}

Utah 2023 also demonstrated why F1 score alone is insufficient for DAS microseismic transfer studies. Although the detector achieved seemingly acceptable F1 values near 0.73, specificity was near zero in most settings and often exactly zero in the AGC-based variants. In practice, this means that the model was not learning a balanced decision boundary, but was instead labeling nearly all windows as events. Balanced accuracy and specificity therefore become essential for diagnosing collapse under shifted or skewed target-domain conditions. The Utah 2023 case provides a concrete example where F1 alone would have led to an overly optimistic interpretation of transferability.

\subsubsection{Implications of preprocessing sensitivity and target-domain mismatch}

The Site C results further show that preprocessing sensitivity is itself domain-dependent. In Site A, AGC-based preprocessing improved low-label separability; in Site B, AGC with robust normalization was required before recovery became visible. In Site C, however, neither AGC alone nor AGC with robust normalization recovered a usable detector. Offline latent diagnostics reinforce this interpretation: the event--noise Wasserstein distance remained extremely small in the AGC variants and weak even in the baseline setting. Thus, Site C is best interpreted not as a weaker success case, but as a dataset-level failure case in which preprocessing and initialization are insufficient to overcome the target-domain mismatch.

\subsection{Preprocessing Sensitivity and Latent Stability}

\subsubsection{Effect of filtering on label efficiency}

The preprocessing ablation confirms that filtering and normalization do not have a universal effect across DAS domains. In Pohang, AGC without additional robust normalization produced the clearest gains in the low-label regime, improving balanced accuracy from 0.5000 to 0.7345 at 5\% labels and from 0.7095 to 0.9758 at 25\% labels relative to the baseline trajectory. In Utah 2019, the strongest improvements emerged only after robust normalization was added after AGC, with balanced accuracy rising to 0.7754 at 50\% labels and 0.9173 at 100\% labels. Utah 2023 again behaved differently: all preprocessing variants remained near chance-level balanced accuracy despite superficially similar F1 values. These observations show that filtering choices strongly modulate label efficiency, but in a manner that is controlled by target-domain signal statistics rather than by a universal recipe.

\subsubsection{Effect of filtering on latent center dynamics}

Latent center diagnostics help explain these site-dependent preprocessing effects. In Pohang, AGC-based preprocessing greatly enlarged the final validation center gap and the sliced Wasserstein distance between event and noise distributions even at 10\% labels, showing that filtering accelerated the emergence of class-separable latent geometry. In Utah 2019, by contrast, low-label center gaps remained close to zero regardless of AGC, and a meaningful separation appeared only in the AGC+robust setting at full supervision. This means that filtering does not simply improve raw classification scores; it changes whether the latent representation becomes stably separable at all. The preprocessing sensitivity observed in the performance curves is therefore reflected directly in the center dynamics and latent-distribution distances.
"""
    (ROOT / "remaining_results_sections_draft.tex").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    _ensure()
    fr = build_failure_recovery_table()
    plot_failure_recovery(fr)
    perf_c, latent_c = build_utah2023_failure_tables()
    plot_utah2023_failure(perf_c, latent_c)
    perf_p, latent_p = build_preprocessing_tables()
    plot_preprocessing_label_efficiency(perf_p)
    plot_preprocessing_latent_dynamics(latent_p)
    write_draft()


if __name__ == "__main__":
    main()
