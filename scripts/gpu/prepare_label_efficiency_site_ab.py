#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "label_efficiency_site_ab_v1"


def _load(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_figure() -> pd.DataFrame:
    pohang = _load("runs/pohang_main_study/summary.csv").copy()
    pohang = pohang[pohang["method"].isin(["scratch", "reconst", "reconst_noanom", "contrast"])]

    u19_main = _load("runs/utah_2019_main_study/summary.csv").copy()
    u19_main = u19_main[u19_main["method"].isin(["scratch", "reconst", "reconst_noanom", "contrast"])]
    u19_main["curve"] = u19_main["method"]

    u19_prep = _load("runs/utah_2019_preprocess_study/bandpass_agc/summary.csv").copy()
    u19_prep["curve"] = "reconst+AGC"

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    style = {
        "scratch": ("#4c72b0", "o"),
        "reconst": ("#dd8452", "s"),
        "reconst_noanom": ("#55a868", "^"),
        "contrast": ("#c44e52", "D"),
        "reconst+AGC": ("#8172b3", "P"),
    }

    for metric, col in [("Balanced Accuracy", "test_balanced_acc"), ("F1", "test_f1")]:
        ax = axes[0, 0] if col == "test_balanced_acc" else axes[0, 1]
        for method in ["scratch", "reconst", "reconst_noanom", "contrast"]:
            sub = pohang[pohang["method"] == method].sort_values("fraction")
            color, marker = style[method]
            ax.plot(sub["fraction"], sub[col], color=color, marker=marker, linewidth=2, markersize=6, label=method)
        ax.set_title(f"Pohang - {metric}")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(pohang["fraction"].unique()))
        ax.set_xticklabels([f"{x:.2f}" for x in sorted(pohang["fraction"].unique())])

    for metric, col in [("Balanced Accuracy", "test_balanced_acc"), ("F1", "test_f1")]:
        ax = axes[1, 0] if col == "test_balanced_acc" else axes[1, 1]
        for curve in ["scratch", "reconst", "reconst_noanom", "contrast"]:
            sub = u19_main[u19_main["curve"] == curve].sort_values("fraction")
            color, marker = style[curve]
            ax.plot(sub["fraction"], sub[col], color=color, marker=marker, linewidth=2, markersize=6, label=curve)
        sub = u19_prep.sort_values("fraction")
        color, marker = style["reconst+AGC"]
        ax.plot(sub["fraction"], sub[col], color=color, marker=marker, linewidth=2.5, markersize=6, label="reconst+AGC")
        ax.set_title(f"Utah 2019 - {metric}")
        ax.set_ylabel("Score")
        ax.set_xlabel("Label fraction")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(u19_prep["fraction"].unique()))
        ax.set_xticklabels([f"{x:.2f}" for x in sorted(u19_prep["fraction"].unique())])

    for ax in axes[0]:
        ax.set_xlabel("Label fraction")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in list(zip(handles, labels)) + list(zip(handles2, labels2)):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)
    fig.legend(uniq_h, uniq_l, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    _save(fig, "label_efficiency_site_ab")

    rows = []
    rows.append(
        pohang.assign(site="Pohang", curve=lambda df: df["method"])[
            ["site", "curve", "fraction", "test_f1", "test_balanced_acc", "test_specificity"]
        ]
    )
    rows.append(
        u19_main.assign(site="Utah 2019")[
            ["site", "curve", "fraction", "test_f1", "test_balanced_acc", "test_specificity"]
        ]
    )
    rows.append(
        u19_prep.assign(site="Utah 2019")[
            ["site", "curve", "fraction", "test_f1", "test_balanced_acc", "test_specificity"]
        ]
    )
    table = pd.concat(rows, ignore_index=True).sort_values(["site", "curve", "fraction"])
    table.to_csv(OUT_DIR / "label_efficiency_site_ab_table.csv", index=False)
    return table


def write_draft(table: pd.DataFrame) -> None:
    text = r"""
\subsection{Label Efficiency in Site A and Site B}
\label{sec:label_efficiency_site_ab}

We next evaluated in-domain label efficiency in the two recoverable domains, Site A (Pohang) and Site B (Utah 2019). This experiment asked whether pretrained representations improved detection performance when only a fraction of the labeled target-domain training set was available. We emphasize balanced accuracy and specificity in addition to F1, because low-label DAS detection can otherwise appear deceptively strong under event-heavy prediction behavior.

In Site A (Pohang), the reconstruction-pretrained model showed the clearest label-efficiency gain once the label fraction exceeded 0.25. At 25\% labels, reconstruction pretraining improved balanced accuracy to 0.7095, while the same model reached near-saturated performance at 50\% and 100\% labels with balanced accuracies of 0.9919 and 0.9913, respectively. By contrast, the contrastive-pretrained model remained much weaker across the entire range, and the anomaly-free reconstruction variant improved over contrastive pretraining but still remained below the full reconstruction-pretrained detector in the moderate-label and high-label regimes.

Site B (Utah 2019) exhibited a different behavior. Using the main-study configuration alone, none of the candidate methods yielded a uniformly dominant label-efficiency curve: scratch, reconstruction pretraining, and contrastive pretraining all remained in a modest performance range, with balanced accuracy typically between approximately 0.58 and 0.69. This instability indicates that pretraining alone was insufficient to guarantee recovery in Utah 2019. However, once preprocessing was aligned using the AGC-based setting, the same reconstruction-pretrained model showed a markedly different trend. Balanced accuracy increased to 0.7754 at 50\% labels and 0.9173 at 100\% labels, demonstrating that Utah 2019 is not intrinsically intractable, but rather requires a target-domain-aligned preprocessing pipeline before the benefits of pretrained initialization become visible.

Taken together, the Site A/Site B label-efficiency results support two conclusions. First, transfer learning can reduce the burden of labeled target data in favorable or recoverable domains. Second, the gain is conditional rather than universal: in Pohang, reconstruction pretraining alone was sufficient to deliver strong label-efficiency improvements, whereas in Utah 2019, the same benefit became apparent only after preprocessing alignment.

\begin{table}[t]
\centering
\caption{Representative label-efficiency results in Site A (Pohang) and Site B (Utah 2019). For Utah 2019, the AGC-aligned reconstruction-pretrained curve is shown separately because it captures the recoverable target-domain behavior more faithfully than the main-study baseline alone.}
\label{tab:label_efficiency_site_ab}
\begin{tabular}{llcccc}
\toprule
Site & Method/setting & Fraction & F1 & Balanced acc. & Specificity \\
\midrule
Pohang & scratch & 0.05 & 0.6667 & 0.6781 & 0.6519 \\
Pohang & reconst & 0.25 & 0.7266 & 0.7095 & 0.5407 \\
Pohang & reconst & 0.50 & 0.9913 & 0.9919 & 0.9926 \\
Pohang & reconst\_noanom & 0.50 & 0.8476 & 0.8647 & 0.9556 \\
Pohang & contrast & 0.50 & 0.5891 & 0.5596 & 0.4148 \\
\midrule
Utah 2019 & scratch & 0.50 & 0.5289 & 0.6918 & 0.8222 \\
Utah 2019 & reconst & 0.50 & 0.3974 & 0.5830 & 0.6222 \\
Utah 2019 & contrast & 0.50 & 0.4769 & 0.6553 & 0.7667 \\
Utah 2019 & reconst+AGC & 0.50 & 0.6500 & 0.7754 & 0.8667 \\
Utah 2019 & reconst+AGC & 1.00 & 0.8455 & 0.9173 & 0.9222 \\
\bottomrule
\end{tabular}
\end{table}
"""
    (ROOT / "label_efficiency_site_ab_draft.tex").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    _ensure()
    table = build_figure()
    write_draft(table)


if __name__ == "__main__":
    main()
