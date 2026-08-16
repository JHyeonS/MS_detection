#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "mixed_domain_joint_v1"


def _load(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_table() -> pd.DataFrame:
    mixed = _load("runs/pair_pohang_utah2019_parallel_v3/mixed_pohang_utah_2019/summary.csv").copy()
    mixed["setting"] = "Mixed A+B"

    p2u = _load("runs/pair_pohang_utah2019_parallel_v3/cross_pohang_to_utah_2019/summary.csv").copy()
    p2u["setting"] = "A→B"

    u2p = _load("runs/pair_pohang_utah2019_parallel_v3/cross_utah_2019_to_pohang/summary.csv").copy()
    u2p["setting"] = "B→A"

    df = pd.concat([mixed, p2u, u2p], ignore_index=True)
    keep = [
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
    table = df[keep].copy().sort_values(["setting", "method"])
    table.to_csv(OUT_DIR / "mixed_domain_joint_table.csv", index=False)
    return table


def build_figure(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    settings = ["A→B", "B→A", "Mixed A+B"]
    x = range(len(settings))
    width = 0.34
    color = {"scratch": "#4c72b0", "reconst": "#dd8452"}

    for ax, metric, title in [
        (axes[0], "test_balanced_acc", "Balanced Accuracy"),
        (axes[1], "test_specificity", "Specificity"),
    ]:
        for i, method in enumerate(["scratch", "reconst"]):
            vals = []
            for s in settings:
                row = table[(table["setting"] == s) & (table["method"] == method)].iloc[0]
                vals.append(row[metric])
            pos = [v + (i - 0.5) * width for v in x]
            ax.bar(pos, vals, width=width, color=color[method], label=method if metric == "test_balanced_acc" else None)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(settings)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    _save(fig, "mixed_domain_joint")


def write_draft(table: pd.DataFrame) -> None:
    text = r"""
\subsection{Mixed-Domain Training and Joint-Domain Performance}
\label{sec:mixed_domain_joint}

We next examined whether exposing the detector to both Site A (Pohang) and Site B (Utah 2019) during finetuning produced a more stable compromise than one-way cross-domain transfer. Unlike the purely directional transfer settings, mixed-domain training allows the detector to observe both target-domain distributions simultaneously and therefore provides a direct test of whether joint-domain exposure can mitigate the brittleness of cross-site adaptation.

The mixed-domain results show a clear advantage for reconstruction-pretrained initialization. Under scratch finetuning, the joint-domain model achieved an F1 of only 0.2785 and a balanced accuracy of 0.5196, indicating that naive joint optimization was not sufficient to produce a reliable detector. In contrast, the reconstruction-pretrained mixed-domain model increased F1 to 0.6420 and balanced accuracy to 0.7542. Most importantly, specificity increased from 0.5810 to 0.9667, showing that the gain was not merely due to higher recall, but reflected a substantially stronger rejection of non-event samples.

When placed alongside the one-way transfer results, the mixed-domain setting appears to provide the most stable trade-off. It does not exceed the best F1 obtained in the stronger directional case (Site B $\rightarrow$ Site A), but it combines competitive balanced accuracy with the highest specificity among the cross/mixed-domain experiments. This suggests that mixed-domain finetuning can partially regularize the detector against site-specific collapse by forcing the representation to accommodate both domains simultaneously.

Taken together, the mixed-domain experiment indicates that domain diversity during finetuning is beneficial when combined with a pretrained initialization. While scratch training struggled to form a useful joint decision boundary, the reconstruction-pretrained model leveraged the shared structure of the two sites and produced the most balanced overall compromise between event recall and noise rejection.

\begin{table}[t]
\centering
\caption{Mixed-domain and directional transfer comparison between Site A (Pohang) and Site B (Utah 2019). The mixed-domain setting provides the highest specificity and a strong balanced-accuracy trade-off under reconstruction-pretrained initialization.}
\label{tab:mixed_domain_joint}
\begin{tabular}{llccc}
\toprule
Setting & Method & F1 & Balanced acc. & Specificity \\
\midrule
A$\rightarrow$B & scratch & 0.2954 & 0.5043 & 0.3322 \\
A$\rightarrow$B & reconst & 0.3781 & 0.6202 & 0.4722 \\
\midrule
B$\rightarrow$A & scratch & 0.5287 & 0.5833 & 0.3493 \\
B$\rightarrow$A & reconst & 0.6848 & 0.7639 & 0.6400 \\
\midrule
Mixed A+B & scratch & 0.2785 & 0.5196 & 0.5810 \\
Mixed A+B & reconst & 0.6420 & 0.7542 & 0.9667 \\
\bottomrule
\end{tabular}
\end{table}
"""
    (ROOT / "mixed_domain_joint_draft.tex").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    _ensure()
    table = build_table()
    build_figure(table)
    write_draft(table)


if __name__ == "__main__":
    main()
