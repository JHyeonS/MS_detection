#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "cross_domain_transfer_ab_v1"


def _load(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_table() -> pd.DataFrame:
    mixed = _load("runs/pair_pohang_utah2019_parallel_v3/mixed_pohang_utah_2019/summary.csv").copy()
    mixed["setting"] = "Mixed A+B"

    p2u = _load("runs/pair_pohang_utah2019_parallel_v3/cross_pohang_to_utah_2019/summary.csv").copy()
    p2u["setting"] = "Pretrain A -> Finetune B"

    u2p = _load("runs/pair_pohang_utah2019_parallel_v3/cross_utah_2019_to_pohang/summary.csv").copy()
    u2p["setting"] = "Pretrain B -> Finetune A"

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
    table.to_csv(OUT_DIR / "cross_domain_transfer_ab_table.csv", index=False)
    return table


def build_figure(table: pd.DataFrame) -> None:
    settings = ["Pretrain A -> Finetune B", "Pretrain B -> Finetune A", "Mixed A+B"]
    metrics = [
        ("test_balanced_acc", "Balanced Accuracy"),
        ("test_f1", "F1"),
        ("test_specificity", "Specificity"),
    ]
    color = {"scratch": "#4c72b0", "reconst": "#dd8452"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    x = range(len(settings))
    width = 0.34

    for ax, (metric, title) in zip(axes, metrics):
        for i, method in enumerate(["scratch", "reconst"]):
            vals = []
            for s in settings:
                row = table[(table["setting"] == s) & (table["method"] == method)].iloc[0]
                vals.append(row[metric])
            pos = [v + (i - 0.5) * width for v in x]
            ax.bar(pos, vals, width=width, color=color[method], label=method if metric == "test_balanced_acc" else None)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(["A→B", "B→A", "Mixed"])
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    _save(fig, "cross_domain_transfer_ab")


def write_draft(table: pd.DataFrame) -> None:
    text = r"""
\subsection{Cross-Domain Transfer Between Site A and Site B}
\label{sec:cross_domain_transfer_ab}

We next evaluated cross-domain transfer between Site A (Pohang) and Site B (Utah 2019), together with a mixed-domain training setting that pooled both domains. At this stage, the completed transfer experiments do not constitute a full label-fraction sweep; instead, they provide full-split cross-domain comparisons between scratch finetuning and reconstruction-pretrained initialization. We therefore interpret them as transfer-performance diagnostics rather than as direct extensions of the in-domain label-efficiency curves.

The results show that reconstruction pretraining consistently improved over scratch in all three settings, but the magnitude and character of the gain were highly asymmetric. In the transfer from Site A to Site B, reconstruction pretraining raised balanced accuracy from 0.5043 to 0.6202 and increased F1 from 0.2954 to 0.3781. Although this is a measurable improvement, the absolute performance remained modest, indicating that Site B remained difficult even when initialized from Site A pretraining. In the opposite direction, from Site B to Site A, reconstruction pretraining yielded a much stronger improvement, increasing balanced accuracy from 0.5833 to 0.7639 and F1 from 0.5287 to 0.6848. This asymmetry suggests that the transferability of the learned representation is not symmetric across sites, and that Site B contains features that transfer more effectively into Site A than vice versa.

The mixed-domain setting produced the strongest overall trade-off. When training jointly on both domains, reconstruction pretraining improved balanced accuracy from 0.5196 to 0.7542 and increased specificity from 0.5810 to 0.9667. This result is important because it shows that pretraining does not merely improve recall; under joint-domain exposure it also stabilizes noise rejection, which was a key weakness in several site-shifted settings.

Overall, the cross-domain and mixed-domain experiments support three conclusions. First, reconstruction-pretrained initialization is consistently more effective than scratch under domain shift. Second, the benefit is asymmetric: transfer from Utah 2019 to Pohang is substantially stronger than transfer from Pohang to Utah 2019. Third, the mixed-domain setting provides the most stable compromise, suggesting that domain diversity during finetuning can partially mitigate the brittleness of purely one-way transfer.

\begin{table}[t]
\centering
\caption{Cross-domain and mixed-domain transfer results between Site A (Pohang) and Site B (Utah 2019). These experiments use full-split transfer settings rather than a label-fraction sweep.}
\label{tab:cross_domain_transfer_ab}
\begin{tabular}{llccc}
\toprule
Setting & Method & F1 & Balanced acc. & Specificity \\
\midrule
Pretrain A $\rightarrow$ Finetune B & scratch & 0.2954 & 0.5043 & 0.3322 \\
Pretrain A $\rightarrow$ Finetune B & reconst & 0.3781 & 0.6202 & 0.4722 \\
\midrule
Pretrain B $\rightarrow$ Finetune A & scratch & 0.5287 & 0.5833 & 0.3493 \\
Pretrain B $\rightarrow$ Finetune A & reconst & 0.6848 & 0.7639 & 0.6400 \\
\midrule
Mixed A+B & scratch & 0.2785 & 0.5196 & 0.5810 \\
Mixed A+B & reconst & 0.6420 & 0.7542 & 0.9667 \\
\bottomrule
\end{tabular}
\end{table}
"""
    (ROOT / "cross_domain_transfer_ab_draft.tex").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    _ensure()
    table = build_table()
    build_figure(table)
    write_draft(table)


if __name__ == "__main__":
    main()
