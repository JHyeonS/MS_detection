#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "in_domain_center_dynamics_v1"


def _load(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_curves() -> pd.DataFrame:
    configs = [
        ("Pohang", "AGC-no-norm", 0.10, "runs/preprocessing_center_diagnostics_v2/pohang/bandpass_agc_none/finetune/pohang__frac0p1/center_history.csv"),
        ("Pohang", "AGC-no-norm", 0.50, "runs/preprocessing_center_diagnostics_v2/pohang/bandpass_agc_none/finetune/pohang__frac0p5/center_history.csv"),
        ("Pohang", "AGC-no-norm", 1.00, "runs/preprocessing_center_diagnostics_v2/pohang/bandpass_agc_none/finetune/pohang__frac1/center_history.csv"),
        ("Utah 2019", "AGC+robust", 0.10, "runs/preprocessing_center_diagnostics_v2/utah_2019/bandpass_agc_robust/finetune/base_utah_2019__frac0p1/center_history.csv"),
        ("Utah 2019", "AGC+robust", 0.50, "runs/preprocessing_center_diagnostics_v2/utah_2019/bandpass_agc_robust/finetune/base_utah_2019__frac0p5/center_history.csv"),
        ("Utah 2019", "AGC+robust", 1.00, "runs/preprocessing_center_diagnostics_v2/utah_2019/bandpass_agc_robust/finetune/base_utah_2019__frac1/center_history.csv"),
    ]

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=False)
    colors = {0.10: "#4c72b0", 0.50: "#dd8452", 1.00: "#55a868"}

    for site, label, frac, path in configs:
        df = _load(path)
        df = df.copy()
        df["site"] = site
        df["setting"] = label
        df["fraction"] = frac
        rows.append(df)

        row = 0 if site == "Pohang" else 1
        axes[row, 0].plot(df["epoch"], df["val_dist_gap_event_minus_noise"], color=colors[frac], linewidth=2, label=f"{frac:.2f}")
        axes[row, 1].plot(df["epoch"], df["val_event_noise_swd"], color=colors[frac], linewidth=2, label=f"{frac:.2f}")

    axes[0, 0].set_title("Pohang - Validation center gap")
    axes[0, 1].set_title("Pohang - Validation latent SWD")
    axes[1, 0].set_title("Utah 2019 - Validation center gap")
    axes[1, 1].set_title("Utah 2019 - Validation latent SWD")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, title="Label fraction")
    _save(fig, "in_domain_center_dynamics")

    full = pd.concat(rows, ignore_index=True)
    full.to_csv(OUT_DIR / "in_domain_center_dynamics_full.csv", index=False)

    final = (
        full.sort_values("epoch")
        .groupby(["site", "setting", "fraction"], as_index=False)
        .tail(1)[
            [
                "site",
                "setting",
                "fraction",
                "epoch",
                "center_delta_from_initial",
                "val_dist_gap_event_minus_noise",
                "val_dist_ratio_event_over_noise",
                "val_event_noise_swd",
            ]
        ]
        .sort_values(["site", "fraction"])
    )
    final.to_csv(OUT_DIR / "in_domain_center_dynamics_final.csv", index=False)
    return final


def add_performance(final: pd.DataFrame) -> pd.DataFrame:
    ph = _load("runs/pohang_normalization_ablation_v2/bandpass_agc_none/summary.csv")
    ph = ph[["fraction", "test_f1", "test_balanced_acc", "test_specificity"]].copy()
    ph["site"] = "Pohang"
    ph["setting"] = "AGC-no-norm"

    u19 = _load("runs/utah_2019_preprocess_study/bandpass_agc/summary.csv")
    u19 = u19[["fraction", "test_f1", "test_balanced_acc", "test_specificity"]].copy()
    u19["site"] = "Utah 2019"
    u19["setting"] = "AGC+robust"

    perf = pd.concat([ph, u19], ignore_index=True)
    merged = final.merge(perf, on=["site", "setting", "fraction"], how="left")
    merged.to_csv(OUT_DIR / "in_domain_center_dynamics_summary_table.csv", index=False)
    return merged


def write_draft(summary: pd.DataFrame) -> None:
    text = r"""
\subsection{Latent Center Dynamics in Site A and Site B}
\label{sec:in_domain_center_dynamics}

To understand why the in-domain label-efficiency curves differed between Site A (Pohang) and Site B (Utah 2019), we examined the dynamics of the latent center and the event--noise feature distributions during finetuning. We used the preprocessing settings that best represented the recoverable behavior of each site: AGC without additional normalization for Pohang, and AGC with robust normalization for Utah 2019. For each setting, we tracked (1) the validation event--noise center gap, defined as the difference between the event and noise distances to the latent center, and (2) the validation sliced Wasserstein distance (SWD) between event and noise latent distributions.

In Site A (Pohang), the latent structure separated rapidly once the detector moved beyond the collapsed low-label regime. Even at 10\% labels, the final validation center gap reached 104.23 and the validation event--noise SWD reached 0.2148, indicating that event and noise features were already distinctly separated in latent space. At 50\% and 100\% labels, this separation became extreme: the center gap exceeded 250 and the SWD remained around 0.40, matching the near-saturated test performance observed in the label-efficiency analysis.

In Site B (Utah 2019), the same diagnostics revealed a delayed recovery. At 10\% labels, the final validation center gap was close to zero and the SWD remained near zero, indicating that event and noise samples were still poorly separated in latent space. At 50\% labels, only a weak separation emerged, with a validation center gap of approximately 0.55 and SWD of 0.0160. A qualitatively different regime appeared only at full supervision, where the validation center gap increased to 69.60 and the SWD rose to 0.1725. This behavior mirrors the label-efficiency result: Utah 2019 did not benefit reliably from pretrained initialization alone, but became recoverable once sufficient target labels and preprocessing alignment were available.

These latent-center results reinforce the main in-domain interpretation of this study. In Pohang, reconstruction-based transfer succeeded because event and noise distributions became cleanly separated even in the low-label regime. In Utah 2019, by contrast, the model required both target-domain preprocessing alignment and larger label fractions before a stable event--noise geometry emerged in latent space. Thus, the difference between Site A and Site B is not simply a matter of final accuracy, but also a difference in how quickly and how stably the latent representation becomes class-separable during finetuning.

\begin{table}[t]
\centering
\caption{Final-epoch latent center diagnostics for the representative in-domain settings used in Site A (Pohang) and Site B (Utah 2019).}
\label{tab:in_domain_center_dynamics}
\begin{tabular}{llcccccc}
\toprule
Site & Setting & Fraction & Epoch & Center drift & Gap & SWD & Test bal. acc. \\
\midrule
Pohang & AGC-no-norm & 0.10 & 36 & 17.48 & 104.23 & 0.2148 & 0.7378 \\
Pohang & AGC-no-norm & 0.50 & 66 & 23.68 & 370.62 & 0.4044 & 0.9870 \\
Pohang & AGC-no-norm & 1.00 & 62 & 10.99 & 257.26 & 0.4125 & 0.9870 \\
\midrule
Utah 2019 & AGC+robust & 0.10 & 21 & 20.52 & 0.14 & 0.0092 & 0.4852 \\
Utah 2019 & AGC+robust & 0.50 & 21 & 8.91 & 0.55 & 0.0160 & 0.7754 \\
Utah 2019 & AGC+robust & 1.00 & 52 & 7.79 & 69.60 & 0.1725 & 0.9173 \\
\bottomrule
\end{tabular}
\end{table}
"""
    (ROOT / "in_domain_center_dynamics_draft.tex").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    _ensure()
    final = build_curves()
    summary = add_performance(final)
    write_draft(summary)


if __name__ == "__main__":
    main()
