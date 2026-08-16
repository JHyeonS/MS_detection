#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("runs/hpo")
OUT_PATH = Path("hpo_tables.tex")


def fmt_float(x, ndigits=4):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{ndigits}f}"


def fmt_drop(x):
    if pd.isna(x):
        return "--"
    return f"{float(x):.1f}"


def escape_tex(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    out = str(text)
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name / "leaderboard.csv")


def main_table_architecture(df: pd.DataFrame) -> str:
    top = df.sort_values("metric_value", ascending=False).iloc[0]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Selected encoder architecture from architecture HPO on the Pohang dataset.}")
    lines.append(r"\label{tab:hpo_architecture_selected}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Layers & Base channels & Latent dim & Dropout & Val. F1 \\")
    lines.append(r"\midrule")
    lines.append(
        "Pohang"
        f" & {int(top['sampled_model.encoder.num_layers'])}"
        f" & {int(top['sampled_model.encoder.base_channels'])}"
        f" & {int(top['sampled_model.encoder.latent_dim'])}"
        f" & {fmt_drop(top['sampled_model.encoder.dropout'])}"
        f" & {fmt_float(top['metric_value'])} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main_table_finetune(pohang: pd.DataFrame, ut19: pd.DataFrame, ut23: pd.DataFrame, ut23_targeted: pd.DataFrame) -> str:
    def best_row(df: pd.DataFrame):
        return df.sort_values("metric_value", ascending=False).iloc[0]

    rows = [
        ("Pohang", "Val. F1", best_row(pohang)),
        ("Utah 2019", "Val. F1", best_row(ut19)),
        ("Utah 2023", "Val. F1", best_row(ut23)),
        ("Utah 2023 (targeted)", "Balanced acc.", best_row(ut23_targeted)),
    ]
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Best finetuning hyperparameters selected for each target site after fixing the encoder architecture.}")
    lines.append(r"\label{tab:hpo_finetune_selected}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Target site & Selection metric & Learning rate & Weight decay & Batch size & Anomaly weight & Best metric \\")
    lines.append(r"\midrule")
    for site, metric_name, row in rows:
        wd = row["sampled_train.weight_decay"] if "sampled_train.weight_decay" in row.index else "--"
        bs = row["sampled_train.batch_size"] if "sampled_train.batch_size" in row.index else "--"
        lines.append(
            f"{site} & {metric_name}"
            f" & {fmt_float(row['sampled_train.lr'], 4)}"
            f" & {fmt_float(wd, 4) if wd != '--' else '--'}"
            f" & {int(bs) if bs != '--' else '--'}"
            f" & {fmt_float(row['sampled_train.anomaly_loss_weight'], 2)}"
            f" & {fmt_float(row['metric_value'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def appendix_longtable_arch(df: pd.DataFrame) -> str:
    df = df.sort_values("metric_value", ascending=False).copy()
    lines = []
    lines.append(r"\begin{longtable}{cccccc}")
    lines.append(r"\caption{Full architecture HPO results on the Pohang dataset.}\label{tab:hpo_architecture_full}\\")
    lines.append(r"\toprule")
    lines.append(r"Trial & Layers & Base channels & Latent dim & Dropout & Val. F1 \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Trial & Layers & Base channels & Latent dim & Dropout & Val. F1 \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for _, row in df.iterrows():
        lines.append(
            f"{int(row['trial_index'])}"
            f" & {int(row['sampled_model.encoder.num_layers'])}"
            f" & {int(row['sampled_model.encoder.base_channels'])}"
            f" & {int(row['sampled_model.encoder.latent_dim'])}"
            f" & {fmt_drop(row['sampled_model.encoder.dropout'])}"
            f" & {fmt_float(row['metric_value'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def appendix_longtable_finetune(df: pd.DataFrame, caption: str, label: str, include_weight_decay: bool = True) -> str:
    df = df.sort_values("metric_value", ascending=False).copy()
    if include_weight_decay:
        colspec = "cccccc"
        header = r"Trial & Learning rate & Weight decay & Batch size & Anomaly weight & Metric \\"
    else:
        colspec = "ccccc"
        header = r"Trial & Learning rate & BCE neg. weight & Anomaly weight & Metric \\"

    lines = []
    lines.append(rf"\begin{{longtable}}{{{colspec}}}")
    lines.append(rf"\caption{{{caption}}}\label{{{label}}}\\")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for _, row in df.iterrows():
        if include_weight_decay:
            lines.append(
                f"{int(row['trial_index'])}"
                f" & {fmt_float(row['sampled_train.lr'], 4)}"
                f" & {fmt_float(row['sampled_train.weight_decay'], 4)}"
                f" & {int(row['sampled_train.batch_size'])}"
                f" & {fmt_float(row['sampled_train.anomaly_loss_weight'], 2)}"
                f" & {fmt_float(row['metric_value'])} \\\\"
            )
        else:
            lines.append(
                f"{int(row['trial_index'])}"
                f" & {fmt_float(row['sampled_train.lr'], 4)}"
                f" & {fmt_float(row['sampled_train.bce_neg_weight'], 2)}"
                f" & {fmt_float(row['sampled_train.anomaly_loss_weight'], 2)}"
                f" & {fmt_float(row['metric_value'])} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def main():
    arch = load_csv("architecture_stage1_pohang")
    pohang_ft = load_csv("finetune_stage1_pohang_best_arch_refined")
    ut19_ft = load_csv("finetune_stage1_utah_2019_best_arch_refined")
    ut23_ft = load_csv("finetune_stage1_utah_2023_best_arch_refined")
    ut23_targeted = load_csv("finetune_utah_2023_silu_bce_weight_targeted")

    parts = []
    parts.append("% Auto-generated by scripts/gpu/generate_hpo_latex_tables.py")
    parts.append("% Requires: \\usepackage{booktabs}, \\usepackage{longtable}")
    parts.append("")
    parts.append("% =====================")
    parts.append("% Main-text tables")
    parts.append("% =====================")
    parts.append(main_table_architecture(arch))
    parts.append("")
    parts.append(main_table_finetune(pohang_ft, ut19_ft, ut23_ft, ut23_targeted))
    parts.append("")
    parts.append("% =====================")
    parts.append("% Appendix tables")
    parts.append("% =====================")
    parts.append(appendix_longtable_arch(arch))
    parts.append("")
    parts.append(
        appendix_longtable_finetune(
            pohang_ft,
            caption="Full finetuning HPO results for Pohang after fixing the encoder architecture.",
            label="tab:hpo_pohang_full",
            include_weight_decay=True,
        )
    )
    parts.append("")
    parts.append(
        appendix_longtable_finetune(
            ut19_ft,
            caption="Full finetuning HPO results for Utah 2019 after fixing the encoder architecture.",
            label="tab:hpo_utah2019_full",
            include_weight_decay=True,
        )
    )
    parts.append("")
    parts.append(
        appendix_longtable_finetune(
            ut23_ft,
            caption="Full finetuning HPO results for Utah 2023 after fixing the encoder architecture.",
            label="tab:hpo_utah2023_full",
            include_weight_decay=True,
        )
    )
    parts.append("")
    parts.append(
        appendix_longtable_finetune(
            ut23_targeted,
            caption="Full targeted finetuning HPO results for Utah 2023 using SiLU and weighted BCE.",
            label="tab:hpo_utah2023_targeted_full",
            include_weight_decay=False,
        )
    )

    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"[DONE] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
