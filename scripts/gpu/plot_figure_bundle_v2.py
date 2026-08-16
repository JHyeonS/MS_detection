#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "figure_bundle_v2_seed_runs"
OUT_DIR = ROOT / "runs" / "paper_figure_bundle_v2"
SEEDS = [int(x.strip()) for x in os.environ.get("FIGURE_V2_SEEDS", "41,42,43,44,45").split(",") if x.strip()]
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]


def _ensure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _seed_tag(seed: int) -> str:
    return f"seed_{seed:02d}"


def load_main(site_key: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        path = RUN_ROOT / f"{site_key}_main" / _seed_tag(seed) / "summary.csv"
        df = _load_csv(path).copy()
        df["seed"] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_norm(site_key: str, variant: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        path = RUN_ROOT / f"{site_key}_norm" / _seed_tag(seed) / variant / "summary.csv"
        df = _load_csv(path).copy()
        df["seed"] = seed
        df["variant"] = variant
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_pair(bundle_name: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        path = RUN_ROOT / "pair" / _seed_tag(seed) / bundle_name / "summary.csv"
        df = _load_csv(path).copy()
        df["seed"] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def agg_curve(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    out = (
        df.groupby([group_col, "fraction"], dropna=False)[metric]
        .agg(["mean", "min", "max", "std", "count"])
        .reset_index()
        .sort_values([group_col, "fraction"])
    )
    out["std"] = out["std"].fillna(0.0)
    return out


def agg_scalar(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(["mean", "min", "max", "std", "count"])
        .reset_index()
    )
    out["std"] = out["std"].fillna(0.0)
    return out


def plot_band(ax, sub: pd.DataFrame, x: str, color: str, marker: str, label: str) -> None:
    sub = sub.sort_values(x)
    xv = sub[x].astype(float).to_numpy()
    y = sub["mean"].to_numpy()
    ylo = sub["mean"].to_numpy() - sub["min"].to_numpy()
    yhi = sub["max"].to_numpy() - sub["mean"].to_numpy()
    ax.plot(xv, y, color=color, marker=marker, linewidth=2.4, markersize=6, label=label)
    ax.errorbar(
        xv,
        y,
        yerr=np.vstack([ylo, yhi]),
        fmt="none",
        ecolor=color,
        elinewidth=1.4,
        capsize=4,
        alpha=0.95,
    )


def _configure_fraction_axis(ax):
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels([f"{x:.2f}" for x in FRACTIONS])
    ax.set_xlim(0.03, 1.07)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)


def build_ablation_model_comparison() -> None:
    ph = load_main("pohang")
    ph = ph[ph["method"].isin(["scratch", "reconst", "reconst_noanom", "contrast"])].copy()
    u19 = load_main("utah_2019")
    u19 = u19[u19["method"].isin(["scratch", "reconst", "reconst_noanom", "contrast"])].copy()

    ph_ba = agg_curve(ph, "method", "test_balanced_acc")
    ph_f1 = agg_curve(ph, "method", "test_f1")
    u19_ba = agg_curve(u19, "method", "test_balanced_acc")
    u19_f1 = agg_curve(u19, "method", "test_f1")

    style = {
        "scratch": ("#4c72b0", "o"),
        "reconst": ("#dd8452", "s"),
        "reconst_noanom": ("#55a868", "^"),
        "contrast": ("#c44e52", "D"),
    }
    order = ["scratch", "reconst", "reconst_noanom", "contrast"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    targets = [
        (axes[0, 0], ph_ba, "Pohang - Balanced Accuracy"),
        (axes[0, 1], ph_f1, "Pohang - F1"),
        (axes[1, 0], u19_ba, "Utah 2019 - Balanced Accuracy"),
        (axes[1, 1], u19_f1, "Utah 2019 - F1"),
    ]
    for ax, data, title in targets:
        for method in order:
            sub = data[data["method"] == method]
            color, marker = style[method]
            plot_band(ax, sub, "fraction", color, marker, method)
        ax.set_title(title)
        ax.set_ylabel("Score")
        _configure_fraction_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel("Label fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
    _save(fig, "ablation_model_comparison")


def build_label_efficiency_site_ab() -> None:
    ph = load_main("pohang")
    ph = ph[ph["method"].isin(["scratch", "reconst", "reconst_noanom", "contrast"])].copy()
    ph["curve"] = ph["method"]
    ph_agcr = load_norm("pohang", "bandpass_agc_robust").copy()
    ph_agcr["curve"] = "reconst+AGC+R"
    u19 = load_main("utah_2019")
    u19 = u19[u19["method"].isin(["scratch", "reconst", "reconst_noanom", "contrast"])].copy()
    u19["curve"] = u19["method"]
    u19_agcr = load_norm("utah_2019", "bandpass_agc_robust").copy()
    u19_agcr["curve"] = "reconst+AGC+R"

    style = {
        "scratch": ("#4c72b0", "o"),
        "reconst": ("#dd8452", "s"),
        "reconst_noanom": ("#55a868", "^"),
        "contrast": ("#c44e52", "D"),
        "reconst+AGC+R": ("#8172b3", "P"),
    }

    ph_curve = pd.concat(
        [
            ph[["curve", "fraction", "seed", "test_balanced_acc", "test_f1"]],
            ph_agcr[["curve", "fraction", "seed", "test_balanced_acc", "test_f1"]],
        ],
        ignore_index=True,
    )
    ph_ba = agg_curve(ph_curve.rename(columns={"curve": "method"}), "method", "test_balanced_acc")
    ph_f1 = agg_curve(ph_curve.rename(columns={"curve": "method"}), "method", "test_f1")
    u19_curve = pd.concat(
        [
            u19[["curve", "fraction", "seed", "test_balanced_acc", "test_f1"]],
            u19_agcr[["curve", "fraction", "seed", "test_balanced_acc", "test_f1"]],
        ],
        ignore_index=True,
    )
    u19_ba = agg_curve(u19_curve.rename(columns={"curve": "method"}), "method", "test_balanced_acc")
    u19_f1 = agg_curve(u19_curve.rename(columns={"curve": "method"}), "method", "test_f1")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    for method in ["scratch", "reconst", "reconst_noanom", "contrast", "reconst+AGC+R"]:
        color, marker = style[method]
        plot_band(axes[0, 0], ph_ba[ph_ba["method"] == method], "fraction", color, marker, method)
        plot_band(axes[0, 1], ph_f1[ph_f1["method"] == method], "fraction", color, marker, method)
    for curve in ["scratch", "reconst", "reconst_noanom", "contrast", "reconst+AGC+R"]:
        color, marker = style[curve]
        plot_band(axes[1, 0], u19_ba[u19_ba["method"] == curve], "fraction", color, marker, curve)
        plot_band(axes[1, 1], u19_f1[u19_f1["method"] == curve], "fraction", color, marker, curve)

    titles = [
        "Pohang - Balanced Accuracy",
        "Pohang - F1",
        "Utah 2019 - Balanced Accuracy",
        "Utah 2019 - F1",
    ]
    for ax, title in zip(axes.flatten(), titles):
        ax.set_title(title)
        ax.set_ylabel("Score")
        _configure_fraction_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel("Label fraction")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    _save(fig, "label_efficiency_site_ab")


def build_ablation_preprocessing() -> None:
    main_ph = load_main("pohang")
    main_u19 = load_main("utah_2019")
    main_u23 = load_main("utah_2023")

    frames = []
    for site_key, site_name, main_df in [
        ("pohang", "Pohang", main_ph),
        ("utah_2019", "Utah 2019", main_u19),
        ("utah_2023", "Utah 2023", main_u23),
    ]:
        base = main_df[main_df["method"] == "reconst"].copy()
        base["site"] = site_name
        base["variant"] = "baseline"
        frames.append(base)
        for variant in ["bandpass_agc_none", "bandpass_agc_robust"]:
            df = load_norm(site_key, variant).copy()
            df["site"] = site_name
            df["variant"] = variant
            frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    colors = {"baseline": "#4c72b0", "bandpass_agc_none": "#55a868", "bandpass_agc_robust": "#c44e52"}
    markers = {"baseline": "o", "bandpass_agc_none": "s", "bandpass_agc_robust": "D"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for i, site in enumerate(["Pohang", "Utah 2019", "Utah 2023"]):
        site_df = all_df[all_df["site"] == site]
        ba = agg_curve(site_df, "variant", "test_balanced_acc")
        sp = agg_curve(site_df, "variant", "test_specificity")
        for variant in ["baseline", "bandpass_agc_none", "bandpass_agc_robust"]:
            plot_band(axes[i, 0], ba[ba["variant"] == variant], "fraction", colors[variant], markers[variant], variant)
            plot_band(axes[i, 1], sp[sp["variant"] == variant], "fraction", colors[variant], markers[variant], variant)
        axes[i, 0].set_title(f"{site} - Balanced Accuracy")
        axes[i, 1].set_title(f"{site} - Specificity")
        axes[i, 0].set_ylabel("Score")
        axes[i, 1].set_ylabel("Score")
        axes[i, 0].set_xlabel("Label fraction")
        axes[i, 1].set_xlabel("Label fraction")
        _configure_fraction_axis(axes[i, 0])
        _configure_fraction_axis(axes[i, 1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    _save(fig, "ablation_preprocessing")


def build_cross_domain_transfer_ab() -> None:
    mixed = load_pair("mixed_pohang_utah_2019").copy()
    mixed["setting"] = "Mixed"
    p2u = load_pair("cross_pohang_to_utah_2019").copy()
    p2u["setting"] = "A→B"
    u2p = load_pair("cross_utah_2019_to_pohang").copy()
    u2p["setting"] = "B→A"
    df = pd.concat([p2u, u2p, mixed], ignore_index=True)

    style = {"scratch": "#4c72b0", "reconst": "#dd8452"}
    settings = ["A→B", "B→A", "Mixed"]
    methods = ["scratch", "reconst"]
    metrics = [("test_balanced_acc", "Balanced Accuracy"), ("test_f1", "F1")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    width = 0.36
    x = np.arange(len(settings))
    for ax, (metric, title) in zip(axes, metrics):
        agg = agg_scalar(df, ["setting", "method"], metric)
        for j, method in enumerate(methods):
            vals = []
            lower = []
            upper = []
            for setting in settings:
                row = agg[(agg["setting"] == setting) & (agg["method"] == method)].iloc[0]
                vals.append(row["mean"])
                lower.append(row["mean"] - row["min"])
                upper.append(row["max"] - row["mean"])
            pos = x + (j - 0.5) * width
            ax.bar(pos, vals, width=width, color=style[method], alpha=0.9, label=method)
            ax.errorbar(pos, vals, yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=4, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(settings)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(alpha=0.25, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    _save(fig, "cross_domain_transfer_ab")


def build_mixed_domain_joint() -> None:
    mixed = load_pair("mixed_pohang_utah_2019").copy()
    metrics = [("test_f1", "F1"), ("test_balanced_acc", "Balanced Accuracy"), ("test_specificity", "Specificity")]
    methods = ["scratch", "reconst"]
    colors = {"scratch": "#4c72b0", "reconst": "#dd8452"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    x = np.arange(len(methods))
    for ax, (metric, title) in zip(axes, metrics):
        agg = agg_scalar(mixed, ["method"], metric)
        vals, lower, upper = [], [], []
        for method in methods:
            row = agg[agg["method"] == method].iloc[0]
            vals.append(row["mean"])
            lower.append(row["mean"] - row["min"])
            upper.append(row["max"] - row["mean"])
        ax.bar(x, vals, color=[colors[m] for m in methods], width=0.6)
        ax.errorbar(x, vals, yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=4, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    _save(fig, "mixed_domain_joint")


def build_failure_recovery_site_ab() -> None:
    mixed = load_pair("mixed_pohang_utah_2019").copy()
    mixed["setting"] = "Mixed"
    p2u = load_pair("cross_pohang_to_utah_2019").copy()
    p2u["setting"] = "A→B"
    u2p = load_pair("cross_utah_2019_to_pohang").copy()
    u2p["setting"] = "B→A"
    df = pd.concat([p2u, u2p, mixed], ignore_index=True)
    settings = ["A→B", "B→A", "Mixed"]
    metrics = [("test_f1", "F1 gain"), ("test_balanced_acc", "Balanced Accuracy gain"), ("test_specificity", "Specificity gain")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    x = np.arange(len(settings))
    for ax, (metric, title) in zip(axes, metrics):
        gains = []
        lows = []
        highs = []
        for setting in settings:
            sub = df[df["setting"] == setting]
            pivot = sub.pivot(index="seed", columns="method", values=metric)
            gain = pivot["reconst"] - pivot["scratch"]
            gains.append(gain.mean())
            lows.append(gain.mean() - gain.min())
            highs.append(gain.max() - gain.mean())
        ax.bar(x, gains, color="#55a868", width=0.62)
        ax.errorbar(x, gains, yerr=np.vstack([lows, highs]), fmt="none", ecolor="black", capsize=4, linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(settings)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    _save(fig, "failure_recovery_site_ab")


def build_site_c_failure_case() -> None:
    main = load_main("utah_2023")
    base = main[main["method"] == "reconst"].copy()
    base["variant"] = "baseline"
    none = load_norm("utah_2023", "bandpass_agc_none").copy()
    robust = load_norm("utah_2023", "bandpass_agc_robust").copy()
    df = pd.concat([base, none, robust], ignore_index=True)
    colors = {"baseline": "#4c72b0", "bandpass_agc_none": "#55a868", "bandpass_agc_robust": "#c44e52"}
    markers = {"baseline": "o", "bandpass_agc_none": "s", "bandpass_agc_robust": "D"}
    metrics = [("test_f1", "F1"), ("test_balanced_acc", "Balanced Accuracy"), ("test_specificity", "Specificity")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (metric, title) in zip(axes, metrics):
        agg = agg_curve(df, "variant", metric)
        for variant in ["baseline", "bandpass_agc_none", "bandpass_agc_robust"]:
            plot_band(ax, agg[agg["variant"] == variant], "fraction", colors[variant], markers[variant], variant)
        ax.set_title(f"Utah 2023 - {title}")
        ax.set_ylabel("Score")
        ax.set_xlabel("Label fraction")
        _configure_fraction_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    _save(fig, "site_c_failure_case")


def build_preprocessing_sensitivity_label_efficiency() -> None:
    main_ph = load_main("pohang")
    main_u19 = load_main("utah_2019")
    main_u23 = load_main("utah_2023")
    frames = []
    for site_key, site_name, main_df in [
        ("pohang", "Pohang", main_ph),
        ("utah_2019", "Utah 2019", main_u19),
        ("utah_2023", "Utah 2023", main_u23),
    ]:
        base = main_df[main_df["method"] == "reconst"].copy()
        base["site"] = site_name
        base["variant"] = "baseline"
        frames.append(base)
        for variant in ["bandpass_agc_none", "bandpass_agc_robust"]:
            df = load_norm(site_key, variant).copy()
            df["site"] = site_name
            df["variant"] = variant
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    colors = {"baseline": "#4c72b0", "bandpass_agc_none": "#55a868", "bandpass_agc_robust": "#c44e52"}
    markers = {"baseline": "o", "bandpass_agc_none": "s", "bandpass_agc_robust": "D"}
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.5))
    for i, site in enumerate(["Pohang", "Utah 2019", "Utah 2023"]):
        sub_df = df[df["site"] == site]
        ba = agg_curve(sub_df, "variant", "test_balanced_acc")
        sp = agg_curve(sub_df, "variant", "test_specificity")
        for variant in ["baseline", "bandpass_agc_none", "bandpass_agc_robust"]:
            plot_band(axes[i, 0], ba[ba["variant"] == variant], "fraction", colors[variant], markers[variant], variant)
            plot_band(axes[i, 1], sp[sp["variant"] == variant], "fraction", colors[variant], markers[variant], variant)
        axes[i, 0].set_title(f"{site} - Balanced Accuracy")
        axes[i, 1].set_title(f"{site} - Specificity")
        axes[i, 0].set_ylabel("Score")
        axes[i, 1].set_ylabel("Score")
        axes[i, 0].set_xlabel("Label fraction")
        axes[i, 1].set_xlabel("Label fraction")
        _configure_fraction_axis(axes[i, 0])
        _configure_fraction_axis(axes[i, 1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    _save(fig, "preprocessing_sensitivity_label_efficiency")


def write_readme() -> None:
    text = (
        "paper_figure_bundle_v2\n"
        f"Seeds: {SEEDS}\n"
        "Quantitative line figures use mean curves with asymmetric min-max error bars.\n"
        "Quantitative bar figures use mean bars with asymmetric min-max error bars.\n"
        "Qualitative UMAP/t-SNE panels are not aggregated in v2 and should remain representative-seed figures.\n"
    )
    (OUT_DIR / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    _ensure()
    build_ablation_model_comparison()
    build_label_efficiency_site_ab()
    build_ablation_preprocessing()
    build_cross_domain_transfer_ab()
    build_mixed_domain_joint()
    build_failure_recovery_site_ab()
    build_site_c_failure_case()
    build_preprocessing_sensitivity_label_efficiency()
    write_readme()


if __name__ == "__main__":
    main()
