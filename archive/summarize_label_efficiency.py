#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt

BRANCH_KEY_MAP = {
    "anomaly": "anomaly_metrics_fixed_threshold",
    "fc": "fc_metrics_fixed_threshold",
    "or": "or_metrics_fixed_threshold",
    "and": "and_metrics_fixed_threshold",
}

def safe_load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
        return None

def parse_branch_metrics(test_metrics_json: dict, branch: str) -> Dict:
    key = BRANCH_KEY_MAP[branch]
    sub = test_metrics_json.get(key, {})
    thresholds = test_metrics_json.get("thresholds", {})
    return {
        "branch": branch,
        "acc": sub.get("acc"),
        "precision": sub.get("precision"),
        "recall": sub.get("recall"),
        "f1": sub.get("f1"),
        "tp": sub.get("tp"),
        "tn": sub.get("tn"),
        "fp": sub.get("fp"),
        "fn": sub.get("fn"),
        "threshold": sub.get("threshold"),
        "score_col": sub.get("score_col"),
        "pred_col": sub.get("pred_col"),
        "rule": sub.get("rule"),
        "threshold_anomaly_score": thresholds.get("anomaly_score"),
        "threshold_fc_prob": thresholds.get("fc_prob"),
    }

def collect_records(run_root: Path, branches: List[str]) -> List[Dict]:
    test_root = run_root / "test"
    finetune_root = run_root / "finetune"

    if not test_root.exists():
        raise FileNotFoundError(f"test root not found: {test_root}")
    if not finetune_root.exists():
        raise FileNotFoundError(f"finetune root not found: {finetune_root}")

    records = []
    for test_json in sorted(test_root.glob("*/test_metrics_fixed_threshold.json")):
        exp_name = test_json.parent.name
        finetune_info_path = finetune_root / exp_name / "label_efficiency_info.json"

        test_js = safe_load_json(test_json)
        if test_js is None:
            continue

        label_js = safe_load_json(finetune_info_path)
        if label_js is None:
            print(f"[WARN] skip {exp_name}: no label_efficiency_info.json")
            continue

        base_row = {
            "experiment": exp_name,
            "run_root": str(run_root),
            "fraction_enabled": label_js.get("enabled"),
            "labeled_fraction": label_js.get("labeled_fraction"),
            "original_num_rows": label_js.get("original_num_rows"),
            "effective_num_rows": label_js.get("effective_num_rows"),
            "fraction_seed": label_js.get("fraction_seed"),
            "original_csv": label_js.get("original_csv"),
            "effective_csv": label_js.get("effective_csv"),
            "per_class_original": json.dumps(label_js.get("per_class_original", {}), ensure_ascii=False),
            "per_class_effective": json.dumps(label_js.get("per_class_effective", {}), ensure_ascii=False),
        }

        for branch in branches:
            row = dict(base_row)
            row.update(parse_branch_metrics(test_js, branch))
            records.append(row)

    return records

def save_branch_csvs(df: pd.DataFrame, out_dir: Path, branches: List[str]) -> None:
    for branch in branches:
        sub = df[df["branch"] == branch].copy()
        sub = sub.sort_values(["effective_num_rows", "labeled_fraction", "f1"], ascending=[True, True, False])
        out_csv = out_dir / f"label_efficiency_{branch}.csv"
        sub.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[INFO] saved: {out_csv}")

def save_summary_csv(df: pd.DataFrame, out_dir: Path) -> None:
    out_csv = out_dir / "label_efficiency_all_branches.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[INFO] saved: {out_csv}")

def save_best_per_fraction_csv(df: pd.DataFrame, out_dir: Path, branches: List[str]) -> None:
    rows = []
    for branch in branches:
        sub = df[df["branch"] == branch].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["labeled_fraction", "f1", "recall", "precision", "acc"], ascending=[True, False, False, False, False])
        best = sub.groupby("labeled_fraction", as_index=False).first()
        rows.append(best)

    if rows:
        best_df = pd.concat(rows, axis=0, ignore_index=True)
        out_csv = out_dir / "label_efficiency_best_per_fraction.csv"
        best_df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[INFO] saved: {out_csv}")

def make_metric_plot(df: pd.DataFrame, out_dir: Path, branch: str, metric: str) -> None:
    sub = df[df["branch"] == branch].copy()
    if sub.empty:
        return

    sub = sub.sort_values(["labeled_fraction", metric, "recall", "precision", "acc"], ascending=[True, False, False, False, False])
    best = sub.groupby("labeled_fraction", as_index=False).first()
    best = best.sort_values("labeled_fraction")

    x = best["effective_num_rows"].tolist()
    y = best[metric].tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Number of labeled train samples")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Label efficiency ({branch}, best per fraction)")
    ax.grid(True, alpha=0.3)

    for _, row in best.iterrows():
        ax.annotate(
            f"{row['labeled_fraction']}",
            (row["effective_num_rows"], row[metric]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    out_png = out_dir / f"plot_{branch}_{metric}.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved: {out_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--branches", type=str, nargs="+", default=["anomaly", "fc"])
    parser.add_argument("--make_plots", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"run_root not found: {run_root}")

    branches = args.branches
    for b in branches:
        if b not in BRANCH_KEY_MAP:
            raise ValueError(f"unsupported branch: {b}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_root / "label_eff_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(run_root, branches)
    if not records:
        raise RuntimeError("No label-efficiency records found.")

    df = pd.DataFrame(records)
    numeric_cols = [
        "labeled_fraction", "original_num_rows", "effective_num_rows", "fraction_seed",
        "acc", "precision", "recall", "f1", "tp", "tn", "fp", "fn",
        "threshold", "threshold_anomaly_score", "threshold_fc_prob",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["branch", "effective_num_rows", "labeled_fraction", "f1"], ascending=[True, True, True, False]).reset_index(drop=True)

    save_summary_csv(df, out_dir)
    save_branch_csvs(df, out_dir, branches)
    save_best_per_fraction_csv(df, out_dir, branches)

    if args.make_plots:
        for branch in branches:
            for metric in ["f1", "acc", "recall", "precision"]:
                make_metric_plot(df, out_dir, branch, metric)

    print(f"[DONE] summary dir: {out_dir}")

if __name__ == "__main__":
    main()
