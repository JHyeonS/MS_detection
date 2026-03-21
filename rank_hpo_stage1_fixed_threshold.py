#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rank_hpo_stage1_fixed_threshold.py

Usage
-----
python rank_hpo_stage1_fixed_threshold.py \
    --run_root /home/ted1204/MS_Detection/runs/hpo_stage1_contrast \
    --topk 5

Optional
--------
python rank_hpo_stage1_fixed_threshold.py \
    --run_root /home/ted1204/MS_Detection/runs/hpo_stage1_contrast \
    --topk 5 \
    --out_csv /home/ted1204/MS_Detection/runs/hpo_stage1_contrast/hpo_top5_fixed_threshold.csv

What it does
------------
- Finds all files matching:
    <run_root>/test/*/test_metrics_fixed_threshold.json
- Reads metrics from each json
- Ranks experiments by:
    1) f1 (descending)
    2) recall (descending)
    3) precision (descending)
    4) acc (descending)
- Prints top-k results
- Optionally saves full ranking to CSV
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_experiment_name(exp_name: str) -> Dict[str, Any]:
    """
    Example:
        pohang_hpo_contrast_L4_D64_BP3_50
        pohang_hpo2_reconst_L5_D512_BP5_80_S43
    """
    out = {
        "method": None,
        "num_layers": None,
        "latent_dim": None,
        "bp_low": None,
        "bp_high": None,
        "seed": None,
    }

    if "_contrast_" in exp_name:
        out["method"] = "contrast"
    elif "_reconst_" in exp_name:
        out["method"] = "reconst"

    m = re.search(r"_L(\d+)_D(\d+)_BP(\d+)_(\d+)(?:_S(\d+))?$", exp_name)
    if m:
        out["num_layers"] = int(m.group(1))
        out["latent_dim"] = int(m.group(2))
        out["bp_low"] = int(m.group(3))
        out["bp_high"] = int(m.group(4))
        if m.group(5) is not None:
            out["seed"] = int(m.group(5))

    return out


def load_metrics(json_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {json_path}: {e}")
        return None

    exp_name = json_path.parent.name

    row = {
        "experiment": exp_name,
        "metrics_path": str(json_path),
        "acc": js.get("acc"),
        "precision": js.get("precision"),
        "recall": js.get("recall"),
        "f1": js.get("f1"),
        "tp": js.get("tp"),
        "tn": js.get("tn"),
        "fp": js.get("fp"),
        "fn": js.get("fn"),
        "threshold": js.get("threshold"),
    }

    row.update(parse_experiment_name(exp_name))
    return row


def collect_results(run_root: Path) -> pd.DataFrame:
    paths = sorted(run_root.glob("test/*/test_metrics_fixed_threshold.json"))

    if not paths:
        raise FileNotFoundError(
            f"No files found under: {run_root}/test/*/test_metrics_fixed_threshold.json"
        )

    rows = []
    for p in paths:
        row = load_metrics(p)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("Found metric files, but failed to load all of them.")

    df = pd.DataFrame(rows)

    numeric_cols = [
        "acc", "precision", "recall", "f1",
        "tp", "tn", "fp", "fn",
        "threshold", "num_layers", "latent_dim",
        "bp_low", "bp_high", "seed",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def rank_results(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(
        by=["f1", "recall", "precision", "acc"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def print_topk(df: pd.DataFrame, topk: int):
    cols = [
        "rank",
        "experiment",
        "method",
        "num_layers",
        "latent_dim",
        "bp_low",
        "bp_high",
        "seed",
        "f1",
        "recall",
        "precision",
        "acc",
        "threshold",
        "fp",
        "fn",
    ]

    cols = [c for c in cols if c in df.columns]
    show_df = df[cols].head(topk).copy()

    float_cols = ["f1", "recall", "precision", "acc", "threshold"]
    for c in float_cols:
        if c in show_df.columns:
            show_df[c] = show_df[c].map(lambda x: f"{x:.6f}" if pd.notna(x) else "NaN")

    print("\n" + "=" * 120)
    print(f"TOP {topk} RESULTS (fixed-threshold test metrics)")
    print("=" * 120)
    print(show_df.to_string(index=False))
    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_root",
        type=str,
        required=True,
        help="Example: /home/ted1204/MS_Detection/runs/hpo_stage1_contrast",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"run_root does not exist: {run_root}")

    df = collect_results(run_root)
    ranked = rank_results(df)

    print(f"[INFO] Found {len(ranked)} experiments under: {run_root}")
    print_topk(ranked, args.topk)

    if args.out_csv is not None:
        out_csv = Path(args.out_csv)
    else:
        out_csv = run_root / f"top{args.topk}_fixed_threshold_ranking.csv"

    ranked.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[INFO] Saved full ranking CSV: {out_csv}")


if __name__ == "__main__":
    main()