#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_test_branch(test_summary: dict):
    candidates = []
    for key in [
        "anomaly_metrics_fixed_threshold",
        "fc_metrics_fixed_threshold",
        "or_metrics_fixed_threshold",
        "and_metrics_fixed_threshold",
    ]:
        metrics = test_summary.get(key)
        if isinstance(metrics, dict) and "f1" in metrics:
            candidates.append((key.replace("_metrics_fixed_threshold", ""), metrics))
    if not candidates:
        return "", {}
    return max(candidates, key=lambda item: item[1].get("f1", -1.0))


def parse_method_and_fraction(summary_path: Path, root: Path):
    rel = summary_path.relative_to(root)
    method = rel.parts[0]
    experiment = rel.parts[2] if len(rel.parts) >= 3 else ""
    fraction = ""
    if "__frac" in experiment:
        fraction = experiment.rsplit("__frac", 1)[1].replace("p", ".")
    return method, fraction, experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []

    for ft_summary_path in sorted(root.glob("*/finetune/*/finetune_summary.json")):
        method, fraction, experiment = parse_method_and_fraction(ft_summary_path, root)
        ft = load_json(ft_summary_path) or {}
        test_summary_path = root / method / "test" / experiment / "test_metrics_fixed_threshold.json"
        test = load_json(test_summary_path) or {}
        branch, metrics = best_test_branch(test) if test else ("", {})
        label_info = load_json(ft_summary_path.with_name("label_efficiency_info.json")) or {}

        rows.append(
            {
                "method": method,
                "fraction": fraction,
                "experiment": experiment,
                "best_epoch": ft.get("best_epoch"),
                "val_best_metric": ft.get("best_metric"),
                "completed_epochs": ft.get("completed_epochs"),
                "stopped_early": ft.get("stopped_early"),
                "train_rows": label_info.get("effective_num_rows"),
                "train_label_counts": json.dumps(label_info.get("per_class_effective", {}), sort_keys=True),
                "test_best_branch": branch,
                "test_f1": metrics.get("f1"),
                "test_acc": metrics.get("acc"),
                "test_precision": metrics.get("precision"),
                "test_recall": metrics.get("recall"),
                "test_specificity": metrics.get("specificity"),
                "test_balanced_acc": metrics.get("balanced_acc"),
                "test_tp": metrics.get("tp"),
                "test_tn": metrics.get("tn"),
                "test_fp": metrics.get("fp"),
                "test_fn": metrics.get("fn"),
                "finetune_summary": str(ft_summary_path),
                "test_summary": str(test_summary_path),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "fraction",
        "experiment",
        "best_epoch",
        "val_best_metric",
        "completed_epochs",
        "stopped_early",
        "train_rows",
        "train_label_counts",
        "test_best_branch",
        "test_f1",
        "test_acc",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_balanced_acc",
        "test_tp",
        "test_tn",
        "test_fp",
        "test_fn",
        "finetune_summary",
        "test_summary",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
