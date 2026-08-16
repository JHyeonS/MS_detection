#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rank HPO results from finetune_summary.json files")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--metric", type=str, default="val_f1")
    parser.add_argument("--mode", type=str, default="max", choices=["max", "min"])
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for summary_path in root.rglob("finetune_summary.json"):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        last_val = summary.get("last_val_metrics") or {}
        metric_value = last_val.get(args.metric[len("val_"):]) if args.metric.startswith("val_") else summary.get(args.metric)
        rows.append(
            {
                "experiment": summary.get("experiment"),
                "metric": args.metric,
                "metric_value": metric_value,
                "best_epoch": summary.get("best_epoch"),
                "best_metric": summary.get("best_metric"),
                "save_dir": summary.get("save_dir"),
                "summary_path": str(summary_path),
            }
        )

    if args.mode == "max":
        rows.sort(key=lambda x: (x["metric_value"] is not None, x["metric_value"]), reverse=True)
    else:
        rows.sort(key=lambda x: (x["metric_value"] is None, x["metric_value"] if x["metric_value"] is not None else float("inf")))

    out_path = root / "ranked_hpo_results.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("experiment,metric,metric_value,best_epoch,best_metric,save_dir,summary_path\n")
        for row in rows:
            f.write(
                f"{row['experiment']},{row['metric']},{row['metric_value']},{row['best_epoch']},"
                f"{row['best_metric']},{row['save_dir']},{row['summary_path']}\n"
            )

    print(f"[DONE] wrote ranking to {out_path}")


if __name__ == "__main__":
    main()
