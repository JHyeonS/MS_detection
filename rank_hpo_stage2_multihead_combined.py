#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd


def parse_experiment_name(exp_name: str) -> Dict[str, Any]:
    out = {
        "method": None, "num_layers": None, "latent_dim": None,
        "bp_low": None, "bp_high": None, "seed": None,
        "stage2_cfg": None, "freeze_encoder": None, "lr": None, "anomaly_weight": None,
    }
    if "_contrast_" in exp_name:
        out["method"] = "contrast"
    elif "_reconst_" in exp_name:
        out["method"] = "reconst"

    if "__" in exp_name:
        base_exp, stage2_cfg = exp_name.split("__", 1)
        out["stage2_cfg"] = stage2_cfg
    else:
        base_exp = exp_name

    m = re.search(r"_L(\d+)_D(\d+)_BP(\d+)_(\d+)(?:_S(\d+))?$", base_exp)
    if m:
        out["num_layers"] = int(m.group(1))
        out["latent_dim"] = int(m.group(2))
        out["bp_low"] = int(m.group(3))
        out["bp_high"] = int(m.group(4))
        if m.group(5) is not None:
            out["seed"] = int(m.group(5))

    if out["stage2_cfg"] is not None:
        s = out["stage2_cfg"]
        if "_freeze_" in s:
            out["freeze_encoder"] = True
        elif "_unfreeze_" in s:
            out["freeze_encoder"] = False

        m_lr = re.search(r"_lr([0-9eE\-\+\.]+)_aw", s)
        if m_lr:
            out["lr"] = m_lr.group(1)

        m_aw = re.search(r"_aw([0-9p]+)$", s)
        if m_aw:
            out["anomaly_weight"] = m_aw.group(1).replace("p", ".")
    return out


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def _extract_test_branch(js: dict, branch: str) -> Optional[Dict[str, Any]]:
    branch_map = {
        "anomaly": "anomaly_metrics_fixed_threshold",
        "fc": "fc_metrics_fixed_threshold",
        "or": "or_metrics_fixed_threshold",
        "and": "and_metrics_fixed_threshold",
    }
    key = branch_map.get(branch)
    if key is None:
        raise ValueError(f"Invalid test branch: {branch}")
    sub = js.get(key)
    if sub is None:
        return None
    thresholds = js.get("thresholds", {})
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
        "score_col": sub.get("score_col"),
        "pred_col": sub.get("pred_col"),
        "rule": sub.get("rule"),
        "threshold": sub.get("threshold"),
        "threshold_anomaly_score": thresholds.get("anomaly_score"),
        "threshold_fc_prob": thresholds.get("fc_prob"),
    }


def _extract_analyze_branch(js: dict, branch: str) -> Optional[Dict[str, Any]]:
    pred_map = {
        "anomaly": "pred_anomaly_summary",
        "fc": "pred_fc_summary",
        "or": "pred_or_summary",
        "and": "pred_and_summary",
    }
    score_map = {
        "anomaly_score": "anomaly_score_summary",
        "fc_prob": "fc_prob_summary",
        "fc_logit": "fc_logit_summary",
    }

    if branch in pred_map:
        sub = js.get(pred_map[branch])
        if sub is None:
            return None
        return {
            "branch": branch, "kind": "pred",
            "acc": sub.get("acc"), "precision": sub.get("precision"),
            "recall": sub.get("recall"), "f1": sub.get("f1"),
            "tp": sub.get("tp"), "tn": sub.get("tn"),
            "fp": sub.get("fp"), "fn": sub.get("fn"),
            "pred_col": sub.get("pred_col"),
        }

    if branch in score_map:
        sub = js.get(score_map[branch])
        if sub is None:
            return None
        best = sub.get("best_threshold_by_f1", {})
        return {
            "branch": branch, "kind": "score",
            "acc": best.get("acc"), "precision": best.get("precision"),
            "recall": best.get("recall"), "f1": best.get("f1"),
            "tp": best.get("tp"), "tn": best.get("tn"),
            "fp": best.get("fp"), "fn": best.get("fn"),
            "threshold": best.get("threshold"),
            "score_col": sub.get("score_col"),
            "score_min": sub.get("score_min"),
            "score_max": sub.get("score_max"),
            "pr_auc": sub.get("pr_auc"),
            "roc_auc": sub.get("roc_auc"),
        }

    raise ValueError(f"Invalid analyze branch: {branch}")


def load_result(json_path: Path, source: str, branch: str, run_root: Path) -> Optional[Dict[str, Any]]:
    js = _safe_read_json(json_path)
    if js is None:
        return None

    exp_name = json_path.parent.name if source == "test" else json_path.parent.parent.name
    row = {
        "experiment": exp_name,
        "metrics_path": str(json_path),
        "source": source,
        "run_root": str(run_root),
        "run_group": run_root.name,
    }

    extracted = _extract_test_branch(js, branch) if source == "test" else _extract_analyze_branch(js, branch)
    if extracted is None:
        return None

    row.update(extracted)
    row.update(parse_experiment_name(exp_name))
    return row


def collect_results(run_roots: List[Path], source: str, branch: str) -> pd.DataFrame:
    rows = []

    for run_root in run_roots:
        if source == "test":
            paths = sorted(run_root.glob("test/*/test_metrics_fixed_threshold.json"))
        elif source == "analyze":
            paths = sorted(run_root.glob("test/*/analysis/analysis_overview.json"))
        else:
            raise ValueError(f"Unsupported source: {source}")

        for p in paths:
            row = load_result(p, source=source, branch=branch, run_root=run_root)
            if row is not None:
                rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No usable results found for source='{source}', branch='{branch}' "
            f"under run_roots={[str(r) for r in run_roots]}"
        )

    df = pd.DataFrame(rows)
    numeric_cols = [
        "acc", "precision", "recall", "f1", "tp", "tn", "fp", "fn",
        "threshold", "threshold_anomaly_score", "threshold_fc_prob",
        "num_layers", "latent_dim", "bp_low", "bp_high", "seed",
        "score_min", "score_max", "pr_auc", "roc_auc", "anomaly_weight",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def rank_results(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sort_by = [c for c in [metric, "recall", "precision", "acc"] if c in df.columns]
    ranked = df.sort_values(by=sort_by, ascending=[False] * len(sort_by), na_position="last").reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def print_topk(df: pd.DataFrame, topk: int, source: str, branch: str, metric: str):
    cols = [
        "rank", "experiment", "run_group", "method", "num_layers", "latent_dim",
        "bp_low", "bp_high", "seed", "freeze_encoder", "lr", "anomaly_weight",
        "branch", "f1", "recall", "precision", "acc", "threshold",
        "pr_auc", "roc_auc", "fp", "fn",
    ]
    cols = [c for c in cols if c in df.columns]
    show_df = df[cols].head(topk).copy()

    for c in ["f1", "recall", "precision", "acc", "threshold", "pr_auc", "roc_auc", "anomaly_weight"]:
        if c in show_df.columns:
            show_df[c] = show_df[c].map(lambda x: f"{x:.6f}" if pd.notna(x) else "NaN")

    print("\n" + "=" * 160)
    print(f"TOP {topk} RESULTS | source={source} | branch={branch} | rank_metric={metric}")
    print("=" * 160)
    print(show_df.to_string(index=False))
    print("=" * 160 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_roots", type=str, nargs="+", required=True,
                        help="One or more run_root paths, e.g. runs/hpo_stage2_contrast runs/hpo_stage2_reconst")
    parser.add_argument("--source", type=str, choices=["test", "analyze"], required=True)
    parser.add_argument("--branch", type=str, required=True,
                        help="test: anomaly/fc/or/and | analyze: anomaly/fc/or/and/anomaly_score/fc_prob/fc_logit")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rank_metric", type=str, default="f1", choices=["f1", "recall", "precision", "acc"])
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    run_roots = [Path(p) for p in args.run_roots]
    for r in run_roots:
        if not r.exists():
            raise FileNotFoundError(f"run_root does not exist: {r}")

    df = collect_results(run_roots, args.source, args.branch)
    ranked = rank_results(df, args.rank_metric)

    print(f"[INFO] Found {len(ranked)} experiments across {len(run_roots)} run_roots")
    print_topk(ranked, args.topk, args.source, args.branch, args.rank_metric)

    if args.out_csv is not None:
        out_csv = Path(args.out_csv)
    else:
        base_dir = run_roots[0].parent if len(run_roots) > 0 else Path(".")
        roots_name = "_".join([r.name for r in run_roots])
        out_csv = base_dir / f"combined_{roots_name}_{args.source}_{args.branch}_ranking.csv"

    ranked.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[INFO] Saved full ranking CSV: {out_csv}")


if __name__ == "__main__":
    main()
