#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/analysis/analyze.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.manifold import TSNE


# =========================================================
# config utils
# =========================================================
class AttrDict(dict):
    def __getattr__(self, item):
        v = self.get(item)
        if isinstance(v, dict) and not isinstance(v, AttrDict):
            v = AttrDict(v)
            self[item] = v
        return v

    def __setattr__(self, key, value):
        self[key] = value


def _to_attrdict(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    return obj


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(base_cfg_path: str | Path, stage_cfg_path: str | Path):
    base_cfg = _load_yaml(base_cfg_path)
    stage_cfg = _load_yaml(stage_cfg_path)
    merged = _deep_update(base_cfg, stage_cfg)
    return _to_attrdict(merged)


def cfg_get(cfg: Any, *keys: str, default=None):
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
    return cur


# =========================================================
# basic utils
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json_if_exists(path: Optional[str | Path]) -> Optional[dict]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        print(f"[WARN] json not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def to_binary_int(arr: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(arr)

    if np.issubdtype(x.dtype, np.number):
        return x.astype(np.int64)

    mapping = {
        "0": 0,
        "1": 1,
        "noise": 0,
        "event": 1,
        "normal": 0,
        "anomaly": 1,
        "false": 0,
        "true": 1,
    }

    out = []
    for v in x:
        s = str(v).strip().lower()

        if s.startswith("tensor(") and s.endswith(")"):
            inner = s[len("tensor("):-1].strip()
            try:
                out.append(int(float(inner)))
                continue
            except ValueError:
                pass

        try:
            out.append(int(float(s)))
            continue
        except ValueError:
            pass

        if s in mapping:
            out.append(mapping[s])
            continue

        raise ValueError(f"Unsupported label value: {v}")

    return np.asarray(out, dtype=np.int64)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_id_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "npy_path", "path", "file_path", "filename", "file_name",
        "sample_id", "id", "batch_idx", "sample_idx"
    ]
    return [c for c in candidates if c in df.columns]


def validate_prediction_df(df: pd.DataFrame) -> None:
    if "label" not in df.columns:
        raise ValueError("test_predictions.csv must contain 'label'")

    required_score_cols = ["anomaly_score", "fc_prob", "fc_logit"]
    existing_score_cols = [c for c in required_score_cols if c in df.columns]
    if not existing_score_cols:
        raise ValueError(
            f"test_predictions.csv must contain at least one of {required_score_cols}"
        )

    required_pred_cols = ["pred_anomaly", "pred_fc", "pred_or", "pred_and"]
    existing_pred_cols = [c for c in required_pred_cols if c in df.columns]
    if not existing_pred_cols:
        raise ValueError(
            f"test_predictions.csv must contain at least one of {required_pred_cols}"
        )


# =========================================================
# plotting
# =========================================================
def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["noise", "event"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_score_histogram(
    y_true: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(scores[y_true == 0], bins=50, alpha=0.6, label="noise")
    ax.hist(scores[y_true == 1], bins=50, alpha=0.6, label="event")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
    title: str,
) -> Dict[str, float]:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"pr_auc": float(pr_auc)}


def save_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
    title: str,
) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"roc_auc": float(roc_auc)}


# =========================================================
# threshold sweep
# =========================================================
def threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        y_pred = (scores >= th).astype(np.int64)
        row = compute_binary_metrics(y_true, y_pred)
        row["threshold"] = float(th)
        rows.append(row)
    return pd.DataFrame(rows)


def save_threshold_plot(
    df_th: pd.DataFrame,
    out_png: Path,
    title: str,
    best_metric: str = "f1",
) -> dict:
    best_idx = df_th[best_metric].idxmax()
    best_row = df_th.loc[best_idx].to_dict()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_th["threshold"], df_th["precision"], label="precision")
    ax.plot(df_th["threshold"], df_th["recall"], label="recall")
    ax.plot(df_th["threshold"], df_th["f1"], label="f1")
    ax.axvline(
        best_row["threshold"],
        linestyle="--",
        label=f"best_{best_metric}={best_row['threshold']:.4f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return best_row


# =========================================================
# error export
# =========================================================
def save_error_cases(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_fp_csv: Path,
    out_fn_csv: Path,
) -> Tuple[int, int]:
    fp_df = df[(y_true == 0) & (y_pred == 1)].copy()
    fn_df = df[(y_true == 1) & (y_pred == 0)].copy()

    fp_df.to_csv(out_fp_csv, index=False)
    fn_df.to_csv(out_fn_csv, index=False)
    return len(fp_df), len(fn_df)


# =========================================================
# optional tsne
# =========================================================
def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"embedding file not found: {path}")

    if path.suffix == ".npy":
        emb = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        emb = data["embeddings"] if "embeddings" in data.files else data[data.files[0]]
    else:
        raise ValueError(f"Unsupported embedding extension: {path.suffix}")

    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape={emb.shape}")
    return emb


def save_tsne_plot(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    out_png: Path,
    max_points: int = 3000,
) -> None:
    if len(embeddings) != len(y_true):
        raise ValueError(f"embeddings rows ({len(embeddings)}) != labels rows ({len(y_true)})")

    if len(embeddings) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings), size=max_points, replace=False)
        embeddings = embeddings[idx]
        y_true = y_true[idx]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    xy = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xy[y_true == 0, 0], xy[y_true == 0, 1], s=10, alpha=0.6, label="noise")
    ax.scatter(xy[y_true == 1, 0], xy[y_true == 1, 1], s=10, alpha=0.6, label="event")
    ax.set_title("t-SNE of latent embeddings")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# analysis blocks
# =========================================================
def analyze_prediction_column(df: pd.DataFrame, pred_col: str, out_dir: Path) -> dict:
    mode_dir = out_dir / pred_col
    ensure_dir(mode_dir)

    y_true = to_binary_int(df["label"])
    y_pred = to_binary_int(df[pred_col])

    metrics = compute_binary_metrics(y_true, y_pred)
    save_json(metrics, mode_dir / "metrics.json")

    save_confusion_matrix(
        y_true,
        y_pred,
        mode_dir / "confusion_matrix.png",
        title=f"Confusion Matrix ({pred_col})",
    )

    fp_n, fn_n = save_error_cases(
        df=df,
        y_true=y_true,
        y_pred=y_pred,
        out_fp_csv=mode_dir / "false_positive.csv",
        out_fn_csv=mode_dir / "false_negative.csv",
    )

    summary = {
        "pred_col": pred_col,
        **metrics,
        "false_positive_rows": fp_n,
        "false_negative_rows": fn_n,
    }
    save_json(summary, mode_dir / "summary.json")
    return summary


def analyze_score_threshold(
    df: pd.DataFrame,
    score_col: str,
    out_dir: Path,
    n_thresholds: int,
) -> dict:
    y_true = to_binary_int(df["label"])
    scores = df[score_col].astype(float).to_numpy()

    mode_dir = out_dir / f"{score_col}_analysis"
    ensure_dir(mode_dir)

    save_score_histogram(
        y_true,
        scores,
        mode_dir / "score_histogram.png",
        title=f"Score Distribution ({score_col})",
        xlabel=score_col,
    )

    pr_info = save_pr_curve(
        y_true,
        scores,
        mode_dir / "pr_curve.png",
        title=f"PR Curve ({score_col})",
    )

    roc_info = save_roc_curve(
        y_true,
        scores,
        mode_dir / "roc_curve.png",
        title=f"ROC Curve ({score_col})",
    )

    smin, smax = float(np.min(scores)), float(np.max(scores))
    thresholds = np.array([smin]) if np.isclose(smin, smax) else np.linspace(smin, smax, n_thresholds)

    df_th = threshold_sweep(y_true, scores, thresholds)
    df_th.to_csv(mode_dir / "threshold_sweep.csv", index=False)

    best_row = save_threshold_plot(
        df_th,
        mode_dir / "threshold_sweep.png",
        title=f"Threshold Sweep ({score_col})",
        best_metric="f1",
    )
    save_json(best_row, mode_dir / "best_threshold_by_f1.json")

    y_pred_best = (scores >= float(best_row["threshold"])).astype(np.int64)

    save_confusion_matrix(
        y_true,
        y_pred_best,
        mode_dir / "confusion_matrix_best_f1.png",
        title=f"Confusion Matrix ({score_col}, best F1 threshold)",
    )

    fp_n, fn_n = save_error_cases(
        df=df.assign(**{f"{score_col}_best_pred": y_pred_best}),
        y_true=y_true,
        y_pred=y_pred_best,
        out_fp_csv=mode_dir / "false_positive_best_f1.csv",
        out_fn_csv=mode_dir / "false_negative_best_f1.csv",
    )

    summary = {
        "score_col": score_col,
        "score_min": smin,
        "score_max": smax,
        "best_threshold_by_f1": best_row,
        "pr_auc": pr_info["pr_auc"],
        "roc_auc": roc_info["roc_auc"],
        "false_positive_rows_best_f1": fp_n,
        "false_negative_rows_best_f1": fn_n,
    }
    save_json(summary, mode_dir / "summary.json")
    return summary


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze DAS microseismic test results")
    parser.add_argument("--base_cfg", type=str, default="configs/train/base.yaml")
    parser.add_argument("--stage_cfg", type=str, default="configs/train/analyze.yaml")
    args = parser.parse_args()

    cfg = load_config(args.base_cfg, args.stage_cfg)

    exp_name = cfg_get(cfg, "data", "experiment", default="default")
    run_root = Path(cfg_get(cfg, "paths", "run_root", default="./runs"))
    test_root = run_root / "test"
    save_dir = Path(cfg_get(cfg, "test", "save_dir", default=test_root / exp_name))

    pred_csv = cfg_get(cfg, "analyze", "pred_csv", default=None)
    metrics_json = cfg_get(cfg, "analyze", "metrics_json", default=None)
    embedding_path = cfg_get(cfg, "analyze", "embedding_path", default=None)
    out_dir = cfg_get(cfg, "analyze", "out_dir", default=None)

    if pred_csv is None:
        pred_csv = save_dir / "test_predictions.csv"
    else:
        pred_csv = Path(pred_csv)

    if metrics_json is None:
        metrics_json = save_dir / "test_metrics_fixed_threshold.json"
    else:
        metrics_json = Path(metrics_json)

    if out_dir is None:
        out_dir = save_dir / "analysis"
    else:
        out_dir = Path(out_dir)

    save_tsne = bool(cfg_get(cfg, "analyze", "save_tsne", default=False))
    tsne_max_points = int(cfg_get(cfg, "analyze", "tsne_max_points", default=3000))
    threshold_points = int(cfg_get(cfg, "analyze", "threshold_points", default=201))

    ensure_dir(out_dir)

    if not pred_csv.exists():
        raise FileNotFoundError(f"test_predictions.csv not found: {pred_csv}")

    df = pd.read_csv(pred_csv)
    validate_prediction_df(df)

    metrics_from_test = load_json_if_exists(metrics_json)
    if metrics_from_test is not None:
        save_json(metrics_from_test, out_dir / "metrics_from_test_json.json")

    y_true = to_binary_int(df["label"])

    overview = {
        "pred_csv": str(pred_csv),
        "metrics_json": str(metrics_json) if metrics_json is not None else None,
        "out_dir": str(out_dir),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "id_columns_detected": find_id_columns(df),
    }

    for pred_col in ["pred_anomaly", "pred_fc", "pred_or", "pred_and"]:
        if pred_col in df.columns:
            overview[f"{pred_col}_summary"] = analyze_prediction_column(df, pred_col, out_dir)

    for score_col in ["anomaly_score", "fc_prob", "fc_logit"]:
        if score_col in df.columns:
            overview[f"{score_col}_summary"] = analyze_score_threshold(
                df=df,
                score_col=score_col,
                out_dir=out_dir,
                n_thresholds=threshold_points,
            )

    save_json(overview, out_dir / "analysis_overview.json")

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        for k, v in overview.items():
            f.write(f"{k}: {v}\n")

    print(f"[DONE] pred_csv : {pred_csv}")
    print(f"[DONE] out_dir  : {out_dir}")


if __name__ == "__main__":
    main()