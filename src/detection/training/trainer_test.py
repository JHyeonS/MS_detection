#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.models.cnn_encoder import cnn_encoder
from src.detection.utils.config_io import (
    ensure_dir,
    save_merged_config,
    copy_config_snapshots,
    save_run_metadata,
)
from src.detection.utils.device import setup_device_from_cfg


# =========================================================
# config utils
# =========================================================
class AttrDict(dict):
    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(item)
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
# dataloader resolve
# =========================================================
def resolve_val_test_dataloaders(cfg):
    errors = []

    try:
        from src.detection.dataloader.finetune_dataloader import build_finetune_dataloaders
        out = build_finetune_dataloaders(cfg)
        if isinstance(out, tuple):
            if len(out) >= 3:
                return out[1], out[2]  # val, test
            if len(out) == 2:
                return out[0], out[1]
        errors.append("build_finetune_dataloaders did not return usable tuple")
    except Exception as e:
        errors.append(f"build_finetune_dataloaders: {repr(e)}")

    try:
        from src.detection.dataloader.test_dataloader import build_test_dataloader
        test_loader = build_test_dataloader(cfg)
        return None, test_loader
    except Exception as e:
        errors.append(f"build_test_dataloader: {repr(e)}")

    raise ImportError("Could not resolve val/test dataloaders.\n" + "\n".join(errors))


# =========================================================
# model
# =========================================================
class TestMSDNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = cnn_encoder(cfg)
        self.latent_dim = int(cfg_get(cfg, "model", "encoder", "latent_dim", default=128))
        self.head = nn.Linear(self.latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logit = self.head(z)
        return z, logit


def parse_batch(batch):
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]
        if len(batch) >= 2:
            return batch[0], batch[1], None
        raise ValueError("Batch tuple/list must have at least 2 items.")
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("input", batch.get("waveform", None)))
        y = batch.get("y", batch.get("label", batch.get("target", None)))
        meta = batch.get("meta", None)
        if x is None or y is None:
            raise ValueError(f"Unsupported batch dict keys: {list(batch.keys())}")
        return x, y, meta
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def load_finetuned_model_and_center(model: TestMSDNet, ckpt_path: str | Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt)}")

    state_dict = ckpt.get("model_state_dict", ckpt)
    center_c = ckpt.get("center_c", None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] loaded finetuned model from: {ckpt_path}")
    print(f"[INFO] load missing keys   : {len(missing)}")
    print(f"[INFO] load unexpected keys: {len(unexpected)}")

    if center_c is None:
        raise ValueError("Finetune checkpoint does not contain 'center_c'.")

    return center_c.to(device).float(), ckpt


# =========================================================
# metrics utils
# =========================================================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def collect_predictions(model, loader, device, center_c):
    model.eval()
    rows = []

    for batch_idx, batch in enumerate(loader):
        x, y, meta = parse_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1).long()

        z, logit = model(x)
        fc_logit = logit.view(-1)
        fc_prob = torch.sigmoid(fc_logit)
        anomaly_score = torch.sum((z - center_c.unsqueeze(0)) ** 2, dim=1)

        bs = x.size(0)
        for i in range(bs):
            row = {
                "batch_idx": batch_idx,
                "sample_idx": i,
                "label": int(y[i].item()),
                "fc_logit": float(fc_logit[i].item()),
                "fc_prob": float(fc_prob[i].item()),
                "anomaly_score": float(anomaly_score[i].item()),
            }

            if meta is not None:
                if isinstance(meta, dict):
                    for k, v in meta.items():
                        try:
                            row[f"meta_{k}"] = v[i] if hasattr(v, "__len__") and len(v) == bs else v
                        except Exception:
                            row[f"meta_{k}"] = str(v)
                else:
                    row["meta"] = str(meta)

            rows.append(row)

    return pd.DataFrame(rows)


def sweep_threshold_by_f1(pred_df: pd.DataFrame, score_col: str):
    if pred_df.empty:
        raise ValueError("Prediction dataframe is empty.")

    y_true = pred_df["label"].to_numpy().astype(int)
    scores = pred_df[score_col].to_numpy().astype(float)

    thresholds = np.unique(scores)
    best = None
    best_f1 = -1.0

    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        metrics["threshold"] = float(th)
        metrics["score_col"] = score_col

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best = metrics

    if best is None:
        raise RuntimeError(f"Failed to sweep thresholds for score_col={score_col}")

    return best


def apply_threshold(pred_df: pd.DataFrame, score_col: str, threshold: float, pred_col: str):
    y_true = pred_df["label"].to_numpy().astype(int)
    scores = pred_df[score_col].to_numpy().astype(float)
    y_pred = (scores >= threshold).astype(int)

    out_df = pred_df.copy()
    out_df[pred_col] = y_pred

    metrics = compute_metrics(y_true, y_pred)
    metrics["threshold"] = float(threshold)
    metrics["score_col"] = score_col
    metrics["pred_col"] = pred_col
    return out_df, metrics


def make_combined_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pred_anomaly" in out.columns and "pred_fc" in out.columns:
        out["pred_or"] = ((out["pred_anomaly"].astype(int) == 1) | (out["pred_fc"].astype(int) == 1)).astype(int)
        out["pred_and"] = ((out["pred_anomaly"].astype(int) == 1) & (out["pred_fc"].astype(int) == 1)).astype(int)
    return out


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cfg", type=str, required=True)
    parser.add_argument("--stage_cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.base_cfg, args.stage_cfg)

    run_root = Path(cfg_get(cfg, "paths", "run_root", default="./runs"))
    experiment = str(cfg_get(cfg, "data", "experiment", default="default_exp"))
    save_dir = run_root / "test" / experiment
    ensure_dir(save_dir)

    save_merged_config(cfg, save_dir)
    copy_config_snapshots(
        base_cfg_path=args.base_cfg,
        stage_cfg_path=args.stage_cfg,
        save_dir=save_dir / "config_snapshot",
    )
    save_run_metadata({"task": "test", "experiment": experiment}, save_dir)

    device = setup_device_from_cfg(cfg)
    print(f"[INFO] device: {device}")
    print(f"[INFO] save_dir: {save_dir}")

    val_loader, test_loader = resolve_val_test_dataloaders(cfg)
    if test_loader is None:
        raise ValueError("test_loader is None. test evaluation cannot proceed.")

    model = TestMSDNet(cfg).to(device)

    ckpt_path = cfg_get(cfg, "test", "checkpoint_path", default=None)
    if ckpt_path is None:
        ckpt_path = run_root / "finetune" / experiment / "best.pt"

    center_c, _ = load_finetuned_model_and_center(model, ckpt_path, device)

    # optional fixed thresholds from config
    anomaly_threshold = cfg_get(cfg, "test", "anomaly_score_threshold", default=None)
    fc_threshold = cfg_get(cfg, "test", "fc_prob_threshold", default=None)
    if fc_threshold is None:
        fc_threshold = cfg_get(cfg, "test", "prob_threshold", default=None)

    val_threshold_summary = {
        "anomaly": None,
        "fc": None,
    }

    # -------------------------------------------------
    # 1) Validation threshold tuning for BOTH branches
    # -------------------------------------------------
    if val_loader is not None:
        val_pred_df = collect_predictions(model, val_loader, device, center_c)
        val_pred_path = save_dir / "val_predictions.csv"
        val_pred_df.to_csv(val_pred_path, index=False)

        val_threshold_summary["anomaly"] = {
            "score_col": "anomaly_score",
            "score_min": float(val_pred_df["anomaly_score"].min()),
            "score_max": float(val_pred_df["anomaly_score"].max()),
            "best_threshold_by_f1": sweep_threshold_by_f1(val_pred_df, score_col="anomaly_score"),
        }
        val_threshold_summary["fc"] = {
            "score_col": "fc_prob",
            "score_min": float(val_pred_df["fc_prob"].min()),
            "score_max": float(val_pred_df["fc_prob"].max()),
            "best_threshold_by_f1": sweep_threshold_by_f1(val_pred_df, score_col="fc_prob"),
        }

        anomaly_threshold = float(val_threshold_summary["anomaly"]["best_threshold_by_f1"]["threshold"])
        fc_threshold = float(val_threshold_summary["fc"]["best_threshold_by_f1"]["threshold"])

        val_summary_path = save_dir / "val_threshold_summary.json"
        with open(val_summary_path, "w", encoding="utf-8") as f:
            json.dump(val_threshold_summary, f, indent=2)

        print(f"[INFO] saved val predictions       : {val_pred_path}")
        print(f"[INFO] saved val threshold summary: {val_summary_path}")
        print(f"[INFO] selected anomaly threshold : {anomaly_threshold:.6f}")
        print(f"[INFO] selected fc threshold      : {fc_threshold:.6f}")

    if anomaly_threshold is None:
        raise ValueError("No anomaly threshold available. Need val_loader or cfg['test']['anomaly_score_threshold'].")

    if fc_threshold is None:
        raise ValueError("No fc threshold available. Need val_loader or cfg['test']['fc_prob_threshold'].")

    # -------------------------------------------------
    # 2) Test with fixed thresholds for BOTH branches
    # -------------------------------------------------
    test_pred_df = collect_predictions(model, test_loader, device, center_c)

    test_pred_df, anomaly_metrics = apply_threshold(
        test_pred_df,
        score_col="anomaly_score",
        threshold=float(anomaly_threshold),
        pred_col="pred_anomaly",
    )
    test_pred_df, fc_metrics = apply_threshold(
        test_pred_df,
        score_col="fc_prob",
        threshold=float(fc_threshold),
        pred_col="pred_fc",
    )

    test_pred_df = make_combined_predictions(test_pred_df)

    y_true = test_pred_df["label"].to_numpy().astype(int)
    or_metrics = compute_metrics(y_true, test_pred_df["pred_or"].to_numpy().astype(int))
    and_metrics = compute_metrics(y_true, test_pred_df["pred_and"].to_numpy().astype(int))

    or_metrics["pred_col"] = "pred_or"
    or_metrics["rule"] = "pred_anomaly OR pred_fc"
    and_metrics["pred_col"] = "pred_and"
    and_metrics["rule"] = "pred_anomaly AND pred_fc"

    summary = {
        "thresholds": {
            "anomaly_score": float(anomaly_threshold),
            "fc_prob": float(fc_threshold),
        },
        "anomaly_metrics_fixed_threshold": anomaly_metrics,
        "fc_metrics_fixed_threshold": fc_metrics,
        "or_metrics_fixed_threshold": or_metrics,
        "and_metrics_fixed_threshold": and_metrics,
    }

    test_pred_path = save_dir / "test_predictions.csv"
    with open(save_dir / "test_metrics_fixed_threshold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    test_pred_df.to_csv(test_pred_path, index=False)

    print(f"[INFO] saved test predictions: {test_pred_path}")
    print(f"[INFO] saved test metrics    : {save_dir / 'test_metrics_fixed_threshold.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()