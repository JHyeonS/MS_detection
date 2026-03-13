#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/training/trainer_test.py

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.models.cnn_encoder import cnn_encoder
from src.utils.device import setup_device_from_cfg
from src.utils.config_io import ensure_dir, save_merged_config, copy_config_snapshots, save_run_metadata


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


def resolve_test_dataloader(cfg):
    errors = []
    candidates = [
        ("src.dataloader.finetune_dataloader", "build_finetune_dataloaders"),
        ("src.dataloader.test_dataloader", "build_test_dataloader"),
        ("src.dataloader.test_dataloader", "build_test_loader"),
    ]

    for module_name, fn_name in candidates:
        try:
            module = __import__(module_name, fromlist=[fn_name])
            fn = getattr(module, fn_name)
            out = fn(cfg)

            if isinstance(out, tuple):
                if len(out) >= 3:
                    return out[2]
                if len(out) == 2:
                    return out[1]
                if len(out) == 1:
                    return out[0]
            return out
        except Exception as e:
            errors.append(f"{module_name}.{fn_name}: {repr(e)}")

    msg = "\n".join(errors)
    raise ImportError("Could not resolve test dataloader builder.\n" + msg)


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


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
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
def main():
    parser = argparse.ArgumentParser(description="Test trainer for DAS microseismic detection")
    parser.add_argument("--base_cfg", type=str, default="config/base.yaml")
    parser.add_argument("--stage_cfg", type=str, default="config/test.yaml")
    args = parser.parse_args()

    cfg = load_config(args.base_cfg, args.stage_cfg)
    device = setup_device_from_cfg(cfg)

    exp_name = cfg_get(cfg, "data", "experiment", default="default")
    run_root = Path(cfg_get(cfg, "paths", "run_root", default="./runs"))
    test_root = run_root / "test"
    finetune_root = run_root / "finetune"

    save_dir = cfg_get(cfg, "test", "save_dir", default=None)
    if save_dir is None:
        save_dir = test_root / exp_name
    else:
        save_dir = Path(save_dir)
    ensure_dir(save_dir)
    print(f"[INFO] test save_dir: {save_dir}")

    finetune_ckpt = cfg_get(cfg, "test", "finetune_ckpt", default=None)
    if finetune_ckpt is None:
        finetune_ckpt = finetune_root / exp_name / "best.pt"
    else:
        finetune_ckpt = Path(finetune_ckpt)

    print(f"[INFO] finetune_ckpt: {finetune_ckpt}")
    if not finetune_ckpt.exists():
        raise FileNotFoundError(
            f"Finetune checkpoint not found: {finetune_ckpt}\n"
            f"Set cfg.test.finetune_ckpt explicitly or place best.pt under {finetune_ckpt.parent}/"
        )

    test_loader = resolve_test_dataloader(cfg)
    model = TestMSDNet(cfg).to(device)
    center_c, _ = load_finetuned_model_and_center(model, finetune_ckpt, device)
    model.eval()

    normal_label = int(cfg_get(cfg, "train", "normal_label", default=0))
    anomaly_label = int(cfg_get(cfg, "train", "anomaly_label", default=1))

    rows = []
    all_y_true = []
    all_y_pred = []

    for batch_idx, batch in enumerate(test_loader):
        x, y, meta = parse_batch(batch)
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long().view(-1)

        z, logits = model(x)
        probs = torch.sigmoid(logits.view(-1))
        pred = (probs >= 0.5).long()

        dist = torch.sum((z - center_c.unsqueeze(0)) ** 2, dim=1)

        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        prob_np = probs.detach().cpu().numpy()
        dist_np = dist.detach().cpu().numpy()

        all_y_true.append(y_np)
        all_y_pred.append(pred_np)

        for i in range(len(y_np)):
            row = {
                "batch_idx": batch_idx,
                "sample_idx": i,
                "label": int(y_np[i]),
                "pred": int(pred_np[i]),
                "prob_anomaly": float(prob_np[i]),
                "anomaly_score": float(dist_np[i]),
            }

            if isinstance(meta, list) and i < len(meta) and isinstance(meta[i], dict):
                row.update(meta[i])
            elif isinstance(meta, dict):
                for k, v in meta.items():
                    try:
                        row[k] = v[i]
                    except Exception:
                        row[k] = v

            rows.append(row)

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)

    mask = (y_true == normal_label) | (y_true == anomaly_label)
    y_true_bin = (y_true[mask] == anomaly_label).astype(np.int64)
    y_pred_bin = (y_pred[mask] == anomaly_label).astype(np.int64)

    metrics = binary_metrics(y_true_bin, y_pred_bin)

    pred_df = pd.DataFrame(rows)
    pred_csv = save_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_merged_config(cfg, save_dir)
    copy_config_snapshots(args.base_cfg, args.stage_cfg, save_dir)
    save_run_metadata(
        {
            "finetune_ckpt": str(finetune_ckpt),
            "device": str(device),
            "cwd": os.getcwd(),
        },
        save_dir,
    )

    print(f"[DONE] saved predictions: {pred_csv}")
    print(f"[DONE] saved metrics    : {metrics_path}")
    print(f"[DONE] metrics: {metrics}")


if __name__ == "__main__":
    main()
