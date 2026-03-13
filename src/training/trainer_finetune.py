#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/training/trainer_finetune.py

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.models.cnn_encoder import cnn_encoder
from src.utils.visualize import save_loss_curve, save_train_history_csv
from src.utils.device import setup_device_from_cfg

from src.utils.config_io import (
    save_merged_config,
    copy_config_snapshots,
    save_run_metadata,
)

# -----------------------------------------------------------------------------
# config utils
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# misc utils
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# dataloader resolver
# -----------------------------------------------------------------------------
def resolve_finetune_dataloaders(cfg):
    """
    Tries a few common project-specific builder names so the script can fit into
    slightly different repo layouts.

    Expected return:
        train_loader, val_loader (val_loader can be None)
    """
    errors = []

    candidates = [
        ("src.dataloader.finetune_dataloader", "build_finetune_dataloaders"),
        ("src.dataloader.finetune_dataloader", "build_train_val_dataloaders"),
        ("src.dataloader.finetune_dataloader", "build_finetune_train_val_dataloaders"),
        ("src.dataloader.train_dataloader", "build_train_val_dataloaders"),
        ("src.dataloader.train_dataloader", "build_finetune_dataloaders"),
        ("src.dataloader.train_dataloader", "build_train_dataloader"),
    ]

    for module_name, fn_name in candidates:
        try:
            module = __import__(module_name, fromlist=[fn_name])
            fn = getattr(module, fn_name)
            out = fn(cfg)

            if isinstance(out, tuple):
                if len(out) == 2:
                    return out[0], out[1]
                if len(out) >= 3:
                    return out[0], out[1]
                raise ValueError(
                    f"{module_name}.{fn_name} returned tuple of len {len(out)}, expected >=2"
                )

            return out, None
        except Exception as e:
            errors.append(f"{module_name}.{fn_name}: {repr(e)}")

    msg = "\n".join(errors)
    raise ImportError(
        "Could not resolve finetune dataloader builder. "
        "Add one of the supported builder functions or adjust resolve_finetune_dataloaders().\n"
        f"Tried:\n{msg}"
    )


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
class FinetuneMSDNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = cnn_encoder(cfg)
        self.latent_dim = int(cfg_get(cfg, "model", "encoder", "latent_dim", default=128))
        self.head = nn.Linear(self.latent_dim, 1)

        freeze_encoder = bool(cfg_get(cfg, "train", "freeze_encoder", default=False))
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)      # (B, D)
        logit = self.head(z)     # (B, 1)
        return z, logit


# -----------------------------------------------------------------------------
# batch parsing
# -----------------------------------------------------------------------------
def parse_finetune_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Supported batch formats:
      - (x, y)
      - [x, y]
      - {"x": x, "y": y}
      - {"input": x, "label": y}
      - {"waveform": x, "target": y}
      - {"data": x, "labels": y}
    """
    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list must have at least 2 items: (x, y)")
        return batch[0], batch[1]

    if isinstance(batch, dict):
        x_keys = ["x", "input", "waveform", "data"]
        y_keys = ["y", "label", "target", "labels"]

        x = None
        y = None

        for k in x_keys:
            if k in batch:
                x = batch[k]
                break

        for k in y_keys:
            if k in batch:
                y = batch[k]
                break

        if x is None or y is None:
            raise ValueError(f"Unsupported batch dict keys: {list(batch.keys())}")
        return x, y

    raise TypeError(f"Unsupported batch type: {type(batch)}")


# -----------------------------------------------------------------------------
# checkpoint loading / center computation
# -----------------------------------------------------------------------------
def _strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


@torch.no_grad()
def compute_center_c_from_loader(
    model: FinetuneMSDNet,
    loader,
    device: torch.device,
    normal_label: int = 0,
    max_batches: Optional[int] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Recompute center_c from normal/noise samples in loader.

    Only samples with label == normal_label are used.
    This is mainly for contrastive pretrain checkpoints where center_c is not saved.
    """
    model.eval()
    z_list = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long().view(-1)

        normal_mask = (y == normal_label)
        if not normal_mask.any():
            continue

        x_normal = x[normal_mask]
        z, _ = model(x_normal)

        if z.ndim != 2:
            raise ValueError(f"Expected z shape (B, D), got {tuple(z.shape)}")

        z_list.append(z.detach().cpu())

    if len(z_list) == 0:
        raise ValueError(
            "No normal/noise samples were found in loader. "
            "Cannot compute center_c for contrastive pretrained encoder."
        )

    z_all = torch.cat(z_list, dim=0)   # (N, D)
    center_c = z_all.mean(dim=0)

    center_c[(center_c.abs() < eps) & (center_c < 0)] = -eps
    center_c[(center_c.abs() < eps) & (center_c >= 0)] = eps

    return center_c.to(device).float()


def load_pretrained_encoder_and_center(
    model: FinetuneMSDNet,
    ckpt_path: str | Path,
    device: torch.device,
    center_loader=None,
    normal_label: int = 0,
    max_center_batches: Optional[int] = None,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt)}")

    pretrain_mode = str(ckpt.get("mode", "unknown")).lower()
    center_c = ckpt.get("center_c", None)

    state_dict = ckpt.get("model_state_dict", None)
    if state_dict is None:
        # allow encoder-only checkpoint as fallback
        state_dict = ckpt

    encoder_state = _strip_prefix_if_present(state_dict, "encoder.")

    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)

    print(f"[INFO] loaded encoder from: {ckpt_path}")
    print(f"[INFO] encoder load missing keys   : {len(missing)}")
    print(f"[INFO] encoder load unexpected keys: {len(unexpected)}")
    print(f"[INFO] pretrain checkpoint mode    : {pretrain_mode}")

    # reconstruction pretrain: center already saved
    if center_c is not None:
        center_c = center_c.to(device).float()
        print(f"[INFO] loaded center_c from checkpoint: {tuple(center_c.shape)}")
        return center_c, ckpt

    # contrast pretrain (or encoder-only ckpt): recompute center from loader
    if center_loader is not None:
        print("[INFO] center_c is None in checkpoint. Recomputing center_c from loader...")
        center_c = compute_center_c_from_loader(
            model=model,
            loader=center_loader,
            device=device,
            normal_label=normal_label,
            max_batches=max_center_batches,
        )
        print(f"[INFO] computed center_c from loader: {tuple(center_c.shape)}")
        return center_c, ckpt

    raise ValueError(
        "Checkpoint does not contain 'center_c', and no center_loader was provided. "
        "For contrastive pretrain, pass a loader containing normal/noise samples."
    )


# -----------------------------------------------------------------------------
# losses / metrics
# -----------------------------------------------------------------------------
def compute_fcl_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    normal_label: int = 0,
    anomaly_label: int = 1,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Classification loss for labeled normal/anomaly only.
    Unlabeled samples are ignored.
    """
    logits = logits.view(-1)
    labels = labels.view(-1)

    mask = (labels == normal_label) | (labels == anomaly_label)
    num_labeled = int(mask.sum().item())

    if num_labeled == 0:
        return logits.new_zeros(()), {"num_labeled": 0}

    y_bin = (labels[mask] == anomaly_label).float()
    loss = F.binary_cross_entropy_with_logits(logits[mask], y_bin)
    return loss, {"num_labeled": num_labeled}


def compute_deep_sad_anomaly_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    center_c: torch.Tensor,
    normal_label: int = 0,
    anomaly_label: int = 1,
    unlabeled_label: int = 2,
    eta: float = 1.0,
    eps: float = 1e-6,
    treat_unlabeled_as_normal: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Deep SAD-style loss.
    - normal    : minimize distance to center
    - anomaly   : minimize inverse distance (push away from center)
    - unlabeled : ignored by default, optionally treated as normal
    """
    labels = labels.view(-1)
    dist = torch.sum((z - center_c.unsqueeze(0)) ** 2, dim=1)

    normal_mask = (labels == normal_label)
    anomaly_mask = (labels == anomaly_label)

    if treat_unlabeled_as_normal:
        normal_mask = normal_mask | (labels == unlabeled_label)

    loss_normal = dist[normal_mask].mean() if normal_mask.any() else dist.new_zeros(())
    loss_anomaly = (
        (1.0 / (dist[anomaly_mask] + eps)).mean()
        if anomaly_mask.any()
        else dist.new_zeros(())
    )

    loss = loss_normal + float(eta) * loss_anomaly

    stats = {
        "dist_mean": float(dist.mean().item()),
        "dist_normal": float(dist[normal_mask].mean().item()) if normal_mask.any() else 0.0,
        "dist_anomaly": float(dist[anomaly_mask].mean().item()) if anomaly_mask.any() else 0.0,
        "n_normal": int(normal_mask.sum().item()),
        "n_anomaly": int(anomaly_mask.sum().item()),
    }
    return loss, stats


def binary_classification_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    normal_label: int = 0,
    anomaly_label: int = 1,
):
    logits = logits.view(-1)
    labels = labels.view(-1)

    mask = (labels == normal_label) | (labels == anomaly_label)
    if not mask.any():
        return {
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_labeled": 0,
        }

    y_true = (labels[mask] == anomaly_label).long()
    y_pred = (torch.sigmoid(logits[mask]) >= 0.5).long()

    tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
    fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true == 1)).sum().item())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_labeled": int(mask.sum().item()),
    }


# -----------------------------------------------------------------------------
# optimizer
# -----------------------------------------------------------------------------
def build_optimizer(cfg, model: nn.Module):
    lr = float(cfg_get(cfg, "train", "lr", default=1e-3))
    weight_decay = float(cfg_get(cfg, "train", "wd", default=0.0))
    name = str(cfg_get(cfg, "train", "optimizer", default="adamw")).lower()

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg_get(cfg, "train", "momentum", default=0.9))
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


# -----------------------------------------------------------------------------
# train / eval loops
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    center_c: torch.Tensor,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
    lambda_anomaly: float,
    eta: float,
    grad_clip: Optional[float],
    normal_label: int,
    anomaly_label: int,
    unlabeled_label: int,
    treat_unlabeled_as_normal: bool,
) -> Dict[str, float]:
    model.train()

    running_total = 0.0
    running_fcl = 0.0
    running_anom = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    running_dist = 0.0
    n_batches = 0

    for batch in loader:
        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long().view(-1)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=use_amp):
                z, logits = model(x)
                loss_fcl, _ = compute_fcl_loss(
                    logits=logits,
                    labels=y,
                    normal_label=normal_label,
                    anomaly_label=anomaly_label,
                )
                loss_anom, anom_stats = compute_deep_sad_anomaly_loss(
                    z=z,
                    labels=y,
                    center_c=center_c,
                    normal_label=normal_label,
                    anomaly_label=anomaly_label,
                    unlabeled_label=unlabeled_label,
                    eta=eta,
                    treat_unlabeled_as_normal=treat_unlabeled_as_normal,
                )
                loss = loss_fcl + float(lambda_anomaly) * loss_anom

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

        else:
            z, logits = model(x)
            loss_fcl, _ = compute_fcl_loss(
                logits=logits,
                labels=y,
                normal_label=normal_label,
                anomaly_label=anomaly_label,
            )
            loss_anom, anom_stats = compute_deep_sad_anomaly_loss(
                z=z,
                labels=y,
                center_c=center_c,
                normal_label=normal_label,
                anomaly_label=anomaly_label,
                unlabeled_label=unlabeled_label,
                eta=eta,
                treat_unlabeled_as_normal=treat_unlabeled_as_normal,
            )
            loss = loss_fcl + float(lambda_anomaly) * loss_anom
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        cls_stats = binary_classification_stats(
            logits.detach(),
            y.detach(),
            normal_label=normal_label,
            anomaly_label=anomaly_label,
        )

        running_total += float(loss.item())
        running_fcl += float(loss_fcl.item())
        running_anom += float(loss_anom.item())
        running_acc += cls_stats["acc"]
        running_f1 += cls_stats["f1"]
        running_dist += anom_stats["dist_mean"]
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": running_total / n,
        "loss_fcl": running_fcl / n,
        "loss_anomaly": running_anom / n,
        "acc": running_acc / n,
        "f1": running_f1 / n,
        "dist_mean": running_dist / n,
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    center_c: torch.Tensor,
    lambda_anomaly: float,
    eta: float,
    normal_label: int,
    anomaly_label: int,
    unlabeled_label: int,
    treat_unlabeled_as_normal: bool,
) -> Dict[str, float]:
    model.eval()

    running_total = 0.0
    running_fcl = 0.0
    running_anom = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    running_dist = 0.0
    n_batches = 0

    for batch in loader:
        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long().view(-1)

        z, logits = model(x)
        loss_fcl, _ = compute_fcl_loss(
            logits=logits,
            labels=y,
            normal_label=normal_label,
            anomaly_label=anomaly_label,
        )
        loss_anom, anom_stats = compute_deep_sad_anomaly_loss(
            z=z,
            labels=y,
            center_c=center_c,
            normal_label=normal_label,
            anomaly_label=anomaly_label,
            unlabeled_label=unlabeled_label,
            eta=eta,
            treat_unlabeled_as_normal=treat_unlabeled_as_normal,
        )
        loss = loss_fcl + float(lambda_anomaly) * loss_anom

        cls_stats = binary_classification_stats(
            logits,
            y,
            normal_label=normal_label,
            anomaly_label=anomaly_label,
        )

        running_total += float(loss.item())
        running_fcl += float(loss_fcl.item())
        running_anom += float(loss_anom.item())
        running_acc += cls_stats["acc"]
        running_f1 += cls_stats["f1"]
        running_dist += anom_stats["dist_mean"]
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": running_total / n,
        "loss_fcl": running_fcl / n,
        "loss_anomaly": running_anom / n,
        "acc": running_acc / n,
        "f1": running_f1 / n,
        "dist_mean": running_dist / n,
    }


# -----------------------------------------------------------------------------
# checkpoint utils
# -----------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    center_c: Optional[torch.Tensor],
    cfg: Any,
):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "center_c": None if center_c is None else center_c.detach().cpu(),
        "train_mode": "finetune",
        "pretrained_ckpt": cfg_get(cfg, "train", "pretrained_ckpt", default=None),
    }
    torch.save(ckpt, path)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning trainer for DAS microseismic detection"
    )
    parser.add_argument("--base_cfg", type=str, default="config/base.yaml")
    parser.add_argument("--stage_cfg", type=str, default="config/train.yaml")
    args = parser.parse_args()

    cfg = load_config(args.base_cfg, args.stage_cfg)

    seed = int(cfg_get(cfg, "seed", default=42))
    set_seed(seed)

    device = setup_device_from_cfg(cfg)

    exp_name = cfg_get(cfg, "data", "experiment", default="default")

    save_dir = cfg_get(cfg, "train", "save_dir", default=None)
    if save_dir is None:
        finetune_root = cfg_get(cfg, "paths", "finetune_root", default=None)

        if finetune_root is None:
            run_root = cfg_get(cfg, "paths", "run_root", default="./runs")
            finetune_root = Path(run_root) / "finetune"
        else:
            finetune_root = Path(finetune_root)

        save_dir = finetune_root / exp_name
    else:
        save_dir = Path(save_dir)

    ensure_dir(save_dir)

    save_merged_config(cfg, save_dir)
    copy_config_snapshots(args.base_cfg, args.stage_cfg, save_dir)
    save_run_metadata(
        {
            "stage": "finetune",
            "device": str(device),
        },
        save_dir,
    )

    print(f"[INFO] finetune save_dir: {save_dir}")

    pretrained_ckpt = cfg_get(cfg, "train", "pretrained_ckpt", default=None)
    if pretrained_ckpt is None:
        run_root = cfg_get(cfg, "paths", "run_root", default="./runs")
        pretrained_ckpt = Path(run_root) / "pretrain" / exp_name / "best.pt"
    else:
        pretrained_ckpt = Path(pretrained_ckpt)

    print(f"[INFO] pretrained_ckpt: {pretrained_ckpt}")

    if not pretrained_ckpt.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {pretrained_ckpt}\n"
            f"Set cfg.train.pretrained_ckpt explicitly or place best.pt under "
            f"{pretrained_ckpt.parent}/"
        )

    train_loader, val_loader = resolve_finetune_dataloaders(cfg)
    print(f"[INFO] val loader: {'enabled' if val_loader is not None else 'disabled'}")

    model = FinetuneMSDNet(cfg).to(device)
    print(f"[INFO] model params (trainable): {count_parameters(model):,}")

    normal_label = int(cfg_get(cfg, "train", "normal_label", default=0))
    anomaly_label = int(cfg_get(cfg, "train", "anomaly_label", default=1))
    unlabeled_label = int(cfg_get(cfg, "train", "unlabeled_label", default=2))
    treat_unlabeled_as_normal = bool(
        cfg_get(cfg, "train", "treat_unlabeled_as_normal", default=False)
    )
    max_center_batches = cfg_get(cfg, "train", "max_center_batches", default=None)

    center_c, pretrain_ckpt_obj = load_pretrained_encoder_and_center(
        model=model,
        ckpt_path=pretrained_ckpt,
        device=device,
        center_loader=train_loader,
        normal_label=normal_label,
        max_center_batches=max_center_batches,
    )

    optimizer = build_optimizer(cfg, model)

    epochs = int(cfg_get(cfg, "train", "epochs", default=100))
    use_amp = bool(cfg_get(cfg, "train", "use_amp", default=True)) and (device.type == "cuda")
    grad_clip = cfg_get(cfg, "train", "grad_clip", default=None)

    lambda_anomaly = float(cfg_get(cfg, "train", "lambda_anomaly", default=0.1))
    eta = float(cfg_get(cfg, "train", "eta", default=1.0))

    monitor = str(cfg_get(cfg, "train", "monitor", default="loss")).lower()
    if monitor not in {"loss", "f1", "acc"}:
        raise ValueError(f"Unsupported train.monitor: {monitor}")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_metric = math.inf if monitor == "loss" else -math.inf
    best_epoch = -1
    history_loss = []
    history_f1 = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            center_c=center_c,
            scaler=scaler,
            use_amp=use_amp,
            lambda_anomaly=lambda_anomaly,
            eta=eta,
            grad_clip=grad_clip,
            normal_label=normal_label,
            anomaly_label=anomaly_label,
            unlabeled_label=unlabeled_label,
            treat_unlabeled_as_normal=treat_unlabeled_as_normal,
        )

        if val_loader is not None:
            val_metrics = evaluate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                center_c=center_c,
                lambda_anomaly=lambda_anomaly,
                eta=eta,
                normal_label=normal_label,
                anomaly_label=anomaly_label,
                unlabeled_label=unlabeled_label,
                treat_unlabeled_as_normal=treat_unlabeled_as_normal,
            )
        else:
            val_metrics = None

        ref = val_metrics if val_metrics is not None else train_metrics
        monitor_value = ref[monitor]

        history_loss.append(ref["loss"])
        history_f1.append(ref["f1"])

        save_train_history_csv(history_loss, save_dir / "finetune_history_loss.csv")
        save_loss_curve(
            history_loss,
            save_dir / "finetune_loss_curve.png",
            title="Fine-tuning Loss",
        )
        save_train_history_csv(history_f1, save_dir / "finetune_history_f1.csv")
        save_loss_curve(
            history_f1,
            save_dir / "finetune_f1_curve.png",
            title="Fine-tuning F1",
        )

        save_checkpoint(
            path=save_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            center_c=center_c,
            cfg=cfg,
        )

        is_better = (monitor_value < best_metric) if monitor == "loss" else (monitor_value > best_metric)
        if is_better:
            best_metric = monitor_value
            best_epoch = epoch
            save_checkpoint(
                path=save_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                center_c=center_c,
                cfg=cfg,
            )

        elapsed = time.time() - epoch_start
        msg = (
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"(fcl={train_metrics['loss_fcl']:.6f}, anom={train_metrics['loss_anomaly']:.6f}) | "
            f"train_acc={train_metrics['acc']:.4f} train_f1={train_metrics['f1']:.4f} | "
        )
        if val_metrics is not None:
            msg += (
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} | "
            )
        msg += f"best_{monitor}={best_metric:.6f} (epoch {best_epoch}) | time={elapsed:.1f}s"
        print(msg)

    total_elapsed = time.time() - start_time
    print(f"[DONE] fine-tuning finished in {total_elapsed / 60.0:.2f} min")
    print(f"[DONE] best {monitor} = {best_metric:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()