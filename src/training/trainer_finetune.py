#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/training/trainer_finetune.py

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any, Optional, Tuple

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


def _to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    return obj


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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_finetune_dataloaders(cfg):
    errors = []
    candidates = [
        ("src.dataloader.finetune_dataloader", "build_finetune_dataloaders"),
        ("src.dataloader.finetune_dataloader", "build_finetune_dataloader"),
        ("src.dataloader.finetune_dataloader", "build_train_val_dataloaders"),
    ]

    for module_name, fn_name in candidates:
        try:
            module = __import__(module_name, fromlist=[fn_name])
            fn = getattr(module, fn_name)
            out = fn(cfg)

            if isinstance(out, tuple):
                if len(out) >= 2:
                    return out[0], out[1]
                if len(out) == 1:
                    return out[0], None
            return out, None
        except Exception as e:
            errors.append(f"{module_name}.{fn_name}: {repr(e)}")

    msg = "\n".join(errors)
    raise ImportError("Could not resolve finetune dataloader builder.\n" + msg)


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
        z = self.encoder(x)
        logit = self.head(z)
        return z, logit


def parse_finetune_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
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


def load_encoder_weights(model: FinetuneMSDNet, encoder_ckpt_path: str | Path):
    ckpt = torch.load(encoder_ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Encoder checkpoint must be a dict, got {type(ckpt)}")

    state_dict = ckpt.get("encoder_state_dict", None)
    if state_dict is None:
        state_dict = ckpt.get("model_state_dict", None)
        if state_dict is not None:
            state_dict = {
                k[len("encoder.") :]: v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }

    if state_dict is None:
        raise ValueError("Could not find encoder weights in checkpoint.")

    missing, unexpected = model.encoder.load_state_dict(state_dict, strict=False)
    print(f"[INFO] loaded encoder weights from: {encoder_ckpt_path}")
    print(f"[INFO] encoder load missing keys   : {len(missing)}")
    print(f"[INFO] encoder load unexpected keys: {len(unexpected)}")


def build_optimizer(cfg, model: nn.Module):
    lr = float(cfg_get(cfg, "train", "lr", default=1e-4))
    weight_decay = float(cfg_get(cfg, "train", "weight_decay", default=1e-5))
    name = str(cfg_get(cfg, "train", "optimizer", default="adamw")).lower()

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg_get(cfg, "train", "momentum", default=0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {name}")


def build_loss_weights(cfg) -> Tuple[float, float]:
    cls_w = float(cfg_get(cfg, "train", "cls_loss_weight", default=1.0))
    anomaly_w = float(cfg_get(cfg, "train", "anomaly_loss_weight", default=1.0))
    return cls_w, anomaly_w


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
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _to_plain_dict(cfg),
    }
    if center_c is not None:
        ckpt["center_c"] = center_c.detach().cpu()
    torch.save(ckpt, path)


@torch.no_grad()
def compute_center_from_loader(model: FinetuneMSDNet, loader, device: torch.device):
    model.eval()
    z_list = []

    for batch in loader:
        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        normal_mask = (y == 0)
        if normal_mask.sum().item() == 0:
            continue

        z, _ = model(x)
        z_list.append(z[normal_mask].detach().cpu())

    if len(z_list) == 0:
        raise ValueError("No normal samples found to compute center_c.")

    c = torch.cat(z_list, dim=0).mean(dim=0)
    eps = 1e-6
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c > 0)] = eps
    return c


def train_one_epoch(
    model: FinetuneMSDNet,
    loader,
    optimizer,
    device,
    center_c: torch.Tensor,
    cls_loss_weight: float,
    anomaly_loss_weight: float,
    scaler,
    use_amp: bool,
    grad_clip: Optional[float] = None,
):
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_anom = 0.0
    total_n = 0

    for batch in loader:
        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            z, logit = model(x)
            logit = logit.view(-1)

            cls_loss = F.binary_cross_entropy_with_logits(logit, y)

            dist = torch.sum((z - center_c.unsqueeze(0)) ** 2, dim=1)
            anomaly_loss = torch.where(y < 0.5, dist, 1.0 / (dist + 1e-6)).mean()

            loss = cls_loss_weight * cls_loss + anomaly_loss_weight * anomaly_loss

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls += cls_loss.item() * bs
        total_anom += anomaly_loss.item() * bs
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1),
        "cls_loss": total_cls / max(total_n, 1),
        "anomaly_loss": total_anom / max(total_n, 1),
    }


@torch.no_grad()
def evaluate(
    model: FinetuneMSDNet,
    loader,
    device,
    center_c: torch.Tensor,
    cls_loss_weight: float,
    anomaly_loss_weight: float,
):
    if loader is None:
        return None

    model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_anom = 0.0
    total_n = 0
    total_correct = 0

    for batch in loader:
        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        z, logit = model(x)
        logit = logit.view(-1)
        prob = torch.sigmoid(logit)
        pred = (prob >= 0.5).float()

        cls_loss = F.binary_cross_entropy_with_logits(logit, y)
        dist = torch.sum((z - center_c.unsqueeze(0)) ** 2, dim=1)
        anomaly_loss = torch.where(y < 0.5, dist, 1.0 / (dist + 1e-6)).mean()
        loss = cls_loss_weight * cls_loss + anomaly_loss_weight * anomaly_loss

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls += cls_loss.item() * bs
        total_anom += anomaly_loss.item() * bs
        total_correct += (pred == y).sum().item()
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1),
        "cls_loss": total_cls / max(total_n, 1),
        "anomaly_loss": total_anom / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cfg", type=str, required=True)
    parser.add_argument("--stage_cfg", type=str, required=True)
    parser.add_argument("--exp_suffix", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.base_cfg, args.stage_cfg)

    seed = int(cfg_get(cfg, "seed", default=42))
    set_seed(seed)

    run_root = Path(cfg_get(cfg, "paths", "run_root", default="./runs"))

    base_experiment = str(cfg_get(cfg, "data", "experiment", default="default_exp"))
    finetune_experiment = base_experiment
    if args.exp_suffix is not None and str(args.exp_suffix).strip() != "":
        finetune_experiment = f"{base_experiment}__{args.exp_suffix}"

    save_dir = run_root / "finetune" / finetune_experiment
    ensure_dir(save_dir)

    save_merged_config(cfg, save_dir)
    copy_config_snapshots(
        base_cfg_path=args.base_cfg,
        stage_cfg_path=args.stage_cfg,
        save_dir=save_dir / "config_snapshot",
    )
    save_run_metadata(
        {"task": "finetune", "experiment": finetune_experiment, "base_experiment": base_experiment},
        save_dir,
    )

    device = setup_device_from_cfg(cfg)
    print(f"[INFO] device: {device}")
    print(f"[INFO] save_dir: {save_dir}")

    train_loader, val_loader = resolve_finetune_dataloaders(cfg)

    model = FinetuneMSDNet(cfg).to(device)
    print(f"[INFO] model params: {count_parameters(model):,}")

    encoder_ckpt = cfg_get(cfg, "train", "pretrained_encoder_path", default=None)

    if encoder_ckpt is None:
        auto_ckpt = run_root / "pretrain" / base_experiment / "best_encoder.pt"
        if auto_ckpt.exists():
            encoder_ckpt = str(auto_ckpt)

    print(f"[DEBUG] pretrained_encoder_path = {encoder_ckpt}")

    if encoder_ckpt:
        load_encoder_weights(model, encoder_ckpt)
    else:
        print("[WARN] No pretrained encoder checkpoint found. Finetune starts from random initialization.")

    optimizer = build_optimizer(cfg, model)
    epochs = int(cfg_get(cfg, "train", "epochs", default=100))
    use_amp = bool(cfg_get(cfg, "train", "use_amp", default=True)) and (device.type == "cuda")
    grad_clip = cfg_get(cfg, "train", "grad_clip", default=None)
    cls_loss_weight, anomaly_loss_weight = build_loss_weights(cfg)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    center_c = compute_center_from_loader(model, train_loader, device).to(device)
    print(f"[INFO] center_c shape: {tuple(center_c.shape)}")

    best_metric = -math.inf
    best_epoch = -1
    train_loss_history = []
    val_loss_history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            center_c=center_c,
            cls_loss_weight=cls_loss_weight,
            anomaly_loss_weight=anomaly_loss_weight,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            center_c=center_c,
            cls_loss_weight=cls_loss_weight,
            anomaly_loss_weight=anomaly_loss_weight,
        )

        train_loss_history.append(train_metrics["loss"])
        if val_metrics is not None:
            val_loss_history.append(val_metrics["loss"])

        save_train_history_csv(
            losses=train_loss_history,
            save_path=save_dir / "train_history.csv",
        )
        save_loss_curve(
            losses=train_loss_history,
            save_path=save_dir / "train_loss_curve.png",
            title="Finetune Train Loss",
        )
        if len(val_loss_history) > 0:
            save_loss_curve(
                losses=val_loss_history,
                save_path=save_dir / "val_loss_curve.png",
                title="Finetune Val Loss",
            )

        current_metric = val_metrics["acc"] if val_metrics is not None else -train_metrics["loss"]

        save_checkpoint(
            path=save_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            center_c=center_c,
            cfg=cfg,
        )

        if current_metric > best_metric:
            best_metric = current_metric
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
        if val_metrics is not None:
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_acc={val_metrics['acc']:.4f} | "
                f"best_metric={best_metric:.4f} (epoch {best_epoch}) | "
                f"time={elapsed:.1f}s"
            )
        else:
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"best_metric={best_metric:.4f} (epoch {best_epoch}) | "
                f"time={elapsed:.1f}s"
            )

    total_elapsed = time.time() - start_time
    print(f"[DONE] finetune finished in {total_elapsed / 60.0:.2f} min")
    print(f"[DONE] best metric = {best_metric:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
