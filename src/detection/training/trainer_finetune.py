#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_encoder import cnn_encoder
from src.detection.utils.visualize import save_loss_curve, save_metrics_history_csv
from src.detection.utils.device import setup_device_from_cfg
from src.detection.utils.process_title import set_process_title
from src.detection.utils.config_io import (
    cfg_get,
    copy_config_snapshots,
    ensure_dir,
    load_config,
    save_merged_config,
    save_run_metadata,
)


def _to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    return obj

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_finetune_dataloaders(cfg):
    errors = []
    candidates = [
        ("src.detection.dataloader.finetune_dataloader", "build_finetune_dataloaders"),
        ("src.detection.dataloader.finetune_dataloader", "build_finetune_dataloader"),
        ("src.detection.dataloader.finetune_dataloader", "build_train_val_dataloaders"),
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
    raise ImportError("Could not resolve finetune dataloader builder.\n" + "\n".join(errors))


class FinetuneMSDNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = cnn_encoder(cfg)
        self.latent_dim = int(cfg_get(cfg, "model", "encoder", "latent_dim", default=128))
        self.head = nn.Linear(self.latent_dim, 1)
        if bool(cfg_get(cfg, "train", "freeze_encoder", default=False)):
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        z = self.encoder(x)
        logit = self.head(z)
        return z, logit


def parse_finetune_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("input", batch.get("waveform", batch.get("data"))))
        y = batch.get("y", batch.get("label", batch.get("target", batch.get("labels"))))
        if x is None or y is None:
            raise ValueError(f"Unsupported batch dict keys: {list(batch.keys())}")
        return x, y
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def load_encoder_weights(model, encoder_ckpt_path):
    ckpt = torch.load(encoder_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("encoder_state_dict", None) if isinstance(ckpt, dict) else None
    if state_dict is None and isinstance(ckpt, dict):
        msd = ckpt.get("model_state_dict", None)
        if msd is not None:
            state_dict = {k[len("encoder."):]: v for k, v in msd.items() if k.startswith("encoder.")}
    if state_dict is None:
        raise ValueError("Could not find encoder weights in checkpoint.")
    missing, unexpected = model.encoder.load_state_dict(state_dict, strict=False)
    print(f"[INFO] loaded encoder weights from: {encoder_ckpt_path}")
    print(f"[INFO] encoder load missing keys   : {len(missing)}")
    print(f"[INFO] encoder load unexpected keys: {len(unexpected)}")


def load_fixed_center(center_ckpt_path):
    ckpt = torch.load(center_ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "center_c" not in ckpt:
        raise ValueError(f"center_c not found in checkpoint: {center_ckpt_path}")
    center_c = ckpt["center_c"]
    if not torch.is_tensor(center_c):
        center_c = torch.tensor(center_c)
    return center_c.float()


@torch.no_grad()
def compute_center_from_loader(model, loader, device, center_mode="target_noise"):
    model.eval()
    z_list = []
    for batch in loader:
        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z, _ = model(x)
        if center_mode == "target_all":
            z_list.append(z.detach().cpu())
        elif center_mode == "target_noise":
            mask = (y == 0)
            if mask.sum().item() > 0:
                z_list.append(z[mask].detach().cpu())
        else:
            raise ValueError(f"Unsupported center_mode for recompute: {center_mode}")
    if len(z_list) == 0:
        raise ValueError(f"No samples found to compute center_c for center_mode='{center_mode}'.")
    c = torch.cat(z_list, dim=0).mean(dim=0)
    eps = 1e-6
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c > 0)] = eps
    return c


def resolve_center(cfg, model, train_loader, device, run_root, base_experiment):
    center_mode = str(cfg_get(cfg, "train", "center_mode", default="target_noise")).lower()
    center_info = {"center_mode": center_mode}
    if center_mode == "fixed":
        fixed_center_path = cfg_get(cfg, "train", "fixed_center_checkpoint_path", default=None)
        if fixed_center_path is None:
            fixed_center_path = run_root / "pretrain" / base_experiment / "best.pt"
        center_c = load_fixed_center(fixed_center_path).to(device)
        center_info["center_source"] = "checkpoint"
        center_info["center_checkpoint_path"] = str(fixed_center_path)
        return center_c, center_info
    if center_mode in {"target_all", "target_noise"}:
        center_c = compute_center_from_loader(model, train_loader, device, center_mode=center_mode).to(device)
        center_info["center_source"] = "train_loader_recomputed"
        return center_c, center_info
    raise ValueError(f"Unsupported train.center_mode: {center_mode}")


@torch.no_grad()
def compute_center_diagnostics(model, loader, device, center_c, prefix: str):
    if loader is None:
        return {}
    model.eval()
    noise_dist = []
    event_dist = []
    for batch in loader:
        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)
        z, _ = model(x)
        dist = torch.sum((z - center_c.unsqueeze(0)) ** 2, dim=1)
        noise_mask = y < 0.5
        event_mask = y >= 0.5
        if noise_mask.any():
            noise_dist.append(dist[noise_mask].detach().cpu())
        if event_mask.any():
            event_dist.append(dist[event_mask].detach().cpu())

    out = {}
    if noise_dist:
        values = torch.cat(noise_dist)
        out[f"{prefix}_noise_count"] = int(values.numel())
        out[f"{prefix}_noise_dist_mean"] = float(values.mean().item())
        out[f"{prefix}_noise_dist_std"] = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    else:
        out[f"{prefix}_noise_count"] = 0
        out[f"{prefix}_noise_dist_mean"] = None
        out[f"{prefix}_noise_dist_std"] = None

    if event_dist:
        values = torch.cat(event_dist)
        out[f"{prefix}_event_count"] = int(values.numel())
        out[f"{prefix}_event_dist_mean"] = float(values.mean().item())
        out[f"{prefix}_event_dist_std"] = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    else:
        out[f"{prefix}_event_count"] = 0
        out[f"{prefix}_event_dist_mean"] = None
        out[f"{prefix}_event_dist_std"] = None

    noise_mean = out[f"{prefix}_noise_dist_mean"]
    event_mean = out[f"{prefix}_event_dist_mean"]
    out[f"{prefix}_dist_gap_event_minus_noise"] = (
        float(event_mean - noise_mean) if noise_mean is not None and event_mean is not None else None
    )
    out[f"{prefix}_dist_ratio_event_over_noise"] = (
        float(event_mean / max(noise_mean, 1e-12)) if noise_mean is not None and event_mean is not None else None
    )
    return out


def _compute_quantile_wasserstein_1d(x: np.ndarray, y: np.ndarray, num_quantiles: int = 128) -> float | None:
    if x.size == 0 or y.size == 0:
        return None
    q = np.linspace(0.0, 1.0, int(num_quantiles), dtype=np.float64)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))


def _compute_sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    num_projections: int = 32,
    num_quantiles: int = 128,
    seed: int = 42,
) -> float | None:
    if x.size == 0 or y.size == 0:
        return None
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Sliced Wasserstein expects 2D arrays: (n_samples, latent_dim).")

    latent_dim = x.shape[1]
    rng = np.random.default_rng(int(seed))
    projections = rng.normal(size=(int(num_projections), latent_dim)).astype(np.float64)
    projections /= np.clip(np.linalg.norm(projections, axis=1, keepdims=True), a_min=1e-12, a_max=None)

    values = []
    for proj in projections:
        xp = x @ proj
        yp = y @ proj
        wd = _compute_quantile_wasserstein_1d(xp, yp, num_quantiles=num_quantiles)
        if wd is not None:
            values.append(wd)
    if not values:
        return None
    return float(np.mean(values))


@torch.no_grad()
def compute_wasserstein_diagnostics(model, loader, device, prefix: str, cfg):
    if loader is None:
        return {}
    model.eval()
    noise_z = []
    event_z = []
    for batch in loader:
        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)
        z, _ = model(x)
        z_np = z.detach().cpu().float().numpy()
        y_np = y.detach().cpu().numpy()
        noise_mask = y_np < 0.5
        event_mask = y_np >= 0.5
        if noise_mask.any():
            noise_z.append(z_np[noise_mask])
        if event_mask.any():
            event_z.append(z_np[event_mask])

    out = {}
    num_projections = int(cfg_get(cfg, "train", "wasserstein_num_projections", default=32))
    num_quantiles = int(cfg_get(cfg, "train", "wasserstein_num_quantiles", default=128))
    seed = int(cfg_get(cfg, "train", "seed", default=42))
    out[f"{prefix}_event_noise_swd_num_projections"] = num_projections
    out[f"{prefix}_event_noise_swd_num_quantiles"] = num_quantiles
    if not noise_z or not event_z:
        out[f"{prefix}_event_noise_swd"] = None
        return out

    noise_z = np.concatenate(noise_z, axis=0)
    event_z = np.concatenate(event_z, axis=0)
    out[f"{prefix}_event_noise_swd"] = _compute_sliced_wasserstein(
        event_z,
        noise_z,
        num_projections=num_projections,
        num_quantiles=num_quantiles,
        seed=seed,
    )
    out[f"{prefix}_event_latent_count"] = int(event_z.shape[0])
    out[f"{prefix}_noise_latent_count"] = int(noise_z.shape[0])
    out[f"{prefix}_event_latent_mean_norm"] = float(np.linalg.norm(event_z.mean(axis=0)))
    out[f"{prefix}_noise_latent_mean_norm"] = float(np.linalg.norm(noise_z.mean(axis=0)))
    return out


def build_optimizer(cfg, model):
    lr = float(cfg_get(cfg, "train", "lr", default=1e-4))
    wd = float(cfg_get(cfg, "train", "weight_decay", default=1e-5))
    name = str(cfg_get(cfg, "train", "optimizer", default="adamw")).lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(cfg_get(cfg, "train", "momentum", default=0.9)), weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_loss_weights(cfg):
    return float(cfg_get(cfg, "train", "cls_loss_weight", default=1.0)), float(cfg_get(cfg, "train", "anomaly_loss_weight", default=1.0))


def build_classwise_loss_weights(cfg):
    return {
        "bce_pos": float(cfg_get(cfg, "train", "bce_pos_weight", default=1.0)),
        "bce_neg": float(cfg_get(cfg, "train", "bce_neg_weight", default=1.0)),
        "anomaly_pos": float(cfg_get(cfg, "train", "anomaly_pos_weight", default=1.0)),
        "anomaly_neg": float(cfg_get(cfg, "train", "anomaly_neg_weight", default=1.0)),
    }


def _weighted_mean_by_label(values, y, pos_weight: float, neg_weight: float):
    weights = torch.where(
        y >= 0.5,
        torch.full_like(y, float(pos_weight)),
        torch.full_like(y, float(neg_weight)),
    )
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def compute_weighted_branch_losses(logit, y, z, center_c, classwise_weights):
    cls_terms = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    cls_loss = _weighted_mean_by_label(
        cls_terms,
        y,
        pos_weight=classwise_weights["bce_pos"],
        neg_weight=classwise_weights["bce_neg"],
    )

    dist = torch.sum((z - center_c.unsqueeze(0)) ** 2, dim=1)
    anomaly_terms = torch.where(y < 0.5, dist, 1.0 / (dist + 1e-6))
    anomaly_loss = _weighted_mean_by_label(
        anomaly_terms,
        y,
        pos_weight=classwise_weights["anomaly_pos"],
        neg_weight=classwise_weights["anomaly_neg"],
    )
    return cls_loss, anomaly_loss


def compute_binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_acc": float(balanced_acc),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def save_checkpoint(path, model, optimizer, epoch, best_metric, center_c, cfg):
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    center_c,
    cls_loss_weight,
    anomaly_loss_weight,
    classwise_weights,
    scaler,
    use_amp,
    grad_clip=None,
):
    model.train()
    total_loss = total_cls = total_anom = 0.0
    total_n = 0
    for batch in loader:
        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            z, logit = model(x)
            logit = logit.view(-1)
            cls_loss, anomaly_loss = compute_weighted_branch_losses(
                logit, y, z, center_c, classwise_weights
            )
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
    return {"loss": total_loss / max(total_n, 1), "cls_loss": total_cls / max(total_n, 1), "anomaly_loss": total_anom / max(total_n, 1)}


@torch.no_grad()
def evaluate(model, loader, device, center_c, cls_loss_weight, anomaly_loss_weight, classwise_weights):
    if loader is None:
        return None
    model.eval()
    total_loss = total_cls = total_anom = 0.0
    total_n = 0
    y_true_all = []
    y_pred_all = []
    for batch in loader:
        x, y = parse_finetune_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)
        z, logit = model(x)
        logit = logit.view(-1)
        pred = (torch.sigmoid(logit) >= 0.5).float()
        cls_loss, anomaly_loss = compute_weighted_branch_losses(
            logit, y, z, center_c, classwise_weights
        )
        loss = cls_loss_weight * cls_loss + anomaly_loss_weight * anomaly_loss
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls += cls_loss.item() * bs
        total_anom += anomaly_loss.item() * bs
        total_n += bs
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    if y_true_all:
        metrics = compute_binary_metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))
    else:
        metrics = compute_binary_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    metrics.update(
        {
            "loss": total_loss / max(total_n, 1),
            "cls_loss": total_cls / max(total_n, 1),
            "anomaly_loss": total_anom / max(total_n, 1),
        }
    )
    return metrics


def monitor_value(metrics, monitor_name: str):
    if metrics is None:
        return None
    if monitor_name not in metrics:
        raise ValueError(f"Unsupported monitor '{monitor_name}'. Available keys: {sorted(metrics.keys())}")
    return float(metrics[monitor_name])


def is_improved(current: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


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
    finetune_experiment = base_experiment if not args.exp_suffix else f"{base_experiment}__{args.exp_suffix}"

    save_dir = run_root / "finetune" / finetune_experiment
    ensure_dir(save_dir)

    save_merged_config(cfg, save_dir)
    copy_config_snapshots(base_cfg_path=args.base_cfg, stage_cfg_path=args.stage_cfg, save_dir=save_dir / "config_snapshot")
    save_run_metadata({"task": "finetune", "experiment": finetune_experiment, "base_experiment": base_experiment}, save_dir)

    process_title = set_process_title("finetune")
    print(f"[INFO] process_title: {process_title}")
    device = setup_device_from_cfg(cfg)
    print(f"[INFO] device: {device}")
    print(f"[INFO] save_dir: {save_dir}")

    train_loader, val_loader = resolve_finetune_dataloaders(cfg)

    train_label_eff_info = getattr(train_loader.dataset, "label_efficiency_info", None)
    if train_label_eff_info is not None:
        info_path = save_dir / "label_efficiency_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(train_label_eff_info, f, indent=2, ensure_ascii=False)
        print("[INFO] label efficiency info:")
        print(json.dumps(train_label_eff_info, indent=2, ensure_ascii=False))
        print(f"[INFO] saved label efficiency info: {info_path}")

    print(f"[INFO] train dataset rows: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"[INFO] val dataset rows  : {len(val_loader.dataset)}")

    model = FinetuneMSDNet(cfg).to(device)
    print(f"[INFO] model params: {count_parameters(model):,}")

    use_pretrained_encoder = bool(cfg_get(cfg, "train", "use_pretrained_encoder", default=True))
    encoder_ckpt = cfg_get(cfg, "train", "pretrained_encoder_path", default=None)
    if encoder_ckpt is None:
        auto_ckpt = run_root / "pretrain" / base_experiment / "best_encoder.pt"
        if auto_ckpt.exists():
            encoder_ckpt = str(auto_ckpt)

    print(f"[DEBUG] use_pretrained_encoder = {use_pretrained_encoder}")
    print(f"[DEBUG] pretrained_encoder_path = {encoder_ckpt}")

    if use_pretrained_encoder:
        if encoder_ckpt:
            load_encoder_weights(model, encoder_ckpt)
        else:
            print("[WARN] use_pretrained_encoder=True but no checkpoint found. Starting from random initialization.")
    else:
        print("[INFO] use_pretrained_encoder=False -> random initialization")

    center_c, center_info = resolve_center(cfg, model, train_loader, device, run_root, base_experiment)
    with open(save_dir / "center_info.json", "w", encoding="utf-8") as f:
        json.dump(center_info, f, indent=2, ensure_ascii=False)
    print(f"[INFO] center_c shape: {tuple(center_c.shape)}")
    print(f"[INFO] center info: {json.dumps(center_info, indent=2, ensure_ascii=False)}")

    optimizer = build_optimizer(cfg, model)
    epochs = int(cfg_get(cfg, "train", "epochs", default=100))
    use_amp = bool(cfg_get(cfg, "train", "use_amp", default=True)) and (device.type == "cuda")
    grad_clip = cfg_get(cfg, "train", "grad_clip", default=None)
    cls_loss_weight, anomaly_loss_weight = build_loss_weights(cfg)
    classwise_weights = build_classwise_loss_weights(cfg)
    print(f"[INFO] branch loss weights: {json.dumps(classwise_weights, sort_keys=True)}")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    monitor_name = str(cfg_get(cfg, "train", "monitor", default="loss")).lower()
    monitor_mode = str(cfg_get(cfg, "train", "monitor_mode", default="min")).lower()
    if monitor_mode not in {"min", "max"}:
        raise ValueError(f"Unsupported train.monitor_mode: {monitor_mode}")

    best_metric = math.inf if monitor_mode == "min" else -math.inf
    best_epoch = -1
    history_rows = []
    center_history_rows = []
    center_update = str(cfg_get(cfg, "train", "center_update", default="once")).lower()
    center_mode = str(cfg_get(cfg, "train", "center_mode", default="target_noise")).lower()
    log_center_diagnostics = bool(cfg_get(cfg, "train", "log_center_diagnostics", default=False))
    log_wasserstein_diagnostics = bool(cfg_get(cfg, "train", "log_wasserstein_diagnostics", default=False))
    center_diagnostics_interval = int(cfg_get(cfg, "train", "center_diagnostics_interval", default=1))
    if center_diagnostics_interval <= 0:
        center_diagnostics_interval = 1
    initial_center_c = center_c.detach().float().cpu().clone()
    previous_epoch_center_c = None
    early_stopping_patience = cfg_get(cfg, "train", "early_stopping_patience", default=None)
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        if early_stopping_patience <= 0:
            early_stopping_patience = None
    early_stopping_min_delta = float(cfg_get(cfg, "train", "early_stopping_min_delta", default=0.0))
    early_stopping_warmup_epochs = int(cfg_get(cfg, "train", "early_stopping_warmup_epochs", default=0))
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        if epoch > 1 and center_mode in {"target_all", "target_noise"} and center_update == "every_epoch":
            center_c = compute_center_from_loader(model, train_loader, device, center_mode=center_mode).to(device)
            print(f"[INFO] recomputed center_c at epoch {epoch}")

        epoch_center_c = center_c.detach().float().cpu().clone()
        should_log_center_diagnostics = (
            log_center_diagnostics
            and (epoch == 1 or epoch % center_diagnostics_interval == 0 or epoch == epochs)
        )
        center_row = None
        if should_log_center_diagnostics:
            center_row = {
                "epoch": epoch,
                "center_mode": center_mode,
                "center_update": center_update,
                "center_norm": float(torch.linalg.vector_norm(epoch_center_c).item()),
                "center_delta_from_initial": float(torch.linalg.vector_norm(epoch_center_c - initial_center_c).item()),
                "center_delta_from_previous_epoch": (
                    0.0
                    if previous_epoch_center_c is None
                    else float(torch.linalg.vector_norm(epoch_center_c - previous_epoch_center_c).item())
                ),
            }

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            center_c,
            cls_loss_weight,
            anomaly_loss_weight,
            classwise_weights,
            scaler,
            use_amp,
            grad_clip,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            center_c,
            cls_loss_weight,
            anomaly_loss_weight,
            classwise_weights,
        )

        if should_log_center_diagnostics and center_row is not None:
            center_row.update(compute_center_diagnostics(model, train_loader, device, center_c, prefix="train"))
            if val_loader is not None:
                center_row.update(compute_center_diagnostics(model, val_loader, device, center_c, prefix="val"))
            if log_wasserstein_diagnostics:
                center_row.update(compute_wasserstein_diagnostics(model, train_loader, device, prefix="train", cfg=cfg))
                if val_loader is not None:
                    center_row.update(compute_wasserstein_diagnostics(model, val_loader, device, prefix="val", cfg=cfg))
            center_history_rows.append(center_row)
            save_metrics_history_csv(rows=center_history_rows, save_path=save_dir / "center_history.csv")
            previous_epoch_center_c = epoch_center_c

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_cls_loss": train_metrics["cls_loss"],
            "train_anomaly_loss": train_metrics["anomaly_loss"],
            "elapsed_sec": round(elapsed := (time.time() - epoch_start), 4),
        }
        if val_metrics is not None:
            row.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_cls_loss": val_metrics["cls_loss"],
                    "val_anomaly_loss": val_metrics["anomaly_loss"],
                    "val_acc": val_metrics["acc"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_specificity": val_metrics["specificity"],
                    "val_balanced_acc": val_metrics["balanced_acc"],
                    "val_f1": val_metrics["f1"],
                    "val_tp": val_metrics["tp"],
                    "val_tn": val_metrics["tn"],
                    "val_fp": val_metrics["fp"],
                    "val_fn": val_metrics["fn"],
                }
            )
        history_rows.append(row)

        save_metrics_history_csv(rows=history_rows, save_path=save_dir / "train_history.csv")
        save_loss_curve(losses=[r["train_loss"] for r in history_rows], save_path=save_dir / "train_loss_curve.png", title="Finetune Train Loss")
        val_losses = [r["val_loss"] for r in history_rows if "val_loss" in r]
        if val_losses:
            save_loss_curve(losses=val_losses, save_path=save_dir / "val_loss_curve.png", title="Finetune Val Loss")

        current_metrics = val_metrics if val_metrics is not None else train_metrics
        current_metric = monitor_value(current_metrics, monitor_name)
        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_metric, center_c, cfg)
        if is_improved(current_metric, best_metric, monitor_mode, min_delta=early_stopping_min_delta):
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_metric, center_c, cfg)
        else:
            epochs_without_improvement += 1

        if val_metrics is not None:
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_acc={val_metrics['acc']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"best_{monitor_name}={best_metric:.4f} (epoch {best_epoch}) | "
                f"time={elapsed:.1f}s"
            )
        else:
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"best_{monitor_name}={best_metric:.4f} (epoch {best_epoch}) | "
                f"time={elapsed:.1f}s"
            )

        if (
            early_stopping_patience is not None
            and epoch >= early_stopping_warmup_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            stop_reason = (
                f"no {monitor_name} improvement for {epochs_without_improvement} epochs "
                f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
            )
            print(f"[EARLY_STOP] epoch={epoch} | best_epoch={best_epoch} | {stop_reason}")
            break

    total_elapsed = time.time() - start_time
    summary = {
        "task": "finetune",
        "experiment": finetune_experiment,
        "base_experiment": base_experiment,
        "save_dir": str(save_dir),
        "monitor": monitor_name,
        "monitor_mode": monitor_mode,
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "epochs": int(epochs),
        "completed_epochs": int(history_rows[-1]["epoch"]) if history_rows else 0,
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_warmup_epochs": int(early_stopping_warmup_epochs),
        "cls_loss_weight": float(cls_loss_weight),
        "anomaly_loss_weight": float(anomaly_loss_weight),
        "classwise_loss_weights": classwise_weights,
        "center_update": center_update,
        "log_center_diagnostics": bool(log_center_diagnostics),
        "log_wasserstein_diagnostics": bool(log_wasserstein_diagnostics),
        "total_elapsed_sec": float(total_elapsed),
        "last_train_metrics": train_metrics,
        "last_val_metrics": val_metrics,
    }
    with open(save_dir / "finetune_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] finetune finished in {total_elapsed / 60.0:.2f} min")
    print(f"[DONE] best metric = {best_metric:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
