#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import argparse
from pathlib import Path

import h5py
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.msd_net import MSDNet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, override: dict):
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def dict_to_namespace(d):
    if isinstance(d, dict):
        return type("Config", (), {k: dict_to_namespace(v) for k, v in d.items()})()
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


class H5MSDDataset(Dataset):
    """
    Expected HDF5 format:
        X: (N, C, T)
        y: (N,)
    Returns:
        x: (1, C, T)
        y: scalar float tensor
    """
    def __init__(
        self,
        h5_path,
        x_key="X",
        y_key="y",
        allowed_labels=None,
        normalize=False,
        positive_labels=None,
    ):
        self.h5_path = str(h5_path)
        self.x_key = x_key
        self.y_key = y_key
        self.allowed_labels = allowed_labels
        self.normalize = normalize
        self.positive_labels = positive_labels if positive_labels is not None else [1]

        with h5py.File(self.h5_path, "r") as f:
            if self.x_key not in f:
                raise KeyError(f"x_key='{self.x_key}' not found in {self.h5_path}")
            if self.y_key not in f:
                raise KeyError(f"y_key='{self.y_key}' not found in {self.h5_path}")

            y_all = f[self.y_key][:]
            n = f[self.x_key].shape[0]

            if len(y_all) != n:
                raise ValueError(f"Length mismatch: X={n}, y={len(y_all)}")

            if self.allowed_labels is not None:
                mask = np.isin(y_all, np.array(self.allowed_labels))
                self.indices = np.where(mask)[0].astype(np.int64)
            else:
                self.indices = np.arange(n, dtype=np.int64)

        self._h5 = None

    def __len__(self):
        return len(self.indices)

    def _lazy_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    @staticmethod
    def _normalize(x: np.ndarray, eps=1e-8):
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + eps)

    def __getitem__(self, idx):
        self._lazy_open()
        real_idx = int(self.indices[idx])

        x = self._h5[self.x_key][real_idx]   # (C, T)
        y = self._h5[self.y_key][real_idx]

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected sample shape (C, T), but got {x.shape}")

        if self.normalize:
            x = self._normalize(x)

        # binary target
        y_bin = 1.0 if int(y) in self.positive_labels else 0.0

        x = torch.from_numpy(x).float().unsqueeze(0)   # (1, C, T)
        y_bin = torch.tensor(y_bin, dtype=torch.float32)

        return x, y_bin

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


def build_dataloader(cfg, split="train"):
    if split == "train":
        h5_path = cfg.data.train_h5
        shuffle = True
    elif split == "val":
        h5_path = cfg.data.val_h5
        shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    allowed_labels = None
    if hasattr(cfg.data, "allowed_labels") and cfg.data.allowed_labels is not None:
        allowed_labels = list(cfg.data.allowed_labels)

    positive_labels = [1]
    if hasattr(cfg.train, "positive_labels") and cfg.train.positive_labels is not None:
        positive_labels = list(cfg.train.positive_labels)

    dataset = H5MSDDataset(
        h5_path=h5_path,
        x_key=cfg.data.x_key,
        y_key=cfg.data.y_key,
        allowed_labels=allowed_labels,
        normalize=cfg.data.normalize,
        positive_labels=positive_labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )
    return loader


def build_scheduler(cfg, optimizer):
    if not hasattr(cfg, "scheduler") or not cfg.scheduler.use:
        return None

    sched_type = cfg.scheduler.type.lower()
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler.t_max,
            eta_min=cfg.scheduler.eta_min,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")


def load_pretrained_encoder(model, ckpt_path, device):
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"pretrained encoder not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    # encoder-only state_dict expected
    missing, unexpected = model.cnn_encoder.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded pretrained encoder from: {ckpt_path}")
    print(f"[INFO] Missing keys    : {missing}")
    print(f"[INFO] Unexpected keys : {unexpected}")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)


@torch.no_grad()
def evaluate_metrics_from_logits(logits, targets, threshold=0.5):
    """
    logits: (N,)
    targets: (N,)
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, cfg):
    model.train()
    total_loss = 0.0

    all_logits = []
    all_targets = []

    use_amp = bool(cfg.train.amp) and device.type == "cuda"

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            x_vec, logits = model(x)             # logits: (B,1) expected
            logits = logits.squeeze(-1)          # (B,)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if cfg.train.grad_clip is not None and cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                cfg.train.grad_clip
            )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

        if (step + 1) % cfg.logging.log_interval == 0:
            print(f"[train] step {step+1}/{len(loader)} loss={loss.item():.6f}")

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = evaluate_metrics_from_logits(
        all_logits,
        all_targets,
        threshold=cfg.train.threshold,
    )

    return total_loss / max(len(loader), 1), metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0

    all_logits = []
    all_targets = []

    use_amp = bool(cfg.train.amp) and device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            x_vec, logits = model(x)
            logits = logits.squeeze(-1)
            loss = criterion(logits, y)

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = evaluate_metrics_from_logits(
        all_logits,
        all_targets,
        threshold=cfg.train.threshold,
    )

    return total_loss / max(len(loader), 1), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    override_cfg = load_yaml(args.config)
    merged_cfg = deep_update(base_cfg, override_cfg)
    cfg = dict_to_namespace(merged_cfg)

    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    print(f"[INFO] train samples = {len(train_loader.dataset)}")
    print(f"[INFO] val samples   = {len(val_loader.dataset)}")

    model = MSDNet(cfg).to(device)

    # load pretrained encoder if provided
    if hasattr(cfg.train, "pretrained_encoder_path") and cfg.train.pretrained_encoder_path:
        load_pretrained_encoder(model, cfg.train.pretrained_encoder_path, device)

    # freeze 여부는 MSDNet 내부에서 처리된다고 가정
    params = filter(lambda p: p.requires_grad, model.parameters())

    pos_weight = None
    if hasattr(cfg.train, "pos_weight") and cfg.train.pos_weight is not None:
        pos_weight = torch.tensor([float(cfg.train.pos_weight)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.train.amp and device.type == "cuda"))

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            cfg=cfg,
        )

        val_loss, val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"train_f1={train_metrics['f1']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_loss:.6f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"lr={lr_now:.8f}"
        )

        save_checkpoint(
            path=save_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val=best_val,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

            save_checkpoint(
                path=save_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val=best_val,
            )
            print(f"[INFO] best model updated at epoch {epoch} (val_loss={val_loss:.6f})")

    print(f"[DONE] best_epoch={best_epoch}, best_val={best_val:.6f}")


if __name__ == "__main__":
    main()