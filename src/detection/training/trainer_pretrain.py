#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/training/trainer_pretrain.py

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

from src.models.pretrain_reconstruction import CAE
from src.models.pretrain_contrastive import ContrastivePretrainModel
from src.utils.device import setup_device_from_cfg
from src.utils.visualize import save_loss_curve, save_train_history_csv
from src.utils.config_io import (
    save_merged_config,
    copy_config_snapshots,
    save_run_metadata,
)


class AttrDict(dict):
    def __getattr__(self, item):
        v = self.get(item)
        if isinstance(v, dict) and not isinstance(v, AttrDict):
            v = AttrDict(v)
            self[item] = v
        return v

    def __setattr__(self, key, value):
        self[key] = value


def _to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    return obj

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


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_pretrain_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode in ["reconstruction", "reconst", "recon", "cae"]:
        return "reconstruction"
    if mode in ["contrast", "contrastive", "simclr"]:
        return "contrast"
    raise ValueError(f"Wrong Pretrain Mode, check yaml: {mode}")


def resolve_pretrain_dataloader(cfg, mode: str):
    from src.dataloader.pretrain_dataloader import build_pretrain_dataloader
    return build_pretrain_dataloader(cfg)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got z1={z1.shape}, z2={z2.shape}")

    # fp16 overflow 방지 위해 loss 계산은 float32로 고정
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # float32

    mask = torch.eye(2 * batch_size, device=sim.device, dtype=torch.bool)

    # fp16/amp에서도 안전한 큰 음수
    sim = sim.masked_fill(mask, -1e4)

    targets = torch.arange(batch_size, device=sim.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)

    return F.cross_entropy(sim, targets)


def parse_reconstruction_batch(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        for key in ["x", "input", "waveform", "data"]:
            if key in batch:
                return batch[key]
    raise TypeError(f"Unsupported reconstruction batch type: {type(batch)}")


def parse_contrast_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        x1 = batch.get("x1", None)
        x2 = batch.get("x2", None)
        if x1 is not None and x2 is not None:
            return x1, x2
    raise TypeError(f"Unsupported contrast batch type: {type(batch)}")


def compute_center_c(z_buffer, eps: float = 1e-6) -> torch.Tensor:
    if len(z_buffer) == 0:
        raise ValueError("z_buffer is empty. Cannot compute hypersphere center c.")
    z_all = torch.cat(z_buffer, dim=0)
    c = z_all.mean(dim=0)
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c > 0)] = eps
    return c


def build_optimizer(cfg, model: nn.Module):
    lr = float(cfg_get(cfg, "pretrain", "lr", default=1e-3))
    weight_decay = float(cfg_get(cfg, "pretrain", "weight_decay", default=1e-5))
    name = str(cfg_get(cfg, "pretrain", "optimizer", default="adamw")).lower()

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg_get(cfg, "pretrain", "momentum", default=0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def save_encoder_only(path: str | Path, model: nn.Module):
    path = Path(path)
    if hasattr(model, "encoder"):
        state = {"encoder_state_dict": model.encoder.state_dict()}
    else:
        state = {"model_state_dict": model.state_dict()}
    torch.save(state, path)


def save_checkpoint(path, model, optimizer, epoch, best_loss, cfg, center_c=None):
    ckpt = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _to_plain_dict(cfg),
    }
    if center_c is not None:
        ckpt["center_c"] = center_c.detach().cpu()
    torch.save(ckpt, path)


def train_one_epoch_reconstruction(model, loader, optimizer, device, scaler, use_amp, grad_clip=None):
    model.train()
    total_loss = 0.0
    total_n = 0
    z_buffer = []

    for batch in loader:
        x = parse_reconstruction_batch(batch).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)

            if isinstance(out, (tuple, list)) and len(out) >= 2:
                # CAE가 (x_hat, z) 또는 (z, x_hat) 둘 다 가능하므로 shape으로 판별
                a, b = out[0], out[1]

                if a.shape == x.shape:
                    x_hat, z = a, b
                elif b.shape == x.shape:
                    z, x_hat = a, b
                else:
                    raise ValueError(
                        f"Neither output matches input shape. "
                        f"out[0]={tuple(a.shape)}, out[1]={tuple(b.shape)}, x={tuple(x.shape)}"
                    )

            elif isinstance(out, dict):
                z = out.get("z", None)
                x_hat = out.get("x_hat", out.get("recon", None))
                if z is None or x_hat is None:
                    raise ValueError(f"Unsupported CAE output keys: {list(out.keys())}")

            else:
                raise ValueError("CAE model output must provide both latent z and reconstruction x_hat.")

            loss = F.mse_loss(x_hat, x)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        z_buffer.append(z.detach().cpu())

    return {"loss": total_loss / max(total_n, 1), "z_buffer": z_buffer}


def train_one_epoch_contrast(model, loader, optimizer, device, scaler, use_amp, temperature=0.1, grad_clip=None):
    model.train()
    total_loss = 0.0
    total_n = 0
    z_buffer = []

    for batch in loader:
        x1, x2 = parse_contrast_batch(batch)
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out1 = model(x1)
            out2 = model(x2)

            if isinstance(out1, (tuple, list)):
                z1 = out1[0]
            elif isinstance(out1, dict):
                z1 = out1.get("z", out1.get("proj", None))
            else:
                z1 = out1

            if isinstance(out2, (tuple, list)):
                z2 = out2[0]
            elif isinstance(out2, dict):
                z2 = out2.get("z", out2.get("proj", None))
            else:
                z2 = out2

            if z1 is None or z2 is None:
                raise ValueError("Contrastive model output must provide embeddings.")
            loss = nt_xent_loss(z1, z2, temperature=temperature)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = x1.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        z_buffer.append(z1.detach().cpu())
        z_buffer.append(z2.detach().cpu())

    return {"loss": total_loss / max(total_n, 1), "z_buffer": z_buffer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cfg", type=str, required=True)
    parser.add_argument("--stage_cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.base_cfg, args.stage_cfg)

    seed = int(cfg_get(cfg, "seed", default=42))
    set_seed(seed)

    run_root = Path(cfg_get(cfg, "paths", "run_root", default="./runs"))
    experiment = str(cfg_get(cfg, "data", "experiment", default="default_exp"))
    mode = normalize_pretrain_mode(cfg_get(cfg, "pretrain", "mode", default="reconstruction"))

    save_dir = run_root / "pretrain" / experiment
    ensure_dir(save_dir)

    save_merged_config(cfg, save_dir)
    copy_config_snapshots(
        base_cfg_path=args.base_cfg,
        stage_cfg_path=args.stage_cfg,
        save_dir=save_dir / "config_snapshot",
    )
    save_run_metadata(
        {
            "task": "finetune",
            "experiment": experiment,
        },
        save_dir,
    )

    device = setup_device_from_cfg(cfg)
    print(f"[INFO] device: {device}")
    print(f"[INFO] pretrain mode: {mode}")
    print(f"[INFO] save_dir: {save_dir}")

    train_loader = resolve_pretrain_dataloader(cfg, mode=mode)
    if mode == "reconstruction":
        model = CAE(cfg).to(device)
    else:
        model = ContrastivePretrainModel(cfg).to(device)

    print(f"[INFO] model params: {count_parameters(model):,}")

    optimizer = build_optimizer(cfg, model)
    epochs = int(cfg_get(cfg, "pretrain", "epochs", default=100))
    use_amp = bool(cfg_get(cfg, "pretrain", "use_amp", default=True)) and (device.type == "cuda")
    grad_clip = cfg_get(cfg, "pretrain", "grad_clip", default=None)
    temperature = float(cfg_get(cfg, "pretrain", "temperature", default=0.1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss = math.inf
    best_epoch = -1
    train_loss_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        if mode == "reconstruction":
            metrics = train_one_epoch_reconstruction(
                model, train_loader, optimizer, device, scaler, use_amp, grad_clip
            )
        else:
            metrics = train_one_epoch_contrast(
                model, train_loader, optimizer, device, scaler, use_amp, temperature, grad_clip
            )

        center_c = compute_center_c(metrics["z_buffer"])
        train_loss = metrics["loss"]
        train_loss_history.append(train_loss)

        save_train_history_csv(train_loss_history, save_dir / "train_history.csv")
        save_loss_curve(train_loss_history, save_dir / "train_loss_curve.png", title=f"Pretrain Loss ({mode})")

        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_loss, cfg, center_c=center_c)
        save_encoder_only(save_dir / "last_encoder.pt", model)

        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_loss, cfg, center_c=center_c)
            save_encoder_only(save_dir / "best_encoder.pt", model)

        elapsed = time.time() - epoch_start
        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"loss={train_loss:.6f} | best={best_loss:.6f} (epoch {best_epoch}) | time={elapsed:.1f}s"
        )

    total_elapsed = time.time() - start_time
    print(f"[DONE] training finished in {total_elapsed / 60.0:.2f} min")
    print(f"[DONE] best loss = {best_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
