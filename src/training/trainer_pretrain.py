#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/training/trainer_pretrain.py

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.dataloader.pretrain_dataloader import (
    build_reconst_pretrain_dataloader,
    build_contrast_pretrain_dataloader,
)
from src.models.pretrain_reconstruction import CAE

from src.utils.visualize import save_loss_curve, save_train_history_csv

# -----------------------------------------------------------------------------
# config utils
# -----------------------------------------------------------------------------

class AttrDict(dict):
    """
    dict -> attribute access
    """
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


def get_device(cfg) -> torch.device:
    device_str = cfg_get(cfg, "device", default=None)
    if device_str is not None:
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# loss functions
# -----------------------------------------------------------------------------

def reconstruction_loss_fn(
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_name: str = "mse",
) -> torch.Tensor:
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return F.mse_loss(recon, target)
    if loss_name == "l1":
        return F.l1_loss(recon, target)
    raise ValueError(f"Unsupported reconstruction loss: {loss_name}")


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    SimCLR-style NT-Xent loss.
    Input:
        z1, z2: (B, D)
    """
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got z1={z1.shape}, z2={z2.shape}")

    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    mask = torch.eye(2 * batch_size, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    targets = torch.arange(batch_size, device=sim.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)

    loss = F.cross_entropy(sim, targets)
    return loss


# -----------------------------------------------------------------------------
# model output parsing
# -----------------------------------------------------------------------------

def parse_reconstruction_output(output: Any, x: torch.Tensor) -> torch.Tensor:
    """
    Flexible parser for reconstruction mode.

    Supported:
    - dict with key 'recon'
    - tensor shaped like x
    - tuple/list containing a tensor shaped like x
    """
    if isinstance(output, dict):
        if "recon" in output:
            return output["recon"]
        raise ValueError("Reconstruction output dict must contain key 'recon'.")

    if torch.is_tensor(output):
        if output.shape == x.shape:
            return output
        raise ValueError(f"Tensor output shape {output.shape} does not match input {x.shape}")

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.shape == x.shape:
                return item
        raise ValueError("Could not find reconstruction tensor matching input shape in tuple/list output.")

    raise TypeError(f"Unsupported reconstruction output type: {type(output)}")


def parse_contrast_output(output: Any) -> torch.Tensor:
    """
    Flexible parser for contrastive mode.

    Preferred outputs:
    - dict with key 'proj' or 'z'
    - tensor of shape (B, D)
    - tuple/list containing a 2D tensor
    """
    if isinstance(output, dict):
        if "proj" in output:
            return output["proj"]
        if "z" in output:
            return output["z"]
        raise ValueError("Contrast output dict must contain 'proj' or 'z'.")

    if torch.is_tensor(output):
        if output.ndim == 2:
            return output
        raise ValueError(f"Contrast tensor output must be 2D, got {output.shape}")

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
        raise ValueError("Could not find 2D projection tensor in tuple/list output.")

    raise TypeError(f"Unsupported contrast output type: {type(output)}")


# -----------------------------------------------------------------------------
# checkpoint utils
# -----------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    cfg: Any,
):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "mode": cfg.pretrain.mode,
    }
    torch.save(ckpt, path)


def save_encoder_only(path: str | Path, model: nn.Module):
    if hasattr(model, "encoder"):
        torch.save(model.encoder.state_dict(), path)


# -----------------------------------------------------------------------------
# train steps
# -----------------------------------------------------------------------------

def train_one_epoch_reconstruction(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
    recon_loss_name: str = "mse",
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for x in loader:
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(x)
                recon = parse_reconstruction_output(output, x)
                loss = reconstruction_loss_fn(recon, x, loss_name=recon_loss_name)

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

        else:
            output = model(x)
            recon = parse_reconstruction_output(output, x)
            loss = reconstruction_loss_fn(recon, x, loss_name=recon_loss_name)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    return {"loss": avg_loss}


def train_one_epoch_contrast(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
    temperature: float = 0.1,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for x1, x2 in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=use_amp):
                out1 = model(x1)
                out2 = model(x2)
                z1 = parse_contrast_output(out1)
                z2 = parse_contrast_output(out2)
                loss = nt_xent_loss(z1, z2, temperature=temperature)

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

        else:
            out1 = model(x1)
            out2 = model(x2)
            z1 = parse_contrast_output(out1)
            z2 = parse_contrast_output(out2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    return {"loss": avg_loss}


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def build_optimizer(cfg, model: nn.Module):
    lr = float(cfg_get(cfg, "pretrain", "lr", default=1e-3))
    weight_decay = float(cfg_get(cfg, "pretrain", "weight_decay", default=1e-5))
    optimizer_name = str(cfg_get(cfg, "pretrain", "optimizer", default="adamw")).lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cfg", type=str, required=True)
    parser.add_argument("--stage_cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.base_cfg, args.stage_cfg)

    seed = int(cfg_get(cfg, "seed", default=42))
    set_seed(seed)

    device = get_device(cfg)
    print(f"[INFO] device: {device}")

    mode = str(cfg.pretrain.mode).lower()
    if mode not in ["reconstruction", "contrast"]:
        raise ValueError(f"Unsupported pretrain.mode: {cfg.pretrain.mode}")

    save_dir = Path(cfg_get(cfg, "pretrain", "save_dir", default="./runs/pretrain"))
    ensure_dir(save_dir)

    print(f"[INFO] pretrain mode: {mode}")
    print(f"[INFO] save_dir: {save_dir}")

    if mode == "reconstruction":
        train_loader = build_reconst_pretrain_dataloader(cfg)
    else:
        train_loader = build_contrast_pretrain_dataloader(cfg)

    model = CAE(cfg).to(device)
    print(f"[INFO] model params: {count_parameters(model):,}")

    optimizer = build_optimizer(cfg, model)

    epochs = int(cfg_get(cfg, "pretrain", "epochs", default=100))
    use_amp = bool(cfg_get(cfg, "pretrain", "use_amp", default=True)) and (device.type == "cuda")
    grad_clip = cfg_get(cfg, "pretrain", "grad_clip", default=None)
    recon_loss_name = str(cfg_get(cfg, "pretrain", "recon_loss", default="mse"))
    temperature = float(cfg_get(cfg, "pretrain", "temperature", default=0.1))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss = math.inf
    best_epoch = -1
    start_time = time.time()
    train_loss_history =[]

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        if mode == "reconstruction":
            metrics = train_one_epoch_reconstruction(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                recon_loss_name=recon_loss_name,
                grad_clip=grad_clip,
            )
        else:
            metrics = train_one_epoch_contrast(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                temperature=temperature,
                grad_clip=grad_clip,
            )

        train_loss = metrics["loss"]
        train_loss_history.append(train_loss)

        save_train_history_csv(
            losses=train_loss_history,
            save_path=save_dir / "train_history.csv",
        )

        save_loss_curve(
            losses=train_loss_history,
            save_path=save_dir / "train_loss_curve.png",
            title=f"Pretrain Loss ({mode})",
        )

        # save last
        save_checkpoint(
            path=save_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_loss=best_loss,
            cfg=cfg,
        )
        save_encoder_only(save_dir / "last_encoder.pt", model)

        # save best
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch

            save_checkpoint(
                path=save_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_loss=best_loss,
                cfg=cfg,
            )
            save_encoder_only(save_dir / "best_encoder.pt", model)

        elapsed = time.time() - epoch_start
        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"loss={train_loss:.6f} | "
            f"best={best_loss:.6f} (epoch {best_epoch}) | "
            f"time={elapsed:.1f}s"
        )

    total_elapsed = time.time() - start_time
    print(f"[DONE] training finished in {total_elapsed / 60.0:.2f} min")
    print(f"[DONE] best loss = {best_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()