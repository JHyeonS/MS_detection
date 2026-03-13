#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/dataloader/pretrain_dataloader.py

from __future__ import annotations

from typing import Any, Callable, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.pretrain_dataset import PretrainNPYDataset
from src.dataset.transforms import (
    build_reconstruction_transform,
    build_contrast_transform,
)


def _cfg_get(cfg: Any, *keys: str, default=None):
    """
    Safe nested getter for dict / OmegaConf-like objects.
    """
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


def _resolve_pretrain_csv(cfg: Any) -> str:
    csv_path = _cfg_get(cfg, "data", "pretrain_csv", default=None)
    if csv_path is None:
        csv_path = _cfg_get(cfg, "pretrain", "pretrain_csv", default=None)
    if csv_path is None:
        raise ValueError("pretrain_csv not found in cfg.data.pretrain_csv or cfg.pretrain.pretrain_csv")
    return csv_path


def _resolve_allowed_labels(cfg: Any):
    allowed_labels = _cfg_get(cfg, "data", "allowed_labels", default=None)
    if allowed_labels is None:
        allowed_labels = _cfg_get(cfg, "pretrain", "allowed_labels", default=None)
    return allowed_labels


def _resolve_batch_size(cfg: Any) -> int:
    batch_size = _cfg_get(cfg, "pretrain", "batch_size", default=None)
    if batch_size is None:
        batch_size = _cfg_get(cfg, "train", "batch_size", default=32)
    return int(batch_size)


def _resolve_num_workers(cfg: Any) -> int:
    return int(_cfg_get(cfg, "data", "num_workers", default=4))


def _resolve_pin_memory(cfg: Any) -> bool:
    return bool(_cfg_get(cfg, "data", "pin_memory", default=True))


def _resolve_add_channel_dim(cfg: Any, default: bool = True) -> bool:
    return bool(_cfg_get(cfg, "data", "add_channel_dim", default=default))


def _resolve_normalize(cfg: Any, default: Optional[str] = "robust") -> Optional[str]:
    return _cfg_get(cfg, "data", "normalize", default=default)


def build_reconst_pretrain_dataloader(
    cfg: Any,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Reconstruction pretraining dataloader.

    Output batch:
        x: torch.Tensor, shape (B, 1, C, T) if add_channel_dim=True
    """
    if transform is None:
        transform = build_reconstruction_transform(cfg)

    dataset = PretrainNPYDataset(
        csv_path=_resolve_pretrain_csv(cfg),
        add_channel_dim=_resolve_add_channel_dim(cfg, default=True),
        normalize=_resolve_normalize(cfg, default="robust"),
        transform=transform,
        return_meta=False,
        allowed_labels=_resolve_allowed_labels(cfg),
    )

    loader = DataLoader(
        dataset,
        batch_size=_resolve_batch_size(cfg),
        shuffle=True,
        num_workers=_resolve_num_workers(cfg),
        pin_memory=_resolve_pin_memory(cfg),
        drop_last=True,
    )
    return loader


def build_contrast_collate_fn(
    cfg: Any,
    base_transform: Optional[Callable] = None,
):
    """
    Returns a collate_fn that converts a batch of samples into two augmented views.

    Expected dataset sample shape:
        - (C, T)        if add_channel_dim=False
        - (1, C, T)     if add_channel_dim=True

    Returned batch:
        x1: (B, 1, C, T)
        x2: (B, 1, C, T)
    """
    if base_transform is None:
        base_transform = build_contrast_transform(cfg)

    def collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x1_list = []
        x2_list = []

        for x in batch:
            if isinstance(x, np.ndarray):
                arr = x.astype(np.float32)
            elif torch.is_tensor(x):
                arr = x.detach().cpu().numpy().astype(np.float32)
            else:
                raise TypeError(f"Unsupported sample type in contrast collate_fn: {type(x)}")

            # dataset output can be (C,T) or (1,C,T)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim != 2:
                raise ValueError(f"Expected sample shape (C,T) or (1,C,T), got {arr.shape}")

            v1 = base_transform(arr.copy())   # (C, T)
            v2 = base_transform(arr.copy())   # (C, T)

            v1 = torch.from_numpy(v1).float().unsqueeze(0)  # (1, C, T)
            v2 = torch.from_numpy(v2).float().unsqueeze(0)  # (1, C, T)

            x1_list.append(v1)
            x2_list.append(v2)

        x1 = torch.stack(x1_list, dim=0)  # (B, 1, C, T)
        x2 = torch.stack(x2_list, dim=0)  # (B, 1, C, T)
        return x1, x2

    return collate_fn


def build_contrast_pretrain_dataloader(
    cfg: Any,
    base_transform: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    Contrastive pretraining dataloader.

    Output batch:
        x1, x2
        each shape: (B, 1, C, T)
    """
    if collate_fn is None:
        collate_fn = build_contrast_collate_fn(cfg, base_transform=base_transform)

    dataset = PretrainNPYDataset(
        csv_path=_resolve_pretrain_csv(cfg),
        # contrastive는 collate_fn에서 (1,C,T)로 맞추므로 여기선 2D로 받는 게 깔끔함
        add_channel_dim=False,
        normalize=_resolve_normalize(cfg, default="robust"),
        transform=None,
        return_meta=False,
        allowed_labels=_resolve_allowed_labels(cfg),
    )

    loader = DataLoader(
        dataset,
        batch_size=_resolve_batch_size(cfg),
        shuffle=True,
        num_workers=_resolve_num_workers(cfg),
        pin_memory=_resolve_pin_memory(cfg),
        drop_last=True,
        collate_fn=collate_fn,
    )
    return loader