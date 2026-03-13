#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/dataloader/test_dataloader.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from src.dataset.finetune_dataset import FineTuneNPYDataset


def _get_attr(cfg, path: str, default=None):
    cur = cfg
    for key in path.split("."):
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
    return cur


def _resolve_test_csv(cfg) -> Path:
    """
    Priority:
      1) cfg.data.test_csv
      2) cfg.data.split_dir / "test.csv"
      3) cfg.data.metadata_dir / "test.csv"
    """
    direct_path = _get_attr(cfg, "data.test_csv", None)
    if direct_path is not None:
        p = Path(direct_path)
        if not p.is_absolute():
            split_dir = _get_attr(cfg, "data.split_dir", None)
            if split_dir is not None:
                p = Path(split_dir) / p
        return p

    split_dir = _get_attr(cfg, "data.split_dir", None)
    if split_dir is not None:
        p = Path(split_dir) / "test.csv"
        if p.exists():
            return p

    metadata_dir = _get_attr(cfg, "data.metadata_dir", None)
    if metadata_dir is not None:
        p = Path(metadata_dir) / "test.csv"
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not resolve test.csv. "
        "Set cfg.data.test_csv or cfg.data.split_dir / cfg.data.metadata_dir."
    )


def build_test_dataset(cfg) -> FineTuneNPYDataset:
    csv_path = _resolve_test_csv(cfg)

    normalize = _get_attr(cfg, "data.normalize", "robust")
    add_channel_dim = _get_attr(cfg, "data.add_channel_dim", True)
    return_meta = _get_attr(cfg, "data.return_meta", True)

    label_map = _get_attr(cfg, "train.label_map", None)
    if label_map is None:
        label_map = {0: 0, 1: 1}

    ds = FineTuneNPYDataset(
        csv_path=csv_path,
        add_channel_dim=add_channel_dim,
        normalize=normalize,
        transform=None,
        labeled_fraction=1.0,
        seed=int(_get_attr(cfg, "train.seed", 42)),
        balance_fraction_by_class=False,
        min_samples_per_class=1,
        label_map=label_map,
        return_meta=return_meta,
    )
    return ds


def build_test_dataloader(cfg) -> DataLoader:
    dataset = build_test_dataset(cfg)

    batch_size = int(_get_attr(cfg, "test.batch_size", 16))
    num_workers = int(_get_attr(cfg, "test.num_workers", _get_attr(cfg, "data.num_workers", 4)))
    pin_memory = bool(_get_attr(cfg, "test.pin_memory", _get_attr(cfg, "data.pin_memory", True)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    print("[INFO] Test dataset summary")
    print(f"  test_csv: {dataset.csv_path}")
    print(f"  n_test:   {len(dataset)}")
    print(f"  counts:   {dataset.class_counts()}")

    return loader


# alias
def build_test_loader(cfg) -> DataLoader:
    return build_test_dataloader(cfg)