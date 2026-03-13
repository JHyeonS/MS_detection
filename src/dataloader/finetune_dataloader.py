#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset.finetune_dataset import FineTuneNPYDataset


def _get_attr(cfg, path: str, default=None):
    """
    Safe nested attribute getter.
    Example:
        _get_attr(cfg, "train.batch_size", 16)
    """
    cur = cfg
    for key in path.split("."):
        if not hasattr(cur, key):
            return default
        cur = getattr(cur, key)
    return cur


def _resolve_csv_path(cfg, split: str) -> Path:
    """
    Resolve CSV path for split in the following priority:
      1) cfg.data.<split>_csv
      2) cfg.data.split_dir / f"{split}.csv"
      3) cfg.data.metadata_dir / f"{split}.csv"
    """
    direct_key = f"data.{split}_csv"
    direct_path = _get_attr(cfg, direct_key, None)
    if direct_path is not None:
        p = Path(direct_path)
        if not p.is_absolute():
            split_dir = _get_attr(cfg, "data.split_dir", None)
            if split_dir is not None:
                p = Path(split_dir) / p
        return p

    split_dir = _get_attr(cfg, "data.split_dir", None)
    if split_dir is not None:
        p = Path(split_dir) / f"{split}.csv"
        if p.exists():
            return p

    metadata_dir = _get_attr(cfg, "data.metadata_dir", None)
    if metadata_dir is not None:
        p = Path(metadata_dir) / f"{split}.csv"
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not resolve {split}.csv. "
        f"Set cfg.data.{split}_csv or cfg.data.split_dir / cfg.data.metadata_dir."
    )


def _build_dataset(cfg, split: str) -> FineTuneNPYDataset:
    """
    Build FineTuneNPYDataset for a given split.
    """
    csv_path = _resolve_csv_path(cfg, split)

    normalize = _get_attr(cfg, "data.normalize", "robust")
    add_channel_dim = _get_attr(cfg, "data.add_channel_dim", True)
    return_meta = _get_attr(cfg, "data.return_meta", False)

    label_map = _get_attr(cfg, "train.label_map", None)
    if label_map is None:
        label_map = {0: 0, 1: 1}

    if split == "train":
        labeled_fraction = float(_get_attr(cfg, "train.labeled_fraction", 1.0))
        balance_fraction_by_class = bool(
            _get_attr(cfg, "train.balance_fraction_by_class", True)
        )
        min_samples_per_class = int(
            _get_attr(cfg, "train.min_samples_per_class", 1)
        )
    else:
        labeled_fraction = 1.0
        balance_fraction_by_class = False
        min_samples_per_class = 1

    seed = int(_get_attr(cfg, "train.seed", 42))

    ds = FineTuneNPYDataset(
        csv_path=csv_path,
        add_channel_dim=add_channel_dim,
        normalize=normalize,
        transform=None,
        labeled_fraction=labeled_fraction,
        seed=seed,
        balance_fraction_by_class=balance_fraction_by_class,
        min_samples_per_class=min_samples_per_class,
        label_map=label_map,
        return_meta=return_meta,
    )
    return ds


def _build_weighted_sampler(dataset: FineTuneNPYDataset) -> WeightedRandomSampler:
    """
    Optional weighted sampler for class imbalance.
    Assumes labels are 0/1 after dataset filtering.
    """
    labels = dataset.get_labels()
    class_counts = {}
    for y in labels:
        class_counts[int(y)] = class_counts.get(int(y), 0) + 1

    weights = [1.0 / class_counts[int(y)] for y in labels]
    weights = torch.tensor(weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


def _build_loader(
    dataset: FineTuneNPYDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    weighted_sampling: bool = False,
) -> DataLoader:
    """
    Build a torch DataLoader.
    """
    sampler = None
    if weighted_sampling:
        sampler = _build_weighted_sampler(dataset)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    return loader


def build_finetune_datasets(
    cfg,
    with_test: bool = True,
) -> Tuple[FineTuneNPYDataset, Optional[FineTuneNPYDataset], Optional[FineTuneNPYDataset]]:
    """
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    train_dataset = _build_dataset(cfg, "train")

    val_dataset = None
    try:
        val_dataset = _build_dataset(cfg, "val")
    except FileNotFoundError:
        pass

    test_dataset = None
    if with_test:
        try:
            test_dataset = _build_dataset(cfg, "test")
        except FileNotFoundError:
            pass

    return train_dataset, val_dataset, test_dataset


def build_finetune_dataloaders(cfg):
    """
    Main entry point.

    Expected cfg examples:
    ----------------------
    data:
      train_csv: "./data/metadata/train.csv"
      val_csv: "./data/metadata/val.csv"
      test_csv: "./data/metadata/test.csv"
      normalize: "robust"
      add_channel_dim: true
      return_meta: false

    train:
      batch_size: 16
      num_workers: 4
      pin_memory: true
      drop_last: true
      seed: 42
      labeled_fraction: 1.0
      balance_fraction_by_class: true
      min_samples_per_class: 1
      weighted_sampling: false
      label_map:
        0: 0
        1: 1
    """
    batch_size = int(_get_attr(cfg, "train.batch_size", 16))
    num_workers = int(_get_attr(cfg, "train.num_workers", 4))
    pin_memory = bool(_get_attr(cfg, "train.pin_memory", True))
    drop_last = bool(_get_attr(cfg, "train.drop_last", True))
    weighted_sampling = bool(_get_attr(cfg, "train.weighted_sampling", False))

    eval_batch_size = int(_get_attr(cfg, "train.eval_batch_size", batch_size))

    train_dataset, val_dataset, test_dataset = build_finetune_datasets(cfg, with_test=True)

    train_loader = _build_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(not weighted_sampling),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        weighted_sampling=weighted_sampling,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = _build_loader(
            dataset=val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            weighted_sampling=False,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = _build_loader(
            dataset=test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            weighted_sampling=False,
        )

    print("[INFO] Fine-tune dataset summary")
    print(f"  train: {len(train_dataset)} samples | class_counts={train_dataset.class_counts()}")
    if val_dataset is not None:
        print(f"  val:   {len(val_dataset)} samples | class_counts={val_dataset.class_counts()}")
    else:
        print("  val:   None")
    if test_dataset is not None:
        print(f"  test:  {len(test_dataset)} samples | class_counts={test_dataset.class_counts()}")
    else:
        print("  test:  None")

    return train_loader, val_loader, test_loader