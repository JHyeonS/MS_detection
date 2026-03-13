#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _find_npy_column(df: pd.DataFrame) -> str:
    for col in ["npy_path", "path"]:
        if col in df.columns:
            return col
    raise ValueError("CSV must contain either 'npy_path' or 'path' column.")


def _load_npy_2d(npy_path: str | Path) -> np.ndarray:
    x = np.load(npy_path)

    if x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        else:
            raise ValueError(f"Expected (1,C,T) or (C,T), got shape={x.shape} for {npy_path}")
    elif x.ndim != 2:
        raise ValueError(f"Expected 2D or 3D array, got shape={x.shape} for {npy_path}")

    return x.astype(np.float32)


def _zscore_global(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)


def _robust_norm_global(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * mad + eps)


class PretrainNPYDataset(Dataset):
    """
    Dataset for SSL / pretraining.

    Reads:
        pretrain.csv

    Returns:
        x                  if return_meta=False
        (x, meta_dict)     if return_meta=True

    Notes:
        - label is not used for training by default
        - unlabeled(2), noise(0), event(1) can all be included
    """

    def __init__(
        self,
        csv_path: str | Path,
        add_channel_dim: bool = True,
        normalize: Optional[str] = "robust",
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        return_meta: bool = False,
        allowed_labels: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        self.csv_path = Path(csv_path)
        self.add_channel_dim = add_channel_dim
        self.normalize = normalize
        self.transform = transform
        self.return_meta = return_meta

        self.df = pd.read_csv(self.csv_path)
        self.npy_col = _find_npy_column(self.df)

        if allowed_labels is not None:
            if "label" not in self.df.columns:
                raise ValueError("allowed_labels was provided, but CSV has no 'label' column.")
            self.df = self.df[self.df["label"].isin(allowed_labels)].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize is None or self.normalize == "none":
            return x
        if self.normalize == "zscore":
            return _zscore_global(x)
        if self.normalize == "robust":
            return _robust_norm_global(x)
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npy_path = row[self.npy_col]

        x = _load_npy_2d(npy_path)
        x = self._normalize(x)

        if self.transform is not None:
            x = self.transform(x)

        if self.add_channel_dim:
            x = np.expand_dims(x, axis=0)  # (1, C, T)

        x = torch.from_numpy(x).float()

        if not self.return_meta:
            return x

        meta: Dict[str, Any] = {}
        for key in ["site", "label", "label_name", "group_id", "file_stem"]:
            if key in row.index:
                meta[key] = row[key]
        meta["npy_path"] = str(npy_path)

        return x, meta