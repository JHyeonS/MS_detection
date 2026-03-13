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


class FineTuneNPYDataset(Dataset):
    """
    Dataset for supervised fine-tuning / validation / test.

    Reads:
        train.csv / val.csv / test.csv

    Supports:
        - labeled_fraction for label-efficiency experiments
        - optional class-balanced subsampling
    """

    def __init__(
        self,
        csv_path: str | Path,
        add_channel_dim: bool = True,
        normalize: Optional[str] = "robust",
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        labeled_fraction: float = 1.0,
        seed: int = 42,
        balance_fraction_by_class: bool = True,
        min_samples_per_class: int = 1,
        label_map: Optional[dict[int, int]] = None,
        return_meta: bool = False,
    ) -> None:
        super().__init__()

        self.csv_path = Path(csv_path)
        self.add_channel_dim = add_channel_dim
        self.normalize = normalize
        self.transform = transform
        self.labeled_fraction = labeled_fraction
        self.seed = seed
        self.balance_fraction_by_class = balance_fraction_by_class
        self.min_samples_per_class = min_samples_per_class
        self.label_map = label_map or {0: 0, 1: 1}
        self.return_meta = return_meta

        self.df = pd.read_csv(self.csv_path)
        self.npy_col = _find_npy_column(self.df)

        if "label" not in self.df.columns:
            raise ValueError(f"{self.csv_path} must contain 'label' column for fine-tuning.")

        self.df = self.df[self.df["label"].isin([0, 1])].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No labeled rows (0/1) found in {self.csv_path}")

        self.df = self._apply_label_fraction(self.df)
        self.df = self.df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError("Dataset became empty after applying labeled_fraction.")

    def _apply_label_fraction(self, df: pd.DataFrame) -> pd.DataFrame:
        frac = float(self.labeled_fraction)
        if not (0 < frac <= 1.0):
            raise ValueError(f"labeled_fraction must be in (0,1], got {frac}")

        if frac >= 1.0:
            return df.reset_index(drop=True)

        if self.balance_fraction_by_class:
            parts = []
            for cls in [0, 1]:
                sub = df[df["label"] == cls].copy()
                if len(sub) == 0:
                    raise ValueError(f"Class {cls} has zero samples in {self.csv_path}")

                n_keep = max(self.min_samples_per_class, int(round(len(sub) * frac)))
                n_keep = min(n_keep, len(sub))
                sub = sub.sample(n=n_keep, random_state=self.seed)
                parts.append(sub)

            out = pd.concat(parts, axis=0).reset_index(drop=True)
            return out

        n_keep = max(1, int(round(len(df) * frac)))
        n_keep = min(n_keep, len(df))
        return df.sample(n=n_keep, random_state=self.seed).reset_index(drop=True)

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

    def get_labels(self) -> np.ndarray:
        return self.df["label"].map(self.label_map).to_numpy(dtype=np.int64)

    def class_counts(self) -> Dict[int, int]:
        counts = self.df["label"].value_counts().to_dict()
        return {int(k): int(v) for k, v in counts.items()}

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npy_path = row[self.npy_col]

        x = _load_npy_2d(npy_path)
        x = self._normalize(x)

        if self.transform is not None:
            x = self.transform(x)

        if self.add_channel_dim:
            x = np.expand_dims(x, axis=0)  # (1, C, T)

        y = int(self.label_map[int(row["label"])])

        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)

        if not self.return_meta:
            return x, y

        meta: Dict[str, Any] = {}
        for key in ["site", "label", "label_name", "group_id", "file_stem"]:
            if key in row.index:
                meta[key] = row[key]
        meta["npy_path"] = str(npy_path)

        return x, y, meta