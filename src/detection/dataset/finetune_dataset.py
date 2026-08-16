#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.detection.dataset.preprocessing import agc_filter, bandpass_filter, remove_mean, robust_norm


class FinetuneDataset(Dataset):
    def __init__(
        self,
        csv_path,
        normalize="robust",
        transform=None,
        preprocess=None,
        add_channel_dim=True,
        return_meta=False,
        label_map=None,
        cache_mode="none",
    ):
        self.df = pd.read_csv(csv_path).copy()
        self.normalize = normalize
        self.transform = transform
        self.add_channel_dim = bool(add_channel_dim)
        self.return_meta = bool(return_meta)
        self.label_map = label_map or {}
        self.cache_mode = str(cache_mode or "none").strip().lower()
        if self.cache_mode not in {"none", "ram"}:
            raise ValueError(f"Unsupported cache_mode={cache_mode!r}. Use 'none' or 'ram'.")

        self.preprocess = preprocess or {}
        self.load_only = bool(self.preprocess.get("load_only", False))
        self.use_detrend = self.preprocess.get("detrend", False)
        self.use_bandpass = self.preprocess.get("bandpass", False)
        self.bandpass_low = self.preprocess.get("bandpass_low", 5)
        self.bandpass_high = self.preprocess.get("bandpass_high", 80)
        self.bandpass_order = self.preprocess.get("bandpass_order", 4)
        self.use_agc = self.preprocess.get("agc", False)
        self.agc_window_sec = self.preprocess.get("agc_window_sec", 0.2)
        self.agc_window_samples = self.preprocess.get("agc_window_samples", None)
        self.agc_target_rms = self.preprocess.get("agc_target_rms", 1.0)
        self.agc_eps = self.preprocess.get("agc_eps", 1e-6)
        self.agc_clip = self.preprocess.get("agc_clip", None)
        self.fs = self.preprocess.get("sampling_rate", 1000)
        self._x_cache = self._build_ram_cache() if self.cache_mode == "ram" else None

    def __len__(self):
        return len(self.df)

    def _normalize(self, x):
        if self.normalize == "robust":
            return robust_norm(x)
        if self.normalize == "zscore":
            return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)
        return x.astype(np.float32)

    def _prepare(self, x):
        x = x.astype(np.float32)
        if x.ndim == 3:
            x = x[0]
        if self.load_only:
            return x.astype(np.float32)
        if self.use_detrend:
            x = remove_mean(x)
        if self.use_bandpass:
            x = bandpass_filter(
                x,
                fs=self.fs,
                fmin=self.bandpass_low,
                fmax=self.bandpass_high,
                order=self.bandpass_order,
            )
        if self.use_agc:
            x = agc_filter(
                x,
                fs=self.fs,
                window_sec=self.agc_window_sec,
                window_samples=self.agc_window_samples,
                target_rms=self.agc_target_rms,
                eps=self.agc_eps,
                clip=self.agc_clip,
            )
        x = self._normalize(x)
        return x

    def _to_tensor(self, x):
        t = torch.from_numpy(x.astype(np.float32))
        if self.add_channel_dim:
            t = t.unsqueeze(0)
        return t

    def _build_ram_cache(self):
        if "npy_path" not in self.df.columns:
            raise ValueError("RAM cache requires an 'npy_path' column.")
        cache = []
        for npy_path in self.df["npy_path"].tolist():
            cache.append(np.load(npy_path).astype(np.float32))
        return cache

    def _load_sample(self, idx, npy_path):
        if self._x_cache is not None:
            return self._x_cache[idx]
        return np.load(npy_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = row["npy_path"]
        label = int(row["label"])
        label = int(self.label_map.get(label, label))

        x = self._load_sample(idx, npy_path)
        x = self._prepare(x)
        x = self.transform(x) if self.transform is not None else x
        x = self._to_tensor(x)

        y = torch.tensor(label, dtype=torch.long)

        if self.return_meta:
            meta = {}
            for key in ["npy_path", "source", "split", "label"]:
                if key in row.index:
                    meta[key] = row[key]
            return x, y, meta

        return x, y
