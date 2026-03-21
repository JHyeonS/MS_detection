#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt


def bandpass_filter(x, fs, fmin, fmax, order=4):
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x, axis=1).astype(np.float32)


def remove_mean(x):
    return (x - x.mean(axis=1, keepdims=True)).astype(np.float32)


def robust_norm(x, eps=1e-8):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return ((x - med) / (1.4826 * mad + eps)).astype(np.float32)


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
    ):
        self.df = pd.read_csv(csv_path).copy()
        self.normalize = normalize
        self.transform = transform
        self.add_channel_dim = bool(add_channel_dim)
        self.return_meta = bool(return_meta)
        self.label_map = label_map or {}

        self.preprocess = preprocess or {}
        self.use_detrend = self.preprocess.get("detrend", False)
        self.use_bandpass = self.preprocess.get("bandpass", False)
        self.bandpass_low = self.preprocess.get("bandpass_low", 5)
        self.bandpass_high = self.preprocess.get("bandpass_high", 80)
        self.fs = self.preprocess.get("sampling_rate", 1000)

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
        if self.use_detrend:
            x = remove_mean(x)
        if self.use_bandpass:
            x = bandpass_filter(
                x,
                fs=self.fs,
                fmin=self.bandpass_low,
                fmax=self.bandpass_high,
            )
        x = self._normalize(x)
        return x

    def _to_tensor(self, x):
        t = torch.from_numpy(x.astype(np.float32))
        if self.add_channel_dim:
            t = t.unsqueeze(0)
        return t

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = row["npy_path"]
        label = int(row["label"])
        label = int(self.label_map.get(label, label))

        x = np.load(npy_path)
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
