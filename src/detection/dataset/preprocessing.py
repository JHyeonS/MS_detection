#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(x, fs, fmin, fmax, order=4):
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    if not 0.0 < low < high < 1.0:
        raise ValueError(
            f"Invalid bandpass range: fmin={fmin}, fmax={fmax}, fs={fs}. "
            "Expected 0 < fmin < fmax < fs/2."
        )
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x, axis=1).astype(np.float32)


def remove_mean(x):
    return (x - x.mean(axis=1, keepdims=True)).astype(np.float32)


def robust_norm(x, eps=1e-8):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return ((x - med) / (1.4826 * mad + eps)).astype(np.float32)


def _moving_average_axis1(x, window_samples):
    window_samples = max(1, int(window_samples))
    if window_samples <= 1:
        return x.astype(np.float32)

    if x.shape[1] <= 1:
        return x.astype(np.float32)

    pad_left = window_samples // 2
    pad_right = window_samples - 1 - pad_left
    padded = np.pad(x, ((0, 0), (pad_left, pad_right)), mode="reflect")
    csum = np.cumsum(padded, axis=1, dtype=np.float64)
    csum = np.concatenate([np.zeros((x.shape[0], 1), dtype=np.float64), csum], axis=1)
    out = (csum[:, window_samples:] - csum[:, :-window_samples]) / float(window_samples)
    return out.astype(np.float32)


def agc_filter(x, fs, window_sec=0.2, window_samples=None, target_rms=1.0, eps=1e-6, clip=None):
    """Apply per-trace automatic gain control using a centered moving RMS."""
    if window_samples is None:
        window_samples = max(1, int(round(float(window_sec) * float(fs))))
    power = _moving_average_axis1(np.square(x.astype(np.float32)), window_samples)
    gain = float(target_rms) / (np.sqrt(power) + float(eps))
    y = x.astype(np.float32) * gain.astype(np.float32)
    if clip is not None and float(clip) > 0:
        y = np.clip(y, -float(clip), float(clip))
    return y.astype(np.float32)
