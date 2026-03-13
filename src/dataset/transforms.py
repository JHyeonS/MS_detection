#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/dataset/transforms.py

from __future__ import annotations

from typing import List, Sequence, Optional
import random

import numpy as np


class Compose:
    """
    Sequentially apply transforms to a numpy array.

    Input / output:
        x: np.ndarray of shape (C, T), dtype float32 preferred
    """

    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        names = [t.__class__.__name__ for t in self.transforms]
        return f"Compose({names})"


class Identity:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


class WithProb:
    """
    Apply a transform with probability p.
    """

    def __init__(self, transform, p: float = 0.5):
        self.transform = transform
        self.p = float(p)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return self.transform(x)
        return x

    def __repr__(self) -> str:
        return f"WithProb(transform={self.transform}, p={self.p})"


class RandomAmplitudeScale:
    """
    Multiply the whole patch by a random scalar.
    """

    def __init__(self, min_scale: float = 0.9, max_scale: float = 1.1):
        assert min_scale > 0 and max_scale > 0
        assert max_scale >= min_scale
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return (x * scale).astype(np.float32)


class RandomChannelGain:
    """
    Apply per-channel random gain.
    gain shape: (C, 1)
    """

    def __init__(self, min_scale: float = 0.95, max_scale: float = 1.05):
        assert min_scale > 0 and max_scale > 0
        assert max_scale >= min_scale
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        c, _ = x.shape
        gain = np.random.uniform(self.min_scale, self.max_scale, size=(c, 1)).astype(np.float32)
        return (x * gain).astype(np.float32)


class AddGaussianNoise:
    """
    Add i.i.d. Gaussian noise to the patch.
    """

    def __init__(self, std: float = 0.01):
        assert std >= 0
        self.std = float(std)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.std == 0:
            return x.astype(np.float32)
        noise = np.random.normal(0.0, self.std, size=x.shape).astype(np.float32)
        return (x + noise).astype(np.float32)


class AddGaussianNoiseRelative:
    """
    Add Gaussian noise scaled by current sample std.

    noise_std = sample_std * relative_std
    """

    def __init__(self, relative_std: float = 0.05, eps: float = 1e-8):
        assert relative_std >= 0
        self.relative_std = float(relative_std)
        self.eps = float(eps)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.relative_std == 0:
            return x.astype(np.float32)
        sample_std = float(np.std(x))
        noise_std = max(sample_std * self.relative_std, self.eps)
        noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        return (x + noise).astype(np.float32)


class RandomTimeShift:
    """
    Circular shift along time axis.
    axis=1 (T)

    Note:
        np.roll is circular. For DAS detection this is often acceptable for augmentation.
    """

    def __init__(self, max_shift: int = 30):
        assert max_shift >= 0
        self.max_shift = int(max_shift)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.max_shift == 0:
            return x.astype(np.float32)
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(x, shift=shift, axis=1).astype(np.float32)


class RandomChannelShift:
    """
    Circular shift along channel axis.
    axis=0 (C)
    """

    def __init__(self, max_shift: int = 5):
        assert max_shift >= 0
        self.max_shift = int(max_shift)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.max_shift == 0:
            return x.astype(np.float32)
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(x, shift=shift, axis=0).astype(np.float32)


class RandomTimeMask:
    """
    Zero out a contiguous time region.

    max_width: maximum mask width in time samples
    fill_value: default 0.0
    """

    def __init__(self, max_width: int = 50, fill_value: float = 0.0):
        assert max_width >= 0
        self.max_width = int(max_width)
        self.fill_value = float(fill_value)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.max_width <= 0:
            return x.astype(np.float32)

        _, t = x.shape
        width = np.random.randint(1, min(self.max_width, t) + 1)
        start = np.random.randint(0, t - width + 1)

        out = x.copy()
        out[:, start:start + width] = self.fill_value
        return out.astype(np.float32)


class RandomChannelMask:
    """
    Zero out a contiguous channel region.

    max_width: maximum mask width in channel dimension
    """

    def __init__(self, max_width: int = 16, fill_value: float = 0.0):
        assert max_width >= 0
        self.max_width = int(max_width)
        self.fill_value = float(fill_value)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.max_width <= 0:
            return x.astype(np.float32)

        c, _ = x.shape
        width = np.random.randint(1, min(self.max_width, c) + 1)
        start = np.random.randint(0, c - width + 1)

        out = x.copy()
        out[start:start + width, :] = self.fill_value
        return out.astype(np.float32)


class RandomPolarityFlip:
    """
    Multiply by -1 with probability 0.5.
    """

    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return (-x).astype(np.float32)
        return x.astype(np.float32)


class RandomHorizontalFlipTime:
    """
    Reverse time axis with probability p.
    Usually NOT recommended for strict physical realism,
    but can be tested experimentally.
    """

    def __init__(self, p: float = 0.0):
        self.p = float(p)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return np.flip(x, axis=1).copy().astype(np.float32)
        return x.astype(np.float32)


class ClipAmplitude:
    """
    Clip amplitudes to [-clip_value, clip_value].
    Useful if augmentations occasionally create extreme values.
    """

    def __init__(self, clip_value: float = 10.0):
        assert clip_value > 0
        self.clip_value = float(clip_value)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, -self.clip_value, self.clip_value).astype(np.float32)


class EnsureFloat32:
    """
    Ensure dtype float32.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)


# -----------------------------------------------------------------------------
# Preset builders
# -----------------------------------------------------------------------------

def _cfg_get(cfg, *keys, default=None):
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


def build_reconstruction_transform(cfg) -> Compose:
    """
    Mild augmentation for reconstruction pretraining.
    Reads augmentation strength from cfg.pretrain.aug.*
    """
    amp_min = _cfg_get(cfg, "pretrain", "aug", "amp_min", default=0.9)
    amp_max = _cfg_get(cfg, "pretrain", "aug", "amp_max", default=1.1)
    noise_std = _cfg_get(cfg, "pretrain", "aug", "noise_std", default=0.01)
    time_shift = _cfg_get(cfg, "pretrain", "aug", "time_shift", default=30)
    clip_value = _cfg_get(cfg, "pretrain", "aug", "clip_value", default=10.0)

    return Compose([
        RandomAmplitudeScale(amp_min, amp_max),
        AddGaussianNoise(std=noise_std),
        RandomTimeShift(max_shift=time_shift),
        ClipAmplitude(clip_value),
        EnsureFloat32(),
    ])


def build_contrast_transform(cfg) -> Compose:
    """
    Moderately strong augmentation for contrastive pretraining.
    Reads augmentation strength from cfg.pretrain.aug.*
    """
    amp_min = _cfg_get(cfg, "pretrain", "aug", "amp_min", default=0.8)
    amp_max = _cfg_get(cfg, "pretrain", "aug", "amp_max", default=1.2)
    noise_std = _cfg_get(cfg, "pretrain", "aug", "noise_std", default=0.015)
    time_shift = _cfg_get(cfg, "pretrain", "aug", "time_shift", default=50)

    channel_gain_min = _cfg_get(cfg, "pretrain", "aug", "channel_gain_min", default=0.95)
    channel_gain_max = _cfg_get(cfg, "pretrain", "aug", "channel_gain_max", default=1.05)
    channel_gain_p = _cfg_get(cfg, "pretrain", "aug", "channel_gain_p", default=0.5)

    channel_mask_width = _cfg_get(cfg, "pretrain", "aug", "channel_mask_width", default=16)
    channel_mask_p = _cfg_get(cfg, "pretrain", "aug", "channel_mask_p", default=0.5)

    time_mask_width = _cfg_get(cfg, "pretrain", "aug", "time_mask_width", default=60)
    time_mask_p = _cfg_get(cfg, "pretrain", "aug", "time_mask_p", default=0.5)

    polarity_flip_p = _cfg_get(cfg, "pretrain", "aug", "polarity_flip_p", default=0.1)

    channel_shift = _cfg_get(cfg, "pretrain", "aug", "channel_shift", default=0)
    channel_shift_p = _cfg_get(cfg, "pretrain", "aug", "channel_shift_p", default=0.0)

    clip_value = _cfg_get(cfg, "pretrain", "aug", "clip_value", default=10.0)

    ops = [
        RandomAmplitudeScale(amp_min, amp_max),
        WithProb(RandomChannelGain(channel_gain_min, channel_gain_max), p=channel_gain_p),
        AddGaussianNoise(std=noise_std),
        RandomTimeShift(max_shift=time_shift),
    ]

    if channel_shift > 0 and channel_shift_p > 0:
        ops.append(WithProb(RandomChannelShift(max_shift=channel_shift), p=channel_shift_p))

    ops.extend([
        WithProb(RandomChannelMask(max_width=channel_mask_width, fill_value=0.0), p=channel_mask_p),
        WithProb(RandomTimeMask(max_width=time_mask_width, fill_value=0.0), p=time_mask_p),
        RandomPolarityFlip(p=polarity_flip_p),
        ClipAmplitude(clip_value),
        EnsureFloat32(),
    ])

    return Compose(ops)


def build_finetune_transform(cfg) -> Compose:
    """
    Weak augmentation for supervised fine-tuning.
    Reads augmentation strength from cfg.train.aug.*
    """
    amp_min = _cfg_get(cfg, "train", "aug", "amp_min", default=0.95)
    amp_max = _cfg_get(cfg, "train", "aug", "amp_max", default=1.05)
    noise_std = _cfg_get(cfg, "train", "aug", "noise_std", default=0.005)
    time_shift = _cfg_get(cfg, "train", "aug", "time_shift", default=15)
    clip_value = _cfg_get(cfg, "train", "aug", "clip_value", default=10.0)

    return Compose([
        RandomAmplitudeScale(amp_min, amp_max),
        AddGaussianNoise(std=noise_std),
        RandomTimeShift(max_shift=time_shift),
        ClipAmplitude(clip_value),
        EnsureFloat32(),
    ])


def build_eval_transform(cfg=None) -> Compose:
    """
    No augmentation for validation / test.
    """
    clip_value = 10.0
    if cfg is not None:
        clip_value = _cfg_get(cfg, "train", "aug", "clip_value", default=10.0)

    return Compose([
        ClipAmplitude(clip_value),
        EnsureFloat32(),
    ])


# -----------------------------------------------------------------------------
# Contrastive helper
# -----------------------------------------------------------------------------

class TwoCropsTransform:
    """
    Create two independently augmented views from the same input sample.

    Example:
        base_transform = build_contrast_transform()
        two_crop = TwoCropsTransform(base_transform)

        x1, x2 = two_crop(x)
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x: np.ndarray):
        x1 = self.base_transform(x.copy())
        x2 = self.base_transform(x.copy())
        return x1.astype(np.float32), x2.astype(np.float32)

    def __repr__(self) -> str:
        return f"TwoCropsTransform(base_transform={self.base_transform})"