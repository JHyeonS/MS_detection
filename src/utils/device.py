#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/utils/device.py

from __future__ import annotations

import os
from typing import Any, List

import torch


def _cfg_get(cfg: Any, *keys: str, default=None):
    '''
    Safe nested getter for dict / AttrDict-like objects.
    Supports:
      _cfg_get(cfg, "device", "type", default="cuda")
      _cfg_get(cfg, "device.type", default="cuda")
    '''
    if len(keys) == 1 and isinstance(keys[0], str) and "." in keys[0]:
        keys = tuple(keys[0].split("."))

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


def resolve_visible_gpu_ids(cfg: Any) -> List[int]:
    '''
    Resolve GPU ids from config.

    Supported config styles:
    1) New style:
        device:
          type: cuda
          ids: [1]

    2) Old style:
        device: "cuda"
        or
        device: "cuda:0"

    Returns:
        list[int]
    '''
    device_cfg = _cfg_get(cfg, "device", default=None)

    if isinstance(device_cfg, dict) or hasattr(device_cfg, "type"):
        ids = _cfg_get(cfg, "device", "ids", default=[0])
        if ids is None:
            return [0]
        if isinstance(ids, int):
            return [ids]
        return [int(x) for x in ids]

    if isinstance(device_cfg, str):
        if device_cfg.startswith("cuda:"):
            try:
                return [int(device_cfg.split(":")[1])]
            except Exception:
                return [0]
        if device_cfg == "cuda":
            return [0]

    return [0]


def setup_device_from_cfg(cfg: Any, verbose: bool = True) -> torch.device:
    '''
    Setup CUDA_VISIBLE_DEVICES from config and return torch.device.

    Priority:
      1) device.type / device.ids
      2) legacy string device: "cuda", "cuda:0", "cpu"

    Important:
      This should be called as early as possible in main() before model creation.

    Returns:
        torch.device("cuda") or torch.device("cpu")
    '''
    device_cfg = _cfg_get(cfg, "device", default="cuda")

    # New structured style:
    # device:
    #   type: cuda
    #   ids: [1]
    if isinstance(device_cfg, dict) or hasattr(device_cfg, "type"):
        device_type = _cfg_get(cfg, "device", "type", default="cuda")
        gpu_ids = resolve_visible_gpu_ids(cfg)

        if device_type == "cuda" and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # Legacy string style:
    # device: "cuda" / "cuda:0" / "cpu"
    elif isinstance(device_cfg, str):
        if device_cfg.startswith("cuda"):
            if ":" in device_cfg:
                try:
                    gpu_id = int(device_cfg.split(":")[1])
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                except Exception:
                    pass

            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_cfg)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"[INFO] device: {device}")
        print(f"[INFO] visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")

    return device
