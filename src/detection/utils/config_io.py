#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/utils/config_io.py

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml


class AttrDict(dict):
    def __getattr__(self, item: str):
        if item not in self:
            raise AttributeError(item)
        value = self.get(item)
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_plain(obj: Any):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool, type(None))):
        try:
            return {k: _to_plain(v) for k, v in vars(obj).items()}
        except Exception:
            return str(obj)
    return obj


def to_attrdict(obj: Any):
    if isinstance(obj, dict):
        return AttrDict({k: to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attrdict(v) for v in obj]
    return obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(base_cfg_path: str | Path, stage_cfg_path: str | Path):
    base_cfg = load_yaml(base_cfg_path)
    stage_cfg = load_yaml(stage_cfg_path)
    return to_attrdict(deep_update(base_cfg, stage_cfg))


def cfg_get(cfg: Any, *keys: str, default=None):
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


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_merged_config(cfg: Any, save_dir: str | Path, filename: str = "merged_config.yaml") -> Path:
    save_dir = ensure_dir(save_dir)
    out_path = save_dir / filename
    plain = _to_plain(cfg)

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(plain, f, sort_keys=False, allow_unicode=True)

    return out_path


def copy_config_snapshots(
    base_cfg_path: str | Path | None,
    stage_cfg_path: str | Path | None,
    save_dir: str | Path,
    base_name: str = "base_config.yaml",
    stage_name: str = "stage_config.yaml",
) -> dict[str, str]:
    save_dir = ensure_dir(save_dir)
    saved = {}

    if base_cfg_path is not None:
        src = Path(base_cfg_path)
        if src.exists():
            dst = save_dir / base_name
            shutil.copy2(src, dst)
            saved["base"] = str(dst)

    if stage_cfg_path is not None:
        src = Path(stage_cfg_path)
        if src.exists():
            dst = save_dir / stage_name
            shutil.copy2(src, dst)
            saved["stage"] = str(dst)

    return saved


def save_run_metadata(metadata: dict, save_dir: str | Path, filename: str = "run_metadata.json") -> Path:
    save_dir = ensure_dir(save_dir)
    out_path = save_dir / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return out_path
