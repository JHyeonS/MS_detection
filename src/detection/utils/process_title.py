#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import os


def _short_title(full_title: str) -> str:
    return full_title.encode("utf-8", errors="ignore")[:15].decode("utf-8", errors="ignore")


def set_process_title(task: str, *, gpu_hint: str | None = None) -> str:
    owner = os.environ.get("MS_JOB_OWNER") or os.environ.get("USER") or "unknown"
    visible = gpu_hint or os.environ.get("CUDA_VISIBLE_DEVICES") or "all"
    full_title = f"{owner}:{task}:gpu{visible}"
    short_title = _short_title(f"{owner}:{task}")

    try:
        libc = ctypes.CDLL(None)
        prctl = libc.prctl
        prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        prctl(15, short_title.encode("utf-8"), 0, 0, 0)
    except Exception:
        pass

    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(full_title)
    except Exception:
        pass

    return full_title
