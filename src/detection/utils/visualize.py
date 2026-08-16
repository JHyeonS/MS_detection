#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/utils/visualize.py

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt


def save_loss_curve(
    losses: Sequence[float],
    save_path: str | Path,
    title: str = "Training Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close("all")


def save_train_history_csv(
    losses: Sequence[float],
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss\n")
        for i, loss in enumerate(losses, start=1):
            f.write(f"{i},{loss}\n")


def save_metrics_history_csv(
    rows,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    if not rows:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("")
        return

    columns = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for key in columns:
                value = row.get(key, "")
                values.append(str(value))
            f.write(",".join(values) + "\n")
