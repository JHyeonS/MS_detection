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