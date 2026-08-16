#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a Utah 2023 split where label-0 noise windows are replaced by synthetic Gaussian noise."
    )
    p.add_argument("--src-split-dir", default="data/0406/metadata/experiments/stage1_utah_2023_only")
    p.add_argument("--out-split-dir", default="runs/utah_2023_gaussian_noise_split/splits")
    p.add_argument("--noise-root", default="runs/utah_2023_gaussian_noise_split/gaussian_noise")
    p.add_argument("--mean", type=float, default=0.0)
    p.add_argument("--std", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def gaussian_path(noise_root: Path, split: str, row: pd.Series) -> Path:
    label_name = str(row.get("label_name", "noise"))
    sample_id = str(row.get("sample_index_within_class", row.name)).zfill(7)
    return noise_root / split / label_name / f"{sample_id}.npy"


def generate_noise_like(src_path: Path, out_path: Path, mean: float, std: float, seed: int, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return

    src = np.load(src_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=std, size=src.shape).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, noise)


def main() -> None:
    args = parse_args()
    src_split_dir = Path(args.src_split_dir)
    out_split_dir = Path(args.out_split_dir)
    noise_root = Path(args.noise_root)
    out_split_dir.mkdir(parents=True, exist_ok=True)
    noise_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "src_split_dir": str(src_split_dir),
        "out_split_dir": str(out_split_dir),
        "noise_root": str(noise_root),
        "gaussian_mean": args.mean,
        "gaussian_std": args.std,
        "seed": args.seed,
        "splits": {},
    }

    for split in ["pretrain", "train", "val", "test"]:
        src_csv = src_split_dir / f"{split}.csv"
        df = pd.read_csv(src_csv).copy()
        if "synthetic_noise" not in df.columns:
            df["synthetic_noise"] = False
        if "original_npy_path" not in df.columns:
            df["original_npy_path"] = ""
        if "gaussian_mean" not in df.columns:
            df["gaussian_mean"] = np.nan
        if "gaussian_std" not in df.columns:
            df["gaussian_std"] = np.nan
        replaced = 0

        for idx, row in df.iterrows():
            if int(row["label"]) != 0:
                continue

            src_path = Path(str(row["npy_path"]))
            out_path = gaussian_path(noise_root, split, row)
            row_seed = int(args.seed + idx + 100000 * ["pretrain", "train", "val", "test"].index(split))
            generate_noise_like(src_path, out_path, args.mean, args.std, row_seed, args.overwrite)

            df.at[idx, "original_npy_path"] = str(src_path)
            df.at[idx, "npy_path"] = str(out_path.resolve())
            df.at[idx, "synthetic_noise"] = True
            df.at[idx, "gaussian_mean"] = args.mean
            df.at[idx, "gaussian_std"] = args.std
            replaced += 1

        df["synthetic_noise"] = df["synthetic_noise"].fillna(False)
        df["original_npy_path"] = df["original_npy_path"].fillna("")

        out_csv = out_split_dir / f"{split}.csv"
        df.to_csv(out_csv, index=False)
        summary["splits"][split] = {
            "src_csv": str(src_csv),
            "out_csv": str(out_csv),
            "n_rows": int(len(df)),
            "n_synthetic_noise": int(replaced),
            "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()},
        }
        print(f"[WRITE] {out_csv} rows={len(df)} synthetic_noise={replaced}")

    with (out_split_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[WRITE] {out_split_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
