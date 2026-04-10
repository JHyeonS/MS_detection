from pathlib import Path
import math
import pandas as pd

from torch.utils.data import DataLoader

from src.detection.dataset.finetune_dataset import FinetuneDataset
from src.detection.dataset.transforms import build_transforms


def _cfg_get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
        if cur is None:
            return default
    return cur


def _split_dir(cfg):
    return Path(_cfg_get(cfg, "data", "split_dir"))


def _run_root(cfg):
    return Path(_cfg_get(cfg, "paths", "run_root", default="./runs"))


def _experiment_name(cfg):
    return str(_cfg_get(cfg, "data", "experiment", default="default_exp"))


def _sanitize_float_for_name(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _load_split_df(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"'label' column is required for label-fraction sampling: {csv_path}")
    return df


def _apply_labeled_fraction(cfg, csv_path: Path, split: str):
    if split != "train":
        return csv_path, None

    use_labeled_fraction = bool(_cfg_get(cfg, "train", "use_labeled_fraction", default=False))
    labeled_fraction = float(_cfg_get(cfg, "train", "labeled_fraction", default=1.0))
    balance_fraction_by_class = bool(_cfg_get(cfg, "train", "balance_fraction_by_class", default=True))
    min_samples_per_class = int(_cfg_get(cfg, "train", "min_samples_per_class", default=1))
    fraction_seed = _cfg_get(cfg, "train", "fraction_seed", default=None)
    if fraction_seed is None:
        fraction_seed = int(_cfg_get(cfg, "train", "seed", default=42))

    full_df = _load_split_df(csv_path)

    if (not use_labeled_fraction) or labeled_fraction >= 1.0:
        info = {
            "enabled": False,
            "split": split,
            "original_csv": str(csv_path),
            "effective_csv": str(csv_path),
            "original_num_rows": int(len(full_df)),
            "effective_num_rows": int(len(full_df)),
            "labeled_fraction": 1.0,
            "balance_fraction_by_class": balance_fraction_by_class,
            "min_samples_per_class": min_samples_per_class,
            "fraction_seed": int(fraction_seed),
            "per_class_original": {str(k): int(v) for k, v in full_df["label"].value_counts().sort_index().to_dict().items()},
            "per_class_effective": {str(k): int(v) for k, v in full_df["label"].value_counts().sort_index().to_dict().items()},
        }
        return csv_path, info

    if labeled_fraction <= 0.0:
        raise ValueError(f"train.labeled_fraction must be > 0, got {labeled_fraction}")

    if balance_fraction_by_class:
        sampled_parts = []
        for label_value, g in full_df.groupby("label", sort=True):
            target_n = max(int(math.floor(len(g) * labeled_fraction)), min_samples_per_class)
            target_n = min(target_n, len(g))
            sampled = g.sample(n=target_n, random_state=int(fraction_seed))
            sampled_parts.append(sampled)
        sampled_df = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=int(fraction_seed)).reset_index(drop=True)
    else:
        target_n = max(int(math.floor(len(full_df) * labeled_fraction)), min_samples_per_class)
        target_n = min(target_n, len(full_df))
        sampled_df = full_df.sample(n=target_n, random_state=int(fraction_seed)).reset_index(drop=True)

    cache_dir = _run_root(cfg) / "_label_fraction_cache" / _experiment_name(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frac_name = _sanitize_float_for_name(labeled_fraction)
    out_csv = cache_dir / f"{split}_frac{frac_name}_seed{int(fraction_seed)}.csv"
    sampled_df.to_csv(out_csv, index=False)

    original_counts = full_df["label"].value_counts().sort_index().to_dict()
    effective_counts = sampled_df["label"].value_counts().sort_index().to_dict()
    info = {
        "enabled": True,
        "split": split,
        "original_csv": str(csv_path),
        "effective_csv": str(out_csv),
        "original_num_rows": int(len(full_df)),
        "effective_num_rows": int(len(sampled_df)),
        "labeled_fraction": float(labeled_fraction),
        "balance_fraction_by_class": bool(balance_fraction_by_class),
        "min_samples_per_class": int(min_samples_per_class),
        "fraction_seed": int(fraction_seed),
        "per_class_original": {str(k): int(v) for k, v in original_counts.items()},
        "per_class_effective": {str(k): int(v) for k, v in effective_counts.items()},
    }
    return out_csv, info


def build_finetune_dataloader(cfg, csv_path, split="train"):
    csv_path = Path(csv_path)
    effective_csv_path, label_efficiency_info = _apply_labeled_fraction(cfg, csv_path, split=split)

    transform = build_transforms("finetune" if split == "train" else "eval")
    label_map = _cfg_get(cfg, "train", "label_map", default={}) or {}

    dataset = FinetuneDataset(
        csv_path=str(effective_csv_path),
        normalize=_cfg_get(cfg, "data", "normalize", default="robust"),
        transform=transform,
        preprocess=_cfg_get(cfg, "data", "preprocess", default={}),
        add_channel_dim=bool(_cfg_get(cfg, "data", "add_channel_dim", default=True)),
        return_meta=bool(_cfg_get(cfg, "data", "return_meta", default=False)),
        label_map=label_map,
    )

    dataset.label_efficiency_info = label_efficiency_info
    dataset.original_csv_path = str(csv_path)
    dataset.effective_csv_path = str(effective_csv_path)

    if split == "train":
        batch_size = int(_cfg_get(cfg, "train", "batch_size", default=16))
        shuffle = True
        drop_last = bool(_cfg_get(cfg, "train", "drop_last", default=True))
    else:
        batch_size = int(_cfg_get(cfg, "train", "eval_batch_size", default=_cfg_get(cfg, "train", "batch_size", default=16)))
        shuffle = False
        drop_last = False

    num_workers = int(_cfg_get(cfg, "data", "num_workers", default=4))
    pin_memory = bool(_cfg_get(cfg, "data", "pin_memory", default=True))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_finetune_dataloaders(cfg):
    split_dir = _split_dir(cfg)
    train_csv = split_dir / _cfg_get(cfg, "data", "train_csv", default="train.csv")
    val_csv = split_dir / _cfg_get(cfg, "data", "val_csv", default="val.csv")
    test_csv = split_dir / _cfg_get(cfg, "data", "test_csv", default="test.csv")

    train_loader = build_finetune_dataloader(cfg, train_csv, split="train")
    val_loader = build_finetune_dataloader(cfg, val_csv, split="val") if val_csv.exists() else None
    test_loader = build_finetune_dataloader(cfg, test_csv, split="test") if test_csv.exists() else None
    return train_loader, val_loader, test_loader
