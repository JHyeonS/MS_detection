from pathlib import Path

from torch.utils.data import DataLoader

from src.dataset.finetune_dataset import FinetuneDataset
from src.dataset.transforms import build_transforms


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


def build_finetune_dataloader(cfg, csv_path, split="train"):
    transform = build_transforms("finetune" if split == "train" else "eval")
    label_map = _cfg_get(cfg, "train", "label_map", default={}) or {}

    dataset = FinetuneDataset(
        csv_path=str(csv_path),
        normalize=_cfg_get(cfg, "data", "normalize", default="robust"),
        transform=transform,
        preprocess=_cfg_get(cfg, "data", "preprocess", default={}),
        add_channel_dim=bool(_cfg_get(cfg, "data", "add_channel_dim", default=True)),
        return_meta=bool(_cfg_get(cfg, "data", "return_meta", default=False)),
        label_map=label_map,
    )

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
