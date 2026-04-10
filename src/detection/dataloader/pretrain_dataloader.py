from pathlib import Path

from torch.utils.data import DataLoader

from src.detection.dataset.pretrain_dataset import PretrainDataset
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


def _resolve_csv_path(cfg, csv_path=None):
    if csv_path is not None:
        return str(csv_path)
    split_dir = Path(_cfg_get(cfg, "data", "split_dir"))
    csv_name = _cfg_get(cfg, "data", "pretrain_csv", default="pretrain.csv")
    return str(split_dir / csv_name)


def build_pretrain_dataloader(cfg, csv_path=None, split="train"):
    mode = str(_cfg_get(cfg, "pretrain", "mode", default="contrast")).lower()
    transform = build_transforms("contrast" if mode in ["contrast", "contrastive", "simclr"] else "reconstruction")

    dataset = PretrainDataset(
        csv_path=_resolve_csv_path(cfg, csv_path),
        mode=mode,
        normalize=_cfg_get(cfg, "data", "normalize", default="robust"),
        transform=transform,
        preprocess=_cfg_get(cfg, "data", "preprocess", default={}),
        add_channel_dim=bool(_cfg_get(cfg, "data", "add_channel_dim", default=True)),
        allowed_labels=_cfg_get(cfg, "data", "allowed_labels", default=None),
    )

    batch_size = int(_cfg_get(cfg, "pretrain", "batch_size", default=16))
    num_workers = int(_cfg_get(cfg, "data", "num_workers", default=4))
    pin_memory = bool(_cfg_get(cfg, "data", "pin_memory", default=True))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )
    return loader


def build_contrast_pretrain_dataloader(cfg, csv_path=None, split="train"):
    return build_pretrain_dataloader(cfg, csv_path=csv_path, split=split)


def build_reconst_pretrain_dataloader(cfg, csv_path=None, split="train"):
    return build_pretrain_dataloader(cfg, csv_path=csv_path, split=split)
