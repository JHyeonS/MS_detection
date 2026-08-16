from pathlib import Path

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


def _num_workers(cfg) -> int:
    return int(_cfg_get(cfg, "data", "num_workers", default=_cfg_get(cfg, "test", "num_workers", default=4)))


def _pin_memory(cfg) -> bool:
    return bool(_cfg_get(cfg, "data", "pin_memory", default=_cfg_get(cfg, "test", "pin_memory", default=True)))


def build_test_dataloader(cfg, csv_path=None):
    split_dir = Path(_cfg_get(cfg, "data", "split_dir"))
    if csv_path is None:
        csv_path = split_dir / _cfg_get(cfg, "data", "test_csv", default="test.csv")

    dataset = FinetuneDataset(
        csv_path=str(csv_path),
        normalize=_cfg_get(cfg, "data", "normalize", default="robust"),
        transform=build_transforms("eval"),
        preprocess=_cfg_get(cfg, "data", "preprocess", default={}),
        add_channel_dim=bool(_cfg_get(cfg, "data", "add_channel_dim", default=True)),
        return_meta=bool(_cfg_get(cfg, "data", "return_meta", default=True)),
        label_map=_cfg_get(cfg, "train", "label_map", default={}) or {},
    )

    batch_size = int(_cfg_get(cfg, "test", "batch_size", default=16))
    num_workers = _num_workers(cfg)
    pin_memory = _pin_memory(cfg)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_test_loader(cfg, csv_path=None):
    return build_test_dataloader(cfg, csv_path=csv_path)
