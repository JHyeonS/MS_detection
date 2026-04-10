#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/analysis/tsne_splits.py

목표
1) leakage check:
   - raw     : train / val / test
   - encoder : train / val / test

2) paper figure:
   - raw     : site (pohang / utah)
   - encoder : site (pohang / utah)

3) 기존 style outputs도 유지:
   - raw / encoder events-only
   - raw / encoder events+noises
   - raw / encoder domain-labeled

중요
- dataloader는 수정하지 않는다.
- site/domain은 batch meta가 아니라 CSV를 직접 읽어서 복원한다.
- train loader가 shuffle=True여도, TSNE용 no-shuffle loader를 별도로 다시 생성한다.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.models.cnn_encoder import cnn_encoder
from src.detection.dataloader.finetune_dataloader import build_finetune_dataloaders, build_finetune_dataloader


# =========================================================
# Config helpers
# =========================================================
class AttrDict(dict):
    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(item)
        v = self.get(item)
        if isinstance(v, dict) and not isinstance(v, AttrDict):
            v = AttrDict(v)
            self[item] = v
        return v

    def __setattr__(self, key, value):
        self[key] = value


def _to_attrdict(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    return obj


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(base_cfg_path: str | Path, stage_cfg_path: str | Path):
    base_cfg = _load_yaml(base_cfg_path)
    stage_cfg = _load_yaml(stage_cfg_path)
    merged = _deep_update(base_cfg, stage_cfg)
    return _to_attrdict(merged)


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


# =========================================================
# Dataloader resolve
# =========================================================
def resolve_train_val_test_dataloaders(cfg):
    out = build_finetune_dataloaders(cfg)
    if not isinstance(out, tuple) or len(out) < 3:
        raise RuntimeError("build_finetune_dataloaders(cfg) must return (train, val, test)")
    return out[0], out[1], out[2]


def rebuild_loader_noshuffle(loader, split_name: str):
    """
    기존 dataloader의 dataset/effective_csv_path를 사용해서
    TSNE용 no-shuffle loader를 다시 만든다.
    """
    dataset = loader.dataset
    csv_path = getattr(dataset, "effective_csv_path", None)
    if csv_path is None:
        raise RuntimeError(f"{split_name} loader.dataset.effective_csv_path not found")

    batch_size = loader.batch_size
    num_workers = loader.num_workers
    pin_memory = getattr(loader, "pin_memory", True)

    new_loader = build_finetune_dataloader(
        cfg=loader._tsne_cfg,   # 아래 attach_cfg_to_loader에서 붙여줌
        csv_path=csv_path,
        split=split_name,
    )

    # 강제로 no-shuffle DataLoader 재구성
    new_loader = DataLoader(
        new_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return new_loader


def attach_cfg_to_loader(loader, cfg):
    setattr(loader, "_tsne_cfg", cfg)
    return loader


# =========================================================
# Model
# =========================================================
class EncoderOnly(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = cnn_encoder(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def load_encoder_weights(model: EncoderOnly, ckpt_path: str | Path, device: torch.device):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = None
    if isinstance(ckpt, dict):
        if "encoder_state_dict" in ckpt:
            state_dict = ckpt["encoder_state_dict"]
        elif "model_state_dict" in ckpt:
            msd = ckpt["model_state_dict"]
            state_dict = {k[len("encoder."):]: v for k, v in msd.items() if k.startswith("encoder.")}
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]

    if state_dict is None:
        # fallback
        state_dict = ckpt

    missing, unexpected = model.encoder.load_state_dict(state_dict, strict=False)
    print(f"[INFO] loaded encoder weights from: {ckpt_path}")
    print(f"[INFO] missing keys   : {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()


# =========================================================
# Batch/meta utils
# =========================================================
def parse_batch(batch):
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("Batch tuple/list must have at least 2 items.")

    if isinstance(batch, dict):
        x = batch.get("x", batch.get("input", batch.get("waveform", batch.get("data", None))))
        y = batch.get("y", batch.get("label", batch.get("target", batch.get("labels", None))))
        if x is None or y is None:
            raise ValueError(f"Unsupported batch dict keys: {list(batch.keys())}")
        return x, y

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def label_to_name(y: int) -> str:
    return "event" if int(y) == 1 else "noise"


# =========================================================
# CSV-based site recovery
# =========================================================
def infer_site_from_row(row: pd.Series) -> str:
    search_cols = [
        "site",
        "source",
        "source_csv",
        "raw_file_path",
        "npy_path",
        "file_name",
        "file_stem",
        "path",
    ]

    text_parts = []
    for col in search_cols:
        if col in row.index:
            text_parts.append(str(row[col]).lower())

    joined = " | ".join(text_parts)

    if "pohang" in joined:
        return "pohang"
    if "utah" in joined or "forge" in joined:
        return "utah"
    return "unknown"


def load_csv_meta(csv_path: str | Path, split_name: str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path).reset_index(drop=True)

    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in CSV: {csv_path}")

    df["split"] = split_name
    df["class_name"] = df["label"].apply(label_to_name)
    df["domain"] = df.apply(infer_site_from_row, axis=1)
    return df


# =========================================================
# Feature collection
# =========================================================
@torch.no_grad()
def collect_split_features_from_loader_and_csv(
    loader,
    csv_df: pd.DataFrame,
    split_name: str,
    model: Optional[EncoderOnly],
    device: torch.device,
    max_per_group: int,
    use_encoder: bool,
) -> pd.DataFrame:
    """
    batch meta를 쓰지 않고, CSV row 순서를 기준으로 site/domain을 직접 붙인다.
    train은 반드시 no-shuffle loader를 써야 한다.
    """
    rows: List[Dict[str, Any]] = []
    group_counts: Dict[str, int] = {}
    global_row_idx = 0

    for batch_idx, batch in enumerate(loader):
        x, y = parse_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.view(-1).long()

        if use_encoder:
            feat = model(x).detach().cpu().numpy()
        else:
            feat = x.detach().cpu().numpy().reshape(x.size(0), -1)

        bs = x.size(0)

        for i in range(bs):
            if global_row_idx >= len(csv_df):
                break

            meta_row = csv_df.iloc[global_row_idx]
            global_row_idx += 1

            yi = int(y[i].item())
            csv_label = int(meta_row["label"])

            # sanity check
            if yi != csv_label:
                print(
                    f"[WARN] label mismatch at split={split_name}, row={global_row_idx-1}: "
                    f"loader={yi}, csv={csv_label}"
                )

            cls_name = label_to_name(yi)
            group_name = f"{split_name}_{cls_name}"

            cur_n = group_counts.get(group_name, 0)
            if cur_n >= max_per_group:
                continue

            row = {
                "split": split_name,
                "label": yi,
                "class_name": cls_name,
                "group": group_name,
                "domain": str(meta_row["domain"]),
                "feat": feat[i].astype(np.float32),
                "batch_idx": batch_idx,
                "sample_idx": i,
                "csv_row_idx": int(global_row_idx - 1),
            }

            # CSV 컬럼도 같이 저장
            for col in csv_df.columns:
                val = meta_row[col]
                if isinstance(val, (str, int, float, np.integer, np.floating)):
                    row[f"meta_{col}"] = val

            rows.append(row)
            group_counts[group_name] = cur_n + 1

    df = pd.DataFrame(rows)
    print(f"[INFO] collected {len(df)} samples for split={split_name}, use_encoder={use_encoder}")
    if not df.empty:
        print(df.groupby(["split", "class_name"]).size())
        if "domain" in df.columns:
            print(df.groupby(["split", "domain"]).size())
    return df


# =========================================================
# TSNE / Plot
# =========================================================
def run_tsne(feat_matrix: np.ndarray, seed: int, perplexity: float, pca_dim: int) -> np.ndarray:
    feat_matrix = StandardScaler().fit_transform(feat_matrix)

    n_samples = feat_matrix.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for t-SNE")

    pca_dim = max(2, min(pca_dim, feat_matrix.shape[1], n_samples - 1))
    feat_matrix = PCA(n_components=pca_dim, random_state=seed).fit_transform(feat_matrix)

    perplexity = min(perplexity, max(5, n_samples - 1))
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 3)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    emb = tsne.fit_transform(feat_matrix)
    return emb


def compute_tsne_df(df: pd.DataFrame, seed: int, perplexity: float, pca_dim: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    feats = np.stack(df["feat"].to_list(), axis=0)
    emb = run_tsne(feats, seed=seed, perplexity=perplexity, pca_dim=pca_dim)

    out = df.copy()
    out["tsne_x"] = emb[:, 0]
    out["tsne_y"] = emb[:, 1]
    return out


def save_tsne_csv(df: pd.DataFrame, out_png: Path):
    csv_path = out_png.with_suffix(".csv")
    save_df = df.copy()
    if "feat" in save_df.columns:
        save_df = save_df.drop(columns=["feat"])
    save_df.to_csv(csv_path, index=False)
    print(f"[INFO] saved: {csv_path}")


def plot_split_style(df: pd.DataFrame, title: str, out_png: Path, figsize=(11, 9), dpi=220):
    if df.empty:
        print(f"[WARN] empty df: {out_png}")
        return

    color_map = {"event": "tab:red", "noise": "tab:blue"}
    marker_map = {"train": "o", "val": "s", "test": "^"}

    plt.figure(figsize=figsize)

    for cls_name in ["event", "noise"]:
        for split_name in ["train", "val", "test"]:
            sub = df[(df["class_name"] == cls_name) & (df["split"] == split_name)]
            if sub.empty:
                continue
            plt.scatter(
                sub["tsne_x"].to_numpy(),
                sub["tsne_y"].to_numpy(),
                s=18,
                alpha=0.72,
                c=color_map.get(cls_name, "gray"),
                marker=marker_map.get(split_name, "x"),
                label=f"{split_name}_{cls_name}",
            )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.2, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

    print(f"[INFO] saved: {out_png}")
    save_tsne_csv(df, out_png)


def plot_site_style(df: pd.DataFrame, title: str, out_png: Path, figsize=(11, 9), dpi=220):
    if df.empty:
        print(f"[WARN] empty df: {out_png}")
        return

    known_df = df[df["domain"].isin(["pohang", "utah"])].copy()
    if known_df.empty:
        print(f"[WARN] no known domains for site plot: {out_png}")
        return

    color_map = {"event": "tab:red", "noise": "tab:blue"}
    marker_map = {"pohang": "o", "utah": "^"}

    plt.figure(figsize=figsize)

    for cls_name in ["event", "noise"]:
        for domain_name in ["pohang", "utah"]:
            sub = known_df[(known_df["class_name"] == cls_name) & (known_df["domain"] == domain_name)]
            if sub.empty:
                continue
            plt.scatter(
                sub["tsne_x"].to_numpy(),
                sub["tsne_y"].to_numpy(),
                s=20,
                alpha=0.75,
                c=color_map.get(cls_name, "gray"),
                marker=marker_map.get(domain_name, "x"),
                label=f"{domain_name}_{cls_name}",
            )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

    print(f"[INFO] saved: {out_png}")
    save_tsne_csv(known_df, out_png)


def plot_by_label(df: pd.DataFrame, label_col: str, title: str, out_png: Path, figsize=(10, 8), dpi=220):
    if df.empty:
        print(f"[WARN] empty dataframe, skip: {out_png}")
        return

    plt.figure(figsize=figsize)
    for lab in df[label_col].unique().tolist():
        sub = df[df[label_col] == lab]
        plt.scatter(
            sub["tsne_x"].to_numpy(),
            sub["tsne_y"].to_numpy(),
            s=18,
            alpha=0.7,
            label=str(lab),
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

    print(f"[INFO] saved: {out_png}")
    save_tsne_csv(df, out_png)


def make_subset(df: pd.DataFrame, groups: List[str], display_mode: str = "group") -> pd.DataFrame:
    out = df[df["group"].isin(groups)].copy()
    if out.empty:
        return out

    if display_mode == "group":
        out["display_label"] = out["group"]
    elif display_mode == "group_domain":
        out["display_label"] = out["group"] + "_" + out["domain"]
    else:
        raise ValueError(f"Unknown display_mode: {display_mode}")

    return out


def has_known_domain(df: pd.DataFrame) -> bool:
    if df.empty or "domain" not in df.columns:
        return False
    return (df["domain"] != "unknown").any()


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cfg", type=str, required=True)
    parser.add_argument("--stage_cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help="encoder/finetune/pretrain checkpoint")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_per_group", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--pca_dim", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.base_cfg, args.stage_cfg)

    # TSNE는 CPU로 돌리는 게 안정적
    device = torch.device("cpu")
    print(f"[INFO] device: {device}")

    # 원래 loader
    train_loader, val_loader, test_loader = resolve_train_val_test_dataloaders(cfg)
    train_loader = attach_cfg_to_loader(train_loader, cfg)
    if val_loader is not None:
        val_loader = attach_cfg_to_loader(val_loader, cfg)
    if test_loader is not None:
        test_loader = attach_cfg_to_loader(test_loader, cfg)

    # TSNE용 no-shuffle loader 재생성
    train_loader_tsne = rebuild_loader_noshuffle(train_loader, "train")
    val_loader_tsne = rebuild_loader_noshuffle(val_loader, "val") if val_loader is not None else None
    test_loader_tsne = rebuild_loader_noshuffle(test_loader, "test") if test_loader is not None else None

    # CSV 메타 불러오기
    train_csv = getattr(train_loader.dataset, "effective_csv_path", None)
    val_csv = getattr(val_loader.dataset, "effective_csv_path", None) if val_loader is not None else None
    test_csv = getattr(test_loader.dataset, "effective_csv_path", None) if test_loader is not None else None

    train_csv_df = load_csv_meta(train_csv, "train")
    val_csv_df = load_csv_meta(val_csv, "val") if val_csv is not None else pd.DataFrame()
    test_csv_df = load_csv_meta(test_csv, "test") if test_csv is not None else pd.DataFrame()

    # -------------------------
    # raw features
    # -------------------------
    raw_train = collect_split_features_from_loader_and_csv(
        loader=train_loader_tsne,
        csv_df=train_csv_df,
        split_name="train",
        model=None,
        device=device,
        max_per_group=args.max_per_group,
        use_encoder=False,
    )
    raw_val = collect_split_features_from_loader_and_csv(
        loader=val_loader_tsne,
        csv_df=val_csv_df,
        split_name="val",
        model=None,
        device=device,
        max_per_group=args.max_per_group,
        use_encoder=False,
    ) if val_loader_tsne is not None else pd.DataFrame()
    raw_test = collect_split_features_from_loader_and_csv(
        loader=test_loader_tsne,
        csv_df=test_csv_df,
        split_name="test",
        model=None,
        device=device,
        max_per_group=args.max_per_group,
        use_encoder=False,
    ) if test_loader_tsne is not None else pd.DataFrame()

    raw_parts = [d for d in [raw_train, raw_val, raw_test] if not d.empty]
    raw_df = pd.concat(raw_parts, ignore_index=True) if len(raw_parts) > 0 else pd.DataFrame()
    if not raw_df.empty:
        raw_df.to_pickle(out_dir / "raw_features.pkl")
        raw_df.drop(columns=["feat"]).to_csv(out_dir / "raw_features_meta.csv", index=False)

    # -------------------------
    # encoder features
    # -------------------------
    enc_df = pd.DataFrame()
    if args.ckpt is not None:
        model = EncoderOnly(cfg).to(device)
        load_encoder_weights(model, args.ckpt, device)

        enc_train = collect_split_features_from_loader_and_csv(
            loader=train_loader_tsne,
            csv_df=train_csv_df,
            split_name="train",
            model=model,
            device=device,
            max_per_group=args.max_per_group,
            use_encoder=True,
        )
        enc_val = collect_split_features_from_loader_and_csv(
            loader=val_loader_tsne,
            csv_df=val_csv_df,
            split_name="val",
            model=model,
            device=device,
            max_per_group=args.max_per_group,
            use_encoder=True,
        ) if val_loader_tsne is not None else pd.DataFrame()
        enc_test = collect_split_features_from_loader_and_csv(
            loader=test_loader_tsne,
            csv_df=test_csv_df,
            split_name="test",
            model=model,
            device=device,
            max_per_group=args.max_per_group,
            use_encoder=True,
        ) if test_loader_tsne is not None else pd.DataFrame()

        enc_parts = [d for d in [enc_train, enc_val, enc_test] if not d.empty]
        enc_df = pd.concat(enc_parts, ignore_index=True) if len(enc_parts) > 0 else pd.DataFrame()
        if not enc_df.empty:
            enc_df.to_pickle(out_dir / "encoder_features.pkl")
            enc_df.drop(columns=["feat"]).to_csv(out_dir / "encoder_features_meta.csv", index=False)
    else:
        print("[WARN] --ckpt not provided. encoder TSNE will be skipped.")

    # =====================================================
    # 0) split plots (leakage check)
    # =====================================================
    if not raw_df.empty:
        raw_split_df = compute_tsne_df(raw_df, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
        plot_split_style(
            raw_split_df,
            title="RAW t-SNE: split leakage check",
            out_png=out_dir / "tsne_raw_split.png",
        )

    if not enc_df.empty:
        enc_split_df = compute_tsne_df(enc_df, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
        plot_split_style(
            enc_split_df,
            title="ENCODER t-SNE: split leakage check",
            out_png=out_dir / "tsne_encoder_split.png",
        )

    # =====================================================
    # 1) site plots (paper figure)
    # =====================================================
    if not raw_df.empty:
        raw_site_df = compute_tsne_df(raw_df, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
        plot_site_style(
            raw_site_df,
            title="RAW t-SNE: site distribution",
            out_png=out_dir / "tsne_raw_site.png",
        )

    if not enc_df.empty:
        enc_site_df = compute_tsne_df(enc_df, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
        plot_site_style(
            enc_site_df,
            title="ENCODER t-SNE: site distribution",
            out_png=out_dir / "tsne_encoder_site.png",
        )

    # =====================================================
    # 2) 기존 style outputs도 유지
    # =====================================================
    event_only_groups = ["train_event", "val_event", "test_event"]
    event_noise_groups = [
        "train_event", "val_event", "test_event",
        "train_noise", "val_noise", "test_noise",
    ]

    if not raw_df.empty:
        raw_event_df = make_subset(raw_df, event_only_groups, display_mode="group")
        if not raw_event_df.empty:
            raw_event_df = compute_tsne_df(raw_event_df, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
            plot_by_label(
                raw_event_df,
                label_col="display_label",
                title="RAW t-SNE: train/val/test events",
                out_png=out_dir / "tsne_raw_events_only.png",
                figsize=(10, 8),
            )

        raw_all_df = make_subset(raw_df, event_noise_groups, display_mode="group")
        if not raw_all_df.empty:
            raw_all_df = compute_tsne_df(raw_all_df, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
            plot_by_label(
                raw_all_df,
                label_col="display_label",
                title="RAW t-SNE: train/val/test events + noises",
                out_png=out_dir / "tsne_raw_events_noises.png",
                figsize=(11, 9),
            )

    if not enc_df.empty:
        enc_event_df = make_subset(enc_df, event_only_groups, display_mode="group")
        if not enc_event_df.empty:
            enc_event_df = compute_tsne_df(enc_event_df, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
            plot_by_label(
                enc_event_df,
                label_col="display_label",
                title="ENCODER t-SNE: train/val/test events",
                out_png=out_dir / "tsne_encoder_events_only.png",
                figsize=(10, 8),
            )

        enc_all_df = make_subset(enc_df, event_noise_groups, display_mode="group")
        if not enc_all_df.empty:
            enc_all_df = compute_tsne_df(enc_all_df, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
            plot_by_label(
                enc_all_df,
                label_col="display_label",
                title="ENCODER t-SNE: train/val/test events + noises",
                out_png=out_dir / "tsne_encoder_events_noises.png",
                figsize=(11, 9),
            )

    # =====================================================
    # 3) domain-labeled outputs
    # =====================================================
    if has_known_domain(raw_df):
        raw_domain_df = make_subset(raw_df, event_noise_groups, display_mode="group_domain")
        if not raw_domain_df.empty:
            raw_domain_df = compute_tsne_df(raw_domain_df, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
            plot_by_label(
                raw_domain_df,
                label_col="display_label",
                title="RAW t-SNE with domain labels",
                out_png=out_dir / "tsne_raw_events_noises_domain.png",
                figsize=(12, 10),
            )

    if not enc_df.empty and has_known_domain(enc_df):
        enc_domain_df = make_subset(enc_df, event_noise_groups, display_mode="group_domain")
        if not enc_domain_df.empty:
            enc_domain_df = compute_tsne_df(enc_domain_df, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
            plot_by_label(
                enc_domain_df,
                label_col="display_label",
                title="ENCODER t-SNE with domain labels",
                out_png=out_dir / "tsne_encoder_events_noises_domain.png",
                figsize=(12, 10),
            )

    summary = {
        "out_dir": str(out_dir),
        "raw_total": int(len(raw_df)),
        "encoder_total": int(len(enc_df)),
        "raw_domain_known": bool(has_known_domain(raw_df)) if not raw_df.empty else False,
        "encoder_domain_known": bool(has_known_domain(enc_df)) if not enc_df.empty else False,
        "requested_outputs": [
            "tsne_raw_split.png",
            "tsne_encoder_split.png",
            "tsne_raw_site.png",
            "tsne_encoder_site.png",
            "tsne_raw_events_only.png",
            "tsne_raw_events_noises.png",
            "tsne_encoder_events_only.png",
            "tsne_encoder_events_noises.png",
        ],
        "optional_outputs": [
            "tsne_raw_events_noises_domain.png",
            "tsne_encoder_events_noises_domain.png",
        ],
    }

    with open(out_dir / "tsne_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()