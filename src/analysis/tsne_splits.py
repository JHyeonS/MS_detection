#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/analysis/tsne_splits.py

목표:
1) raw [train event / val event / test event]
2) raw [train event / val event / test event / train noise / val noise / test noise]
3) encoder [train event / val event / test event]
4) encoder [train event / val event / test event / train noise / val noise / test noise]

추가:
- meta/path/dataset/source 등에 "pohang" / "utah" 문자열이 있으면
  domain-separated TSNE도 자동 생성
- GPU 서버에서 feature extraction + TSNE까지 한 번에 수행 가능
  (주의: sklearn TSNE는 내부적으로 CPU 사용이 큼)

실행 예시:
python src/analysis/tsne_splits.py \
    --base_cfg config/base.yaml \
    --stage_cfg config/train.yaml \
    --ckpt runs/finetune/EXP_NAME/best.pt \
    --out_dir runs/analysis/EXP_NAME_tsne \
    --max_per_group 400 \
    --tsne_perplexity 30 \
    --seed 42
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

from src.models.cnn_encoder import cnn_encoder


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
    errors = []

    try:
        from src.dataloader.finetune_dataloader import build_finetune_dataloaders
        out = build_finetune_dataloaders(cfg)
        if isinstance(out, tuple) and len(out) >= 3:
            return out[0], out[1], out[2]
        errors.append("build_finetune_dataloaders(cfg) did not return (train, val, test)")
    except Exception as e:
        errors.append(f"build_finetune_dataloaders: {repr(e)}")

    raise ImportError(
        "Could not resolve train/val/test dataloaders.\n" + "\n".join(errors)
    )


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

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt

    # 1) direct load
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded_any = (len(missing) < len(model.state_dict())) or (len(unexpected) >= 0)

    # 2) encoder-only filtered load fallback
    if len(missing) == len(model.state_dict()):
        filtered = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                filtered[k] = v
            elif k.startswith("model.encoder."):
                filtered[k.replace("model.", "", 1)] = v
            elif k.startswith("backbone.encoder."):
                filtered[k.replace("backbone.", "", 1)] = v

        if filtered:
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            loaded_any = True

    print(f"[INFO] loaded ckpt: {ckpt_path}")
    print(f"[INFO] missing keys   : {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")

    if not loaded_any:
        raise RuntimeError(f"Could not load encoder weights from: {ckpt_path}")

    model.to(device)
    model.eval()


# =========================================================
# Batch/meta utils
# =========================================================
def parse_batch(batch):
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]
        if len(batch) >= 2:
            return batch[0], batch[1], None
        raise ValueError("Batch tuple/list must have at least 2 items.")

    if isinstance(batch, dict):
        x = batch.get("x", batch.get("input", batch.get("waveform", None)))
        y = batch.get("y", batch.get("label", batch.get("target", None)))
        meta = batch.get("meta", None)
        if x is None or y is None:
            raise ValueError(f"Unsupported batch dict keys: {list(batch.keys())}")
        return x, y, meta

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def to_python_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        return x.tolist()
    return x


def extract_meta_for_index(meta, idx: int, batch_size: int) -> Dict[str, Any]:
    out = {}

    if meta is None:
        return out

    if isinstance(meta, dict):
        for k, v in meta.items():
            try:
                if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) == batch_size:
                    val = v[idx]
                else:
                    val = v
            except Exception:
                val = v
            out[k] = to_python_scalar(val)
        return out

    out["meta"] = str(meta)
    return out


def infer_domain(meta_dict: Dict[str, Any]) -> str:
    search_keys = [
        "dataset", "source", "domain", "path", "filepath", "file", "h5_path", "root"
    ]
    text_parts = []

    for k, v in meta_dict.items():
        if k in search_keys or "path" in k.lower() or "file" in k.lower() or "dataset" in k.lower():
            text_parts.append(str(v).lower())

    joined = " | ".join(text_parts)

    if "pohang" in joined:
        return "pohang"
    if "utah" in joined:
        return "utah"
    return "unknown"


def label_to_name(y: int) -> str:
    return "event" if int(y) == 1 else "noise"


# =========================================================
# Feature collection
# =========================================================
@torch.no_grad()
def collect_split_features(
    loader,
    split_name: str,
    model: Optional[EncoderOnly],
    device: torch.device,
    max_per_group: int,
    use_encoder: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_counts: Dict[str, int] = {}

    for batch_idx, batch in enumerate(loader):
        x, y, meta = parse_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.view(-1).long()

        if use_encoder:
            feat = model(x).detach().cpu().numpy()
        else:
            feat = x.detach().cpu().numpy().reshape(x.size(0), -1)

        bs = x.size(0)

        for i in range(bs):
            yi = int(y[i].item())
            cls_name = label_to_name(yi)
            group_name = f"{split_name}_{cls_name}"

            cur_n = group_counts.get(group_name, 0)
            if cur_n >= max_per_group:
                continue

            meta_i = extract_meta_for_index(meta, i, bs)
            domain_i = infer_domain(meta_i)

            row = {
                "split": split_name,
                "label": yi,
                "class_name": cls_name,
                "group": group_name,
                "domain": domain_i,
                "feat": feat[i].astype(np.float32),
                "batch_idx": batch_idx,
                "sample_idx": i,
            }

            for k, v in meta_i.items():
                row[f"meta_{k}"] = v

            rows.append(row)
            group_counts[group_name] = cur_n + 1

    df = pd.DataFrame(rows)
    print(f"[INFO] collected {len(df)} samples for split={split_name}, use_encoder={use_encoder}")
    if not df.empty:
        print(df.groupby(["split", "class_name"]).size())
    return df


# =========================================================
# TSNE / Plot
# =========================================================
def run_tsne(feat_matrix: np.ndarray, seed: int, perplexity: float, pca_dim: int) -> np.ndarray:
    feat_matrix = StandardScaler().fit_transform(feat_matrix)

    n_samples = feat_matrix.shape[0]
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


def plot_tsne(df: pd.DataFrame, out_png: Path, title: str, color_col: str = "display_label"):
    if df.empty:
        print(f"[WARN] empty dataframe, skip: {out_png}")
        return

    feats = np.stack(df["feat"].to_list(), axis=0)
    emb = run_tsne(
        feat_matrix=feats,
        seed=42,
        perplexity=30,
        pca_dim=50,
    )

    plot_df = df.copy()
    plot_df["tsne_x"] = emb[:, 0]
    plot_df["tsne_y"] = emb[:, 1]

    plt.figure(figsize=(10, 8))
    unique_labels = plot_df[color_col].unique().tolist()

    for lab in unique_labels:
        sub = plot_df[plot_df[color_col] == lab]
        plt.scatter(
            sub["tsne_x"].to_numpy(),
            sub["tsne_y"].to_numpy(),
            s=18,
            alpha=0.7,
            label=lab,
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

    csv_path = out_png.with_suffix(".csv")
    plot_df.drop(columns=["feat"]).to_csv(csv_path, index=False)
    print(f"[INFO] saved: {out_png}")
    print(f"[INFO] saved: {csv_path}")


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
    parser.add_argument("--max_per_group", type=int, default=400)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    train_loader, val_loader, test_loader = resolve_train_val_test_dataloaders(cfg)

    # -------------------------
    # raw features
    # -------------------------
    raw_train = collect_split_features(
        loader=train_loader,
        split_name="train",
        model=None,
        device=device,
        max_per_group=args.max_per_group,
        use_encoder=False,
    )
    raw_val = collect_split_features(
        loader=val_loader,
        split_name="val",
        model=None,
        device=device,
        max_per_group=args.max_per_group,
        use_encoder=False,
    )
    raw_test = collect_split_features(
        loader=test_loader,
        split_name="test",
        model=None,
        device=device,
        max_per_group=args.max_per_group,
        use_encoder=False,
    )

    raw_df = pd.concat([raw_train, raw_val, raw_test], ignore_index=True)
    raw_df.to_pickle(out_dir / "raw_features.pkl")
    raw_df.drop(columns=["feat"]).to_csv(out_dir / "raw_features_meta.csv", index=False)

    # -------------------------
    # encoder features
    # -------------------------
    enc_df = pd.DataFrame()
    if args.ckpt is not None:
        model = EncoderOnly(cfg).to(device)
        load_encoder_weights(model, args.ckpt, device)

        enc_train = collect_split_features(
            loader=train_loader,
            split_name="train",
            model=model,
            device=device,
            max_per_group=args.max_per_group,
            use_encoder=True,
        )
        enc_val = collect_split_features(
            loader=val_loader,
            split_name="val",
            model=model,
            device=device,
            max_per_group=args.max_per_group,
            use_encoder=True,
        )
        enc_test = collect_split_features(
            loader=test_loader,
            split_name="test",
            model=model,
            device=device,
            max_per_group=args.max_per_group,
            use_encoder=True,
        )

        enc_df = pd.concat([enc_train, enc_val, enc_test], ignore_index=True)
        enc_df.to_pickle(out_dir / "encoder_features.pkl")
        enc_df.drop(columns=["feat"]).to_csv(out_dir / "encoder_features_meta.csv", index=False)
    else:
        print("[WARN] --ckpt not provided. encoder TSNE will be skipped.")

    # =====================================================
    # Requested 4 figures
    # =====================================================
    event_only_groups = ["train_event", "val_event", "test_event"]
    event_noise_groups = [
        "train_event", "val_event", "test_event",
        "train_noise", "val_noise", "test_noise",
    ]

    # raw 1
    raw_event_df = make_subset(raw_df, event_only_groups, display_mode="group")
    if not raw_event_df.empty:
        feats = np.stack(raw_event_df["feat"].to_list(), axis=0)
        emb = run_tsne(feats, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
        raw_event_df["tsne_x"] = emb[:, 0]
        raw_event_df["tsne_y"] = emb[:, 1]
        plt.figure(figsize=(10, 8))
        for lab in raw_event_df["display_label"].unique():
            sub = raw_event_df[raw_event_df["display_label"] == lab]
            plt.scatter(sub["tsne_x"], sub["tsne_y"], s=18, alpha=0.7, label=lab)
        plt.title("RAW t-SNE: train/val/test events")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "tsne_raw_events_only.png", dpi=220)
        plt.close()
        raw_event_df.drop(columns=["feat"]).to_csv(out_dir / "tsne_raw_events_only.csv", index=False)

    # raw 2
    raw_all_df = make_subset(raw_df, event_noise_groups, display_mode="group")
    if not raw_all_df.empty:
        feats = np.stack(raw_all_df["feat"].to_list(), axis=0)
        emb = run_tsne(feats, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
        raw_all_df["tsne_x"] = emb[:, 0]
        raw_all_df["tsne_y"] = emb[:, 1]
        plt.figure(figsize=(11, 9))
        for lab in raw_all_df["display_label"].unique():
            sub = raw_all_df[raw_all_df["display_label"] == lab]
            plt.scatter(sub["tsne_x"], sub["tsne_y"], s=16, alpha=0.7, label=lab)
        plt.title("RAW t-SNE: train/val/test events + noises")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "tsne_raw_events_noises.png", dpi=220)
        plt.close()
        raw_all_df.drop(columns=["feat"]).to_csv(out_dir / "tsne_raw_events_noises.csv", index=False)

    # encoder 3
    if not enc_df.empty:
        enc_event_df = make_subset(enc_df, event_only_groups, display_mode="group")
        if not enc_event_df.empty:
            feats = np.stack(enc_event_df["feat"].to_list(), axis=0)
            emb = run_tsne(feats, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
            enc_event_df["tsne_x"] = emb[:, 0]
            enc_event_df["tsne_y"] = emb[:, 1]
            plt.figure(figsize=(10, 8))
            for lab in enc_event_df["display_label"].unique():
                sub = enc_event_df[enc_event_df["display_label"] == lab]
                plt.scatter(sub["tsne_x"], sub["tsne_y"], s=18, alpha=0.7, label=lab)
            plt.title("ENCODER t-SNE: train/val/test events")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.legend(fontsize=9)
            plt.tight_layout()
            plt.savefig(out_dir / "tsne_encoder_events_only.png", dpi=220)
            plt.close()
            enc_event_df.drop(columns=["feat"]).to_csv(out_dir / "tsne_encoder_events_only.csv", index=False)

        # encoder 4
        enc_all_df = make_subset(enc_df, event_noise_groups, display_mode="group")
        if not enc_all_df.empty:
            feats = np.stack(enc_all_df["feat"].to_list(), axis=0)
            emb = run_tsne(feats, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
            enc_all_df["tsne_x"] = emb[:, 0]
            enc_all_df["tsne_y"] = emb[:, 1]
            plt.figure(figsize=(11, 9))
            for lab in enc_all_df["display_label"].unique():
                sub = enc_all_df[enc_all_df["display_label"] == lab]
                plt.scatter(sub["tsne_x"], sub["tsne_y"], s=16, alpha=0.7, label=lab)
            plt.title("ENCODER t-SNE: train/val/test events + noises")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.legend(fontsize=9)
            plt.tight_layout()
            plt.savefig(out_dir / "tsne_encoder_events_noises.png", dpi=220)
            plt.close()
            enc_all_df.drop(columns=["feat"]).to_csv(out_dir / "tsne_encoder_events_noises.csv", index=False)

    # =====================================================
    # Optional domain-separated outputs
    # =====================================================
    if has_known_domain(raw_df):
        raw_domain_df = make_subset(raw_df, event_noise_groups, display_mode="group_domain")
        feats = np.stack(raw_domain_df["feat"].to_list(), axis=0)
        emb = run_tsne(feats, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
        raw_domain_df["tsne_x"] = emb[:, 0]
        raw_domain_df["tsne_y"] = emb[:, 1]
        plt.figure(figsize=(12, 10))
        for lab in raw_domain_df["display_label"].unique():
            sub = raw_domain_df[raw_domain_df["display_label"] == lab]
            plt.scatter(sub["tsne_x"], sub["tsne_y"], s=15, alpha=0.7, label=lab)
        plt.title("RAW t-SNE with domain labels")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / "tsne_raw_events_noises_domain.png", dpi=220)
        plt.close()
        raw_domain_df.drop(columns=["feat"]).to_csv(out_dir / "tsne_raw_events_noises_domain.csv", index=False)

    if not enc_df.empty and has_known_domain(enc_df):
        enc_domain_df = make_subset(enc_df, event_noise_groups, display_mode="group_domain")
        feats = np.stack(enc_domain_df["feat"].to_list(), axis=0)
        emb = run_tsne(feats, seed=args.seed, perplexity=args.tsne_perplexity, pca_dim=args.pca_dim)
        enc_domain_df["tsne_x"] = emb[:, 0]
        enc_domain_df["tsne_y"] = emb[:, 1]
        plt.figure(figsize=(12, 10))
        for lab in enc_domain_df["display_label"].unique():
            sub = enc_domain_df[enc_domain_df["display_label"] == lab]
            plt.scatter(sub["tsne_x"], sub["tsne_y"], s=15, alpha=0.7, label=lab)
        plt.title("ENCODER t-SNE with domain labels")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / "tsne_encoder_events_noises_domain.png", dpi=220)
        plt.close()
        enc_domain_df.drop(columns=["feat"]).to_csv(out_dir / "tsne_encoder_events_noises_domain.csv", index=False)

    summary = {
        "out_dir": str(out_dir),
        "raw_total": int(len(raw_df)),
        "encoder_total": int(len(enc_df)),
        "raw_domain_known": bool(has_known_domain(raw_df)) if not raw_df.empty else False,
        "encoder_domain_known": bool(has_known_domain(enc_df)) if not enc_df.empty else False,
        "requested_outputs": [
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