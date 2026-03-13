#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/models/pretrain_contrastive.py

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cnn_encoder import cnn_encoder


def _get_attr(obj, name, default=None):
    """
    Safe getter for:
        - OmegaConf-like object
        - argparse/namespace-like object
        - dict

    If the attribute/key exists but its value is None,
    this function returns default.
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        value = obj.get(name, default)
        return default if value is None else value

    if hasattr(obj, name):
        value = getattr(obj, name)
        return default if value is None else value

    return default


class ProjectionHead(nn.Module):
    """
    SimCLR-style MLP projection head.

    Input:
        z: (B, latent_dim)

    Output:
        proj: (B, proj_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        use_bn: bool = False,
    ):
        super().__init__()

        if use_bn:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim, bias=True),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastivePretrainModel(nn.Module):
    """
    Contrastive pretraining model for DAS microseismic.

    Input:
        x: (B, 1, C, T)

    Output:
        dict:
            feat: (B, latent_dim, C', T')
            z   : (B, latent_dim)
            proj: (B, proj_dim)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_cfg = _get_attr(cfg, "model", None)
        enc_cfg = _get_attr(model_cfg, "encoder", None)
        pre_cfg = _get_attr(cfg, "pretrain", None)

        latent_dim = int(_get_attr(enc_cfg, "latent_dim", 128))
        proj_hidden_dim = int(_get_attr(pre_cfg, "proj_hidden_dim", 256))
        proj_dim = int(_get_attr(pre_cfg, "proj_dim", 128))
        proj_use_bn = bool(_get_attr(pre_cfg, "proj_use_bn", False))

        self.encoder = cnn_encoder(cfg)
        self.projection_head = ProjectionHead(
            in_dim=latent_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_dim,
            use_bn=proj_use_bn,
        )

    def forward(self, x: torch.Tensor):
        feat = self.encoder.forward_features(x)   # (B, latent_dim, C', T')
        z = self.encoder.forward_embedding(x)     # (B, latent_dim)
        proj = self.projection_head(z)            # (B, proj_dim)

        return {
            "feat": feat,
            "z": z,
            "proj": proj,
        }


if __name__ == "__main__":
    class DummyEncoderCfg:
        in_channels = 1
        num_layers = 4
        base_channels = 32
        latent_dim = 128
        norm = "bn"
        act = "relu"
        dropout = 0.0

    class DummyModelCfg:
        encoder = DummyEncoderCfg()

    class DummyPretrainCfg:
        mode = "contrast"
        proj_hidden_dim = 256
        proj_dim = 128
        proj_use_bn = False

    class DummyCfg:
        model = DummyModelCfg()
        pretrain = DummyPretrainCfg()

    cfg = DummyCfg()
    model = ContrastivePretrainModel(cfg)

    x = torch.randn(2, 1, 64, 512)
    out = model(x)

    print("feat shape :", out["feat"].shape)
    print("z shape    :", out["z"].shape)
    print("proj shape :", out["proj"].shape)