#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_attr(obj, key, default=None):
    """
    Safe getter for:
        - dict-like objects
        - attribute-like objects (AttrDict / OmegaConf / namespace)

    If the key/attribute exists but value is None, return default.
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        value = obj.get(key, default)
        return default if value is None else value

    if hasattr(obj, key):
        value = getattr(obj, key)
        return default if value is None else value

    return default


def _resolve_encoder_dropout(cfg, default=0.0) -> float:
    """
    Priority:
        1) cfg.pretrain.encoder_dropout
        2) cfg.train.encoder_dropout
        3) cfg.model.encoder.dropout
        4) default
    """
    pretrain_cfg = _get_attr(cfg, "pretrain", None)
    train_cfg = _get_attr(cfg, "train", None)
    model_cfg = _get_attr(cfg, "model", None)
    enc_cfg = _get_attr(model_cfg, "encoder", None)

    value = _get_attr(pretrain_cfg, "encoder_dropout", None)
    if value is not None:
        return float(value)

    value = _get_attr(train_cfg, "encoder_dropout", None)
    if value is not None:
        return float(value)

    value = _get_attr(enc_cfg, "dropout", None)
    if value is not None:
        return float(value)

    return float(default)


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    norm_type = (norm_type or "bn").lower()

    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm_type == "gn":
        num_groups = 8 if num_channels >= 8 else 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif norm_type in ["none", "identity", None]:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


def _make_act(act_type: str) -> nn.Module:
    act_type = (act_type or "relu").lower()

    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type in ["none", "identity", None]:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


class ConvBlock(nn.Module):
    """
    Conv2D -> Norm -> Activation -> Dropout2D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        norm="bn",
        act="relu",
        dropout=0.0,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            _make_norm(norm, out_channels),
            _make_act(act),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GeneralizedMeanPooling(nn.Module):
    """
    Sign-preserving generalized mean pooling.

    Standard GeM assumes non-negative features. Here the final 1x1 projection can
    produce signed activations, so the pooled signed moment is converted back with
    a sign-preserving root to avoid NaNs from fractional powers of negatives.
    """

    def __init__(
        self,
        num_channels=None,
        p=3.0,
        eps=1e-6,
        p_min=1.0,
        p_max=10.0,
        output_size=(1, 1),
    ):
        super().__init__()
        if num_channels is not None:
            self.p = nn.Parameter(float(p) * torch.ones(int(num_channels)))
        else:
            self.p = nn.Parameter(float(p) * torch.ones(1))
        self.eps = float(eps)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(self.p_min, self.p_max)
        if p.dim() == 1:
            p = p.view(1, -1, 1, 1)

        x_abs = x.abs().clamp(min=self.eps, max=1e4)
        signed_power = x.sign() * x_abs.pow(p)
        pooled = F.adaptive_avg_pool2d(signed_power, output_size=self.output_size)
        pooled_abs = pooled.abs().clamp(min=self.eps, max=1e12)
        return pooled.sign() * pooled_abs.pow(1.0 / p)


def _make_pooling(pool_type: str, latent_dim: int, enc_cfg) -> nn.Module:
    pool_type = str(pool_type or "avg").lower()
    if pool_type in {"avg", "average", "gap", "adaptive_avg"}:
        return nn.AdaptiveAvgPool2d((1, 1))
    if pool_type in {"gem", "signed_gem", "generalized_mean"}:
        channelwise = bool(_get_attr(enc_cfg, "pooling_channelwise", True))
        return GeneralizedMeanPooling(
            num_channels=latent_dim if channelwise else None,
            p=float(_get_attr(enc_cfg, "pooling_p", 3.0)),
            eps=float(_get_attr(enc_cfg, "pooling_eps", 1e-6)),
            p_min=float(_get_attr(enc_cfg, "pooling_p_min", 1.0)),
            p_max=float(_get_attr(enc_cfg, "pooling_p_max", 10.0)),
            output_size=(1, 1),
        )
    raise ValueError(f"Unsupported pooling type: {pool_type}")


class CNNEncoder(nn.Module):
    """
    Auto-generated CNN encoder for DAS microseismic input.

    Expected input:
        x: (B, 1, C, T)

    Main behavior:
        - forward_features(x): feature map for CAE
        - forward(x): pooled latent vector for classifier / contrastive

    Recommended config:

    base.yaml
    ----------
    model:
      encoder:
        in_channels: 1
        num_layers: 4
        base_channels: 32
        latent_dim: 128
        norm: "bn"
        act: "relu"
        dropout: 0.0    # fallback only
        pooling: "avg"  # "avg" or "gem"
        pooling_p: 3.0

    pretrain.yaml
    -------------
    pretrain:
      encoder_dropout: 0.0

    train.yaml
    ----------
    train:
      encoder_dropout: 0.1
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_cfg = _get_attr(cfg, "model", None)
        enc_cfg = _get_attr(model_cfg, "encoder", None)

        self.in_channels = int(_get_attr(enc_cfg, "in_channels", 1))
        self.num_layers = int(_get_attr(enc_cfg, "num_layers", 4))
        self.base_channels = int(_get_attr(enc_cfg, "base_channels", 32))
        self.latent_dim = int(_get_attr(enc_cfg, "latent_dim", 128))
        self.norm = _get_attr(enc_cfg, "norm", "bn")
        self.act = _get_attr(enc_cfg, "act", "relu")
        self.pooling = _get_attr(enc_cfg, "pooling", "avg")
        self.dropout = _resolve_encoder_dropout(cfg, default=0.0)

        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, but got {self.num_layers}")

        layers = []
        prev_ch = self.in_channels

        for i in range(self.num_layers):
            out_ch = self.base_channels * (2 ** i)

            # first layer: reduce time first
            if i == 0:
                kernel_size = (5, 5)
                stride = (1, 2)
                padding = (2, 2)
            else:
                kernel_size = (3, 3)
                stride = (2, 2)
                padding = (1, 1)

            layers.append(
                ConvBlock(
                    in_channels=prev_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    norm=self.norm,
                    act=self.act,
                    dropout=self.dropout,
                )
            )
            prev_ch = out_ch

        self.encoder = nn.Sequential(*layers)

        # final projection to latent_dim
        self.proj = nn.Conv2d(
            in_channels=prev_ch,
            out_channels=self.latent_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.global_pool = _make_pooling(self.pooling, self.latent_dim, enc_cfg)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns feature map for CAE / anomaly localization.

        Input:
            x: (B, 1, C, T)

        Output:
            feat: (B, latent_dim, C', T')
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, 1, C, T), but got {tuple(x.shape)}"
            )

        feat = self.encoder(x)
        feat = self.proj(feat)
        return feat

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns pooled embedding vector.

        Output:
            z: (B, latent_dim)
        """
        feat = self.forward_features(x)
        z = self.global_pool(feat)   # (B, latent_dim, 1, 1)
        z = torch.flatten(z, 1)      # (B, latent_dim)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default forward returns pooled latent vector.
        """
        return self.forward_embedding(x)


def cnn_encoder(cfg):
    """
    Factory function for compatibility:
        from src.models.cnn_encoder import cnn_encoder
    """
    return CNNEncoder(cfg)


if __name__ == "__main__":
    class DummyEncoderCfg:
        in_channels = 1
        num_layers = 4
        base_channels = 32
        latent_dim = 128
        norm = "bn"
        act = "relu"
        dropout = 0.0
        pooling = "avg"

    class DummyModelCfg:
        encoder = DummyEncoderCfg()

    class DummyPretrainCfg:
        encoder_dropout = 0.0

    class DummyCfg:
        model = DummyModelCfg()
        pretrain = DummyPretrainCfg()

    cfg = DummyCfg()
    model = CNNEncoder(cfg)

    x = torch.randn(2, 1, 64, 512)

    feat = model.forward_features(x)
    z = model(x)

    print("Input shape         :", x.shape)
    print("Feature map shape   :", feat.shape)
    print("Embedding shape     :", z.shape)
    print("Resolved dropout    :", model.dropout)
