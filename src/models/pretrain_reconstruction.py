#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

    Important:
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
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type in ["none", "identity", None]:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


def _make_final_act(act_type: str) -> nn.Module:
    act_type = (act_type or "identity").lower()

    if act_type in ["identity", "none"]:
        return nn.Identity()
    elif act_type == "tanh":
        return nn.Tanh()
    elif act_type == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported final_act: {act_type}")


class DeconvBlock(nn.Module):
    """
    ConvTranspose2d -> Norm -> Activation -> Dropout2D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(4, 4),
        stride=(2, 2),
        padding=(1, 1),
        output_padding=(0, 0),
        norm="bn",
        act="relu",
        dropout=0.0,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            _make_norm(norm, out_channels),
            _make_act(act),
            nn.Dropout2d(dropout) if float(dropout) > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CAEDecoder(nn.Module):
    """
    Mirror decoder for the auto-generated CNNEncoder.

    Encoder rule recap:
        layer 0 : kernel=(5,5), stride=(1,2), padding=(2,2)
        layer 1+: kernel=(3,3), stride=(2,2), padding=(1,1)
        final   : proj 1x1 -> latent_dim

    Decoder rule:
        reverse channel progression
        reverse stride schedule
        final reconstruction head -> out_channels (= encoder in_channels)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_cfg = _get_attr(cfg, "model", None)
        enc_cfg = _get_attr(model_cfg, "encoder", None)
        cae_cfg = _get_attr(model_cfg, "cae", None)

        self.in_channels = int(_get_attr(enc_cfg, "in_channels", 1))
        self.num_layers = int(_get_attr(enc_cfg, "num_layers", 4))
        self.base_channels = int(_get_attr(enc_cfg, "base_channels", 32))
        self.latent_dim = int(_get_attr(enc_cfg, "latent_dim", 128))
        self.norm = _get_attr(enc_cfg, "norm", "bn")
        self.act = _get_attr(enc_cfg, "act", "relu")
        self.dropout = _resolve_encoder_dropout(cfg, default=0.0)

        self.mirror_decoder = bool(_get_attr(cae_cfg, "mirror_decoder", True))
        self.final_act = _get_attr(cae_cfg, "final_act", "identity")

        if not self.mirror_decoder:
            raise NotImplementedError(
                "Currently only mirror_decoder=True is supported."
            )

        enc_channels = [self.base_channels * (2 ** i) for i in range(self.num_layers)]
        rev_channels = list(reversed(enc_channels))

        layers = []
        prev_ch = self.latent_dim

        for i, out_ch in enumerate(rev_channels):
            if i < self.num_layers - 1:
                kernel_size = (4, 4)
                stride = (2, 2)
                padding = (1, 1)
                output_padding = (0, 0)
            else:
                kernel_size = (3, 4)
                stride = (1, 2)
                padding = (1, 1)
                output_padding = (0, 0)

            layers.append(
                DeconvBlock(
                    in_channels=prev_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    norm=self.norm,
                    act=self.act,
                    dropout=self.dropout,
                )
            )
            prev_ch = out_ch

        self.decoder = nn.Sequential(*layers)

        self.recon_head = nn.Sequential(
            nn.Conv2d(
                in_channels=prev_ch,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            _make_final_act(self.final_act),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, output_size=None) -> torch.Tensor:
        """
        Args:
            feat: (B, latent_dim, C', T')
            output_size: original input shape, e.g. x.shape = (B,1,C,T)

        Returns:
            x_hat: reconstructed tensor, approximately same shape as input
        """
        x_hat = self.decoder(feat)
        x_hat = self.recon_head(x_hat)

        if output_size is not None:
            _, _, target_h, target_w = output_size
            x_hat = x_hat[..., :target_h, :target_w]

        return x_hat


class CAE(nn.Module):
    """
    Convolutional Autoencoder for DAS reconstruction.

    Input:
        x: (B, 1, C, T)

    Output:
        x_hat: (B, 1, C, T)
        feat : (B, latent_dim, C', T')
        z    : (B, latent_dim)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = cnn_encoder(cfg)
        self.decoder = CAEDecoder(cfg)

    def forward(self, x: torch.Tensor):
        feat = self.encoder.forward_features(x)   # (B, latent_dim, C', T')
        z = self.encoder.forward_embedding(x)     # (B, latent_dim)
        x_hat = self.decoder(feat, output_size=x.shape)
        return x_hat, feat, z


def build_recon_loss(cfg):
    """
    Build reconstruction criterion from cfg.pretrain.recon_loss
    """
    pre_cfg = _get_attr(cfg, "pretrain", None)
    loss_name = str(_get_attr(pre_cfg, "recon_loss", "l1")).lower()

    if loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported recon_loss: {loss_name}")


if __name__ == "__main__":
    class DummyEncoderCfg:
        in_channels = 1
        num_layers = 4
        base_channels = 32
        latent_dim = 128
        norm = "bn"
        act = "relu"
        dropout = 0.0

    class DummyCAECfg:
        mirror_decoder = True
        final_act = "identity"

    class DummyModelCfg:
        encoder = DummyEncoderCfg()
        cae = DummyCAECfg()

    class DummyPretrainCfg:
        mode = "reconstruction"
        recon_loss = "l1"
        encoder_dropout = 0.0

    class DummyCfg:
        model = DummyModelCfg()
        pretrain = DummyPretrainCfg()

    cfg = DummyCfg()
    model = CAE(cfg)
    criterion = build_recon_loss(cfg)

    x = torch.randn(2, 1, 64, 512)
    x_hat, feat, z = model(x)
    loss = criterion(x_hat, x)

    print("input shape  :", x.shape)
    print("recon shape  :", x_hat.shape)
    print("feat shape   :", feat.shape)
    print("embed shape  :", z.shape)
    print("loss         :", float(loss))