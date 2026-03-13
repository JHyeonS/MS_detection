import torch
import torch.nn as nn
from src.models.cnn_encoder import cnn_encoder


class MSDNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.cnn_encoder = cnn_encoder(cfg)

        self.latent_dim = cfg.model.encoder.latent_dim
        self.fc = nn.Linear(self.latent_dim, 1)

        if cfg.train.freeze_encoder:
            for p in self.cnn_encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        x_vec = self.cnn_encoder(x)

        if x_vec.ndim == 4:
            x_vec = x_vec.mean(dim=(2, 3))

        y = self.fc(x_vec)
        return x_vec, y