# models/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import WINDOW_SIZE

class CustomTransformerLayer(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, save_attn=True):
        attn_output, attn_weights = self.self_attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ff(x))
        return x, attn_weights if save_attn else None

class HMTransformerTokenEncoder(nn.Module):
    def __init__(self, input_dim=6, embed_dim=64, num_layers=4, nhead=4, z_dim=32):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, WINDOW_SIZE, embed_dim))  # positional encoding
        self.layers = nn.ModuleList([
            CustomTransformerLayer(embed_dim, nhead) for _ in range(num_layers)
        ])
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, z_dim)
        )
        self.attn_weights = []

    def forward(self, x, save_attn=True):
        """
        x: [B, W, C] - batch of histone modification windows
        return: [B, W, z_dim] - projected representation
        """
        B, W, C = x.size()
        h = self.embed(x) + self.pos  # [B, W, D]
        self.attn_weights = []

        for layer in self.layers:
            h, attn = layer(h, save_attn=save_attn)
            if save_attn and attn is not None:
                self.attn_weights.append(attn.detach().cpu())

        h_flat = h.view(B * W, -1)
        z_flat = self.projector(h_flat)
        return z_flat.view(B, W, -1)