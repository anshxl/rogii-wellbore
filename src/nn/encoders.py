"""Phase 3 NN encoders — CNN-TCN, Transformer, typewell."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel,
                              padding=pad, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        h = self.conv(x)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.drop(h)
        return x + h


class TypewellEncoder(nn.Module):
    """Small dilated CNN over typewell rows. Outputs the K/V bank."""

    def __init__(self, in_features: int = 8, d_model: int = 128, n_blocks: int = 3):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.blocks = nn.ModuleList([
            _DilatedConvBlock(channels=d_model, kernel=3, dilation=2 ** i, dropout=0.1)
            for i in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F], mask: [B, L]
        h = self.proj(x)             # [B, L, D]
        h = h * mask.unsqueeze(-1)
        h = h.transpose(1, 2)        # [B, D, L]
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)        # [B, L, D]
        return h
