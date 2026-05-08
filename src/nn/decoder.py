"""Phase 3 NN decoder — cross-attention into typewell + per-row TVT residual head."""

import torch
import torch.nn as nn


class _CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor,
    ) -> torch.Tensor:
        # kv_mask: [B, L_kv]; True positions are *padding* in PyTorch's MHA, so invert.
        key_padding_mask = (kv_mask == 0)
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        # If a row has all-padding K/V, key_padding_mask is all True for that row.
        # MHA returns NaN — guard by giving at least one valid key.
        all_pad = key_padding_mask.all(dim=1, keepdim=True)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad.squeeze(1), 0] = False
        attn_out, _ = self.attn(q_n, kv_n, kv_n, key_padding_mask=key_padding_mask)
        h = q + self.drop(attn_out)
        h = h + self.drop(self.ff(self.norm_ff(h)))
        return h


class CrossAttentionDecoder(nn.Module):
    """Two cross-attention blocks + per-row 2-layer MLP head.

    Output is a *residual* added to TVT_input_filled by the parent Model.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Initialize the final layer's bias/weights to ~0 so untrained output ≈ TVT anchor.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        h_well: torch.Tensor,
        h_tw:   torch.Tensor,
        well_mask: torch.Tensor,
        tw_mask:   torch.Tensor,
    ) -> torch.Tensor:
        h = h_well
        for block in self.blocks:
            h = block(h, h_tw, tw_mask)
        out = self.head(h).squeeze(-1)            # [B, L]
        out = out * well_mask                      # zero out padding rows
        return out
