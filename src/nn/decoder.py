"""Phase 3 NN decoder — cross-attention into typewell + per-row TVT residual head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + feedforward, using SDPA for memory efficiency.

    Uses `torch.nn.functional.scaled_dot_product_attention` instead of
    `nn.MultiheadAttention` so the full [B, H, L_q, L_kv] attention matrix
    is never materialized. On T4 the mem-efficient backend kicks in
    automatically.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

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
        B, Lq, D = q.shape
        Lkv = kv.shape[1]
        H, Dh = self.n_heads, self.d_head

        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)

        Q = self.q_proj(q_n).view(B, Lq, H, Dh).transpose(1, 2)   # [B, H, Lq, Dh]
        K = self.k_proj(kv_n).view(B, Lkv, H, Dh).transpose(1, 2) # [B, H, Lkv, Dh]
        V = self.v_proj(kv_n).view(B, Lkv, H, Dh).transpose(1, 2) # [B, H, Lkv, Dh]

        # SDPA bool attn_mask: True = attend, False = mask out (opposite of MHA).
        # Our convention: kv_mask 1 = real, 0 = padding.
        attend_mask = (kv_mask == 1)                              # [B, Lkv]

        # Guard rows that are entirely padding — softmax over all-False would
        # produce NaN. Force position 0 valid for those rows; the model's
        # well_mask zeroes out the corresponding output anyway.
        all_pad = ~attend_mask.any(dim=1, keepdim=True)
        if all_pad.any():
            attend_mask = attend_mask.clone()
            attend_mask[all_pad.squeeze(1), 0] = True

        # Broadcast to [B, 1, 1, Lkv] so it covers all heads and queries.
        attn_mask = attend_mask[:, None, None, :]

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                                         # [B, H, Lq, Dh]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, D)
        attn_out = self.o_proj(attn_out)

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
