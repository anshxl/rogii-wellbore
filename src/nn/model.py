"""Phase 3 NN models — dummy sanity model + (later) full encoder/decoder."""

import torch
import torch.nn as nn

from src.nn.data import WELL_FEATURE_NAMES
from src.nn.encoders import CNNEncoder, TypewellEncoder
from src.nn.decoder import CrossAttentionDecoder

TVT_INPUT_IDX = WELL_FEATURE_NAMES.index("tvt_input_filled")


class DummyMLP(nn.Module):
    """Per-row 2-layer MLP. Predicts a residual added to TVT_input_filled.

    Sanity-check baseline for M1. Doesn't see the typewell — this is intentional;
    we just want a model that's not broken to validate the pipeline floor.
    """

    def __init__(self, n_well_features: int = 12, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(n_well_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        # Initialize the head to ~0 so untrained output ≈ tvt_input_filled.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, well_inputs: torch.Tensor, well_mask: torch.Tensor, **_) -> torch.Tensor:
        # well_inputs: [B, L, F]; well_mask: [B, L]
        h = torch.relu(self.fc1(well_inputs))
        h = torch.relu(self.fc2(h))
        residual = self.head(h).squeeze(-1)              # [B, L]
        tvt_anchor = well_inputs[..., TVT_INPUT_IDX]     # [B, L]
        return tvt_anchor + residual


class Model(nn.Module):
    """Full Phase 3 model: well encoder + typewell encoder + cross-attention decoder.

    Output is `TVT_input_filled + decoder_residual`. Untrained, the residual
    starts at ~0 (head is zero-initialized) so the model emits the anchor.
    """

    def __init__(
        self,
        encoder_kind: str = "cnn",
        n_well_features: int = 12,
        n_typewell_features: int = 8,
        d_model: int = 128,
        n_well_blocks: int = 6,
        n_tw_blocks: int = 3,
        n_decoder_blocks: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_kind = encoder_kind
        if encoder_kind == "cnn":
            self.well_encoder = CNNEncoder(
                in_features=n_well_features, d_model=d_model, n_blocks=n_well_blocks,
            )
        else:
            raise ValueError(f"Unknown encoder kind: {encoder_kind!r}")
        self.tw_encoder = TypewellEncoder(
            in_features=n_typewell_features, d_model=d_model, n_blocks=n_tw_blocks,
        )
        self.decoder = CrossAttentionDecoder(
            d_model=d_model, n_heads=n_heads, n_blocks=n_decoder_blocks, dropout=dropout,
        )

    def forward(
        self,
        well_inputs: torch.Tensor,
        well_mask: torch.Tensor,
        typewell_inputs: torch.Tensor,
        typewell_mask: torch.Tensor,
    ) -> torch.Tensor:
        h_well = self.well_encoder(well_inputs, well_mask)
        h_tw   = self.tw_encoder(typewell_inputs, typewell_mask)
        residual = self.decoder(h_well, h_tw, well_mask, typewell_mask)
        anchor = well_inputs[..., TVT_INPUT_IDX]
        return anchor + residual


def masked_mse(pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error averaged only over rows where target_mask == 1."""
    diff = (pred - target) * target_mask
    sse = (diff * diff).sum()
    n = target_mask.sum().clamp_min(1.0)
    return sse / n
