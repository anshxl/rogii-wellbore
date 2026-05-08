"""Phase 3 NN models — dummy sanity model + (later) full encoder/decoder."""

import torch
import torch.nn as nn

from src.nn.data import WELL_FEATURE_NAMES

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


def masked_mse(pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error averaged only over rows where target_mask == 1."""
    diff = (pred - target) * target_mask
    sse = (diff * diff).sum()
    n = target_mask.sum().clamp_min(1.0)
    return sse / n
