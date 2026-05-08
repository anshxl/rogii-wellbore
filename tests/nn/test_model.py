import torch
import pytest

from src.nn.model import DummyMLP, masked_mse


def test_dummy_mlp_forward_shape():
    model = DummyMLP(n_well_features=12, hidden=32)
    well_inputs = torch.randn(2, 100, 12)
    well_mask = torch.ones(2, 100)
    out = model(well_inputs=well_inputs, well_mask=well_mask)
    assert out.shape == (2, 100)


def test_dummy_mlp_residual_to_tvt_input():
    """Untrained DummyMLP should output ≈ TVT_input_filled (residual = 0 init)."""
    torch.manual_seed(0)
    model = DummyMLP(n_well_features=12, hidden=32)
    # Build inputs where tvt_input_filled column is a ramp
    well_inputs = torch.zeros(1, 50, 12)
    tvt_idx = 7  # WELL_FEATURE_NAMES.index("tvt_input_filled") = 7
    ramp = torch.linspace(1000.0, 1050.0, 50)
    well_inputs[0, :, tvt_idx] = ramp
    well_mask = torch.ones(1, 50)
    out = model(well_inputs=well_inputs, well_mask=well_mask)
    # The residual head's bias should be near zero, so out ≈ ramp
    assert torch.allclose(out[0], ramp, atol=5.0)


def test_masked_mse_only_counts_target_mask():
    """Loss must average only over rows where target_mask = 1."""
    pred = torch.tensor([[10.0, 20.0, 30.0]])
    target = torch.tensor([[12.0, 99.0, 33.0]])
    target_mask = torch.tensor([[1.0, 0.0, 1.0]])
    loss = masked_mse(pred, target, target_mask)
    # MSE on rows 0 and 2: ((12-10)^2 + (33-30)^2) / 2 = (4 + 9) / 2 = 6.5
    assert loss.item() == pytest.approx(6.5, rel=1e-5)
