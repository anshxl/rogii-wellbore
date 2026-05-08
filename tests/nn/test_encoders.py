import torch

from src.nn.encoders import TypewellEncoder


def test_typewell_encoder_shape():
    enc = TypewellEncoder(in_features=8, d_model=128)
    x = torch.randn(2, 300, 8)
    mask = torch.ones(2, 300)
    out = enc(x, mask)
    assert out.shape == (2, 300, 128)


def test_typewell_encoder_handles_padding():
    """Padded rows should still produce finite outputs (mask used downstream)."""
    enc = TypewellEncoder(in_features=8, d_model=128)
    x = torch.randn(2, 300, 8)
    mask = torch.zeros(2, 300)
    mask[:, :150] = 1.0
    out = enc(x, mask)
    assert torch.isfinite(out).all()
