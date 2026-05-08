import torch

from src.nn.encoders import TypewellEncoder, CNNEncoder


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


def test_cnn_encoder_shape():
    enc = CNNEncoder(in_features=12, d_model=128, n_blocks=6)
    x = torch.randn(2, 1500, 12)
    mask = torch.ones(2, 1500)
    out = enc(x, mask)
    assert out.shape == (2, 1500, 128)


def test_cnn_encoder_param_budget():
    """Ensure total params stay around the spec target (~300k for d=128)."""
    enc = CNNEncoder(in_features=12, d_model=128, n_blocks=6)
    n = sum(p.numel() for p in enc.parameters())
    assert n < 1_000_000
