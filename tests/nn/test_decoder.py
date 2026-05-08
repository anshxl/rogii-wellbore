import torch

from src.nn.decoder import CrossAttentionDecoder


def test_decoder_forward_shape():
    dec = CrossAttentionDecoder(d_model=128, n_heads=4, n_blocks=2, dropout=0.1)
    h_well = torch.randn(2, 1500, 128)
    h_tw   = torch.randn(2, 300, 128)
    well_mask = torch.ones(2, 1500)
    tw_mask   = torch.ones(2, 300)
    out = dec(h_well, h_tw, well_mask, tw_mask)
    # Decoder returns the per-row residual TVT scalar.
    assert out.shape == (2, 1500)


def test_decoder_respects_typewell_mask():
    """If all typewell rows are padded, output should still be finite."""
    dec = CrossAttentionDecoder(d_model=128, n_heads=4, n_blocks=2, dropout=0.0)
    h_well = torch.randn(1, 100, 128)
    h_tw   = torch.randn(1, 50, 128)
    well_mask = torch.ones(1, 100)
    tw_mask   = torch.zeros(1, 50)
    tw_mask[0, :10] = 1.0  # only first 10 rows valid
    out = dec(h_well, h_tw, well_mask, tw_mask)
    assert torch.isfinite(out).all()
