import numpy as np
import pandas as pd
import pytest

from src.nn.data import compute_well_stats, build_well_inputs, WELL_FEATURE_NAMES

LEAK_COLS = ["TVT", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def test_compute_well_stats_basic():
    df = pd.DataFrame({
        "MD": np.arange(0, 100, 1.0),
        "GR": np.array([10.0, 20.0] * 50),
        "Z":  np.linspace(1000, 1100, 100),
        "X":  np.linspace(0, 50, 100),
        "Y":  np.linspace(0, 50, 100),
    })
    stats = compute_well_stats(df)
    assert stats["gr_mean"] == pytest.approx(15.0)
    assert stats["gr_std"] == pytest.approx(5.0, rel=1e-3)
    assert stats["md_step_median"] == pytest.approx(1.0)
    assert "z_mean" in stats and "z_std" in stats
    assert "x_mean" in stats and "y_mean" in stats


def _make_synthetic_well(n_rows=200, prefix_len=50):
    md = np.arange(0, n_rows, 1.0)
    gr = 50.0 + 5.0 * np.sin(md * 0.1)
    tvt_input = np.where(np.arange(n_rows) < prefix_len, 1000.0 + np.arange(n_rows) * 0.5, np.nan)
    tvt = 1000.0 + np.arange(n_rows) * 0.5
    df = pd.DataFrame({
        "MD": md, "GR": gr,
        "Z": 1500 + 0.1 * md, "X": 100 + md, "Y": 200 - md,
        "TVT_input": tvt_input, "TVT": tvt,
        # add train-only leakage columns to verify they don't leak
        "ANCC": 0.0, "ASTNU": 0.0, "ASTNL": 0.0,
        "EGFDU": 0.0, "EGFDL": 0.0, "BUDA": 0.0,
    })
    return df


def test_build_well_inputs_shape_and_features():
    df = _make_synthetic_well(n_rows=200, prefix_len=50)
    stats = compute_well_stats(df)
    inputs = build_well_inputs(df, stats)
    assert inputs.shape == (200, len(WELL_FEATURE_NAMES))
    assert len(WELL_FEATURE_NAMES) == 12
    assert not np.isnan(inputs).any()
    # is_known_mask should be 1 on prefix rows, 0 on hidden rows
    is_known_idx = WELL_FEATURE_NAMES.index("is_known_mask")
    assert (inputs[:50, is_known_idx] == 1.0).all()
    assert (inputs[50:, is_known_idx] == 0.0).all()


def test_build_well_inputs_no_leakage():
    """build_well_inputs must not surface train-only columns."""
    # leakage columns should not appear in the feature names list
    for col in LEAK_COLS:
        assert col not in WELL_FEATURE_NAMES, f"{col} leaked into feature list"
