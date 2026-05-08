import numpy as np
import pandas as pd
import pytest

from src.nn.data import compute_well_stats, build_well_inputs, WELL_FEATURE_NAMES, build_typewell_inputs, TYPEWELL_FEATURE_NAMES, GEOLOGY_NAMES, apply_prefix_augmentation

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


def _make_synthetic_typewell(n=300):
    return pd.DataFrame({
        "TVT": np.linspace(900, 1300, n),
        "GR":  60 + 10 * np.sin(np.linspace(0, 6.28, n)),
        "Geology": (["EGFDU"] * (n // 3) + ["EGFDL"] * (n // 3) + ["ANCC"] * (n - 2 * (n // 3))),
    })


def test_build_typewell_inputs_shape():
    well_df = _make_synthetic_well(200, 50)
    well_stats = compute_well_stats(well_df)
    tw = _make_synthetic_typewell(300)
    out = build_typewell_inputs(tw, well_stats)
    assert out.shape == (300, len(TYPEWELL_FEATURE_NAMES))
    assert len(TYPEWELL_FEATURE_NAMES) == 8
    assert not np.isnan(out).any()


def test_build_typewell_geology_onehot():
    well_df = _make_synthetic_well(200, 50)
    well_stats = compute_well_stats(well_df)
    tw = _make_synthetic_typewell(300)
    out = build_typewell_inputs(tw, well_stats)
    geo_idx_start = TYPEWELL_FEATURE_NAMES.index(f"geo_{GEOLOGY_NAMES[0]}")
    geo_idx_end = geo_idx_start + len(GEOLOGY_NAMES)
    geo_block = out[:, geo_idx_start:geo_idx_end]
    # each row has exactly one 1 across the 6 geology columns
    assert (geo_block.sum(axis=1) == 1.0).all()


def test_augmentation_preserves_total_length():
    df = _make_synthetic_well(n_rows=200, prefix_len=50)
    stats = compute_well_stats(df)
    inputs = build_well_inputs(df, stats)
    rng = np.random.default_rng(42)

    aug_inputs, target, target_mask = apply_prefix_augmentation(
        well_df=df, well_inputs=inputs, well_stats=stats, p=0.5, rng=rng,
    )
    assert aug_inputs.shape == inputs.shape
    assert target.shape == (200,)
    assert target_mask.shape == (200,)


def test_augmentation_target_only_on_hidden():
    """Loss must contribute only on rows where is_known_aug = 0."""
    df = _make_synthetic_well(n_rows=200, prefix_len=50)
    stats = compute_well_stats(df)
    inputs = build_well_inputs(df, stats)
    rng = np.random.default_rng(42)

    p = 0.30
    aug_inputs, target, target_mask = apply_prefix_augmentation(
        well_df=df, well_inputs=inputs, well_stats=stats, p=p, rng=rng,
    )
    is_known_idx = WELL_FEATURE_NAMES.index("is_known_mask")
    is_known = aug_inputs[:, is_known_idx].astype(bool)
    # target_mask is 1 where target counts; this should be exactly the hidden rows
    assert (target_mask == (~is_known).astype(np.float32)).all()


def test_augmentation_input_anchor_correctness():
    """At hidden rows, TVT_input_filled must equal last_known TVT under p."""
    df = _make_synthetic_well(n_rows=200, prefix_len=50)
    stats = compute_well_stats(df)
    inputs = build_well_inputs(df, stats)
    rng = np.random.default_rng(42)

    p = 0.30  # prefix to MD-row 60
    aug_inputs, target, target_mask = apply_prefix_augmentation(
        well_df=df, well_inputs=inputs, well_stats=stats, p=p, rng=rng,
    )
    is_known_idx = WELL_FEATURE_NAMES.index("is_known_mask")
    tvt_idx = WELL_FEATURE_NAMES.index("tvt_input_filled")

    is_known = aug_inputs[:, is_known_idx].astype(bool)
    tvt_filled = aug_inputs[:, tvt_idx]

    # last_known under augmentation = TVT at the last is_known_aug=1 row
    lkt_aug = float(df["TVT"].to_numpy()[is_known][-1])
    # all hidden rows have tvt_filled == lkt_aug
    assert np.allclose(tvt_filled[~is_known], lkt_aug)
    # known rows have tvt_filled == ground truth TVT
    assert np.allclose(tvt_filled[is_known], df["TVT"].to_numpy()[is_known])
