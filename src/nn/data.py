"""Phase 3 NN pipeline — data loading + augmentation + batching."""

import numpy as np
import pandas as pd


def compute_well_stats(well_df: pd.DataFrame) -> dict:
    """Per-well normalization statistics.

    Used to z-score per-row inputs so each well is on its own scale.
    """
    md = well_df["MD"].to_numpy(dtype=np.float64)
    gr = well_df["GR"].to_numpy(dtype=np.float64)
    z  = well_df["Z"].to_numpy(dtype=np.float64)
    x  = well_df["X"].to_numpy(dtype=np.float64)
    y  = well_df["Y"].to_numpy(dtype=np.float64)
    md_step = np.diff(md)
    return {
        "gr_mean": float(np.nanmean(gr)),
        "gr_std":  float(np.nanstd(gr) or 1.0),
        "z_mean":  float(np.mean(z)),
        "z_std":   float(np.std(z) or 1.0),
        "x_mean":  float(np.mean(x)),
        "x_std":   float(np.std(x) or 1.0),
        "y_mean":  float(np.mean(y)),
        "y_std":   float(np.std(y) or 1.0),
        "md_min":  float(md.min()),
        "md_max":  float(md.max()),
        "md_step_median": float(np.median(md_step)) if len(md_step) else 1.0,
    }


WELL_FEATURE_NAMES = [
    "gr_z",
    "md_norm",
    "dmd",
    "z_z",
    "dz",
    "x_z",
    "y_z",
    "tvt_input_filled",
    "is_known_mask",
    "dz_dmd",
    "dx_dmd",
    "dy_dmd",
]


def build_well_inputs(well_df: pd.DataFrame, stats: dict) -> np.ndarray:
    """Build [L, 12] per-row well inputs.

    Order: WELL_FEATURE_NAMES.
    No NaNs in the output. TVT_input_filled is `last_known_TVT` on the
    hidden suffix.
    """
    n = len(well_df)
    md = well_df["MD"].to_numpy(dtype=np.float64)
    gr = well_df["GR"].to_numpy(dtype=np.float64)
    z  = well_df["Z"].to_numpy(dtype=np.float64)
    x  = well_df["X"].to_numpy(dtype=np.float64)
    y  = well_df["Y"].to_numpy(dtype=np.float64)
    tvt_input = well_df["TVT_input"].to_numpy(dtype=np.float64)

    is_known = (~np.isnan(tvt_input)).astype(np.float32)
    if is_known.sum() == 0:
        raise ValueError("Well has no known prefix")
    last_known_tvt = float(tvt_input[is_known.astype(bool)][-1])
    tvt_filled = np.where(np.isnan(tvt_input), last_known_tvt, tvt_input)

    md_range = max(stats["md_max"] - stats["md_min"], 1e-6)
    md_norm = (md - stats["md_min"]) / md_range

    md_step_med = max(stats["md_step_median"], 1e-6)
    dmd = np.diff(md, prepend=md[0]) / md_step_med
    dz  = np.diff(z,  prepend=z[0])
    dx  = np.diff(x,  prepend=x[0])
    dy  = np.diff(y,  prepend=y[0])

    sdmd = np.maximum(np.diff(md, prepend=md[0]), 1e-6)
    dz_dmd = dz / sdmd
    dx_dmd = dx / sdmd
    dy_dmd = dy / sdmd

    z_std = max(stats["z_std"], 1e-6)
    out = np.stack([
        ((gr - stats["gr_mean"]) / max(stats["gr_std"], 1e-6)).astype(np.float32),
        md_norm.astype(np.float32),
        dmd.astype(np.float32),
        ((z - stats["z_mean"]) / z_std).astype(np.float32),
        (dz / z_std).astype(np.float32),
        ((x - stats["x_mean"]) / max(stats["x_std"], 1e-6)).astype(np.float32),
        ((y - stats["y_mean"]) / max(stats["y_std"], 1e-6)).astype(np.float32),
        tvt_filled.astype(np.float32),
        is_known.astype(np.float32),
        dz_dmd.astype(np.float32),
        dx_dmd.astype(np.float32),
        dy_dmd.astype(np.float32),
    ], axis=1)
    assert out.shape == (n, len(WELL_FEATURE_NAMES))
    assert not np.isnan(out).any()
    return out
