"""EDA Phase 1 pipeline.

Single pass over all train + test wells. Builds per-well summary tables,
caches them as parquet, and produces per-well GR-vs-typewell scale stats.

Imported by notebooks/eda_phase1.ipynb. Pure functions; no side effects on
import beyond defining the loaders.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
CACHE_DIR = REPO / "eda_outputs"
CACHE_DIR.mkdir(exist_ok=True)
FIGS_DIR = CACHE_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

LEAK_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "TVT"]


def list_wells(split_dir: Path) -> list[str]:
    names = set()
    for p in split_dir.iterdir():
        if p.suffix == ".csv":
            stem = p.name.split("__")[0]
            names.add(stem)
    return sorted(names)


def file_inventory() -> pd.DataFrame:
    """Return a DataFrame of which files exist for every well in train and test."""
    rows = []
    for split, d in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        for w in list_wells(d):
            rows.append(
                dict(
                    split=split,
                    well=w,
                    has_horizontal=(d / f"{w}__horizontal_well.csv").exists(),
                    has_typewell=(d / f"{w}__typewell.csv").exists(),
                    has_png=(d / f"{w}.png").exists(),
                )
            )
    return pd.DataFrame(rows)


def _load_horizontal(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_typewell(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@dataclass
class WellSummary:
    """Per-well aggregates we compute in a single pass."""

    well: str
    split: str

    # geometry / sizes
    n_rows: int
    md_min: float
    md_max: float
    md_step_med: float
    z_min: float
    z_max: float
    x_mean: float
    y_mean: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    # mask / hidden zone
    mask_start_idx: int  # first NaN in TVT_input (-1 if no NaN)
    known_len: int
    hidden_len: int
    hidden_ratio: float
    md_at_mask: float
    md_known_span: float
    md_hidden_span: float

    # TVT (train only — NaN otherwise)
    tvt_known_min: float
    tvt_known_max: float
    dtvtdmd_mean: float
    dtvtdmd_std: float
    abs_dtvtdmd_mean: float
    abs_dtvtdmd_max: float
    d2tvtdmd2_std: float
    d2tvtdmd2_max_abs: float
    n_jumps_3sigma: int  # count of |d2| > 3*global_sigma proxy (per-well via |d2| > 5*per-well median-abs-dev)

    # GR (whole well)
    gr_mean: float
    gr_std: float
    gr_p10: float
    gr_p50: float
    gr_p90: float
    gr_min: float
    gr_max: float

    # GR known prefix (for fair vs typewell comparison)
    gr_pref_mean: float
    gr_pref_std: float
    gr_pref_p10: float
    gr_pref_p50: float
    gr_pref_p90: float

    # Typewell
    tw_n_rows: int
    tw_tvt_min: float
    tw_tvt_max: float
    tw_gr_mean: float
    tw_gr_std: float
    tw_gr_p10: float
    tw_gr_p50: float
    tw_gr_p90: float
    tw_has_geology: bool

    # Coverage / mismatch
    tvt_input_min: float
    tvt_input_max: float
    cov_inside: bool          # is [tvt_input_min, tvt_input_max] inside [tw_tvt_min, tw_tvt_max]?
    cov_low_margin: float     # tvt_input_min - tw_tvt_min  (>=0 means inside on low side)
    cov_high_margin: float    # tw_tvt_max - tvt_input_max  (>=0 means inside on high side)
    gr_scale_offset: float    # gr_pref_mean - tw_gr_mean
    gr_scale_ratio: float     # gr_pref_std / tw_gr_std
    gr_ks_proxy: float        # max |empirical CDF diff| at deciles


def _safe(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def summarize_well(well: str, split: str) -> WellSummary:
    d = TRAIN_DIR if split == "train" else TEST_DIR
    h = _load_horizontal(d / f"{well}__horizontal_well.csv")
    tw = _load_typewell(d / f"{well}__typewell.csv")

    md = h["MD"].to_numpy()
    tvt_input = h["TVT_input"].to_numpy()
    gr = h["GR"].to_numpy(dtype=float)

    # mask
    nan_mask = np.isnan(tvt_input)
    if nan_mask.any():
        first_nan = int(np.argmax(nan_mask))
    else:
        first_nan = -1
    n = len(h)
    known_len = first_nan if first_nan >= 0 else n
    hidden_len = n - known_len
    md_at_mask = float(md[first_nan]) if first_nan > 0 else float("nan")

    # TVT smoothness — train only on the known prefix
    if "TVT" in h.columns and split == "train":
        tvt = h["TVT"].to_numpy(dtype=float)
        # Use known prefix where TVT is observed AND TVT_input is known too
        kp = slice(0, known_len)
        tvt_kp = tvt[kp]
        md_kp = md[kp]
        if len(tvt_kp) >= 3 and np.all(np.diff(md_kp) > 0):
            dtvt = np.diff(tvt_kp) / np.diff(md_kp)
            d2 = np.diff(dtvt)
            dtvtdmd_mean = float(np.nanmean(dtvt))
            dtvtdmd_std = float(np.nanstd(dtvt))
            abs_mean = float(np.nanmean(np.abs(dtvt)))
            abs_max = float(np.nanmax(np.abs(dtvt)))
            d2_std = float(np.nanstd(d2))
            d2_max = float(np.nanmax(np.abs(d2)))
            mad = float(np.nanmedian(np.abs(d2 - np.nanmedian(d2))) + 1e-9)
            n_jumps = int(np.sum(np.abs(d2) > 8 * mad))
        else:
            dtvtdmd_mean = dtvtdmd_std = abs_mean = abs_max = float("nan")
            d2_std = d2_max = float("nan")
            n_jumps = 0
        tvt_known_min = float(np.nanmin(tvt_kp)) if len(tvt_kp) else float("nan")
        tvt_known_max = float(np.nanmax(tvt_kp)) if len(tvt_kp) else float("nan")
    else:
        dtvtdmd_mean = dtvtdmd_std = abs_mean = abs_max = float("nan")
        d2_std = d2_max = float("nan")
        n_jumps = 0
        tvt_known_min = tvt_known_max = float("nan")

    # GR (whole well)
    gr_stats = {
        "mean": _safe(np.nanmean(gr)),
        "std": _safe(np.nanstd(gr)),
        "p10": _safe(np.nanpercentile(gr, 10)),
        "p50": _safe(np.nanpercentile(gr, 50)),
        "p90": _safe(np.nanpercentile(gr, 90)),
        "min": _safe(np.nanmin(gr)),
        "max": _safe(np.nanmax(gr)),
    }
    gr_pref = gr[:known_len] if known_len > 0 else gr
    pref_stats = {
        "mean": _safe(np.nanmean(gr_pref)),
        "std": _safe(np.nanstd(gr_pref)),
        "p10": _safe(np.nanpercentile(gr_pref, 10)),
        "p50": _safe(np.nanpercentile(gr_pref, 50)),
        "p90": _safe(np.nanpercentile(gr_pref, 90)),
    }

    # Typewell
    tw_tvt = tw["TVT"].to_numpy(dtype=float)
    tw_gr = tw["GR"].to_numpy(dtype=float)
    tw_stats = {
        "mean": _safe(np.nanmean(tw_gr)),
        "std": _safe(np.nanstd(tw_gr)),
        "p10": _safe(np.nanpercentile(tw_gr, 10)),
        "p50": _safe(np.nanpercentile(tw_gr, 50)),
        "p90": _safe(np.nanpercentile(tw_gr, 90)),
    }

    # Coverage
    if known_len > 0:
        tvt_input_kp = tvt_input[:known_len]
        ti_min = _safe(np.nanmin(tvt_input_kp))
        ti_max = _safe(np.nanmax(tvt_input_kp))
    else:
        ti_min = ti_max = float("nan")
    cov_low = ti_min - _safe(np.nanmin(tw_tvt))
    cov_high = _safe(np.nanmax(tw_tvt)) - ti_max
    cov_inside = bool((cov_low >= 0) and (cov_high >= 0))

    # GR scale offsets
    offset = pref_stats["mean"] - tw_stats["mean"]
    ratio = (pref_stats["std"] / tw_stats["std"]) if tw_stats["std"] and tw_stats["std"] > 0 else float("nan")
    # crude KS: max diff of decile breakpoints
    deciles = np.linspace(10, 90, 9)
    h_q = np.nanpercentile(gr_pref, deciles)
    t_q = np.nanpercentile(tw_gr, deciles)
    if pref_stats["std"] > 0 and tw_stats["std"] > 0:
        ks_proxy = float(np.max(np.abs(h_q - t_q)) / max(pref_stats["std"], tw_stats["std"]))
    else:
        ks_proxy = float("nan")

    return WellSummary(
        well=well,
        split=split,
        n_rows=int(n),
        md_min=_safe(md.min()),
        md_max=_safe(md.max()),
        md_step_med=_safe(np.median(np.diff(md))) if n > 1 else float("nan"),
        z_min=_safe(np.nanmin(h["Z"])),
        z_max=_safe(np.nanmax(h["Z"])),
        x_mean=_safe(np.nanmean(h["X"])),
        y_mean=_safe(np.nanmean(h["Y"])),
        x_min=_safe(np.nanmin(h["X"])),
        x_max=_safe(np.nanmax(h["X"])),
        y_min=_safe(np.nanmin(h["Y"])),
        y_max=_safe(np.nanmax(h["Y"])),
        mask_start_idx=int(first_nan),
        known_len=int(known_len),
        hidden_len=int(hidden_len),
        hidden_ratio=float(hidden_len / n) if n > 0 else float("nan"),
        md_at_mask=md_at_mask,
        md_known_span=float(md[known_len - 1] - md[0]) if known_len > 1 else float("nan"),
        md_hidden_span=float(md[-1] - md[known_len]) if hidden_len > 0 else float("nan"),
        tvt_known_min=tvt_known_min,
        tvt_known_max=tvt_known_max,
        dtvtdmd_mean=dtvtdmd_mean,
        dtvtdmd_std=dtvtdmd_std,
        abs_dtvtdmd_mean=abs_mean,
        abs_dtvtdmd_max=abs_max,
        d2tvtdmd2_std=d2_std,
        d2tvtdmd2_max_abs=d2_max,
        n_jumps_3sigma=n_jumps,
        gr_mean=gr_stats["mean"],
        gr_std=gr_stats["std"],
        gr_p10=gr_stats["p10"],
        gr_p50=gr_stats["p50"],
        gr_p90=gr_stats["p90"],
        gr_min=gr_stats["min"],
        gr_max=gr_stats["max"],
        gr_pref_mean=pref_stats["mean"],
        gr_pref_std=pref_stats["std"],
        gr_pref_p10=pref_stats["p10"],
        gr_pref_p50=pref_stats["p50"],
        gr_pref_p90=pref_stats["p90"],
        tw_n_rows=int(len(tw)),
        tw_tvt_min=_safe(np.nanmin(tw_tvt)),
        tw_tvt_max=_safe(np.nanmax(tw_tvt)),
        tw_gr_mean=tw_stats["mean"],
        tw_gr_std=tw_stats["std"],
        tw_gr_p10=tw_stats["p10"],
        tw_gr_p50=tw_stats["p50"],
        tw_gr_p90=tw_stats["p90"],
        tw_has_geology=("Geology" in tw.columns),
        tvt_input_min=ti_min,
        tvt_input_max=ti_max,
        cov_inside=cov_inside,
        cov_low_margin=cov_low,
        cov_high_margin=cov_high,
        gr_scale_offset=_safe(offset),
        gr_scale_ratio=_safe(ratio),
        gr_ks_proxy=ks_proxy,
    )


def build_summary(force: bool = False) -> pd.DataFrame:
    cache = CACHE_DIR / "well_summary.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    rows = []
    for split, d in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        wells = list_wells(d)
        for i, w in enumerate(wells):
            try:
                s = summarize_well(w, split)
                rows.append(s.__dict__)
            except Exception as e:
                rows.append({"well": w, "split": split, "_error": str(e)})
            if (i + 1) % 100 == 0 or i == len(wells) - 1:
                print(f"  {split}: {i+1}/{len(wells)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    return df


def list_train_pngs_present() -> int:
    return sum(
        1 for w in list_wells(TRAIN_DIR) if (TRAIN_DIR / f"{w}.png").exists()
    )


def list_test_pngs_present() -> int:
    return sum(
        1 for w in list_wells(TEST_DIR) if (TEST_DIR / f"{w}.png").exists()
    )


def leakage_audit() -> dict:
    """Return per-test-well leakage / structure checks."""
    out = {}
    for w in list_wells(TEST_DIR):
        h = pd.read_csv(TEST_DIR / f"{w}__horizontal_well.csv")
        bad = [c for c in LEAK_COLS if c in h.columns]
        ti = h["TVT_input"].to_numpy()
        nan_mask = np.isnan(ti)
        first_nan = int(np.argmax(nan_mask)) if nan_mask.any() else -1
        # confirm: no NaNs before first_nan, all NaN after first_nan
        if first_nan >= 0:
            before_clean = bool(np.all(~np.isnan(ti[:first_nan])))
            after_all_nan = bool(np.all(np.isnan(ti[first_nan:])))
        else:
            before_clean = bool(np.all(~np.isnan(ti)))
            after_all_nan = True
        out[w] = dict(
            leak_cols_present=bad,
            first_nan_idx=first_nan,
            n_total=len(ti),
            n_nan=int(nan_mask.sum()),
            before_clean=before_clean,
            after_all_nan=after_all_nan,
        )
    return out
