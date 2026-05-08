"""Phase 3 NN pipeline — data loading + augmentation + batching."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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


def apply_prefix_augmentation(
    well_df: pd.DataFrame,
    well_inputs: np.ndarray,
    well_stats: dict,
    p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random prefix-length augmentation for training.

    Args:
        well_df: original horizontal CSV (must have a `TVT` column — train only).
        well_inputs: `build_well_inputs` output for this well.
        well_stats: as returned by `compute_well_stats`.
        p: prefix-end MD-fraction in [0, 1].
        rng: numpy Generator (kept for future stochastic extensions; unused here).

    Returns:
        aug_inputs: copy of `well_inputs` with `is_known_mask` and
            `tvt_input_filled` overwritten according to `p`.
        target: per-row TVT (ground truth).
        target_mask: 1 on rows where loss contributes (hidden suffix under p),
            0 elsewhere.
    """
    if "TVT" not in well_df.columns:
        raise ValueError("apply_prefix_augmentation needs ground-truth TVT (train only)")

    md = well_df["MD"].to_numpy(dtype=np.float64)
    md_range = max(well_stats["md_max"] - well_stats["md_min"], 1e-6)
    md_norm = (md - well_stats["md_min"]) / md_range
    is_known_aug = (md_norm <= p).astype(np.float32)
    if is_known_aug.sum() == 0:
        # degenerate: at least keep the first row known
        is_known_aug[0] = 1.0

    tvt = well_df["TVT"].to_numpy(dtype=np.float64)
    last_idx = int(np.flatnonzero(is_known_aug)[-1])
    lkt_aug = float(tvt[last_idx])

    is_known_idx = WELL_FEATURE_NAMES.index("is_known_mask")
    tvt_idx = WELL_FEATURE_NAMES.index("tvt_input_filled")

    aug_inputs = well_inputs.copy()
    aug_inputs[:, is_known_idx] = is_known_aug
    aug_inputs[:, tvt_idx] = np.where(is_known_aug.astype(bool), tvt, lkt_aug).astype(np.float32)

    target = tvt.astype(np.float32)
    target_mask = (1.0 - is_known_aug).astype(np.float32)
    return aug_inputs, target, target_mask


GEOLOGY_NAMES = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
TYPEWELL_FEATURE_NAMES = [
    "tw_gr_z",
    "tw_tvt_z",
] + [f"geo_{g}" for g in GEOLOGY_NAMES]


def build_typewell_inputs(tw_df: pd.DataFrame, well_stats: dict) -> np.ndarray:
    """Build [L_tw, 8] per-row typewell inputs.

    GR z-scored against the *well's* GR statistics (cross-well normalization).
    TVT z-scored against the typewell's own TVT statistics.
    Geology one-hot over the 6 known classes; unknown geologies -> all zeros.
    """
    n = len(tw_df)
    tw_gr = tw_df["GR"].to_numpy(dtype=np.float64)
    tw_tvt = tw_df["TVT"].to_numpy(dtype=np.float64)

    gr_z = (tw_gr - well_stats["gr_mean"]) / max(well_stats["gr_std"], 1e-6)
    tvt_mean = float(np.mean(tw_tvt))
    tvt_std = float(np.std(tw_tvt) or 1.0)
    tvt_z = (tw_tvt - tvt_mean) / tvt_std

    geo_strings = (
        tw_df["Geology"].astype(str).to_numpy()
        if "Geology" in tw_df.columns
        else np.array(["UNK"] * n)
    )
    onehot = np.zeros((n, len(GEOLOGY_NAMES)), dtype=np.float32)
    for i, g in enumerate(GEOLOGY_NAMES):
        onehot[:, i] = (geo_strings == g).astype(np.float32)

    out = np.concatenate([
        gr_z.astype(np.float32)[:, None],
        tvt_z.astype(np.float32)[:, None],
        onehot,
    ], axis=1)
    assert out.shape == (n, len(TYPEWELL_FEATURE_NAMES))
    assert not np.isnan(out).any()
    return out


class WellDataset(Dataset):
    """Per-well dataset. One well = one example.

    For training: applies random prefix-length augmentation each __getitem__.
    For validation: uses the natural prefix split (TVT_input mask).
    """

    def __init__(
        self,
        wells: list[str],
        data_dir: Path,
        training: bool,
        seed: int = 42,
        prefix_p_min: float = 0.10,
        prefix_p_max: float = 0.90,
    ):
        self.wells = list(wells)
        self.data_dir = Path(data_dir)
        self.training = training
        self.prefix_p_min = prefix_p_min
        self.prefix_p_max = prefix_p_max
        self._seed = seed
        self._rng: np.random.Generator | None = None

        # Cache parsed dataframes + stats once.
        self._cache: dict[str, dict] = {}

    def _ensure_rng(self):
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            wid = worker.id if worker else 0
            self._rng = np.random.default_rng(self._seed + wid)

    def _load_well(self, well: str) -> dict:
        if well in self._cache:
            return self._cache[well]
        h_path = self.data_dir / f"{well}__horizontal_well.csv"
        t_path = self.data_dir / f"{well}__typewell.csv"
        well_df = pd.read_csv(h_path)
        tw_df = pd.read_csv(t_path)
        stats = compute_well_stats(well_df)
        well_inputs = build_well_inputs(well_df, stats)
        tw_inputs = build_typewell_inputs(tw_df, stats)
        self._cache[well] = {
            "well_df": well_df,
            "stats": stats,
            "well_inputs": well_inputs,
            "tw_inputs": tw_inputs,
        }
        return self._cache[well]

    def __len__(self) -> int:
        return len(self.wells)

    def __getitem__(self, idx: int) -> dict:
        self._ensure_rng()
        well = self.wells[idx]
        rec = self._load_well(well)
        well_df = rec["well_df"]
        well_inputs = rec["well_inputs"]
        tw_inputs = rec["tw_inputs"]
        stats = rec["stats"]

        if self.training:
            p = float(self._rng.uniform(self.prefix_p_min, self.prefix_p_max))
            aug_inputs, target, target_mask = apply_prefix_augmentation(
                well_df=well_df,
                well_inputs=well_inputs,
                well_stats=stats,
                p=p,
                rng=self._rng,
            )
        else:
            # Natural prefix: target = TVT (train only); mask is on hidden rows.
            is_known_idx = WELL_FEATURE_NAMES.index("is_known_mask")
            is_known = well_inputs[:, is_known_idx].astype(bool)
            target_mask = (~is_known).astype(np.float32)
            if "TVT" in well_df.columns:
                target = well_df["TVT"].to_numpy(dtype=np.float32)
            else:
                target = np.zeros(len(well_inputs), dtype=np.float32)
            aug_inputs = well_inputs

        return {
            "well": well,
            "well_inputs": torch.from_numpy(aug_inputs),
            "typewell_inputs": torch.from_numpy(tw_inputs),
            "target": torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_mask),
        }


def pad_collate(batch: list[dict]) -> dict:
    """Pad variable-length wells/typewells to per-batch max with attention masks."""
    B = len(batch)
    L_well = max(item["well_inputs"].shape[0] for item in batch)
    L_tw   = max(item["typewell_inputs"].shape[0] for item in batch)
    F_well = batch[0]["well_inputs"].shape[1]
    F_tw   = batch[0]["typewell_inputs"].shape[1]

    well_inputs = torch.zeros(B, L_well, F_well, dtype=torch.float32)
    typewell_inputs = torch.zeros(B, L_tw, F_tw, dtype=torch.float32)
    well_mask = torch.zeros(B, L_well, dtype=torch.float32)
    typewell_mask = torch.zeros(B, L_tw, dtype=torch.float32)
    target = torch.zeros(B, L_well, dtype=torch.float32)
    target_mask = torch.zeros(B, L_well, dtype=torch.float32)

    wells = []
    for i, item in enumerate(batch):
        nw = item["well_inputs"].shape[0]
        nt = item["typewell_inputs"].shape[0]
        well_inputs[i, :nw] = item["well_inputs"]
        well_mask[i, :nw] = 1.0
        typewell_inputs[i, :nt] = item["typewell_inputs"]
        typewell_mask[i, :nt] = 1.0
        target[i, :nw] = item["target"]
        target_mask[i, :nw] = item["target_mask"]
        wells.append(item["well"])

    return {
        "wells": wells,
        "well_inputs": well_inputs,
        "well_mask": well_mask,
        "typewell_inputs": typewell_inputs,
        "typewell_mask": typewell_mask,
        "target": target,
        "target_mask": target_mask,
    }
