#!/usr/bin/env python
# coding: utf-8
"""Local fork of the rank-#2 reference pipeline at
references/top-2-rank-10-784-physics-informed-baseline.py.

Modifications vs the reference:
- DATA_DIR / ARTEFACT_DIR / OUTPUT_DIR resolved relative to the repo root
  instead of /kaggle/ paths.
- Default MODE switched to "cv" (the reference defaults to "infer", which
  needs Kaggle-hosted artefacts that don't exist locally).
- Top-level dispatch wrapped in `if __name__ == "__main__":` so the module
  is importable for analysis without running.
- Added per_well_oof_rmse() helper.

Sections:
  0. Imports & Config        8. Per-Well Feature Builder
  1. Artifact Manager        9. Dataset Builder
  2. Math & Logging         10. Model Registry
  3. GR Feature Engineering 11. Cross-Validation
  4. Geometry Features      12. Final Training
  5. Beam Search            13. Ensemble Optimisation
  6. TVT Particle Filter    14. Inference & Submission
  7. ANCC Particle Filter   15. Main Dispatch
"""

# ## 0. Imports & Config

# In[ ]:


import gc
import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION CONFIG — edit these before running
# ══════════════════════════════════════════════════════════════════════════════

# MODE choices: "train" | "infer" | "cv" | "features_only" | "ensemble_only"
MODE = os.environ.get("BASELINE_MODE", "cv")

# Models to use — any subset of ["lgb", "xgb", "cb"]
# LGB dropped 2026-05-08: NM weight = 0.000 in 3-model ensemble (xgb+cb suffice).
ACTIVE_MODELS = ["xgb", "cb"]

# ── Debug flags ──
DEBUG_MAX_WELLS    = int(os.environ["DEBUG_MAX_WELLS"]) if os.environ.get("DEBUG_MAX_WELLS") else None
DEBUG_ONE_FOLD     = os.environ.get("DEBUG_ONE_FOLD", "0") == "1"
DEBUG_INSPECT_PF   = False
DEBUG_INSPECT_BEAM = False

# ── Paths ── (local repo layout; kaggle fallback retained for parity with the reference)
REPO_ROOT = Path(__file__).resolve().parents[1]
_KAGGLE_CANDIDATES = [
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
]
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else \
           next((p for p in _KAGGLE_CANDIDATES if (p / "test").exists()),
                REPO_ROOT / "data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

# Overridable via env vars so notebooks running from a read-only mount
# (e.g. /kaggle/input/) can stage artefacts to a writable working dir
# without forking the file.
ARTEFACT_DIR = Path(os.environ.get("ARTEFACT_DIR", str(REPO_ROOT / "artefacts")))
OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR",   str(REPO_ROOT / "outputs")))
ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ──
SEED     = 42
N_SPLITS = 5

# ── Feature config ──
# TVT context offsets for typewell-diff features
TVT_OFFSETS = np.array(
    [-120, -80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80, 120], dtype=np.float32
)

# Beam search configs — 4 priors on geological smoothness
BEAM_CONFIGS = [
    dict(beam_size=5,  move_cost=50.0, emit_scale=200.0, radius=1, name="tight"),
    dict(beam_size=10, move_cost=20.0, emit_scale=144.0, radius=2, name="cons"),
    dict(beam_size=15, move_cost=8.0,  emit_scale=64.0,  radius=2, name="loose"),
    dict(beam_size=20, move_cost=3.0,  emit_scale=25.0,  radius=3, name="vloose"),
]

# ── Particle Filter — TVT (Z-velocity model) ──
PF_N_PARTICLES          = 500
PF_MOMENTUM_ALPHA       = 0.993
PF_Z_SIGMA_FLOOR        = 0.005
PF_Z_SIGMA_SCALE        = 2.0
PF_VELOCITY_NOISE_STD   = 0.005
PF_POSITION_NOISE_STD   = 0.01
PF_INIT_VELOCITY_STD    = 0.02
PF_GR_SIGMA_MIN         = 10.0
PF_GR_SIGMA_MAX         = 60.0
PF_GR_SIGMA_DEFAULT     = 30.0
PF_INIT_SPREAD_STD      = 0.5
PF_RESAMPLE_THRESHOLD   = 0.5
PF_ROUGHENING_STD_POS   = 0.2
PF_ROUGHENING_STD_VEL   = 0.003
PF_GR_ROLLING_WINDOW    = 5
PF_GR_ROLLING_WEIGHT    = 0.3

# ── Particle Filter — ANCC (TVT+Z composite) ──
ANCC_ALPHA               = 0.998
ANCC_RATE_NOISE_STD      = 0.002
ANCC_POS_NOISE_STD       = 0.005
ANCC_INIT_RATE_STD       = 0.01
ANCC_INIT_SPREAD_STD     = 0.3
ANCC_ROUGHENING_STD_POS  = 0.1
ANCC_ROUGHENING_STD_RATE = 0.001
ANCC_N_PARTICLES         = 500

# ── Auto-detect GPU ──
def _has_gpu():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False

_GPU = _has_gpu()
print(f"GPU available: {_GPU}")

# ── Model hyperparameters ──
LGB_PARAMS = dict(
    n_estimators      = 3000,
    learning_rate     = 0.03,
    num_leaves        = 127,
    min_child_samples = 20,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.05,
    reg_lambda        = 1.0,
    objective         = "regression",
    metric            = "rmse",
    random_state      = SEED,
    n_jobs            = -1,
    verbosity         = -1,
    device            = "gpu" if _GPU else "cpu",
)

XGB_PARAMS = dict(
    n_estimators          = 3000,
    learning_rate         = 0.03,
    max_depth             = 7,
    min_child_weight      = 20,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    reg_alpha             = 0.05,
    reg_lambda            = 1.0,
    random_state          = SEED,
    tree_method           = "hist",
    device                = "cuda" if _GPU else "cpu",
    eval_metric           = "rmse",
    early_stopping_rounds = 100,
)

CB_PARAMS = dict(
    iterations            = 3000,
    learning_rate         = 0.03,
    depth                 = 7,
    l2_leaf_reg           = 3.0,
    subsample             = 0.8,
    bootstrap_type        = "Poisson" if _GPU else "Bernoulli",
    random_seed           = SEED,
    task_type             = "GPU" if _GPU else "CPU",
    eval_metric           = "RMSE",
    verbose               = 0,
    early_stopping_rounds = 100,
)

EARLY_STOP_ROUNDS = 100
LOG_EVERY         = 200
FINAL_ITER_SCALE  = 1.10

# ── Meta cols excluded from ML features ──
_META_COLS = {"well", "prediction_id", "target", "group_id", "row_idx", "x", "y"}

print(f"MODE={MODE}  |  ACTIVE_MODELS={ACTIVE_MODELS}")
print(f"DATA_DIR={DATA_DIR}")
print(f"ARTEFACT_DIR={ARTEFACT_DIR}")

assert TEST_DIR.exists(), f"TEST_DIR not found: {TEST_DIR}"
assert (DATA_DIR / "sample_submission.csv").exists(), \
    f"sample_submission.csv not found in {DATA_DIR}"


# ## 1. Artifact Manager

# In[ ]:


def _json_default(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


class ArtifactManager:
    """Single point of control for all pipeline I/O."""

    TRAIN_DF           = "train_df"
    TEST_DF            = "test_df"
    OOF_PREDICTIONS    = "oof_predictions"
    BEST_ITERS         = "best_iters"
    ENSEMBLE_WEIGHTS   = "ensemble_weights"
    FEATURES_LIST      = "features"
    FOLD_METRICS       = "fold_metrics"
    FEATURE_IMPORTANCE = "feature_importance"

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, obj: Any, name: str) -> Path:
        path = self.dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        print(f"  ✔ saved  {path.name}  ({path.stat().st_size/1e6:.1f} MB)")
        return path

    def load(self, name: str) -> Any:
        path = self.dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Artefact not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_json(self, obj: Any, name: str) -> Path:
        path = self.dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        print(f"  ✔ saved  {path.name}")
        return path

    def load_json(self, name: str) -> Any:
        path = self.dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {path}")
        with open(path) as f:
            return json.load(f)

    def save_df(self, df: pd.DataFrame, name: str) -> Path:
        path = self.dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  ✔ saved  {path.name}  ({path.stat().st_size/1e6:.1f} MB)")
        return path

    def load_df(self, name: str) -> pd.DataFrame:
        path = self.dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"DataFrame not found: {path}")
        return pd.read_parquet(path)

    def exists(self, name: str, ext: str = ".pkl") -> bool:
        return (self.dir / f"{name}{ext}").exists()

    def list(self):
        for p in sorted(self.dir.iterdir()):
            print(f"  {p.name:45s}  {p.stat().st_size/1e6:.2f} MB")


am = ArtifactManager(ARTEFACT_DIR)


# ## 2. Math & Logging Utilities

# In[ ]:


# ── Logging ──
_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"

def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_LOG_FMT, datefmt="%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.DEBUG)
        log.propagate = False
    return log

@contextmanager
def timer(log, label: str):
    log.info(f"▶  {label} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info(f"✔  {label} — {time.perf_counter()-t0:.2f}s")

def section(log, title: str):
    bar = "=" * 65
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)

log = get_logger("pipeline")

# ── Math helpers ──
def nearest_index(sorted_values: np.ndarray, target: float) -> int:
    """Binary-search for closest index in a sorted array."""
    idx = int(np.searchsorted(sorted_values, target, side="left"))
    if idx >= len(sorted_values):
        return len(sorted_values) - 1
    if idx > 0 and abs(sorted_values[idx-1] - target) <= abs(sorted_values[idx] - target):
        return idx - 1
    return idx


def recent_mean_diff(values: np.ndarray, window: int) -> float:
    vals = values[-(window + 1):]
    return float(np.diff(vals).mean()) if len(vals) >= 2 else 0.0


def recent_slope(y_vals: np.ndarray, x_vals: np.ndarray, window: int) -> float:
    y = y_vals[-window:]; x = x_vals[-window:]
    if len(y) < 2: return 0.0
    cx = x - x.mean(); denom = float(np.dot(cx, cx))
    return 0.0 if denom == 0.0 else float(np.dot(cx, y - y.mean()) / denom)


def safe_interp(x, xp, fp) -> np.ndarray:
    xp, fp = np.asarray(xp, float), np.asarray(fp, float)
    mask = np.isfinite(xp) & np.isfinite(fp)
    if mask.sum() < 2:
        return np.full(len(np.asarray(x)), np.nan)
    order = np.argsort(xp[mask])
    return np.interp(np.asarray(x, float), xp[mask][order], fp[mask][order],
                     left=np.nan, right=np.nan)


def fill_and_smooth_gr(values: np.ndarray, fallback: float, radius: int) -> np.ndarray:
    s = pd.Series(values, dtype="float32").interpolate(limit_direction="both").fillna(fallback)
    if radius > 0:
        s = s.rolling(radius * 2 + 1, center=True, min_periods=1).mean()
    return s.to_numpy(dtype=np.float32)


def systematic_resample(weights: np.ndarray, n: int) -> np.ndarray:
    cum = np.cumsum(weights)
    pos = (np.arange(n) + np.random.uniform()) / n
    return np.searchsorted(cum, pos)


print("Math & logging utilities loaded.")


# ## 3. GR Feature Engineering
# 
# **Leakage audit — `gr_lead1 / gr_lead5 / gr_lead10`**
# 
# GR is *fully observed* for the entire borehole. Only `TVT_input` is withheld for the hidden section.
# Therefore lead-GR features look ahead **in depth along the borehole**, not ahead in the prediction target.
# They are **legal** and among the most informative features — they provide stratigraphic context
# about the geological unit being entered. **Keep them.**

# In[ ]:


def compute_gr_features(gr_raw: pd.Series, fallback: float) -> dict:
    """
    Compute all GR-derived Series on the full well (crosses known/hidden boundary
    intentionally so rolling windows are correct at the boundary).

    Returns a dict of named pd.Series all aligned to the original well index.
    """
    def roll(s, w, fn="mean"):
        return getattr(s.rolling(w, center=True, min_periods=1), fn)()

    gf = gr_raw.astype("float32").interpolate(limit_direction="both").fillna(fallback)

    mn5  = roll(gf, 5,  "min");  mx5  = roll(gf, 5,  "max")
    mn21 = roll(gf, 21, "min");  mx21 = roll(gf, 21, "max")
    grad = gf.diff().fillna(0.0)

    return dict(
        gr_filled = gf,
        roll3     = roll(gf, 3),
        roll5     = roll(gf, 5),
        roll11    = roll(gf, 11),
        roll21    = roll(gf, 21),
        roll51    = roll(gf, 51),
        roll151   = roll(gf, 151),
        std5      = roll(gf, 5,  "std").fillna(0),
        std21     = roll(gf, 21, "std").fillna(0),
        min5      = mn5,  max5  = mx5,  range5  = mx5  - mn5,
        min21     = mn21, max21 = mx21, range21 = mx21 - mn21,
        grad      = grad,
        grad2     = grad.diff().fillna(0.0),
        lag1      = gf.shift(1).bfill(),
        lag5      = gf.shift(5).bfill(),
        lag10     = gf.shift(10).bfill(),
        lead1     = gf.shift(-1).ffill(),   # legal — GR fully observed
        lead5     = gf.shift(-5).ffill(),   # legal
        lead10    = gf.shift(-10).ffill(),  # legal
        cumsum    = gf.cumsum(),
    )


def extract_prefix_gr_stats(gf: pd.Series, mask_start: int, fallback: float) -> dict:
    known_gr = gf.iloc[:mask_start]
    n = mask_start
    return dict(
        prefix_gr_mean   = float(known_gr.mean())        if n > 0 else fallback,
        prefix_gr_std    = float(known_gr.std())         if n > 1 else 0.0,
        prefix_gr_last5  = float(known_gr.iloc[-5:].mean())  if n >= 5  else
                           (float(known_gr.mean()) if n > 0 else fallback),
        prefix_gr_last20 = float(known_gr.iloc[-20:].mean()) if n >= 20 else
                           (float(known_gr.mean()) if n > 0 else fallback),
    )

print("GR feature functions loaded.")


# ## 4. Geometry Features

# In[ ]:


def compute_geometry_features(
    hidden:         pd.DataFrame,
    last_known_md:  float,
    last_known_x:   float,
    last_known_y:   float,
    last_known_z:   float,
) -> dict:
    """Spatial displacement features from trajectory start to each hidden row."""
    md = hidden["MD"].to_numpy(dtype=np.float32)
    x  = hidden["X"].to_numpy(dtype=np.float32)
    y  = hidden["Y"].to_numpy(dtype=np.float32)
    z  = hidden["Z"].to_numpy(dtype=np.float32)

    dmd = (md - last_known_md).astype(np.float32)
    dz  = (z  - last_known_z).astype(np.float32)
    dx  = (x  - last_known_x).astype(np.float32)
    dy  = (y  - last_known_y).astype(np.float32)
    sdmd = np.maximum(dmd, 1e-5)

    return dict(
        md       = md, z  = z, x = x, y = y,
        dmd      = dmd, dz = dz, dx = dx, dy = dy,
        dx_dmd   = (dx / sdmd).astype(np.float32),
        dy_dmd   = (dy / sdmd).astype(np.float32),
        dz_dmd   = (dz / sdmd).astype(np.float32),
        dist_xy  = np.sqrt(dx**2 + dy**2).astype(np.float32),
        dist_xyz = np.sqrt(dx**2 + dy**2 + dz**2).astype(np.float32),
        md_diff  = np.diff(md,  prepend=last_known_md).astype(np.float32),
        z_diff   = np.diff(z,   prepend=last_known_z).astype(np.float32),
    )

print("Geometry feature function loaded.")


# ## 5. Beam Search Trajectory Estimation
# 
# Treats TVT prediction as a pathfinding problem on the typewell GR template.
# Four beam configurations represent different priors on geological smoothness:
# - **tight** — slow TVT drift (high move_cost)
# - **cons** — balanced
# - **loose** — allows faster excursions
# - **vloose** — minimal penalty, follows GR aggressively
# 
# Disagreement between beams = path uncertainty → key ML signal.

# In[ ]:


def beam_predict(
    gr_values:  np.ndarray,
    tw_tvt:     np.ndarray,
    tw_gr:      np.ndarray,
    start_tvt:  float,
    beam_size:  int,
    move_cost:  float,
    emit_scale: float,
    radius:     int,
) -> np.ndarray:
    """Run beam search; returns absolute TVT path aligned to gr_values length."""
    start_idx   = nearest_index(tw_tvt, start_tvt)
    fallback    = float(np.nanmean(tw_gr))
    smoothed_gr = fill_and_smooth_gr(gr_values, fallback, radius)
    T           = len(tw_tvt)

    states: Dict[int, float]    = {start_idx: 0.0}
    backpointers: List[Dict]    = []

    for gr_val in smoothed_gr:
        cands: Dict[int, float] = {}
        pars:  Dict[int, int]   = {}
        for idx, cost in states.items():
            for delta in (-1, 0, 1):
                nxt = idx + delta
                if nxt < 0 or nxt >= T: continue
                total = cost + (gr_val - tw_gr[nxt])**2 / emit_scale + move_cost * abs(delta)
                if nxt not in cands or total < cands[nxt]:
                    cands[nxt] = total; pars[nxt] = idx
        kept   = sorted(cands.items(), key=lambda kv: kv[1])[:beam_size]
        states = {i: c for i, c in kept}
        backpointers.append({i: pars[i] for i, _ in kept})

    final_idx = min(states, key=states.get)
    path = [final_idx]
    for step in range(len(backpointers) - 1, 0, -1):
        path.append(backpointers[step][path[-1]])
    path.reverse()
    return tw_tvt[np.asarray(path, dtype=np.int32)]


def compute_beam_features(
    hidden_gr_filled: np.ndarray,
    tw_tvt:           np.ndarray,
    tw_gr:            np.ndarray,
    last_known_tvt:   float,
    debug:            bool = False,
) -> dict:
    """Run all 4 beam configs; return dict of arrays (len = hidden section)."""
    paths = {}
    for cfg in BEAM_CONFIGS:
        p = beam_predict(hidden_gr_filled, tw_tvt, tw_gr, last_known_tvt,
                         cfg["beam_size"], cfg["move_cost"],
                         cfg["emit_scale"], cfg["radius"])
        paths[cfg["name"]] = p
        if debug:
            print(f"  beam [{cfg['name']}] range: {p.min():.2f}–{p.max():.2f}")

    lkt    = np.float32(last_known_tvt)
    tight  = paths["tight"]; cons  = paths["cons"]
    loose  = paths["loose"]; vloose= paths["vloose"]
    stack  = np.stack([tight, cons, loose, vloose], axis=1)

    out = {}
    for name, p in paths.items():
        out[f"beam_{name}_delta"] = (p - lkt).astype(np.float32)

    out["beam_mean"]   = (stack.mean(axis=1) - lkt).astype(np.float32)
    out["beam_std"]    = stack.std(axis=1).astype(np.float32)
    out["beam_spread"] = (vloose - tight).astype(np.float32)
    out["beam_gap"]    = (loose  - cons).astype(np.float32)

    out["tw_gr_at_beam_cons"]     = np.interp(cons,  tw_tvt, tw_gr).astype(np.float32)
    out["tw_gr_at_beam_loose"]    = np.interp(loose, tw_tvt, tw_gr).astype(np.float32)
    out["gr_minus_tw_beam_cons"]  = (hidden_gr_filled - out["tw_gr_at_beam_cons"]).astype(np.float32)
    out["gr_minus_tw_beam_loose"] = (hidden_gr_filled - out["tw_gr_at_beam_loose"]).astype(np.float32)
    return out

print("Beam search functions loaded.")


# ## 5b. DTW alignment (third aligner, decorrelated from beam + PF)

# Key differences from beam search:
#   - Global DP optimum within a Sakoe-Chiba band (no pruning).
#   - Quadratic move cost on (Δj - expected_dj), vs beam's linear |Δj|.
#   - Step pattern allows ±2 backward and up to +3 forward layer skips per
#     MD step, so non-monotone TVT(MD) is permitted (EDA showed dTVT/dMD has
#     mass on both sides).
# Same emission likelihood (GR squared error) as beam/PF; the decorrelation
# is in the search procedure and move-cost shape.

DTW_STEPS         = (-1, 0, 1)             # ref-index deltas per query step
DTW_EMIT_SCALE    = 144.0                  # match beam_cons emit_scale
DTW_SLOPE_PENALTY = 20.0                   # quadratic; matches beam_cons move_cost at |Δ|=1
DTW_BAND_CELLS    = 100                    # ± typewell cells around prior path
DTW_USE_PREFIX_SLOPE = False               # band centred on start_idx (flat); slope from data via cost


def dtw_predict(
    gr_values:      np.ndarray,
    tw_tvt:         np.ndarray,
    tw_gr:          np.ndarray,
    start_tvt:      float,
    expected_slope: float,
    band_cells:     int = DTW_BAND_CELLS,
    emit_scale:     float = DTW_EMIT_SCALE,
    slope_penalty:  float = DTW_SLOPE_PENALTY,
    radius:         int = 2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Banded DTW alignment of hidden-GR query to typewell-GR reference.

    Args:
      gr_values: hidden-zone GR (NaNs OK; will be filled+smoothed).
      tw_tvt, tw_gr: typewell reference (sorted ascending tw_tvt).
      start_tvt: TVT at last known row (alignment start).
      expected_slope: prior dTVT/dMD from the known prefix; the band is
        centred on a linear extrapolation at this slope. A unit of 1.0
        corresponds to 1 typewell cell per query step.
      band_cells: half-width of the Sakoe-Chiba band, in typewell cells.

    Returns:
      tvt_path:   absolute TVT per query step (length = len(gr_values))
      local_cost: emission cost at the matched cell, per query step
      total_cost: cumulative cost / n_steps  (overall fit quality)
    """
    n         = len(gr_values)
    T         = len(tw_tvt)
    fallback  = float(np.nanmean(tw_gr))
    smoothed  = fill_and_smooth_gr(gr_values, fallback, radius)
    start_idx = int(nearest_index(tw_tvt, start_tvt))

    # Cells per typewell step (~1 if tw is densely sampled in TVT).
    # The "expected_slope" input is in TVT-units / query-step, so we convert
    # to cells / query-step by dividing by the median typewell step in TVT.
    tw_step  = float(np.median(np.diff(tw_tvt))) if T > 1 else 1.0
    exp_dj   = float(expected_slope) / max(tw_step, 1e-6)

    # Prior trajectory in cell-space, then band each step. If
    # DTW_USE_PREFIX_SLOPE is False, the band is centred on a constant line
    # at start_idx — prefix slope can mislead on wells whose dipping rate
    # changes in the hidden zone (case in point: well 028d7b28). The slope
    # information is still used by the move-cost penalty via exp_dj.
    drift   = exp_dj if DTW_USE_PREFIX_SLOPE else 0.0
    prior_j = start_idx + drift * np.arange(n, dtype=np.float64)
    j_lo    = np.clip(np.floor(prior_j - band_cells).astype(np.int64), 0, T - 1)
    j_hi    = np.clip(np.ceil( prior_j + band_cells).astype(np.int64), 0, T - 1)
    band_w  = (j_hi - j_lo + 1).astype(np.int64)
    max_w   = int(band_w.max())

    INF       = np.float64(1e18)
    steps_arr = np.array(DTW_STEPS, dtype=np.int64)
    move_pen  = (steps_arr.astype(np.float64) - exp_dj) ** 2 * slope_penalty

    cost   = np.full((n, max_w), INF, dtype=np.float64)
    parent = np.full((n, max_w), -1,  dtype=np.int8)  # index into DTW_STEPS

    # Step 0: only the start cell is seeded.
    s0 = max(0, min(int(band_w[0]) - 1, start_idx - int(j_lo[0])))
    cost[0, s0] = (smoothed[0] - tw_gr[int(j_lo[0]) + s0]) ** 2 / emit_scale

    # Forward DP — vectorised over band cells; loop only over n and over the
    # 6 transitions in DTW_STEPS (a constant), so total work is O(n × |steps| × max_w).
    for i in range(1, n):
        lo_i, w_i = int(j_lo[i]),     int(band_w[i])
        lo_p, w_p = int(j_lo[i - 1]), int(band_w[i - 1])
        lo_diff   = lo_i - lo_p
        prev_row  = cost[i - 1]           # shape (max_w,) — first w_p valid

        best   = np.full(w_i, INF, dtype=np.float64)
        best_d = np.full(w_i, -1,  dtype=np.int8)
        ks     = np.arange(w_i)

        for d_idx, d in enumerate(steps_arr):
            p_k   = ks + lo_diff - int(d)
            valid = (p_k >= 0) & (p_k < w_p)
            if not valid.any():
                continue
            cand = np.where(
                valid,
                prev_row[np.clip(p_k, 0, max_w - 1)] + move_pen[d_idx],
                INF,
            )
            improve = cand < best
            best    = np.where(improve, cand,         best)
            best_d  = np.where(improve, d_idx,        best_d).astype(np.int8)

        emit = (smoothed[i] - tw_gr[lo_i:lo_i + w_i]) ** 2 / emit_scale
        cost[i, :w_i]   = best + emit
        parent[i, :w_i] = best_d

    # Termination: minimum over the last row.
    end_k = int(np.argmin(cost[n - 1]))
    total_cost = float(cost[n - 1, end_k]) / max(n, 1)

    path_k = np.empty(n, dtype=np.int64)
    path_k[n - 1] = end_k
    for i in range(n - 1, 0, -1):
        d_idx = parent[i, path_k[i]]
        if d_idx < 0:
            # No legal parent (shouldn't happen if start was set correctly).
            path_k[i - 1] = path_k[i]
            continue
        d = int(steps_arr[d_idx])
        prev_abs = (int(j_lo[i]) + path_k[i]) - d
        path_k[i - 1] = prev_abs - int(j_lo[i - 1])

    path_j = (j_lo + path_k).astype(np.int64)
    path_j = np.clip(path_j, 0, T - 1)
    tvt_path  = tw_tvt[path_j].astype(np.float32)
    local_cost = ((smoothed - tw_gr[path_j]) ** 2 / emit_scale).astype(np.float32)
    return tvt_path, local_cost, total_cost


def compute_dtw_features(
    hidden_gr_filled: np.ndarray,
    tw_tvt:           np.ndarray,
    tw_gr:            np.ndarray,
    last_known_tvt:   float,
    expected_slope:   float,
    beam_cons:        np.ndarray,
    beam_loose:       np.ndarray,
    pf_pred:          np.ndarray,
    ancc_pred:        np.ndarray,
) -> dict:
    """Run banded DTW; return feature dict (arrays of length = hidden section)."""
    tvt_path, local_cost, total_cost = dtw_predict(
        hidden_gr_filled, tw_tvt, tw_gr, last_known_tvt, expected_slope,
    )
    lkt = np.float32(last_known_tvt)
    out = {
        "dtw_delta"            : (tvt_path - lkt).astype(np.float32),
        "dtw_local_cost"       : local_cost,
        "dtw_total_cost"       : np.full_like(local_cost, np.float32(total_cost)),
        "dtw_minus_beam_cons"  : (tvt_path - beam_cons ).astype(np.float32),
        "dtw_minus_beam_loose" : (tvt_path - beam_loose).astype(np.float32),
        "dtw_minus_pf"         : (tvt_path - (pf_pred  if len(pf_pred ) == len(tvt_path) else np.full_like(tvt_path, lkt))).astype(np.float32),
        "dtw_minus_ancc"       : (tvt_path - (ancc_pred if len(ancc_pred) == len(tvt_path) else np.full_like(tvt_path, lkt))).astype(np.float32),
        "tw_gr_at_dtw"         : np.interp(tvt_path, tw_tvt, tw_gr).astype(np.float32),
    }
    out["gr_minus_tw_dtw"] = (hidden_gr_filled - out["tw_gr_at_dtw"]).astype(np.float32)
    return out


print("DTW alignment functions loaded.")


# ## 6. TVT Particle Filter (Z-velocity model)
# 
# Models TVT as a particle cloud driven by the Z-coordinate velocity signal
# and constrained by GR template matching. Returns predicted TVT + uncertainty std.

# In[ ]:


def _pf_calibrate_gr_sigma(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> float:
    known    = hw[hw["TVT_input"].notna()]
    known_gr = known[known["GR"].notna()]
    if len(known_gr) < 20:
        return PF_GR_SIGMA_DEFAULT
    tw_func  = interp1d(tw_tvt, tw_gr, bounds_error=False,
                        fill_value=(tw_gr[0], tw_gr[-1]))
    residuals = known_gr["GR"].values - tw_func(known_gr["TVT_input"].values)
    return float(np.clip(np.std(residuals), PF_GR_SIGMA_MIN, PF_GR_SIGMA_MAX))


def _pf_estimate_init_velocity(hw: pd.DataFrame) -> float:
    known = hw[hw["TVT_input"].notna()]
    if len(known) < 10: return 0.0
    tail = known.tail(20)
    dtvt = np.diff(tail["TVT_input"].values); dmd = np.diff(tail["MD"].values)
    mask = dmd > 0
    return float(np.median(dtvt[mask] / dmd[mask])) if mask.sum() >= 3 else 0.0


def _pf_learn_z_beta(hw: pd.DataFrame) -> Tuple[float, float, float]:
    """Fit: dTVT/dMD = beta*(dZ/dMD) + intercept. Returns (beta, intercept, sigma)."""
    known = hw[hw["TVT_input"].notna()]
    if len(known) < 30: return -1.0, 0.0, 0.1
    dz   = np.diff(known["Z"].values);           dtvt = np.diff(known["TVT_input"].values)
    dmd  = np.diff(known["MD"].values);          mask = dmd > 0
    if mask.sum() < 10: return -1.0, 0.0, 0.1
    vz   = dz[mask] / dmd[mask];                 vt   = dtvt[mask] / dmd[mask]
    coef, _, _, _ = np.linalg.lstsq(
        np.column_stack([vz, np.ones_like(vz)]), vt, rcond=None)
    return float(coef[0]), float(coef[1]), max(float(np.std(vt - (coef[0]*vz + coef[1]))), 0.001)


def run_pf_z_velocity(
    hw:   pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr:  np.ndarray,
    n_particles: int = PF_N_PARTICLES,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pred_tvts, pred_stds) for hidden rows; empty arrays if none."""
    tvt_min, tvt_max = tw_tvt.min(), tw_tvt.max()
    gr_sigma         = _pf_calibrate_gr_sigma(hw, tw_tvt, tw_gr)
    beta, intercept, z_sigma = _pf_learn_z_beta(hw)

    tw_func_pt = interp1d(tw_tvt, tw_gr, bounds_error=False,
                          fill_value=(tw_gr[0], tw_gr[-1]))
    tw_gr_sm   = pd.Series(tw_gr).rolling(PF_GR_ROLLING_WINDOW, center=True, min_periods=1).mean().values
    tw_func_sm = interp1d(tw_tvt, tw_gr_sm, bounds_error=False,
                          fill_value=(tw_gr_sm[0], tw_gr_sm[-1]))

    known = hw[hw["TVT_input"].notna()]
    evalz = hw[hw["TVT_input"].isna()]
    if len(evalz) == 0: return np.array([]), np.array([])

    hw_gr_sm = hw["GR"].rolling(PF_GR_ROLLING_WINDOW, center=True, min_periods=1).mean()

    positions  = float(known["TVT_input"].iloc[-1]) + np.random.normal(0, PF_INIT_SPREAD_STD, n_particles)
    velocities = _pf_estimate_init_velocity(hw) + np.random.normal(0, PF_INIT_VELOCITY_STD, n_particles)
    weights    = np.ones(n_particles) / n_particles

    md_vals  = evalz["MD"].values;  gr_vals = evalz["GR"].values;  z_vals = evalz["Z"].values
    prev_md  = float(known["MD"].iloc[-1]);  prev_z = float(known["Z"].iloc[-1])
    pred_tvts = np.empty(len(evalz));  pred_stds = np.empty(len(evalz))

    for i, orig_idx in enumerate(evalz.index):
        d_md       = max(md_vals[i] - prev_md, 1.0)
        dz_dmd     = (z_vals[i] - prev_z) / d_md
        v_expected = beta * dz_dmd + intercept

        velocities = PF_MOMENTUM_ALPHA * velocities + np.random.normal(0, PF_VELOCITY_NOISE_STD, n_particles)
        positions  = positions + velocities * d_md + np.random.normal(0, PF_POSITION_NOISE_STD, n_particles)
        positions  = np.clip(positions, tvt_min - 50, tvt_max + 50)

        if not np.isnan(gr_vals[i]):
            lik  = np.exp(-0.5 * ((gr_vals[i] - tw_func_pt(positions)) / gr_sigma)**2)
            gr_s = hw_gr_sm.iloc[hw.index.get_loc(orig_idx)]
            if not np.isnan(gr_s):
                lik_s = np.exp(-0.5 * ((gr_s - tw_func_sm(positions)) / (gr_sigma*1.5))**2)
                lik   = (1 - PF_GR_ROLLING_WEIGHT)*lik + PF_GR_ROLLING_WEIGHT*lik_s
            weights = np.maximum(lik, 1e-300) * weights
            s = weights.sum(); weights = weights/s if s > 0 else np.full(n_particles, 1/n_particles)

        lik_z  = np.exp(-0.5 * ((velocities - v_expected) / max(z_sigma*PF_Z_SIGMA_SCALE, PF_Z_SIGMA_FLOOR))**2)
        weights= np.maximum(lik_z, 1e-300) * weights
        s = weights.sum(); weights = weights/s if s > 0 else np.full(n_particles, 1/n_particles)

        n_eff = 1.0 / np.sum(weights**2)
        if n_eff < PF_RESAMPLE_THRESHOLD * n_particles:
            idx        = systematic_resample(weights, n_particles)
            positions  = positions[idx]  + np.random.normal(0, PF_ROUGHENING_STD_POS, n_particles)
            velocities = velocities[idx] + np.random.normal(0, PF_ROUGHENING_STD_VEL, n_particles)
            weights[:] = 1.0 / n_particles

        mu = np.average(positions, weights=weights)
        pred_tvts[i] = mu
        pred_stds[i] = np.sqrt(np.average((positions - mu)**2, weights=weights))
        prev_md = md_vals[i];  prev_z = z_vals[i]

        if debug and i % 50 == 0:
            print(f"    PF_TVT step {i:4d}: mu={mu:.3f}  std={pred_stds[i]:.3f}  n_eff={n_eff:.0f}")

    return pred_tvts, pred_stds

print("TVT Particle Filter loaded.")


# ## 7. ANCC Particle Filter (TVT+Z composite state)
# 
# Tracks `S = TVT + Z`, roughly conserved in horizontal drilling (apparent net closure change).
# Provides an independent trajectory hypothesis orthogonal to the Z-velocity PF.
# Disagreement between PF_TVT and ANCC is itself a key uncertainty feature.

# In[ ]:


def _ancc_init_rate(hw: pd.DataFrame) -> float:
    known = hw[hw["TVT_input"].notna()]
    if len(known) < 10: return 0.0
    tail = known.tail(30)
    dtvt = np.diff(tail["TVT_input"].values); dz  = np.diff(tail["Z"].values)
    dmd  = np.diff(tail["MD"].values);        mask = dmd > 0
    return float(np.median((dtvt[mask] + dz[mask]) / dmd[mask])) if mask.sum() >= 3 else 0.0


def run_pf_ancc(
    hw:   pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr:  np.ndarray,
    n_particles: int = ANCC_N_PARTICLES,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pred_tvts, pred_stds) for hidden rows."""
    tvt_min, tvt_max = tw_tvt.min(), tw_tvt.max()
    gr_sigma         = _pf_calibrate_gr_sigma(hw, tw_tvt, tw_gr)

    known = hw[hw["TVT_input"].notna()]
    evalz = hw[hw["TVT_input"].isna()]
    if len(evalz) == 0: return np.array([]), np.array([])

    pos  = float(known["TVT_input"].iloc[-1]) + float(known["Z"].iloc[-1]) \
           + np.random.normal(0, ANCC_INIT_SPREAD_STD, n_particles)
    rate = _ancc_init_rate(hw) + np.random.normal(0, ANCC_INIT_RATE_STD, n_particles)
    w    = np.ones(n_particles) / n_particles

    md_vals = evalz["MD"].values;  z_vals = evalz["Z"].values;  gr_vals = evalz["GR"].values
    prev_md = float(known["MD"].iloc[-1])
    pred_tvts = np.empty(len(evalz));  pred_stds = np.empty(len(evalz))

    for i in range(len(evalz)):
        d_md = max(md_vals[i] - prev_md, 1.0)
        rate = ANCC_ALPHA * rate + np.random.normal(0, ANCC_RATE_NOISE_STD, n_particles)
        pos  = pos + rate * d_md + np.random.normal(0, ANCC_POS_NOISE_STD, n_particles)
        tvt_c = np.clip(pos - z_vals[i], tvt_min - 50, tvt_max + 50)
        pos   = tvt_c + z_vals[i]

        if not np.isnan(gr_vals[i]):
            lik = np.exp(-0.5 * ((gr_vals[i] - np.interp(tvt_c, tw_tvt, tw_gr)) / gr_sigma)**2)
            w  *= np.maximum(lik, 1e-300)
            s = w.sum(); w = w/s if s > 0 else np.full(n_particles, 1/n_particles)

        n_eff = 1.0 / np.sum(w**2)
        if n_eff < PF_RESAMPLE_THRESHOLD * n_particles:
            idx  = systematic_resample(w, n_particles)
            pos  = pos[idx]  + np.random.normal(0, ANCC_ROUGHENING_STD_POS,  n_particles)
            rate = rate[idx] + np.random.normal(0, ANCC_ROUGHENING_STD_RATE, n_particles)
            w[:] = 1.0 / n_particles

        tvt_w = pos - z_vals[i]
        mu    = float(np.average(tvt_w, weights=w))
        pred_tvts[i] = mu
        pred_stds[i] = float(np.sqrt(np.average((tvt_w - mu)**2, weights=w)))
        prev_md = md_vals[i]

        if debug and i % 50 == 0:
            print(f"    ANCC step {i:4d}: mu={mu:.3f}  std={pred_stds[i]:.3f}  n_eff={n_eff:.0f}")

    return pred_tvts, pred_stds

print("ANCC Particle Filter loaded.")


# ## 8. Per-Well Feature Builder
# 
# Orchestrates all feature modules for a single well into one flat DataFrame.

# In[ ]:


def build_well_features(
    horizontal_path: Path,
    typewell_path:   Path,
    is_train:        bool,
    debug_pf:        bool = False,
    debug_beam:      bool = False,
) -> Optional[pd.DataFrame]:
    """
    Build complete feature DataFrame for one well.
    Returns None if no usable hidden rows found.
    """
    well = horizontal_path.name.split("__")[0]
    df   = pd.read_csv(horizontal_path)

    mask       = df["TVT_input"].isna().to_numpy()
    if not mask.any(): return None
    mask_start = int(np.flatnonzero(mask)[0])
    if mask_start == 0: return None

    known  = df.iloc[:mask_start].copy()
    hidden = df.iloc[mask_start:].copy()
    if is_train:
        hidden = hidden[hidden["TVT"].notna()].copy()
    if len(hidden) == 0: return None

    if not typewell_path.exists(): return None
    tw = pd.read_csv(typewell_path)
    if not {"TVT", "GR"}.issubset(tw.columns) or len(tw) < 2: return None
    tw_tvt = tw["TVT"].to_numpy(dtype=np.float32)
    tw_gr  = tw["GR"].to_numpy(dtype=np.float32)

    # ── GR features (full well) ───────────────────────────────────────────
    gr_mean_tw = float(np.nanmean(tw_gr))
    grd        = compute_gr_features(df["GR"], fallback=gr_mean_tw)
    pfx        = extract_prefix_gr_stats(grd["gr_filled"], mask_start, gr_mean_tw)

    # ── Known section anchors ─────────────────────────────────────────────
    last_k         = known.iloc[-1]
    known_tvt      = known["TVT_input"].to_numpy(dtype=np.float32)
    known_md       = known["MD"].to_numpy(dtype=np.float32)
    known_z        = known["Z"].to_numpy(dtype=np.float32)
    lkt            = float(last_k["TVT_input"])
    last_known_md  = float(last_k["MD"])
    last_known_z   = float(last_k["Z"])
    last_known_x   = float(last_k["X"])
    last_known_y   = float(last_k["Y"])
    last_known_gr  = float(last_k["GR"]) if not np.isnan(last_k["GR"]) else gr_mean_tw

    # ── Typewell at last known ────────────────────────────────────────────
    lk_tw_idx       = nearest_index(tw_tvt, lkt)
    tw_gr_at_last   = float(tw_gr[lk_tw_idx])
    local_win       = tw_gr[max(0, lk_tw_idx-5): lk_tw_idx+6]
    tw_gr_std_local = float(np.std(local_win)) if len(local_win) > 1 else 0.0

    # ── Prefix typewell residuals ─────────────────────────────────────────
    pr = grd["gr_filled"].iloc[:mask_start].to_numpy(np.float32) - np.interp(known_tvt, tw_tvt, tw_gr)
    prefix_tw_rmse = float(np.sqrt(np.mean(pr**2))) if len(pr) else 0.0
    prefix_tw_mae  = float(np.mean(np.abs(pr)))     if len(pr) else 0.0
    prefix_tw_bias = float(pr.mean())               if len(pr) else 0.0

    # ── Prefix trend features ─────────────────────────────────────────────
    pfx_tvt_step20   = recent_mean_diff(known_tvt, 20)
    pfx_tvt_step100  = recent_mean_diff(known_tvt, 100)
    pfx_tvt_md_slope = recent_slope(known_tvt, known_md, 100)
    pfx_tvt_z_slope  = recent_slope(known_tvt, known_z,  100)

    # ── Index mappings ────────────────────────────────────────────────────
    sel_full  = hidden.index.to_numpy(dtype=np.int64)
    sel_local = sel_full - mask_start

    # ── Hidden GR filled ──────────────────────────────────────────────────
    hgr_full = grd["gr_filled"].iloc[mask_start:].to_numpy(dtype=np.float32)
    hgr_sel  = hgr_full[sel_local]
    n_hidden = int(df["TVT_input"].isna().sum())

    # ── Geometry ──────────────────────────────────────────────────────────
    geo = compute_geometry_features(hidden, last_known_md, last_known_x,
                                    last_known_y, last_known_z)

    # ── Beam features ─────────────────────────────────────────────────────
    bf = compute_beam_features(hgr_full, tw_tvt, tw_gr, lkt, debug=debug_beam)

    # ── Particle filters ──────────────────────────────────────────────────
    np.random.seed(SEED)
    pf_pred,   pf_std   = run_pf_z_velocity(df, tw_tvt, tw_gr, debug=debug_pf)
    ancc_pred, ancc_std = run_pf_ancc(df,       tw_tvt, tw_gr, debug=debug_pf)

    if len(pf_pred)   == 0: pf_pred   = np.zeros(n_hidden); pf_std   = np.ones(n_hidden)
    if len(ancc_pred) == 0: ancc_pred = np.zeros(n_hidden); ancc_std = np.ones(n_hidden)

    pf_delta   = (pf_pred   - lkt).astype(np.float32)
    ancc_delta = (ancc_pred - lkt).astype(np.float32)

    # ── DTW alignment ─────────────────────────────────────────────────────
    # expected_slope is TVT-units per query step (one query step = one MD row).
    md_step      = float(np.median(np.diff(df["MD"].to_numpy())))
    exp_slope    = float(pfx_tvt_md_slope) * md_step
    beam_cons_abs  = bf["beam_cons_delta"]  + lkt
    beam_loose_abs = bf["beam_loose_delta"] + lkt
    df_dtw = compute_dtw_features(
        hgr_full, tw_tvt, tw_gr, lkt, exp_slope,
        beam_cons_abs, beam_loose_abs, pf_pred, ancc_pred,
    )

    # ── TVT context offset features ───────────────────────────────────────
    offset_feats = {
        f"tw_diff_{int(o):+d}": (
            hgr_sel - np.float32(np.interp(lkt + float(o), tw_tvt, tw_gr))
        ).astype(np.float32)
        for o in TVT_OFFSETS
    }

    # ── GR cumsum relative to mask start ─────────────────────────────────
    cs_offset = grd["cumsum"].iloc[mask_start-1] if mask_start > 0 else 0.0
    hcs = (grd["cumsum"].iloc[mask_start:] - cs_offset).to_numpy(dtype=np.float32)

    # ── Positional fraction ───────────────────────────────────────────────
    frac = (sel_local / max(n_hidden - 1, 1)).astype(np.float32)

    # ── Baseline slope features ───────────────────────────────────────────
    baseline_slope = (lkt + pfx_tvt_md_slope * geo["dmd"]).astype(np.float32)
    tw_gr_at_slope = safe_interp(baseline_slope, tw_tvt, tw_gr).astype(np.float32)

    # ── Assemble DataFrame ────────────────────────────────────────────────
    out = pd.DataFrame({
        # meta
        "well"          : well,
        "prediction_id" : [f"{well}_{i}" for i in sel_full],
        "row_idx"       : sel_full.astype(np.int32),
        # section info
        "last_known_tvt": np.float32(lkt),
        "known_len"     : np.int32(mask_start),
        "hidden_len"    : np.int32(n_hidden),
        "frac_hidden"   : frac,
        # geometry
        **geo,
        # GR raw
        "gr"            : hgr_sel,
        "gr_missing"    : hidden["GR"].isna().to_numpy(dtype=np.int8),
        "last_known_gr" : np.float32(last_known_gr),
        # GR rolling
        "gr_roll3"   : grd["roll3"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_roll5"   : grd["roll5"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_roll11"  : grd["roll11"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_roll21"  : grd["roll21"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_roll51"  : grd["roll51"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_roll151" : grd["roll151"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_std5"    : grd["std5"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_std21"   : grd["std21"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_min5"    : grd["min5"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_max5"    : grd["max5"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_min21"   : grd["min21"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_max21"   : grd["max21"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_range5"  : grd["range5"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_range21" : grd["range21"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_grad"    : grd["grad"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_grad2"   : grd["grad2"].iloc[sel_full].to_numpy(dtype=np.float32),
        # GR lags / leads (leads LEGAL — GR fully observed)
        "gr_lag1"    : grd["lag1"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_lag5"    : grd["lag5"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_lag10"   : grd["lag10"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_lead1"   : grd["lead1"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_lead5"   : grd["lead5"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_lead10"  : grd["lead10"].iloc[sel_full].to_numpy(dtype=np.float32),
        "gr_cumsum"  : hcs[sel_local],
        "gr_minus_last_known": (hgr_sel - last_known_gr).astype(np.float32),
        # prefix stats
        "prefix_gr_mean"      : np.float32(pfx["prefix_gr_mean"]),
        "prefix_gr_std"       : np.float32(pfx["prefix_gr_std"]),
        "prefix_gr_last5"     : np.float32(pfx["prefix_gr_last5"]),
        "prefix_gr_last20"    : np.float32(pfx["prefix_gr_last20"]),
        "prefix_tw_rmse"      : np.float32(prefix_tw_rmse),
        "prefix_tw_mae"       : np.float32(prefix_tw_mae),
        "prefix_tw_bias"      : np.float32(prefix_tw_bias),
        "prefix_tvt_step20"   : np.float32(pfx_tvt_step20),
        "prefix_tvt_step100"  : np.float32(pfx_tvt_step100),
        "prefix_tvt_md_slope" : np.float32(pfx_tvt_md_slope),
        "prefix_tvt_z_slope"  : np.float32(pfx_tvt_z_slope),
        # typewell stats
        "tw_gr_at_last"   : np.float32(tw_gr_at_last),
        "tw_gr_std_local" : np.float32(tw_gr_std_local),
        "tw_tvt_range"    : np.float32(float(tw_tvt[-1] - tw_tvt[0])),
        "tw_gr_global_mean": np.float32(float(tw_gr.mean())),
        "tw_gr_global_std" : np.float32(float(tw_gr.std())),
        # known TVT stats
        "known_tvt_min"   : np.float32(known_tvt.min()),
        "known_tvt_max"   : np.float32(known_tvt.max()),
        "known_tvt_range" : np.float32(known_tvt.max() - known_tvt.min()),
        "known_tvt_std"   : np.float32(known_tvt.std()),
        # beam features (indexed into hidden section)
        **{k: v[sel_local] for k, v in bf.items()},
        # PF features
        "pf_delta"           : pf_delta[sel_local],
        "pf_std"             : pf_std[sel_local],
        "pf_beam_cons_diff"  : (pf_delta - bf["beam_cons_delta"])[sel_local],
        "pf_beam_loose_diff" : (pf_delta - bf["beam_loose_delta"])[sel_local],
        "ancc_delta"         : ancc_delta[sel_local],
        "ancc_std"           : ancc_std[sel_local],
        "ancc_beam_cons_diff": (ancc_delta - bf["beam_cons_delta"])[sel_local],
        "ancc_pf_diff"       : (ancc_delta - pf_delta)[sel_local],
        # DTW features (indexed into hidden section)
        **{k: v[sel_local] for k, v in df_dtw.items()},
        # baseline / typewell residuals
        "baseline_slope"     : baseline_slope,
        "tw_gr_at_slope"     : tw_gr_at_slope,
        "gr_minus_tw_lkt"    : (hgr_sel - tw_gr_at_last).astype(np.float32),
        "gr_minus_tw_slope"  : (hgr_sel - tw_gr_at_slope).astype(np.float32),
    })

    for col, vals in offset_feats.items():
        out[col] = vals

    if is_train:
        out["target"] = (
            hidden["TVT"].to_numpy(dtype=np.float32) - np.float32(lkt)
        ).astype(np.float32)

    return out.reset_index(drop=True)

print("Per-well feature builder loaded.")


# ## 9. Dataset Builder

# In[ ]:


def build_dataset(
    data_dir:   Path,
    is_train:   bool,
    max_wells:  Optional[int] = None,
    debug_pf:   bool = False,
    debug_beam: bool = False,
) -> pd.DataFrame:
    hw_files = sorted(data_dir.glob("*__horizontal_well.csv"))
    if max_wells is not None:
        hw_files = hw_files[:max_wells]
        print(f"  DEBUG: limited to {max_wells} wells")

    split = "train" if is_train else "test"
    print(f"Building {split} dataset from {len(hw_files)} wells …")

    parts: List[pd.DataFrame] = []
    n_skip = 0

    for hp in tqdm(hw_files, desc=split):
        well = hp.name.split("__")[0]
        tp   = data_dir / f"{well}__typewell.csv"
        try:
            feat = build_well_features(hp, tp, is_train=is_train,
                                       debug_pf=debug_pf, debug_beam=debug_beam)
            if feat is not None and len(feat) > 0:
                parts.append(feat)
            else:
                n_skip += 1
        except Exception as exc:
            print(f"  WARNING [{well}]: {exc}")
            n_skip += 1

    if not parts:
        raise RuntimeError("No wells produced features — check data paths")
    df = pd.concat(parts, ignore_index=True)
    print(f"  → {df.shape}  wells={df['well'].nunique()}  skipped={n_skip}")
    return df


def _typewell_hash(well: str, data_dir: Path) -> str:
    with open(data_dir / f"{well}__typewell.csv", "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def assign_groups(df: pd.DataFrame, data_dir: Path = TRAIN_DIR,
                  pad_eps_frac: float = 0.005) -> pd.DataFrame:
    """Assign group_ids by union of (shared typewell-file hash) ∪ (pad cluster).

    Pads are DBSCAN clusters on per-well (x_mean, y_mean) centroids with
    eps = pad_eps_frac × (X-Y bbox diagonal). DBSCAN noise (-1) wells stay
    singletons unless linked via typewell-hash. EDA Phase 1 found
    13 typewell hashes shared by ≥2 wells (33 wells) and ~106 pad clusters,
    so vanilla per-well GroupKFold leaks both signals.
    """
    wells = sorted(df["well"].unique())

    tw_hash = {w: _typewell_hash(w, data_dir) for w in wells}

    centroids = df.groupby("well")[["x", "y"]].mean().loc[wells]
    bbox_diag = float(np.hypot(
        centroids["x"].max() - centroids["x"].min(),
        centroids["y"].max() - centroids["y"].min(),
    ))
    eps = pad_eps_frac * bbox_diag
    pad_label = dict(zip(
        wells,
        DBSCAN(eps=eps, min_samples=2).fit_predict(centroids.values),
    ))

    parent = {w: w for w in wells}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_hash: Dict[str, List[str]] = {}
    for w in wells:
        by_hash.setdefault(tw_hash[w], []).append(w)
    for grp in by_hash.values():
        for other in grp[1:]:
            union(grp[0], other)

    by_pad: Dict[int, List[str]] = {}
    for w in wells:
        if pad_label[w] != -1:
            by_pad.setdefault(int(pad_label[w]), []).append(w)
    for grp in by_pad.values():
        for other in grp[1:]:
            union(grp[0], other)

    roots = sorted({find(w) for w in wells})
    root_to_id = {r: i for i, r in enumerate(roots)}
    well_to_gid = {w: root_to_id[find(w)] for w in wells}

    sizes = pd.Series(list(well_to_gid.values())).value_counts()
    n_groups = len(sizes)
    n_singletons = int((sizes == 1).sum())
    n_multi = n_groups - n_singletons
    print(f"  groups: {n_groups} total ({n_singletons} singletons, "
          f"{n_multi} multi-well, largest={int(sizes.max())})  "
          f"[eps={eps:.0f}]")

    df["group_id"] = df["well"].map(well_to_gid).astype(np.int32)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _META_COLS]

print("Dataset builder loaded.")


# ## 10. Model Registry

# In[ ]:


def make_model(key: str, params: dict = None):
    defaults = {"lgb": LGB_PARAMS, "xgb": XGB_PARAMS, "cb": CB_PARAMS}
    if key not in defaults: raise ValueError(f"Unknown model key: {key}")
    p = {**defaults[key], **(params or {})}
    return {"lgb": LGBMRegressor, "xgb": XGBRegressor, "cb": CatBoostRegressor}[key](**p)


def fit_model_cv(key: str, model, Xtr, ytr, Xva, yva):
    """Fit with early stopping for CV."""
    if key == "lgb":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                              log_evaluation(LOG_EVERY)])
    elif key == "xgb":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=LOG_EVERY)
    else:
        model.fit(Xtr, ytr, eval_set=(Xva, yva))
    return model


def get_best_iteration(key: str, model) -> int:
    if key == "lgb": return int(model.best_iteration_)
    if key == "xgb": return int(model.best_iteration)
    return int(model.best_iteration_)


def fit_model_final(key: str, params: dict, X, y):
    """Fit on full data — no early stopping."""
    if key == "lgb":
        p = {k: v for k, v in params.items() if k != "metric"}
        m = LGBMRegressor(**p)
        m.fit(X, y, callbacks=[log_evaluation(LOG_EVERY)])
    elif key == "xgb":
        p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        m = XGBRegressor(**p); m.fit(X, y, verbose=LOG_EVERY)
    else:
        p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        p["verbose"] = LOG_EVERY
        m = CatBoostRegressor(**p); m.fit(X, y)
    return m


def save_model(key: str, model, directory: Path) -> Path:
    if key == "lgb":
        path = directory / "final_lgb.txt"; model.booster_.save_model(str(path))
    elif key == "xgb":
        path = directory / "final_xgb.json"; model.save_model(str(path))
    else:
        path = directory / "final_cb.cbm"; model.save_model(str(path))
    print(f"  ✔ saved {path.name}")
    return path


def load_model(key: str, directory: Path):
    if key == "lgb":
        return lgb.Booster(model_file=str(directory / "final_lgb.txt"))
    elif key == "xgb":
        m = XGBRegressor(); m.load_model(str(directory / "final_xgb.json")); return m
    else:
        m = CatBoostRegressor(); m.load_model(str(directory / "final_cb.cbm")); return m


def model_predict(key: str, model, X) -> np.ndarray:
    return model.predict(X).astype(np.float64)

print("Model registry loaded.")


# ## 11. Cross-Validation (GroupKFold + OOF)

# In[ ]:


def run_cv(
    train_df:      pd.DataFrame,
    feature_cols:  List[str],
    active_models: List[str] = None,
    one_fold:      bool = False,
) -> Tuple[dict, dict, pd.DataFrame]:
    """
    GroupKFold CV. Wells stay whole within folds.
    Returns (oof_preds, best_iters, fold_metrics_df).
    """
    active = active_models or ACTIVE_MODELS
    X      = train_df
    y      = train_df["target"]
    groups = train_df["group_id"]
    n      = len(train_df)

    cv     = GroupKFold(n_splits=N_SPLITS)
    splits = list(cv.split(X, y, groups=groups))
    if one_fold:
        splits = splits[:1]; print("DEBUG: running only fold 0")

    oof_preds  = {k: np.zeros(n, dtype=np.float64) for k in active}
    best_iters = {k: [] for k in active}
    fold_scores= {k: [] for k in active}
    fold_ids   = []

    section(log, f"CROSS-VALIDATION  ({N_SPLITS}-fold GroupKFold)")

    for fold, (tr_idx, va_idx) in enumerate(splits):
        Xtr = X.iloc[tr_idx][feature_cols]; ytr = y.iloc[tr_idx]
        Xva = X.iloc[va_idx][feature_cols]; yva = y.iloc[va_idx]
        w_tr = X.iloc[tr_idx]["well"].nunique(); w_va = X.iloc[va_idx]["well"].nunique()
        log.info(f"\n── Fold {fold+1}/{len(splits)}  "
                 f"(train={len(tr_idx):,}  val={len(va_idx):,}  "
                 f"wells_tr={w_tr}  wells_va={w_va}) ──")
        fold_ids.append(fold + 1)

        for key in active:
            m     = make_model(key)
            m     = fit_model_cv(key, m, Xtr, ytr, Xva, yva)
            preds = model_predict(key, m, Xva)
            oof_preds[key][va_idx] = preds
            bi    = get_best_iteration(key, m)
            best_iters[key].append(bi)
            rmse  = root_mean_squared_error(yva, preds)
            fold_scores[key].append(rmse)
            log.info(f"  {key.upper():3s}  iter={bi:5d}  fold_RMSE={rmse:.5f}")

    # Summary table
    section(log, "CV SUMMARY")
    rows = {k: fold_scores[k] + [np.mean(fold_scores[k]), np.std(fold_scores[k])] for k in active}
    idx  = [f"fold_{i}" for i in fold_ids] + ["mean", "std"]
    fm_df = pd.DataFrame(rows, index=idx)
    try:    log.info("\n" + fm_df.to_markdown(floatfmt=".5f"))
    except: log.info("\n" + fm_df.to_string(float_format=lambda v: f"{v:.5f}"))

    for key in active:
        log.info(f"  OOF RMSE  {key.upper():3s} = "
                 f"{root_mean_squared_error(y, oof_preds[key]):.5f}")

    return oof_preds, best_iters, fm_df

print("Cross-validation function loaded.")


# ## 12. Final Training

# In[ ]:


def run_final_training(
    train_df:      pd.DataFrame,
    feature_cols:  List[str],
    best_iters:    dict,
    active_models: List[str] = None,
) -> Tuple[dict, Optional[pd.DataFrame]]:
    """
    Retrain on full data at (mean_best_iter × FINAL_ITER_SCALE) per model.
    Returns (models_dict, feature_importance_df).
    """
    active = active_models or ACTIVE_MODELS
    section(log, "FINAL TRAINING — FULL DATA")
    X = train_df[feature_cols]; y = train_df["target"]
    models = {}

    for key in active:
        n_iters = max(50, int(round(np.mean(best_iters[key]) * FINAL_ITER_SCALE)))
        base    = {"lgb": LGB_PARAMS, "xgb": XGB_PARAMS, "cb": CB_PARAMS}[key]
        params  = dict(base)
        params.pop("early_stopping_rounds", None)
        if key in ("lgb", "xgb"): params["n_estimators"] = n_iters
        else:                      params["iterations"]   = n_iters
        log.info(f"  {key.upper():3s}  n_iters={n_iters}")
        m = fit_model_final(key, params, X, y)
        save_model(key, m, ARTEFACT_DIR)
        models[key] = m

    fi_df = None
    fi_source = None
    if "lgb" in models:
        try:    imps = models["lgb"].feature_importances_
        except: imps = models["lgb"].booster_.feature_importance(importance_type="gain")
        fi_source = "LGB gain"
    elif "xgb" in models:
        imps = models["xgb"].feature_importances_
        fi_source = "XGB gain"
    elif "cb" in models:
        imps = models["cb"].get_feature_importance()
        fi_source = "CatBoost"
    if fi_source is not None:
        fi_df = (pd.DataFrame({"feature": feature_cols, "importance": imps})
                 .sort_values("importance", ascending=False).reset_index(drop=True))
        log.info(f"\nTop-25 features ({fi_source}):")
        log.info("\n" + fi_df.head(25).to_string(index=False))

    return models, fi_df

print("Final training function loaded.")


# ## 13. Ensemble Optimisation (Nelder-Mead)

# In[ ]:


def optimise_ensemble_weights(
    oof_preds:     dict,
    y_true:        np.ndarray,
    active_models: List[str] = None,
    n_restarts:    int = 5,
) -> dict:
    """
    Minimise OOF RMSE over convex weight combinations.
    Multiple random restarts to avoid local minima.
    """
    active     = [k for k in (active_models or ACTIVE_MODELS) if k in oof_preds]
    oof_matrix = np.column_stack([oof_preds[k] for k in active])
    n_models   = len(active)

    def objective(raw_w):
        w = np.maximum(raw_w, 0.0); w /= (w.sum() + 1e-12)
        return root_mean_squared_error(y_true, oof_matrix @ w)

    rng = np.random.default_rng(SEED)
    best_res = None
    for trial in range(n_restarts):
        w0  = np.ones(n_models)/n_models if trial == 0 else rng.dirichlet(np.ones(n_models))
        res = minimize(objective, w0, method="Nelder-Mead",
                       options={"maxiter": 30_000, "xatol": 1e-12, "fatol": 1e-12})
        if best_res is None or res.fun < best_res.fun: best_res = res

    w = np.maximum(best_res.x, 0.0); w /= (w.sum() + 1e-12)
    weights = {k: float(w[i]) for i, k in enumerate(active)}

    section(log, "ENSEMBLE WEIGHTS (Nelder-Mead)")
    for k, wv in weights.items(): log.info(f"  {k.upper():6s}: {wv:.4f}")
    ens_rmse = root_mean_squared_error(y_true, oof_matrix @ w)
    eq_rmse  = root_mean_squared_error(y_true, oof_matrix.mean(axis=1))
    log.info(f"\n  OOF ensemble RMSE : {ens_rmse:.6f}")
    log.info(f"  Equal-weight RMSE : {eq_rmse:.6f}  (Δ={eq_rmse-ens_rmse:+.6f})")
    return weights


def apply_ensemble(test_preds: dict, weights: dict) -> np.ndarray:
    keys = list(weights.keys())
    mat  = np.column_stack([test_preds[k] for k in keys])
    w    = np.array([weights[k] for k in keys])
    return mat @ w

print("Ensemble optimisation loaded.")


# ## 14. Inference & Submission

# In[ ]:


def generate_test_predictions(
    test_df:      pd.DataFrame,
    feature_cols: List[str],
    models:       dict,
    weights:      dict,
) -> np.ndarray:
    section(log, "INFERENCE")
    X_test = test_df[feature_cols]
    test_preds = {}
    for key, model in models.items():
        preds = model_predict(key, model, X_test)
        test_preds[key] = preds
        log.info(f"  {key.upper():3s} delta range: {preds.min():.2f}–{preds.max():.2f}")

    ensemble_delta = apply_ensemble(test_preds, weights)
    abs_tvt = ensemble_delta + test_df["last_known_tvt"].to_numpy()
    log.info(f"  abs TVT range: {abs_tvt.min():.2f}–{abs_tvt.max():.2f}")
    return abs_tvt


def build_submission(
    test_df:     pd.DataFrame,
    abs_tvt:     np.ndarray,
    sample_sub:  pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Map predictions onto the sample submission.

    Raises immediately if ANY prediction ID in the sample is missing from
    test_df — no silent zero-fill. This turns a scoring disaster into a
    loud, debuggable error caught before submission.
    """
    pred_map = dict(zip(test_df["prediction_id"], abs_tvt))
    sub = sample_sub.copy()
    sub["tvt"] = sub["id"].map(pred_map)

    missing_mask = sub["tvt"].isna()
    miss = int(missing_mask.sum())

    if miss > 0:
        missing_ids = sub.loc[missing_mask, "id"].tolist()
        n_show = min(10, len(missing_ids))
        raise RuntimeError(
            f"{miss} prediction IDs in sample_submission.csv have no match in test_df.\n"
            f"First {n_show} missing: {missing_ids[:n_show]}\n\n"
            f"Likely cause: TEST_DIR is pointing at the wrong folder, or "
            f"build_well_features silently skipped wells.\n"
            f"TEST_DIR = {TEST_DIR}\n"
            f"test_df prediction_id sample: {test_df['prediction_id'].iloc[:5].tolist()}"
        )

    sub.to_csv(output_path, index=False)
    log.info(f"Submission → {output_path}  ({len(sub):,} rows, 0 missing)")
    log.info(f"\n{sub.head()}")
    return sub

print("Inference & submission functions loaded.")


# ## 15. Main Dispatch
# 
# Controlled by `MODE` set at the top of the notebook.

# In[ ]:


def _get_or_build_test_df() -> pd.DataFrame:
    """
    Always build test features fresh from the live TEST_DIR during inference.

    The cache is ONLY used in non-infer modes (train / cv / features_only)
    where the test dir is stable. In infer mode the competition swaps in
    hidden wells that share no prediction IDs with any previously cached file,
    so loading from cache produces a total ID mismatch and zero-fills the sub.
    """
    if MODE != "infer" and am.exists(am.TEST_DF, ".parquet"):
        log.info("Loading cached test dataset …")
        return am.load_df(am.TEST_DF)

    # Always rebuild when inferring against the live test set
    log.info(f"Building test features fresh from {TEST_DIR} …")
    with timer(log, "building test features"):
        test_df = build_dataset(
            TEST_DIR,
            is_train=False,
            max_wells=DEBUG_MAX_WELLS,
            debug_pf=DEBUG_INSPECT_PF,
            debug_beam=DEBUG_INSPECT_BEAM,
        )

    # Cache for reuse within the same non-infer session only
    if MODE != "infer":
        am.save_df(test_df, am.TEST_DF)

    return test_df


def _get_or_build_train_df():
    try:
        log.info("Loading cached train dataset …")
        train_df     = am.load_df(am.TRAIN_DF)
        train_df     = assign_groups(train_df)
        feature_cols = am.load_json(am.FEATURES_LIST)
        return train_df, feature_cols
    except FileNotFoundError:
        pass
    log.info("Cache miss — building train features …")
    with timer(log, "train feature engineering"):
        train_df = build_dataset(TRAIN_DIR, is_train=True,
                                 max_wells=DEBUG_MAX_WELLS,
                                 debug_pf=DEBUG_INSPECT_PF,
                                 debug_beam=DEBUG_INSPECT_BEAM)
    train_df = assign_groups(train_df)
    am.save_df(train_df, am.TRAIN_DF)
    feature_cols = get_feature_columns(train_df)
    am.save_json(feature_cols, am.FEATURES_LIST)
    return train_df, feature_cols


def per_well_oof_rmse(oof_df: pd.DataFrame, ensemble_col: Optional[str] = None) -> pd.DataFrame:
    """Per-well RMSE from saved OOF predictions.

    `oof_df` must contain columns: well, target, and one or more `oof_<model>`.
    If `ensemble_col` is provided it must exist as a column; otherwise the
    equal-weight mean of the available `oof_*` columns is used.
    """
    pred_cols = [c for c in oof_df.columns if c.startswith("oof_")]
    if ensemble_col is None:
        oof_df = oof_df.copy()
        oof_df["_ens"] = oof_df[pred_cols].mean(axis=1)
        ensemble_col = "_ens"
    rows = []
    for w, g in oof_df.groupby("well"):
        y, p = g["target"].to_numpy(), g[ensemble_col].to_numpy()
        per_model = {f"rmse_{c[4:]}": float(np.sqrt(np.mean((g[c].to_numpy() - y) ** 2))) for c in pred_cols}
        rows.append({
            "well": w,
            "n_rows": len(g),
            "rmse_ens": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae_ens": float(np.mean(np.abs(p - y))),
            "bias_ens": float(np.mean(p - y)),
            **per_model,
        })
    return pd.DataFrame(rows).sort_values("rmse_ens", ascending=False).reset_index(drop=True)


def _dispatch():
    """Run the configured MODE. Called only from `__main__`."""
    section(log, f"GEO-PIPELINE  |  MODE={MODE}  |  MODELS={ACTIVE_MODELS}")
    log.info(f"Data dir   : {DATA_DIR}")
    log.info(f"Artefacts  : {ARTEFACT_DIR}")
    np.random.seed(SEED)

    if MODE == "features_only":
        train_df, feature_cols = _get_or_build_train_df()
        log.info(f"Train: {train_df.shape}  features: {len(feature_cols)}")
        test_df = _get_or_build_test_df()
        log.info(f"Test : {test_df.shape}")

    elif MODE == "cv":
        train_df, feature_cols = _get_or_build_train_df()
        oof_preds, best_iters, fm_df = run_cv(
            train_df, feature_cols,
            active_models=ACTIVE_MODELS,
            one_fold=DEBUG_ONE_FOLD,
        )
        oof_df = train_df[["well", "prediction_id", "target"]].copy()
        for k, p in oof_preds.items():
            oof_df[f"oof_{k}"] = p
        am.save_df(oof_df, am.OOF_PREDICTIONS)
        am.save_json(best_iters, am.BEST_ITERS)
        am.save_df(fm_df.reset_index(), am.FOLD_METRICS)
        log.info("CV complete. OOF + best_iters saved.")

    elif MODE == "ensemble_only":
        oof_df    = am.load_df(am.OOF_PREDICTIONS)
        y_true    = oof_df["target"].to_numpy()
        oof_preds = {k: oof_df[f"oof_{k}"].to_numpy()
                     for k in ACTIVE_MODELS if f"oof_{k}" in oof_df.columns}
        weights   = optimise_ensemble_weights(oof_preds, y_true, ACTIVE_MODELS)
        am.save_json(weights, am.ENSEMBLE_WEIGHTS)

    elif MODE == "finalize_only":
        # Skip CV entirely. Use cached best_iters from a previous CV run to
        # train final models on full data, then compute ensemble weights from
        # cached OOF. No test inference. Use this when the CV pass has
        # already produced best_iters + oof_predictions and you just want
        # the trained model artefacts for off-machine inference (e.g. Kaggle).
        train_df, feature_cols = _get_or_build_train_df()
        best_iters = am.load_json(am.BEST_ITERS)
        active = [k for k in ACTIVE_MODELS if k in best_iters]
        if active != ACTIVE_MODELS:
            log.warning(f"best_iters has {list(best_iters)} but ACTIVE_MODELS={ACTIVE_MODELS}; "
                        f"finalising for intersection {active}")

        models, fi_df = run_final_training(train_df, feature_cols, best_iters,
                                           active_models=active)
        if fi_df is not None:
            am.save_df(fi_df, am.FEATURE_IMPORTANCE)

        oof_df    = am.load_df(am.OOF_PREDICTIONS)
        y_true    = oof_df["target"].to_numpy()
        oof_preds = {k: oof_df[f"oof_{k}"].to_numpy()
                     for k in active if f"oof_{k}" in oof_df.columns}
        weights   = optimise_ensemble_weights(oof_preds, y_true, active)
        am.save_json(weights, am.ENSEMBLE_WEIGHTS)
        log.info("Final training + ensemble weights saved. No test inference.")

    elif MODE == "train":
        train_df, feature_cols = _get_or_build_train_df()
        y = train_df["target"]

        oof_preds, best_iters, fm_df = run_cv(
            train_df, feature_cols,
            active_models=ACTIVE_MODELS,
            one_fold=DEBUG_ONE_FOLD,
        )
        oof_df = train_df[["well", "prediction_id", "target"]].copy()
        for k, p in oof_preds.items():
            oof_df[f"oof_{k}"] = p
        am.save_df(oof_df, am.OOF_PREDICTIONS)
        am.save_json(best_iters, am.BEST_ITERS)
        am.save_df(fm_df.reset_index(), am.FOLD_METRICS)

        models, fi_df = run_final_training(train_df, feature_cols, best_iters,
                                           active_models=ACTIVE_MODELS)
        if fi_df is not None:
            am.save_df(fi_df, am.FEATURE_IMPORTANCE)

        weights = optimise_ensemble_weights(oof_preds, y.to_numpy(), ACTIVE_MODELS)
        am.save_json(weights, am.ENSEMBLE_WEIGHTS)

        test_df = _get_or_build_test_df()
        abs_tvt = generate_test_predictions(test_df, feature_cols, models, weights)
        sample  = pd.read_csv(DATA_DIR / "sample_submission.csv")
        build_submission(test_df, abs_tvt, sample, OUTPUT_DIR / "submission.csv")

    elif MODE == "infer":
        feature_cols = am.load_json(am.FEATURES_LIST)
        weights      = am.load_json(am.ENSEMBLE_WEIGHTS)
        models       = {k: load_model(k, ARTEFACT_DIR) for k in ACTIVE_MODELS}
        log.info(f"Loaded models: {list(models.keys())}")

        test_df = _get_or_build_test_df()
        abs_tvt = generate_test_predictions(test_df, feature_cols, models, weights)
        sample  = pd.read_csv(DATA_DIR / "sample_submission.csv")
        build_submission(test_df, abs_tvt, sample, OUTPUT_DIR / "submission.csv")

    else:
        log.error(f"Unknown MODE='{MODE}'. "
                  f"Choose from: train | infer | cv | features_only | ensemble_only | finalize_only")

    section(log, "DONE")


if __name__ == "__main__":
    _dispatch()

