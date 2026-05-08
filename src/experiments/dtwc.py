"""Composite-emit DTW aligner (NEGATIVE RESULT — kept for reference).

Status: TESTED 2026-05-08. Does not improve OOF or LB. Removed from
[src/baseline.py](../baseline.py) on the same day. Do NOT wire back in
without re-evaluating the whole approach.

Why kept: the design is a clean instance of "different-observation aligner"
informed by Phase 2 EDA. If we revisit alignment-ambiguity wells with a
different angle (e.g., a per-row distinctiveness gate, a different
search procedure, or a non-DTW model), this code is a reasonable starting
point. Throwing it away would lose the design notes and the tuned scales.

## Design

The standard DTW (5b in [src/baseline.py](../baseline.py)) and beam/PF
aligners share a single GR-magnitude likelihood and lock onto the same
wrong layer in low-distinctiveness typewell regions
([JOURNAL.md](../../JOURNAL.md) entry 2026-05-08, Phase 2 EDA). This
variant replaces the emit cost with a weighted sum of three components:

    E(i,j) = w_mag  * (gr[i] - tw[j])^2          / mag_scale
           + w_grad * (dgr[i] - dtw_dgr[j])^2    / grad_scale
           + w_lp   * (lp_gr[i] - lp_tw[j])^2    / lp_scale

- `mag` (magnitude): same as existing DTW, kept at half weight.
- `grad` (gradient): central difference. Penalises candidates whose
  typewell-GR is locally rising while observed-GR is falling. Targets
  the trajectory-shape failure (Panel C of the Phase 2 EDA).
- `lp` (low-pass): wide-radius smoothing so the slow trend dominates.
  Targets the look-alike-layer failure (Panel B).

Search procedure is the same banded DP as 5b: steps `{-1, 0, 1}`,
quadratic slope penalty, flat band. Only the emit cost differs.

## Smoke-test result (2026-05-08)

On the 25 EDA Phase 2 wells, dtwc raw RMSE vs dtw raw RMSE:

| group   | mean dtw RMSE | mean dtwc RMSE | mean Δ      |
|---------|---------------|----------------|-------------|
| Worst-20| 41.16         | 35.23          | **−5.92**   |
| Easy-5  | 14.13         | 21.89          | **+7.76**   |

Strong improvement on the worst wells, regression on easy wells.

## Full-CV result (2026-05-08)

| metric            | dtw only (12.531 OOF) | dtw + dtwc       | Δ        |
|-------------------|-----------------------|------------------|----------|
| XGB OOF           | 12.612                | 12.625           | +0.013   |
| CB OOF            | 12.661                | 12.697           | +0.036   |
| Ensemble OOF (NM) | **12.531**            | **12.551**       | **+0.020** |
| p50 per-well RMSE | 7.63                  | 7.74             | +0.11    |
| p90 per-well RMSE | 18.46                 | 18.73            | +0.27    |
| p99 per-well RMSE | 39.72                 | 38.56            | −1.16    |
| Worst-20 mean     | 38.28                 | 38.23            | −0.05    |

The right tail compressed slightly (p99 −1.16) but the bulk of the
distribution shifted worse. Worst-20 improved 9 / 20 — essentially flat.
The dtwc raw improvement on hard wells did not translate to the ensemble:
the GBDTs already had enough signal from existing aligners + disagreement
features, and on easy wells the new `dtwc_minus_*` features look like
noise (dtwc disagrees with dtw because dtwc is wrong, not because dtw is).

## What this rules out (do not retry naively)

- More-aggressive composite weights (raising w_grad / w_lp, lowering
  w_mag): likely makes the easy-well regression worse. Not promising.
- Tuning emit scales: the smoke-test wins were 6+ RMSE on hard wells
  yet ensemble OOF moved 0.02. Re-tuning emit scales at the alignment
  level cannot unlock signal that the GBDTs aren't capable of arbitrating.
- Adding dtwc as a single-model submission (no ensemble): would lose
  the diversification advantage that XGB+CB ensemble already provides.

## What might still work (future directions)

- A per-row distinctiveness gate: feed the GBDTs a feature that says
  "at this predicted TVT, is the typewell GR pattern unique enough that
  any aligner's prediction is trustworthy?" — gives the GBDTs the
  per-row arbitration signal they currently lack between dtw and dtwc.
- A non-DTW model (sequence neural net) where the observation model is
  learned, not hand-coded. Out of scope for the current GBDT pipeline.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# ---- Tuning constants (do not edit without re-running the full smoke test) ----

DTWC_STEPS         = (-1, 0, 1)
DTWC_BAND_CELLS    = 100
DTWC_SLOPE_PENALTY = 20.0
DTWC_RADIUS        = 2
DTWC_GRAD_RADIUS   = 5
DTWC_LP_RADIUS_TW  = 100
DTWC_LP_RADIUS_HZ  = 70
DTWC_W_MAG         = 0.5
DTWC_W_GRAD        = 1.0
DTWC_W_LP          = 1.0
DTWC_MAG_SCALE     = 144.0
DTWC_GRAD_SCALE    = 25.0
DTWC_LP_SCALE      = 144.0


def _central_diff(arr: np.ndarray, half_window: int) -> np.ndarray:
    """Symmetric central difference over a fixed half-window, edge-safe."""
    n = len(arr)
    if n == 0:
        return arr.astype(np.float64)
    a = arr.astype(np.float64)
    out = np.empty(n, dtype=np.float64)
    hw = max(1, int(half_window))
    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n - 1, i + hw)
        out[i] = (a[hi] - a[lo]) / max(hi - lo, 1)
    return out


def _box_smooth(arr: np.ndarray, radius: int) -> np.ndarray:
    """Centered moving-average smoothing, edge-safe."""
    n = len(arr)
    if n == 0 or radius <= 0:
        return arr.astype(np.float64)
    a = arr.astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(a)))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out[i] = (cs[hi] - cs[lo]) / max(hi - lo, 1)
    return out


def dtwc_predict(
    gr_values:      np.ndarray,
    tw_tvt:         np.ndarray,
    tw_gr:          np.ndarray,
    start_tvt:      float,
    expected_slope: float,
    fill_and_smooth_gr,  # passed in to avoid importing baseline.py
    nearest_index,
    band_cells:     int = DTWC_BAND_CELLS,
    slope_penalty:  float = DTWC_SLOPE_PENALTY,
    radius:         int = DTWC_RADIUS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Banded DTW with composite (magnitude + gradient + low-pass) emit cost.

    Returns (tvt_path, local_cost, total_cost) — same interface as
    baseline.dtw_predict. To run, pass `baseline.fill_and_smooth_gr` and
    `baseline.nearest_index` so this module stays importable on its own.
    """
    n         = len(gr_values)
    T         = len(tw_tvt)
    fallback  = float(np.nanmean(tw_gr))
    smoothed  = fill_and_smooth_gr(gr_values, fallback, radius)
    start_idx = int(nearest_index(tw_tvt, start_tvt))

    tw_step  = float(np.median(np.diff(tw_tvt))) if T > 1 else 1.0
    exp_dj   = float(expected_slope) / max(tw_step, 1e-6)

    tw_grad = _central_diff(tw_gr, DTWC_GRAD_RADIUS)
    tw_lp   = _box_smooth(tw_gr, DTWC_LP_RADIUS_TW)
    q_grad  = _central_diff(smoothed, DTWC_GRAD_RADIUS)
    q_lp    = _box_smooth(smoothed, DTWC_LP_RADIUS_HZ)

    drift   = 0.0
    prior_j = start_idx + drift * np.arange(n, dtype=np.float64)
    j_lo    = np.clip(np.floor(prior_j - band_cells).astype(np.int64), 0, T - 1)
    j_hi    = np.clip(np.ceil( prior_j + band_cells).astype(np.int64), 0, T - 1)
    band_w  = (j_hi - j_lo + 1).astype(np.int64)
    max_w   = int(band_w.max())

    INF       = np.float64(1e18)
    steps_arr = np.array(DTWC_STEPS, dtype=np.int64)
    move_pen  = (steps_arr.astype(np.float64) - exp_dj) ** 2 * slope_penalty

    cost   = np.full((n, max_w), INF, dtype=np.float64)
    parent = np.full((n, max_w), -1,  dtype=np.int8)

    def emit_row(i: int, lo_i: int, w_i: int) -> np.ndarray:
        sl = slice(lo_i, lo_i + w_i)
        e_mag  = (smoothed[i] - tw_gr[sl]) ** 2 / DTWC_MAG_SCALE
        e_grad = (q_grad[i]  - tw_grad[sl]) ** 2 / DTWC_GRAD_SCALE
        e_lp   = (q_lp[i]    - tw_lp[sl]) ** 2 / DTWC_LP_SCALE
        return DTWC_W_MAG * e_mag + DTWC_W_GRAD * e_grad + DTWC_W_LP * e_lp

    s0 = max(0, min(int(band_w[0]) - 1, start_idx - int(j_lo[0])))
    cost[0, s0] = emit_row(0, int(j_lo[0]), int(band_w[0]))[s0]

    for i in range(1, n):
        lo_i, w_i = int(j_lo[i]),     int(band_w[i])
        lo_p, w_p = int(j_lo[i - 1]), int(band_w[i - 1])
        lo_diff   = lo_i - lo_p
        prev_row  = cost[i - 1]

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

        cost[i, :w_i]   = best + emit_row(i, lo_i, w_i)
        parent[i, :w_i] = best_d

    end_k = int(np.argmin(cost[n - 1]))
    total_cost = float(cost[n - 1, end_k]) / max(n, 1)

    path_k = np.empty(n, dtype=np.int64)
    path_k[n - 1] = end_k
    for i in range(n - 1, 0, -1):
        d_idx = parent[i, path_k[i]]
        if d_idx < 0:
            path_k[i - 1] = path_k[i]
            continue
        d = int(steps_arr[d_idx])
        prev_abs = (int(j_lo[i]) + path_k[i]) - d
        path_k[i - 1] = prev_abs - int(j_lo[i - 1])

    path_j     = np.clip((j_lo + path_k).astype(np.int64), 0, T - 1)
    tvt_path   = tw_tvt[path_j].astype(np.float32)
    local_cost = ((smoothed - tw_gr[path_j]) ** 2 / DTWC_MAG_SCALE).astype(np.float32)
    return tvt_path, local_cost, total_cost
