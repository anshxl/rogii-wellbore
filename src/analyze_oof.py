"""Per-well OOF RMSE breakdown.

Run after `BASELINE_MODE=cv uv run python src/baseline.py` has populated
artefacts/oof_predictions.parquet. Joins to eda_outputs/well_summary.parquet
so we can see which well-properties correlate with high OOF error.

Outputs:
- artefacts/per_well_oof.parquet  — full per-well table
- prints overall + per-model OOF RMSE
- prints top-20 worst wells with their cluster + cov margins
- prints overall and Nelder-Mead-optimal ensemble RMSE
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error

REPO = Path(__file__).resolve().parents[1]
ART  = REPO / "artefacts"
EDA  = REPO / "eda_outputs"


def _ensemble_weights(oof_matrix: np.ndarray, y: np.ndarray, n_restarts: int = 5):
    rng = np.random.default_rng(42)
    best = None
    n = oof_matrix.shape[1]

    def obj(raw):
        w = np.maximum(raw, 0.0); w /= w.sum() + 1e-12
        return root_mean_squared_error(y, oof_matrix @ w)

    for t in range(n_restarts):
        w0 = np.ones(n) / n if t == 0 else rng.dirichlet(np.ones(n))
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 30_000, "xatol": 1e-12, "fatol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    w = np.maximum(best.x, 0.0); w /= w.sum() + 1e-12
    return w, float(best.fun)


def main():
    oof = pd.read_parquet(ART / "oof_predictions.parquet")
    pred_cols = sorted(c for c in oof.columns if c.startswith("oof_"))
    model_keys = [c[4:] for c in pred_cols]
    y = oof["target"].to_numpy()

    print("=" * 70)
    print(f"Rows: {len(oof):,}    Wells: {oof['well'].nunique()}    Models: {model_keys}")
    print("=" * 70)

    # Per-model OOF RMSE
    print("\nPer-model OOF RMSE:")
    for c in pred_cols:
        rmse = root_mean_squared_error(y, oof[c].to_numpy())
        print(f"  {c:>10s}  {rmse:.5f}")

    # Equal-weight ensemble
    eq = oof[pred_cols].mean(axis=1).to_numpy()
    print(f"\n  equal-weight ens   {root_mean_squared_error(y, eq):.5f}")

    # Nelder-Mead-optimal ensemble
    M = oof[pred_cols].to_numpy()
    w, ens_rmse = _ensemble_weights(M, y)
    print(f"  NM-optimal ens     {ens_rmse:.5f}")
    for k, wt in zip(model_keys, w):
        print(f"     w[{k}] = {wt:.4f}")
    oof["oof_ens"] = M @ w

    # Per-well table
    rows = []
    for well, g in oof.groupby("well"):
        yw = g["target"].to_numpy()
        d = {"well": well, "n_rows": len(g)}
        for c in pred_cols:
            d[f"rmse_{c[4:]}"] = float(np.sqrt(np.mean((g[c].to_numpy() - yw) ** 2)))
        d["rmse_ens"] = float(np.sqrt(np.mean((g["oof_ens"].to_numpy() - yw) ** 2)))
        d["bias_ens"] = float(np.mean(g["oof_ens"].to_numpy() - yw))
        d["target_std"] = float(np.std(yw))
        rows.append(d)
    pwr = pd.DataFrame(rows).sort_values("rmse_ens", ascending=False).reset_index(drop=True)

    # Join EDA per-well summary if available
    eda_path = EDA / "well_summary.parquet"
    if eda_path.exists():
        eda = pd.read_parquet(eda_path)
        eda = eda[eda["split"] == "train"]
        keep_cols = [
            "well", "hidden_len", "known_len", "hidden_ratio",
            "abs_dtvtdmd_max", "d2tvtdmd2_max_abs", "n_jumps_3sigma",
            "gr_scale_offset", "gr_scale_ratio", "gr_ks_proxy",
            "cov_low_margin", "cov_high_margin", "cov_inside",
        ]
        pwr = pwr.merge(eda[keep_cols], on="well", how="left")

    pwr.to_parquet(ART / "per_well_oof.parquet", index=False)
    print(f"\nSaved per-well OOF table → artefacts/per_well_oof.parquet  ({len(pwr)} wells)")

    # Tail of distribution
    print("\nDistribution of per-well rmse_ens:")
    print(pwr["rmse_ens"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]).round(3))

    # Top-20 worst
    print("\n--- Top-20 worst wells (by ensemble RMSE) ---")
    show = ["well", "n_rows", "rmse_ens", "bias_ens"]
    if "cov_low_margin" in pwr.columns:
        show += ["cov_low_margin", "cov_high_margin", "abs_dtvtdmd_max", "hidden_ratio"]
    print(pwr[show].head(20).round(2).to_string(index=False))

    # Cumulative error contribution
    sse = (pwr["rmse_ens"] ** 2 * pwr["n_rows"]).cumsum()
    total_sse = (pwr["rmse_ens"] ** 2 * pwr["n_rows"]).sum()
    pwr["cum_sse_share"] = sse / total_sse
    print("\nCumulative SSE contribution by worst-well rank:")
    for k in (5, 10, 20, 50, 100, 200):
        if k <= len(pwr):
            share = pwr["cum_sse_share"].iloc[k - 1]
            print(f"  top-{k:>3d} worst wells account for {share*100:5.1f}% of total SSE")

    # Same-name-as-test holdout sanity
    test_names = {"000d7d20", "00bbac68", "00e12e8b"}
    if test_names.issubset(set(pwr["well"])):
        print("\n--- The 3 wells that share names with the local test set ---")
        print(pwr[pwr["well"].isin(test_names)][show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
