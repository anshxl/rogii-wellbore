"""LGB-only fork with Optuna hyperparameter tuning.

Reuses the feature engineering pipeline from `src/baseline.py` (build_dataset,
assign_groups, ArtifactManager, etc.) and replaces the XGB+CB ensemble with a
single tuned LightGBM model.

Why this exists:
- Single-model LGB iteration is ~3× faster than the XGB+CB ensemble per CV
  run, which is useful for fast feature ablations.
- Optuna study reveals which LGB hyperparameters matter on this data; the
  insight transfers to XGB/CB if needed.

Caveat: In the default-params 3-model ensemble run on 2026-05-08, LGB had
Nelder-Mead weight 0.000 (fully redundant with XGB+CB). Tuned-LGB-alone is
unlikely to match the 2-model ensemble OOF, but may close the gap and add
ensemble diversity if reintroduced into a 3-model setup.

Modes (env var BASELINE_MODE):
  cv     — 5-fold tag-grouped CV with current best params (or defaults).
  tune   — Optuna study, save best params, then run CV with them.
  train  — load best params, retrain on full data, build submission.

Env vars (all optional):
  BASELINE_MODE     {cv, tune, train}                default: tune
  N_TRIALS          int                              default: 30
  TUNE_TIMEOUT      seconds (None disables)          default: 3600
  STUDY_NAME        string                           default: lgb_optuna_v1
  DEBUG_MAX_WELLS   int                              default: None (use all)
  DEBUG_ONE_FOLD    "0" / "1"                        default: "0"
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

# Reuse FE + helpers from the main baseline. Importing has side-effects
# (it prints "X loaded." lines and instantiates the main artefacts manager),
# but does not run the dispatch — that's gated by `if __name__ == "__main__"`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline as bl  # noqa: E402

import lightgbm as lgb  # noqa: E402
from lightgbm import LGBMRegressor, early_stopping, log_evaluation  # noqa: E402

try:
    import optuna  # noqa: E402
    from optuna.samplers import TPESampler  # noqa: E402
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


# ── Config ────────────────────────────────────────────────────────────────────
MODE          = os.environ.get("BASELINE_MODE", "tune")
N_TRIALS      = int(os.environ.get("N_TRIALS", "30"))
_t            = os.environ.get("TUNE_TIMEOUT", "3600")
TUNE_TIMEOUT  = None if _t in ("0", "none", "None", "") else int(_t)
STUDY_NAME    = os.environ.get("STUDY_NAME", "lgb_optuna_v1")
DEBUG_MAX_WELLS = int(os.environ["DEBUG_MAX_WELLS"]) if os.environ.get("DEBUG_MAX_WELLS") else None
DEBUG_ONE_FOLD  = os.environ.get("DEBUG_ONE_FOLD", "0") == "1"

# Fast-tuning mode: subsample groups, fewer folds, lower n_estimators cap,
# narrower LR range. The subsample preserves the tag-grouped CV semantics
# (groups are kept whole). Final CV after tuning always runs on full data
# at full N_SPLITS with the full n_estimators cap.
TUNE_FAST         = os.environ.get("TUNE_FAST", "1") == "1"
TUNE_SAMPLE_FRAC  = float(os.environ.get("TUNE_SAMPLE_FRAC", "0.30"))
TUNE_N_SPLITS     = int(os.environ.get("TUNE_N_SPLITS", "3"))
TUNE_N_EST_CAP    = int(os.environ.get("TUNE_N_EST_CAP", "3000"))
TUNE_EARLY_STOP   = int(os.environ.get("TUNE_EARLY_STOP", "100"))
TUNE_LR_MIN       = float(os.environ.get("TUNE_LR_MIN", "0.02"))
TUNE_LR_MAX       = float(os.environ.get("TUNE_LR_MAX", "0.10"))

SEED       = bl.SEED
N_SPLITS   = bl.N_SPLITS

# Separate artefacts dir so we don't clobber the main pipeline's outputs.
LGB_DIR = bl.REPO_ROOT / "artefacts" / "lgb_optuna"
LGB_DIR.mkdir(parents=True, exist_ok=True)
am_lgb = bl.ArtifactManager(LGB_DIR)

# Train_df + features cache from the main pipeline (same FE).
am_main = bl.am

log = bl.get_logger("lgb_optuna")


# ── Default LGB params (used until Optuna finds something better) ─────────────
DEFAULT_LGB_PARAMS: Dict[str, Any] = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l1=0.0,
    lambda_l2=0.0,
    max_depth=-1,
    n_estimators=10_000,
    early_stopping_rounds=200,
    verbose=-1,
    random_state=SEED,
    n_jobs=-1,
)


# ── Train/feature cache ───────────────────────────────────────────────────────
def _get_or_build_train_df() -> Tuple[pd.DataFrame, List[str]]:
    """Reuse the main pipeline's cached train_df. Build if missing."""
    try:
        log.info("Loading cached train dataset from main pipeline …")
        train_df     = am_main.load_df(am_main.TRAIN_DF)
        train_df     = bl.assign_groups(train_df)
        feature_cols = am_main.load_json(am_main.FEATURES_LIST)
        return train_df, feature_cols
    except FileNotFoundError:
        pass
    log.info("Cache miss — building train features (this is the slow path) …")
    with bl.timer(log, "train feature engineering"):
        train_df = bl.build_dataset(bl.TRAIN_DIR, is_train=True,
                                    max_wells=DEBUG_MAX_WELLS)
    train_df = bl.assign_groups(train_df)
    am_main.save_df(train_df, am_main.TRAIN_DF)
    feature_cols = bl.get_feature_columns(train_df)
    am_main.save_json(feature_cols, am_main.FEATURES_LIST)
    return train_df, feature_cols


# ── CV runner (single LGB model) ──────────────────────────────────────────────
def run_cv_lgb(
    train_df:     pd.DataFrame,
    feature_cols: List[str],
    params:       Dict[str, Any],
    one_fold:     bool = False,
    verbose:      bool = True,
) -> Tuple[np.ndarray, List[int], pd.DataFrame]:
    """5-fold tag-grouped CV with LGB. Returns (oof, best_iters, fold_metrics_df)."""
    y      = train_df["target"].to_numpy()
    groups = train_df["group_id"].to_numpy()
    n      = len(train_df)

    cv     = GroupKFold(n_splits=N_SPLITS)
    splits = list(cv.split(train_df, y, groups=groups))
    if one_fold:
        splits = splits[:1]
        if verbose:
            log.info("DEBUG: running only fold 0")

    oof         = np.zeros(n, dtype=np.float64)
    best_iters  = []
    fold_scores = []
    fold_ids    = []

    if verbose:
        bl.section(log, f"CV  ({len(splits)}-fold GroupKFold, LGB)")

    for fold, (tr_idx, va_idx) in enumerate(splits):
        Xtr = train_df.iloc[tr_idx][feature_cols]
        ytr = y[tr_idx]
        Xva = train_df.iloc[va_idx][feature_cols]
        yva = y[va_idx]

        if verbose:
            w_tr = train_df.iloc[tr_idx]["well"].nunique()
            w_va = train_df.iloc[va_idx]["well"].nunique()
            log.info(f"\n── Fold {fold+1}/{len(splits)}  "
                     f"(train={len(tr_idx):,}  val={len(va_idx):,}  "
                     f"wells_tr={w_tr}  wells_va={w_va}) ──")
        fold_ids.append(fold + 1)

        m = LGBMRegressor(**params)
        cb = [early_stopping(stopping_rounds=params.get("early_stopping_rounds", 200), verbose=False)]
        if verbose:
            cb.append(log_evaluation(period=200))
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=cb)

        preds = m.predict(Xva)
        oof[va_idx] = preds
        best_iters.append(int(m.best_iteration_))
        rmse = root_mean_squared_error(yva, preds)
        fold_scores.append(rmse)
        if verbose:
            log.info(f"  LGB  iter={int(m.best_iteration_):5d}  fold_RMSE={rmse:.5f}")

    fm_df = pd.DataFrame(
        {"lgb": fold_scores + [float(np.mean(fold_scores)), float(np.std(fold_scores))]},
        index=[f"fold_{i}" for i in fold_ids] + ["mean", "std"],
    )

    if verbose:
        bl.section(log, "CV SUMMARY")
        try:
            log.info("\n" + fm_df.to_markdown(floatfmt=".5f"))
        except Exception:
            log.info("\n" + fm_df.to_string(float_format=lambda v: f"{v:.5f}"))
        log.info(f"  OOF RMSE  LGB = {root_mean_squared_error(y, oof):.5f}")

    return oof, best_iters, fm_df


# ── Optuna search space + objective ───────────────────────────────────────────
def _suggest_params(trial: "optuna.trial.Trial", fast: bool = TUNE_FAST) -> Dict[str, Any]:
    """LGB hyperparameter search space.

    `fast=True` narrows the LR range and lowers n_estimators to fit a 30-trial
    study into ~1-2 hours. The relative ranking of param combinations under
    these caps usually transfers to full-budget training.
    """
    lr_min     = TUNE_LR_MIN     if fast else 0.005
    lr_max     = TUNE_LR_MAX     if fast else 0.1
    n_est      = TUNE_N_EST_CAP  if fast else 10_000
    early_stop = TUNE_EARLY_STOP if fast else 200
    return dict(
        objective="regression",
        metric="rmse",
        learning_rate=trial.suggest_float("learning_rate", lr_min, lr_max, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 256, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 200, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 0, 10),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        max_depth=trial.suggest_int("max_depth", -1, 16),
        min_gain_to_split=trial.suggest_float("min_gain_to_split", 1e-8, 1.0, log=True),
        n_estimators=n_est,
        early_stopping_rounds=early_stop,
        verbose=-1,
        random_state=SEED,
        n_jobs=-1,
    )


def _subsample_by_group(train_df: pd.DataFrame, frac: float, seed: int = SEED) -> pd.DataFrame:
    """Sample whole groups (all rows of group_id stay together)."""
    rng = np.random.default_rng(seed)
    all_groups = train_df["group_id"].unique()
    n_keep = max(1, int(round(len(all_groups) * frac)))
    keep = rng.choice(all_groups, size=n_keep, replace=False)
    sub = train_df[train_df["group_id"].isin(keep)].reset_index(drop=True)
    return sub


def _make_objective(train_df: pd.DataFrame, feature_cols: List[str]):
    """Returns an Optuna objective: K-fold CV mean RMSE on (optionally subsampled) data.

    In fast-tuning mode (`TUNE_FAST=1`), the objective uses
    `TUNE_N_SPLITS`-fold CV on a `TUNE_SAMPLE_FRAC` group-subsample of
    `train_df`. Final CV after the study runs separately on full data.
    """
    if TUNE_FAST:
        sub = _subsample_by_group(train_df, TUNE_SAMPLE_FRAC)
        n_splits = TUNE_N_SPLITS
        log.info(f"  fast tuning: subsampled to {len(sub):,} rows  "
                 f"({sub['well'].nunique()} wells, {sub['group_id'].nunique()} groups)  "
                 f"using {n_splits}-fold")
    else:
        sub = train_df
        n_splits = N_SPLITS
        log.info(f"  full-data tuning: {len(sub):,} rows  using {n_splits}-fold")

    y      = sub["target"].to_numpy()
    groups = sub["group_id"].to_numpy()
    cv     = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(sub, y, groups=groups))
    log_folds = os.environ.get("LOG_FOLDS", "1") == "1"

    def objective(trial: "optuna.trial.Trial") -> float:
        params = _suggest_params(trial)
        early_stop = params.get("early_stopping_rounds", 200)
        scores = []
        t0 = time.perf_counter()
        for fold, (tr_idx, va_idx) in enumerate(splits):
            Xtr = sub.iloc[tr_idx][feature_cols]
            Xva = sub.iloc[va_idx][feature_cols]
            ytr = y[tr_idx]
            yva = y[va_idx]

            m = LGBMRegressor(**params)
            m.fit(
                Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[early_stopping(stopping_rounds=early_stop, verbose=False)],
            )
            preds = m.predict(Xva)
            rmse  = root_mean_squared_error(yva, preds)
            scores.append(rmse)

            if log_folds:
                log.info(f"  trial {trial.number:3d}  fold {fold+1}/{n_splits}: "
                         f"rmse={rmse:.5f}  iter={int(m.best_iteration_):5d}  "
                         f"running_mean={float(np.mean(scores)):.5f}  "
                         f"elapsed={time.perf_counter()-t0:.0f}s")

            # Report after each fold so MedianPruner can prune slow trials.
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                if log_folds:
                    log.info(f"  trial {trial.number:3d}  PRUNED at fold {fold+1}")
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    return objective


def _log_trial_done(study: "optuna.Study", trial: "optuna.trial.FrozenTrial"):
    """Per-trial summary callback. Runs after each completed/pruned trial."""
    state = trial.state.name
    val   = f"{trial.value:.5f}" if trial.value is not None else "—"
    best  = f"{study.best_value:.5f}" if study.best_trial else "—"
    pretty_params = {k: (round(v, 5) if isinstance(v, float) else v)
                     for k, v in trial.params.items()}
    dur = trial.duration.total_seconds() if trial.duration else 0.0
    log.info(f"trial {trial.number:3d} {state:8s}  value={val}  best={best}  "
             f"({dur:.0f}s)  params={pretty_params}")


def run_optuna(
    train_df:     pd.DataFrame,
    feature_cols: List[str],
    n_trials:     int = N_TRIALS,
    timeout:      Optional[int] = TUNE_TIMEOUT,
    study_name:   str = STUDY_NAME,
) -> Dict[str, Any]:
    """Run an Optuna study; return best params merged with fixed defaults."""
    if not _HAS_OPTUNA:
        raise RuntimeError("optuna is not installed. Run: uv add optuna")

    storage_path = LGB_DIR / f"{study_name}.db"
    storage_url  = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    bl.section(log, f"OPTUNA  (study={study_name}  n_trials={n_trials}  timeout={timeout}s)")
    log.info(f"  storage: {storage_url}")
    log.info(f"  existing trials: {len(study.trials)}")

    objective = _make_objective(train_df, feature_cols)
    # Keep optuna's INFO logs (trial-finished messages) and add our own
    # per-trial callback for richer one-line summaries.
    optuna.logging.set_verbosity(optuna.logging.INFO)
    t0 = time.perf_counter()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        gc_after_trial=True,
        catch=(),
        callbacks=[_log_trial_done],
    )
    elapsed = time.perf_counter() - t0

    log.info(f"\nFinished in {elapsed:.0f}s. Total trials in study: {len(study.trials)}")
    log.info(f"Best value (CV RMSE): {study.best_value:.5f}")
    log.info("Best params:")
    for k, v in study.best_params.items():
        log.info(f"  {k:>22s} = {v}")

    # Merge tuned params with fixed bookkeeping fields.
    best = dict(DEFAULT_LGB_PARAMS)
    best.update(study.best_params)
    am_lgb.save_json(best, "best_params")

    # Save lightweight trials summary too.
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "duration", "params"))
    am_lgb.save_df(trials_df, "trials")
    return best


# ── Final-train (full data) for submission ────────────────────────────────────
def run_final_lgb(train_df: pd.DataFrame, feature_cols: List[str],
                  params: Dict[str, Any], best_iters: List[int]) -> LGBMRegressor:
    n_iters = max(50, int(round(np.mean(best_iters) * 1.10)))
    p = dict(params)
    p.pop("early_stopping_rounds", None)
    p["n_estimators"] = n_iters

    bl.section(log, "FINAL TRAINING — FULL DATA  (LGB)")
    log.info(f"  n_iters={n_iters} (mean best_iter × 1.10)")
    X = train_df[feature_cols]
    y = train_df["target"]
    m = LGBMRegressor(**p)
    m.fit(X, y)
    booster_path = LGB_DIR / "final_lgb.txt"
    m.booster_.save_model(str(booster_path))
    log.info(f"  ✔ saved {booster_path}")

    # Feature importance
    try:
        imps = m.feature_importances_
    except Exception:
        imps = m.booster_.feature_importance(importance_type="gain")
    fi_df = (pd.DataFrame({"feature": feature_cols, "importance": imps})
             .sort_values("importance", ascending=False).reset_index(drop=True))
    am_lgb.save_df(fi_df, "feature_importance")
    log.info("\nTop-25 features (LGB gain):")
    log.info("\n" + fi_df.head(25).to_string(index=False))
    return m


# ── Main dispatch ─────────────────────────────────────────────────────────────
def _dispatch():
    bl.section(log, f"LGB-Optuna  |  MODE={MODE}  |  N_TRIALS={N_TRIALS}  |  STUDY={STUDY_NAME}")
    log.info(f"Main artefacts : {bl.ARTEFACT_DIR}")
    log.info(f"LGB artefacts  : {LGB_DIR}")
    np.random.seed(SEED)

    train_df, feature_cols = _get_or_build_train_df()
    log.info(f"train_df: {train_df.shape}  features: {len(feature_cols)}")

    if MODE == "tune":
        best = run_optuna(train_df, feature_cols)
        # Final CV on full data uses full early-stopping budget regardless of
        # tuning caps (the search-space n_estimators/early_stop were chosen
        # for trial speed, not final-fit quality).
        final_params = dict(best)
        final_params["n_estimators"] = 10_000
        final_params["early_stopping_rounds"] = 200
        log.info("Running final CV on full data with full-budget early stopping …")
        oof, best_iters, fm_df = run_cv_lgb(
            train_df, feature_cols, final_params, one_fold=DEBUG_ONE_FOLD,
        )
        oof_df = train_df[["well", "prediction_id", "target"]].copy()
        oof_df["oof_lgb"] = oof
        am_lgb.save_df(oof_df, "oof_predictions")
        am_lgb.save_json({"lgb": best_iters}, "best_iters")
        am_lgb.save_df(fm_df.reset_index(), "fold_metrics")
        log.info("Tune + CV complete.")

    elif MODE == "cv":
        try:
            params = am_lgb.load_json("best_params")
            log.info("Loaded best_params from previous tune run.")
        except FileNotFoundError:
            log.info("No best_params found — using DEFAULT_LGB_PARAMS.")
            params = DEFAULT_LGB_PARAMS
        oof, best_iters, fm_df = run_cv_lgb(
            train_df, feature_cols, params, one_fold=DEBUG_ONE_FOLD,
        )
        oof_df = train_df[["well", "prediction_id", "target"]].copy()
        oof_df["oof_lgb"] = oof
        am_lgb.save_df(oof_df, "oof_predictions")
        am_lgb.save_json({"lgb": best_iters}, "best_iters")
        am_lgb.save_df(fm_df.reset_index(), "fold_metrics")

    elif MODE == "train":
        params = am_lgb.load_json("best_params")
        bi     = am_lgb.load_json("best_iters")["lgb"]
        m      = run_final_lgb(train_df, feature_cols, params, bi)

        # Submission path — needs the test_df cached by the main pipeline.
        try:
            test_df = am_main.load_df(am_main.TEST_DF)
        except FileNotFoundError:
            log.warning("test_df not cached. Run the main pipeline once with "
                        "MODE=train (or features_only) to build it. Skipping submission.")
            return

        delta_pred = m.predict(test_df[feature_cols])
        abs_tvt    = delta_pred.astype(np.float64) + test_df["last_known_tvt"].to_numpy()
        sample = pd.read_csv(bl.DATA_DIR / "sample_submission.csv")
        bl.build_submission(test_df, abs_tvt, sample, bl.OUTPUT_DIR / "submission_lgb_optuna.csv")
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    _dispatch()
