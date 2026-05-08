# Kaggle submission workflow

Two notebooks in this directory submit our two best pipelines:
- `submit_ensemble.ipynb` — XGB + CB ensemble with DTW features (OOF 12.531).
- `submit_lgb_optuna.ipynb` — Tuned LightGBM single model with DTW features (OOF 12.640).

Both load locally-trained models and run inference on the Kaggle hidden test set. They do **not** retrain on Kaggle — that happens locally so we get reproducible model files.

## One-time local setup

After running CV (which we have), produce the trained model artefacts:

```bash
# 1. XGB + CB final fit on full data (uses cached best_iters from CV)
unset VIRTUAL_ENV
BASELINE_MODE=finalize_only uv run python src/baseline.py

# 2. Tuned LGB final fit on full data (uses cached best_params + best_iters)
BASELINE_MODE=train uv run python src/baseline_lgb.py
```

Estimated time: ~10–15 min total. Each writes its own model files; no overlap.

After both runs, you should have:

```
artefacts/
  features.json
  best_iters.json                 # for xgb + cb
  ensemble_weights.json           # NM-optimal cb/xgb weights
  final_xgb.json                  # XGBRegressor.save_model() output
  final_cb.cbm                    # CatBoostRegressor.save_model() output
  lgb_optuna/
    best_params.json
    best_iters.json
    final_lgb.txt                 # LGB Booster.save_model() output
```

## Bundle the artefacts as a Kaggle dataset

Put together a private dataset containing **only** the files the notebooks need (don't ship `train_df.parquet` — it's 1 GB and only useful for retraining):

```
wellbore-baseline-bundle/
  src/baseline.py
  artefacts/features.json
  artefacts/best_iters.json
  artefacts/ensemble_weights.json
  artefacts/final_xgb.json
  artefacts/final_cb.cbm
  artefacts/lgb_optuna/best_params.json
  artefacts/lgb_optuna/final_lgb.txt
```

Total size should be well under 100 MB.

Upload via `kaggle datasets create` or the web UI. If you change the dataset slug, update `BUNDLE = Path('/kaggle/input/<slug>')` at the top of each notebook.

## On Kaggle

For each notebook:

1. Create a new notebook on the competition page (so `rogii-wellbore-geology-prediction` is auto-attached as input).
2. Add your `wellbore-baseline-bundle` dataset as an additional input.
3. Upload the `.ipynb` file (or paste cells).
4. Run all. Each notebook will:
   - Stage artefacts to `/kaggle/working/artefacts/` (writable).
   - Build `test_df` from competition test wells (~3–5 min for FE).
   - Load models, predict, write `/kaggle/working/submission.csv`.
5. Submit via the Kaggle UI.

The notebooks also save `test_df.parquet` to `/kaggle/working/`. The LGB notebook will reuse it if it's already there in the same session, so you can run both notebooks back-to-back without re-running FE.

## Sanity checks before submitting

The notebooks already check:
- All feature columns present in `test_df`.
- All `prediction_id`s in `sample_submission.csv` map to a `test_df` row (`build_submission` raises if not).
- TVT prediction range falls in a sensible band.

If FE fails on a single well, the dataset builder logs a warning and skips it. **If that happens, the submission build will raise** because some `prediction_id`s won't have predictions. Fix the underlying issue rather than masking it — silent zero-fills score 0 on those rows.
