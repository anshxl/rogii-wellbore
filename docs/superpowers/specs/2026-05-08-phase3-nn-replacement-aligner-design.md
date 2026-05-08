# Phase 3 — Neural Replacement Aligner: Design

**Date:** 2026-05-08
**Status:** Approved for implementation
**Phase:** 3 (post-GBDT-saturation; sequence-model replacement aligner)

## Context

Phase 1/2 produced a tag-grouped XGB+CB ensemble at OOF 12.531 / public LB
10.364 (Kaggle rank #20). Phase 2's EDA + three negative-result experiments
(dtwc, distinctiveness, pruning) converged on a single diagnosis: every
hand-engineered aligner (4 beam configs, 2 PFs, 1 DTW) shares a GR-magnitude
likelihood and locks onto the same wrong stratigraphic candidate in
low-distinctiveness regions of the typewell. The GBDT cannot arbitrate when
all base aligners agree on a wrong answer, and no GBDT-level feature
addresses this — the bottleneck is the observation model itself.

Phase 3 replaces the hand-engineered aligner stack with a neural network
that learns the observation likelihood end-to-end. The well's GR is fully
observed in the eval zone (CLAUDE.md leakage rules permit this), so the
problem is **sequence labeling with cross-attention**, not autoregressive
forecasting.

## Goals & success criteria

**Goal:** Replace the alignment stack with a learned observation model;
produce per-row TVT predictions that the GBDT pipeline can ensemble against.

**Success thresholds (in order of strictness):**

| Threshold | Standalone OOF | Action |
|---|---|---|
| Floor (architecture not broken) | < 14.0 | Continue iterating |
| Aspirational standalone | beat XGB-only OOF 12.612 | Validates structural-replacement thesis |
| Aspirational ensemble (NN-pair) | beat current best 12.531 | Move to NN+GBDT integration |
| Bail-out to plan C | both encoders > 13.5 after one regularization round | Pivot to feature-extractor mode |

**Submission policy:** every full-CV milestone produces `submission.csv` as a
deliverable artefact regardless of OOF. Whether to upload to Kaggle is a
manual per-artefact decision; sample variance on the 52-well public LB is
±~1 RMSE so submissions are spent only on changes with material OOF
movement.

**Out of scope for Phase 3:** hyperparameter tuning beyond a small grid;
multi-task learning; PNG-based aux signals; learned-aligner-as-feature-bank
for GBDTs (that is plan C if we need it).

## Architecture

```
                    ┌─────────────────────────┐
   well sequence →  │  ENCODER (CNN | xfmr)   │ → H_well  [L_well, d]
   (per-row inputs) └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Cross-attention block   │ ← K, V from typewell
                    │ (queries = H_well)      │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
   typewell rows →  │ TYPEWELL ENCODER (1D    │ → H_tw   [L_tw, d]
   (GR, TVT, geo)   │  CNN, small)            │
                    └─────────────────────────┘    (used as K/V)
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   per-row TVT head      │ → ŷ  [L_well]
                    │   (2-layer MLP)         │
                    └─────────────────────────┘
```

**Encoders (variable across the comparison):**

- **CNN-TCN encoder.** Stack of dilated Conv1d blocks (kernel 3; dilations
  1, 2, 4, 8, 16, 32; ~6 blocks; channels 128). Receptive field ≈190 rows
  on the MD-axis. ~300k params.
- **Transformer encoder.** 4 layers, 4 heads, d_model=128, FFN=256, RoPE
  positional encoding on MD-position. ~600k params. Attention masking
  handles variable lengths.

**Shared (constant across both encoders):**

- **Typewell encoder.** Small dilated CNN (3 blocks, d=128). Encodes
  typewell rows into a K/V bank for cross-attention.
- **Cross-attention.** 2 stacked blocks, multihead (4 heads), residual +
  LayerNorm. Queries from well encoder; K/V from typewell encoder.
- **Output head.** 2-layer MLP, GELU, dropout 0.1 → scalar TVT per row.

**Total budget.** ≤1M params either variant. Comfortably fits T4 with
batch sizes 8–32.

**Why this shape.** Cross-attention encodes the inductive bias that "TVT
comes from somewhere in the typewell". The encoder builds a per-row query
representation; the decoder finds the matching typewell region. CNN tests
"is local context enough"; Transformer tests "do we need global
self-attention". Both are the same size class so the comparison is fair.

## Inputs and preprocessing

### Per-row well inputs (12 features)

Computed once per well at data-loading time:

| Feature | Definition | Why |
|---|---|---|
| `GR` | z-scored per-well using own GR mean/std | Per-well GR-normalization, free here |
| `MD_norm` | `(MD - MD_min) / (MD_max - MD_min)` | Scale-free MD position |
| `dMD` | `MD - MD_prev`, normalized by per-well median dMD | Captures non-uniform sampling |
| `Z_norm` | z-scored Z (TVD) per well | Absolute-depth anchor |
| `dZ` | first-diff of Z, z-scored | Vertical motion rate |
| `X_norm` | z-scored X per well | Weak spatial signal |
| `Y_norm` | z-scored Y per well | Weak spatial signal |
| `TVT_input_filled` | TVT_input with NaNs replaced by `last_known_TVT` | Prefix anchor |
| `is_known_mask` | 1 in prefix, 0 in hidden zone | Critical: tells model which rows are anchor vs target |
| `dz_dmd` | z_diff per MD step (dipping rate) | Geologically meaningful, hard for small encoder to discover |
| `dx_dmd` | x_diff per MD step (lateral drift) | Same |
| `dy_dmd` | y_diff per MD step | Same |

**Deliberate exclusions:** all `gr_roll{N}`, `gr_std{N}`, `gr_lag/lead{N}`,
`gr_grad`, `gr_cumsum` — encoder is meant to learn these from raw GR;
pre-computing imposes fixed-window inductive bias. All Phase 1 prefix-summary
scalars (`prefix_gr_*`, `prefix_tw_*`, `prefix_tvt_*`, `known_tvt_*`) —
derivable from prefix rows. All aligner outputs and disagreements (`beam_*`,
`pf_*`, `ancc_*`, `dtw_*`, `*_diff`, `tw_uniq_*`) — excluded by the
"replacement aligner = pure raw" choice. All `tw_diff_{±N}` fixed-offset
similarities — cross-attention does this end-to-end, in learned form.

### Per-row typewell inputs (8 features)

| Feature | Definition | Why |
|---|---|---|
| `GR_tw` | z-scored against the *same well's* GR statistics | Cross-well normalization |
| `TVT_tw` | z-scored against per-well typewell TVT statistics | Position in typewell |
| `Geology_onehot` | 6-way: `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA` | Free signal; Phase 2 showed it can't disambiguate worst wells but doesn't hurt elsewhere |

### Leakage compliance

- Train-only horizontal columns (`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`,
  `TVT`) never enter inputs.
- `TVT_input` is leakage-safe by construction (NaN in eval zone).
- Typewell `Geology` is permitted by CLAUDE.md.
- GR is fully observed everywhere (including eval zone) by problem
  definition; the encoder reads the entire well GR sequence per row.

### Variable-length handling

Per-batch padding to max-length-in-batch with attention masks. Wells top out
~2000 rows; typewells ~3000. Both well within T4 memory at d=128.

## Targets, loss, training

### Target

Per-row absolute TVT. Loss applied only on rows where the *current epoch's*
`is_known_mask = 0` (the simulated hidden zone). Known-prefix rows are
inputs, not targets.

### Loss

Standard MSE on per-row TVT, averaged over hidden-zone rows. Competition
metric is RMSE; train against it directly. No Huber / no tail-weighting in
v1 — add only if training instability or right-tail divergence is observed.

### Random prefix-length augmentation (mandatory)

For each training well on each epoch:

1. Sample a prefix-end MD-fraction `p ~ Uniform(0.10, 0.90)`.
2. Build the augmented input mask: `is_known_aug[i] = 1` for rows with
   `MD_norm[i] ≤ p`, else `0`.
3. Compute the augmented prefix anchor: `lkt_aug` = `TVT` at the last row
   with `is_known_aug = 1`.
4. Per-row inputs:
   - `is_known_mask[i]` ← `is_known_aug[i]`
   - `TVT_input_filled[i]` ← `TVT[i]` if `is_known_aug[i] = 1`, else
     `lkt_aug`
5. Per-row target: `TVT[i]` at rows where `is_known_aug[i] = 0`. Loss
   contributes only on these rows.

Note: feeding ground-truth `TVT` as input at extended-prefix rows is not
leakage — those rows are inputs, not targets, on this epoch's draw of `p`.

**Validation uses the natural prefix split** (the well's actual `TVT_input`
mask) so OOF is comparable to the GBDT pipeline.

### Other augmentations (start off, add only if overfitting)

- GR Gaussian noise σ=0.05 (z-scored units), well rows only.
- Random typewell-row dropout 5%.
- MD-axis stretch ±5% (kept off in v1).

### Optimizer / schedule

- AdamW, lr 3e-4, weight_decay 1e-4, betas default.
- Cosine schedule with 5% warmup, ~50 epochs total.
- Early-stop on validation OOF, patience 10.
- Gradient clipping at norm 1.0.
- Mixed precision (fp16/bf16 on T4).
- Per-fold checkpoint saved at best validation.

### Batching

Group wells by length to minimize pad waste; effective batch size 16 wells.
Each well is one example; random-prefix augmentation gives ~50 effective
training examples per well over 50 epochs.

### Reproducibility

`SEED=42` per CLAUDE.md. Random-prefix augmentation introduces meaningful
epoch-to-epoch variance even with seeded RNG; expect OOF noise larger than
GBDT's ~0.05.

### Per-fold training time estimate

~50 epochs × ~620 wells × forward+backward, T4: order of 10–30 min per
fold per encoder. 5 folds × 2 encoders ≈ 2–5 hours per full CV run.

## CV scheme, evaluation, comparison protocol

### CV scheme

Same tag-grouped 5-fold splits as Phase 1/2 (union-find over typewell-md5 +
DBSCAN pad-cluster, 251 groups). Reuse `assign_groups` from `src/baseline.py`.
**Same fold assignments as the GBDT pipeline** so per-well OOF is directly
comparable.

### Per-encoder OOF

For each encoder (CNN, Transformer):

- 5 fold-models trained, each predicts on its held-out fold using the
  *natural* prefix split.
- Stitch into a single OOF parquet matching the GBDT format:
  `well, prediction_id, row_idx, fold, target, pred_<encoder>`.
- Save to `artefacts/nn/<encoder>/oof_predictions.parquet`.

### Comparison metrics (against existing GBDT OOF)

- Row-level RMSE (primary; what LB measures).
- Per-well RMSE distribution: median, p90, p99, max.
- Worst-20-wells overlap with the GBDT ensemble's worst-20.
- `bias_ens` per-well for the worst wells — does the NN escape the
  "all aligners agree on a wrong layer" signature?

### Decision points

| Outcome | Action |
|---|---|
| Both encoders OOF > 13.5 after one regularization round | Bail to plan C (feature extractor) |
| One encoder ≤ 12.6, other > 13.5 | Use the working one; drop the broken one |
| Both ≤ 12.6, residual correlation < 0.7 | Strong ensemble candidate — proceed to layer 1 |
| Both ≤ 12.6, residual correlation > 0.9 | Ensemble redundant; pick the better one |

## Ensembling strategy

**Layer 1 — NN-pair (CNN + Transformer).** Stack the two NN OOF columns;
Nelder-Mead on weights to minimize OOF RMSE. Same procedure as XGB+CB.
If NM gives one model weight 0, drop it.

**Layer 2 — NN + GBDT integration.** Two strategies, in order:

1. **NM weights over 4 models** (XGB, CB, CNN, Transformer). Re-run NM on
   the 4-column OOF stack. Submit if it beats 12.531.
2. **Stacking.** Train a small ridge or shallow LGB (depth 2) on the 4 OOF
   columns + a few cheap meta-features (prefix length, `prefix_tw_rmse`,
   well length). Only worth doing if NM leaves material OOF on the table.

**Not doing in Phase 3:** training the NN on GBDT residuals. Phase 2 dtwc
rule-out: GBDTs already have all the alignment-output signal; the point is
to escape that subspace, not extend it.

## Code organization

```
src/
  nn/
    __init__.py
    data.py              # well/typewell loaders, augmentation, batching
    encoders.py          # CNN-TCN encoder, Transformer encoder
    decoder.py           # shared cross-attention + head
    model.py             # full model class, swappable encoder
    train.py             # CV loop, optimizer, scheduler, checkpointing
    predict.py           # OOF and test-time inference
    cli.py               # mode=tune|cv|train|predict, env-var driven
artefacts/nn/
  cnn/
    fold_models/{1..5}.pt
    oof_predictions.parquet
    metrics.json
  transformer/
    fold_models/{1..5}.pt
    oof_predictions.parquet
    metrics.json
  ensemble_weights.json   # NM weights over {xgb, cb, cnn, transformer}
notebooks/
  nn_phase3_eda.ipynb     # comparison plots, residual analysis
```

### Reuse from `src/baseline.py`

- `assign_groups` (CV grouping).
- Path / env-var conventions (`DATA_DIR`, `ARTEFACT_DIR`, `OUTPUT_DIR`).
- Logging idiom (`get_logger`, `timer`, `section`).
- OOF parquet schema.

### Submission notebook

`notebooks/submit_nn_ensemble.ipynb`. Loads NN checkpoints + XGB/CB models
from a Kaggle dataset bundle; emits one CSV per submission strategy. Run
after each full-CV milestone; uploading to Kaggle is a manual decision.

## Where it runs

| Stage | Where | Why |
|---|---|---|
| Code editing, lint, type-check | Local Mac | Fast iteration |
| Smoke tests (5 wells, 1 fold, CPU) | Local Mac (CPU) | Verify correctness, no GPU needed |
| Full CV runs | Kaggle T4 notebook | The actual training |
| Submission inference | Kaggle T4 notebook | Same as Phase 1/2 |

### Kaggle dataset bundle

- `wellbore-prediction-code` — private dataset, ZIP of `src/`. Re-uploaded
  per code change.
- `wellbore-prediction-train-cache` — `train_df.parquet`,
  `tag_groups.parquet`, per-well dicts. Already exists.
- `wellbore-prediction-nn-checkpoints` — new. Per-fold `.pt` files
  consumed by the submission notebook.

### Compute budget

Kaggle free tier: 30 GPU-hours/week, dual T4 available, 9-hour session cap.
Full CV ≈ 2–5 hours per run. Budget ≈ 6–10 full runs per week.

### Resume-from-checkpoint

Per-fold checkpointing must support resumption to handle the 9-hour
session cap. The training CLI must accept a `RESUME_FROM_FOLD` env var
that picks up at fold `N` and reuses prior `.pt` files for folds `< N`.

### Memory pre-loading

Random-prefix augmentation regenerates batches per epoch; pre-load all 773
wells into RAM at notebook start (tens of MB; trivial for T4 RAM).

## Phase 3 milestones

| # | Deliverable | Goal |
|---|---|---|
| M1 | Data pipeline + dummy model (MLP-on-pooled-features) | Verify shapes, masking, leakage; establish pipeline floor (~OOF 13–14) |
| M2 | CNN encoder, fold 0 only | Validate convergence; iterate on lr / depth |
| M3 | CNN full 5-fold OOF + submission.csv | First real CNN number; comparison to GBDT |
| M4 | Transformer encoder, fold 0 only | Same drill |
| M5 | Transformer full 5-fold OOF + submission.csv | Decision point on dropping one |
| M6 | NN-pair NM ensemble + comparison report + submission.csv | Per-well diff vs GBDT, worst-20 overlap |
| M7 | NN+GBDT NM integration + submission.csv | Final Phase 3 deliverable; manual call on uploading |

Each milestone gets a JOURNAL.md entry per CLAUDE.md conventions.

**Implementation order:** session 1 — M1+M2; session 2 — M3+M4; session 3 —
M5+M6+M7. Each session ends on a deliverable.

## Open questions deferred to implementation

- Exact handling of wells whose hidden zone exits the typewell TVT range (13
  wells per Phase 1 EDA; 10 share a typewell). Cross-attention will
  attempt to align outside the K/V range; behavior will be observed and
  patched at M2 if it causes catastrophic failures.
- Whether `dz_dmd`/`dx_dmd`/`dy_dmd` need their own normalization or can
  ride on Z/X/Y per-well z-scoring. Decide at M1 from training stability.
- Loss spike on bad-well batches → switch MSE to Huber with generous δ.
  Defer the decision to first observed run.
