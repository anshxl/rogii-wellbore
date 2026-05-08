# JOURNAL

Chronological experiment log. Append-only, oldest at top. See `CLAUDE.md` for
the entry template and conventions.

## 2026-05-07 — EDA Phase 1: inventory, distributions, leakage audit

**What:** Built a single-pass EDA pipeline at
[notebooks/eda_pipeline.py](notebooks/eda_pipeline.py) that loops once over all
wells, computes per-well aggregates, and caches them at
[eda_outputs/well_summary.parquet](eda_outputs/well_summary.parquet) (776 rows
× 59 cols). Driven by the notebook
[notebooks/eda_phase1.ipynb](notebooks/eda_phase1.ipynb), which renders all
plots inline and writes them to [eda_outputs/figs/](eda_outputs/figs/). Full
write-up in [eda_findings.md](eda_findings.md).

**Findings:**
- **Local data:** 773 train wells (each with horizontal + typewell + PNG) and
  only 3 test wells (no PNGs). The 3 "test" files are byte-identical to the
  3 same-named train files in MD/X/Y/Z/GR with the same TVT_input mask
  position — the test files just strip TVT, the six formation-top columns
  (`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`), and the typewell `Geology`
  column. Treat the local test as a smoke sandbox, not the official held-out
  set; CLAUDE.md's "~200 wellbores" figure refers to the official Kaggle test.
- **Hidden zone:** every train well has a hidden zone. `hidden_ratio` is
  tightly distributed (median 0.74, IQR 0.70–0.78, full range 0.20–0.88).
  Models must extrapolate ~3× the known-prefix length.
- **TVT smoothness:** dTVT/dMD is concentrated near zero; d²TVT/dMD² is
  sharply peaked with rare heavy-tail events. A naive 8×MAD jump detector
  flags 94% of wells (false positive cliff) — a more conservative
  `abs_dtvtdmd_max > 5` threshold flags only 1 well as a true-fault candidate.
  Hard discontinuities are rare; v1 does not need explicit fault handling.
- **GR calibration:** per-well horizontal-vs-typewell mean offset spans a
  wide ±20 GR-unit range; std-ratio centered near 0.9. A per-well GR
  normalisation (offset + scale to typewell) is essentially mandatory before
  any GR-similarity model.
- **Typewell coverage:** 760/773 wells (98.3%) have their lateral fully inside
  the typewell TVT range. Of the 13 wells that exit, **10 share a single
  typewell file** (md5 `a23359a1…`) and exit on the low side by 100s of TVT
  units — those 10 wells likely need a low-side extrapolation rule.
- **Difficulty clustering (KMeans k=6, silhouette 0.20):** ~85% of wells fall
  into 3 "easy" clusters that differ mostly on GR offset. The non-trivial
  groups are a single jump-like outlier (cluster 3, n=1) and the shared-
  typewell low-side exits (cluster 4, n=10). Most of the LB error budget will
  come from a small minority of wells.
- **Spatial / typewell sharing:** wells form many small pads (DBSCAN @ 0.5%-
  bbox finds ~106 clusters + 274 singletons). 13 typewell file hashes are
  shared by 33 wells (largest shared group n=10). **GroupKFold should group
  on pad and/or typewell-hash, not just on well-id**, to avoid optimistic CV.
- **Train/test alignment:** with only 3 test wells we can't run a robust shift
  test, but every aggregate (known_len, hidden_len, GR mean, Z, X, Y) sits
  inside the train distribution.
- **Leakage audit:** test horizontals have no leak columns; `TVT_input` is
  non-NaN exactly on the prefix and NaN exactly on the suffix; no test PNGs.
  ✓ Sanity confirmed.

**Decisions / next steps:** Three candidate v1 architectures to consider
(no decision made — that's for the next session).

1. **Replicate the rank-#2 baseline (LGB+XGB+CatBoost on hand-engineered
   features, predict `TVT - last_known_TVT`).** Justification: known to score
   10.78 LB; gives us a hard floor and a debugging substrate. The 107-feature
   set is mostly window/lag aggregates plus the beam-search and particle-
   filter outputs — straightforward to re-implement. Risks: this won't beat
   #2 by itself, only matches it; treat as the *baseline*, not the goal.

2. **Sequence model over GR (1D CNN or small Transformer encoder), with
   typewell GR encoded as a key/value bank for cross-attention.** Each well
   becomes a sequence over MD; the model learns the GR-vs-typewell alignment
   directly instead of through hand-crafted features. Justification: GR is
   fully observed including the eval zone (CLAUDE.md leakage rules
   explicitly permit forward-looking GR features), and the rank-#2 reference
   has no neural component — this is a real diversification axis. Risks:
   harder to debug than GBDT; needs careful per-well GR normalization (Q3);
   may overfit the small training set (~773 wells).

3. **Alignment-first: dynamic programming / beam search over typewell TVT
   indices using a GR-similarity likelihood, then a small ML residual
   correction.** Replace the 107 features with a single strong feature —
   the alignment's predicted TVT trajectory — plus a GBDT residual model on
   simple geometric features (MD-position, dipping rate from known prefix,
   well-cluster id, etc.). Justification: the underlying problem is literally
   sequence alignment; making that the primary signal rather than one of 107
   inputs should be cleaner. The rank-#2 PFs already exploit this idea but
   bury it among many other features. Risks: the alignment likelihood needs
   careful tuning; we'll need to handle the 13 typewell-coverage failure
   wells explicitly.

Operational decisions inferred from EDA (apply regardless of which v1 we
pick):
- Always GroupKFold on (pad-cluster ∪ typewell-hash), not on well-id.
- Always per-well GR-normalize against the typewell before any similarity
  computation.
- Hold out the 3 local test wells from training entirely (they are
  duplicates) so local "test" remains a clean signal.
- For the 10 shared-typewell low-side-exit wells, flag and either drop from
  the alignment likelihood or extend the typewell with a constant-rate
  extrapolation.

**Surprises:**
- The local "test" directory is not the official test set — only 3 wells, and
  they're identical-content copies of 3 train wells. CLAUDE.md's "~200
  wellbores" figure does not match disk reality. This is a sandbox.
- 10 wells share a single typewell file, and that exact typewell is too short
  (lateral exits its TVT range by hundreds of units on the low side). Any
  typewell-anchored model will degrade on those 10 unless explicitly
  patched.
- The naive MAD-based jump detector is essentially useless (94% positive
  rate); fault detection in this dataset needs a magnitude-based threshold,
  not a relative one. (Updated `n_jumps_3sigma` interpretation in findings;
  the underlying field stays in the parquet for future use.)

## 2026-05-08 — Reproduce rank-#2 baseline + per-well OOF breakdown

**What:** Forked the rank-#2 reference into [src/baseline.py](src/baseline.py)
(local paths, `__main__` guard, env-var overrides for `BASELINE_MODE` /
`DEBUG_MAX_WELLS` / `DEBUG_ONE_FOLD`). Added `libomp` via brew + uv-added
`lightgbm xgboost catboost scipy tqdm`. Smoke-tested 10 wells / 1 fold, then
ran full `MODE=cv` on all 773 train wells. Wrote
[src/analyze_oof.py](src/analyze_oof.py) which loads
[artefacts/oof_predictions.parquet](artefacts/oof_predictions.parquet),
recomputes per-model + equal-weight + NM-optimal ensemble RMSE, builds a
per-well RMSE table joined to our EDA summary, and saves it to
[artefacts/per_well_oof.parquet](artefacts/per_well_oof.parquet). Full CV
log: [outputs/baseline_cv_full.log](outputs/baseline_cv_full.log).

**Findings:**
- **Clean reproduction.** Per-fold RMSE matches the reference within ~0.02–0.2.
  OOF: LGB 12.486 (ref 12.472), XGB 12.293 (ref 12.315), CB 12.324 (ref 12.292).
  Mean of fold RMSEs across the three models is identical to the ref to 0.02.
- **Ensemble OOF RMSE = 12.205** (Nelder-Mead). Weights: `cb=0.4616,
  xgb=0.5384, lgb=0.0000`. The "3 diverse GBDTs" claim is overstated — LGB
  contributes literally nothing once XGB+CB are present. Reference reported
  similar (LGB=4.4%); on our run it collapses to zero. Removing LGB from the
  pipeline costs nothing and would cut FE-independent training time by ~33%.
- **Reference's reported LB of 10.78 is ~1.4 RMSE better than our OOF**
  (12.21). Consistent with the user's note that public LB = 26% of test data
  (~52 wells); our per-well RMSE distribution is heavy-tailed enough that a
  52-well sample's variance can easily explain a 1.4-RMSE gap.
- **Per-well RMSE is dramatically heavy-tailed**: median 7.55, p90 18.15,
  p99 40.18, max 53.94 (~7× spread). Top-50 worst wells (6.5% of training)
  account for 45% of total SSE; top-100 = 60%; top-200 = 77%. **Most of the
  competition's error budget lives in a small minority of wells.**
- **The failure mode is alignment ambiguity, not typewell coverage.** Top-20
  worst wells *all* have `cov_low_margin ≈ 12` (i.e., firmly inside their
  typewells), unremarkable `abs_dtvtdmd_max` (mostly 1–2), and unremarkable
  `hidden_ratio`. What they share is huge `bias_ens` of ±25 to ±49 — the
  predictions are systematically off by 25–50 stratigraphic units, which is
  the signature of beam search + PFs locking onto the wrong stratigraphic
  candidate (similar-looking GR pattern in a different layer) with no signal
  for the GBDT to override. **None** of the 10 shared-typewell low-side-exit
  wells from the EDA (`02e7fe5a`, `10b89021`, etc.) appear in the top-20
  worst — they're handled fine.
- **Test-name well OOF**: `000d7d20` RMSE 5.60 (good), `00e12e8b` 8.65 (mid),
  `00bbac68` 21.98 (notably bad — worse than median, despite being included
  in training in 4 of 5 folds via the duplicated train file). Provisionally:
  expect official-test ensemble RMSE in the 10.7–12.5 range, not the 10.78
  marketing number.

**Decisions / next steps:** The Tier-B/Tier-C ordering from yesterday is
revised based on the per-well breakdown:

1. **Tier A (newly highest priority): a structurally different aligner**
   (was item 6). The dominant failure mode is multi-hypothesis alignment
   ambiguity — beam + 2 PFs all use the same GR-likelihood and lock onto the
   same wrong candidate together. A DTW alignment with a smoothness penalty,
   exposed as `dtw_tvt`/`dtw_cost`/`dtw_minus_beam_cons` features, is a real
   diversification axis. Estimated upside on those top-50 wells is large
   given they currently have 30–50 absolute bias.
2. **Tier A: drop LGB.** Zero ensemble weight, costs ~33% of training time
   for no gain. One-line ablation to confirm OOF unchanged, then remove.
3. **Tier A: better grouping for CV.** Was already next-up (item 3 yesterday).
   Still worth doing: ensures DTW/other improvements are evaluated on a CV
   that reflects the official-test variance, not on a grouping that leaks
   typewells.
4. **Tier C (deprioritized): patch the 10 shared-typewell low-side-exit
   wells** (was item 4 yesterday). Per the per-well breakdown, this group is
   not in the top-20 worst — patching them won't move the needle.
5. **Tier B unchanged: per-well GR normalization upstream of beam + PF**
   (was item 5). Still measurable; do as an ablation alongside DTW.

Operational follow-ups from this session:
- The 22.0 OOF on `00bbac68` is suspicious given it's in the train set;
  worth a one-off look at what beam/PF produce for it. May be a useful
  diagnostic case for testing DTW.
- Final-training step (`MODE=train`) is unrun; not needed for CV-based
  iteration but will be needed to actually generate a submission file.

**Surprises:**
- My EDA-based Tier-B priority (patch the 10 shared-typewell low-side-exit
  wells) was wrong. None of them are in the top-20 worst by OOF RMSE. The
  EDA's "difficulty cluster 4" was about a *property* of the wells, not
  about *where the model fails*. Lesson: per-well OOF error is a stronger
  prioritization signal than EDA-derived difficulty proxies. Always run CV
  before guessing where the budget lives.
- LGB's ensemble weight is literally 0.0000, not 4.4% as the reference
  reports. Either the reference was tuning slightly differently, or this is
  a CPU-vs-GPU artifact of the OOF predictions feeding Nelder-Mead.
- Per-well RMSE has p99=40 — a single bad well contributes more error than
  100 good wells combined. This is the kind of distribution where any
  improvement that flips even 10–20 wells from disastrous to merely-bad
  could move OOF by 0.5+ RMSE.

## 2026-05-08 — Drop LGB + tag-based CV grouping

**What:** Two ablations on [src/baseline.py](src/baseline.py).
(1) Removed LGB from `ACTIVE_MODELS` (was zero-weighted in NM ensemble).
(2) Replaced per-well `assign_groups` with a union-find over (typewell-file
md5 hash) ∪ (DBSCAN pad-cluster on per-well X/Y centroids, eps = 0.5% of
the X-Y bounding-box diagonal ≈ 1107 ft). 773 wells collapse to 251 groups
(110 singletons, 141 multi-well, largest group n=24). Old artefacts backed
up at `artefacts/{oof_predictions,per_well_oof,fold_metrics}_lgb-xgb-cb_wellgroups.parquet`.
Re-ran `MODE=cv`; full log at
[outputs/cv_xgb-cb_taggroups.log](outputs/cv_xgb-cb_taggroups.log).

**Findings:**

| metric | per-well groups, lgb+xgb+cb | tag groups, xgb+cb | Δ |
|---|---|---|---|
| XGB OOF | 12.293 | 12.804 | +0.511 |
| CB OOF | 12.324 | 12.719 | +0.395 |
| LGB OOF | 12.486 | (dropped) | — |
| Ensemble OOF (NM) | 12.205 | **12.650** | +0.445 |
| Equal-weight ens | ~12.21 | 12.655 | comparable |
| NM weights | cb=.46 xgb=.54 lgb=.00 | cb=.60 xgb=.40 | — |
| Median per-well RMSE | 7.55 | 7.53 | flat |
| p99 per-well RMSE | 40.18 | 39.51 | flat |
| Max per-well RMSE | 53.94 | 57.51 | +3.6 |
| Top-50 worst share of SSE | 45% | 46% | flat |
| Fold-RMSE std (across 5 folds) | ~0.20 | 0.30–0.40 | larger |

- The +0.45 OOF jump is **leakage closure**, not a regression. Old folds
  routinely had a val well's typewell-twin or pad-neighbor in train; the
  GBDTs were partially memorising shared-typewell GR signatures. New OOF
  is the honest estimate; trust it over the old number going forward.
- LGB drop is free: ensemble OOF basically unchanged once XGB+CB take its
  weight. Confirmed — leave LGB out.
- The per-well distribution is essentially unchanged — same median, p99,
  same alignment-ambiguity wells in the top-20 (`1b1eba53`, `86454a6f`,
  `389ae58f`, `c8d9680c`, etc.) with the same ±25 to ±53 `bias_ens`. The
  improvement target for DTW (next item) is unchanged.
- Fold-RMSE std rose from ~0.20 to 0.30–0.40. Folds are now structurally
  heterogeneous (a fold containing the 24-well group is unlike a fold full
  of singletons). Accept higher OOF noise (~±0.05 from group-assignment
  reshuffles) as the cost of honesty.
- The `00bbac68` test-name well OOF is now 21.34 (was 22.0). Still anomalously
  bad given it's in the train file, still a useful diagnostic case for DTW.

**OOF vs LB — explicit:**
- 12.65 is our **OOF on 5-fold tag-grouped CV** of the 773 train wells.
  This is *not* a leaderboard number.
- We have no LB anchor of our own yet. The reference's 10.78 is *their*
  public-LB number on 52 hidden-test wells, not transferable to our pipeline.
- Public LB samples 26% of the hidden test (~52 wells). Given our top-50
  worst wells contribute ~46% of total SSE, a public-LB number from this
  pipeline could land anywhere in roughly 9–14 RMSE depending on which 52
  hidden wells happen to fall on the public split. The private LB (148 wells)
  will be more representative.
- **Implication:** OOF is the right iteration metric. Submit to Kaggle for
  a real LB number only when (a) we want a calibration point or (b) we have
  a meaningful improvement we want to confirm.

**Decisions / next steps:**
1. **DTW alignment** (Tier A) — unchanged from yesterday's plan. Top-20
  worst wells still dominate by alignment bias; this remains the highest-
  upside item. Design pass next.
2. **First Kaggle submission** (new item) — set up a Kaggle notebook that
  pulls cached `train_df.parquet` from a private dataset, retrains XGB+CB
  on full data, and emits `submission.csv`. Goal: get our first own LB
  anchor before iterating further. Plays with Kaggle workflow we'll need
  for every future submission anyway. Estimate: small if the cached parquet
  upload works; larger if FE has to re-run inside the notebook.
3. **Per-well GR normalisation upstream of beam/PF** (Tier B, was item 5) —
  still pending.
4. (Deprioritised, recorded for completeness) The 10 shared-typewell low-
  side-exit wells from the EDA still aren't in the top-20 worst (none of
  `02e7fe5a`/`10b89021`/`3417285d`/`bc4381e2`/`ecdab904` appear). The
  Tier-C demotion stands.

**Surprises:**
- I almost wrote "12.65 is our private-LB anchor" in the session summary
  before being corrected. None of our internal numbers are LB numbers
  until we actually submit. Fixing the framing here so future-me reads it
  correctly: OOF is generalization estimate; LB is what Kaggle returns
  after a submission; the two are not interchangeable.
- The OOF jump (+0.45) was bigger than I'd guessed (~+0.1–0.2). The shared
  typewells (33 wells) and pads (~106 clusters covering most non-singleton
  wells) were leaking more signal than the EDA suggested. Worth remembering
  whenever someone proposes a CV setup: "wells are independent" was wrong
  here, and the test for whether that's true is rarely visible from EDA
  summary stats alone.

## 2026-05-08 — Add DTW as third aligner; XGB+CB+DTW OOF 12.531

**What:** Implemented banded subsequence DTW as a third aligner in
[src/baseline.py](src/baseline.py) (`dtw_predict` + `compute_dtw_features`,
section 5b). Steps `{-1, 0, 1}`, emit_scale 144 (matches beam_cons),
quadratic slope_penalty 20 referenced to `exp_dj` from the prefix slope,
band ±100 typewell cells centred on a *flat* line at `start_idx`
(prefix-slope drift was killing wells whose dipping rate changed in the
hidden zone — e.g. 028d7b28). Vectorised inner DP over the band; ~1.1s/well
on macOS. Exposed 9 features per row: `dtw_delta`, `dtw_local_cost`,
`dtw_total_cost`, `dtw_minus_{beam_cons, beam_loose, pf, ancc}`,
`tw_gr_at_dtw`, `gr_minus_tw_dtw`. Rebuilt train_df (773 wells, 116 cols
total, 24 of which are now alignment outputs) and re-ran 5-fold tag-grouped
CV with XGB+CB. Full log:
[outputs/cv_xgb-cb_taggroups_dtw.log](outputs/cv_xgb-cb_taggroups_dtw.log).

**Findings:**

| metric | tag groups, xgb+cb (no DTW) | + DTW | Δ |
|---|---|---|---|
| XGB OOF | 12.804 | **12.612** | **-0.192** |
| CB OOF | 12.719 | 12.661 | -0.058 |
| Equal-weight ens | 12.655 | 12.533 | -0.122 |
| NM-optimal ens | 12.650 | **12.531** | **-0.119** |
| NM weights | cb=.60 xgb=.40 | cb=.44 xgb=.56 | XGB pulled weight |
| Median per-well RMSE | 7.53 | 7.63 | +0.10 |
| Mean per-well RMSE | 9.81 | 9.77 | -0.04 |
| Max per-well RMSE | 57.51 | 56.60 | -0.91 |
| Top-50 worst share of SSE | 45.7% | 45.8% | flat |

- **DTW is a real win, primarily via XGB.** -0.19 OOF for XGB vs -0.06 for
  CB. XGB exploits the disagreement features (`dtw_minus_beam_cons` and
  friends) more aggressively; CB's ordered boosting tends to discount weak
  features. Net ensemble OOF -0.12.
- **DTW does NOT correlate with beam_cons** despite using the same emit
  cost (corr 0.15 on the smoke-test 20 wells). The "global DP, no pruning"
  + "quadratic move cost" + "flat band centring" combination genuinely
  produces different paths from beam search. ~37% of rows have
  `|dtw - beam_cons| > 5`, ~6.5% > 20.
- **The win is from feature diversification, not from fixing the worst
  wells.** Top-20 worst by ensemble RMSE is essentially the same set as
  before (`1b1eba53`, `86454a6f`, `c8d9680c`, `5f4d2a52`, `8f201368`,
  `7e721392`, `389ae58f`, `708caea9`, `91db7070`, `81bf5923`, `fef8af96`,
  `a959858c`, `77e4821c`, …). Same ±30–55 `bias_ens` signature. DTW shares
  the GR-magnitude likelihood with beam/PF, so it gets misled on the same
  alignment-ambiguous wells.
- **Median per-well slightly worse (+0.10) but mean -0.04 and max -0.91.**
  DTW adds noise on easy/mid wells (median drift) but shaves the tail.
  Row-level OOF (which is what gets reported) is dominated by long wells
  and the right tail, hence the net OOF improvement.
- **Test-name wells mixed:** `000d7d20` 8.34 → 7.64 (better),
  `00e12e8b` 7.91 → 8.21 (~flat), `00bbac68` 21.34 → 25.46 (worse).
  Small-N noise on n=3.
- **Initial DTW configuration was much worse.** First pass used
  `{-2..+3}` step set, slope_penalty 5.0, prior-slope-drifted band — DTW
  RMSE on `028d7b28` was 127 (vs beam_cons 6.92) and several wells were
  > 80 RMSE. Diagnosis: prefix dipping rate doesn't predict hidden-zone
  dipping on wells where the geology shifts mid-lateral, and a wide step
  set with weak penalty lets DTW chase spurious far-away GR matches.
  Tightening to `{-1, 0, 1}` + 4× higher penalty + flat band fixed it
  (worst dropped to 32.77, median DTW = median beam_cons over 20-well sample).

**OOF vs LB — still:**
- 12.531 is OOF on tag-grouped 5-fold CV. Not an LB number.
- A first own LB anchor still requires a Kaggle submission.

**Decisions / next steps:**
1. **First Kaggle submission** (still pending from yesterday). Now even
   more justified — we have a meaningfully better pipeline than yesterday's
   12.65 baseline. Set up a notebook that pulls the cached `train_df` from
   a private Kaggle dataset, retrains XGB+CB on full data, emits
   `submission.csv`. Goal: get our first own LB anchor.
2. **LGB-with-Optuna fork** (parallel track requested by user): wrote
   [src/baseline_lgb.py](src/baseline_lgb.py) — single LGB model with
   Optuna hyperparameter search. Reuses the same `train_df.parquet` so it
   includes DTW features. Separate artefacts at `artefacts/lgb_optuna/`.
   Default 30 trials, 1-hour timeout, resumable SQLite study. Honest take:
   tuned LGB alone unlikely to beat the 2-model ensemble (LGB had NM weight
   0.000 in the 3-model run); main value is iteration speed for future
   ablations and possible 3-model ensemble reintroduction.
3. **Different-observation aligner** (newly added). The persistent
   alignment-ambiguity wells share the same GR-magnitude likelihood across
   all four aligners (beam × 4, PF × 2, DTW × 1 — wait that's 7 paths,
   none with a different observation). A DTW variant on `dGR/dMD`
   (gradient similarity) or on smoothed-GR with a different smoothing
   radius would hit a different cost surface. This is the main lever
   left for the right tail.
4. Per-well GR normalisation upstream of beam/PF/DTW (Tier B) — still
   pending. Could be combined with item 3.

**Surprises:**
- DTW correlation with beam_cons is 0.15 — much lower than expected given
  identical emit cost. The non-pruning + quadratic-cost + flat-band
  combination is a more meaningful change than I'd predicted. Note for
  future: "same cost function" doesn't mean "same predictions" — the
  inductive bias of the search procedure dominates on ambiguous problems.
- XGB benefits ~3× more than CB from DTW features. Earlier I assumed both
  GBDTs would benefit roughly equally from any new feature; CB's
  conservatism on weak/noisy features is a real factor on this dataset
  and worth remembering when evaluating future feature additions.
- DTW v1 with prefix-slope-drifted band failed catastrophically on a
  well (`028d7b28`) where the lateral's true slope (~0.005) was 7× smaller
  than the prefix-estimated slope (0.036). The prefix is *not* a reliable
  predictor of hidden-zone slope on the hard wells — generalises beyond
  DTW: any model that extrapolates the prefix-slope as a confident prior
  will fail on the same wells.

## 2026-05-08 — LGB-only fork with Optuna; submission notebooks ready

**What:** Built [src/baseline_lgb.py](src/baseline_lgb.py) — single-LGB fork
of the main pipeline with Optuna hyperparameter search. Reuses the cached
`train_df.parquet` (so it inherits DTW features) and writes its own
artefacts to `artefacts/lgb_optuna/`. Three modes (`tune`, `cv`, `train`),
SQLite-resumable study, per-trial+per-fold logging, group-aware subsampling
for fast tuning. Ran a 30-trial study with `TUNE_FAST=1` (30% group
subsample, 3-fold during tuning, n_estimators capped at 3000, lr ∈
[0.02, 0.10]) followed by automatic 5-fold full-data CV with the best
params at full early-stop budget. Full log:
[outputs/lgb_optuna_v1.log](outputs/lgb_optuna_v1.log).

Also patched `src/baseline.py`:
- `ARTEFACT_DIR`/`OUTPUT_DIR`/`DATA_DIR` overridable via env vars (Kaggle
  read-only mounts no longer fail at import).
- New `finalize_only` mode that uses cached `best_iters` to skip CV and
  go straight to full-data final training + ensemble-weight saving.
  Used to produce model artefacts for the Kaggle submission notebooks.

Created two Kaggle submission notebooks:
- [notebooks/submit_ensemble.ipynb](notebooks/submit_ensemble.ipynb) —
  XGB + CB ensemble with DTW features.
- [notebooks/submit_lgb_optuna.ipynb](notebooks/submit_lgb_optuna.ipynb) —
  Tuned LGB single model with DTW features.
- [notebooks/KAGGLE_SETUP.md](notebooks/KAGGLE_SETUP.md) — workflow:
  finalize models locally → bundle artefacts as private dataset → run
  notebooks on Kaggle.

**Findings:**

LGB-Optuna study (30 trials, ~29 min wall):
- Best 3-fold subsample CV RMSE: 12.331
- Best full-data 5-fold CV RMSE (final, with full early-stop budget): **12.640**
- 0.31 RMSE gap between subsample-CV and full-CV. The 30% subsample with
  3-fold has notably easier mean (heterogeneous fold composition: trial
  scores routinely showed f1≈10.5, f2≈12.5, f3≈14.5 — fold 3 was always
  the hard one). Subsample tuning is fine for *ranking* params but reads
  0.3 RMSE optimistic compared to full CV.
- Best params: `learning_rate=0.0364, num_leaves=36, min_data_in_leaf=110,
  max_depth=2, feature_fraction=0.96, bagging_fraction=0.75, bagging_freq=7,
  λ_l1≈0, λ_l2≈0, min_gain≈0`. **Heavy regularisation, very shallow trees.**
  Trial 19 found a similar configuration (`max_depth=3, num_leaves=153,
  min_data_in_leaf=122`) with same value 12.331. Both convergent on
  "shallow + lots of data per leaf", which fits the heavy-tailed per-well
  RMSE distribution — wide-but-shallow trees average over wells without
  overfitting any single hard well.

Comparison vs current best ensemble:

| pipeline | OOF RMSE | notes |
|---|---|---|
| XGB only (DTW)         | 12.612 | strongest single model |
| **LGB tuned (DTW)**    | **12.640** | new, single-model, Optuna-tuned |
| CB only (DTW)          | 12.661 | |
| XGB + CB ensemble (DTW)| **12.531** | current best, NM weights cb=.44 xgb=.56 |
| LGB default (no DTW)   | 12.486 | reference, in old per-well CV (optimistic) |

- **Tuned LGB beats default-LGB-on-its-own-axis** (would-be ~12.7 with
  default LGB on the new tag-grouped CV with DTW; the 12.486 from the
  old run was on optimistic per-well groups). Real improvement from
  tuning.
- **Tuned LGB does NOT beat the 2-model ensemble.** Confirms the prior:
  single tuned LGB ≈ standalone XGB; can't match XGB+CB diversification.
- **LGB is a credible 3rd ensemble member.** Per-trial fold variance
  patterns differed from XGB and CB (e.g., shallower-tree LGB stayed
  under 12 on fold 1 more reliably). Worth re-introducing into a 3-model
  ensemble in a future session — last time LGB at default params had
  NM weight 0.000, but tuned-LGB's prediction surface is meaningfully
  different and may take non-zero weight.

**OOF vs LB — still pending:**
- Both pipelines now have a clear path to a real LB number via the two
  submission notebooks.
- Submission setup is the next required step (run `finalize_only` for
  XGB+CB, `train` for LGB, bundle artefacts, upload to Kaggle, run
  notebooks). Estimated ~30 min total local + ~10 min per Kaggle
  notebook run.

**Decisions / next steps:**
1. **Finalize models + submit both pipelines** (the immediate plan).
   Two submissions will give us our first own LB anchors and tell us
   the OOF↔LB gap. We expect public LB to land somewhere in 9–14 RMSE
   given heavy-tailed per-well distribution and 26%-sample variance.
2. **3-model ensemble** (XGB + CB + tuned-LGB) — try after the first
   submission. Re-run NM weight optimisation on the OOF stack of all
   three; if tuned-LGB takes >5% weight, integrate into a new submission.
3. **Different-observation aligner** (Tier A from the DTW entry) — still
   unaddressed. The right-tail wells need this; submission/ensembling
   won't help them.

**Surprises:**
- The fast-tuning subsample read **0.31 RMSE optimistic** vs full CV.
  Worth flagging: subsample-based tuning gives reliable param *rankings*
  but absolute OOF predictions on subsamples are not honest estimates of
  full-data generalisation. For future hyperparameter tuning, always
  re-run a final-CV step at full budget before trusting a number.
- Optuna landed on max_depth=2 (literally 2-level trees) with num_leaves=36
  (a constraint that's effectively meaningless given depth=2 caps the
  tree at ≤4 leaves). The TPE sampler doesn't enforce param-set
  consistency — `num_leaves=36` is dead code in the final config. Worth
  noting that hyperparameter searches can find redundant configurations
  that nonetheless score well.
- The first attempt to run the LGB study with full-data 5-fold and a
  wider lr range was projected to take ~14 hours for 30 trials. Fast
  tuning (subsample + smaller folds + tighter ranges) cut that to
  29 minutes for the same trial count. Per-trial speedup ~30×, with
  only the 0.3 RMSE optimism gap as the cost. Strong default for
  future tuning rounds: always tune on a subsample first.

## 2026-05-08 — First Kaggle submissions: LB anchors established

**What:** Bundled local artefacts (`src/baseline.py`, trained
xgb/cb/lgb model files, features.json, best_iters.json,
ensemble_weights.json, lgb best_params.json) into a 3.2 MB private Kaggle
dataset and ran both submission notebooks against the official ~200-well
hidden test set.

**Findings — first own LB numbers:**

| pipeline | OOF (5-fold tag) | Public LB | LB rank | OOF − LB |
|---|---|---|---|---|
| XGB + CB ensemble (DTW) | 12.531 | **10.364** | **#20** | -2.17 |
| Tuned LGB single (DTW)  | 12.640 | 10.531 | — | -2.11 |
| Reference (rank-#2 in writeup) | (their) 12.30 | 10.784 | (their) #2 | -1.55 |

- **Both submissions beat the reference's reported 10.784** by 0.25–0.42.
  Ensemble is 0.42 better. Confirms DTW + tag-grouped CV + dropping LGB
  was a real improvement over the reference's pipeline, not just a
  reproduction.
- **Both submissions sit ~2.1 RMSE *better* on LB than on OOF.** Direction
  matches the heavy-right-tail prediction from the 2026-05-08 entry:
  public LB samples ~52 of ~200 wells (26%), and our top-50 worst wells
  contribute ~46% of total SSE. The public split apparently undersampled
  the right tail. Private LB (148 wells) should regress most of this
  optimism — provisionally expect private LB ≈ 11.5–12.5 for the ensemble.
- **Ensemble beats single-LGB by 0.17 on LB** (10.364 vs 10.531). Smaller
  margin than the OOF gap (0.11 vs 0.20 — wait, OOF gap was 0.11 so LB
  margin is *larger*). Ensemble's diversification advantage holds up on
  the LB sample.
- **Leaderboard has shifted dramatically since the reference writeup.**
  Reference was #2 at 10.784; we're #20 at 10.364. Top of current LB is
  likely in the 9.x range. The competition is much more crowded /
  competitive than the rank-#2 writeup suggests.
- **Submission notebooks worked end-to-end on first run.** No silent
  zero-fills; all `prediction_id`s in `sample_submission.csv` matched
  `test_df` rows. The Kaggle bundle approach (3.2 MB zip, models loaded
  at inference time, FE rebuilt on Kaggle test wells) takes ~5 min per
  submission run.

**OOF vs LB — calibrated:**
- OOF reads ~2.1 RMSE pessimistic vs the *public* LB sample on this data.
  Direction is right (heavy tail underrepresented in 52-well subsample),
  magnitude is consistent with the SSE-share analysis.
- Private LB will be more representative — expect a meaningful regression
  toward OOF when private numbers settle.
- Future iterations: trust OOF Δ as the reliable signal of improvement.
  Don't chase LB-Δ on a single submission — sample-noise on 52 wells is
  large.

**Decisions / next steps (Phase 2):**

1. **3-model ensemble (XGB + CB + tuned-LGB)** — re-run NM weight
   optimisation on the OOF stack of all three; if tuned-LGB takes
   non-trivial weight (>5%), submit. Estimated upside on LB: 0.05–0.15
   RMSE.
2. **Different-observation aligner** (still Tier A) — DTW on `dGR/dMD`
   gradient or a dramatically different smoothing radius. The persistent
   alignment-ambiguity wells (`1b1eba53`, `86454a6f`, `c8d9680c`, …) all
   share the GR-magnitude likelihood; nothing in the current pipeline
   addresses this. Estimated upside: substantial on the right tail,
   uncertain magnitude on overall RMSE.
3. **Per-well GR normalisation upstream of all aligners** (Tier B) —
   still pending. Expect modest improvement.
4. **Stacking / second-level model** — instead of NM weights, train a
   small ridge or shallow GBDT on the OOF stack as features. Has
   marginal upside but worth a half-day investment if (1) and (2) are
   exhausted.
5. **EDA Phase 2** — now that we have a strong baseline, look at the
   top-20 worst wells in detail (raw GR, typewell GR, beam/PF/DTW paths
   overlaid on truth) to find a structural pattern. Drives item 2.

Phase 1 (build a strong baseline + first LB anchor) is **done**.

**Surprises:**
- Both submissions read 2+ RMSE better on public LB than on OOF. We knew
  the direction was likely (right-tail-heavy distribution + small public
  sample) but the magnitude of the gap is still notable. Lesson: the
  variance from the 26% public sample on this data is enormous; treat
  individual public-LB numbers as ±1 RMSE noise around the true
  generalisation, and weight private-LB more heavily.
- We're at LB rank #20 with a configuration that's only mildly more
  sophisticated than the rank-#2 writeup (drop one model, fix CV
  grouping, add one aligner). The competition is much more competitive
  than the writeup-era leaderboard. Implication: incremental
  improvements at this point are unlikely to move us much in rank
  unless they're substantial. The next real lever is structural
  (different-observation aligner) or model-class (e.g., a sequence
  model) — not feature-engineering tweaks.

## 2026-05-08 — Phase 2 EDA: per-well diagnosis of worst 20 wells

**What:** Built [src/eda_phase2_plots.py](src/eda_phase2_plots.py) — a
3-panel diagnostic plot per well: (A) horizontal GR vs MD with end-of-prefix
marker; (B) typewell GR vs TVT with shaded bands for the true hidden-zone
TVT range (green) and each of 7 aligners' predicted TVT ranges (beam_tight,
beam_cons, beam_loose, beam_vloose, pf, ancc, dtw); (C) TVT vs MD with all
7 aligner paths + ensemble + truth + `TVT_input` prefix. Driven by
[notebooks/eda_phase2.ipynb](notebooks/eda_phase2.ipynb) which selects the
top-20 worst wells by ensemble OOF RMSE and 5 wells near median RMSE as
contrast. 25 PNGs in [eda_outputs/figs/phase2/](eda_outputs/figs/phase2/).
Findings appended to [eda_findings.md](eda_findings.md). Spec at
[docs/superpowers/specs/2026-05-08-phase2-eda-worst-wells-design.md](docs/superpowers/specs/2026-05-08-phase2-eda-worst-wells-design.md).

**Findings:**
- **Look-alike-layer failure confirmed visually.** On 14 of 14 worst wells
  inspected, Panel B shows the truth band and the stacked aligner bands
  sitting 30–50 TVT units apart over typewell regions with visually
  similar GR character (similar mean/spread, similar peak structure).
  Classic alignment ambiguity — the wells aren't pathological in any
  EDA-derivable way, they just live in low-distinctiveness GR
  neighborhoods of their typewells.
- **All 7 aligners cluster together and disagree with truth in lockstep.**
  Structural signature of a shared GR-magnitude likelihood. DTW (added
  yesterday) helped via feature diversification, not by escaping the
  shared observation model.
- **Bias is sign-mixed, not directional.** Of the 14 worst inspected,
  9 are negative (predict too shallow) and 5 are positive (too deep).
  The failure is "lock onto wrong layer", not a systematic offset.
- **Easy-well contrast (47222616, 5aa03df7, dc7f9757, all RMSE 7.6):**
  truth and aligner bands *overlap* in Panel B, and the relevant typewell
  region contains a distinctive feature (sharp transition, plateau
  boundary, unique trough). Easy ↔ hard is determined by *local typewell
  distinctiveness* near the truth, not by intrinsic well properties.
- **Trajectory-shape failures observed (Panel C).** 1b1eba53, c8d9680c,
  91db7070, 5f4d2a52: aligners trajectory is flat while truth dips 30–50
  units. ba48188d, 7e721392: aligners over-extrapolate prefix slope while
  truth flattens. The aligners are not getting dipping-rate signal from
  GR — they inherit it from the prefix or the smoothness prior. A
  trend-aware likelihood would constrain trajectory shape directly.

**Decisions / next steps:**
1. **Build a different-observation DTW aligner** with a composite
   likelihood combining `dGR/dMD` (gradient) and a low-pass GR (σ ≈ 50–100
   ft TVT) component, with a small weight on the existing GR-magnitude
   term. Both are CSV-derivable and test-time legal. The two components
   target the two failure modes seen in the panels — gradient breaks the
   look-alike ambiguity in regions where adjacent typewell layers have
   similar magnitudes but different dynamics; low-pass GR makes the slow
   trend the primary match signal rather than mixing it with HF noise.
   Implement on top of the existing DTW code in [src/baseline.py](src/baseline.py).
2. **Run the 3-model ensemble (XGB+CB+tuned-LGB) in parallel** as the
   cheap side track. Re-run NM weight optimisation; if tuned-LGB takes
   >5% weight, submit.
3. **Recorded for v2 EDA:** typewell `Geology`-segment overlay on Panel B
   would test whether geology is a free disambiguator (currently unused
   by every aligner). Cheap to add if the gradient/low-pass aligner
   doesn't fully close the gap.
4. **Recorded for future:** PNG-derived features from `data/train/{well}.png`
   as a training-only auxiliary signal. Out of scope until current
   alternate-observation work is done.

**Surprises:**
- The look-alike-layer hypothesis from the per-well-OOF entry survived
  contact with the actual plots. Every worst well has a clean visual
  signature for it. I had budgeted 1–2 sessions to potentially fall back
  to "patterns are diffuse" and need to add derivatives/multi-scale to
  the panels (the v2 path in the spec). Didn't need it.
- The easy-vs-hard split is not a property of the *well* — it's a property
  of the *typewell region* that the well's hidden zone happens to traverse.
  Wells in distinctive-GR neighborhoods are easy regardless of hidden_ratio,
  prefix length, or pad cluster. Wells in low-distinctiveness neighborhoods
  are hard regardless of any of those. This is a stronger framing than
  the EDA's "difficulty cluster" (which was about wells) and explains why
  the per-well RMSE distribution is so heavy-tailed: difficulty is a
  function of the local typewell, and a small minority of typewell
  regions are degenerate.
- Several worst wells fail trajectory-shape, not just absolute position.
  This wasn't visible in any aggregate metric — `bias_ens` only tells you
  the centroid of the residual, not whether the residual is a flat offset
  or a shape mismatch. Suggests RMSE-of-residual-trend (or a per-well
  fit quality of `pred - truth` against MD) would be a useful future
  diagnostic.

## 2026-05-08 — Composite-emit DTW (dtwc): negative result, removed

**What:** Built a fourth aligner — composite-emit DTW (5c) — informed by
Phase 2 EDA. Same banded DP as the existing DTW, but with an emit cost
that combines (a) GR-magnitude (existing, weight 0.5), (b) gradient
`dGR/dMD` (new, weight 1.0), and (c) low-pass GR (new, weight 1.0).
Implementation in [src/experiments/dtwc.py](src/experiments/dtwc.py) — kept
as documented experimental code, removed from
[src/baseline.py](src/baseline.py) after the negative result. Geology
overlay on Panel B was the prerequisite step (added to
[src/eda_phase2_plots.py](src/eda_phase2_plots.py)) and itself produced a
decisive negative result — geology does not disambiguate the look-alike
failure because EGFDL is the dominant zone for both truth and prediction.

**Findings:**

Smoke test (raw aligner RMSE on 25 EDA wells, dtw vs dtwc):

| group   | mean dtw RMSE | mean dtwc RMSE | mean Δ |
|---------|---------------|----------------|--------|
| Worst-20| 41.16         | 35.23          | −5.92  |
| Easy-5  | 14.13         | 21.89          | +7.76  |

Strong improvement on hard wells (e.g., 2fd68f7b 69 → 32, a959858c 78 →
45, 6d6d93af 56 → 19) but regressions on easy wells (fb3848a1 5.6 → 39).

Full XGB+CB CV (tag-grouped, 5-fold):

| metric            | dtw only       | dtw + dtwc     | Δ        |
|-------------------|----------------|----------------|----------|
| XGB OOF           | 12.612         | 12.625         | +0.013   |
| CB OOF            | 12.661         | 12.697         | +0.036   |
| Ensemble OOF (NM) | **12.531**     | **12.551**     | **+0.020** |
| p50 per-well RMSE | 7.63           | 7.74           | +0.11    |
| p90 per-well RMSE | 18.46          | 18.73          | +0.27    |
| p99 per-well RMSE | 39.72          | 38.56          | −1.16    |
| Worst-20 mean RMSE| 38.28          | 38.23          | −0.05    |

Right tail compressed slightly (p99 −1.16) but the bulk of the
distribution shifted worse. Worst-20 ensemble RMSE is essentially flat
(9 / 20 improved, 11 / 20 regressed). The 6-RMSE per-well wins from the
smoke test did not translate to the ensemble — XGB+CB already had enough
signal from existing aligners + disagreement features to handle most of
those wells, and on easy wells the new `dtwc_minus_*` features look like
noise.

**Interpretation:** The smoke-test pre-test asked the wrong question. It
measured raw dtwc RMSE vs raw dtw RMSE, which would matter if the
pipeline used a single aligner. But the GBDT ensemble's job is to
arbitrate between aligners per row, and the new disagreement features
(`dtwc_minus_dtw`, `dtwc_minus_beam_cons`, …) don't tell the GBDTs
*which* aligner is right when they differ. Adding aligners helps only
when the GBDTs can either (a) consistently arbitrate or (b) get a stable
diversification effect via NM ensembling. dtw already provided the
latter (NM weight from CB was reduced 0.60 → 0.44 when dtw was added);
dtwc provides neither.

**Decisions / next steps:**

1. **Drop dtwc.** Reverted to dtw-only pipeline. Rationale per
   [docs/superpowers/specs/...](docs/superpowers/specs/) and prior
   journal lesson: "trust OOF Δ as the reliable signal of improvement;
   don't chase LB-Δ on a single submission — sample-noise on 52 wells
   is large." Submitting on +0.02 OOF would burn a submission to learn
   nothing.
2. **Pivot to FE Round 2** (the user's plan #4 from this session, now
   reordered to #1). Two cheap items:
   - Drop zero-gain features from existing 117 — `feature_importance.parquet`
     already exists for the previous final XGB run.
   - **Add a per-row typewell-distinctiveness feature** — at the
     predicted TVT, how unique is the typewell GR pattern within the
     same typewell? Phase 2 EDA's deepest insight was that the easy↔hard
     split is a property of *local typewell distinctiveness*. This
     feature gives the GBDTs the per-row arbitration signal they
     currently lack. Design pass next session.
3. (Recorded for completeness) The negative-result rule-outs for dtwc:
   tuning emit weights/scales is unlikely to flip +0.02 to a meaningful
   improvement; an aligner-only signal that the GBDTs can't arbitrate
   is fundamentally limited; future direction is per-row arbitration
   features or a learned-observation model (sequence neural net).

**Surprises:**

- **Geology was a non-disambiguator.** I'd budgeted geology as a
  possible v2 fallback if gradient/low-pass didn't suffice. The Panel B
  re-render with geology bands plus the per-row geo-agreement check
  showed truth and prediction in the *same* geology segment for 14/20
  worst wells (median 100% same-geo). The look-alike failure is
  *within-EGFDL*, not across geology boundaries. Future EDA should
  always include this kind of programmatic agreement check, not rely
  solely on visual inspection of bands.
- **The smoke-test → CV gap was bigger than the prior DTW pass.** When
  DTW was added (2026-05-08 earlier entry), its raw RMSE wins translated
  to a +0.12 ensemble OOF improvement. dtwc's larger smoke-test wins
  delivered −0.02 ensemble OOF. The difference is that DTW's path was
  consistently better than beam/PF on the same kinds of wells where
  beam/PF struggled, so the GBDTs learned a stable preference; dtwc's
  path is sometimes much better and sometimes much worse than dtw, with
  no per-row signal of which case applies. Lesson for future aligner
  additions: the right pre-test isn't "is the new aligner more accurate
  on average?" but "is its prediction reliably arbitrable from existing
  aligners' predictions, given the features the GBDTs see?"
- **Per-row arbitration is the real missing piece.** Phase 2 EDA showed
  the failure is alignment ambiguity in low-distinctiveness regions of
  the typewell. The GBDTs currently have no signal to identify which
  rows are in those regions. This reframes FE Round 2: the goal isn't
  feature pruning + cosmetics, it's giving the GBDTs the input they
  need to arbitrate.

## 2026-05-08 — Phase 2 wrap-up: distinctiveness fails LGB transferability + LGB-zero pruning is neutral

**What:** Two LGB CV experiments to wrap up Phase 2 GBDT exploration.
Reused the cached Optuna-tuned `best_params` (heavy regularisation,
`max_depth=2`, `min_data_in_leaf=110`) so each experiment isolates a
single feature-set change. v1 LGB used the original 117-feature
train_df. The current `train_df.parquet` has 6 distinctiveness lookups
added and 3 zero-importance features dropped (`md_diff`, `gr_missing`,
`gr_grad2`) — left in place since they were already wired and matched
the EDA insight; the experiments toggle other dimensions on top.

**Findings:**

| run                                    | features | LGB OOF    | Δ vs v1   |
|----------------------------------------|----------|------------|-----------|
| v1 LGB (Optuna-tuned, original 117)    | 117      | **12.640** | —         |
| Exp A: +6 uniq, −3 zero-gain           | 120      | 12.697     | +0.057    |
| Exp B: pruned (drop 42 LGB-zero + 6 uniq) | 65    | 12.652     | +0.012    |

- **Exp A confirms transferability of the XGB+CB result.** Distinctiveness
  features hurt LGB by +0.057 — same direction as the XGB+CB regression
  earlier today (+0.020). The diagnosis from earlier in the day stands:
  distinctiveness tells the GBDT *"this prediction is in an ambiguous
  zone"* but provides no alternative answer when all aligners agree on a
  wrong layer. Half the signal isn't enough.
- **Exp B confirms pruning is near-neutral.** Dropping 48 features
  (41% reduction, including all the GR-derivative rolls/lags/leads
  LGB already gave zero importance) moves OOF +0.012 — well within
  fold-RMSE std (0.30 in this run). The LGB-zero list was honest: those
  features were neither helping nor hurting. Pruning gives a smaller
  faster model with the same accuracy.
- Fold-RMSE std is 0.31 (Exp B) and 0.35 (Exp A), comparable to v1's
  0.30. Folds 2 and 4 are consistently the hardest (12.88–13.07);
  fold 5 the easiest (12.11–12.12). The fold composition is stable
  across all three runs.

**Decisions / next steps — Phase 2 closes:**

1. **Phase 2 ends here.** The XGB+CB ensemble (LB 10.364, OOF 12.531)
   is the Phase 2 best, achieved before any of today's experiments.
   No new submission warranted. Current ranking on Kaggle: #20.
2. **Pruning Exp B is informative but not a winning ablation.** A
   leaner pipeline with the same OOF could matter for compute budgets
   later, but doesn't move the leaderboard. Not productionising the
   prune unless we revisit GBDTs.
3. **Phase 3: pivot to sequence models.** The repeated dead end of the
   last three experiments (dtwc, distinctiveness, pruning) all confirm
   the same diagnosis: the GBDTs have absorbed all the signal the
   hand-engineered aligner outputs and disagreement features can
   provide, and they can't arbitrate between aligners on a per-row
   basis. The remaining error budget lives in the alignment-ambiguity
   wells where all aligners agree on a wrong layer; no GBDT-level
   feature can fix this. The next lever is a model class that learns
   the observation likelihood from the GR sequence directly — a 1D
   CNN or small transformer encoder over MD with cross-attention to
   the typewell GR. That replaces the hand-engineered aligners with
   a learned alignment. CLAUDE.md's Q1 from EDA Phase 1 already
   flagged this as a candidate; today's results make it the clear
   next step.
4. **Cleanup completed for disk space.** Deleted ~3.4 GB of stale
   backups, pre-DTW train_df, and historical OOF parquets. Current
   live artefacts: tag-grouped XGB+CB OOF (`oof_predictions.parquet`),
   final XGB + CB models, cached LGB best_params + Optuna study,
   train_df with 120 cols (distinctiveness in but not used by current
   submission). Experiment A and B LGB OOFs preserved at
   `artefacts/lgb_optuna/{uniq,v1}/` for reference.

**Surprises:**

- **Pruning is neutral, not free.** Dropping 41% of features for
  +0.012 OOF (within fold noise) is a clean ablation showing that
  importance-based pruning of GBDT-zero features is genuinely
  information-preserving — the features were zero for a reason. But
  it's not an improvement. Worth remembering for future reference:
  "the dropped features were noise" is a true claim only when you
  measure that the model *can* be trained without them at the same
  performance.
- **Distinctiveness was a clean negative result.** All three GBDT
  variants tested today (XGB+CB ensemble, single LGB) regressed
  with the feature added. The hypothesis was sound — Phase 2 EDA
  established that easy↔hard correlates with local typewell
  distinctiveness — but the implementation gap (no per-row
  alternative answer to redirect to) is a real conceptual hole, not
  a tuning issue. Lesson: when a feature provides only a confidence
  signal and no redirect signal, GBDTs can't exploit it on
  wells where all base predictions are wrong together.
- **Fast iteration paid off as predicted.** Today's three OOF runs
  (XGB+CB aborted, LGB Exp A, LGB Exp B) cost combined ~25 min, vs
  ~75 min for three full XGB+CB runs. The per-iteration speedup is
  the right call when the *direction* of feature impact is the
  question (transferability across GBDT models is high in
  practice, regardless of magnitude differences).
