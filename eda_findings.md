# EDA Phase 1 — Findings

Numbers below are produced by `notebooks/eda_phase1.ipynb` (one-pass aggregation
in `notebooks/eda_pipeline.py`, cached at `eda_outputs/well_summary.parquet`,
figures at `eda_outputs/figs/`). Run with `uv run jupyter nbconvert --execute --inplace notebooks/eda_phase1.ipynb`.

## 0. Inventory & schema

- **Train**: 773 wells. Every well has all three files (`__horizontal_well.csv`,
  `__typewell.csv`, `.png`). No missing files.
- **Test**: **only 3 wells** locally — `000d7d20`, `00bbac68`, `00e12e8b`. No
  PNGs (consistent with the data dictionary). This contradicts the
  "~200 wellbores" figure quoted in `CLAUDE.md`; the local test is clearly a
  *sanity sandbox*, not the official held-out set.
- **The 3 test files are derivative of identically-named train files.** For
  each test well, MD/X/Y/Z/GR are byte-identical to the train file, the mask
  index in `TVT_input` is identical, and the typewell `TVT`/`GR` columns are
  identical. The test versions simply strip `TVT`, the six formation-top
  columns, and the typewell's `Geology` column. Treat the local test as a
  smoke test only; never use these 3 wells in train when developing CV.
- Train horizontal columns: `MD,X,Y,Z,ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA,TVT,GR,TVT_input` (all float64).
- Train typewell columns: `TVT,GR,Geology`. Test typewell columns: `TVT,GR` only — `Geology` absent at test time. **Implication for v1**: `Geology` is a train-only signal unless the official test set still ships it (the rank-#2 reference also did not use it).
- Train PNGs are large (~5000×2800 RGBA). They cannot be inputs at test time.

## Q1 — Hidden vs known zone sizes

- Every train well has a hidden zone (no well is fully observed).
- `hidden_ratio` is tightly distributed: median 0.74, IQR 0.70–0.78, 5th–95th
  pctl 0.625–0.819, full range 0.20–0.88. Effectively, the eval zone is ~3×
  longer than the known prefix for the typical well.
- `known_len` median 1714 rows, IQR roughly 1500–1860, range 851–2392; in MD
  this is ~1700 ft of known prefix per well (5th–95th pctl 1345–2052 ft).
- `hidden_len` median ~4839 ft, range 406–10,051 ft.
- The 3 test wells fall well inside the train distribution: hidden_ratio
  0.67/0.80/0.67, known_len 1442/1545/2083.
- **Implication**: any model needs to extrapolate ~3× the length of the known
  prefix in MD-space. A purely autoregressive prefix-based model is unlikely
  to be enough on its own — the typewell anchor is essential.

## Q2 — TVT trajectory smoothness (train only, known prefix)

- `dTVT/dMD` distribution is tightly concentrated near zero with some mass on
  both sides; per-well mean |dTVT/dMD| median ≈ 0.45 (i.e., the bit moves
  ~0.45 stratigraphic units per MD-foot on average).
- `d²TVT/dMD²` is sharply peaked at zero with heavy tails — TVT is mostly
  smooth with occasional curvature events.
- The naive 8×MAD jump detector flags 94% of wells as having "≥1 jump", which
  is a false-positive cliff and not a useful threshold. A more conservative
  proxy: wells with `abs_dtvtdmd_max > 5` (i.e., one MD-step where TVT changes
  by ≥5 units) are the *real* fault candidates — that flags only ~1 well in
  this dataset (cluster 3 in Q5, median `abs_dtvtdmd_max` = 151). Most wells
  have `abs_dtvtdmd_max` between 1 and 3, consistent with smooth dipping
  laterals.
- **Implication**: an alignment / smoothness prior on TVT is well-justified.
  Hard discontinuities (true faults) are rare; v1 does not need a special
  fault-handling mechanism.

## Q3 — GR scale consistency

- Train horizontal GR (across-well averages): mean of per-well means 88.1,
  mean of per-well stds 17.9, with substantial across-well variation
  (sd-of-means 15.4, sd-of-stds 4.0).
- Train typewells: similar mean-of-means (~88) but lower sd-of-means.
- **Per-well horizontal-vs-typewell calibration**:
  - Mean offset (`gr_pref_mean - tw_gr_mean`): median ≈ 0, but with noticeable
    spread; the central 90% lies roughly within ±20 GR units. Six clusters
    later separate primarily on this axis.
  - Std ratio (`gr_pref_std / tw_gr_std`): median ≈ 0.9, with several wells
    ratio ≫ 1 (lateral noisier than typewell) and some ratio ≪ 1.
- Train vs test GR (only n=3 test wells, so use cautiously): test means are
  inside the train distribution but with very small variance (test GR-mean
  range 85.0–88.9 vs train 44–135). Nothing alarming.
- **Implication**: a per-well GR normalization (subtract horizontal-vs-typewell
  offset, optionally scale to typewell std) is a near-mandatory preprocessing
  step before any GR-similarity model. The rank-#2 reference also normalizes;
  we should too.

## Q4 — Typewell coverage

- 760 of 773 train wells (98.3%) have their `[TVT_input_min, TVT_input_max]`
  fully inside `[typewell.TVT_min, typewell.TVT_max]`.
- The typical low-side margin (`TVT_input_min - typewell_TVT_min`) is ~12,
  and the high-side margin (`typewell_TVT_max - TVT_input_max`) is ~150
  median, 5th–95th pctl 84–248. The typewell typically has lots of headroom
  above the deepest known TVT but only a thin cushion below.
- **13 wells exit the typewell**: 10 of them on the low side by very large
  amounts (-117 to -645 stratigraphic units, e.g. `02e7fe5a`, `10b89021`,
  `3417285d`, `bc4381e2`, `ecdab904`, …). These 10 wells *all share the same
  typewell file* (md5 `a23359a1…`, see Q6), suggesting the typewell here is
  too short. A few wells exit on the high side by tiny amounts (≤ −1.3).
- All 3 test wells are fully inside their typewells.
- **Implication**: for ~98% of wells we can rely on typewell coverage. For the
  shared-typewell outliers, an extrapolation strategy on the low side will be
  needed (or those wells should be flagged as a separate handling case).

## Q5 — Per-well difficulty proxies & clustering

KMeans on standardized features (`hidden_len, known_len, abs_dtvtdmd_mean,
abs_dtvtdmd_max, d2tvtdmd2_std, n_jumps_3sigma, gr_scale_offset,
gr_scale_ratio, gr_ks_proxy, cov_low_margin, cov_high_margin`) — silhouette
peaks at k=6 (0.20). Cluster medians:

| cluster | size | gr_offset | gr_ratio | cov_low | abs_d2_std | abs_dtvtdmd_max | character |
|---|---|---|---|---|---|---|---|
| 0 | 296 | -1.6 | 0.78 |  12.3 | 0.02 | 1.35 | calibrated, smooth |
| 1 | 167 | -3.4 | 0.76 |  12.3 | 0.01 | 1.06 | calibrated, very smooth |
| 2 | 250 | +5.5 | 0.72 |  12.4 | 0.02 | 1.13 | mild positive offset |
| 3 |   1 | +18  | 1.09 | -118  | 5.02 | 151  | **single fault-like outlier** |
| 4 |  10 | +12.6| 0.83 | -453  | 0.01 | 1.02 | shared-typewell low-side exits |
| 5 |  49 | -16.3| 0.98 |  12.2 | 0.01 | 1.13 | strong negative offset |

- ~85% of wells are "easy": calibrated GR, smooth TVT, lateral inside typewell.
- The non-trivial groups are cluster 4 (typewell-coverage failure, n=10) and
  cluster 3 (an isolated true fault candidate, n=1).
- **Implication**: most of the leaderboard error budget will come from a small
  set of wells. A v1 model that does well on the bulk (~700 easy wells) will
  not beat #2 by chasing them — improvements over #2 will likely require
  better handling of the harder edge cases or a structurally different model.

## Q6 — Spatial layout

- All 773 train + 3 test centroids occupy a roughly 180k × 130k bounding box
  in the (X, Y) coordinate system (likely state-plane feet).
- DBSCAN at eps = 0.5% of the bounding box (≈893 ft) finds 106 clusters and
  274 noise points — i.e., wells are organized into many small pads with a
  long tail of singletons. Reasonable as an initial pad assignment for
  GroupKFold-style validation.
- Each test well's nearest train well is the same-named train well at distance
  0 (confirming the train↔test duplication noted in §0). The next-nearest
  *distinct* train neighbors sit 370–2400 ft away — i.e., test wells are not
  geographic outliers; they sit in well-populated pads.
- **Typewell file sharing**: 13 typewell file-content hashes are shared by ≥2
  wells, accounting for 33 wells in total. The largest shared group is 10
  wells (md5 `a23359a1…`) — the same 10 wells that fail typewell coverage in
  Q4. **Implication**: GroupKFold should group on shared typewell hash AND/OR
  pad cluster, not just on individual well, to avoid optimistic CV.

## Q7 — Train/test distribution alignment

(Cautious: only n=3 test wells locally.)

| feature | train mean (std) | test mean (std) | aligned? |
|---|---|---|---|
| known_len | 1692 (217) | 1690 (344) | yes |
| hidden_len | 4895 (1301) | 4717 (1147) | yes |
| hidden_ratio | 0.73 (0.06) | 0.73 (0.06) | yes |
| md_known_span | 1691 | 1689 | yes |
| GR mean (pref) | 88.1 (13.7) | 87.1 (1.98) | within range |
| Z min | -9530 (610) | -9759 (422) | within range |
| X mean | 2.967e6 (47k) | 2.987e6 (20k) | within range |
| Y mean | 1.076e6 (35k) | 1.074e6 (12k) | within range |

No covariate shift signals at this sample size.

## Sanity / leakage audit

Per-test-well audit (`leakage_audit()`):

- `000d7d20`: leak-cols-present=[], first_nan=1442, n_total=5278, n_nan=3836, before_clean=True, after_all_nan=True.
- `00bbac68`: leak-cols-present=[], first_nan=1545, n_total=7559, n_nan=6014, before_clean=True, after_all_nan=True.
- `00e12e8b`: leak-cols-present=[], first_nan=2083, n_total=6384, n_nan=4301, before_clean=True, after_all_nan=True.

✓ No formation-top columns in test horizontals. ✓ `TVT_input` is non-NaN
exactly on the prefix, NaN exactly on the suffix. ✓ No PNGs for test wells.
✓ Spot-checked train PNGs are RGBA ~5000×2800 — exist and openable.

---

## Phase 2 — per-well diagnosis of the worst 20 wells (2026-05-08)

Driver: [notebooks/eda_phase2.ipynb](notebooks/eda_phase2.ipynb) +
[src/eda_phase2_plots.py](src/eda_phase2_plots.py). 25 figures rendered to
[eda_outputs/figs/phase2/](eda_outputs/figs/phase2/) — top-20 worst wells by
ensemble OOF RMSE (tag-grouped CV) plus 5 wells closest to the median RMSE
as visual contrast.

### Failure mode: confirmed look-alike-layer degeneracy

Panel B (typewell GR vs TVT, with shaded bands for truth and each aligner's
predicted TVT range) makes the failure mode visible directly:

- On every worst well inspected (86454a6f, 1b1eba53, ba48188d, c8d9680c,
  2fd68f7b, 5f4d2a52, 7e721392, 8f201368, 389ae58f, 708caea9, 91db7070,
  fef8af96, 77e4821c, f6d009f4 — 14/20), the truth band and the stacked
  aligner bands are **narrow and adjacent**, typically 30–50 TVT units
  apart, and **both lie over typewell regions with visually similar GR
  character** (similar mean/spread, similar peak structure, similar
  baseline). The aligners are matching a look-alike layer.
- All 7 aligners (4 beam configs, pf, ancc, dtw) cluster *together* —
  they agree with each other and disagree with truth in lockstep. That's
  the structural signature of a shared GR-magnitude likelihood.
- Bias direction is mixed: of the 14 worst inspected, 9 negative (predicted
  too shallow), 5 positive (too deep). Not a directional offset, just a
  layer-locking ambiguity.

### Easy-well contrast

- 47222616 (RMSE 7.6, bias −0.18), 5aa03df7 (RMSE 7.6, bias +5.7),
  dc7f9757 (RMSE 7.6, bias +5.0): truth and aligner bands **overlap** in
  Panel B, and the relevant typewell region contains a distinctive feature
  (sharp GR transition, plateau boundary, or unique low-GR trough) close
  to where the bands sit.
- Implication: aligners do *not* fail on hard wells; they fail in
  low-distinctiveness GR neighborhoods of the typewell. The signal that
  separates "easy" from "hard" is the *local distinctiveness* of the
  truth-region's GR pattern within the typewell.

### Trajectory shape disagreement (Panel C)

Several worst wells show qualitatively wrong trajectory shape, not just an
offset:

- 1b1eba53, c8d9680c, 91db7070, 5f4d2a52 — aligners track essentially
  flat through the hidden zone while truth dips by 30–50 units.
- ba48188d, 7e721392 — aligners over-extrapolate the prefix slope while
  truth flattens.

The aligners are not getting dipping-rate signal from GR — they inherit it
from the prefix slope or the smoothness prior. A likelihood that responds
to *trend* in GR (not just magnitude) would constrain trajectory shape.

### Named alternate-observation candidates for the new aligner

CSV-derivable, test-time legal:

**(A) `dGR/dMD` likelihood.** Compute the gradient of GR along MD on the
horizontal and along TVT on the typewell. Match candidate TVTs by gradient
sign and magnitude. Directly attacks the look-alike degeneracy when
adjacent typewell regions have similar GR magnitudes but different
*dynamics* (rising vs falling vs flat). Best evidence: 91db7070 and
5f4d2a52, where truth has a clear slope through a region that aligners
treat as flat.

**(B) Low-pass (multi-scale) GR likelihood.** Smooth typewell GR with a
large radius (σ ≈ 50–100 ft TVT), and match against horizontal GR smoothed
at the same scale. Separates the slow trend (more diagnostic for absolute
TVT position) from high-frequency noise. Pairs naturally with (A).

**Recommendation:** build a DTW variant on top of the existing DTW
implementation that uses a composite likelihood combining (A) + (B), with
a small weight on the existing GR-magnitude likelihood to preserve
high-frequency matching where it helps. Justified by the panels: every
worst-well failure has a clean look-alike-layer signature in Panel B that
either trend (A) or scale (B) should disambiguate.

### Recorded-but-deferred candidates

- **Typewell `Geology`-segment-aware likelihood.** The typewell carries a
  `Geology` column unused by all current aligners. If the truth band and
  predicted bands overlap *different* geology segments, geology is a free
  disambiguator. Worth a v2 EDA pass that adds geology bands to Panel B
  before committing to a separate aligner.
- **PNG-derived features.** Train-only `data/train/{well}.png` files
  contain GR, vertical-plan, and per-well TVT plots. Out of scope for this
  EDA; recorded for a possible training-only auxiliary signal experiment.

### Geology bands on Panel B — decisive negative result

Re-rendered Panel B with the typewell `Geology` column overlaid as a colored
strip at the top of the panel, then computed per-row geology agreement
between truth-TVT and ensemble-predicted-TVT for the worst 20 wells.

| metric | value |
|---|---|
| Worst wells where truth & pred share at least one geology | 20/20 |
| Worst wells where 100% of rows have same geo for truth & pred | 14/20 |
| Median `pct_rows_same_geo` | 1.00 |

**Geology does not disambiguate the dominant failure.** EGFDL is the
target zone for the overwhelming majority of worst wells, and both
truth and predicted bands sit *inside* EGFDL. The look-alike-layer
ambiguity is *within* a geology segment, not across them.

Wells where geology would help: `389ae58f` (truth spans EGFDL/LBHL/LTGT/LTHL,
pred only EGFDL — 18% same-geo), `91db7070` (truth in EGFDL/MNSS, 67%),
`5f4d2a52` (truth in BUDA/Clay/EGFDL, 64%), plus a few smaller deviations
(`f6d009f4`, `a959858c`, `ba48188d`). That's ~5 of 20 — a real but minor
secondary signal.

**Decision: skip geology as a likelihood term in the new aligner.** Within-
EGFDL ambiguity is the dominant failure (~75% of worst wells); geology
boundaries don't constrain it. Original A+B (gradient + low-pass GR)
composite stands as the design. Geology is recorded as a possible v2
refinement aimed at the ~25% boundary-spanning subset, after the gradient/
low-pass aligner is in place.
