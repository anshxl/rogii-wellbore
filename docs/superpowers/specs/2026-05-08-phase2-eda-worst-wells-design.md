# Phase 2 EDA: per-well diagnosis of the worst 20 wells

**Date:** 2026-05-08
**Status:** Approved design, ready for implementation plan

## Context

Phase 1 (see [JOURNAL.md](../../../JOURNAL.md)) ended with two LB submissions:
XGB+CB ensemble at public LB 10.364 (rank #20) and tuned-LGB at 10.531. The
journal's per-well OOF analysis identified a stable failure mode: ~20 wells
with `|bias_ens|` of 25–55 RMSE units that all 7 aligners (beam × 4, particle
filter × 2, DTW × 1) get wrong in the same direction. They share the same
GR-magnitude likelihood, so adding a 7th aligner with that same likelihood
(DTW) helped via feature diversification but did not fix the right tail.

The Phase 2 anchor item is a **different-observation aligner** (Tier A in the
journal). Before designing it, we need to look at the worst wells directly and
identify which alternate observation would actually disambiguate them.

## Goal

Produce a per-well visual diagnosis of the 20 worst wells by ensemble OOF
RMSE + 5 easy contrast wells (RMSE near median). Output: figures + a short
findings update naming the CSV-derivable alternate observation(s) we will
build into the new aligner.

## Well selection

- **Worst 20**: top 20 by `ensemble_rmse` in
  [artefacts/per_well_oof.parquet](../../../artefacts/per_well_oof.parquet),
  using the tag-grouped CV from the 2026-05-08 DTW entry.
- **Easy 5**: 5 wells whose `ensemble_rmse` is closest to the median, as a
  visual control for "what normal looks like".
- Both groups are written into the notebook explicitly (well-id list at the
  top) so reruns are deterministic.

## Plot per well (3 panels, single figure)

### Panel A — Horizontal GR vs MD
The bit-side observed signal. Line plot. Vertical dashed line at the end of
the known prefix (last non-NaN `TVT_input`). Title annotation: well id,
ensemble RMSE, `bias_ens`, `hidden_ratio`.

### Panel B — Typewell GR vs TVT (the look-alike check, load-bearing)
Line plot of typewell GR as a function of typewell TVT. Overlay:
- **Green shaded band**: the *true* TVT range traversed in the hidden zone,
  `[min(true_TVT_hidden), max(true_TVT_hidden)]`.
- **Red shaded bands**: the predicted TVT range from each aligner column
  exposed in `train_df.parquet` (e.g., beam_cons, beam_loose, pf, ancc, dtw —
  exact set determined by inspecting the parquet at implementation time).
  Thin and partially transparent so overlap is visible.

If the green and red bands sit over visually similar GR shapes at different
TVTs, that confirms the look-alike-layer failure mode and indicates which
GR feature (shape, derivative, scale) would disambiguate.

### Panel C — TVT vs MD trajectory
Line plot. Each aligner's predicted TVT path (same set of columns as Panel
B's red bands) plus the ensemble prediction, the true TVT, and the
`TVT_input` prefix. Shows what the residual `bias_ens` of ±35 actually looks
like along the lateral and where each aligner diverges from truth.

## Inputs

- [artefacts/per_well_oof.parquet](../../../artefacts/per_well_oof.parquet)
  — for ranking and selection.
- [artefacts/oof_predictions.parquet](../../../artefacts/oof_predictions.parquet)
  — per-row OOF predictions.
- [artefacts/train_df.parquet](../../../artefacts/train_df.parquet) — per-row
  outputs for each aligner (beam_cons, beam_loose, pf, ancc, dtw) and
  `tw_gr_at_*` columns.
- Raw `data/train/{well}__horizontal_well.csv` — for `MD`, `GR`,
  `TVT_input`, true `TVT`.
- Raw `data/train/{well}__typewell.csv` — for `TVT` and `GR`.

## Deliverables

- [notebooks/eda_phase2.ipynb](../../../notebooks/eda_phase2.ipynb) —
  selects the 25 wells and renders panels via the helper.
- [src/eda_phase2_plots.py](../../../src/eda_phase2_plots.py) — plotting
  helper with one function per panel and a `plot_well(well_id) -> Figure`
  composer.
- 25 PNGs in
  [eda_outputs/figs/phase2/](../../../eda_outputs/figs/phase2/), one per well.
- Findings appended to
  [eda_findings.md](../../../eda_findings.md): named CSV-derivable patterns
  observed across the 20 bad wells, named alternate-observation
  candidate(s), evidence cited per pattern.
- Journal entry in [JOURNAL.md](../../../JOURNAL.md) summarizing the
  above.

## Out of scope

- The new aligner implementation (separate spec, written after this EDA
  decides the observation).
- Derivatives, multi-scale GR, and Geology-band overlays in the panels.
  Deferred to a v2 EDA pass if the look-alike panel doesn't make the
  pattern obvious.
- PNG-derived features. The cross-section PNGs in `data/train/{well}.png`
  contain GR and per-well TVT plots that may be useful as a training-only
  auxiliary signal, but are out of scope here. Recorded for a future
  experiment.

## Decision criterion

The pass succeeds if we can name 1–2 concrete CSV-derivable alternate
observations to try in the new aligner — for example: "dGR/dMD with σ=10 ft
smoothing", "low-pass GR at 50-MD radius", "geology-segment-aware
likelihood" — with cited evidence from specific wells' panels. If the
patterns are diffuse, we widen to derivatives / multi-scale / Geology bands
in a v2 EDA pass before committing to an aligner spec.
