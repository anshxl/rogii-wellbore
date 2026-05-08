# CLAUDE.md — ROGII Wellbore Geology Prediction

## Project goal
Top-tier leaderboard finish. Current public LB #2 is 10.784 RMSE (a reference
solution we've reviewed but are not committing to reproduce).

## Problem
Predict `TVT` (stratigraphic position, *not* literal thickness) for the hidden
evaluation zone of ~200 horizontal wellbores. Per-row RMSE.

Each well has:
- `{WELLNAME}__horizontal_well.csv` — trajectory (MD, X, Y, Z), `GR`, `TVT_input`
  (copy of TVT, NaN in the eval zone), and in train only: target `TVT` plus
  formation-top columns `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`
- `{WELLNAME}__typewell.csv` — vertical reference log with `TVT`, `GR`, `Geology`
- `{WELLNAME}.png` — cross-section image. **Train only — does not exist for
  test wells.**

## Critical domain concepts
- **MD**: length along wellbore (monotonic). **Z (TVD)**: vertical depth.
  **TVT**: stratigraphic position — i.e., where in the typewell's column the bit
  is equivalent to. TVT lives in stratigraphy-space, not physical-space.
- **Typewell is the Rosetta stone**: a function `f(TVT) → (GR, Geology)`. Live GR
  is matched to f(·) to localize the bit's TVT.
- Each well is split into a *known prefix* (TVT_input populated) and a *hidden
  suffix* (TVT_input is NaN). Predict TVT for the hidden suffix.

## Data leakage rules — STRICT
- **Train-only columns** in horizontal CSVs: `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`,
  `EGFDL`, `BUDA` (formation tops), and `TVT`. Never use as features.
- `TVT_input` is OK as a feature (NaN in eval zone by construction).
- **GR is fully observed everywhere, including the eval zone.** GR-derived
  features that look forward (leads, centered rolling windows) are explicitly
  LEGAL. The rank-#2 solution exploits this and so should we.
- Typewell `Geology` is available everywhere. Currently unused by rank-#2.
- Cross-section PNGs exist only for train wells. They cannot be a test-time
  input, but could serve as a training-only auxiliary signal.

## Reference solution (context, not to reproduce)
Public rank #2, LB 10.784:
- Beam search (4 smoothness configs) over typewell TVT-indices
- Two particle filters: state `(TVT, dTVT/dMD)` and `(TVT+Z, rate)`; both use
  GR-vs-typewell Gaussian likelihood
- 107 hand-engineered features → LGB + XGB + CatBoost, Nelder-Mead OOF weights
- Target = `TVT - last_known_TVT` (delta), not absolute
- 5-fold GroupKFold by well; OOF ~12.3, ensemble 10.78

Known gaps in this solution we may attack later:
- No sequence model (1D CNN / transformer over GR could replace much FE)
- `Geology` column unused
- No cross-well structure
- 107 features almost certainly contains noise; selection unexplored
- Both PFs share an observation model — diversity is overstated

## Conventions
- Predict `target = TVT - last_known_TVT`; reconstruct absolute at inference.
- Never use train-only columns as features.
- GR-spanning features (leads, centered windows) are explicitly legal.
- Reproducibility: `SEED = 42`.
- Repo structure and data paths: **TBD — confirm and edit this file.**

## Journal-driven workflow
Claude maintains `JOURNAL.md` at the repo root as a chronological log of every
experiment — EDA passes, model iterations, Kaggle submissions, anything that
produced a finding or a number. Entries are append-only and ordered oldest at
top, newest at bottom.

**At the start of every Claude Code session**, read `CLAUDE.md` then read
`JOURNAL.md` end-to-end to load project context. Do not edit prior journal
entries; they are historical record.

**At the end of every experiment**, append a new entry using this template:

    ## YYYY-MM-DD — <short title>

    **What:** One or two sentences on what was done. Reference notebooks or
    other artifacts by path.

    **Findings:** Concrete numbers and observations that matter. Include CV
    scores and LB score if a submission was made. Note CV-LB gap explicitly
    when both exist.

    **Decisions / next steps:** What this changes about the plan. If nothing,
    say so.

    **Surprises:** Anything that contradicted prior assumptions in `CLAUDE.md`,
    earlier journal entries, or the rank-#2 reference. "None" is a valid value.

Keep entries tight — a journal entry is a summary, not a report. Detailed
outputs live in their own files (notebooks, findings docs, submission CSVs);
the entry references them.