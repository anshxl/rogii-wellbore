"""Read-only Optuna study monitor. Safe to run while a study is in progress."""
from __future__ import annotations
import os, sys
from pathlib import Path

import optuna

REPO = Path(__file__).resolve().parents[1]
DB   = REPO / "artefacts" / "lgb_optuna" / "lgb_optuna_v1.db"
NAME = os.environ.get("STUDY_NAME", "lgb_optuna_v1")

if not DB.exists():
    print(f"No DB at {DB}", file=sys.stderr); sys.exit(1)

s = optuna.load_study(study_name=NAME, storage=f"sqlite:///{DB}")
print(f"study: {NAME}  |  total trials: {len(s.trials)}")

try:
    print(f"best so far: {s.best_value:.5f}  (trial #{s.best_trial.number})")
except ValueError:
    print("best so far: (no completed trial yet)")

print()
state_counts = {}
for t in s.trials:
    state_counts[t.state.name] = state_counts.get(t.state.name, 0) + 1
print(f"by state: {state_counts}")
print()

print("most recent 8 trials:")
for t in s.trials[-8:]:
    iv = sorted(t.intermediate_values.items())
    folds = ", ".join(f"f{k+1}={v:.4f}" for k, v in iv) or "no folds yet"
    val   = f"{t.value:.5f}" if t.value is not None else "—"
    dur   = (t.duration.total_seconds() if t.duration else 0.0)
    print(f"  #{t.number:3d} {t.state.name:8s}  value={val}  dur={dur:.0f}s  folds=[{folds}]")
