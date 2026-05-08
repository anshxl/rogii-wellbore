"""Phase 3 NN CLI — env-var-driven entry points.

MODE values:
    smoke   — 5 wells, 1 fold, 2 epochs, CPU. Sanity check on disk.
    fold    — full training of a single fold (FOLD env var, default 0).
    cv      — full 5-fold CV (M3+ milestones).

Other env vars:
    MODEL_KIND  — dummy | cnn | transformer (default: dummy)
    DATA_DIR    — overrides repo data dir
    ARTEFACT_DIR — overrides repo artefacts dir
    SEED        — default 42
    DEBUG_MAX_WELLS — for smoke runs
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.nn.train import train_one_fold
from src.nn.predict import predict_oof_fold


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_paths():
    data_dir = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
    art_root = Path(os.environ.get("ARTEFACT_DIR", REPO_ROOT / "artefacts"))
    model_kind = os.environ.get("MODEL_KIND", "dummy")
    art = art_root / "nn" / model_kind
    art.mkdir(parents=True, exist_ok=True)
    return data_dir, art, model_kind


def _list_train_wells(data_dir: Path, max_wells: int | None) -> list[str]:
    train = data_dir / "train"
    wells = sorted({
        p.name.split("__")[0] for p in train.glob("*__horizontal_well.csv")
    })
    if max_wells is not None:
        wells = wells[:max_wells]
    return wells


def main_smoke():
    data_dir, art, model_kind = _resolve_paths()
    seed = int(os.environ.get("SEED", "42"))
    max_w = int(os.environ.get("DEBUG_MAX_WELLS", "5"))
    wells = _list_train_wells(data_dir, max_w)
    if len(wells) < 2:
        raise RuntimeError(f"Smoke needs at least 2 wells; found {len(wells)}")
    train_wells = wells[:-1]
    val_wells   = wells[-1:]

    metrics = train_one_fold(
        train_wells=train_wells,
        val_wells=val_wells,
        data_dir=data_dir / "train",
        artefact_dir=art,
        model_kind=model_kind,
        n_epochs=2,
        batch_size=2,
        device="cpu",
        seed=seed,
        fold_idx=0,
    )
    print(json.dumps(metrics, indent=2, default=str))

    oof = predict_oof_fold(
        val_wells=val_wells,
        data_dir=data_dir / "train",
        checkpoint_path=art / "fold_models" / "fold_0.pt",
        fold_idx=0,
        device="cpu",
    )
    oof_path = art / "oof_smoke.parquet"
    oof.to_parquet(oof_path)
    print(f"Wrote {oof_path} with {len(oof)} rows")


if __name__ == "__main__":
    mode = os.environ.get("MODE", "smoke")
    if mode == "smoke":
        main_smoke()
    else:
        raise NotImplementedError(f"MODE={mode!r} not implemented in M1")
