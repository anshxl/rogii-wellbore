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
    AUGMENT     — on (default) | off. When off, train set uses natural prefix
                  split (no random prefix augmentation). Diagnostic only.
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


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main_fold():
    """Train one fold using assign_groups for the split. FOLD env var picks fold idx."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from src.baseline import assign_groups, TRAIN_DIR

    data_dir, art, model_kind = _resolve_paths()
    seed = int(os.environ.get("SEED", "42"))
    fold_idx = int(os.environ.get("FOLD", "0"))
    n_epochs = int(os.environ.get("N_EPOCHS", "50"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    device = os.environ.get("DEVICE", "cuda" if _cuda_available() else "cpu")
    augment_raw = os.environ.get("AUGMENT", "on").strip().lower()
    augment = augment_raw not in {"off", "0", "false", "no"}
    print(f"[main_fold] fold={fold_idx} model={model_kind} epochs={n_epochs} "
          f"batch={batch_size} device={device} augment={'on' if augment else 'OFF (diagnostic)'}")

    wells = _list_train_wells(data_dir, max_wells=None)

    # Build a df with per-row x/y so assign_groups can compute pad centroids.
    train_dir = data_dir / "train"
    chunks = []
    for w in wells:
        hw = train_dir / f"{w}__horizontal_well.csv"
        chunk = pd.read_csv(hw, usecols=["X", "Y"])
        chunk["well"] = w
        chunk = chunk.rename(columns={"X": "x", "Y": "y"})
        chunks.append(chunk[["well", "x", "y"]])
    df = pd.concat(chunks, ignore_index=True)
    df = assign_groups(df, data_dir=TRAIN_DIR)

    # One group label per well (assign_groups sets group_id per row).
    well_group = df.groupby("well")["group_id"].first().reset_index()
    well_group.columns = ["well", "group"]

    # Same fold construction as src/baseline.py: GroupKFold over `group`.
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(n_splits=5).split(well_group, groups=well_group["group"]))
    train_idx, val_idx = folds[fold_idx]
    train_wells = well_group.iloc[train_idx]["well"].tolist()
    val_wells   = well_group.iloc[val_idx]["well"].tolist()

    metrics = train_one_fold(
        train_wells=train_wells,
        val_wells=val_wells,
        data_dir=data_dir / "train",
        artefact_dir=art,
        model_kind=model_kind,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
        seed=seed,
        fold_idx=fold_idx,
        augment=augment,
    )
    metrics["augment"] = augment
    metrics_path = art / f"fold_{fold_idx}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Wrote {metrics_path}")

    oof = predict_oof_fold(
        val_wells=val_wells,
        data_dir=data_dir / "train",
        checkpoint_path=art / "fold_models" / f"fold_{fold_idx}.pt",
        fold_idx=fold_idx,
        device=device,
    )
    oof_path = art / f"oof_fold_{fold_idx}.parquet"
    oof.to_parquet(oof_path)
    print(f"Wrote {oof_path} ({len(oof)} rows)")


if __name__ == "__main__":
    mode = os.environ.get("MODE", "smoke")
    if mode == "smoke":
        main_smoke()
    elif mode == "fold":
        main_fold()
    else:
        raise NotImplementedError(f"MODE={mode!r} not implemented")
