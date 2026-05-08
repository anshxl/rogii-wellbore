import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.nn.train import train_one_fold


def _build_tiny_dataset(tmp_path: Path) -> Path:
    """Make 5 synthetic train wells on disk."""
    d = tmp_path / "train"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(5):
        n = 200
        md = np.arange(n, dtype=float)
        gr = 50 + 5 * np.sin(md * 0.1) + rng.normal(0, 0.5, n)
        z = 1500 + 0.1 * md
        x = 100 + md
        y = 200 - md
        tvt = 1000 + 0.5 * md + rng.normal(0, 0.05, n)
        prefix = 50
        tvt_input = np.where(np.arange(n) < prefix, tvt, np.nan)
        df = pd.DataFrame(
            dict(MD=md, GR=gr, Z=z, X=x, Y=y, TVT_input=tvt_input, TVT=tvt)
        )
        df.to_csv(d / f"W{i}__horizontal_well.csv", index=False)
        tw = pd.DataFrame(dict(
            TVT=np.linspace(900, 1300, 300),
            GR=60 + 10 * np.sin(np.linspace(0, 6.28, 300)),
            Geology=["EGFDU"] * 100 + ["EGFDL"] * 100 + ["ANCC"] * 100,
        ))
        tw.to_csv(d / f"W{i}__typewell.csv", index=False)
    return d


def test_train_one_fold_smoke(tmp_path):
    data_dir = _build_tiny_dataset(tmp_path)
    artefact_dir = tmp_path / "artefacts"
    artefact_dir.mkdir()

    # 4-train, 1-val split
    train_wells = [f"W{i}" for i in range(4)]
    val_wells = ["W4"]

    metrics = train_one_fold(
        train_wells=train_wells,
        val_wells=val_wells,
        data_dir=data_dir,
        artefact_dir=artefact_dir,
        model_kind="dummy",
        n_epochs=2,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        seed=42,
        fold_idx=0,
    )
    assert "best_val_rmse" in metrics
    assert "train_loss_per_epoch" in metrics
    assert len(metrics["train_loss_per_epoch"]) == 2
    # Loss should not be NaN
    assert not any(np.isnan(metrics["train_loss_per_epoch"]))
    # Checkpoint must exist
    assert (artefact_dir / "fold_models" / "fold_0.pt").exists()
