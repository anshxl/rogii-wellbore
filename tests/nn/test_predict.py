import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.nn.train import train_one_fold
from src.nn.predict import predict_oof_fold
from tests.nn.test_train import _build_tiny_dataset


def test_predict_oof_fold_returns_expected_schema(tmp_path):
    data_dir = _build_tiny_dataset(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    train_one_fold(
        train_wells=["W0", "W1", "W2", "W3"],
        val_wells=["W4"],
        data_dir=data_dir,
        artefact_dir=art,
        model_kind="dummy",
        n_epochs=2,
        batch_size=2,
        device="cpu",
        seed=42,
        fold_idx=0,
    )
    df = predict_oof_fold(
        val_wells=["W4"],
        data_dir=data_dir,
        checkpoint_path=art / "fold_models" / "fold_0.pt",
        fold_idx=0,
        device="cpu",
    )
    assert set(df.columns) >= {
        "well", "prediction_id", "row_idx", "fold", "target", "pred",
    }
    assert (df["fold"] == 0).all()
    # W4 has 200 rows, 50 prefix, 150 hidden → 150 OOF rows
    assert len(df) == 150
