"""Phase 3 NN inference — OOF prediction matching the GBDT parquet schema."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.nn.data import WellDataset, pad_collate, WELL_FEATURE_NAMES
from src.nn.model import DummyMLP

IS_KNOWN_IDX = WELL_FEATURE_NAMES.index("is_known_mask")


def _build_model(kind: str) -> torch.nn.Module:
    if kind == "dummy":
        return DummyMLP(n_well_features=12, hidden=64)
    if kind == "cnn":
        from src.nn.model import Model
        return Model(encoder_kind="cnn")
    raise ValueError(f"Unknown model kind: {kind!r}")


def predict_oof_fold(
    val_wells: list[str],
    data_dir: Path,
    checkpoint_path: Path,
    fold_idx: int,
    device: str = "cpu",
    batch_size: int = 4,
) -> pd.DataFrame:
    """Run inference on `val_wells` using a saved fold checkpoint.

    Returns a DataFrame in the same schema as
    `artefacts/oof_predictions.parquet`:
    `well, prediction_id, row_idx, fold, target, pred`. Rows are restricted
    to the natural hidden suffix (where `is_known_mask = 0`).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _build_model(ckpt["model_kind"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.train(False)

    ds = WellDataset(val_wells, data_dir=data_dir, training=False, seed=0)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=pad_collate, num_workers=0,
    )

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            wi = batch["well_inputs"].to(device)
            wm = batch["well_mask"].to(device)
            ti = batch["typewell_inputs"].to(device)
            tm = batch["typewell_mask"].to(device)
            pred = model(
                well_inputs=wi, well_mask=wm,
                typewell_inputs=ti, typewell_mask=tm,
            )
            wells = batch["wells"]
            target = batch["target"]
            for b, well in enumerate(wells):
                L = int(wm[b].sum().item())
                is_known = wi[b, :L, IS_KNOWN_IDX].cpu().numpy().astype(bool)
                pred_b   = pred[b, :L].cpu().numpy()
                tgt_b    = target[b, :L].cpu().numpy()
                hidden_idx = np.flatnonzero(~is_known)
                for r in hidden_idx:
                    rows.append({
                        "well": well,
                        "prediction_id": f"{well}_{int(r)}",
                        "row_idx": int(r),
                        "fold": int(fold_idx),
                        "target": float(tgt_b[r]),
                        "pred": float(pred_b[r]),
                    })
    return pd.DataFrame(rows)
