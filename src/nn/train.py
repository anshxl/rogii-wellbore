"""Phase 3 NN training loop — single fold."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.nn.data import WellDataset, pad_collate
from src.nn.model import DummyMLP, masked_mse


def _build_model(kind: str) -> torch.nn.Module:
    if kind == "dummy":
        return DummyMLP(n_well_features=12, hidden=64)
    if kind == "cnn":
        from src.nn.model import Model
        return Model(encoder_kind="cnn")
    raise ValueError(f"Unknown model kind: {kind!r}")


def _score_val(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    """Per-row RMSE over hidden-zone rows only, in TVT units."""
    model.train(False)  # PyTorch: switch to inference mode (equivalent to .eval())
    sse, n = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            wi = batch["well_inputs"].to(device)
            wm = batch["well_mask"].to(device)
            ti = batch["typewell_inputs"].to(device)
            tm = batch["typewell_mask"].to(device)
            y  = batch["target"].to(device)
            ym = batch["target_mask"].to(device)
            pred = model(
                well_inputs=wi, well_mask=wm,
                typewell_inputs=ti, typewell_mask=tm,
            )
            diff = (pred - y) * ym
            sse += float((diff * diff).sum().item())
            n   += float(ym.sum().item())
    return math.sqrt(sse / max(n, 1.0))


def train_one_fold(
    train_wells: list[str],
    val_wells:   list[str],
    data_dir:    Path,
    artefact_dir: Path,
    model_kind:  str = "dummy",
    n_epochs:    int = 50,
    batch_size:  int = 16,
    lr:          float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip:   float = 1.0,
    warmup_frac: float = 0.05,
    early_stop_patience: int = 10,
    device:      str = "cpu",
    seed:        int = 42,
    fold_idx:    int = 0,
    augment:     bool = True,
) -> dict:
    """Train a single fold; save best checkpoint; return metrics dict.

    `augment=False` disables random prefix-length augmentation: the train set
    uses each well's natural prefix split, the same as validation. Useful as
    a diagnostic — does the model learn the actual task at all, separated
    from augmentation noise?
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = WellDataset(train_wells, data_dir=data_dir, training=augment, seed=seed)
    val_ds   = WellDataset(val_wells,   data_dir=data_dir, training=False, seed=seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=pad_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=pad_collate, num_workers=0,
    )

    model = _build_model(model_kind).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = max(n_epochs * max(len(train_loader), 1), 1)
    warmup_steps = max(int(warmup_frac * total_steps), 1)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    out_dir = Path(artefact_dir) / "fold_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"fold_{fold_idx}.pt"

    train_loss_per_epoch: list[float] = []
    val_rmse_per_epoch:   list[float] = []
    best_val = float("inf")
    epochs_since_improve = 0
    step = 0

    epoch_bar = tqdm(range(n_epochs), desc=f"fold {fold_idx}", unit="epoch")
    for epoch in epoch_bar:
        model.train(True)
        epoch_loss, n_batches = 0.0, 0
        batch_bar = tqdm(
            train_loader,
            desc=f"  epoch {epoch+1}/{n_epochs}",
            unit="batch",
            leave=False,
        )
        for batch in batch_bar:
            for g in optim.param_groups:
                g["lr"] = lr * lr_at(step)
            optim.zero_grad(set_to_none=True)
            wi = batch["well_inputs"].to(device)
            wm = batch["well_mask"].to(device)
            ti = batch["typewell_inputs"].to(device)
            tm = batch["typewell_mask"].to(device)
            y  = batch["target"].to(device)
            ym = batch["target_mask"].to(device)
            pred = model(
                well_inputs=wi, well_mask=wm,
                typewell_inputs=ti, typewell_mask=tm,
            )
            loss = masked_mse(pred, y, ym)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            step += 1
            epoch_loss += float(loss.item())
            n_batches += 1
            batch_bar.set_postfix(loss=f"{epoch_loss / n_batches:.3f}")
        avg_loss = epoch_loss / max(n_batches, 1)
        train_loss_per_epoch.append(avg_loss)

        val_rmse = _score_val(model, val_loader, device)
        val_rmse_per_epoch.append(val_rmse)

        improved = val_rmse < best_val
        if improved:
            best_val = val_rmse
            epochs_since_improve = 0
            torch.save({
                "model_kind": model_kind,
                "state_dict": model.state_dict(),
                "val_rmse": val_rmse,
                "epoch": epoch,
            }, ckpt_path)
        else:
            epochs_since_improve += 1

        epoch_bar.set_postfix(
            train=f"{avg_loss:.3f}",
            val=f"{val_rmse:.3f}",
            best=f"{best_val:.3f}",
            patience=f"{epochs_since_improve}/{early_stop_patience}",
        )

        if not improved and epochs_since_improve >= early_stop_patience:
            break

    return {
        "fold_idx": fold_idx,
        "best_val_rmse": best_val,
        "train_loss_per_epoch": train_loss_per_epoch,
        "val_rmse_per_epoch": val_rmse_per_epoch,
        "checkpoint": str(ckpt_path),
    }
