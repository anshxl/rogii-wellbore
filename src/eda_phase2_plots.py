"""Phase 2 EDA: per-well 3-panel diagnostic plot.

Panel A: horizontal GR vs MD, with end-of-known-prefix marker.
Panel B: typewell GR vs TVT, with shaded TVT-range bands for truth (green)
         and each aligner (one band per aligner column in train_df).
Panel C: TVT vs MD trajectory with all aligner paths + ensemble + truth +
         TVT_input prefix.

Driven by notebooks/eda_phase2.ipynb.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ALIGNERS: list[tuple[str, str, str]] = [
    ("beam_tight_delta", "beam_tight", "#9ecae1"),
    ("beam_cons_delta", "beam_cons", "#4292c6"),
    ("beam_loose_delta", "beam_loose", "#2171b5"),
    ("beam_vloose_delta", "beam_vloose", "#084594"),
    ("pf_delta", "pf", "#fd8d3c"),
    ("ancc_delta", "ancc", "#9467bd"),
    ("dtw_delta", "dtw", "#d62728"),
]

# Stable color per geology label so segments are comparable across wells.
GEOLOGY_COLORS: dict[str, str] = {
    "ANCC": "#1f77b4",
    "ASTNU": "#ff7f0e",
    "ASTNL": "#2ca02c",
    "EGFDU": "#d62728",
    "EGFDL": "#9467bd",
    "BUDA": "#8c564b",
    "AC_UEF_BHL": "#e377c2",
    "AC_UEF_THL": "#7f7f7f",
    "AC_UEF_TRGT": "#bcbd22",
    "Clay Rich Interval": "#17becf",
    "LBHL": "#aec7e8",
    "LTGT": "#ffbb78",
    "LTHL": "#98df8a",
    "MNSS": "#ff9896",
    "OLMOS": "#c5b0d5",
    "UEGFD BHL": "#c49c94",
    "UEGFD TGT": "#f7b6d2",
    "UEGFD THL": "#dbdb8d",
}


def _geology_segments(tw: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Return contiguous (label, tvt_lo, tvt_hi) runs from the typewell."""
    if "Geology" not in tw.columns:
        return []
    s = tw["Geology"].fillna("(none)")
    runs = s.ne(s.shift()).cumsum()
    out: list[tuple[str, float, float]] = []
    for _, grp in tw.groupby(runs, sort=True):
        label = grp["Geology"].iloc[0]
        if pd.isna(label):
            continue
        out.append((str(label), float(grp["TVT"].min()), float(grp["TVT"].max())))
    return out


def _data_dir() -> Path:
    return Path("data/train")


def load_ensemble_weights(path: Path = Path("artefacts/ensemble_weights.json")) -> dict[str, float]:
    return json.loads(path.read_text())


def load_well_frames(
    well_id: str,
    train_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    ens_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (horizontal_csv, typewell_csv, per_row_df_with_predictions)."""
    data_dir = _data_dir()
    h = pd.read_csv(data_dir / f"{well_id}__horizontal_well.csv")
    tw = pd.read_csv(data_dir / f"{well_id}__typewell.csv")

    tdf = (
        train_df[train_df.well == well_id]
        .sort_values("md")
        .reset_index(drop=True)
    )
    oof = oof_df[oof_df.well == well_id].set_index("prediction_id")

    out = tdf[["prediction_id", "md", "gr", "last_known_tvt"]].copy()
    for col, name, _ in ALIGNERS:
        out[name] = tdf["last_known_tvt"] + tdf[col]

    joined = tdf[["prediction_id", "last_known_tvt", "target"]].merge(
        oof[["oof_xgb", "oof_cb"]],
        on="prediction_id",
        how="left",
    )
    out["ensemble"] = (
        ens_weights["xgb"] * joined["oof_xgb"]
        + ens_weights["cb"] * joined["oof_cb"]
        + joined["last_known_tvt"]
    ).values
    out["truth"] = (joined["target"] + joined["last_known_tvt"]).values
    return h, tw, out


def _plot_panel_a(ax: plt.Axes, h: pd.DataFrame) -> None:
    pre_mask = h["TVT_input"].notna()
    end_known = h.loc[pre_mask, "MD"].max()
    ax.plot(h["MD"], h["GR"], color="black", linewidth=0.6)
    ax.axvline(
        end_known,
        color="grey",
        linestyle="--",
        linewidth=0.8,
        label=f"end of known prefix (MD={end_known:.0f})",
    )
    ax.set_xlabel("MD")
    ax.set_ylabel("GR (horizontal)")
    ax.set_title("Panel A — horizontal GR vs MD")
    ax.legend(loc="upper right", fontsize=7)


def _plot_panel_b(ax: plt.Axes, tw: pd.DataFrame, out: pd.DataFrame) -> None:
    ax.plot(tw["TVT"], tw["GR"], color="black", linewidth=0.6)

    truth_lo, truth_hi = out["truth"].min(), out["truth"].max()
    ax.axvspan(
        truth_lo,
        truth_hi,
        color="#2ca02c",
        alpha=0.30,
        label=f"truth [{truth_lo:.0f}, {truth_hi:.0f}]",
    )

    for _, name, color in ALIGNERS:
        lo, hi = out[name].min(), out[name].max()
        ax.axvspan(lo, hi, color=color, alpha=0.18)
        ax.plot([], [], color=color, linewidth=8, alpha=0.4, label=name)

    # Geology strip at top of panel: thin colored bar per contiguous segment.
    segments = _geology_segments(tw)
    seen_geos: list[str] = []
    for label, lo, hi in segments:
        color = GEOLOGY_COLORS.get(label, "#cccccc")
        ax.axvspan(lo, hi, ymin=0.94, ymax=1.0, color=color, alpha=0.85)
        if label not in seen_geos:
            seen_geos.append(label)
    # Legend entries for geology (only labels actually present in this typewell).
    for label in seen_geos:
        ax.plot(
            [],
            [],
            color=GEOLOGY_COLORS.get(label, "#cccccc"),
            linewidth=8,
            alpha=0.85,
            label=f"geo: {label}",
        )

    ax.set_xlabel("TVT (typewell)")
    ax.set_ylabel("GR (typewell)")
    ax.set_title("Panel B — typewell GR vs TVT (look-alike check, geology strip on top)")
    ax.legend(loc="upper right", fontsize=6, ncol=3)


def _plot_panel_c(ax: plt.Axes, h: pd.DataFrame, out: pd.DataFrame) -> None:
    pre_mask = h["TVT_input"].notna()
    ax.plot(
        h.loc[pre_mask, "MD"],
        h.loc[pre_mask, "TVT_input"],
        color="black",
        linewidth=1.4,
        label="TVT_input (known prefix)",
    )
    ax.plot(out["md"], out["truth"], color="#2ca02c", linewidth=1.6, label="truth")

    for _, name, color in ALIGNERS:
        ax.plot(out["md"], out[name], color=color, linewidth=0.7, alpha=0.7, label=name)

    ax.plot(
        out["md"],
        out["ensemble"],
        color="red",
        linewidth=1.3,
        linestyle="--",
        label="ensemble",
    )
    ax.set_xlabel("MD")
    ax.set_ylabel("TVT")
    ax.set_title("Panel C — TVT vs MD trajectories")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.invert_yaxis()  # geological convention: higher TVT = lower in column


def plot_well(
    well_id: str,
    train_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    ens_weights: dict[str, float],
    per_well_oof: pd.DataFrame,
) -> plt.Figure:
    h, tw, out = load_well_frames(well_id, train_df, oof_df, ens_weights)
    info = per_well_oof[per_well_oof["well"] == well_id].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(13, 13), constrained_layout=True)
    _plot_panel_a(axes[0], h)
    _plot_panel_b(axes[1], tw, out)
    _plot_panel_c(axes[2], h, out)

    fig.suptitle(
        f"{well_id}  |  rmse_ens={info['rmse_ens']:.2f}  "
        f"bias_ens={info['bias_ens']:+.2f}  "
        f"hidden_ratio={info['hidden_ratio']:.2f}  n_rows={info['n_rows']}",
        fontsize=12,
        y=1.01,
    )
    return fig


def render_wells(
    well_ids: list[str],
    out_dir: Path = Path("eda_outputs/figs/phase2"),
    train_df_path: Path = Path("artefacts/train_df.parquet"),
    oof_path: Path = Path("artefacts/oof_predictions.parquet"),
    per_well_oof_path: Path = Path("artefacts/per_well_oof.parquet"),
    weights_path: Path = Path("artefacts/ensemble_weights.json"),
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(train_df_path)
    oof_df = pd.read_parquet(oof_path)
    per_well_oof = pd.read_parquet(per_well_oof_path)
    ens_weights = load_ensemble_weights(weights_path)

    paths: list[Path] = []
    for well_id in well_ids:
        fig = plot_well(well_id, train_df, oof_df, ens_weights, per_well_oof)
        out_path = out_dir / f"{well_id}.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)
    return paths
