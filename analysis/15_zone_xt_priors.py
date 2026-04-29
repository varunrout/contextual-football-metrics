"""
analysis/15_zone_xt_priors.py
==============================
Part 9 — Zone xT Prior Surface (ZoneXTBaseline).

Fits ZoneXTBaseline on features.parquet, extracts the learned value surface,
and saves zone_xt_priors.parquet for downstream CxT model initialisation.

Outputs
-------
data/features/zone_xt_priors.parquet
reports/figures/baselines/zone_xt_value_surface.png
reports/figures/baselines/zone_shot_frequency.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import load_features, save_fig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("15_zone_xt_priors")

_PRIORS_PATH = _ROOT / "data" / "features" / "zone_xt_priors.parquet"


def _fit_zone_xt(df: pd.DataFrame):
    """Instantiate and fit ZoneXTBaseline from src.models.cxt.baseline."""
    from src.models.cxt.baseline import ZoneXTBaseline

    required = ["x_location", "y_location", "event_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing for ZoneXTBaseline: {missing}")

    # goal column: required by Bellman solver
    goal_col = "goal" if "goal" in df.columns else None
    if goal_col is None:
        logger.warning("No 'goal' column — using zero vector as placeholder.")
        df = df.copy()
        df["goal"] = 0

    model = ZoneXTBaseline()
    logger.info("Fitting ZoneXTBaseline on %d rows …", len(df))
    model.fit(df)
    logger.info("ZoneXTBaseline fit complete.")
    return model


def _extract_zone_records(model) -> pd.DataFrame:
    """
    Flatten model value arrays into a tidy DataFrame with one row per zone.

    ZoneXTBaseline stores flattened arrays of length n_zones in this codebase:
      values_       — (n_zones,) xT surface
      shot_prob_    — (n_zones,) shot probability per zone
      shot_value_   — (n_zones,) expected value given shot
      transition_   — (n_zones, n_zones) transition matrix

    We reshape the 1D vectors to (n_y_bins, n_x_bins) using model config.
    """
    values = np.asarray(model.values_)
    shot_prob = np.asarray(model.shot_prob_)
    shot_value = np.asarray(model.shot_value_)

    n_x = int(getattr(model.config, "pitch_zones_x", 16))
    n_y = int(getattr(model.config, "pitch_zones_y", 12))
    expected = n_x * n_y

    if values.ndim == 1:
        if values.size != expected:
            raise ValueError(
                f"values_ size ({values.size}) does not match expected grid size ({expected})."
            )
        values = values.reshape(n_y, n_x)
    elif values.ndim == 2:
        n_y, n_x = values.shape
    else:
        raise ValueError(f"Unexpected values_ ndim: {values.ndim}")

    if shot_prob.ndim == 1:
        shot_prob = shot_prob.reshape(n_y, n_x)
    if shot_value.ndim == 1:
        shot_value = shot_value.reshape(n_y, n_x)

    records = []
    zone_id = 0
    for yi in range(n_y):
        for xi in range(n_x):
            records.append({
                "zone_id": zone_id,
                "x_bin": xi,
                "y_bin": yi,
                "xt_value": float(values[yi, xi]),
                "shot_prob": float(shot_prob[yi, xi]),
                "shot_value": float(shot_value[yi, xi]),
            })
            zone_id += 1

    return pd.DataFrame(records)


def plot_value_surface(priors_df: pd.DataFrame) -> None:
    n_x = priors_df["x_bin"].max() + 1
    n_y = priors_df["y_bin"].max() + 1

    mat = np.full((n_y, n_x), np.nan)
    for _, row in priors_df.iterrows():
        mat[int(row["y_bin"]), int(row["x_bin"])] = row["xt_value"]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        mat, origin="lower", aspect="auto", cmap="YlOrRd",
        vmin=0, vmax=np.nanpercentile(mat, 98),
        extent=[0, 105, 0, 68],
    )
    plt.colorbar(im, ax=ax, label="xT Value")
    ax.add_patch(plt.Rectangle((0, 0), 105, 68, fill=False, edgecolor="black", lw=2))
    ax.add_patch(plt.Rectangle((83, 13.85), 22, 40.3, fill=False, edgecolor="white", lw=1.5))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Zone xT Value Surface  ({n_x}×{n_y} zones)")
    plt.tight_layout()
    save_fig("zone_xt_value_surface", "baselines")


def plot_shot_frequency(priors_df: pd.DataFrame) -> None:
    n_x = priors_df["x_bin"].max() + 1
    n_y = priors_df["y_bin"].max() + 1

    mat = np.full((n_y, n_x), np.nan)
    for _, row in priors_df.iterrows():
        mat[int(row["y_bin"]), int(row["x_bin"])] = row["shot_prob"]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        mat, origin="lower", aspect="auto", cmap="Blues",
        vmin=0, vmax=np.nanpercentile(mat, 98),
        extent=[0, 105, 0, 68],
    )
    plt.colorbar(im, ax=ax, label="Shot Probability")
    ax.add_patch(plt.Rectangle((0, 0), 105, 68, fill=False, edgecolor="black", lw=2))
    ax.add_patch(plt.Rectangle((83, 13.85), 22, 40.3, fill=False, edgecolor="grey", lw=1.5))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Shot Probability by Zone  ({n_x}×{n_y})")
    plt.tight_layout()
    save_fig("zone_shot_frequency", "baselines")


def main() -> None:
    logger.info("Loading features.parquet …")
    df = load_features()

    try:
        model = _fit_zone_xt(df)
    except Exception as e:
        logger.error("ZoneXTBaseline fit failed: %s", e)
        raise

    logger.info("Extracting zone records …")
    priors_df = _extract_zone_records(model)
    logger.info("Zones extracted: %d rows", len(priors_df))
    logger.info("  xt_value range: [%.4f, %.4f]",
                priors_df["xt_value"].min(), priors_df["xt_value"].max())

    logger.info("Saving zone_xt_priors.parquet → %s", _PRIORS_PATH)
    _PRIORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    priors_df.to_parquet(_PRIORS_PATH, index=False)

    logger.info("Plotting zone xT value surface …")
    plot_value_surface(priors_df)

    logger.info("Plotting shot frequency surface …")
    plot_shot_frequency(priors_df)

    logger.info("15_zone_xt_priors.py complete.")


if __name__ == "__main__":
    main()
