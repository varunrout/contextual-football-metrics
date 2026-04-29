"""
analysis/06_bivariate_cxt.py
============================
Part 4c — Bivariate Analysis for CxT.

Target: shot_in_possession (binary proxy — possession contains ≥1 shot).
NOTE: The real CxT training target is 'possession_cxg', computed by
      compute_possession_cxg() in src/models/cxt/state_value_model.py
      during train_cxt.py — that requires fitted CxG predictions not yet
      available at this analysis stage.

Filters features.parquet to passes and carries.

Outputs
-------
reports/figures/bivariate/cxt/sip_rate_{col}.png
reports/figures/bivariate/cxt/correlation_with_sip.png
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

from analysis._utils import (
    derive_shot_in_possession,
    load_features,
    save_fig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("06_bivariate_cxt")

_CXT_TYPES = {"pass", "carry", "cross", "cutback"}
_CAT_COLS = [
    "event_type", "score_state", "sequence_type", "possession_start_zone",
    "transition_or_settled", "phase_of_play",
]
_CORR_COLS = [
    "vertical_progression_speed", "directness", "events_before_action",
    "x_location", "y_location", "distance_to_goal", "pass_length",
    "carry_distance", "progressive_distance", "score_differential",
]


def _sip_rate_bar(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns or "shot_in_possession" not in df.columns:
        return
    rates = (
        df.groupby(col, observed=True)["shot_in_possession"]
        .agg(rate="mean", n="count")
        .reset_index()
        .sort_values("rate", ascending=False)
    )
    if rates.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(rates) * 0.8), 5))
    bars = ax.bar(rates[col].astype(str), rates["rate"], color="#8c564b", alpha=0.8)
    for bar, n in zip(bars, rates["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"n={n:,}",
            ha="center", fontsize=7,
        )
    ax.set_xlabel(col)
    ax.set_ylabel("Shot-in-Possession Rate")
    ax.set_title(f"Shot-in-Possession Rate by {col}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save_fig(f"sip_rate_{col}", "bivariate/cxt")


def _correlation_with_sip(df: pd.DataFrame) -> None:
    if "shot_in_possession" not in df.columns:
        return
    avail = [c for c in _CORR_COLS if c in df.columns]
    if not avail:
        return

    corrs = {}
    for col in avail:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = df[["shot_in_possession"]].assign(feat=s).dropna()
        if len(valid) < 20:
            continue
        r = valid["feat"].corr(valid["shot_in_possession"])
        corrs[col] = round(float(r), 4)

    corrs_sorted = dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(corrs_sorted.keys())
    vals = list(corrs_sorted.values())
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
    ax.bar(names, vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r with shot_in_possession")
    ax.set_title("Numeric Feature Correlation with Shot-in-Possession")
    plt.tight_layout()
    save_fig("correlation_with_sip", "bivariate/cxt")


def main() -> None:
    logger.info("Loading features.parquet …")
    features = load_features()

    # Filter to CxT-relevant action types
    type_col = next(
        (c for c in ["event_type", "action_type"] if c in features.columns), None
    )
    if type_col:
        df = features[features[type_col].astype(str).isin(_CXT_TYPES)].copy()
        logger.info("Filtered to CxT action types: %d rows", len(df))
    else:
        logger.warning("No event_type column found — using all rows.")
        df = features.copy()

    logger.info("Deriving shot_in_possession proxy …")
    df = derive_shot_in_possession(df)

    logger.info("Shot-in-possession rate by categorical features …")
    for col in _CAT_COLS:
        _sip_rate_bar(df, col)

    logger.info("Correlation of numeric features with shot_in_possession …")
    _correlation_with_sip(df)

    logger.info("06_bivariate_cxt.py complete.")


if __name__ == "__main__":
    main()
