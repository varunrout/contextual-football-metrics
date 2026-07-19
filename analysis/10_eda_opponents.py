"""
analysis/10_eda_opponents.py
============================
Part 6c — Opponent Context EDA.

Outputs
-------
reports/figures/eda/opponent_feature_dists.png
reports/figures/eda/opponent_elo_by_competition.png
reports/figures/eda/pressing_vs_goal_rate.png
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

from analysis._utils import (  # noqa: E402
    competition_labels,
    feature_groups,
    load_features,
    load_shots,
    save_fig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("10_eda_opponents")


def opponent_feature_distributions(features_df: pd.DataFrame) -> None:
    """Distribution plots for all opponent-group numeric features."""
    groups = feature_groups()
    opponent_cols = groups.get("opponent", [])
    if not opponent_cols:
        # Fallback: any column containing "opponent" or "opp_"
        opponent_cols = [c for c in features_df.columns if "opponent" in c or c.startswith("opp_")]

    num_cols = [
        c
        for c in opponent_cols
        if c in features_df.columns and pd.api.types.is_numeric_dtype(features_df[c])
    ]

    if not num_cols:
        logger.warning("No numeric opponent features found — skipping.")
        return

    ncols = min(4, len(num_cols))
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, num_cols, strict=False):
        vals = pd.to_numeric(features_df[col], errors="coerce").dropna()
        ax.hist(
            vals.clip(vals.quantile(0.01), vals.quantile(0.99)),
            bins=40,
            color="#e377c2",
            edgecolor="white",
            alpha=0.8,
        )
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylabel("")

    for ax in axes[len(num_cols) :]:
        ax.set_visible(False)

    fig.suptitle("Opponent Feature Distributions", fontsize=12, y=1.01)
    plt.tight_layout()
    save_fig("opponent_feature_dists", "eda")


def opponent_elo_by_competition(features_df: pd.DataFrame) -> None:
    elo_col = next(
        (c for c in features_df.columns if "opponent_team_elo" in c or "opp_elo" in c),
        None,
    )
    if elo_col is None:
        logger.warning("No opponent ELO column found — skipping.")
        return
    if "competition_id" not in features_df.columns:
        return

    labels = competition_labels()

    plot_df = features_df[["competition_id", elo_col]].copy()
    plot_df[elo_col] = pd.to_numeric(plot_df[elo_col], errors="coerce")
    plot_df = plot_df.dropna()
    plot_df["label"] = plot_df["competition_id"].apply(lambda c: labels.get(str(c), str(c)))

    comp_order = (
        plot_df.groupby("label")[elo_col].median().sort_values(ascending=True).index.tolist()
    )

    data = [plot_df[plot_df["label"] == comp][elo_col].tolist() for comp in comp_order]

    fig, ax = plt.subplots(figsize=(10, max(5, len(comp_order) * 0.5)))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6)
    colors = plt.cm.tab10(np.linspace(0, 1, len(comp_order)))
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yticks(range(1, len(comp_order) + 1))
    ax.set_yticklabels(comp_order, fontsize=8)
    ax.set_xlabel(elo_col)
    ax.set_title("Opponent ELO by Competition")
    plt.tight_layout()
    save_fig("opponent_elo_by_competition", "eda")


def pressing_vs_goal_rate(features_df: pd.DataFrame, shots_df: pd.DataFrame) -> None:
    pressing_col = next(
        (c for c in features_df.columns if "pressing" in c and "opponent" in c),
        next((c for c in features_df.columns if "pressing" in c), None),
    )
    if pressing_col is None:
        logger.warning("No opponent pressing column found — skipping.")
        return
    if "goal" not in shots_df.columns:
        return

    # Join pressing intensity to shots
    id_col = next(
        (
            c
            for c in ["match_id", "competition_id"]
            if c in features_df.columns and c in shots_df.columns
        ),
        None,
    )
    if id_col is None:
        logger.warning("No join key for pressing vs goal rate — skipping.")
        return

    press_agg = (
        features_df.groupby(id_col)[pressing_col]
        .median()
        .reset_index()
        .rename(columns={pressing_col: "pressing_median"})
    )

    shot_rates = (
        shots_df.groupby(id_col)["goal"].mean().reset_index().rename(columns={"goal": "goal_rate"})
    )

    plot_df = press_agg.merge(shot_rates, on=id_col).dropna()
    if len(plot_df) < 5:
        logger.warning("Not enough data for pressing vs goal rate scatter — skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df["pressing_median"], plot_df["goal_rate"], alpha=0.6, color="#d62728", s=40)

    # Linear trend line
    m, b = np.polyfit(plot_df["pressing_median"], plot_df["goal_rate"], 1)
    xs = np.linspace(plot_df["pressing_median"].min(), plot_df["pressing_median"].max(), 100)
    ax.plot(xs, m * xs + b, "b--", linewidth=1.5, label=f"Trend (slope={m:.4f})")

    ax.set_xlabel(f"Median {pressing_col}")
    ax.set_ylabel("Goal Rate (per match)")
    ax.set_title("Opponent Pressing Intensity vs Goal Rate")
    ax.legend()
    plt.tight_layout()
    save_fig("pressing_vs_goal_rate", "eda")


def main() -> None:
    logger.info("Loading features.parquet …")
    features = load_features()

    logger.info("Loading shots.parquet …")
    shots = load_shots()

    logger.info("Opponent feature distributions …")
    opponent_feature_distributions(features)

    logger.info("Opponent ELO by competition …")
    opponent_elo_by_competition(features)

    logger.info("Pressing intensity vs goal rate …")
    pressing_vs_goal_rate(features, shots)

    logger.info("10_eda_opponents.py complete.")


if __name__ == "__main__":
    main()
