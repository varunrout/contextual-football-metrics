"""
analysis/09_eda_sequences.py
=============================
Part 6b — Possession/Sequence EDA.

Outputs
-------
reports/figures/eda/possession_length_dist.png
reports/figures/eda/sequence_type_dist.png
reports/figures/eda/directness_vs_progression.png
reports/figures/eda/possession_start_to_final_zone.png
reports/figures/eda/possession_speed_by_sequence.png
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

from analysis._utils import load_features, save_fig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("09_eda_sequences")


def possession_length_dist(df: pd.DataFrame) -> None:
    col = next(
        (
            c
            for c in ["events_in_possession", "possession_length", "events_before_action"]
            if c in df.columns
        ),
        None,
    )
    if col is None:
        logger.warning("No possession-length column found — skipping.")
        return

    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    vals = vals.clip(upper=vals.quantile(0.99))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=50, color="#1f77b4", edgecolor="white", alpha=0.8)
    ax.axvline(vals.median(), color="red", linestyle="--", label=f"Median={vals.median():.1f}")
    ax.axvline(vals.mean(), color="orange", linestyle="--", label=f"Mean={vals.mean():.1f}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(f"Possession Length Distribution  (n={len(vals):,})")
    ax.legend()
    plt.tight_layout()
    save_fig("possession_length_dist", "eda")


def sequence_type_dist(df: pd.DataFrame) -> None:
    if "sequence_type" not in df.columns:
        logger.warning("sequence_type not found — skipping.")
        return

    counts = df["sequence_type"].astype(str).value_counts().reset_index()
    counts.columns = ["sequence_type", "count"]
    counts = counts.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, len(counts) * 0.5)))
    ax.barh(counts["sequence_type"], counts["count"], color="#ff7f0e")
    ax.set_xlabel("Event Count")
    ax.set_title("Event Count by Sequence Type")
    for _, row in counts.iterrows():
        ax.text(
            row["count"] + 5, row["sequence_type"], f"{row['count']:,}", va="center", fontsize=8
        )
    plt.tight_layout()
    save_fig("sequence_type_dist", "eda")


def directness_vs_progression(df: pd.DataFrame) -> None:
    x_col = "directness"
    y_col = "vertical_progression_speed"
    hue_col = "transition_or_settled"

    if x_col not in df.columns or y_col not in df.columns:
        logger.warning("directness or vertical_progression_speed not found — skipping scatter.")
        return

    plot_df = df[[x_col, y_col]].copy()
    if hue_col in df.columns:
        plot_df[hue_col] = df[hue_col].astype(str)
    else:
        plot_df[hue_col] = "all"

    for col in [x_col, y_col]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df.dropna()

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, grp in plot_df.groupby(hue_col):
        ax.scatter(grp[x_col], grp[y_col], label=label, alpha=0.15, s=5, rasterized=True)

    ax.set_xlabel("Directness")
    ax.set_ylabel("Vertical Progression Speed")
    ax.set_title("Directness vs Vertical Progression Speed  (by transition/settled)")
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    save_fig("directness_vs_progression", "eda")


def start_to_final_zone_flow(df: pd.DataFrame) -> None:
    start_col = "possession_start_zone"
    end_col = "final_pass_zone"

    if start_col not in df.columns or end_col not in df.columns:
        logger.warning("%s or %s not found — skipping zone flow.", start_col, end_col)
        return

    flow = (
        df.groupby([start_col, end_col], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(30)  # Top 30 flows
    )

    if flow.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    flow_pivot = flow.pivot(index=start_col, columns=end_col, values="count").fillna(0)
    im = ax.imshow(flow_pivot.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Event count")
    ax.set_xticks(range(len(flow_pivot.columns)))
    ax.set_xticklabels(flow_pivot.columns.astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(flow_pivot.index)))
    ax.set_yticklabels(flow_pivot.index.astype(str), fontsize=8)
    ax.set_xlabel(end_col)
    ax.set_ylabel(start_col)
    ax.set_title("Possession Start Zone → Final Pass Zone Flow")
    plt.tight_layout()
    save_fig("possession_start_to_final_zone", "eda")


def possession_speed_by_sequence(df: pd.DataFrame) -> None:
    speed_col = "possession_speed"
    seq_col = "sequence_type"

    if speed_col not in df.columns:
        # Fallback: try vertical_progression_speed
        speed_col = "vertical_progression_speed"
    if speed_col not in df.columns or seq_col not in df.columns:
        logger.warning("Speed or sequence_type column missing — skipping.")
        return

    plot_df = df[[seq_col, speed_col]].copy()
    plot_df[speed_col] = pd.to_numeric(plot_df[speed_col], errors="coerce")
    plot_df = plot_df.dropna()

    seq_order = (
        plot_df.groupby(seq_col)[speed_col].median().sort_values(ascending=True).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(seq_order) * 0.55)))

    data = [
        plot_df[plot_df[seq_col] == seq][speed_col]
        .clip(
            plot_df[speed_col].quantile(0.01),
            plot_df[speed_col].quantile(0.99),
        )
        .tolist()
        for seq in seq_order
    ]
    bp = ax.boxplot(data, vert=False, patch_artist=True, notch=False)
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(seq_order)))
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yticks(range(1, len(seq_order) + 1))
    ax.set_yticklabels(seq_order, fontsize=8)
    ax.set_xlabel(speed_col)
    ax.set_title(f"{speed_col} by Sequence Type")
    plt.tight_layout()
    save_fig("possession_speed_by_sequence", "eda")


def main() -> None:
    logger.info("Loading features.parquet …")
    df = load_features()

    logger.info("Possession length distribution …")
    possession_length_dist(df)

    logger.info("Sequence type distribution …")
    sequence_type_dist(df)

    logger.info("Directness vs vertical progression speed …")
    directness_vs_progression(df)

    logger.info("Possession start → final zone flow …")
    start_to_final_zone_flow(df)

    logger.info("Possession speed by sequence type …")
    possession_speed_by_sequence(df)

    logger.info("09_eda_sequences.py complete.")


if __name__ == "__main__":
    main()
