"""
analysis/08_eda_shots.py
========================
Part 6a — Deep Shot / CxG EDA.

Outputs
-------
reports/figures/eda/shot_density_map.png
reports/figures/eda/goal_rate_heatmap.png
reports/figures/eda/xg_calibration_curve.png
reports/figures/eda/goal_rate_by_minute.png
reports/figures/eda/goal_rate_by_competition.png
reports/figures/eda/360_vs_non360_distributions.png
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
    competition_labels,
    load_events,
    load_shots,
    save_fig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("08_eda_shots")

_PITCH_X = 105.0
_PITCH_Y = 68.0
_ZONE_X = 16
_ZONE_Y = 12


def shot_density_map(shots: pd.DataFrame) -> None:
    x = pd.to_numeric(shots["x_location"], errors="coerce").dropna()
    y_vals = pd.to_numeric(shots["y_location"], errors="coerce")
    valid = pd.concat([x, y_vals], axis=1).dropna()

    if valid.empty:
        logger.warning("No valid shot locations — skipping shot density map.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    h = ax.hexbin(
        valid["x_location"], valid["y_location"],
        gridsize=30, cmap="YlOrRd", mincnt=1,
    )
    plt.colorbar(h, ax=ax, label="Shot count")

    # Pitch outline
    ax.add_patch(plt.Rectangle((0, 0), 105, 68, fill=False, edgecolor="black", lw=2))
    ax.add_patch(plt.Rectangle((83, 13.85), 22, 40.3, fill=False, edgecolor="grey", lw=1.5))
    ax.add_patch(plt.Rectangle((94.2, 24.85), 10.8, 18.3, fill=False, edgecolor="grey", lw=1.0))

    ax.set_xlim(-2, 107)
    ax.set_ylim(-2, 70)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Shot Density Heatmap  (n={len(valid):,})")
    plt.tight_layout()
    save_fig("shot_density_map", "eda")


def goal_rate_heatmap(shots: pd.DataFrame) -> None:
    if "goal" not in shots.columns:
        return

    x = pd.to_numeric(shots["x_location"], errors="coerce")
    y = pd.to_numeric(shots["y_location"], errors="coerce")
    g = pd.to_numeric(shots["goal"], errors="coerce")
    valid = pd.concat([x, y, g], axis=1).dropna()

    if valid.empty:
        return

    # Bin into 16×12 grid
    valid["xbin"] = pd.cut(valid["x_location"], bins=_ZONE_X, labels=False)
    valid["ybin"] = pd.cut(valid["y_location"], bins=_ZONE_Y, labels=False)

    grid = (
        valid.groupby(["xbin", "ybin"])
        .agg(goal_rate=("goal", "mean"), n=("goal", "count"))
        .reset_index()
    )

    mat = np.full((_ZONE_Y, _ZONE_X), np.nan)
    for _, row in grid.iterrows():
        if row["n"] >= 5:
            mat[int(row["ybin"]), int(row["xbin"])] = row["goal_rate"]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        mat, origin="lower", aspect="auto", cmap="RdYlGn",
        vmin=0, vmax=0.5,
        extent=[0, _PITCH_X, 0, _PITCH_Y],
    )
    plt.colorbar(im, ax=ax, label="Goal Rate")
    ax.add_patch(plt.Rectangle((0, 0), 105, 68, fill=False, edgecolor="black", lw=2))
    ax.add_patch(plt.Rectangle((83, 13.85), 22, 40.3, fill=False, edgecolor="white", lw=1.5))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Goal Rate Heatmap by Pitch Zone  ({_ZONE_X}×{_ZONE_Y} grid, min 5 shots)")
    plt.tight_layout()
    save_fig("goal_rate_heatmap", "eda")


def xg_calibration_curve(shots: pd.DataFrame, events: pd.DataFrame) -> None:
    """xG vs actual goals with confidence bands."""
    xg_col = "shot_statsbomb_xg"

    if xg_col in shots.columns:
        plot_df = shots[["goal", xg_col]].copy()
    elif not events.empty:
        id_col = next((c for c in ["event_id", "internal_id"] if c in shots.columns and c in events.columns), None)
        if id_col and xg_col in events.columns:
            xg_lookup = events[[id_col, xg_col]].drop_duplicates(id_col)
            plot_df = shots[["goal", id_col]].merge(xg_lookup, on=id_col, how="left")
        else:
            logger.warning("Cannot join xG — skipping calibration curve.")
            return
    else:
        logger.warning("shot_statsbomb_xg unavailable — skipping calibration curve.")
        return

    plot_df[xg_col] = pd.to_numeric(plot_df[xg_col], errors="coerce")
    valid = plot_df.dropna(subset=[xg_col, "goal"])
    if len(valid) < 50:
        return

    valid = valid.copy()
    valid["xg_bin"] = pd.cut(valid[xg_col], bins=20, labels=False)
    calib = valid.groupby("xg_bin").agg(
        mean_xg=(xg_col, "mean"),
        actual_rate=("goal", "mean"),
        n=("goal", "count"),
    ).reset_index().dropna()

    # Wilson confidence interval for each bucket
    z = 1.96
    calib["ci_lo"] = ((calib["actual_rate"] + z**2 / (2 * calib["n"]) -
                        z * np.sqrt(calib["actual_rate"] * (1 - calib["actual_rate"]) / calib["n"] +
                                    z**2 / (4 * calib["n"]**2))) /
                       (1 + z**2 / calib["n"]))
    calib["ci_hi"] = ((calib["actual_rate"] + z**2 / (2 * calib["n"]) +
                        z * np.sqrt(calib["actual_rate"] * (1 - calib["actual_rate"]) / calib["n"] +
                                    z**2 / (4 * calib["n"]**2))) /
                       (1 + z**2 / calib["n"]))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(calib["mean_xg"], calib["actual_rate"], "o-", color="#1f77b4", label="Actual goal rate")
    ax.fill_between(calib["mean_xg"], calib["ci_lo"], calib["ci_hi"], alpha=0.2, color="#1f77b4")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlabel("Mean StatsBomb xG (bin)")
    ax.set_ylabel("Actual Goal Rate")
    ax.set_title("StatsBomb xG Calibration Curve  (95% confidence band)")
    ax.legend()
    max_val = max(calib["mean_xg"].max(), calib["actual_rate"].max()) * 1.1
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    plt.tight_layout()
    save_fig("xg_calibration_curve", "eda")


def goal_rate_by_minute(shots: pd.DataFrame) -> None:
    if "minute" not in shots.columns or "goal" not in shots.columns:
        return

    shots = shots.copy()
    shots["minute_bin"] = (pd.to_numeric(shots["minute"], errors="coerce") // 5 * 5).clip(0, 90)
    rates = shots.groupby("minute_bin")["goal"].agg(rate="mean", n="count").reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(rates["minute_bin"], rates["rate"], width=4, color="#17becf", alpha=0.8)
    ax.set_xlabel("Match Minute (5-min bins)")
    ax.set_ylabel("Goal Rate")
    ax.set_title("Goal Rate by Match Minute")
    for _, row in rates.iterrows():
        ax.text(row["minute_bin"] + 2, row["rate"] + 0.001, str(int(row["n"])), fontsize=6, ha="center")
    plt.tight_layout()
    save_fig("goal_rate_by_minute", "eda")


def goal_rate_by_competition(shots: pd.DataFrame) -> None:
    if "competition_id" not in shots.columns or "goal" not in shots.columns:
        return

    labels = competition_labels()
    rates = (
        shots.groupby("competition_id")["goal"]
        .agg(rate="mean", n="count")
        .reset_index()
        .sort_values("rate")
    )
    rates["label"] = rates["competition_id"].apply(lambda c: labels.get(str(c), str(c)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(rates["label"], rates["rate"], color="#9467bd")
    ax.set_xlabel("Goal Rate")
    ax.set_title("Goal Rate by Competition")
    for _, row in rates.iterrows():
        ax.text(row["rate"] + 0.001, row["label"], f"n={int(row['n']):,}", va="center", fontsize=8)
    plt.tight_layout()
    save_fig("goal_rate_by_competition", "eda")


def shot_360_distributions(shots: pd.DataFrame) -> None:
    if "has_360" not in shots.columns:
        return

    cols_360 = ["keeper_distance_to_shooter", "nearest_defender_distance", "visible_area_size"]
    avail = [c for c in cols_360 if c in shots.columns]
    if not avail:
        return

    shots_360 = shots[shots["has_360"].astype(bool)]
    shots_non = shots[~shots["has_360"].astype(bool)]

    ncols = len(avail)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 4))
    if ncols == 1:
        axes = [axes]

    for ax, col in zip(axes, avail):
        s1 = pd.to_numeric(shots_360[col], errors="coerce").dropna()
        s2 = pd.to_numeric(shots_non[col], errors="coerce").dropna()
        if len(s1) > 0:
            bins = np.linspace(s1.quantile(0.01), s1.quantile(0.99), 40)
            ax.hist(s1, bins=bins, density=True, alpha=0.6, color="#2ca02c", label=f"360 (n={len(s1):,})")
        if len(s2) > 0:
            ax.hist(s2, bins=40, density=True, alpha=0.6, color="#aec7e8", label=f"Non-360 (n={len(s2):,})")
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle("360 vs Non-360 Shot Feature Distributions", fontsize=11)
    plt.tight_layout()
    save_fig("360_vs_non360_distributions", "eda")


def main() -> None:
    logger.info("Loading shots.parquet …")
    shots = load_shots()

    logger.info("Loading events.parquet for xG …")
    try:
        events = load_events()
    except FileNotFoundError:
        logger.warning("events.parquet not found — xG calibration curve skipped.")
        events = pd.DataFrame()

    logger.info("Shot density map …")
    shot_density_map(shots)

    logger.info("Goal rate heatmap …")
    goal_rate_heatmap(shots)

    logger.info("xG calibration curve …")
    xg_calibration_curve(shots, events)

    logger.info("Goal rate by minute …")
    goal_rate_by_minute(shots)

    logger.info("Goal rate by competition …")
    goal_rate_by_competition(shots)

    logger.info("360 vs non-360 shot distributions …")
    shot_360_distributions(shots)

    logger.info("08_eda_shots.py complete.")


if __name__ == "__main__":
    main()
