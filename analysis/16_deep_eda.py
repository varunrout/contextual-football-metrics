"""
analysis/16_deep_eda.py
=======================
Part 10 — Deeper EDA: subgroup calibration + interaction effects.

Outputs
-------
reports/figures/eda/deep/cxg_calibration_by_sequence_type.png
reports/figures/eda/deep/cxg_calibration_by_competition.png
reports/figures/eda/deep/cxg_interaction_sequence_vs_start_zone.png
reports/figures/eda/deep/cxg_interaction_sequence_vs_body_part.png
reports/figures/eda/deep/cxa_interaction_sequence_vs_event_type.png
reports/figures/eda/deep/cxa_interaction_sequence_vs_start_zone.png
reports/deep_eda_summary.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import (  # noqa: E402
    competition_labels,
    derive_shot_created,
    load_actions,
    load_events,
    load_shots,
    save_fig,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("16_deep_eda")

_MIN_GROUP_N = 40
_MIN_CELL_N = 25
_MAX_PLOT_GROUPS = 12


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not np.any(mask):
            continue
        frac = mask.sum() / n
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += frac * abs(acc - conf)
    return float(ece)


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_true - y_prob) ** 2))


def _attach_xg(shots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    xg_col = "shot_statsbomb_xg"
    if xg_col in shots.columns:
        out = shots.copy()
    else:
        id_col = next(
            (c for c in ["event_id", "internal_id"] if c in shots.columns and c in events.columns),
            None,
        )
        if id_col is None or xg_col not in events.columns:
            return pd.DataFrame()
        lookup = events[[id_col, xg_col]].drop_duplicates(id_col)
        out = shots.merge(lookup, on=id_col, how="left")

    out["goal"] = pd.to_numeric(out.get("goal", np.nan), errors="coerce")
    out[xg_col] = pd.to_numeric(out.get(xg_col, np.nan), errors="coerce")
    out = out.dropna(subset=["goal", xg_col]).copy()
    out[xg_col] = out[xg_col].clip(1e-7, 1 - 1e-7)
    out["goal"] = out["goal"].astype(int)
    return out


def _group_calibration(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    xg_col = "shot_statsbomb_xg"
    if group_col not in df.columns:
        return pd.DataFrame()

    work = df[[group_col, "goal", xg_col]].dropna().copy()
    if work.empty:
        return work

    recs: list[dict] = []
    for key, grp in work.groupby(group_col, observed=True):
        n = len(grp)
        if n < _MIN_GROUP_N:
            continue
        y = grp["goal"].to_numpy(dtype=float)
        p = grp[xg_col].to_numpy(dtype=float)
        recs.append(
            {
                "group": str(key),
                "n": int(n),
                "actual_rate": float(y.mean()),
                "mean_xg": float(p.mean()),
                "calibration_gap": float(y.mean() - p.mean()),
                "ece": _ece(y, p),
                "brier": _brier(y, p),
            }
        )

    out = pd.DataFrame(recs)
    if out.empty:
        return out
    out = out.sort_values(["n", "group"], ascending=[False, True]).reset_index(drop=True)
    return out


def _plot_group_calibration(cal_df: pd.DataFrame, fig_name: str, title: str) -> None:
    if cal_df.empty:
        return

    plot_df = cal_df.sort_values("n", ascending=False).head(_MAX_PLOT_GROUPS).copy()
    plot_df = plot_df.sort_values("actual_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.45)))
    y = np.arange(len(plot_df))
    ax.scatter(plot_df["mean_xg"], y, s=50, color="#1f77b4", label="Mean xG")
    ax.scatter(plot_df["actual_rate"], y, s=50, color="#d62728", label="Actual goal rate")

    for yi, row in enumerate(plot_df.itertuples(index=False)):
        ax.plot([row.mean_xg, row.actual_rate], [yi, yi], color="#999999", alpha=0.6)
        ax.text(
            max(row.mean_xg, row.actual_rate) + 0.005,
            yi,
            f"n={int(row.n)}",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["group"].astype(str))
    ax.set_xlabel("Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig(fig_name, "eda/deep")


def _interaction_stats(df: pd.DataFrame, row_col: str, col_col: str, target_col: str) -> dict:
    tab = pd.crosstab(
        df[row_col].astype(str) + " | " + df[col_col].astype(str),
        df[target_col].astype(int),
    )
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {}

    chi2, pval, dof, _ = sps.chi2_contingency(tab)
    n = int(tab.to_numpy().sum())
    k = max(min(tab.shape) - 1, 1)
    cramers_v = float(np.sqrt(max(chi2, 0.0) / (n * k)))
    return {
        "chi2": float(chi2),
        "p_value": float(pval),
        "dof": int(dof),
        "cramers_v": cramers_v,
        "n_cells": int(tab.shape[0]),
    }


def _interaction_cell_table(
    df: pd.DataFrame, row_col: str, col_col: str, target_col: str
) -> pd.DataFrame:
    grouped = (
        df.groupby([row_col, col_col], observed=True)[target_col]
        .agg(n="count", positives="sum", rate="mean")
        .reset_index()
    )
    if grouped.empty:
        return grouped

    baseline = float(df[target_col].mean())
    grouped["lift_vs_baseline"] = grouped["rate"] - baseline
    grouped["baseline_rate"] = baseline
    grouped = grouped[grouped["n"] >= _MIN_CELL_N].copy()
    grouped = grouped.sort_values("lift_vs_baseline", ascending=False).reset_index(drop=True)
    return grouped


def _plot_interaction_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    target_col: str,
    fig_name: str,
    title: str,
) -> pd.DataFrame:
    if row_col not in df.columns or col_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    work = df[[row_col, col_col, target_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame()

    top_rows = work[row_col].astype(str).value_counts().head(10).index
    top_cols = work[col_col].astype(str).value_counts().head(8).index
    work = work[
        work[row_col].astype(str).isin(top_rows) & work[col_col].astype(str).isin(top_cols)
    ].copy()

    rates = work.groupby([row_col, col_col], observed=True)[target_col].mean().unstack(col_col)
    counts = (
        work.groupby([row_col, col_col], observed=True)[target_col]
        .count()
        .unstack(col_col)
        .fillna(0)
    )

    if rates.empty:
        return pd.DataFrame()

    mask_low = counts < _MIN_CELL_N
    rates_masked = rates.mask(mask_low)

    fig, ax = plt.subplots(figsize=(11, max(5, len(rates_masked.index) * 0.5)))
    sns.heatmap(
        rates_masked,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": f"{target_col} rate"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    plt.tight_layout()
    save_fig(fig_name, "eda/deep")

    return _interaction_cell_table(work, row_col, col_col, target_col)


def _run_cxg_deep(shots: pd.DataFrame, events: pd.DataFrame) -> dict:
    out: dict = {}
    xg_df = _attach_xg(shots, events)
    if xg_df.empty:
        logger.warning("CxG deep EDA skipped: xG not available.")
        return out

    labels = competition_labels()
    if "competition_id" in xg_df.columns:
        xg_df = xg_df.copy()
        xg_df["competition_label"] = (
            xg_df["competition_id"]
            .astype(str)
            .map(labels)
            .fillna(xg_df["competition_id"].astype(str))
        )

    cal_seq = (
        _group_calibration(xg_df, "sequence_type")
        if "sequence_type" in xg_df.columns
        else pd.DataFrame()
    )
    cal_comp = (
        _group_calibration(xg_df, "competition_label")
        if "competition_label" in xg_df.columns
        else pd.DataFrame()
    )

    _plot_group_calibration(
        cal_seq, "cxg_calibration_by_sequence_type", "CxG Calibration by Sequence Type"
    )
    _plot_group_calibration(
        cal_comp, "cxg_calibration_by_competition", "CxG Calibration by Competition"
    )

    cxg_inter_1 = (
        _plot_interaction_heatmap(
            xg_df,
            row_col="sequence_type",
            col_col="possession_start_zone",
            target_col="goal",
            fig_name="cxg_interaction_sequence_vs_start_zone",
            title="CxG: Goal Rate by Sequence Type x Start Zone",
        )
        if "sequence_type" in xg_df.columns and "possession_start_zone" in xg_df.columns
        else pd.DataFrame()
    )

    cxg_inter_2 = (
        _plot_interaction_heatmap(
            xg_df,
            row_col="sequence_type",
            col_col="body_part",
            target_col="goal",
            fig_name="cxg_interaction_sequence_vs_body_part",
            title="CxG: Goal Rate by Sequence Type x Body Part",
        )
        if "sequence_type" in xg_df.columns and "body_part" in xg_df.columns
        else pd.DataFrame()
    )

    out["cxg"] = {
        "n_shots": int(len(xg_df)),
        "overall_goal_rate": float(xg_df["goal"].mean()),
        "overall_mean_xg": float(xg_df["shot_statsbomb_xg"].mean()),
        "calibration_by_sequence_type": cal_seq.to_dict("records"),
        "calibration_by_competition": cal_comp.to_dict("records"),
        "interaction_sequence_x_start_zone": {
            "stats": _interaction_stats(
                xg_df.dropna(subset=["sequence_type", "possession_start_zone", "goal"]),
                "sequence_type",
                "possession_start_zone",
                "goal",
            )
            if "sequence_type" in xg_df.columns and "possession_start_zone" in xg_df.columns
            else {},
            "top_positive_lift_cells": cxg_inter_1.head(10).to_dict("records")
            if not cxg_inter_1.empty
            else [],
            "top_negative_lift_cells": cxg_inter_1.tail(10)
            .sort_values("lift_vs_baseline", ascending=True)
            .to_dict("records")
            if not cxg_inter_1.empty
            else [],
        },
        "interaction_sequence_x_body_part": {
            "stats": _interaction_stats(
                xg_df.dropna(subset=["sequence_type", "body_part", "goal"]),
                "sequence_type",
                "body_part",
                "goal",
            )
            if "sequence_type" in xg_df.columns and "body_part" in xg_df.columns
            else {},
            "top_positive_lift_cells": cxg_inter_2.head(10).to_dict("records")
            if not cxg_inter_2.empty
            else [],
            "top_negative_lift_cells": cxg_inter_2.tail(10)
            .sort_values("lift_vs_baseline", ascending=True)
            .to_dict("records")
            if not cxg_inter_2.empty
            else [],
        },
    }
    return out


def _run_cxa_deep(actions: pd.DataFrame) -> dict:
    out: dict = {}
    act = derive_shot_created(actions)
    if "shot_created" not in act.columns:
        logger.warning("CxA deep EDA skipped: shot_created unavailable.")
        return out

    cxa_inter_1 = (
        _plot_interaction_heatmap(
            act,
            row_col="sequence_type",
            col_col="event_type",
            target_col="shot_created",
            fig_name="cxa_interaction_sequence_vs_event_type",
            title="CxA: Shot-Created Rate by Sequence Type x Event Type",
        )
        if "sequence_type" in act.columns and "event_type" in act.columns
        else pd.DataFrame()
    )

    cxa_inter_2 = (
        _plot_interaction_heatmap(
            act,
            row_col="sequence_type",
            col_col="possession_start_zone",
            target_col="shot_created",
            fig_name="cxa_interaction_sequence_vs_start_zone",
            title="CxA: Shot-Created Rate by Sequence Type x Start Zone",
        )
        if "sequence_type" in act.columns and "possession_start_zone" in act.columns
        else pd.DataFrame()
    )

    out["cxa"] = {
        "n_actions": int(len(act)),
        "overall_shot_created_rate": float(act["shot_created"].mean()),
        "interaction_sequence_x_event_type": {
            "stats": _interaction_stats(
                act.dropna(subset=["sequence_type", "event_type", "shot_created"]),
                "sequence_type",
                "event_type",
                "shot_created",
            )
            if "sequence_type" in act.columns and "event_type" in act.columns
            else {},
            "top_positive_lift_cells": cxa_inter_1.head(10).to_dict("records")
            if not cxa_inter_1.empty
            else [],
            "top_negative_lift_cells": cxa_inter_1.tail(10)
            .sort_values("lift_vs_baseline", ascending=True)
            .to_dict("records")
            if not cxa_inter_1.empty
            else [],
        },
        "interaction_sequence_x_start_zone": {
            "stats": _interaction_stats(
                act.dropna(subset=["sequence_type", "possession_start_zone", "shot_created"]),
                "sequence_type",
                "possession_start_zone",
                "shot_created",
            )
            if "sequence_type" in act.columns and "possession_start_zone" in act.columns
            else {},
            "top_positive_lift_cells": cxa_inter_2.head(10).to_dict("records")
            if not cxa_inter_2.empty
            else [],
            "top_negative_lift_cells": cxa_inter_2.tail(10)
            .sort_values("lift_vs_baseline", ascending=True)
            .to_dict("records")
            if not cxa_inter_2.empty
            else [],
        },
    }
    return out


def main() -> None:
    logger.info("Loading datasets for deep EDA ...")
    shots = load_shots()
    actions = load_actions()
    try:
        events = load_events()
    except FileNotFoundError:
        events = pd.DataFrame()

    logger.info("Running deep CxG analysis ...")
    summary = _run_cxg_deep(shots, events)

    logger.info("Running deep CxA analysis ...")
    summary.update(_run_cxa_deep(actions))

    save_json(summary, "deep_eda_summary")
    logger.info("16_deep_eda.py complete.")


if __name__ == "__main__":
    main()
