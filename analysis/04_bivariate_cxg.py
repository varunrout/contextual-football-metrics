"""
analysis/04_bivariate_cxg.py
============================
Part 4a — Bivariate Analysis for CxG (target = goal).

Outputs
-------
reports/figures/bivariate/cxg/goal_rate_{cat_col}.png
reports/figures/bivariate/cxg/boxplots_numeric.png
reports/figures/bivariate/cxg/xg_calibration.png
reports/figures/bivariate/cxg/shot_map.png
reports/figures/bivariate/cxg/goal_by_sequence_type.png
reports/figures/bivariate/cxg/sequence_type_goal_standardized_residuals.png
reports/figures/bivariate/cxg/sequence_type_goal_adjusted_odds_ratios.png
reports/cxg_sequence_type_deep_summary.json
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
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import (  # noqa: E402
    load_events,
    load_shots,
    save_fig,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("04_bivariate_cxg")

_CAT_COLS = [
    "body_part",
    "shot_type",
    "set_piece_type",
    "score_state",
    "period",
    "home_or_away",
    "sequence_type",
    "possession_start_zone",
    "final_pass_zone",
]
_NUM_COLS = [
    "x_location",
    "y_location",
    "distance_to_goal",
    "shot_angle",
    "pass_length",
    "nearest_defender_distance",
    "keeper_distance_to_shooter",
    "keeper_angle_coverage",
    "events_before_action",
    "score_differential",
]
_SEQUENCE_MIN_N = 150
_MAX_PAIRWISE_TYPES = 12
_SEQUENCE_CONTROL_COLS = [
    "distance_to_goal",
    "shot_angle",
    "body_part",
    "set_piece_type",
    "under_pressure",
    "score_differential",
]


def _goal_rate_bar(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns or "goal" not in df.columns:
        logger.debug("Skipping goal_rate bar for %s — column missing.", col)
        return
    rates = (
        df.groupby(col, observed=True)["goal"]
        .agg(goal_rate="mean", n="count")
        .reset_index()
        .sort_values("goal_rate", ascending=False)
    )
    if rates.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(rates) * 0.7), 5))
    bars = ax.bar(
        rates[col].astype(str),
        rates["goal_rate"],
        color="#2ca02c",
        alpha=0.8,
    )
    # Add count labels
    for bar, n in zip(bars, rates["n"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"n={n:,}",
            ha="center",
            fontsize=7,
        )
    ax.set_xlabel(col)
    ax.set_ylabel("Goal Rate")
    ax.set_title(f"Goal Rate by {col}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save_fig(f"goal_rate_{col}", "bivariate/cxg")


def _numeric_boxplots(df: pd.DataFrame) -> None:
    if "goal" not in df.columns:
        return
    avail = [c for c in _NUM_COLS if c in df.columns]
    if not avail:
        return

    ncols = 5
    nrows = -(-len(avail) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]

    for i, col in enumerate(avail):
        ax = axes_flat[i]
        s0 = pd.to_numeric(df.loc[df["goal"] == 0, col], errors="coerce").dropna()
        s1 = pd.to_numeric(df.loc[df["goal"] == 1, col], errors="coerce").dropna()
        ax.boxplot(
            [
                s0.clip(*np.percentile(s0, [1, 99])) if len(s0) else [],
                s1.clip(*np.percentile(s1, [1, 99])) if len(s1) else [],
            ],
            tick_labels=["No Goal", "Goal"],
            patch_artist=True,
            boxprops=dict(facecolor="#aec7e8"),
        )
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(avail), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions: Goal vs No Goal", fontsize=11)
    plt.tight_layout()
    save_fig("boxplots_numeric", "bivariate/cxg")


def _xg_calibration(shots: pd.DataFrame, events: pd.DataFrame) -> None:
    """StatsBomb xG vs actual goal rate in decile calibration buckets."""
    xg_col = "shot_statsbomb_xg"
    event_id_col = "event_id"

    # Look for xg in shots first, then join from events
    if xg_col in shots.columns:
        plot_df = shots[["goal", xg_col]].copy()
        plot_df[xg_col] = pd.to_numeric(plot_df[xg_col], errors="coerce")
    elif (
        event_id_col in shots.columns
        and event_id_col in events.columns
        and xg_col in events.columns
    ):
        xg_lookup = events[[event_id_col, xg_col]].drop_duplicates(event_id_col)
        plot_df = shots[["goal", event_id_col]].merge(xg_lookup, on=event_id_col, how="left")
        plot_df[xg_col] = pd.to_numeric(plot_df[xg_col], errors="coerce")
    else:
        # Try internal_id join
        if "internal_id" in events.columns and xg_col in events.columns:
            xg_lookup = events[["internal_id", xg_col]].drop_duplicates("internal_id")
            merge_col = next((c for c in ["event_id", "internal_id"] if c in shots.columns), None)
            if merge_col:
                plot_df = shots[["goal", merge_col]].merge(
                    xg_lookup, left_on=merge_col, right_on="internal_id", how="left"
                )
                plot_df[xg_col] = pd.to_numeric(plot_df[xg_col], errors="coerce")
            else:
                logger.warning("Cannot join shot_statsbomb_xg — skipping xG calibration.")
                return
        else:
            logger.warning("shot_statsbomb_xg not available — skipping xG calibration plot.")
            return

    valid = plot_df.dropna(subset=[xg_col, "goal"])
    if len(valid) < 50:
        logger.warning("Too few rows with xG for calibration plot.")
        return

    valid = valid.copy()
    valid["xg_decile"] = pd.qcut(valid[xg_col], q=10, labels=False, duplicates="drop")
    calib = (
        valid.groupby("xg_decile")
        .agg(
            mean_xg=(xg_col, "mean"),
            actual_goal_rate=("goal", "mean"),
            n=("goal", "count"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        calib["mean_xg"],
        calib["actual_goal_rate"],
        s=calib["n"] / calib["n"].max() * 200,
        color="#1f77b4",
        zorder=3,
        label="Decile bucket",
    )
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlabel("Mean StatsBomb xG in decile")
    ax.set_ylabel("Actual Goal Rate")
    ax.set_title("StatsBomb xG Calibration (decile buckets)")
    ax.legend()
    ax.set_xlim(0, calib["mean_xg"].max() * 1.1)
    ax.set_ylim(0, calib["actual_goal_rate"].max() * 1.1)
    plt.tight_layout()
    save_fig("xg_calibration", "bivariate/cxg")


def _shot_map(df: pd.DataFrame) -> None:
    if "x_location" not in df.columns or "y_location" not in df.columns:
        return
    if "goal" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    no_goal = df[df["goal"] == 0]
    goal = df[df["goal"] == 1]

    ax.scatter(
        pd.to_numeric(no_goal["x_location"], errors="coerce"),
        pd.to_numeric(no_goal["y_location"], errors="coerce"),
        s=8,
        alpha=0.3,
        color="#aec7e8",
        label="No goal",
    )
    ax.scatter(
        pd.to_numeric(goal["x_location"], errors="coerce"),
        pd.to_numeric(goal["y_location"], errors="coerce"),
        s=20,
        alpha=0.7,
        color="#d62728",
        label="Goal",
    )

    # Draw pitch outline (simplified)
    pitch = plt.Rectangle((0, 0), 105, 68, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(pitch)
    box = plt.Rectangle((83, 13.85), 22, 40.3, fill=False, edgecolor="grey", linewidth=1.5)
    ax.add_patch(box)
    ax.set_xlim(-2, 107)
    ax.set_ylim(-2, 70)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Shot Map (n={len(df):,})")
    ax.legend(markerscale=2)
    ax.set_aspect("equal")
    plt.tight_layout()
    save_fig("shot_map", "bivariate/cxg")


def _holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = float(pvals[idx]) * (m - rank)
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def _sequence_rate_table(shots: pd.DataFrame) -> pd.DataFrame:
    work = shots.copy()
    work["sequence_type"] = work["sequence_type"].astype(str)
    grouped = (
        work.groupby("sequence_type", observed=True)["goal"]
        .agg(n="count", goals="sum", rate="mean")
        .reset_index()
    )
    if grouped.empty:
        return grouped

    ci_low, ci_high = proportion_confint(
        grouped["goals"].to_numpy(),
        grouped["n"].to_numpy(),
        alpha=0.05,
        method="wilson",
    )
    baseline = float(work["goal"].mean())
    grouped["ci_low"] = ci_low
    grouped["ci_high"] = ci_high
    grouped["lift_vs_baseline"] = grouped["rate"] - baseline
    grouped["baseline_rate"] = baseline
    grouped = grouped.sort_values("rate", ascending=False).reset_index(drop=True)
    return grouped


def _plot_sequence_goal_rates(rate_df: pd.DataFrame) -> None:
    if rate_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5, len(rate_df) * 0.4)))
    y = np.arange(len(rate_df))
    ax.errorbar(
        rate_df["rate"],
        y,
        xerr=[rate_df["rate"] - rate_df["ci_low"], rate_df["ci_high"] - rate_df["rate"]],
        fmt="o",
        color="#1f77b4",
        ecolor="#7aa6d1",
        capsize=3,
    )
    ax.axvline(float(rate_df["baseline_rate"].iloc[0]), color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(rate_df["sequence_type"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Goal rate (with 95% Wilson CI)")
    ax.set_title("Goal rate by sequence type")
    for yi, n in zip(y, rate_df["n"], strict=False):
        ax.text(
            1.005 * max(rate_df["ci_high"].max(), 0.01),
            yi,
            f"n={int(n):,}",
            va="center",
            fontsize=8,
        )
    plt.tight_layout()
    save_fig("goal_by_sequence_type", "bivariate/cxg")


def _sequence_contingency_tests(shots: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    tab = pd.crosstab(shots["sequence_type"].astype(str), shots["goal"].astype(int))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {}, tab, pd.DataFrame()

    chi2, pval, dof, expected = sps.chi2_contingency(tab)
    n = tab.values.sum()
    r, c = tab.shape
    cramers_v = float(np.sqrt(max(chi2, 0) / (n * max(min(r - 1, c - 1), 1))))
    residuals = (tab.values - expected) / np.sqrt(np.maximum(expected, 1e-9))
    residuals_df = pd.DataFrame(residuals, index=tab.index, columns=tab.columns)
    stats_summary = {
        "chi2": float(chi2),
        "p_value": float(pval),
        "dof": int(dof),
        "cramers_v": cramers_v,
    }
    return stats_summary, tab, residuals_df


def _plot_sequence_residual_heatmap(residuals_df: pd.DataFrame) -> None:
    if residuals_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, len(residuals_df) * 0.35)))
    sns.heatmap(
        residuals_df,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Standardized residual"},
        ax=ax,
    )
    ax.set_title("Sequence type x goal standardized residuals")
    ax.set_xlabel("goal")
    ax.set_ylabel("sequence_type")
    plt.tight_layout()
    save_fig("sequence_type_goal_standardized_residuals", "bivariate/cxg")


def _pairwise_rate_tests(rate_df: pd.DataFrame) -> list[dict]:
    candidates = rate_df[rate_df["n"] >= _SEQUENCE_MIN_N].copy()
    if len(candidates) < 2:
        return []
    candidates = candidates.head(_MAX_PAIRWISE_TYPES)

    pairs: list[dict] = []
    pvals: list[float] = []
    idx = candidates.index.tolist()
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a = candidates.loc[idx[i]]
            b = candidates.loc[idx[j]]
            stat, p = proportions_ztest(
                count=[int(a["goals"]), int(b["goals"])],
                nobs=[int(a["n"]), int(b["n"])],
            )
            pairs.append(
                {
                    "a": str(a["sequence_type"]),
                    "b": str(b["sequence_type"]),
                    "rate_a": float(a["rate"]),
                    "rate_b": float(b["rate"]),
                    "rate_diff": float(a["rate"] - b["rate"]),
                    "z_stat": float(stat),
                    "p_value": float(p),
                }
            )
            pvals.append(float(p))

    adj = _holm_adjust(pvals)
    for rec, p_adj in zip(pairs, adj, strict=False):
        rec["p_value_holm"] = float(p_adj)
    pairs.sort(key=lambda r: r["p_value_holm"])
    return pairs


def _adjusted_logit_sequence_effects(shots: pd.DataFrame) -> list[dict]:
    model_df = shots.copy()
    model_df["sequence_type"] = model_df["sequence_type"].astype(str)

    seq_counts = model_df["sequence_type"].value_counts()
    keep_seq = seq_counts[seq_counts >= _SEQUENCE_MIN_N].index
    model_df = model_df[model_df["sequence_type"].isin(keep_seq)].copy()
    if model_df.empty or model_df["sequence_type"].nunique() < 2:
        return []

    control_terms: list[str] = []
    for col in _SEQUENCE_CONTROL_COLS:
        if col not in model_df.columns:
            continue
        nunique = model_df[col].nunique(dropna=True)
        if nunique < 2:
            continue
        if pd.api.types.is_numeric_dtype(model_df[col]):
            control_terms.append(col)
        else:
            control_terms.append(f"C({col})")

    fit_df = model_df.dropna(subset=["goal", "sequence_type"]).copy()
    if fit_df.empty:
        return []

    model = None
    used_terms: list[str] = []
    for k in range(len(control_terms), -1, -1):
        terms = ["C(sequence_type)", *control_terms[:k]]
        formula = "goal ~ " + " + ".join(terms)
        try:
            model = smf.logit(formula=formula, data=fit_df).fit(disp=False, maxiter=200)
            used_terms = terms
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("CxG adjusted logit failed with %d controls: %s", k, exc)
            continue

    if model is None:
        logger.warning("CxG adjusted logit unavailable after fallback attempts.")
        return []

    effects: list[dict] = []
    for name, coef in model.params.items():
        if not name.startswith("C(sequence_type)"):
            continue
        se = float(model.bse[name])
        lo = coef - 1.96 * se
        hi = coef + 1.96 * se
        level = name.split("T.")[-1].rstrip("]") if "T." in name else name
        effects.append(
            {
                "term": name,
                "sequence_type": level,
                "odds_ratio": float(np.exp(coef)),
                "or_ci_low": float(np.exp(lo)),
                "or_ci_high": float(np.exp(hi)),
                "p_value": float(model.pvalues[name]),
                "model_terms": used_terms,
            }
        )

    effects.sort(key=lambda r: r["odds_ratio"], reverse=True)
    return effects


def _plot_adjusted_or(effects: list[dict]) -> None:
    if not effects:
        return
    df = pd.DataFrame(effects)
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    y = np.arange(len(df))
    ax.errorbar(
        df["odds_ratio"],
        y,
        xerr=[df["odds_ratio"] - df["or_ci_low"], df["or_ci_high"] - df["odds_ratio"]],
        fmt="o",
        color="#2ca02c",
        ecolor="#98d398",
        capsize=3,
    )
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["sequence_type"].astype(str), fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Adjusted odds ratio (log scale)")
    ax.set_title("Adjusted sequence-type effects on goal probability")
    plt.tight_layout()
    save_fig("sequence_type_goal_adjusted_odds_ratios", "bivariate/cxg")


def _sequence_type_deep_analysis(shots: pd.DataFrame) -> None:
    if "sequence_type" not in shots.columns or "goal" not in shots.columns:
        logger.warning("CxG deep sequence analysis skipped: required columns missing.")
        return

    work = shots.dropna(subset=["sequence_type", "goal"]).copy()
    if work.empty:
        logger.warning("CxG deep sequence analysis skipped: no valid rows.")
        return

    rate_df = _sequence_rate_table(work)
    _plot_sequence_goal_rates(rate_df)

    assoc_stats, tab, residuals_df = _sequence_contingency_tests(work)
    _plot_sequence_residual_heatmap(residuals_df)

    pairwise = _pairwise_rate_tests(rate_df)
    effects = _adjusted_logit_sequence_effects(work)
    _plot_adjusted_or(effects)

    summary = {
        "n_shots": int(len(work)),
        "overall_goal_rate": float(work["goal"].mean()),
        "sequence_rate_table": rate_df.to_dict("records"),
        "association_test": assoc_stats,
        "pairwise_rate_tests_holm": pairwise,
        "adjusted_logit_sequence_effects": effects,
        "contingency_table": tab.reset_index().to_dict("records") if not tab.empty else [],
    }
    save_json(summary, "cxg_sequence_type_deep_summary")


def main() -> None:
    logger.info("Loading shots.parquet …")
    shots = load_shots()

    logger.info("Loading events.parquet for shot_statsbomb_xg …")
    try:
        events = load_events()
    except FileNotFoundError:
        logger.warning("events.parquet not found — xG calibration plot skipped.")
        events = pd.DataFrame()

    logger.info("Goal rate by categorical features …")
    for col in _CAT_COLS:
        _goal_rate_bar(shots, col)

    logger.info("Numeric boxplots by goal outcome …")
    _numeric_boxplots(shots)

    logger.info("StatsBomb xG calibration …")
    _xg_calibration(shots, events)

    logger.info("Shot map …")
    _shot_map(shots)

    logger.info("Deep CxG sequence-type analysis …")
    _sequence_type_deep_analysis(shots)

    logger.info("04_bivariate_cxg.py complete.")


if __name__ == "__main__":
    main()
