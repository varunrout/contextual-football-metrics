"""
analysis/03_univariate.py
=========================
Part 3 — Univariate Analysis.

Outputs
-------
reports/figures/univariate/{group}_distributions.png    (per feature group)
reports/figures/univariate/{group}_categoricals.png     (per group with cat features)
reports/figures/univariate/outlier_summary.png
reports/figures/univariate/skewness_ranking.png
reports/univariate_stats.json
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import (
    feature_groups,
    load_features,
    save_fig,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("03_univariate")

_ID_COLS = {"player_id", "team_id", "opponent_id", "competition_id",
            "match_id", "possession_id", "event_id"}
_SKEW_FLAG = 2.0
_ZSCORE_THRESHOLD = 3.0
_MAX_COLS_PER_GRID = 6


def _cap_sample(s: pd.Series, max_n: int, seed: int = 42) -> pd.Series:
    """No-op sampler: keep full data for current analysis subset."""
    return s


def _numeric_stats(series: pd.Series) -> dict:
    s_raw = series.dropna()
    if len(s_raw) == 0:
        return {}

    n_full = int(len(s_raw))
    s = _cap_sample(s_raw, len(s_raw))
    s = pd.to_numeric(s, errors="coerce").astype("float64", copy=False).dropna()
    if len(s) == 0:
        return {}
    mean = float(s.mean())
    std = float(s.std())
    z = (s - mean) / (std + 1e-9)
    outlier_fraction = float((np.abs(z) > _ZSCORE_THRESHOLD).mean())

    return {
        "count": n_full,
        "sample_size": int(len(s)),
        "mean": round(mean, 4),
        "median": round(float(s.median()), 4),
        "std": round(std, 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
        "skew": round(float(stats.skew(s)), 4),
        "kurtosis": round(float(stats.kurtosis(s)), 4),
        "outlier_count": int(round(outlier_fraction * n_full)),
        "outlier_fraction": round(outlier_fraction, 4),
    }


def _categorical_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {}
    n_full = int(len(s))
    s = _cap_sample(s, len(s))
    s = s.astype(str)
    vc = s.value_counts()
    probs = vc / vc.sum()
    entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
    return {
        "count": n_full,
        "sample_size": int(len(s)),
        "n_unique": int(s.nunique()),
        "entropy_bits": round(entropy, 4),
        "top_5": {k: int(v) for k, v in vc.head(5).items()},
    }


def _distribution_grid(group_name: str, cols: list[str], df: pd.DataFrame) -> None:
    num_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return

    ncols = min(_MAX_COLS_PER_GRID, len(num_cols))
    nrows = math.ceil(len(num_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    axes_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]

    for i, col in enumerate(num_cols):
        ax = axes_flat[i]
        s = _cap_sample(df[col].dropna(), len(df[col]))
        s = pd.to_numeric(s, errors="coerce").astype("float64", copy=False).dropna()
        if len(s) == 0:
            ax.set_visible(False)
            continue

        # Binary indicators are better shown as 0/1 class proportions than histograms.
        uniq = sorted(s.dropna().unique().tolist())
        is_binary = len(uniq) <= 2 and set(uniq).issubset({0.0, 1.0})
        if is_binary:
            zero_rate = float((s == 0).mean())
            one_rate = float((s == 1).mean())
            ax.bar([0, 1], [zero_rate, one_rate], color=["#7f7f7f", "#2ca02c"], alpha=0.85)
            ax.set_xticks([0, 1])
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Proportion", fontsize=6)
        else:
            ax.hist(s, bins=40, color="#4878cf", alpha=0.7, density=True, edgecolor="none")
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=6)

    for j in range(len(num_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Feature Distributions — {group_name}", fontsize=10)
    plt.tight_layout()
    save_fig(f"{group_name}_distributions", "univariate")


def _categorical_grid(group_name: str, cols: list[str], df: pd.DataFrame) -> None:
    cat_cols = [
        c for c in cols
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
        and c not in _ID_COLS
    ]
    if not cat_cols:
        return

    ncols = min(3, len(cat_cols))
    nrows = math.ceil(len(cat_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]

    for i, col in enumerate(cat_cols):
        ax = axes_flat[i]
        s = _cap_sample(df[col].dropna(), len(df[col]))
        vc = s.astype(str).value_counts().head(10)
        if vc.empty:
            ax.set_visible(False)
            continue
        ax.bar(range(len(vc)), vc.values, color="#9467bd")
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=45, ha="right", fontsize=7)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(cat_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Categorical Features — {group_name}", fontsize=10)
    plt.tight_layout()
    save_fig(f"{group_name}_categoricals", "univariate")


def outlier_summary(df: pd.DataFrame, stat_results: dict) -> None:
    """Bar chart of top-20 features by outlier fraction."""
    items = [
        (col, v["outlier_fraction"])
        for col, v in stat_results.items()
        if "outlier_fraction" in v
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    top20 = items[:20]
    if not top20:
        return
    feats, fracs = zip(*top20)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(feats)[::-1], list(fracs)[::-1], color="#d62728")
    ax.set_xlabel(f"Outlier Fraction (|z| > {_ZSCORE_THRESHOLD})")
    ax.set_title("Top 20 Features by Outlier Density")
    plt.tight_layout()
    save_fig("outlier_summary", "univariate")


def skewness_ranking(stat_results: dict) -> list[str]:
    """Bar chart of |skewness| per numeric feature; flag > 2 as log-transform candidates."""
    items = [
        (col, abs(v["skew"]))
        for col, v in stat_results.items()
        if "skew" in v
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    top40 = items[:40]
    if not top40:
        return []

    feats, skews = zip(*top40)
    colors = ["#d62728" if s > _SKEW_FLAG else "#1f77b4" for s in skews]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(feats)), skews, color=colors)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.axhline(_SKEW_FLAG, color="red", linestyle="--", label=f"|skew| = {_SKEW_FLAG} (log-transform)")
    ax.set_ylabel("|Skewness|")
    ax.set_title("Feature Skewness Ranking  (red = log-transform candidate)")
    ax.legend()
    plt.tight_layout()
    save_fig("skewness_ranking", "univariate")

    return [col for col, s in items if s > _SKEW_FLAG]


def main() -> None:
    logger.info("Loading features.parquet …")
    df = load_features()
    groups = feature_groups()

    stat_results: dict = {}
    cat_results: dict = {}

    for group_name, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            logger.info("  Group '%s': no columns present — skipping.", group_name)
            continue

        logger.info("  Group '%s': %d features …", group_name, len(present))

        # Numeric stats
        for col in present:
            if pd.api.types.is_numeric_dtype(df[col]):
                stat_results[col] = _numeric_stats(df[col])
            else:
                cat_results[col] = _categorical_stats(df[col])

        # Distribution grid
        _distribution_grid(group_name, present, df)
        _categorical_grid(group_name, present, df)

    logger.info("Plotting outlier summary …")
    outlier_summary(df, stat_results)

    logger.info("Plotting skewness ranking …")
    log_candidates = skewness_ranking(stat_results)
    logger.info("Log-transform candidates (|skew| > %.1f): %s", _SKEW_FLAG, log_candidates[:10])

    summary = {
        "numeric_stats": stat_results,
        "categorical_stats": cat_results,
        "log_transform_candidates": log_candidates,
        "n_numeric_features": len(stat_results),
        "n_categorical_features": len(cat_results),
    }
    save_json(summary, "univariate_stats")
    logger.info("03_univariate.py complete.")


if __name__ == "__main__":
    main()
