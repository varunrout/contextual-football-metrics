"""
analysis/01_data_quality.py
===========================
Part 1 — Data Quality & Coverage.

Outputs
-------
reports/figures/data_quality/missing_rates.png
reports/figures/data_quality/missing_by_group.png
reports/figures/data_quality/coverage_360.png
reports/figures/data_quality/event_type_dist.png
reports/figures/data_quality/row_completeness.png
reports/data_quality_summary.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import (
    competition_labels,
    feature_groups,
    load_features,
    load_matches,
    save_fig,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("01_data_quality")

_ID_COLS = {"player_id", "team_id", "opponent_id", "competition_id",
            "match_id", "possession_id", "event_id"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _ID_COLS]


def missing_rates(df: pd.DataFrame) -> None:
    feat_cols = _feature_cols(df)
    missing = df[feat_cols].isnull().mean().sort_values(ascending=False)
    non_zero = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(12, max(4, len(non_zero) * 0.25)))
    colors = ["#d62728" if v > 0.5 else "#ff7f0e" if v > 0.2 else "#1f77b4"
              for v in non_zero.values]
    ax.barh(non_zero.index[::-1], non_zero.values[::-1], color=colors[::-1])
    ax.set_xlabel("Missing Rate")
    ax.set_title(f"Missing Rates per Feature ({len(non_zero)} columns with any missing)")
    ax.axvline(0.2, color="orange", linestyle="--", linewidth=1, label=">20%")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label=">50%")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig("missing_rates", "data_quality")

    return missing


def missing_by_group(df: pd.DataFrame, groups: dict[str, list[str]]) -> dict:
    group_stats: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    names, rates = [], []
    for group, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        rate = float(df[present].isnull().mean().mean())
        group_stats[group] = {
            "n_features": len(present),
            "mean_missing_rate": round(rate, 4),
        }
        names.append(group)
        rates.append(rate)
    ax.bar(names, rates, color="#4878cf")
    ax.set_ylabel("Mean Missing Rate")
    ax.set_title("Mean Missing Rate by Feature Group")
    ax.set_ylim(0, max(rates + [0.05]) * 1.2)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.005, f"{r:.1%}", ha="center", fontsize=9)
    plt.tight_layout()
    save_fig("missing_by_group", "data_quality")
    return group_stats


def coverage_360(df: pd.DataFrame) -> dict:
    if "has_360" not in df.columns:
        logger.warning("has_360 column not found — skipping 360 coverage plot.")
        return {}

    total = len(df)
    overall_rate = float(df["has_360"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By competition
    comp_labels = competition_labels()
    if "competition_id" in df.columns:
        comp_rates = (
            df.groupby("competition_id")["has_360"]
            .mean()
            .sort_values()
        )
        labels = [comp_labels.get(str(c), str(c)) for c in comp_rates.index]
        axes[0].barh(labels, comp_rates.values, color="#2ca02c")
        axes[0].set_xlabel("360 Coverage Rate")
        axes[0].set_title("360 Coverage by Competition")
        axes[0].set_xlim(0, 1.05)
    else:
        axes[0].text(0.5, 0.5, "No competition_id", ha="center")

    # By event type
    if "event_type" in df.columns:
        type_rates = (
            df.groupby("event_type")["has_360"]
            .mean()
            .sort_values(ascending=False)
        )
        axes[1].bar(type_rates.index, type_rates.values, color="#9467bd")
        axes[1].set_ylabel("360 Coverage Rate")
        axes[1].set_title("360 Coverage by Event Type")
        axes[1].set_ylim(0, 1.05)
        axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(f"360 Data Coverage  (overall: {overall_rate:.1%})", fontsize=12)
    plt.tight_layout()
    save_fig("coverage_360", "data_quality")

    return {
        "total_rows": total,
        "overall_360_rate": round(overall_rate, 4),
    }


def event_type_distribution(df: pd.DataFrame) -> dict:
    if "event_type" not in df.columns:
        logger.warning("event_type column not found — skipping event type distribution.")
        return {}

    counts = df["event_type"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index, counts.values, color="#17becf")
    ax.set_ylabel("Count")
    ax.set_title("Event Type Distribution")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.max() * 0.01, str(v), ha="center", fontsize=7)
    plt.tight_layout()
    save_fig("event_type_dist", "data_quality")

    return {c: int(v) for c, v in counts.items()}


def dtype_audit(df: pd.DataFrame, groups: dict[str, list[str]]) -> list[dict]:
    """Compare actual dtypes vs features.yaml spec."""
    dtype_map = {
        "float32": ["float32", "float64"],
        "int8": ["int8", "int16", "int32", "int64"],
        "int16": ["int8", "int16", "int32", "int64"],
        "int32": ["int16", "int32", "int64"],
        "bool": ["bool"],
        "category": ["category"],
        "str": ["object"],
    }

    registry_path = _ROOT / "configs" / "features.yaml"
    with open(registry_path, encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    mismatches = []
    for group_name, entries in registry.items():
        if group_name == "identifiers" or not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            col = entry.get("name")
            expected_dtype = entry.get("dtype")
            if col not in df.columns or not expected_dtype:
                continue
            actual = str(df[col].dtype)
            allowed = dtype_map.get(expected_dtype, [expected_dtype])
            if actual not in allowed:
                mismatches.append({
                    "column": col,
                    "group": group_name,
                    "expected_dtype": expected_dtype,
                    "actual_dtype": actual,
                })

    if mismatches:
        logger.warning("Dtype mismatches (%d):", len(mismatches))
        for m in mismatches[:20]:
            logger.warning("  %s: expected %s, got %s", m["column"], m["expected_dtype"], m["actual_dtype"])
    else:
        logger.info("No dtype mismatches found.")

    return mismatches


def row_completeness(df: pd.DataFrame) -> None:
    feat_cols = _feature_cols(df)
    n_features = len(feat_cols)
    completeness = df[feat_cols].notnull().sum(axis=1)
    completeness_frac = completeness / n_features

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(completeness_frac, bins=50, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Fraction of features non-null")
    ax.set_ylabel("Number of rows")
    ax.set_title(f"Row-Level Feature Completeness  (n_features={n_features})")
    ax.axvline(completeness_frac.mean(), color="red", linestyle="--",
               label=f"Mean: {completeness_frac.mean():.2f}")
    ax.legend()
    plt.tight_layout()
    save_fig("row_completeness", "data_quality")


def main() -> None:
    logger.info("Loading features.parquet …")
    df = load_features()
    groups = feature_groups()

    logger.info("Step 1: Missing rates …")
    missing = missing_rates(df)

    logger.info("Step 2: Missing by group …")
    group_stats = missing_by_group(df, groups)

    logger.info("Step 3: 360 coverage …")
    coverage = coverage_360(df)

    logger.info("Step 4: Event type distribution …")
    event_counts = event_type_distribution(df)

    logger.info("Step 5: Dtype audit …")
    mismatches = dtype_audit(df, groups)

    logger.info("Step 6: Row completeness …")
    row_completeness(df)

    # ── Summarise ──────────────────────────────────────────────────────────────
    top_missing = missing[missing > 0].head(20).to_dict()
    summary = {
        "n_rows": len(df),
        "n_feature_cols": len(_feature_cols(df)),
        "total_missing_cells": int(df[_feature_cols(df)].isnull().sum().sum()),
        "overall_missing_rate": round(float(df[_feature_cols(df)].isnull().mean().mean()), 4),
        "coverage_360": coverage,
        "event_type_counts": event_counts,
        "feature_groups": group_stats,
        "top_20_missing_features": {k: round(v, 4) for k, v in top_missing.items()},
        "dtype_mismatches": mismatches,
        "n_dtype_mismatches": len(mismatches),
    }
    save_json(summary, "data_quality_summary")
    logger.info("01_data_quality.py complete.")


if __name__ == "__main__":
    main()
