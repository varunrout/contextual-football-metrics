"""
analysis/02_feature_stability.py
=================================
Part 2 — Feature Stability & Validity.

Outputs
-------
reports/figures/stability/ks_heatmap.png
reports/figures/stability/temporal_stability.png
reports/figures/stability/cohort_360_comparison.png
reports/validity_violations.json
"""

from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind
from typing import cast

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import (
    competition_labels,
    load_features,
    numeric_feature_cols,
    save_fig,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("02_feature_stability")

_KS_FLAG_THRESHOLD = 0.2
_TTEST_PVAL_THRESHOLD = 0.01
_MWU_PVAL_THRESHOLD = 0.05
_MAX_FEATURES_PLOT = 40   # cap for readability in heatmaps
_MAX_PAIRS_PLOT = 10            # competition pairs shown in heatmap


def ks_competition_heatmap(df: pd.DataFrame) -> dict:
    """KS-test for each numeric feature between every pair of competitions.

    Single-pass: KS stats are computed once, stored in a dict, then reused
    for both the flagging step and the heatmap matrix.
    """
    if "competition_id" not in df.columns:
        logger.warning("competition_id not in features — skipping KS heatmap.")
        return {}

    num_cols = numeric_feature_cols(df)[:_MAX_FEATURES_PLOT]
    comp_ids = sorted(df["competition_id"].dropna().unique())
    comp_pairs = list(combinations(comp_ids, 2))
    labels = competition_labels()

    if len(comp_ids) < 2 or not num_cols:
        logger.warning("Not enough competitions or features for KS heatmap.")
        return {}

    # Pre-split groups once — avoids repeated boolean indexing inside the loop
    groups: dict = {
        cid: df.loc[df["competition_id"] == cid, num_cols]
        for cid in comp_ids
    }

    # Single pass: ks_stats[(col, c1, c2)] = stat
    ks_stats: dict[tuple, float] = {}
    flagged: dict[str, list[str]] = {}
    max_ks: dict[str, float] = {c: 0.0 for c in num_cols}

    for c1, c2 in comp_pairs:
        g1 = groups[c1]
        g2 = groups[c2]
        pair_label = f"{labels.get(str(c1), str(c1))} vs {labels.get(str(c2), str(c2))}"
        for col in num_cols:
            s1 = g1[col].dropna()
            s2 = g2[col].dropna()
            if len(s1) < 10 or len(s2) < 10:
                continue
            stat = cast(float, ks_2samp(s1, s2)[0])
            ks_stats[(col, c1, c2)] = stat
            if stat > max_ks[col]:
                max_ks[col] = stat
            if stat > _KS_FLAG_THRESHOLD:
                flagged.setdefault(col, [])
                if pair_label not in flagged[col]:
                    flagged[col].append(pair_label)

    # Heatmap — reuse stored stats (no second KS loop)
    top_features = sorted(max_ks, key=lambda c: max_ks[c], reverse=True)[:_MAX_FEATURES_PLOT]
    plot_pairs = comp_pairs[:_MAX_PAIRS_PLOT]
    pair_labels = [
        f"{labels.get(str(c1), str(c1)[:8])}\nvs\n{labels.get(str(c2), str(c2)[:8])}"
        for c1, c2 in plot_pairs
    ]
    matrix = np.array(
        [
            [ks_stats.get((col, c1, c2), 0.0) for col in top_features]
            for c1, c2 in plot_pairs
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(max(10, len(top_features) * 0.35), max(4, len(pair_labels) * 0.6))
    )
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=7)
    ax.set_title(
        f"KS Statistic: top-{len(top_features)} drifted features  (threshold={_KS_FLAG_THRESHOLD})"
    )
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_fig("ks_heatmap", "stability")

    return {
        "n_flagged_features": len(flagged),
        "flagged_features": {k: v[:5] for k, v in flagged.items()},
        "max_ks_per_feature": {k: round(v, 4) for k, v in max_ks.items()},
    }


def temporal_stability(df: pd.DataFrame) -> dict:
    """Split by first/second half of match_id sort order; t-test each feature."""
    if "match_id" not in df.columns:
        logger.warning("match_id not in features — skipping temporal stability.")
        return {}

    match_ids = sorted(df["match_id"].dropna().unique())
    n = len(match_ids)
    if n < 4:
        logger.warning("Too few matches for temporal split.")
        return {}

    early = set(match_ids[: n // 2])
    late = set(match_ids[n // 2 :])
    df_early = df[df["match_id"].isin(early)]
    df_late = df[df["match_id"].isin(late)]

    num_cols = numeric_feature_cols(df)
    flagged = {}
    stats_list = []

    for col in num_cols:
        s1 = df_early[col].dropna()
        s2 = df_late[col].dropna()
        if len(s1) < 10 or len(s2) < 10:
            continue
        pval = cast(float, ttest_ind(s1, s2, equal_var=False)[1])
        mean_diff = float(s2.mean() - s1.mean())
        stats_list.append({"feature": col, "pval": float(pval), "mean_diff": mean_diff})
        if pval < _TTEST_PVAL_THRESHOLD:
            flagged[col] = {"pval": round(float(pval), 6), "mean_diff": round(mean_diff, 4)}

    stats_list.sort(key=lambda x: x["pval"])

    # Plot top-20 by p-value
    top20 = stats_list[:20]
    if top20:
        fig, ax = plt.subplots(figsize=(10, 6))
        feats = [r["feature"] for r in top20]
        pvals = [-np.log10(max(r["pval"], 1e-300)) for r in top20]
        ax.barh(feats[::-1], pvals[::-1], color="#1f77b4")
        ax.axvline(-np.log10(0.01), color="red", linestyle="--", label="p=0.01")
        ax.set_xlabel("-log10(p-value)")
        ax.set_title("Temporal Stability: Top 20 features by t-test p-value (early vs late matches)")
        ax.legend()
        plt.tight_layout()
        save_fig("temporal_stability", "stability")

    return {
        "n_early_matches": len(early),
        "n_late_matches": len(late),
        "n_flagged_features": len(flagged),
        "flagged_features": flagged,
    }


def cohort_360_comparison(df: pd.DataFrame) -> dict:
    """Compare feature distributions between 360 and non-360 events (Mann-Whitney U)."""
    if "has_360" not in df.columns:
        logger.warning("has_360 not present — skipping 360 cohort comparison.")
        return {}

    df_360 = df[df["has_360"].astype(bool)]
    df_non360 = df[~df["has_360"].astype(bool)]

    if len(df_360) < 10 or len(df_non360) < 10:
        logger.warning("Not enough data for 360 cohort comparison.")
        return {}

    # Only test features that don't require 360 (available in both cohorts)
    import yaml
    with open(_ROOT / "configs" / "features.yaml", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    non_360_cols = []
    for group, entries in registry.items():
        if group == "identifiers" or not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and not entry.get("requires_360", False):
                col = entry.get("name")
                if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    non_360_cols.append(col)

    flagged = {}
    mwu_stats = []
    for col in non_360_cols:
        s1 = df_360[col].dropna()
        s2 = df_non360[col].dropna()
        if len(s1) < 10 or len(s2) < 10:
            continue
        res = mannwhitneyu(s1, s2, alternative="two-sided")
        stat, pval = cast(float, res[0]), cast(float, res[1])
        mwu_stats.append({"feature": col, "stat": stat, "pval": pval})
        if pval < _MWU_PVAL_THRESHOLD:
            flagged[col] = {"mwu_stat": round(stat, 2), "pval": round(pval, 6)}

    mwu_stats.sort(key=lambda x: x["pval"])
    top20 = mwu_stats[:20]

    if top20:
        fig, ax = plt.subplots(figsize=(10, 6))
        feats = [r["feature"] for r in top20]
        pvals = [-np.log10(max(r["pval"], 1e-300)) for r in top20]
        ax.barh(feats[::-1], pvals[::-1], color="#9467bd")
        ax.axvline(-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
        ax.set_xlabel("-log10(p-value)")
        ax.set_title("360 vs non-360 cohort comparison: top 20 features (Mann-Whitney U)")
        ax.legend()
        plt.tight_layout()
        save_fig("cohort_360_comparison", "stability")

    return {
        "n_360_rows": len(df_360),
        "n_non360_rows": len(df_non360),
        "n_features_tested": len(non_360_cols),
        "n_flagged": len(flagged),
        "flagged_features": flagged,
    }


def domain_validity_checks(df: pd.DataFrame) -> list[dict]:
    """Hard-coded range assertions on domain-constrained features."""
    violations = []

    checks = [
        ("x_location",         lambda s: s.between(0, 105),          "x_location ∈ [0, 105]"),
        ("y_location",         lambda s: s.between(0, 68),           "y_location ∈ [0, 68]"),
        ("distance_to_goal",   lambda s: s.between(0, 120),          "distance_to_goal ∈ [0, 120]"),
        ("shot_angle",         lambda s: s.between(0, np.pi),        "shot_angle ∈ [0, π]"),
        ("pass_length",        lambda s: s >= 0,                     "pass_length ≥ 0"),
        ("carry_distance",     lambda s: s >= 0,                     "carry_distance ≥ 0"),
    ]

    for col, check_fn, desc in checks:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        bad = (~check_fn(series)).sum()
        if bad > 0:
            violations.append({
                "check": desc,
                "column": col,
                "violation_count": int(bad),
                "violation_fraction": round(float(bad) / len(series), 4),
            })
            logger.warning("  VIOLATION: %s — %d rows", desc, bad)
        else:
            logger.info("  OK: %s", desc)

    # Ordering check: defenders_within_3m ≤ 5m ≤ 10m
    d3 = "defenders_within_3m"
    d5 = "defenders_within_5m"
    d10 = "defenders_within_10m"
    if {d3, d5, d10}.issubset(df.columns):
        sub = df[[d3, d5, d10]].dropna()
        for (col_a, col_b, desc) in [
            (d3, d5, "defenders_within_3m ≤ defenders_within_5m"),
            (d5, d10, "defenders_within_5m ≤ defenders_within_10m"),
        ]:
            a = pd.to_numeric(sub[col_a], errors="coerce")
            b = pd.to_numeric(sub[col_b], errors="coerce")
            bad = int((a > b).sum())
            if bad > 0:
                violations.append({
                    "check": desc,
                    "column": f"{col_a}, {col_b}",
                    "violation_count": bad,
                    "violation_fraction": round(bad / len(sub), 4),
                })
                logger.warning("  VIOLATION: %s — %d rows", desc, bad)
            else:
                logger.info("  OK: %s", desc)

    return violations


def main() -> None:
    logger.info("Loading features.parquet …")
    df = load_features()

    logger.info("Step 1: KS competition drift heatmap …")
    ks_result = ks_competition_heatmap(df)

    logger.info("Step 2: Temporal stability (t-test) …")
    temporal = temporal_stability(df)

    logger.info("Step 3: 360 vs non-360 cohort comparison …")
    cohort = cohort_360_comparison(df)

    logger.info("Step 4: Domain validity checks …")
    violations = domain_validity_checks(df)

    save_json({"violations": violations, "n_violations": len(violations)}, "validity_violations")

    summary = {
        "ks_competition_drift": ks_result,
        "temporal_stability": temporal,
        "cohort_360_comparison": cohort,
        "domain_validity_violations": violations,
    }
    save_json(summary, "feature_stability_summary")
    logger.info("02_feature_stability.py complete.")


if __name__ == "__main__":
    main()
