"""
analysis/07_correlations.py
============================
Part 5 — Correlations & Multicollinearity.

Outputs
-------
reports/figures/correlations/pearson_heatmap.png
reports/figures/correlations/pbcorr_cxg.png
reports/figures/correlations/pbcorr_cxa.png
reports/figures/correlations/cramers_v_heatmap.png
reports/correlation_summary.json
reports/vif_table.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import pointbiserialr, spearmanr

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import (  # noqa: E402
    categorical_feature_cols,
    derive_shot_created,
    load_actions,
    load_features,
    load_shots,
    numeric_feature_cols,
    save_fig,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("07_correlations")

_SPEARMAN_FLAG = 0.85
_VIF_FLAG = 10.0
_MAX_HEATMAP_FEATURES = 35
_FOCUS_360_METRICS = [
    "nearest_defender_distance",
    "second_nearest_defender_distance",
    "nearest_defender_to_receiver",
]


def _cluster_order(corr_matrix: pd.DataFrame) -> list[str]:
    """Return column order based on hierarchical clustering of correlation matrix."""
    dist = 1 - np.abs(corr_matrix.fillna(0).values)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    try:
        Z = linkage(dist, method="average")
        order = leaves_list(Z)
        return list(corr_matrix.columns[order])
    except Exception:
        return list(corr_matrix.columns)


def pearson_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = numeric_feature_cols(df)
    if not num_cols:
        return pd.DataFrame()

    sample = df[num_cols]

    corr = sample.corr(method="pearson")

    # Limit columns for readability
    if len(num_cols) > _MAX_HEATMAP_FEATURES:
        # Keep features with highest mean absolute correlation to others
        mean_abs = corr.abs().mean().sort_values(ascending=False)
        keep = mean_abs.index[:_MAX_HEATMAP_FEATURES].tolist()
        corr = corr.loc[keep, keep]

    ordered = _cluster_order(corr)
    corr = corr.loc[ordered, ordered]

    fig_size = max(12, len(ordered) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.3,
    )
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6, rotation=0)
    ax.set_title("Pearson Correlation Matrix (hierarchically clustered)", fontsize=11)
    plt.tight_layout()
    save_fig("pearson_heatmap", "correlations")
    return corr


def spearman_flagged(df: pd.DataFrame) -> list[dict]:
    """Find feature pairs with |Spearman ρ| > threshold."""
    num_cols = numeric_feature_cols(df)
    if len(num_cols) < 2:
        return []

    sample = df[num_cols]

    high_pairs = []
    cols = list(sample.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            common = sample[[cols[i], cols[j]]].dropna()
            if len(common) < 30:
                continue
            if common[cols[i]].nunique() < 2 or common[cols[j]].nunique() < 2:
                continue
            rho, _ = spearmanr(common[cols[i]], common[cols[j]])
            if abs(rho) > _SPEARMAN_FLAG:
                high_pairs.append(
                    {
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "spearman_rho": round(float(rho), 4),
                    }
                )

    high_pairs.sort(key=lambda x: abs(x["spearman_rho"]), reverse=True)
    logger.info("High Spearman pairs (|ρ| > %.2f): %d", _SPEARMAN_FLAG, len(high_pairs))
    return high_pairs


def point_biserial_plot(df: pd.DataFrame, target_col: str, figure_name: str) -> dict:
    """Point-biserial correlation of each numeric feature vs a binary target."""
    if target_col not in df.columns:
        logger.warning("Target %s not in dataframe — skipping pbcorr.", target_col)
        return {}

    num_cols = numeric_feature_cols(df)
    results = {}

    for col in num_cols:
        if col == target_col:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        t = pd.to_numeric(df[target_col], errors="coerce")
        valid = pd.concat([s, t], axis=1).dropna()
        if len(valid) < 30 or valid[target_col].nunique() < 2 or valid[col].nunique() < 2:
            continue
        r, pval = pointbiserialr(valid[col], valid[target_col])
        results[col] = {"r": round(float(r), 4), "pval": round(float(pval), 6)}

    if not results:
        return results

    sorted_items = sorted(results.items(), key=lambda x: abs(x[1]["r"]), reverse=True)[:30]
    feats = [k for k, _ in sorted_items]
    rs = [v["r"] for _, v in sorted_items]
    colors = ["#2ca02c" if r > 0 else "#d62728" for r in rs]

    fig, ax = plt.subplots(figsize=(10, max(5, len(feats) * 0.35)))
    ax.barh(feats[::-1], rs[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Point-biserial r with {target_col}")
    ax.set_title(f"Feature vs {target_col} (top 30 by |r|)")
    plt.tight_layout()
    save_fig(figure_name, "correlations")
    return results


def cramers_v_heatmap(df: pd.DataFrame) -> None:
    """Cramér's V for all categorical × categorical pairs."""
    cat_cols = categorical_feature_cols(df)
    # Exclude high-cardinality cols
    cat_cols = [c for c in cat_cols if df[c].nunique() <= 50]
    if len(cat_cols) < 2:
        logger.warning("Not enough categorical columns for Cramér's V heatmap.")
        return

    n = len(cat_cols)
    matrix = np.zeros((n, n))

    for i, c1 in enumerate(cat_cols):
        for j, c2 in enumerate(cat_cols):
            if i == j:
                matrix[i, j] = 1.0
                continue
            if i > j:
                matrix[i, j] = matrix[j, i]
                continue
            ct = pd.crosstab(df[c1].astype(str), df[c2].astype(str))
            chi2 = 0.0
            ct_vals = ct.values.astype(float)
            row_sum = ct_vals.sum(axis=1, keepdims=True)
            col_sum = ct_vals.sum(axis=0, keepdims=True)
            total = ct_vals.sum()
            if total == 0:
                continue
            expected = row_sum * col_sum / total
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = float(np.nansum((ct_vals - expected) ** 2 / (expected + 1e-9)))
            k = min(ct.shape) - 1
            v = float(np.sqrt(chi2 / (total * max(k, 1)))) if total > 0 else 0.0
            matrix[i, j] = matrix[j, i] = min(v, 1.0)

    corr_df = pd.DataFrame(matrix, index=cat_cols, columns=cat_cols)
    fig_size = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr_df,
        annot=n <= 15,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.3,
    )
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    ax.set_title("Cramér's V — Categorical Feature Associations", fontsize=10)
    plt.tight_layout()
    save_fig("cramers_v_heatmap", "correlations")


def vif_analysis(df: pd.DataFrame) -> list[dict]:
    """Compute VIF for numeric features; flag VIF > threshold."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Use CxG traditional + contextual numeric features
    num_cols = numeric_feature_cols(df)
    # Subset to columns without too many missing values
    good_cols = [c for c in num_cols if df[c].isnull().mean() < 0.5]
    if len(good_cols) < 2:
        return []

    X = df[good_cols].copy()
    X = X.fillna(X.median())

    # Drop perfectly collinear columns
    corr_mat = X.corr().abs()
    upper = corr_mat.where(np.triu(np.ones_like(corr_mat, dtype=bool), k=1))
    drop_cols = [col for col in upper.columns if any(upper[col] > 0.999)]
    X = X.drop(columns=drop_cols)

    if X.shape[1] < 2:
        return []

    vif_results = []
    for i, col in enumerate(X.columns):
        try:
            vif_val = variance_inflation_factor(X.values, i)
        except Exception:
            vif_val = float("nan")
        vif_results.append(
            {
                "feature": col,
                "vif": round(float(vif_val), 2),
                "flagged": bool(vif_val > _VIF_FLAG),
            }
        )

    vif_results.sort(key=lambda x: x["vif"] if not np.isnan(x["vif"]) else 0, reverse=True)
    flagged = [r for r in vif_results if r["flagged"]]
    logger.info("VIF analysis: %d features flagged (VIF > %.0f)", len(flagged), _VIF_FLAG)
    return vif_results


def focus_360_metric_summary(
    features: pd.DataFrame,
    pbcorr_cxg: dict,
    pbcorr_cxa: dict,
) -> dict:
    """Always report key 360 nearest-distance metrics, even if not top-ranked."""
    summary: dict[str, dict] = {}
    for col in _FOCUS_360_METRICS:
        if col not in features.columns:
            summary[col] = {"present": False}
            continue
        s = pd.to_numeric(features[col], errors="coerce")
        summary[col] = {
            "present": True,
            "null_rate": round(float(s.isna().mean()), 6),
            "zero_rate": round(float((s == 0).mean()), 6),
            "cxg_point_biserial_r": pbcorr_cxg.get(col, {}).get("r"),
            "cxg_point_biserial_p": pbcorr_cxg.get(col, {}).get("pval"),
            "cxa_point_biserial_r": pbcorr_cxa.get(col, {}).get("r"),
            "cxa_point_biserial_p": pbcorr_cxa.get(col, {}).get("pval"),
        }
    return summary


def main() -> None:
    logger.info("Loading features.parquet …")
    features = load_features()

    logger.info("Loading shots.parquet …")
    shots = load_shots()

    logger.info("Loading actions.parquet …")
    actions = load_actions()
    actions = derive_shot_created(actions)

    logger.info("Step 1: Pearson correlation heatmap …")
    pearson_corr = pearson_heatmap(features)

    logger.info("Step 2: Spearman high-correlation pairs …")
    spearman_pairs = spearman_flagged(features)

    logger.info("Step 3: Point-biserial vs goal (shots) …")
    pbcorr_cxg = point_biserial_plot(shots, "goal", "pbcorr_cxg")

    logger.info("Step 4: Point-biserial vs shot_created (actions) …")
    pbcorr_cxa = point_biserial_plot(actions, "shot_created", "pbcorr_cxa")

    logger.info("Step 5: Cramér's V heatmap …")
    cramers_v_heatmap(features)

    logger.info("Step 6: VIF analysis …")
    vif_results = vif_analysis(features)
    save_json({"vif": vif_results, "vif_flag_threshold": _VIF_FLAG}, "vif_table")

    # Top-10 Pearson pairs
    top_pearson = []
    if not pearson_corr.empty:
        p = pearson_corr.abs().unstack().reset_index()
        p.columns = ["feat_a", "feat_b", "abs_corr"]
        p = p[p["feat_a"] < p["feat_b"]].sort_values("abs_corr", ascending=False).head(10)
        top_pearson = p.to_dict("records")

    summary = {
        "n_spearman_high_pairs": len(spearman_pairs),
        "spearman_high_pairs": spearman_pairs[:20],
        "top_10_pearson_pairs": top_pearson,
        "top_features_cxg": sorted(pbcorr_cxg, key=lambda k: abs(pbcorr_cxg[k]["r"]), reverse=True)[
            :10
        ],
        "top_features_cxa": sorted(pbcorr_cxa, key=lambda k: abs(pbcorr_cxa[k]["r"]), reverse=True)[
            :10
        ],
        "focus_360_metrics": focus_360_metric_summary(features, pbcorr_cxg, pbcorr_cxa),
        "n_vif_flagged": sum(1 for r in vif_results if r.get("flagged")),
    }
    save_json(summary, "correlation_summary")
    logger.info("07_correlations.py complete.")


if __name__ == "__main__":
    main()
