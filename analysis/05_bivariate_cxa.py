"""
analysis/05_bivariate_cxa.py
============================
Part 4b — Bivariate Analysis for CxA (target = shot_created).

shot_created is derived in-script: an action's possession contains ≥1 shot.

Outputs
-------
reports/figures/bivariate/cxa/shot_created_rate_{col}.png
reports/figures/bivariate/cxa/boxplots_numeric.png
reports/figures/bivariate/cxa/progressive_distance.png
reports/figures/bivariate/cxa/shot_created_by_sequence_type.png
reports/figures/bivariate/cxa/sequence_type_standardized_residuals.png
reports/figures/bivariate/cxa/sequence_type_adjusted_odds_ratios.png
reports/bivariate_cxa_sequence_type_summary.json
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

from analysis._utils import (
    derive_shot_created,
    load_actions,
    load_features,
    save_fig,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("05_bivariate_cxa")

_CAT_COLS = [
    "event_type", "box_entry", "cross", "through_ball", "central_progression",
    "under_pressure", "pass_height", "pass_body_part", "score_state",
    "possession_start_zone",
]
_NUM_COLS = [
    "x_location", "y_location", "distance_to_goal", "pass_length",
    "carry_distance", "progressive_distance", "events_before_action",
    "score_differential", "nearest_defender_distance",
]
_SEQUENCE_MIN_N = 200
_MAX_PAIRWISE_TYPES = 12
_LOGIT_CONTROL_COLS = [
    "possession_start_zone",
    "regain_zone",
    "directness",
    "possession_speed",
    "score_differential",
]


def _shot_created_rate_bar(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns or "shot_created" not in df.columns:
        return
    rates = (
        df.groupby(col, observed=True)["shot_created"]
        .agg(rate="mean", n="count")
        .reset_index()
        .sort_values("rate", ascending=False)
    )
    if rates.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(rates) * 0.8), 5))
    bars = ax.bar(rates[col].astype(str), rates["rate"], color="#ff7f0e", alpha=0.8)
    for bar, n in zip(bars, rates["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"n={n:,}",
            ha="center", fontsize=7,
        )
    ax.set_xlabel(col)
    ax.set_ylabel("Shot Creation Rate")
    ax.set_title(f"Shot Creation Rate by {col}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save_fig(f"shot_created_rate_{col}", "bivariate/cxa")


def _numeric_boxplots(df: pd.DataFrame) -> None:
    if "shot_created" not in df.columns:
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
        s0 = pd.to_numeric(df.loc[df["shot_created"] == 0, col], errors="coerce").dropna()
        s1 = pd.to_numeric(df.loc[df["shot_created"] == 1, col], errors="coerce").dropna()
        ax.boxplot(
            [s0.clip(*np.percentile(s0, [1, 99])) if len(s0) else [],
             s1.clip(*np.percentile(s1, [1, 99])) if len(s1) else []],
            tick_labels=["No Shot", "Shot Created"],
            patch_artist=True,
            boxprops=dict(facecolor="#ffbb78"),
        )
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(avail), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Numeric Features: Shot Created vs Not", fontsize=11)
    plt.tight_layout()
    save_fig("boxplots_numeric", "bivariate/cxa")


def _progressive_distance_dist(df: pd.DataFrame) -> None:
    prog_col = next(
        (c for c in ["progressive_distance", "carry_progressive_distance", "pass_length"]
         if c in df.columns),
        None
    )
    if prog_col is None or "shot_created" not in df.columns:
        return

    s0 = pd.to_numeric(df.loc[df["shot_created"] == 0, prog_col], errors="coerce").dropna()
    s1 = pd.to_numeric(df.loc[df["shot_created"] == 1, prog_col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(
        min(s0.quantile(0.01), s1.quantile(0.01)),
        max(s0.quantile(0.99), s1.quantile(0.99)),
        50
    )
    ax.hist(s0, bins=bins, density=True, alpha=0.6, color="#aec7e8", label="No shot created")
    ax.hist(s1, bins=bins, density=True, alpha=0.6, color="#d62728", label="Shot created")
    ax.set_xlabel(prog_col)
    ax.set_ylabel("Density")
    ax.set_title(f"{prog_col} distribution: shot-creating vs non-shot-creating actions")
    ax.legend()
    plt.tight_layout()
    save_fig("progressive_distance", "bivariate/cxa")


def _holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
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


def _build_possession_sequence_df() -> pd.DataFrame:
    """Build one-row-per-possession dataframe with shot_created and sequence context."""
    features = load_features()
    if "possession_id" not in features.columns:
        return pd.DataFrame()

    work = features.copy()
    if "event_type" in work.columns:
        work["shot_flag"] = work["event_type"].astype(str).eq("shot").astype(int)
    else:
        work["shot_flag"] = 0

    agg = {
        "shot_flag": "max",
    }
    for col in ["sequence_type", *_LOGIT_CONTROL_COLS]:
        if col in work.columns:
            agg[col] = "first"

    poss = work.groupby("possession_id", observed=True).agg(agg).reset_index()
    poss = poss.rename(columns={"shot_flag": "shot_created"})
    return poss


def _sequence_rate_table(poss_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        poss_df.groupby("sequence_type", observed=True)["shot_created"]
        .agg(n="count", shots="sum", rate="mean")
        .reset_index()
    )
    if grouped.empty:
        return grouped

    ci_low, ci_high = proportion_confint(
        grouped["shots"].to_numpy(),
        grouped["n"].to_numpy(),
        alpha=0.05,
        method="wilson",
    )
    baseline = float(poss_df["shot_created"].mean())
    grouped["ci_low"] = ci_low
    grouped["ci_high"] = ci_high
    grouped["lift_vs_baseline"] = grouped["rate"] - baseline
    grouped["baseline_rate"] = baseline
    grouped = grouped.sort_values("rate", ascending=False).reset_index(drop=True)
    return grouped


def _plot_sequence_rates(rate_df: pd.DataFrame) -> None:
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
    ax.set_xlabel("Shot-created rate (with 95% Wilson CI)")
    ax.set_title("Shot-created rate by possession sequence type")
    for yi, n in zip(y, rate_df["n"]):
        ax.text(1.005 * max(rate_df["ci_high"].max(), 0.01), yi, f"n={int(n):,}", va="center", fontsize=8)
    plt.tight_layout()
    save_fig("shot_created_by_sequence_type", "bivariate/cxa")


def _sequence_contingency_tests(poss_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    tab = pd.crosstab(poss_df["sequence_type"].astype(str), poss_df["shot_created"].astype(int))
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


def _plot_residual_heatmap(residuals_df: pd.DataFrame) -> None:
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
    ax.set_title("Sequence type x shot_created standardized residuals")
    ax.set_xlabel("shot_created")
    ax.set_ylabel("sequence_type")
    plt.tight_layout()
    save_fig("sequence_type_standardized_residuals", "bivariate/cxa")


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
                count=[int(a["shots"]), int(b["shots"])],
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
    for rec, p_adj in zip(pairs, adj):
        rec["p_value_holm"] = float(p_adj)
    pairs.sort(key=lambda r: r["p_value_holm"])
    return pairs


def _adjusted_logit_sequence_effects(poss_df: pd.DataFrame) -> list[dict]:
    if "shot_created" not in poss_df.columns or "sequence_type" not in poss_df.columns:
        return []
    model_df = poss_df.copy()
    model_df["sequence_type"] = model_df["sequence_type"].astype(str)

    # Avoid unstable coefficient blow-ups from very rare sequence classes.
    seq_counts = model_df["sequence_type"].value_counts()
    keep_seq = seq_counts[seq_counts >= _SEQUENCE_MIN_N].index
    model_df = model_df[model_df["sequence_type"].isin(keep_seq)].copy()

    control_terms: list[str] = []
    for col in _LOGIT_CONTROL_COLS:
        if col not in model_df.columns:
            continue
        nunique = model_df[col].nunique(dropna=True)
        if nunique < 2:
            continue
        if pd.api.types.is_numeric_dtype(model_df[col]):
            control_terms.append(col)
        else:
            control_terms.append(f"C({col})")

    if len(model_df["sequence_type"].dropna().unique()) < 2:
        return []

    fit_df = model_df.dropna(subset=["shot_created", "sequence_type"]).copy()
    if fit_df.empty:
        return []

    model = None
    used_terms: list[str] = []
    # Back off controls progressively until we get a stable fit.
    for k in range(len(control_terms), -1, -1):
        terms = ["C(sequence_type)", *control_terms[:k]]
        formula = "shot_created ~ " + " + ".join(terms)
        try:
            model = smf.logit(formula=formula, data=fit_df).fit(disp=False, maxiter=200)
            used_terms = terms
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Adjusted logit failed with %d controls: %s", k, exc)
            continue

    if model is None:
        logger.warning("Adjusted logit unavailable after fallback attempts.")
        return []

    effects: list[dict] = []
    bse = getattr(model, "bse", None)
    pvals = getattr(model, "pvalues", None)
    for name, coef in model.params.items():
        if not name.startswith("C(sequence_type)"):
            continue
        se = float(bse[name]) if bse is not None and name in bse.index else float("nan")
        lo = coef - 1.96 * se if np.isfinite(se) else float("nan")
        hi = coef + 1.96 * se if np.isfinite(se) else float("nan")
        level = name.split("T.")[-1].rstrip("]") if "T." in name else name
        effects.append(
            {
                "term": name,
                "sequence_type": level,
                "odds_ratio": float(np.exp(coef)),
                "or_ci_low": float(np.exp(lo)) if np.isfinite(lo) else float("nan"),
                "or_ci_high": float(np.exp(hi)) if np.isfinite(hi) else float("nan"),
                "p_value": float(pvals[name]) if pvals is not None and name in pvals.index else float("nan"),
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
    ax.set_title("Adjusted sequence-type effects on shot creation")
    plt.tight_layout()
    save_fig("sequence_type_adjusted_odds_ratios", "bivariate/cxa")


def _sequence_type_bivariate_analysis() -> None:
    poss_df = _build_possession_sequence_df()
    if poss_df.empty or "sequence_type" not in poss_df.columns:
        logger.warning("Possession-level sequence analysis skipped: missing sequence data.")
        return

    rate_df = _sequence_rate_table(poss_df)
    _plot_sequence_rates(rate_df)

    assoc_stats, tab, residuals_df = _sequence_contingency_tests(poss_df)
    _plot_residual_heatmap(residuals_df)

    pairwise = _pairwise_rate_tests(rate_df)
    effects = _adjusted_logit_sequence_effects(poss_df)
    _plot_adjusted_or(effects)

    summary = {
        "n_possessions": int(len(poss_df)),
        "overall_shot_created_rate": float(poss_df["shot_created"].mean()),
        "sequence_rate_table": rate_df.to_dict("records"),
        "association_test": assoc_stats,
        "pairwise_rate_tests_holm": pairwise,
        "adjusted_logit_sequence_effects": effects,
        "contingency_table": tab.reset_index().to_dict("records") if not tab.empty else [],
    }
    save_json(summary, "bivariate_cxa_sequence_type_summary")


def main() -> None:
    logger.info("Loading actions.parquet …")
    actions = load_actions()

    logger.info("Deriving shot_created label …")
    actions = derive_shot_created(actions)

    logger.info("Shot creation rate by categorical features …")
    for col in _CAT_COLS:
        _shot_created_rate_bar(actions, col)

    logger.info("Numeric boxplots by shot_created …")
    _numeric_boxplots(actions)

    logger.info("Progressive distance distribution …")
    _progressive_distance_dist(actions)

    logger.info("Possession-level shot_created vs sequence_type analysis …")
    _sequence_type_bivariate_analysis()

    logger.info("05_bivariate_cxa.py complete.")


if __name__ == "__main__":
    main()
