"""
Phase 8: Formal Statistical vs Neural Comparison — model_comparison.py

Run a fixed evaluation protocol across all three metric families (CxG, CxA, CxT)
on a common held-out test set.

Design goals
------------
1. Single entry point: ``ModelComparisonSuite.run(...)`` accepts pre-fitted model
   objects and a test DataFrame, then evaluates every model on every dimension.
2. Framework-agnostic: models only need to expose `predict_proba(df)` (classifiers)
   or `predict(df)` (regressors). No re-fitting occurs here.
3. Bootstrap stability: 20 resamples of the test set, capturing variance of metrics.
4. Odd/even leaderboard reliability: split matches into odd/even sets and measure
   Spearman rank correlation of player CxT/CxA totals.
5. Output: ``ComparisonReport`` dataclass → HTML via ``build_html_report``.

Evaluation dimensions
---------------------
Classification (CxG, CxA-creation):
  log_loss, brier_score, roc_auc, pr_auc, ece, reliability_diagram_data

Regression (CxA-quality, CxT):
  mae, rmse, spearman, calibration_by_bucket, ece_regression

Shared:
  bootstrap_std_{metric}   — standard deviation across 20 bootstrap resamples
  leaderboard_rank_corr    — Spearman ρ of odd/even match player rankings
  promotion_verdict        — 'promote' | 'keep_tree' | 'insufficient_data'

Promotion criteria (from configs/models.yaml):
  • log_loss or Brier improvement > 5 % over best tree model
  • ECE ≤ tree model ECE
  • Leaderboard rank correlation ≥ 0.80
  • Improvement holds on non-360 test set too
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Default bootstrap resamples
N_BOOTSTRAP = 20

# Promotion thresholds (mirrors configs/models.yaml)
PROMO_LOG_LOSS_IMPROVEMENT = 0.05
PROMO_MAX_ECE_RATIO = 1.0
PROMO_MIN_RANK_CORR = 0.80


# ── Metric dataclasses ────────────────────────────────────────────────────────

@dataclass
class ClassificationMetrics:
    """Discrimination + calibration for binary classifiers."""
    log_loss: float
    brier: float
    roc_auc: float | None
    pr_auc: float | None
    ece: float
    reliability_bins: list[dict] = field(default_factory=list)
    # Bootstrap standard deviations (filled by bootstrap pass)
    log_loss_std: float = 0.0
    brier_std: float = 0.0
    roc_auc_std: float = 0.0


@dataclass
class RegressionMetrics:
    """Discrimination + calibration for regression models."""
    mae: float
    rmse: float
    spearman: float | None
    calibration_by_bucket: dict[str, float] = field(default_factory=dict)
    # Bootstrap standard deviations
    mae_std: float = 0.0
    rmse_std: float = 0.0
    spearman_std: float = 0.0


@dataclass
class ModelComparisonResult:
    """Single model's full evaluation result."""
    name: str
    family: str          # 'baseline' | 'glm' | 'tree' | 'neural_tabular' | 'neural_seq' | 'neural_360'
    metric_type: str     # 'cxg' | 'cxa' | 'cxt'
    task_type: str       # 'classification' | 'regression'
    feature_set: str
    metrics: ClassificationMetrics | RegressionMetrics
    leaderboard_rank_corr: float | None = None
    promotion_verdict: str = "not_evaluated"
    is_360_only: bool = False


@dataclass
class ComparisonReport:
    """Full comparison report across all models and metric families."""
    results: list[ModelComparisonResult]
    promotion_summary: dict[str, str]   # metric_type → recommended_model_name
    best_tree: dict[str, str]           # metric_type → best tree model name

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten results to a single DataFrame."""
        rows = []
        for r in self.results:
            m = r.metrics
            row: dict = {
                "name": r.name,
                "family": r.family,
                "metric_type": r.metric_type,
                "task_type": r.task_type,
                "feature_set": r.feature_set,
                "promotion_verdict": r.promotion_verdict,
                "leaderboard_rank_corr": r.leaderboard_rank_corr,
                "is_360_only": r.is_360_only,
            }
            if isinstance(m, ClassificationMetrics):
                row.update({
                    "log_loss": m.log_loss,
                    "brier": m.brier,
                    "roc_auc": m.roc_auc,
                    "pr_auc": m.pr_auc,
                    "ece": m.ece,
                    "log_loss_std": m.log_loss_std,
                    "brier_std": m.brier_std,
                    "roc_auc_std": m.roc_auc_std,
                })
            elif isinstance(m, RegressionMetrics):
                row.update({
                    "mae": m.mae,
                    "rmse": m.rmse,
                    "spearman": m.spearman,
                    "mae_std": m.mae_std,
                    "rmse_std": m.rmse_std,
                    "spearman_std": m.spearman_std,
                })
            rows.append(row)
        return pd.DataFrame(rows)


# ── Metric computation helpers ────────────────────────────────────────────────

def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (classification)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Reliability diagram data for a classifier."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        bins.append({
            "bin_lo": round(lo, 2),
            "bin_hi": round(hi, 2),
            "count": int(mask.sum()),
            "mean_pred": float(y_prob[mask].mean()) if mask.sum() > 0 else 0.0,
            "mean_actual": float(y_true[mask].mean()) if mask.sum() > 0 else 0.0,
        })
    return bins


def _ece_regression(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Calibration by value bucket for regressors."""
    try:
        non_neg = y_true > 0
        if non_neg.sum() < n_bins:
            return {}
        q = np.quantile(y_true[non_neg], np.linspace(0, 1, n_bins + 1))
        buckets: dict[str, float] = {}
        for i in range(n_bins):
            mask = (y_true >= q[i]) & (y_true < q[i + 1])
            if mask.sum() > 0:
                buckets[f"q{i+1}"] = float(np.mean(y_pred[mask]) - np.mean(y_true[mask]))
        return buckets
    except Exception:
        return {}


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> ClassificationMetrics:
    """Compute full classification metric set."""
    ll = float(log_loss(y_true, y_prob))
    br = float(brier_score_loss(y_true, y_prob))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None
    try:
        prauc = float(average_precision_score(y_true, y_prob))
    except Exception:
        prauc = None
    ece = _ece(y_true, y_prob)
    rel_bins = _reliability_bins(y_true, y_prob)
    return ClassificationMetrics(
        log_loss=ll, brier=br, roc_auc=auc, pr_auc=prauc,
        ece=ece, reliability_bins=rel_bins
    )


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute full regression metric set."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    corr, _ = spearmanr(y_true, y_pred)
    sp = float(corr) if not np.isnan(corr) else None
    buckets = _ece_regression(y_true, y_pred)
    return RegressionMetrics(mae=mae, rmse=rmse, spearman=sp, calibration_by_bucket=buckets)


def _bootstrap_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (log_loss_std, brier_std, roc_auc_std) from bootstrap resamples."""
    if rng is None:
        rng = np.random.default_rng(0)
    lls, brs, aucs = [], [], []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        lls.append(float(log_loss(y_true[idx], y_prob[idx])))
        brs.append(float(brier_score_loss(y_true[idx], y_prob[idx])))
        try:
            aucs.append(float(roc_auc_score(y_true[idx], y_prob[idx])))
        except Exception:
            pass
    return (
        float(np.std(lls)),
        float(np.std(brs)),
        float(np.std(aucs)) if aucs else 0.0,
    )


def _bootstrap_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mae_std, rmse_std, spearman_std) from bootstrap resamples."""
    if rng is None:
        rng = np.random.default_rng(0)
    maes, rmses, sps = [], [], []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        maes.append(float(mean_absolute_error(y_true[idx], y_pred[idx])))
        rmses.append(float(np.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))))
        corr, _ = spearmanr(y_true[idx], y_pred[idx])
        if not np.isnan(corr):
            sps.append(float(corr))
    return (
        float(np.std(maes)),
        float(np.std(rmses)),
        float(np.std(sps)) if sps else 0.0,
    )


# ── Leaderboard reliability ───────────────────────────────────────────────────

def leaderboard_rank_correlation(
    scored_df: pd.DataFrame,
    player_id_col: str = "player_id",
    match_id_col: str = "match_id",
    value_col: str = "cxt",
    min_actions: int = 10,
) -> float | None:
    """
    Split matches into odd/even sets and measure Spearman rank correlation
    of per-player value totals between the two halves.

    Returns None if there are fewer than 5 common players across both halves.
    """
    if match_id_col not in scored_df.columns or player_id_col not in scored_df.columns:
        return None
    if value_col not in scored_df.columns:
        return None

    matches = sorted(scored_df[match_id_col].unique())
    odd_matches = {m for i, m in enumerate(matches) if i % 2 == 0}
    even_matches = {m for i, m in enumerate(matches) if i % 2 == 1}

    df_odd = scored_df[scored_df[match_id_col].isin(odd_matches)]
    df_even = scored_df[scored_df[match_id_col].isin(even_matches)]

    odd_totals = df_odd.groupby(player_id_col)[value_col].sum()
    even_totals = df_even.groupby(player_id_col)[value_col].sum()

    common = odd_totals.index.intersection(even_totals.index)
    # Apply minimum actions filter
    odd_counts = df_odd.groupby(player_id_col)[value_col].count()
    even_counts = df_even.groupby(player_id_col)[value_col].count()
    valid = [p for p in common
             if odd_counts.get(p, 0) >= min_actions and even_counts.get(p, 0) >= min_actions]

    if len(valid) < 5:
        return None

    corr, _ = spearmanr(odd_totals[valid].to_numpy(), even_totals[valid].to_numpy())
    return float(corr) if not np.isnan(corr) else None


# ── Promotion logic ───────────────────────────────────────────────────────────

def evaluate_promotion(
    candidate: ModelComparisonResult,
    best_tree: ModelComparisonResult,
    rank_corr: float | None = None,
    min_improvement: float = PROMO_LOG_LOSS_IMPROVEMENT,
    max_ece_ratio: float = PROMO_MAX_ECE_RATIO,
    min_rank_corr: float = PROMO_MIN_RANK_CORR,
) -> str:
    """
    Return 'promote', 'keep_tree', or 'insufficient_data'.

    Promotion requires ALL of:
    - log_loss or Brier improvement > min_improvement (5 %) over best tree
    - ECE ≤ tree ECE (ratio ≤ max_ece_ratio)
    - leaderboard rank correlation ≥ min_rank_corr
    - does not rely on 360 only (checked by caller)
    """
    if candidate.task_type == "classification":
        cand_m = candidate.metrics
        tree_m = best_tree.metrics
        if not isinstance(cand_m, ClassificationMetrics) or not isinstance(tree_m, ClassificationMetrics):
            return "insufficient_data"

        ll_improvement = (tree_m.log_loss - cand_m.log_loss) / max(tree_m.log_loss, 1e-9)
        br_improvement = (tree_m.brier - cand_m.brier) / max(tree_m.brier, 1e-9)
        ece_ratio = cand_m.ece / max(tree_m.ece, 1e-9)

        passes_discrimination = (ll_improvement > min_improvement or br_improvement > min_improvement)
        passes_calibration = ece_ratio <= max_ece_ratio

    elif candidate.task_type == "regression":
        cand_m = candidate.metrics
        tree_m = best_tree.metrics
        if not isinstance(cand_m, RegressionMetrics) or not isinstance(tree_m, RegressionMetrics):
            return "insufficient_data"

        mae_improvement = (tree_m.mae - cand_m.mae) / max(tree_m.mae, 1e-9)
        passes_discrimination = mae_improvement > min_improvement
        # ECE proxy: mean absolute calibration bucket error
        cand_cal = float(np.mean(np.abs(list(cand_m.calibration_by_bucket.values())))) if cand_m.calibration_by_bucket else float("inf")
        tree_cal = float(np.mean(np.abs(list(tree_m.calibration_by_bucket.values())))) if tree_m.calibration_by_bucket else float("inf")
        ece_ratio = cand_cal / max(tree_cal, 1e-9) if tree_cal < float("inf") else 0.0
        passes_calibration = ece_ratio <= max_ece_ratio
    else:
        return "insufficient_data"

    if rank_corr is None:
        passes_rank = None  # unknown
    else:
        passes_rank = rank_corr >= min_rank_corr

    if candidate.is_360_only:
        return "keep_tree"  # Criterion 4: must not rely exclusively on 360

    if not passes_discrimination:
        return "keep_tree"
    if not passes_calibration:
        return "keep_tree"
    if passes_rank is False:
        return "keep_tree"

    return "promote"


# ── Model entry specification ─────────────────────────────────────────────────

@dataclass
class ModelEntry:
    """Descriptor for a model to be evaluated."""
    name: str
    family: str           # 'baseline' | 'glm' | 'gam' | 'tree' | 'neural_tabular' | ...
    metric_type: str      # 'cxg' | 'cxa' | 'cxt'
    task_type: str        # 'classification' | 'regression'
    feature_set: str
    model: Any            # must expose predict_proba(df) or predict(df)
    is_360_only: bool = False
    predict_fn: Callable | None = None   # override for custom predict logic

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Dispatch to predict_proba (classifiers) or predict (regressors)."""
        if self.predict_fn is not None:
            return self.predict_fn(df)
        if self.task_type == "classification":
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba(df)
                # Return positive-class probability
                if hasattr(p, "ndim") and p.ndim == 2:
                    return p[:, 1]
                return p
        return np.clip(self.model.predict(df), 0.0, None)


# ── Main comparison suite ─────────────────────────────────────────────────────

class ModelComparisonSuite:
    """
    Orchestrates model evaluation across all metric families.

    Usage
    -----
    suite = ModelComparisonSuite()
    suite.add_model(ModelEntry(name="glm_cxg", family="glm", metric_type="cxg",
                               task_type="classification", feature_set="contextual",
                               model=glm_model))
    report = suite.run(test_df, target_map={"cxg": "goal", "cxt": "possession_cxg"})
    """

    def __init__(
        self,
        n_bootstrap: int = N_BOOTSTRAP,
        player_id_col: str = "player_id",
        match_id_col: str = "match_id",
        random_state: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.player_id_col = player_id_col
        self.match_id_col = match_id_col
        self.random_state = random_state
        self._entries: list[ModelEntry] = []

    def add_model(self, entry: ModelEntry) -> "ModelComparisonSuite":
        self._entries.append(entry)
        return self

    def add_models(self, entries: Sequence[ModelEntry]) -> "ModelComparisonSuite":
        self._entries.extend(entries)
        return self

    def run(
        self,
        test_df: pd.DataFrame,
        target_map: dict[str, str] | None = None,
        leaderboard_value_col: str = "cxt",
    ) -> ComparisonReport:
        """
        Evaluate all registered models on test_df.

        Parameters
        ----------
        test_df:
            Held-out test DataFrame with all feature columns + target columns.
        target_map:
            Dict mapping metric_type → target_column_name in test_df.
            Defaults: {'cxg': 'goal', 'cxa': 'shot_created', 'cxt': 'possession_cxg'}
        leaderboard_value_col:
            Column to use for leaderboard rank correlation (default 'cxt').
        """
        if not self._entries:
            raise ValueError("No models registered. Call add_model() first.")

        if target_map is None:
            target_map = {
                "cxg": "goal",
                "cxa": "shot_created",
                "cxt": "possession_cxg",
            }

        rng = np.random.default_rng(self.random_state)
        results: list[ModelComparisonResult] = []

        for entry in self._entries:
            target_col = target_map.get(entry.metric_type)
            if target_col is None or target_col not in test_df.columns:
                logger.warning("Skipping %s — target column %r not in test_df", entry.name, target_col)
                continue

            logger.info("Evaluating %s (%s / %s)…", entry.name, entry.metric_type, entry.task_type)
            y_true = test_df[target_col].astype(float).to_numpy()

            try:
                y_pred = entry.predict(test_df)
            except Exception as exc:
                logger.warning("Model %s failed predict: %s", entry.name, exc)
                continue

            if entry.task_type == "classification":
                y_prob = np.clip(y_pred, 1e-7, 1 - 1e-7)
                metrics = compute_classification_metrics(y_true, y_prob)
                ll_std, br_std, auc_std = _bootstrap_classification(
                    y_true, y_prob, n_bootstrap=self.n_bootstrap, rng=rng
                )
                metrics.log_loss_std = ll_std
                metrics.brier_std = br_std
                metrics.roc_auc_std = auc_std
            else:
                metrics = compute_regression_metrics(y_true, y_pred)
                mae_std, rmse_std, sp_std = _bootstrap_regression(
                    y_true, y_pred, n_bootstrap=self.n_bootstrap, rng=rng
                )
                metrics.mae_std = mae_std
                metrics.rmse_std = rmse_std
                metrics.spearman_std = sp_std

            # Leaderboard reliability (requires player_id and match_id in scored df)
            # Build a scored df for this model
            rank_corr: float | None = None
            if self.player_id_col in test_df.columns and self.match_id_col in test_df.columns:
                scored = test_df[[self.player_id_col, self.match_id_col]].copy()
                scored[leaderboard_value_col] = y_pred
                rank_corr = leaderboard_rank_correlation(
                    scored,
                    player_id_col=self.player_id_col,
                    match_id_col=self.match_id_col,
                    value_col=leaderboard_value_col,
                )

            results.append(ModelComparisonResult(
                name=entry.name,
                family=entry.family,
                metric_type=entry.metric_type,
                task_type=entry.task_type,
                feature_set=entry.feature_set,
                metrics=metrics,
                leaderboard_rank_corr=rank_corr,
                is_360_only=entry.is_360_only,
            ))

        # Determine best tree per metric_type
        best_tree: dict[str, ModelComparisonResult] = {}
        for metric_type in {r.metric_type for r in results}:
            trees = [r for r in results if r.metric_type == metric_type and r.family == "tree"]
            if trees:
                if trees[0].task_type == "classification":
                    best_tree[metric_type] = min(trees, key=lambda r: r.metrics.log_loss)
                else:
                    best_tree[metric_type] = min(trees, key=lambda r: r.metrics.mae)

        # Apply promotion logic
        promotion_summary: dict[str, str] = {}
        for r in results:
            if r.family not in ("neural_tabular", "neural_seq", "neural_360"):
                r.promotion_verdict = "not_applicable"
                continue
            bt = best_tree.get(r.metric_type)
            if bt is None:
                r.promotion_verdict = "insufficient_data"
                continue
            r.promotion_verdict = evaluate_promotion(r, bt, rank_corr=r.leaderboard_rank_corr)

        # Summarise: best model per metric_type
        for metric_type in {r.metric_type for r in results}:
            group = [r for r in results if r.metric_type == metric_type]
            best = _pick_best(group)
            if best is not None:
                promotion_summary[metric_type] = best.name

        return ComparisonReport(
            results=results,
            promotion_summary=promotion_summary,
            best_tree={k: v.name for k, v in best_tree.items()},
        )


def _pick_best(results: list[ModelComparisonResult]) -> ModelComparisonResult | None:
    """Pick overall best model by primary metric (log_loss for classifiers, mae for regressors)."""
    if not results:
        return None
    classifiers = [r for r in results if r.task_type == "classification"]
    regressors = [r for r in results if r.task_type == "regression"]
    if classifiers:
        return min(classifiers, key=lambda r: r.metrics.log_loss)
    if regressors:
        return min(regressors, key=lambda r: r.metrics.mae)
    return None


# ── HTML report builder ───────────────────────────────────────────────────────

def build_html_report(report: ComparisonReport, output_path: str | None = None) -> str:
    """
    Generate an HTML model comparison matrix from a ComparisonReport.

    Parameters
    ----------
    report:     ComparisonReport produced by ModelComparisonSuite.run()
    output_path: Optional file path to write the HTML (e.g. 'reports/model_comparison_matrix.html')

    Returns
    -------
    HTML string.
    """
    from pathlib import Path

    df = report.to_dataframe()
    if df.empty or "metric_type" not in df.columns:
        # Return a minimal valid HTML page for empty reports
        empty_html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Model Comparison Matrix</title></head>
<body><h1>Model Comparison Matrix</h1><p>No results to display.</p></body></html>"""
        if output_path is not None:
            from pathlib import Path as _Path
            _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            _Path(output_path).write_text(empty_html, encoding="utf-8")
        return empty_html

    # ── CSS & header ──────────────────────────────────────────────────────────
    html = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Contextual Football Metrics — Model Comparison Matrix</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f7fa; color: #1a202c; margin: 0; padding: 24px; }
  h1   { font-size: 1.6rem; margin-bottom: 4px; }
  h2   { font-size: 1.2rem; margin: 24px 0 8px; color: #2d3748; }
  .subtitle { color: #718096; font-size: 0.9rem; margin-bottom: 24px; }
  .summary-box { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
  .card { background: white; border-radius: 8px; padding: 16px 20px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); min-width: 180px; }
  .card .label { font-size: 0.75rem; text-transform: uppercase;
                 letter-spacing: 0.05em; color: #718096; }
  .card .value { font-size: 1.2rem; font-weight: 600; margin-top: 4px; }
  .card.promote  { border-left: 4px solid #38a169; }
  .card.keep     { border-left: 4px solid #3182ce; }
  table { width: 100%; border-collapse: collapse; background: white;
          border-radius: 8px; overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 32px; }
  th    { background: #edf2f7; padding: 10px 12px; text-align: left;
          font-size: 0.8rem; text-transform: uppercase;
          letter-spacing: 0.04em; color: #4a5568; }
  td    { padding: 8px 12px; font-size: 0.85rem;
          border-top: 1px solid #edf2f7; }
  tr:hover td { background: #f7fafc; }
  .tag  { display: inline-block; padding: 2px 8px; border-radius: 12px;
          font-size: 0.72rem; font-weight: 600; }
  .promote  { background: #c6f6d5; color: #276749; }
  .keep_tree { background: #bee3f8; color: #1a365d; }
  .not_applicable { background: #edf2f7; color: #4a5568; }
  .insufficient_data { background: #fefcbf; color: #744210; }
  .best { font-weight: 700; color: #276749; }
</style>
</head>
<body>
<h1>Model Comparison Matrix</h1>
<p class="subtitle">Contextual Football Metrics — Phase 8 Evaluation &mdash;
  Comparing all model families across CxG, CxA, and CxT on the held-out test set.</p>
"""]

    # ── Promotion summary cards ────────────────────────────────────────────────
    html.append('<div class="summary-box">')
    for metric_type, model_name in report.promotion_summary.items():
        verdict = "promote" if any(
            r.promotion_verdict == "promote" and r.name == model_name
            for r in report.results
        ) else "keep"
        html.append(f"""<div class="card {verdict}">
  <div class="label">{metric_type.upper()} — recommended</div>
  <div class="value">{model_name}</div>
</div>""")
    html.append("</div>")

    # ── Per-metric-type tables ─────────────────────────────────────────────────
    for metric_type in sorted(df["metric_type"].unique()):
        sub = df[df["metric_type"] == metric_type].copy()
        # Determine primary metric column
        is_clf = (sub["task_type"] == "classification").any()
        primary_col = "log_loss" if is_clf else "mae"
        std_col = "log_loss_std" if is_clf else "mae_std"
        secondary_cols = (
            ["brier", "roc_auc", "pr_auc", "ece"] if is_clf
            else ["rmse", "spearman"]
        )

        html.append(f"<h2>{metric_type.upper()}</h2>")
        html.append("<table>")
        headers = ["Model", "Family", "Feature Set"]
        headers += [primary_col.replace("_", " ").title(), "± (std)"]
        headers += [c.replace("_", " ").title() for c in secondary_cols]
        headers += ["Rank Corr", "Verdict"]
        html.append("<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")

        if primary_col in sub.columns:
            sub = sub.sort_values(primary_col, ascending=is_clf or primary_col == "mae")

        best_primary = sub[primary_col].min() if primary_col in sub.columns else None

        for _, row in sub.iterrows():
            verdict = str(row.get("promotion_verdict", ""))
            tag = f'<span class="tag {verdict}">{verdict.replace("_", " ")}</span>'
            primary_val = row.get(primary_col)
            std_val = row.get(std_col, 0.0)
            is_best = (primary_val is not None and best_primary is not None and
                       abs(primary_val - best_primary) < 1e-9)
            cls = 'class="best"' if is_best else ""
            cells = [
                f"<td {cls}>{row['name']}</td>",
                f"<td>{row['family']}</td>",
                f"<td>{row['feature_set']}</td>",
                f"<td {cls}>{_fmt(primary_val, 5)}</td>",
                f"<td>±{_fmt(std_val, 4)}</td>",
            ]
            for c in secondary_cols:
                cells.append(f"<td>{_fmt(row.get(c), 4)}</td>")
            rank_corr = row.get("leaderboard_rank_corr")
            cells.append(f"<td>{_fmt(rank_corr, 3)}</td>")
            cells.append(f"<td>{tag}</td>")
            html.append("<tr>" + "".join(cells) + "</tr>")

        html.append("</table>")

    html.append("</body></html>")
    result = "\n".join(html)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(result, encoding="utf-8")
        logger.info("HTML report written to %s", output_path)

    return result


def _fmt(value, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)
