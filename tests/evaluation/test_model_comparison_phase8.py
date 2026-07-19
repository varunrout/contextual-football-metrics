"""
Phase 8 tests — Formal Statistical vs Neural Comparison.

Covers:
  • compute_classification_metrics — log_loss, brier, roc_auc, pr_auc, ece
  • compute_regression_metrics — mae, rmse, spearman, calibration_by_bucket
  • _ece — calibration error computation
  • _bootstrap_classification — std estimates are non-negative
  • _bootstrap_regression — std estimates are non-negative
  • leaderboard_rank_correlation — odd/even match Spearman rank correlation
  • evaluate_promotion — promote / keep_tree / insufficient_data
  • ModelEntry.predict — dispatches to predict_proba / predict
  • ModelComparisonSuite — add_model, run, promotion logic, report shape
  • ComparisonReport.to_dataframe — column coverage
  • build_html_report — returns valid HTML string, writes file
  • Edge cases: missing target column, empty results, no tree baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.model_comparison import (
    ClassificationMetrics,
    ComparisonReport,
    ModelComparisonResult,
    ModelComparisonSuite,
    ModelEntry,
    RegressionMetrics,
    build_html_report,
    compute_classification_metrics,
    compute_regression_metrics,
    evaluate_promotion,
    leaderboard_rank_correlation,
)

# ── Synthetic data helpers ─────────────────────────────────────────────────────


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _clf_data(n: int = 500, seed: int = 0):
    rng = _rng(seed)
    y_true = rng.integers(0, 2, n).astype(float)
    y_prob = np.clip(y_true * 0.6 + rng.uniform(0, 0.4, n), 0.05, 0.95)
    return y_true, y_prob


def _reg_data(n: int = 500, seed: int = 0):
    rng = _rng(seed)
    y_true = rng.uniform(0, 0.5, n)
    y_pred = np.clip(y_true + rng.normal(0, 0.05, n), 0, None)
    return y_true, y_pred


def _make_test_df(n: int = 300, n_matches: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = _rng(seed)
    match_ids = [f"m{i % n_matches}" for i in range(n)]
    player_ids = [f"p{i % 20}" for i in range(n)]
    return pd.DataFrame(
        {
            "match_id": match_ids,
            "player_id": player_ids,
            "x_location": rng.uniform(20, 105, n),
            "y_location": rng.uniform(0, 68, n),
            "goal": rng.integers(0, 2, n).astype(float),
            "shot_created": rng.integers(0, 2, n).astype(float),
            "possession_cxg": rng.uniform(0, 0.4, n),
        }
    )


class _FakeClassifier:
    """Stub binary classifier for testing."""

    def __init__(self, n: int = 300, seed: int = 0):
        rng = _rng(seed)
        self._proba = rng.uniform(0.1, 0.9, n)
        self._n = n

    def predict_proba(self, df):
        return np.column_stack([1 - self._proba, self._proba])


class _FakeRegressor:
    """Stub regressor for testing."""

    def __init__(self, n: int = 300, seed: int = 0):
        rng = _rng(seed)
        self._pred = rng.uniform(0, 0.3, n)
        self._n = n

    def predict(self, df):
        return self._pred


# ═══════════════════════════════════════════════════════════════════════════════
# Classification Metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeClassificationMetrics:
    def test_returns_classification_metrics_instance(self):
        y_true, y_prob = _clf_data()
        m = compute_classification_metrics(y_true, y_prob)
        assert isinstance(m, ClassificationMetrics)

    def test_log_loss_positive(self):
        y_true, y_prob = _clf_data()
        m = compute_classification_metrics(y_true, y_prob)
        assert m.log_loss > 0

    def test_brier_in_range(self):
        y_true, y_prob = _clf_data()
        m = compute_classification_metrics(y_true, y_prob)
        assert 0.0 <= m.brier <= 1.0

    def test_roc_auc_in_range(self):
        y_true, y_prob = _clf_data()
        m = compute_classification_metrics(y_true, y_prob)
        assert 0.0 <= m.roc_auc <= 1.0

    def test_pr_auc_in_range(self):
        y_true, y_prob = _clf_data()
        m = compute_classification_metrics(y_true, y_prob)
        assert 0.0 <= m.pr_auc <= 1.0

    def test_ece_in_range(self):
        y_true, y_prob = _clf_data()
        m = compute_classification_metrics(y_true, y_prob)
        assert 0.0 <= m.ece <= 1.0

    def test_reliability_bins_length(self):
        y_true, y_prob = _clf_data()
        m = compute_classification_metrics(y_true, y_prob)
        assert len(m.reliability_bins) == 10

    def test_perfect_predictions_low_log_loss(self):
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob = np.array([0.01, 0.01, 0.99, 0.99])
        m = compute_classification_metrics(y_true, y_prob)
        assert m.log_loss < 0.1

    def test_random_predictions_high_log_loss(self):
        rng = _rng(1)
        y_true = rng.integers(0, 2, 500).astype(float)
        y_prob = np.full(500, 0.5)
        m = compute_classification_metrics(y_true, y_prob)
        assert m.log_loss > 0.65  # near max entropy


# ═══════════════════════════════════════════════════════════════════════════════
# Regression Metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeRegressionMetrics:
    def test_returns_regression_metrics_instance(self):
        y_true, y_pred = _reg_data()
        m = compute_regression_metrics(y_true, y_pred)
        assert isinstance(m, RegressionMetrics)

    def test_mae_positive(self):
        y_true, y_pred = _reg_data()
        m = compute_regression_metrics(y_true, y_pred)
        assert m.mae >= 0

    def test_rmse_gte_mae(self):
        y_true, y_pred = _reg_data()
        m = compute_regression_metrics(y_true, y_pred)
        assert m.rmse >= m.mae

    def test_spearman_in_range(self):
        y_true, y_pred = _reg_data()
        m = compute_regression_metrics(y_true, y_pred)
        assert -1.0 <= m.spearman <= 1.0

    def test_perfect_predictions_zero_mae(self):
        y_true = np.array([0.1, 0.2, 0.3, 0.4])
        m = compute_regression_metrics(y_true, y_true)
        assert m.mae == pytest.approx(0.0)
        assert m.rmse == pytest.approx(0.0)
        assert m.spearman == pytest.approx(1.0)

    def test_calibration_by_bucket_is_dict(self):
        y_true, y_pred = _reg_data()
        m = compute_regression_metrics(y_true, y_pred)
        assert isinstance(m.calibration_by_bucket, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# ECE helper
# ═══════════════════════════════════════════════════════════════════════════════


class TestEce:
    def test_perfectly_calibrated_has_near_zero_ece(self):
        from src.evaluation.model_comparison import _ece

        # Create perfectly calibrated predictions
        y_prob = np.linspace(0.05, 0.95, 100)
        y_true = (np.random.default_rng(0).uniform(0, 1, 100) < y_prob).astype(float)
        ece = _ece(y_true, y_prob)
        assert ece >= 0.0  # can't be negative
        assert ece < 0.3  # should be relatively small

    def test_wrong_way_predictions_have_high_ece(self):
        from src.evaluation.model_comparison import _ece

        y_prob = np.array([0.9] * 100)  # all predict positive
        y_true = np.array([0.0] * 100)  # all actually negative
        ece = _ece(y_true, y_prob)
        assert ece > 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestBootstrap:
    def test_bootstrap_classification_returns_non_negative_stds(self):
        from src.evaluation.model_comparison import _bootstrap_classification

        y_true, y_prob = _clf_data(n=200)
        ll_std, br_std, auc_std = _bootstrap_classification(y_true, y_prob, n_bootstrap=10)
        assert ll_std >= 0
        assert br_std >= 0
        assert auc_std >= 0

    def test_bootstrap_regression_returns_non_negative_stds(self):
        from src.evaluation.model_comparison import _bootstrap_regression

        y_true, y_pred = _reg_data(n=200)
        mae_std, rmse_std, sp_std = _bootstrap_regression(y_true, y_pred, n_bootstrap=10)
        assert mae_std >= 0
        assert rmse_std >= 0
        assert sp_std >= 0

    def test_constant_true_values_have_zero_std(self):
        from src.evaluation.model_comparison import _bootstrap_regression

        # When y_true is constant, MAE vs any constant prediction won't vary across resamples
        y_true = np.full(100, 0.3)  # constant true values
        y_pred = np.full(100, 0.5)  # constant predictions
        mae_std, _, _ = _bootstrap_regression(y_true, y_pred, n_bootstrap=5)
        assert mae_std == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Leaderboard rank correlation
# ═══════════════════════════════════════════════════════════════════════════════


class TestLeaderboardRankCorrelation:
    def _make_scored_df(self, n: int = 300, n_matches: int = 20, seed: int = 0) -> pd.DataFrame:
        rng = _rng(seed)
        player_ids = [f"p{i % 15}" for i in range(n)]
        # Give consistent ranking: players with higher id have systematically higher cxt
        cxt = np.array([int(p[1:]) * 0.01 + rng.normal(0, 0.03) for p in player_ids])
        return pd.DataFrame(
            {
                "match_id": [f"m{i % n_matches}" for i in range(n)],
                "player_id": player_ids,
                "cxt": cxt,
            }
        )

    def test_returns_float_for_valid_data(self):
        df = self._make_scored_df()
        corr = leaderboard_rank_correlation(df, min_actions=1)
        assert isinstance(corr, float)

    def test_corr_in_valid_range(self):
        df = self._make_scored_df()
        corr = leaderboard_rank_correlation(df, min_actions=1)
        assert -1.0 <= corr <= 1.0

    def test_stable_player_ordering_gives_high_corr(self):
        """Players with consistently higher values should yield high rank correlation."""
        df = self._make_scored_df(n=600, n_matches=40, seed=0)
        corr = leaderboard_rank_correlation(df, min_actions=1)
        assert corr > 0.5

    def test_returns_none_if_missing_columns(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        corr = leaderboard_rank_correlation(df)
        assert corr is None

    def test_returns_none_if_too_few_common_players(self):
        df = pd.DataFrame(
            {
                "match_id": ["m1", "m2"],
                "player_id": ["p1", "p2"],
                "cxt": [0.1, 0.2],
            }
        )
        corr = leaderboard_rank_correlation(df, min_actions=10)
        assert corr is None


# ═══════════════════════════════════════════════════════════════════════════════
# Promotion evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def _clf_result(name, family, log_loss=0.5, brier=0.15, ece=0.05):
    m = ClassificationMetrics(log_loss=log_loss, brier=brier, roc_auc=0.75, pr_auc=0.5, ece=ece)
    return ModelComparisonResult(
        name=name,
        family=family,
        metric_type="cxg",
        task_type="classification",
        feature_set="contextual",
        metrics=m,
    )


def _reg_result(name, family, mae=0.05, rmse=0.08, cal=None):
    if cal is None:
        cal = {}
    m = RegressionMetrics(mae=mae, rmse=rmse, spearman=0.6, calibration_by_bucket=cal)
    return ModelComparisonResult(
        name=name,
        family=family,
        metric_type="cxt",
        task_type="regression",
        feature_set="contextual",
        metrics=m,
    )


class TestEvaluatePromotion:
    def test_neural_promoted_when_all_criteria_met(self):
        tree = _clf_result("xgb", "tree", log_loss=0.6, ece=0.06)
        neural = _clf_result("ffnn", "neural_tabular", log_loss=0.5, ece=0.05)
        verdict = evaluate_promotion(neural, tree, rank_corr=0.90)
        assert verdict == "promote"

    def test_keep_tree_when_improvement_insufficient(self):
        tree = _clf_result("xgb", "tree", log_loss=0.6, ece=0.06)
        neural = _clf_result("ffnn", "neural_tabular", log_loss=0.59, ece=0.05)  # < 5 % improvement
        verdict = evaluate_promotion(neural, tree, rank_corr=0.90)
        assert verdict == "keep_tree"

    def test_keep_tree_when_ece_worse(self):
        tree = _clf_result("xgb", "tree", log_loss=0.6, ece=0.05)
        neural = _clf_result("ffnn", "neural_tabular", log_loss=0.5, ece=0.08)  # worse calibration
        verdict = evaluate_promotion(neural, tree, rank_corr=0.90)
        assert verdict == "keep_tree"

    def test_keep_tree_when_rank_corr_too_low(self):
        tree = _clf_result("xgb", "tree", log_loss=0.6, ece=0.06)
        neural = _clf_result("ffnn", "neural_tabular", log_loss=0.5, ece=0.05)
        verdict = evaluate_promotion(neural, tree, rank_corr=0.70)  # < 0.80
        assert verdict == "keep_tree"

    def test_keep_tree_if_360_only(self):
        tree = _clf_result("xgb", "tree", log_loss=0.6, ece=0.06)
        neural = _clf_result("gnn", "neural_360", log_loss=0.5, ece=0.05)
        neural.is_360_only = True
        verdict = evaluate_promotion(neural, tree, rank_corr=0.90)
        assert verdict == "keep_tree"

    def test_unknown_rank_corr_allows_promote(self):
        """If rank_corr is None (unknown), promotion can still occur."""
        tree = _clf_result("xgb", "tree", log_loss=0.6, ece=0.06)
        neural = _clf_result("ffnn", "neural_tabular", log_loss=0.5, ece=0.05)
        verdict = evaluate_promotion(neural, tree, rank_corr=None)
        assert verdict == "promote"

    def test_regression_promotion(self):
        tree = _reg_result("xgb_reg", "tree", mae=0.08)
        neural = _reg_result("ffnn_reg", "neural_tabular", mae=0.05)  # > 5 % improvement
        verdict = evaluate_promotion(neural, tree, rank_corr=0.85)
        assert verdict == "promote"

    def test_regression_keep_tree_when_mae_similar(self):
        tree = _reg_result("xgb_reg", "tree", mae=0.08)
        neural = _reg_result("ffnn_reg", "neural_tabular", mae=0.078)  # < 5 % improvement
        verdict = evaluate_promotion(neural, tree, rank_corr=0.85)
        assert verdict == "keep_tree"


# ═══════════════════════════════════════════════════════════════════════════════
# ModelEntry
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelEntry:
    def test_predict_dispatches_to_predict_proba_for_clf(self):
        df = _make_test_df(n=300)
        clf = _FakeClassifier(n=300)
        entry = ModelEntry("clf", "glm", "cxg", "classification", "contextual", clf)
        p = entry.predict(df)
        assert p.ndim == 1
        assert len(p) == 300
        assert (p >= 0).all() and (p <= 1).all()

    def test_predict_dispatches_to_predict_for_reg(self):
        df = _make_test_df(n=300)
        reg = _FakeRegressor(n=300)
        entry = ModelEntry("reg", "glm", "cxt", "regression", "contextual", reg)
        p = entry.predict(df)
        assert p.ndim == 1
        assert len(p) == 300

    def test_predict_uses_custom_predict_fn(self):
        df = _make_test_df(n=10)

        def custom_fn(df):
            return np.ones(len(df)) * 0.42

        entry = ModelEntry(
            "custom", "glm", "cxg", "classification", "contextual", None, predict_fn=custom_fn
        )
        p = entry.predict(df)
        np.testing.assert_array_almost_equal(p, 0.42)


# ═══════════════════════════════════════════════════════════════════════════════
# ModelComparisonSuite
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def test_df():
    return _make_test_df(n=300, n_matches=20, seed=0)


@pytest.fixture(scope="module")
def suite_with_models(test_df):
    n = len(test_df)
    suite = ModelComparisonSuite(n_bootstrap=5, random_state=42)
    # CxG classifiers
    suite.add_model(
        ModelEntry(
            "glm_cxg", "glm", "cxg", "classification", "contextual", _FakeClassifier(n, seed=0)
        )
    )
    suite.add_model(
        ModelEntry(
            "tree_cxg", "tree", "cxg", "classification", "contextual", _FakeClassifier(n, seed=1)
        )
    )
    suite.add_model(
        ModelEntry(
            "ffnn_cxg",
            "neural_tabular",
            "cxg",
            "classification",
            "contextual",
            _FakeClassifier(n, seed=2),
        )
    )
    # CxT regressors
    suite.add_model(
        ModelEntry("gamma_cxt", "glm", "cxt", "regression", "contextual", _FakeRegressor(n, seed=3))
    )
    suite.add_model(
        ModelEntry("tree_cxt", "tree", "cxt", "regression", "contextual", _FakeRegressor(n, seed=4))
    )
    return suite


class TestModelComparisonSuite:
    def test_run_returns_comparison_report(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        assert isinstance(report, ComparisonReport)

    def test_report_results_length(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        assert len(report.results) == 5

    def test_promotion_summary_has_metric_types(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        assert "cxg" in report.promotion_summary or "cxt" in report.promotion_summary

    def test_best_tree_populated(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        assert "cxg" in report.best_tree
        assert "cxt" in report.best_tree

    def test_neural_models_have_verdict(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        neural = [r for r in report.results if r.family == "neural_tabular"]
        assert len(neural) > 0
        for r in neural:
            assert r.promotion_verdict in {"promote", "keep_tree", "insufficient_data"}

    def test_non_neural_models_not_applicable(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        non_neural = [
            r
            for r in report.results
            if r.family not in ("neural_tabular", "neural_seq", "neural_360")
        ]
        for r in non_neural:
            assert r.promotion_verdict == "not_applicable"

    def test_results_have_bootstrap_stds(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        for r in report.results:
            if r.task_type == "classification":
                assert r.metrics.log_loss_std >= 0
                assert r.metrics.brier_std >= 0
            else:
                assert r.metrics.mae_std >= 0

    def test_raises_when_no_models_registered(self, test_df):
        suite = ModelComparisonSuite()
        with pytest.raises(ValueError, match="No models registered"):
            suite.run(test_df)

    def test_skips_missing_target_column(self, test_df):
        n = len(test_df)
        suite = ModelComparisonSuite(n_bootstrap=3)
        suite.add_model(
            ModelEntry("m1", "glm", "cxg", "classification", "contextual", _FakeClassifier(n))
        )
        report = suite.run(test_df, target_map={"cxg": "nonexistent_column"})
        assert len(report.results) == 0

    def test_add_models_bulk(self, test_df):
        n = len(test_df)
        suite = ModelComparisonSuite(n_bootstrap=3)
        entries = [
            ModelEntry("a", "glm", "cxg", "classification", "contextual", _FakeClassifier(n)),
            ModelEntry("b", "tree", "cxg", "classification", "contextual", _FakeClassifier(n)),
        ]
        suite.add_models(entries)
        report = suite.run(test_df)
        assert len(report.results) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# ComparisonReport.to_dataframe
# ═══════════════════════════════════════════════════════════════════════════════


class TestComparisonReportToDataframe:
    def test_returns_dataframe(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        df = report.to_dataframe()
        for col in ("name", "family", "metric_type", "task_type", "promotion_verdict"):
            assert col in df.columns

    def test_row_count_matches_results(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        df = report.to_dataframe()
        assert len(df) == len(report.results)

    def test_classification_rows_have_log_loss(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        df = report.to_dataframe()
        clf_rows = df[df["task_type"] == "classification"]
        assert "log_loss" in df.columns
        assert clf_rows["log_loss"].notna().all()

    def test_regression_rows_have_mae(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        df = report.to_dataframe()
        reg_rows = df[df["task_type"] == "regression"]
        assert "mae" in df.columns
        assert reg_rows["mae"].notna().all()


# ═══════════════════════════════════════════════════════════════════════════════
# HTML report builder
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildHtmlReport:
    def test_returns_string(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        html = build_html_report(report)
        assert isinstance(html, str)

    def test_contains_html_doctype(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        html = build_html_report(report)
        assert "<!DOCTYPE html>" in html

    def test_contains_metric_type_sections(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        html = build_html_report(report)
        assert "CXG" in html
        assert "CXT" in html

    def test_contains_model_names(self, suite_with_models, test_df):
        report = suite_with_models.run(test_df)
        html = build_html_report(report)
        assert "glm_cxg" in html
        assert "tree_cxg" in html

    def test_writes_file(self, suite_with_models, test_df, tmp_path):
        report = suite_with_models.run(test_df)
        output = tmp_path / "comparison.html"
        build_html_report(report, output_path=str(output))
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_empty_report_does_not_crash(self):
        report = ComparisonReport(results=[], promotion_summary={}, best_tree={})
        html = build_html_report(report)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclass construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataclasses:
    def test_classification_metrics_construction(self):
        m = ClassificationMetrics(log_loss=0.5, brier=0.2, roc_auc=0.7, pr_auc=0.4, ece=0.05)
        assert m.log_loss == 0.5
        assert m.brier == 0.2
        assert isinstance(m.reliability_bins, list)

    def test_regression_metrics_construction(self):
        m = RegressionMetrics(mae=0.05, rmse=0.08, spearman=0.6)
        assert m.mae == 0.05
        assert isinstance(m.calibration_by_bucket, dict)

    def test_model_comparison_result_construction(self):
        m = ClassificationMetrics(log_loss=0.5, brier=0.2, roc_auc=0.7, pr_auc=0.4, ece=0.05)
        r = ModelComparisonResult(
            name="test",
            family="glm",
            metric_type="cxg",
            task_type="classification",
            feature_set="contextual",
            metrics=m,
        )
        assert r.name == "test"
        assert r.promotion_verdict == "not_evaluated"
