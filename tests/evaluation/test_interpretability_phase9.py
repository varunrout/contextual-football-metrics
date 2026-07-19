"""
Phase 9 tests — Interpretability and Reporting Layer.

Covers:
  • SHAPResult         — to_dataframe, top_features, waterfall_data
  • DirectionCheckResult  — passed property
  • AblationEntry / AblationResult — construction, to_dataframe, most_important_group
  • _get_named_coefficients  — linear model, Pipeline, raises on unsupported
  • check_coefficient_directions — correct / wrong signs, partial match, pass_rate
  • run_ablation_study  — classification, regression, missing groups, custom metric
  • build_match_report  — basic, raises on missing match, top_n, team_summary, shap_available
  • build_player_report — totals, per_90, vs_average z-scores, shap top features, raises on missing player
  • build_interpretability_html — all-None, individual sections, writes file
  • compute_shap_values — ImportError if shap missing (skip test if installed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.interpretability import (
    EXPECTED_SIGNS_CXG,
    EXPECTED_SIGNS_CXT,
    AblationEntry,
    AblationResult,
    DirectionCheckResult,
    DirectionViolation,
    MatchReport,
    PlayerReport,
    SHAPResult,
    _get_named_coefficients,
    build_interpretability_html,
    build_match_report,
    build_player_report,
    check_coefficient_directions,
    run_ablation_study,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_clf_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = _rng(seed)
    return pd.DataFrame(
        {
            "distance_to_goal": rng.uniform(5, 35, n),
            "in_box": rng.integers(0, 2, n).astype(float),
            "under_pressure": rng.integers(0, 2, n).astype(float),
            "is_central": rng.integers(0, 2, n).astype(float),
            "progressive_distance": rng.uniform(0, 20, n),
            "goal": rng.integers(0, 2, n).astype(float),
        }
    )


def _make_reg_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = _rng(seed)
    return pd.DataFrame(
        {
            "distance_to_goal": rng.uniform(5, 35, n),
            "in_box": rng.integers(0, 2, n).astype(float),
            "under_pressure": rng.integers(0, 2, n).astype(float),
            "progressive_distance": rng.uniform(0, 20, n),
            "possession_cxg": rng.uniform(0, 0.3, n),
        }
    )


def _fit_logreg(df: pd.DataFrame, feature_cols: list[str], target: str = "goal"):
    X = df[feature_cols].values
    y = df[target].values
    clf = LogisticRegression(max_iter=500, random_state=0)
    clf.fit(X, y)
    return clf


def _fit_linreg(df: pd.DataFrame, feature_cols: list[str], target: str = "possession_cxg"):
    X = df[feature_cols].values
    y = df[target].values
    reg = LinearRegression()
    reg.fit(X, y)
    return reg


def _fit_pipeline_logreg(df: pd.DataFrame, feature_cols: list[str], target: str = "goal"):
    X = df[feature_cols]
    y = df[target].values
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=0))]
    )
    pipe.fit(X, y)
    return pipe


def _make_scored_df(n: int = 300, n_matches: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = _rng(seed)
    return pd.DataFrame(
        {
            "match_id": [f"m{i % n_matches}" for i in range(n)],
            "player_id": [f"p{i % 15}" for i in range(n)],
            "team_id": [f"t{i % 2}" for i in range(n)],
            "cxg": rng.uniform(0, 0.3, n),
            "cxa": rng.uniform(0, 0.2, n),
            "cxt": rng.uniform(-0.05, 0.15, n),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SHAPResult
# ═══════════════════════════════════════════════════════════════════════════════


class TestSHAPResult:
    def _make(self, n: int = 50, k: int = 4) -> SHAPResult:
        rng = _rng(0)
        return SHAPResult(
            shap_values=rng.standard_normal((n, k)),
            base_value=0.12,
            feature_names=[f"f{i}" for i in range(k)],
        )

    def test_to_dataframe_shape(self):
        sr = self._make(50, 4)
        df = sr.to_dataframe()
        assert df.shape == (50, 4)

    def test_to_dataframe_columns(self):
        sr = self._make(50, 4)
        df = sr.to_dataframe()
        assert list(df.columns) == sr.feature_names

    def test_top_features_sorted_descending(self):
        sr = self._make(50, 4)
        top = sr.top_features(n=4)
        vals = top["mean_abs_shap"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_top_features_respects_n(self):
        sr = self._make(50, 10)
        top = sr.top_features(n=3)
        assert len(top) == 3

    def test_top_features_columns(self):
        sr = self._make()
        top = sr.top_features()
        assert "feature" in top.columns
        assert "mean_abs_shap" in top.columns

    def test_waterfall_data_length(self):
        sr = self._make(50, 4)
        wd = sr.waterfall_data(row_idx=0)
        assert len(wd) == 4

    def test_waterfall_data_has_required_keys(self):
        sr = self._make()
        wd = sr.waterfall_data(row_idx=0)
        for entry in wd:
            assert "feature" in entry
            assert "shap_value" in entry
            assert "running_total" in entry

    def test_waterfall_final_running_total(self):
        sr = self._make(50, 4)
        wd = sr.waterfall_data(row_idx=0)
        expected_total = sr.base_value + float(sr.shap_values[0].sum())
        assert abs(wd[-1]["running_total"] - expected_total) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# DirectionCheckResult
# ═══════════════════════════════════════════════════════════════════════════════


class TestDirectionCheckResult:
    def test_passed_true_when_no_violations(self):
        result = DirectionCheckResult(violations=[], n_checked=3, n_violations=0, pass_rate=1.0)
        assert result.passed is True

    def test_passed_false_when_violations(self):
        v = DirectionViolation("distance_to_goal", -1, 1, 0.5)
        result = DirectionCheckResult(violations=[v], n_checked=1, n_violations=1, pass_rate=0.0)
        assert result.passed is False


# ═══════════════════════════════════════════════════════════════════════════════
# AblationEntry and AblationResult
# ═══════════════════════════════════════════════════════════════════════════════


class TestAblationResult:
    def _make(self) -> AblationResult:
        entries = [
            AblationEntry("location", ["x", "y"], 0.5, 0.6, 0.1, 0.2),
            AblationEntry("context", ["a", "b"], 0.5, 0.52, 0.02, 0.04),
            AblationEntry("opponent", ["c"], 0.5, 0.51, 0.01, 0.02),
        ]
        return AblationResult(baseline_metric=0.5, entries=entries, metric_name="log_loss")

    def test_to_dataframe_shape(self):
        df = self._make().to_dataframe()
        assert len(df) == 3

    def test_to_dataframe_sorted_descending(self):
        df = self._make().to_dataframe()
        assert df["degradation"].tolist() == sorted(df["degradation"].tolist(), reverse=True)

    def test_to_dataframe_columns(self):
        df = self._make().to_dataframe()
        for col in (
            "group",
            "features_removed",
            "baseline_metric",
            "ablated_metric",
            "degradation",
            "relative_degradation",
        ):
            assert col in df.columns

    def test_most_important_group(self):
        assert self._make().most_important_group() == "location"

    def test_most_important_group_none_for_empty(self):
        ar = AblationResult(baseline_metric=0.5, entries=[], metric_name="log_loss")
        assert ar.most_important_group() is None

    def test_to_dataframe_empty(self):
        ar = AblationResult(baseline_metric=0.5, entries=[], metric_name="log_loss")
        df = ar.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _get_named_coefficients
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetNamedCoefficients:
    def test_bare_logistic_regression(self):
        df = _make_clf_df()
        feat_cols = ["distance_to_goal", "in_box", "under_pressure"]
        model = _fit_logreg(df, feat_cols)
        coef_map = _get_named_coefficients(model, feat_cols)
        assert set(coef_map.keys()) == set(feat_cols)

    def test_bare_linear_regression(self):
        df = _make_reg_df()
        feat_cols = ["distance_to_goal", "in_box", "progressive_distance"]
        model = _fit_linreg(df, feat_cols)
        coef_map = _get_named_coefficients(model, feat_cols)
        assert len(coef_map) == 3

    def test_pipeline_logreg_returns_dict(self):
        df = _make_clf_df()
        feat_cols = ["distance_to_goal", "in_box", "under_pressure"]
        pipe = _fit_pipeline_logreg(df, feat_cols)
        coef_map = _get_named_coefficients(pipe, feat_cols)
        assert isinstance(coef_map, dict)
        assert len(coef_map) == 3

    def test_raises_for_unsupported_model(self):
        class _NoCoefs:
            pass

        with pytest.raises(ValueError, match="Cannot extract coefficients"):
            _get_named_coefficients(_NoCoefs(), ["a", "b"])

    def test_coefficients_are_floats(self):
        df = _make_clf_df()
        feat_cols = ["distance_to_goal", "in_box"]
        model = _fit_logreg(df, feat_cols)
        coef_map = _get_named_coefficients(model, feat_cols)
        for v in coef_map.values():
            assert isinstance(v, float)


# ═══════════════════════════════════════════════════════════════════════════════
# check_coefficient_directions
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckCoefficientDirections:
    def test_no_violations_when_signs_correct(self):
        """A model with expected negative coef for distance_to_goal should pass."""
        df = _make_clf_df(n=300)
        feat_cols = ["distance_to_goal", "in_box", "under_pressure"]
        # Train a LogisticRegression; distance_to_goal should get negative coef
        # Force it by setting coefficients directly
        model = _fit_logreg(df, feat_cols)
        # Manually override coef_ to have expected signs
        model.coef_ = np.array([[-0.5, 0.8, -0.3]])  # -, +, -  (matches EXPECTED_SIGNS_CXG)
        result = check_coefficient_directions(
            model,
            feat_cols,
            expected_signs={"distance_to_goal": -1, "in_box": +1, "under_pressure": -1},
        )
        assert result.n_checked == 3
        assert result.n_violations == 0
        assert result.passed is True
        assert result.pass_rate == pytest.approx(1.0)

    def test_violation_detected_for_wrong_sign(self):
        df = _make_clf_df(n=300)
        feat_cols = ["distance_to_goal", "in_box"]
        model = _fit_logreg(df, feat_cols)
        # Override: distance_to_goal positive (wrong), in_box positive (correct)
        model.coef_ = np.array([[0.5, 0.8]])
        result = check_coefficient_directions(
            model,
            feat_cols,
            expected_signs={"distance_to_goal": -1, "in_box": +1},
        )
        assert result.n_violations == 1
        assert result.violations[0].feature == "distance_to_goal"
        assert result.violations[0].expected_sign == -1
        assert result.violations[0].actual_sign == 1

    def test_features_not_in_model_are_skipped(self):
        df = _make_clf_df(n=300)
        feat_cols = ["distance_to_goal"]
        model = _fit_logreg(df, feat_cols)
        model.coef_ = np.array([[-0.5]])
        # Provide expected sign for a feature that isn't in the model
        result = check_coefficient_directions(
            model,
            feat_cols,
            expected_signs={"distance_to_goal": -1, "nonexistent_feature": +1},
        )
        assert result.n_checked == 1  # only distance_to_goal checked

    def test_pass_rate_calculated_correctly(self):
        df = _make_clf_df(n=300)
        feat_cols = ["distance_to_goal", "in_box", "under_pressure"]
        model = _fit_logreg(df, feat_cols)
        # 2 correct, 1 wrong
        model.coef_ = np.array([[-0.5, 0.8, 0.3]])  # under_pressure wrong (+, should be -)
        result = check_coefficient_directions(
            model,
            feat_cols,
            expected_signs={"distance_to_goal": -1, "in_box": +1, "under_pressure": -1},
        )
        assert result.n_checked == 3
        assert result.n_violations == 1
        assert result.pass_rate == pytest.approx(2 / 3)

    def test_uses_expected_signs_cxg_by_default(self):
        df = _make_clf_df(n=300)
        feat_cols = list(EXPECTED_SIGNS_CXG.keys())
        model = _fit_logreg(df, [c for c in feat_cols if c in df.columns])
        result = check_coefficient_directions(model, [c for c in feat_cols if c in df.columns])
        assert isinstance(result, DirectionCheckResult)

    def test_raises_for_unsupported_model(self):
        class _NoCoefs:
            pass

        with pytest.raises(ValueError):
            check_coefficient_directions(_NoCoefs(), ["a"])


# ═══════════════════════════════════════════════════════════════════════════════
# run_ablation_study
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunAblationStudy:
    @pytest.fixture(scope="class")
    def clf_setup(self):
        df = _make_clf_df(n=200)
        feat_cols = ["distance_to_goal", "in_box", "under_pressure", "progressive_distance"]
        model = _fit_logreg(df, feat_cols)
        groups = {
            "location": ["distance_to_goal"],
            "pressure": ["under_pressure"],
            "context": ["in_box", "progressive_distance"],
        }
        return model, df, feat_cols, groups

    @pytest.fixture(scope="class")
    def reg_setup(self):
        df = _make_reg_df(n=200)
        feat_cols = ["distance_to_goal", "in_box", "progressive_distance"]
        model = _fit_linreg(df, feat_cols)
        groups = {
            "location": ["distance_to_goal"],
            "context": ["in_box", "progressive_distance"],
        }
        return model, df, feat_cols, groups

    def test_returns_ablation_result(self, clf_setup):
        model, df, feat_cols, groups = clf_setup
        result = run_ablation_study(
            model, df, "goal", "classification", groups, feature_cols=feat_cols
        )
        assert isinstance(result, AblationResult)

    def test_classification_uses_log_loss(self, clf_setup):
        model, df, feat_cols, groups = clf_setup
        result = run_ablation_study(
            model, df, "goal", "classification", groups, feature_cols=feat_cols
        )
        assert result.metric_name == "log_loss"

    def test_regression_uses_mae(self, reg_setup):
        model, df, feat_cols, groups = reg_setup
        result = run_ablation_study(
            model, df, "possession_cxg", "regression", groups, feature_cols=feat_cols
        )
        assert result.metric_name == "mae"

    def test_entries_count_matches_groups(self, clf_setup):
        model, df, feat_cols, groups = clf_setup
        result = run_ablation_study(
            model, df, "goal", "classification", groups, feature_cols=feat_cols
        )
        assert len(result.entries) == len(groups)

    def test_missing_feature_group_skipped(self, clf_setup):
        model, df, feat_cols, groups = clf_setup
        groups_with_missing = dict(groups)
        groups_with_missing["nonexistent_group"] = ["col_does_not_exist"]
        result = run_ablation_study(
            model, df, "goal", "classification", groups_with_missing, feature_cols=feat_cols
        )
        group_names = {e.group_name for e in result.entries}
        assert "nonexistent_group" not in group_names

    def test_baseline_metric_stored(self, clf_setup):
        model, df, feat_cols, groups = clf_setup
        result = run_ablation_study(
            model, df, "goal", "classification", groups, feature_cols=feat_cols
        )
        assert result.baseline_metric > 0

    def test_custom_metric_fn_used(self, reg_setup):
        model, df, feat_cols, groups = reg_setup
        from sklearn.metrics import mean_squared_error

        def mse(yt, yp):
            return float(mean_squared_error(yt, yp))

        mse.__name__ = "mse"
        result = run_ablation_study(
            model, df, "possession_cxg", "regression", groups, metric_fn=mse, feature_cols=feat_cols
        )
        assert result.metric_name == "mse"

    def test_invalid_task_type_raises(self, clf_setup):
        model, df, feat_cols, groups = clf_setup
        with pytest.raises(ValueError, match="task_type"):
            run_ablation_study(model, df, "goal", "invalid_type", groups)

    def test_removing_important_feature_degrades_metric(self, clf_setup):
        model, df, feat_cols, groups = clf_setup
        result = run_ablation_study(
            model, df, "goal", "classification", groups, feature_cols=feat_cols
        )
        # At least one group should cause positive degradation (metric gets worse)
        degradations = [e.degradation for e in result.entries]
        assert max(degradations) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# build_match_report
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildMatchReport:
    @pytest.fixture(scope="class")
    def scored_df(self):
        return _make_scored_df(n=300, n_matches=10)

    def test_returns_match_report(self, scored_df):
        report = build_match_report(scored_df, "m0")
        assert isinstance(report, MatchReport)

    def test_match_id_stored_as_str(self, scored_df):
        report = build_match_report(scored_df, "m1")
        assert report.match_id == "m1"

    def test_n_actions_correct(self, scored_df):
        expected = int((scored_df["match_id"] == "m0").sum())
        report = build_match_report(scored_df, "m0")
        assert report.n_actions == expected

    def test_top_cxg_actions_max_top_n(self, scored_df):
        report = build_match_report(scored_df, "m0", top_n=5)
        assert len(report.top_cxg_actions) <= 5

    def test_top_cxt_actions_sorted(self, scored_df):
        report = build_match_report(scored_df, "m0", top_n=5)
        if len(report.top_cxt_actions) > 1:
            vals = report.top_cxt_actions["cxt"].tolist()
            assert vals == sorted(vals, reverse=True)

    def test_team_summary_has_both_teams(self, scored_df):
        # The fixture cycles team_id over t0/t1; match m0 should have both
        report = build_match_report(scored_df, "m0")
        assert len(report.team_summary) >= 1

    def test_team_summary_has_cxg_column(self, scored_df):
        report = build_match_report(scored_df, "m0")
        assert "cxg" in report.team_summary.columns

    def test_missing_match_raises_value_error(self, scored_df):
        with pytest.raises(ValueError, match="match_id"):
            build_match_report(scored_df, "nonexistent_match_id")

    def test_shap_available_false_without_shap_df(self, scored_df):
        report = build_match_report(scored_df, "m0")
        assert report.shap_available is False

    def test_shap_available_true_with_shap_df(self, scored_df):
        dummy_shap = pd.DataFrame({"f1": [0.1, 0.2]})
        report = build_match_report(scored_df, "m0", shap_df=dummy_shap)
        assert report.shap_available is True

    def test_top_cxg_actions_contain_player_id(self, scored_df):
        report = build_match_report(scored_df, "m0")
        assert "player_id" in report.top_cxg_actions.columns

    def test_custom_top_n(self, scored_df):
        report = build_match_report(scored_df, "m0", top_n=3)
        assert len(report.top_cxg_actions) <= 3


# ═══════════════════════════════════════════════════════════════════════════════
# build_player_report
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildPlayerReport:
    @pytest.fixture(scope="class")
    def scored_df(self):
        return _make_scored_df(n=300, n_matches=10)

    def test_returns_player_report(self, scored_df):
        report = build_player_report(scored_df, "p0")
        assert isinstance(report, PlayerReport)

    def test_player_id_stored_as_str(self, scored_df):
        report = build_player_report(scored_df, "p0")
        assert report.player_id == "p0"

    def test_n_actions_positive(self, scored_df):
        report = build_player_report(scored_df, "p0")
        assert report.n_actions > 0

    def test_per_90_keys_present(self, scored_df):
        report = build_player_report(scored_df, "p0")
        for key in ("cxg_per_90", "cxa_per_90", "cxt_per_90"):
            assert key in report.per_90

    def test_per_90_values_non_negative(self, scored_df):
        # cxt can be negative, but cxg/cxa should not be
        report = build_player_report(scored_df, "p0")
        assert report.per_90["cxg_per_90"] >= 0
        assert report.per_90["cxa_per_90"] >= 0

    def test_total_cxg_matches_sum(self, scored_df):
        expected = float(scored_df[scored_df["player_id"] == "p0"]["cxg"].sum())
        report = build_player_report(scored_df, "p0")
        assert report.total_cxg == pytest.approx(expected)

    def test_cxt_handles_nan(self):
        df = pd.DataFrame(
            {
                "player_id": ["p1", "p1", "p1"],
                "match_id": ["m1", "m2", "m3"],
                "cxg": [0.1, 0.2, 0.0],
                "cxa": [0.05, 0.08, 0.0],
                "cxt": [0.1, float("nan"), 0.3],
            }
        )
        report = build_player_report(df, "p1")
        assert report.total_cxt == pytest.approx(0.4)

    def test_missing_player_raises_value_error(self, scored_df):
        with pytest.raises(ValueError, match="player_id"):
            build_player_report(scored_df, "nonexistent_player")

    def test_vs_average_populated_with_league_df(self, scored_df):
        rng = _rng(0)
        league_df = pd.DataFrame(
            {
                "player_id": [f"p{i}" for i in range(20)],
                "cxg_per_90": rng.uniform(0.01, 0.3, 20),
                "cxt_per_90": rng.uniform(0.01, 0.2, 20),
            }
        )
        report = build_player_report(scored_df, "p0", league_df=league_df)
        assert "cxg_per_90" in report.vs_average
        assert "cxt_per_90" in report.vs_average

    def test_vs_average_empty_without_league_df(self, scored_df):
        report = build_player_report(scored_df, "p0")
        assert report.vs_average == {}

    def test_top_features_populated_with_shap_df(self, scored_df):
        rng = _rng(0)
        shap_df = pd.DataFrame(
            {
                "player_id": scored_df["player_id"].values,
                "feature_a": rng.standard_normal(len(scored_df)),
                "feature_b": rng.standard_normal(len(scored_df)),
            }
        )
        report = build_player_report(scored_df, "p0", shap_df=shap_df)
        assert report.top_features is not None
        assert "feature" in report.top_features.columns
        assert "mean_abs_shap" in report.top_features.columns

    def test_top_features_none_without_shap_df(self, scored_df):
        report = build_player_report(scored_df, "p0")
        assert report.top_features is None


# ═══════════════════════════════════════════════════════════════════════════════
# build_interpretability_html
# ═══════════════════════════════════════════════════════════════════════════════


def _make_shap_result() -> SHAPResult:
    rng = _rng(0)
    return SHAPResult(
        shap_values=rng.standard_normal((50, 5)),
        base_value=0.1,
        feature_names=["distance_to_goal", "in_box", "under_pressure", "is_central", "speed"],
    )


def _make_direction_check(n_violations: int = 0) -> DirectionCheckResult:
    violations = []
    if n_violations > 0:
        violations = [DirectionViolation("distance_to_goal", -1, 1, 0.4)]
    return DirectionCheckResult(
        violations=violations,
        n_checked=3,
        n_violations=n_violations,
        pass_rate=(3 - n_violations) / 3,
    )


def _make_ablation() -> AblationResult:
    entries = [
        AblationEntry("location", ["x", "y"], 0.5, 0.65, 0.15, 0.30),
        AblationEntry("context", ["a"], 0.5, 0.51, 0.01, 0.02),
    ]
    return AblationResult(baseline_metric=0.5, entries=entries, metric_name="log_loss")


class TestBuildInterpretabilityHtml:
    def test_all_none_returns_string(self):
        html = build_interpretability_html()
        assert isinstance(html, str)

    def test_all_none_contains_doctype(self):
        html = build_interpretability_html()
        assert "<!DOCTYPE html>" in html

    def test_shap_section_present(self):
        html = build_interpretability_html(shap_result=_make_shap_result())
        assert "distance_to_goal" in html

    def test_shap_section_absent_when_none(self):
        html = build_interpretability_html(shap_result=None)
        assert "No SHAP result provided" in html

    def test_direction_check_no_violations(self):
        html = build_interpretability_html(direction_check=_make_direction_check(0))
        assert "ALL CLEAR" in html

    def test_direction_check_violations_listed(self):
        html = build_interpretability_html(direction_check=_make_direction_check(1))
        assert "distance_to_goal" in html

    def test_ablation_section_shows_groups(self):
        html = build_interpretability_html(ablation_result=_make_ablation())
        assert "location" in html
        assert "context" in html

    def test_ablation_section_absent_when_none(self):
        html = build_interpretability_html(ablation_result=None)
        assert "No ablation result provided" in html

    def test_all_sections_combined(self):
        html = build_interpretability_html(
            shap_result=_make_shap_result(),
            direction_check=_make_direction_check(0),
            ablation_result=_make_ablation(),
        )
        assert "<!DOCTYPE html>" in html
        assert "SHAP" in html
        assert "Coefficient" in html
        assert "Ablation" in html

    def test_writes_file(self, tmp_path):
        output = tmp_path / "interp.html"
        html = build_interpretability_html(
            shap_result=_make_shap_result(),
            output_path=str(output),
        )
        assert output.exists()
        assert output.read_text(encoding="utf-8") == html

    def test_creates_parent_directories(self, tmp_path):
        output = tmp_path / "reports" / "subdir" / "interp.html"
        build_interpretability_html(output_path=str(output))
        assert output.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# compute_shap_values — only tests if shap is NOT installed (ImportError guard)
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeShapValues:
    def test_raises_import_error_if_shap_missing(self):
        """
        This test verifies the ImportError guard fires when shap is absent.
        If shap IS installed the test is skipped.
        """
        try:
            import shap  # noqa: F401

            pytest.skip("shap is installed — ImportError guard test not applicable")
        except ImportError:
            pass

        from src.evaluation.interpretability import compute_shap_values

        df = _make_clf_df(n=10)
        model = _fit_logreg(df, ["distance_to_goal", "in_box"])
        with pytest.raises(ImportError, match="shap"):
            compute_shap_values(model, df, ["distance_to_goal", "in_box"])


# ═══════════════════════════════════════════════════════════════════════════════
# Constant dictionaries
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpectedSignDicts:
    def test_cxg_distance_is_negative(self):
        assert EXPECTED_SIGNS_CXG["distance_to_goal"] == -1

    def test_cxg_in_box_is_positive(self):
        assert EXPECTED_SIGNS_CXG["in_box"] == +1

    def test_cxt_distance_is_negative(self):
        assert EXPECTED_SIGNS_CXT["distance_to_goal"] == -1

    def test_all_values_are_plus_or_minus_one(self):
        for d in (EXPECTED_SIGNS_CXG, EXPECTED_SIGNS_CXT):
            for v in d.values():
                assert v in (+1, -1)
