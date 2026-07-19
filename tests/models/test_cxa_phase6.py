"""
Phase 6 tests — CxA Two-Stage Contextual Models.

Covers:
  • CxAFeatureSetSpec registry and validation
  • GlmShotCreationModel
  • XGBoostShotCreationModel
  • LightGBMShotCreationModel
  • WindowStabilityReport
  • ShotCreationLadder
  • GammaShotQualityModel
  • XGBoostShotQualityModel
  • LightGBMShotQualityModel
  • ShotQualityLadder
  • CxAPipeline
  • build_player_leaderboard / build_team_leaderboard
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Feature-store fixture ──────────────────────────────────────────────────────


def _make_actions_df(n: int = 200, n_matches: int = 8, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic actions DataFrame with all CONTEXTUAL feature columns, plus
    the window labels required for ShotCreationModel testing.
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(20, 105, n).astype(float)
    y = rng.uniform(0, 68, n).astype(float)
    prog_dist = x - rng.uniform(0, 20, n)

    # Binary creation label: logistic driven by distance
    dist_to_goal = np.sqrt((105 - x) ** 2 + (34 - y) ** 2)
    logit = 1.5 - 0.05 * dist_to_goal + 0.04 * prog_dist + rng.normal(0, 0.3, n)
    p_shot = 1 / (1 + np.exp(-logit))
    shot_created = (rng.uniform(0, 1, n) < p_shot).astype(int)

    # Window variants
    shot_w5 = np.clip(shot_created + rng.integers(0, 2, n), 0, 1)
    shot_w10s = np.clip(shot_created + rng.integers(0, 2, n), 0, 1)
    shot_w15s = np.clip(shot_created + rng.integers(0, 2, n), 0, 1)

    # resulting_shot_cxg: positive float only where shot_created=1
    cxg_values = np.where(shot_created == 1, rng.uniform(0.02, 0.35, n), 0.0)

    n_players = max(1, n_matches * 2)
    match_ids = [f"m{i % n_matches}" for i in range(n)]
    player_ids = [f"p{i % n_players}" for i in range(n)]
    team_ids = [f"t{i % (n_matches // 2)}" for i in range(n)]
    action_types = rng.choice(["pass", "carry", "cross", "cutback"], n)
    cat_values = rng.choice(["none", "free_kick", "corner"], n)
    score_states = rng.choice(["winning", "drawing", "losing"], n)
    home_or_away = rng.choice(["home", "away"], n)
    seq_types = rng.choice(["open_play", "transition", "set_piece"], n)
    poss_zones = rng.choice(["defensive", "middle", "attacking"], n)
    t_or_s = rng.choice(["transition", "settled"], n)
    phase = rng.choice(["build_up", "progression", "final_third"], n)
    pass_height = rng.choice(["ground", "low", "high"], n)
    pass_body = rng.choice(["foot", "head", "other"], n)

    df = pd.DataFrame(
        {
            # IDs
            "event_id": [f"e{i}" for i in range(n)],
            "match_id": match_ids,
            "player_id": player_ids,
            "team_id": team_ids,
            "possession_id": [f"pos{i}" for i in range(n)],
            # Spatial
            "x_location": x,
            "y_location": y,
            "pass_length": rng.uniform(5, 40, n),
            "pass_angle": rng.uniform(-3.14, 3.14, n),
            "progressive_distance": prog_dist,
            "end_x": np.clip(x + rng.uniform(-20, 20, n), 0, 105),
            "end_y": np.clip(y + rng.uniform(-10, 10, n), 0, 68),
            "distance_to_goal": dist_to_goal,
            "end_distance_to_goal": rng.uniform(5, 80, n),
            "distance_gained": rng.uniform(-5, 20, n),
            # Boolean flags
            "cross": rng.integers(0, 2, n),
            "cutback": rng.integers(0, 2, n),
            "through_ball": rng.integers(0, 2, n),
            "switch": rng.integers(0, 2, n),
            "central_progression": rng.integers(0, 2, n),
            "box_entry": rng.integers(0, 2, n),
            "under_pressure": rng.integers(0, 2, n),
            # Opponent quality context
            "opponent_xg_conceded_rolling_5": rng.uniform(0.5, 2.0, n),
            "opponent_shots_conceded_rolling_5": rng.uniform(5, 20, n),
            "opponent_defensive_rating": rng.uniform(0.4, 1.0, n),
            "opponent_team_strength": rng.uniform(0.4, 1.0, n),
            # Match context
            "minute": rng.uniform(1, 90, n),
            "score_differential": rng.integers(-3, 4, n),
            # Sequence context
            "events_before_action": rng.integers(0, 15, n),
            "passes_before_action": rng.integers(0, 10, n),
            "carries_before_action": rng.integers(0, 5, n),
            "time_from_possession_start": rng.uniform(0, 30, n),
            "vertical_progression_speed": rng.uniform(-1, 5, n),
            "directness": rng.uniform(0, 1, n),
            # Receiver context
            "receiver_distance_to_goal": rng.uniform(5, 60, n),
            "receiver_x": rng.uniform(20, 105, n),
            "receiver_y": rng.uniform(0, 68, n),
            "receiver_in_box": rng.integers(0, 2, n),
            "receiver_is_central": rng.integers(0, 2, n),
            # Context flags
            "knockout_or_group": rng.integers(0, 2, n),
            "set_piece_flag": rng.integers(0, 2, n),
            "counterpress_regain_flag": rng.integers(0, 2, n),
            # Categorical
            "action_type": action_types,
            "pass_height": pass_height,
            "pass_body_part": pass_body,
            "set_piece_type": cat_values,
            "score_state": score_states,
            "home_or_away": home_or_away,
            "sequence_type": seq_types,
            "possession_start_zone": poss_zones,
            "transition_or_settled": t_or_s,
            "phase_of_play": phase,
            # Labels
            "shot_created": shot_created,
            "shot_within_5_actions": shot_w5,
            "shot_within_10s": shot_w10s,
            "shot_within_15s": shot_w15s,
            "resulting_shot_cxg": cxg_values,
        }
    )
    return df


@pytest.fixture(scope="module")
def actions_df():
    return _make_actions_df(n=200, n_matches=8, seed=0)


@pytest.fixture(scope="module")
def pos_actions_df():
    """Only rows where a shot followed (positive target required by Gamma GLM)."""
    df = _make_actions_df(n=200, n_matches=8, seed=0)
    return df[df["resulting_shot_cxg"] > 0].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Set Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCxAFeatureSets:
    def test_registry_has_three_entries(self):
        from src.models.cxa.feature_sets import _REGISTRY

        assert len(_REGISTRY) == 3

    def test_all_sets_accessible_by_name(self):
        from src.models.cxa.feature_sets import get_feature_set

        for name in ("traditional", "contextual", "full_360"):
            fs = get_feature_set(name)
            assert fs.name == name

    def test_unknown_name_raises_value_error(self):
        from src.models.cxa.feature_sets import get_feature_set

        with pytest.raises(ValueError, match="Unknown CxA feature set"):
            get_feature_set("nonexistent")

    def test_feature_sets_have_no_duplicate_features(self):
        from src.models.cxa.feature_sets import CONTEXTUAL, FULL_360, TRADITIONAL

        for fs in (TRADITIONAL, CONTEXTUAL, FULL_360):
            features = fs.all_features
            assert len(features) == len(set(features)), f"{fs.name} has duplicate features"

    def test_feature_set_ordering_is_nested(self):
        """FULL_360 should have more features than CONTEXTUAL > TRADITIONAL."""
        from src.models.cxa.feature_sets import CONTEXTUAL, FULL_360, TRADITIONAL

        assert len(FULL_360.all_features) > len(CONTEXTUAL.all_features)
        assert len(CONTEXTUAL.all_features) > len(TRADITIONAL.all_features)

    def test_360_flag(self):
        from src.models.cxa.feature_sets import CONTEXTUAL, FULL_360, TRADITIONAL

        assert not TRADITIONAL.requires_360
        assert not CONTEXTUAL.requires_360
        assert FULL_360.requires_360

    def test_numeric_all_is_numeric_plus_boolean(self):
        from src.models.cxa.feature_sets import TRADITIONAL

        expected = TRADITIONAL.numeric + TRADITIONAL.boolean
        assert TRADITIONAL.numeric_all == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Shot-Creation Model Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGlmShotCreation:
    def test_fit_predict_returns_probabilities(self, actions_df):
        from src.models.cxa.shot_creation_model import GlmShotCreationModel

        m = GlmShotCreationModel(feature_set="traditional")
        m.fit(actions_df, "shot_created")
        p = m.predict_proba(actions_df)
        assert p.shape == (len(actions_df),)
        assert (p >= 0).all() and (p <= 1).all()

    def test_evaluate_returns_metrics(self, actions_df):
        from src.models.cxa.shot_creation_model import GlmShotCreationModel, ShotCreationMetrics

        m = GlmShotCreationModel(feature_set="traditional")
        m.fit(actions_df, "shot_created")
        metrics = m.evaluate(actions_df, "shot_created")
        assert isinstance(metrics, ShotCreationMetrics)
        assert 0 <= metrics.log_loss <= 5
        assert 0 <= metrics.brier <= 1
        assert 0 <= metrics.auc <= 1

    def test_contextual_feature_set_fits(self, actions_df):
        from src.models.cxa.shot_creation_model import GlmShotCreationModel

        m = GlmShotCreationModel(feature_set="contextual")
        m.fit(actions_df, "shot_created")
        p = m.predict_proba(actions_df)
        assert p.shape == (len(actions_df),)

    def test_missing_target_raises(self, actions_df):
        from src.models.cxa.shot_creation_model import GlmShotCreationModel

        m = GlmShotCreationModel()
        with pytest.raises((ValueError, KeyError)):
            m.fit(actions_df, "nonexistent_target")

    def test_not_fitted_raises_on_predict(self, actions_df):
        from src.models.cxa.shot_creation_model import GlmShotCreationModel

        m = GlmShotCreationModel()
        with pytest.raises(RuntimeError):
            m.predict_proba(actions_df)


class TestXGBoostShotCreation:
    def test_fit_predict(self, actions_df):
        pytest.importorskip("xgboost")
        from src.models.cxa.shot_creation_model import XGBoostShotCreationModel

        m = XGBoostShotCreationModel(feature_set="contextual", n_estimators=30)
        m.fit(actions_df, "shot_created")
        p = m.predict_proba(actions_df)
        assert p.shape == (len(actions_df),)
        assert (p >= 0).all() and (p <= 1).all()


class TestLightGBMShotCreation:
    def test_fit_predict(self, actions_df):
        pytest.importorskip("lightgbm")
        from src.models.cxa.shot_creation_model import LightGBMShotCreationModel

        m = LightGBMShotCreationModel(feature_set="contextual", n_estimators=30)
        m.fit(actions_df, "shot_created")
        p = m.predict_proba(actions_df)
        assert p.shape == (len(actions_df),)
        assert (p >= 0).all() and (p <= 1).all()


class TestShotCreationPartialFeatures:
    def test_handles_partial_columns(self, actions_df):
        """Model should work when only a subset of feature columns are present."""
        from src.models.cxa.shot_creation_model import GlmShotCreationModel

        # Keep only traditional numeric cols + target
        keep_cols = [
            "x_location",
            "y_location",
            "distance_to_goal",
            "progressive_distance",
            "shot_created",
            "match_id",
        ]
        partial_df = actions_df[[c for c in keep_cols if c in actions_df.columns]].copy()
        m = GlmShotCreationModel(feature_set="traditional")
        m.fit(partial_df, "shot_created")
        p = m.predict_proba(partial_df)
        assert (p >= 0).all() and (p <= 1).all()


class TestWindowStabilityReport:
    def test_stability_report_best_window(self, actions_df):
        from src.models.cxa.shot_creation_model import GlmShotCreationModel

        m = GlmShotCreationModel(feature_set="traditional")
        m.fit(actions_df, "shot_created")
        report = m.stability_analysis(actions_df)
        best = report.best_window()
        assert best in (
            "shot_created",
            "shot_within_5_actions",
            "shot_within_10s",
            "shot_within_15s",
        )

    def test_stability_report_to_dataframe(self, actions_df):
        from src.models.cxa.shot_creation_model import GlmShotCreationModel

        m = GlmShotCreationModel(feature_set="traditional")
        m.fit(actions_df, "shot_created")
        report = m.stability_analysis(actions_df)
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestShotCreationLadder:
    def test_end_to_end(self, actions_df):
        from src.models.cxa.shot_creation_model import ShotCreationLadder

        ladder = ShotCreationLadder()
        results = ladder.run(
            actions_df,
            target_col="shot_created",
            match_id_col="match_id",
            n_folds=3,
            n_estimators=30,
            random_state=42,
        )
        assert len(results) >= 2  # at least GLM variants

    def test_results_sorted_by_log_loss(self, actions_df):
        from src.models.cxa.shot_creation_model import ShotCreationLadder

        ladder = ShotCreationLadder()
        results = ladder.run(
            actions_df,
            target_col="shot_created",
            match_id_col="match_id",
            n_folds=3,
            n_estimators=30,
            random_state=42,
        )
        ll_scores = [r.cv_log_loss for r in results]
        assert ll_scores == sorted(ll_scores), "Ladder not sorted by cv_log_loss"

    def test_leaderboard_returns_dataframe(self, actions_df):
        from src.models.cxa.shot_creation_model import ShotCreationLadder

        ladder = ShotCreationLadder()
        ladder.run(
            actions_df,
            target_col="shot_created",
            match_id_col="match_id",
            n_folds=3,
            n_estimators=30,
            random_state=42,
        )
        lb = ladder.leaderboard()
        assert isinstance(lb, pd.DataFrame)
        assert "cv_log_loss" in lb.columns

    def test_best_returns_top_result(self, actions_df):
        from src.models.cxa.shot_creation_model import ShotCreationLadder

        ladder = ShotCreationLadder()
        results = ladder.run(
            actions_df,
            target_col="shot_created",
            match_id_col="match_id",
            n_folds=3,
            n_estimators=30,
            random_state=42,
        )
        best = ladder.best()
        assert best.cv_log_loss == results[0].cv_log_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Shot-Quality Model Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGammaShotQuality:
    def test_fit_predict_positive_only(self, pos_actions_df):
        from src.models.cxa.shot_quality_model import GammaShotQualityModel

        m = GammaShotQualityModel(feature_set="contextual")
        m.fit(pos_actions_df, "resulting_shot_cxg")
        preds = m.predict(pos_actions_df)
        assert preds.shape == (len(pos_actions_df),)
        assert (preds > 0).all()

    def test_rejects_nonpositive_targets(self, actions_df):
        from src.models.cxa.shot_quality_model import GammaShotQualityModel

        # actions_df has zeros in resulting_shot_cxg
        df_with_zeros = actions_df[actions_df["resulting_shot_cxg"] == 0].head(20).copy()
        if df_with_zeros.empty:
            pytest.skip("No zero CxG rows in fixture")
        m = GammaShotQualityModel(feature_set="contextual")
        with pytest.raises(ValueError, match="strictly positive"):
            m.fit(df_with_zeros, "resulting_shot_cxg")

    def test_evaluate_returns_metrics(self, pos_actions_df):
        from src.models.cxa.shot_quality_model import GammaShotQualityModel, ShotQualityMetrics

        m = GammaShotQualityModel(feature_set="contextual")
        m.fit(pos_actions_df, "resulting_shot_cxg")
        metrics = m.evaluate(pos_actions_df, "resulting_shot_cxg")
        assert isinstance(metrics, ShotQualityMetrics)
        assert metrics.mae >= 0
        assert metrics.rmse >= 0


class TestXGBoostShotQuality:
    def test_fit_predict(self, actions_df):
        pytest.importorskip("xgboost")
        from src.models.cxa.shot_quality_model import XGBoostShotQualityModel

        m = XGBoostShotQualityModel(feature_set="contextual", n_estimators=30)
        pos_df = actions_df[actions_df["resulting_shot_cxg"] > 0]
        m.fit(pos_df, "resulting_shot_cxg")
        preds = m.predict(pos_df)
        assert preds.shape == (len(pos_df),)
        assert (preds > 0).all()

    def test_predicts_on_all_rows(self, actions_df):
        """XGBoost should predict on any row (not just shot rows) after fitting on positive."""
        pytest.importorskip("xgboost")
        from src.models.cxa.shot_quality_model import XGBoostShotQualityModel

        pos_df = actions_df[actions_df["resulting_shot_cxg"] > 0].reset_index(drop=True)
        m = XGBoostShotQualityModel(feature_set="contextual", n_estimators=30)
        m.fit(pos_df, "resulting_shot_cxg")
        preds = m.predict(actions_df)
        assert preds.shape == (len(actions_df),)


class TestLightGBMShotQuality:
    def test_fit_predict(self, actions_df):
        pytest.importorskip("lightgbm")
        from src.models.cxa.shot_quality_model import LightGBMShotQualityModel

        pos_df = actions_df[actions_df["resulting_shot_cxg"] > 0]
        m = LightGBMShotQualityModel(feature_set="contextual", n_estimators=30)
        m.fit(pos_df, "resulting_shot_cxg")
        preds = m.predict(pos_df)
        assert preds.shape == (len(pos_df),)
        assert (preds > 0).all()


class TestShotQualityLadder:
    def test_end_to_end(self, actions_df):
        from src.models.cxa.shot_quality_model import ShotQualityLadder

        ladder = ShotQualityLadder()
        results = ladder.run(
            actions_df,
            target_col="resulting_shot_cxg",
            match_id_col="match_id",
            n_folds=3,
            n_estimators=30,
            random_state=42,
        )
        assert len(results) >= 1

    def test_results_sorted_by_mae(self, actions_df):
        from src.models.cxa.shot_quality_model import ShotQualityLadder

        ladder = ShotQualityLadder()
        results = ladder.run(
            actions_df,
            target_col="resulting_shot_cxg",
            match_id_col="match_id",
            n_folds=3,
            n_estimators=30,
            random_state=42,
        )
        maes = [r.cv_mae for r in results]
        assert maes == sorted(maes), "Ladder not sorted by cv_mae"

    def test_leaderboard_returns_dataframe(self, actions_df):
        from src.models.cxa.shot_quality_model import ShotQualityLadder

        ladder = ShotQualityLadder()
        ladder.run(
            actions_df,
            target_col="resulting_shot_cxg",
            match_id_col="match_id",
            n_folds=3,
            n_estimators=30,
            random_state=42,
        )
        lb = ladder.leaderboard()
        assert isinstance(lb, pd.DataFrame)
        assert "cv_mae" in lb.columns


# ═══════════════════════════════════════════════════════════════════════════════
# CxA Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_pipeline(actions_df):
    """Helper: fit and return a CxAPipeline using fast GLM models."""
    from src.models.cxa.cxa_pipeline import CxAPipeline
    from src.models.cxa.shot_creation_model import GlmShotCreationModel
    from src.models.cxa.shot_quality_model import XGBoostShotQualityModel

    creation = GlmShotCreationModel(feature_set="traditional")
    creation.fit(actions_df, "shot_created")

    pos_df = actions_df[actions_df["resulting_shot_cxg"] > 0].reset_index(drop=True)
    quality = XGBoostShotQualityModel(feature_set="contextual", n_estimators=30)
    quality.fit(pos_df, "resulting_shot_cxg")

    return CxAPipeline.from_models(creation, quality)


class TestCxAPipeline:
    def test_score_output_columns(self, actions_df):
        pytest.importorskip("xgboost")
        pipeline = _make_pipeline(actions_df)
        out = pipeline.score(actions_df)
        for col in (
            "p_shot_created",
            "expected_cxg_if_shot",
            "cxa",
            "realised_cxa",
            "opponent_adjustment_delta",
        ):
            assert col in out.columns, f"Missing column: {col}"

    def test_cxa_formula(self, actions_df):
        """CxA should equal p_shot_created × expected_cxg_if_shot (within floating point)."""
        pytest.importorskip("xgboost")
        pipeline = _make_pipeline(actions_df)
        out = pipeline.score(actions_df)
        expected = out["p_shot_created"] * out["expected_cxg_if_shot"]
        np.testing.assert_allclose(out["cxa"].to_numpy(), expected.to_numpy(), rtol=1e-5)

    def test_cxa_values_are_positive(self, actions_df):
        pytest.importorskip("xgboost")
        pipeline = _make_pipeline(actions_df)
        out = pipeline.score(actions_df)
        assert (out["cxa"] > 0).all()

    def test_probabilities_in_unit_interval(self, actions_df):
        pytest.importorskip("xgboost")
        pipeline = _make_pipeline(actions_df)
        out = pipeline.score(actions_df)
        assert (out["p_shot_created"] >= 0).all() and (out["p_shot_created"] <= 1).all()

    def test_realised_cxa_for_actual_shots(self, actions_df):
        """realised_cxa should be non-NaN only for rows where shot_created==1."""
        pytest.importorskip("xgboost")
        pipeline = _make_pipeline(actions_df)
        out = pipeline.score(actions_df)
        if "shot_created" in out.columns:
            shot_rows = out[out["shot_created"] == 1]
            non_shot_rows = out[out["shot_created"] == 0]
            # Non-shot rows should have NaN realised_cxa
            assert non_shot_rows["realised_cxa"].isna().all()
            # Shot rows should have finite realised_cxa (where resulting_shot_cxg > 0)
            shot_with_cxg = (
                shot_rows[shot_rows["resulting_shot_cxg"] > 0]
                if "resulting_shot_cxg" in out.columns
                else shot_rows
            )
            if not shot_with_cxg.empty:
                assert shot_with_cxg["realised_cxa"].notna().any()

    def test_score_passes_filters_pass_types(self, actions_df):
        """score_passes should return only passes and crosses."""
        pytest.importorskip("xgboost")
        pipeline = _make_pipeline(actions_df)
        out = pipeline.score_passes(actions_df)
        if "action_type" in out.columns:
            assert out["action_type"].isin({"pass", "cross"}).all()

    def test_score_returns_creative_actions_only(self, actions_df):
        """score with filter_creative=True should only include CREATIVE_ACTION_TYPES."""
        pytest.importorskip("xgboost")
        from src.models.cxa.cxa_pipeline import CREATIVE_ACTION_TYPES

        pipeline = _make_pipeline(actions_df)
        out = pipeline.score(actions_df, filter_creative=True)
        if "action_type" in out.columns:
            assert out["action_type"].isin(CREATIVE_ACTION_TYPES).all()

    def test_fit_method(self, actions_df):
        """CxAPipeline.fit() should train both stages from scratch."""
        pytest.importorskip("xgboost")
        from src.models.cxa.cxa_pipeline import CxAPipeline
        from src.models.cxa.shot_creation_model import GlmShotCreationModel
        from src.models.cxa.shot_quality_model import XGBoostShotQualityModel

        creation = GlmShotCreationModel(feature_set="traditional")
        quality = XGBoostShotQualityModel(feature_set="contextual", n_estimators=30)
        pipeline = CxAPipeline(creation, quality)
        pipeline.fit(actions_df, "shot_created", "resulting_shot_cxg")
        out = pipeline.score(actions_df)
        assert len(out) > 0

    def test_save_load_roundtrip(self, actions_df, tmp_path):
        """Pipeline should serialise and deserialise correctly."""
        pytest.importorskip("xgboost")
        from src.models.cxa.cxa_pipeline import CxAPipeline

        pipeline = _make_pipeline(actions_df)
        save_path = tmp_path / "pipeline.pkl"
        pipeline.save(save_path)
        loaded = CxAPipeline.load(save_path)
        out1 = pipeline.score(actions_df)
        out2 = loaded.score(actions_df)
        np.testing.assert_allclose(out1["cxa"].to_numpy(), out2["cxa"].to_numpy(), rtol=1e-6)

    def test_empty_df_raises(self, actions_df):
        pytest.importorskip("xgboost")
        pipeline = _make_pipeline(actions_df)
        empty_all_creative = actions_df[actions_df["action_type"] == "nonexistent_type"]
        out = pipeline.score(empty_all_creative)
        # Should return an empty DataFrame (not raise)
        assert isinstance(out, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════════
# Leaderboard Tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_scored_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Minimal scored DataFrame with required CxA columns."""
    rng = np.random.default_rng(seed)
    n_players = 10
    n_teams = 4
    n_matches = 6
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            "player_id": [f"p{i % n_players}" for i in range(n)],
            "team_id": [f"t{i % n_teams}" for i in range(n)],
            "match_id": [f"m{i % n_matches}" for i in range(n)],
            "action_type": rng.choice(["pass", "carry", "cross", "cutback"], n),
            "cxa": rng.uniform(0.01, 0.15, n),
            "p_shot_created": rng.uniform(0.05, 0.4, n),
            "expected_cxg_if_shot": rng.uniform(0.05, 0.35, n),
            "realised_cxa": np.where(rng.uniform(0, 1, n) > 0.8, rng.uniform(0.05, 0.3, n), np.nan),
            "opponent_adjustment_delta": rng.uniform(-0.02, 0.02, n),
            "sequence_type": rng.choice(["open_play", "transition", "set_piece"], n),
            "transition_or_settled": rng.choice(["transition", "settled"], n),
            "set_piece_flag": rng.integers(0, 2, n).astype(float),
        }
    )


class TestCxAPlayerLeaderboard:
    def test_has_expected_columns(self):
        from src.dashboards.cxa_leaderboard import build_player_leaderboard

        scored = _make_scored_df()
        lb = build_player_leaderboard(scored, min_minutes=0)
        for col in ("CxA_per_90", "CxA_total", "shot_creation_prob_per_90"):
            assert col in lb.columns, f"Missing column: {col}"

    def test_per_90_scaling(self):
        from src.dashboards.cxa_leaderboard import build_player_leaderboard

        scored = _make_scored_df(n=200)
        mins_df = pd.DataFrame(
            {
                "player_id": [f"p{i}" for i in range(10)],
                "minutes_played": [450.0] * 10,
            }
        )
        lb = build_player_leaderboard(scored, minutes_df=mins_df, min_minutes=0)
        assert not lb.empty
        # CxA_per_90 should be positive
        assert (lb["CxA_per_90"] > 0).all()

    def test_min_minutes_filter(self):
        from src.dashboards.cxa_leaderboard import build_player_leaderboard

        scored = _make_scored_df(n=20)  # few actions per player
        mins_df = pd.DataFrame(
            {
                "player_id": [f"p{i}" for i in range(10)],
                "minutes_played": [20.0] * 10,  # below default threshold
            }
        )
        lb = build_player_leaderboard(scored, minutes_df=mins_df, min_minutes=90.0)
        assert lb.empty

    def test_sorted_by_cxa_per_90(self):
        from src.dashboards.cxa_leaderboard import build_player_leaderboard

        scored = _make_scored_df(n=200)
        lb = build_player_leaderboard(scored, min_minutes=0)
        values = lb["CxA_per_90"].tolist()
        assert values == sorted(values, reverse=True)

    def test_transition_and_cutback_columns_present(self):
        from src.dashboards.cxa_leaderboard import build_player_leaderboard

        scored = _make_scored_df()
        lb = build_player_leaderboard(scored, min_minutes=0)
        assert "CxA_transition" in lb.columns
        assert "CxA_cutbacks" in lb.columns


class TestCxATeamLeaderboard:
    def test_has_expected_columns(self):
        from src.dashboards.cxa_leaderboard import build_team_leaderboard

        scored = _make_scored_df()
        lb = build_team_leaderboard(scored, min_minutes=0)
        for col in ("CxA_per_90", "CxA_total", "CxA_passes_per_90", "CxA_carries_per_90"):
            assert col in lb.columns, f"Missing column: {col}"

    def test_returns_non_empty_for_valid_df(self):
        from src.dashboards.cxa_leaderboard import build_team_leaderboard

        scored = _make_scored_df(n=200)
        lb = build_team_leaderboard(scored, min_minutes=0)
        assert not lb.empty

    def test_missing_team_id_returns_empty(self):
        from src.dashboards.cxa_leaderboard import build_team_leaderboard

        scored = _make_scored_df()
        scored = scored.drop(columns=["team_id"])
        lb = build_team_leaderboard(scored, min_minutes=0)
        assert lb.empty

    def test_sorted_by_cxa_per_90(self):
        from src.dashboards.cxa_leaderboard import build_team_leaderboard

        scored = _make_scored_df(n=400)
        lb = build_team_leaderboard(scored, min_minutes=0)
        values = lb["CxA_per_90"].tolist()
        assert values == sorted(values, reverse=True)
