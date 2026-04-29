"""
Phase 7 tests — CxT Contextual Value Models.

Covers:
  • CxTFeatureSetSpec registry and validation
  • BEFORE_TO_AFTER / AFTER_TO_BEFORE mappings
  • compute_possession_cxg helper
  • GammaStateValueModel — fit / predict / evaluate (positive targets)
  • XGBoostStateValueModel — skipped if xgboost not installed
  • LightGBMStateValueModel — skipped if lightgbm not installed
  • FFNNStateValueModel — skipped if torch not installed
  • StateValueLadder — sorting, graceful skip
  • CxTPipeline — score output columns, formula v_after - v_before
  • CxTPipeline — unsuccessful action negation
  • CxTPipeline — decompose returns CxTDecompositionRecord list
  • CxTPipeline — save / load round-trip
  • build_cxt_player_leaderboard — per-90 scaling, min_minutes filter
  • build_cxt_team_leaderboard — per-90 scaling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _make_actions_df(n: int = 200, n_matches: int = 8, seed: int = 42) -> pd.DataFrame:
    """Synthetic actions DataFrame with all CxT CONTEXTUAL feature columns."""
    rng = np.random.default_rng(seed)

    x = rng.uniform(20, 100, n).astype(float)
    y = rng.uniform(5, 63, n).astype(float)
    end_x = np.clip(x + rng.uniform(-20, 25, n), 0, 105)
    end_y = np.clip(y + rng.uniform(-10, 10, n), 0, 68)

    dist_to_goal = np.sqrt((105 - x) ** 2 + (34 - y) ** 2)
    end_dist = np.sqrt((105 - end_x) ** 2 + (34 - end_y) ** 2)
    prog_dist = dist_to_goal - end_dist

    action_types = rng.choice(["pass", "carry", "cross", "cutback"], n)
    seq_types = rng.choice(["open_play", "transition", "set_piece"], n)
    poss_zones = rng.choice(["defensive", "middle", "attacking"], n)
    t_or_s = rng.choice(["transition", "settled"], n)
    phases = rng.choice(["build_up", "progression", "final_third"], n)
    score_states = rng.choice(["winning", "drawing", "losing"], n)
    home_away = rng.choice(["home", "away"], n)
    outcomes = rng.choice(["complete", "incomplete", "out"], n)

    # Target: possession_cxg — simulate non-negative values
    # Some rows have non-zero value (shots resulted from possession)
    logit = -2.0 + 0.03 * prog_dist + rng.normal(0, 0.5, n)
    p_val = 1 / (1 + np.exp(-logit))
    possession_cxg = np.where(
        rng.uniform(0, 1, n) < p_val,
        rng.uniform(0.01, 0.4, n),
        0.0
    )

    match_ids = [f"m{i % n_matches}" for i in range(n)]
    player_ids = [f"p{i % 20}" for i in range(n)]
    team_ids = [f"t{i % 4}" for i in range(n)]
    poss_ids = [f"pos{i % 50}" for i in range(n)]

    df = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)],
        "match_id": match_ids,
        "player_id": player_ids,
        "team_id": team_ids,
        "possession_id": poss_ids,
        # Before-state location
        "x_location": x,
        "y_location": y,
        "distance_to_goal": dist_to_goal,
        # After-state location
        "end_x": end_x,
        "end_y": end_y,
        "end_distance_to_goal": end_dist,
        # Action magnitude
        "progressive_distance": prog_dist,
        "pass_length": rng.uniform(5, 40, n),
        # Boolean flags
        "in_box": (x > 88).astype(int),
        "is_central": ((y > 24) & (y < 44)).astype(int),
        "end_in_box": (end_x > 88).astype(int),
        "end_is_central": ((end_y > 24) & (end_y < 44)).astype(int),
        "under_pressure": rng.integers(0, 2, n),
        "after_under_pressure": rng.integers(0, 2, n),
        "box_entry": ((x <= 88) & (end_x > 88)).astype(int),
        "cross": (action_types == "cross").astype(int),
        "cutback": (action_types == "cutback").astype(int),
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
        # Boolean context
        "knockout_or_group": rng.integers(0, 2, n),
        "set_piece_flag": rng.integers(0, 2, n),
        "counterpress_regain_flag": rng.integers(0, 2, n),
        "central_progression": rng.integers(0, 2, n),
        "through_ball": rng.integers(0, 2, n),
        "switch": rng.integers(0, 2, n),
        # Categorical
        "action_type": action_types,
        "possession_start_zone": poss_zones,
        "score_state": score_states,
        "home_or_away": home_away,
        "sequence_type": seq_types,
        "transition_or_settled": t_or_s,
        "phase_of_play": phases,
        # Outcome (for success flag)
        "outcome": outcomes,
        # Target
        "possession_cxg": possession_cxg,
    })
    return df


def _make_minutes_df(actions_df: pd.DataFrame, id_col: str = "player_id") -> pd.DataFrame:
    ids = actions_df[id_col].unique()
    return pd.DataFrame({id_col: ids, "minutes_played": [900.0] * len(ids)})


@pytest.fixture(scope="module")
def actions_df():
    return _make_actions_df(n=200, n_matches=8, seed=42)


@pytest.fixture(scope="module")
def pos_actions_df():
    """Rows with strictly positive possession_cxg (for Gamma GLM)."""
    df = _make_actions_df(n=200, n_matches=8, seed=42)
    return df[df["possession_cxg"] > 0].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Set Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCxTFeatureSets:
    def test_registry_has_three_entries(self):
        from src.models.cxt.feature_sets import _REGISTRY
        assert len(_REGISTRY) == 3

    def test_all_sets_accessible_by_name(self):
        from src.models.cxt.feature_sets import get_feature_set
        for name in ("traditional", "contextual", "full_360"):
            fs = get_feature_set(name)
            assert fs.name == name

    def test_unknown_name_raises_value_error(self):
        from src.models.cxt.feature_sets import get_feature_set
        with pytest.raises(ValueError, match="Unknown CxT feature set"):
            get_feature_set("nonexistent")

    def test_feature_sets_have_no_duplicates(self):
        from src.models.cxt.feature_sets import TRADITIONAL, CONTEXTUAL, FULL_360
        for fs in (TRADITIONAL, CONTEXTUAL, FULL_360):
            features = fs.all_features
            assert len(features) == len(set(features)), f"{fs.name} has duplicate features"

    def test_feature_set_nesting(self):
        """FULL_360 > CONTEXTUAL > TRADITIONAL in feature count."""
        from src.models.cxt.feature_sets import TRADITIONAL, CONTEXTUAL, FULL_360
        assert len(FULL_360.all_features) > len(CONTEXTUAL.all_features)
        assert len(CONTEXTUAL.all_features) > len(TRADITIONAL.all_features)

    def test_360_flag(self):
        from src.models.cxt.feature_sets import TRADITIONAL, CONTEXTUAL, FULL_360
        assert not TRADITIONAL.requires_360
        assert not CONTEXTUAL.requires_360
        assert FULL_360.requires_360

    def test_numeric_all_is_numeric_plus_boolean(self):
        from src.models.cxt.feature_sets import TRADITIONAL
        expected = TRADITIONAL.numeric + TRADITIONAL.boolean
        assert TRADITIONAL.numeric_all == expected

    def test_before_to_after_has_all_before_location_cols(self):
        from src.models.cxt.feature_sets import BEFORE_TO_AFTER
        assert "x_location" in BEFORE_TO_AFTER
        assert "y_location" in BEFORE_TO_AFTER
        assert "end_x" in BEFORE_TO_AFTER.values()

    def test_after_to_before_is_inverse(self):
        from src.models.cxt.feature_sets import BEFORE_TO_AFTER, AFTER_TO_BEFORE
        for before, after in BEFORE_TO_AFTER.items():
            assert AFTER_TO_BEFORE[after] == before


# ═══════════════════════════════════════════════════════════════════════════════
# Possession CxG Helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePossessionCxg:
    def test_returns_series_same_length(self):
        from src.models.cxt.state_value_model import compute_possession_cxg
        df = pd.DataFrame({
            "match_id": ["m1"] * 5,
            "possession_id": ["p1", "p1", "p2", "p2", "p2"],
            "event_cxg": [0.0, 0.2, 0.0, 0.0, 0.1],
        })
        result = compute_possession_cxg(df)
        assert len(result) == 5

    def test_zero_if_no_shots_in_possession(self):
        from src.models.cxt.state_value_model import compute_possession_cxg
        df = pd.DataFrame({
            "match_id": ["m1"] * 3,
            "possession_id": ["p1", "p1", "p1"],
            "event_cxg": [0.0, 0.0, 0.0],
        })
        result = compute_possession_cxg(df)
        assert (result == 0.0).all()

    def test_discounting_reduces_value_for_later_shots(self):
        """Possession with 2 shots should have value between 1×first and sum of both."""
        from src.models.cxt.state_value_model import compute_possession_cxg, SHOT_DISCOUNT
        cxg1, cxg2 = 0.3, 0.2
        df = pd.DataFrame({
            "match_id": ["m1"] * 3,
            "possession_id": ["p1"] * 3,
            "event_cxg": [cxg1, 0.0, cxg2],
        })
        result = compute_possession_cxg(df)
        expected = cxg1 + SHOT_DISCOUNT * cxg2
        assert abs(result.iloc[0] - expected) < 1e-8

    def test_missing_columns_returns_zeros(self):
        from src.models.cxt.state_value_model import compute_possession_cxg
        df = pd.DataFrame({"match_id": ["m1"] * 3})
        result = compute_possession_cxg(df)
        assert (result == 0.0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Gamma State-Value Model
# ═══════════════════════════════════════════════════════════════════════════════

class TestGammaStateValueModel:
    def test_fit_returns_self(self, pos_actions_df):
        from src.models.cxt.state_value_model import GammaStateValueModel
        m = GammaStateValueModel(feature_set="traditional")
        result = m.fit(pos_actions_df, target_col="possession_cxg")
        assert result is m

    def test_predict_returns_non_negative_array(self, pos_actions_df):
        from src.models.cxt.state_value_model import GammaStateValueModel
        m = GammaStateValueModel(feature_set="traditional").fit(pos_actions_df)
        preds = m.predict(pos_actions_df)
        assert preds.ndim == 1
        assert len(preds) == len(pos_actions_df)
        assert (preds >= 0).all()

    def test_evaluate_returns_metrics(self, pos_actions_df):
        from src.models.cxt.state_value_model import GammaStateValueModel, StateValueMetrics
        m = GammaStateValueModel(feature_set="traditional").fit(pos_actions_df)
        metrics = m.evaluate(pos_actions_df, target_col="possession_cxg")
        assert isinstance(metrics, StateValueMetrics)
        assert metrics.mae >= 0
        assert metrics.rmse >= 0

    def test_raises_on_non_positive_target(self, actions_df):
        """Gamma GLM must reject rows with target == 0."""
        from src.models.cxt.state_value_model import GammaStateValueModel
        m = GammaStateValueModel(feature_set="traditional")
        with pytest.raises(ValueError, match="strictly positive"):
            m.fit(actions_df, target_col="possession_cxg")

    def test_raises_on_empty_df(self):
        from src.models.cxt.state_value_model import GammaStateValueModel
        m = GammaStateValueModel(feature_set="traditional")
        with pytest.raises(ValueError):
            m.fit(pd.DataFrame(), target_col="possession_cxg")

    def test_raises_if_not_fitted(self, actions_df):
        from src.models.cxt.state_value_model import GammaStateValueModel
        m = GammaStateValueModel(feature_set="traditional")
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict(actions_df)

    def test_contextual_feature_set_also_fits(self, pos_actions_df):
        from src.models.cxt.state_value_model import GammaStateValueModel
        m = GammaStateValueModel(feature_set="contextual").fit(pos_actions_df)
        preds = m.predict(pos_actions_df)
        assert len(preds) == len(pos_actions_df)

    def test_save_load_roundtrip(self, pos_actions_df, tmp_path):
        from src.models.cxt.state_value_model import GammaStateValueModel
        m = GammaStateValueModel(feature_set="traditional").fit(pos_actions_df)
        p = tmp_path / "gamma_sv.pkl"
        m.save(p)
        loaded = GammaStateValueModel.load(p)
        np.testing.assert_array_almost_equal(
            m.predict(pos_actions_df), loaded.predict(pos_actions_df)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost State-Value Model (conditionally skipped)
# ═══════════════════════════════════════════════════════════════════════════════

class TestXGBoostStateValueModel:
    def test_fit_predict(self, actions_df):
        pytest.importorskip("xgboost", reason="xgboost not installed")
        from src.models.cxt.state_value_model import XGBoostStateValueModel
        m = XGBoostStateValueModel(feature_set="contextual", n_estimators=20)
        m.fit(actions_df, target_col="possession_cxg")
        preds = m.predict(actions_df)
        assert len(preds) == len(actions_df)
        assert (preds >= 0).all()

    def test_evaluate_returns_metrics(self, actions_df):
        pytest.importorskip("xgboost", reason="xgboost not installed")
        from src.models.cxt.state_value_model import XGBoostStateValueModel, StateValueMetrics
        m = XGBoostStateValueModel(feature_set="contextual", n_estimators=20)
        m.fit(actions_df, target_col="possession_cxg")
        metrics = m.evaluate(actions_df)
        assert isinstance(metrics, StateValueMetrics)
        assert metrics.mae >= 0

    def test_save_load_roundtrip(self, actions_df, tmp_path):
        pytest.importorskip("xgboost", reason="xgboost not installed")
        from src.models.cxt.state_value_model import XGBoostStateValueModel
        m = XGBoostStateValueModel(feature_set="traditional", n_estimators=10)
        m.fit(actions_df)
        p = tmp_path / "xgb_sv.pkl"
        m.save(p)
        loaded = XGBoostStateValueModel.load(p)
        np.testing.assert_array_almost_equal(
            m.predict(actions_df), loaded.predict(actions_df)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LightGBM State-Value Model (conditionally skipped)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLightGBMStateValueModel:
    def test_fit_predict(self, actions_df):
        pytest.importorskip("lightgbm", reason="lightgbm not installed")
        from src.models.cxt.state_value_model import LightGBMStateValueModel
        m = LightGBMStateValueModel(feature_set="contextual", n_estimators=20)
        m.fit(actions_df, target_col="possession_cxg")
        preds = m.predict(actions_df)
        assert len(preds) == len(actions_df)
        assert (preds >= 0).all()

    def test_evaluate_returns_metrics(self, actions_df):
        pytest.importorskip("lightgbm", reason="lightgbm not installed")
        from src.models.cxt.state_value_model import LightGBMStateValueModel, StateValueMetrics
        m = LightGBMStateValueModel(feature_set="contextual", n_estimators=20)
        m.fit(actions_df)
        metrics = m.evaluate(actions_df)
        assert isinstance(metrics, StateValueMetrics)
        assert metrics.mae >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# State-Value Ladder
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateValueLadder:
    def test_run_returns_sorted_by_cv_mae(self, actions_df):
        """Ladder must have at least one candidate (Gamma GLM on positive rows)."""
        from src.models.cxt.state_value_model import StateValueLadder
        ladder = StateValueLadder()
        results = ladder.run(
            actions_df,
            target_col="possession_cxg",
            n_folds=3,
            n_estimators=20,
            random_state=42,
        )
        assert len(results) >= 1
        for i in range(len(results) - 1):
            assert results[i].cv_mae <= results[i + 1].cv_mae

    def test_leaderboard_is_dataframe(self, actions_df):
        from src.models.cxt.state_value_model import StateValueLadder
        ladder = StateValueLadder()
        ladder.run(actions_df, n_folds=3, n_estimators=20, random_state=42)
        lb = ladder.leaderboard()
        assert isinstance(lb, pd.DataFrame)
        assert "cv_mae" in lb.columns

    def test_best_returns_lowest_mae(self, actions_df):
        from src.models.cxt.state_value_model import StateValueLadder
        ladder = StateValueLadder()
        results = ladder.run(actions_df, n_folds=3, n_estimators=20, random_state=42)
        best = ladder.best()
        assert best.cv_mae == results[0].cv_mae
        assert best.rank == 1

    def test_ladder_result_has_required_fields(self, actions_df):
        from src.models.cxt.state_value_model import StateValueLadder, StateValueLadderResult
        ladder = StateValueLadder()
        results = ladder.run(actions_df, n_folds=3, n_estimators=20, random_state=42)
        r = results[0]
        assert isinstance(r, StateValueLadderResult)
        assert r.name != ""
        assert r.family != ""
        assert r.n_cv_folds_used >= 1

    def test_raises_before_run(self):
        from src.models.cxt.state_value_model import StateValueLadder
        ladder = StateValueLadder()
        with pytest.raises(RuntimeError):
            ladder.leaderboard()
        with pytest.raises(RuntimeError):
            ladder.best()

    def test_raises_on_missing_target(self, actions_df):
        from src.models.cxt.state_value_model import StateValueLadder
        ladder = StateValueLadder()
        with pytest.raises(ValueError, match="Missing target"):
            ladder.run(actions_df, target_col="nonexistent_col", n_folds=3, n_estimators=20)


# ═══════════════════════════════════════════════════════════════════════════════
# CxT Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def fitted_pipeline(pos_actions_df, actions_df):
    """A CxTPipeline with a fitted GammaStateValueModel on positive rows."""
    from src.models.cxt.cxt_pipeline import CxTPipeline
    from src.models.cxt.state_value_model import GammaStateValueModel
    sv = GammaStateValueModel(feature_set="traditional").fit(pos_actions_df)
    pipeline = CxTPipeline(state_value_model=sv)
    return pipeline


class TestCxTPipelineScore:
    def test_score_adds_cxt_columns(self, fitted_pipeline, actions_df):
        scored = fitted_pipeline.score(actions_df)
        assert "v_before" in scored.columns
        assert "v_after" in scored.columns
        assert "cxt" in scored.columns

    def test_cxt_equals_v_after_minus_v_before(self, fitted_pipeline, actions_df):
        scored = fitted_pipeline.score(actions_df)
        # Only successful actions have cxt = v_after - v_before;
        # unsuccessful ones are negated (cxt = -abs(v_after - v_before))
        scored_valid = scored.dropna(subset=["cxt"])
        successful = scored_valid[scored_valid["is_successful"].astype(bool)]
        np.testing.assert_array_almost_equal(
            successful["cxt"].to_numpy(),
            (successful["v_after"] - successful["v_before"]).to_numpy(),
            decimal=6,
        )

    def test_score_output_has_same_length(self, fitted_pipeline, actions_df):
        scored = fitted_pipeline.score(actions_df, filter_cxt_actions=False)
        assert len(scored) == len(actions_df)

    def test_score_cxt_is_nan_for_non_cxt_action_types(self, fitted_pipeline, actions_df):
        from src.models.cxt.cxt_pipeline import CXT_ACTION_TYPES
        scored = fitted_pipeline.score(actions_df, filter_cxt_actions=True)
        non_cxt = scored[~scored["action_type"].isin(CXT_ACTION_TYPES)]
        assert non_cxt["cxt"].isna().all()

    def test_unsuccessful_actions_have_negated_cxt(self, fitted_pipeline):
        """Unsuccessful action: CxT should be <= 0."""
        from src.models.cxt.cxt_pipeline import CxTPipeline
        from src.models.cxt.state_value_model import GammaStateValueModel
        pos_df = _make_actions_df(n=50, n_matches=4, seed=7)
        pos_df = pos_df[pos_df["possession_cxg"] > 0].reset_index(drop=True)
        sv = GammaStateValueModel(feature_set="traditional").fit(pos_df)
        pipeline = CxTPipeline(state_value_model=sv)

        # Force incomplete pass (should yield negative CxT)
        df_bad = _make_actions_df(n=20, n_matches=4, seed=8)
        df_bad = df_bad[df_bad["action_type"] == "pass"].head(10).copy()
        df_bad["outcome"] = "incomplete"
        df_bad["possession_cxg"] = 0.1  # doesn't matter — not using for fit

        scored = pipeline.score(df_bad, filter_cxt_actions=False)
        scored_bad = scored[scored["outcome"] == "incomplete"].dropna(subset=["cxt"])
        if len(scored_bad) > 0:
            assert (scored_bad["cxt"] <= 0).all()

    def test_v_before_and_v_after_are_non_negative(self, fitted_pipeline, actions_df):
        scored = fitted_pipeline.score(actions_df)
        valid = scored.dropna(subset=["v_before", "v_after"])
        assert (valid["v_before"] >= 0).all()
        assert (valid["v_after"] >= 0).all()


class TestCxTPipelineDecompose:
    def test_decompose_returns_list_of_records(self, fitted_pipeline, actions_df):
        from src.models.cxt.cxt_pipeline import CxTDecompositionRecord
        records = fitted_pipeline.decompose(actions_df)
        assert isinstance(records, list)
        assert len(records) > 0
        assert isinstance(records[0], CxTDecompositionRecord)

    def test_record_has_required_fields(self, fitted_pipeline, actions_df):
        records = fitted_pipeline.decompose(actions_df)
        r = records[0]
        for attr in ("event_id", "player_id", "team_id", "match_id",
                     "action_type", "x_before", "y_before", "x_after", "y_after",
                     "v_before", "v_after", "cxt", "sequence_type",
                     "opponent_adjustment_delta"):
            assert hasattr(r, attr), f"Missing field: {attr}"

    def test_cxt_equals_v_after_minus_v_before_in_record(self, fitted_pipeline, actions_df):
        records = fitted_pipeline.decompose(actions_df)
        for r in records[:20]:
            expected = r.v_after - r.v_before
            assert abs(r.cxt - expected) < 1e-6 or r.cxt <= 0  # negated for failed


class TestCxTPipelineSaveLoad:
    def test_save_load_roundtrip(self, fitted_pipeline, actions_df, tmp_path):
        from src.models.cxt.cxt_pipeline import CxTPipeline
        p = tmp_path / "cxt_pipeline.pkl"
        fitted_pipeline.save(p)
        loaded = CxTPipeline.load(p)
        scored_orig = fitted_pipeline.score(actions_df)
        scored_load = loaded.score(actions_df)
        np.testing.assert_array_almost_equal(
            scored_orig["cxt"].fillna(0).to_numpy(),
            scored_load["cxt"].fillna(0).to_numpy(),
        )

    def test_from_models_classmethod(self, pos_actions_df):
        from src.models.cxt.cxt_pipeline import CxTPipeline
        from src.models.cxt.state_value_model import GammaStateValueModel
        sv = GammaStateValueModel(feature_set="traditional").fit(pos_actions_df)
        pipeline = CxTPipeline.from_models(state_value_model=sv)
        assert isinstance(pipeline, CxTPipeline)

    def test_fit_method_fits_state_value_model(self, actions_df, pos_actions_df):
        """CxTPipeline.fit() must call state_value_model.fit() on positive rows."""
        from src.models.cxt.cxt_pipeline import CxTPipeline
        from src.models.cxt.state_value_model import GammaStateValueModel
        sv = GammaStateValueModel(feature_set="traditional")
        pipeline = CxTPipeline(state_value_model=sv)
        # fit on positive subset (Gamma requires >0 targets)
        pipeline.fit(pos_actions_df, target_col="possession_cxg")
        # After fit, predict should work
        preds = pipeline.state_value_model.predict(pos_actions_df)
        assert len(preds) == len(pos_actions_df)

    def test_score_raises_if_not_fitted(self, actions_df):
        from src.models.cxt.cxt_pipeline import CxTPipeline
        from src.models.cxt.state_value_model import GammaStateValueModel
        sv = GammaStateValueModel(feature_set="traditional")
        pipeline = CxTPipeline(state_value_model=sv)
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.score(actions_df)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature remapping helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildAfterStateDf:
    def test_after_state_remaps_end_x_to_x_location(self, actions_df):
        from src.models.cxt.cxt_pipeline import _build_after_state_df
        from src.models.cxt.feature_sets import TRADITIONAL
        after_df = _build_after_state_df(actions_df, TRADITIONAL)
        # end_x should now be the x_location column
        np.testing.assert_array_almost_equal(
            after_df["x_location"].to_numpy(),
            actions_df["end_x"].to_numpy(),
        )

    def test_after_state_remaps_end_y_to_y_location(self, actions_df):
        from src.models.cxt.cxt_pipeline import _build_after_state_df
        from src.models.cxt.feature_sets import TRADITIONAL
        after_df = _build_after_state_df(actions_df, TRADITIONAL)
        np.testing.assert_array_almost_equal(
            after_df["y_location"].to_numpy(),
            actions_df["end_y"].to_numpy(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CxT Leaderboards
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def scored_df():
    """Scored DataFrame with cxt column for leaderboard tests."""
    df = _make_actions_df(n=300, n_matches=10, seed=99)
    rng = np.random.default_rng(99)
    df["v_before"] = rng.uniform(0, 0.3, len(df))
    df["v_after"] = rng.uniform(0, 0.4, len(df))
    df["cxt"] = df["v_after"] - df["v_before"]
    df["opponent_adjustment_delta"] = rng.uniform(-0.05, 0.05, len(df))
    return df


class TestCxTPlayerLeaderboard:
    def test_returns_dataframe(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_player_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="player_id")
        lb = build_cxt_player_leaderboard(scored_df, mins_df, min_minutes=0.0)
        assert isinstance(lb, pd.DataFrame)

    def test_has_cxt_per_90_column(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_player_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="player_id")
        lb = build_cxt_player_leaderboard(scored_df, mins_df, min_minutes=0.0)
        assert "CxT_per_90" in lb.columns

    def test_sorted_by_cxt_per_90_descending(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_player_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="player_id")
        lb = build_cxt_player_leaderboard(scored_df, mins_df, min_minutes=0.0)
        vals = lb["CxT_per_90"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_min_minutes_filter(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_player_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="player_id")
        lb_low = build_cxt_player_leaderboard(scored_df, mins_df, min_minutes=0.0)
        lb_high = build_cxt_player_leaderboard(scored_df, mins_df, min_minutes=2000.0)
        assert len(lb_high) <= len(lb_low)

    def test_per_90_scaling_correct(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_player_leaderboard
        rng = np.random.default_rng(1)
        df_single = scored_df.copy()
        # Keep only player_id == "p0"
        p0 = df_single[df_single["player_id"] == "p0"].copy()
        mins_df = pd.DataFrame({"player_id": ["p0"], "minutes_played": [90.0]})
        lb = build_cxt_player_leaderboard(p0, mins_df, min_minutes=0.0)
        expected = p0["cxt"].sum()
        row = lb[lb["player_id"] == "p0"]
        if len(row) > 0:
            assert abs(row["CxT_per_90"].iloc[0] - expected) < 1e-4

    def test_has_action_type_breakdowns(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_player_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="player_id")
        lb = build_cxt_player_leaderboard(scored_df, mins_df, min_minutes=0.0)
        assert "CxT_carries_per_90" in lb.columns
        assert "CxT_passes_per_90" in lb.columns

    def test_cxt_minus_xt_column_present_when_oad_available(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_player_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="player_id")
        lb = build_cxt_player_leaderboard(scored_df, mins_df, min_minutes=0.0)
        assert "CxT_minus_xT_per_90" in lb.columns

    def test_transition_column_present(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_player_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="player_id")
        lb = build_cxt_player_leaderboard(scored_df, mins_df, min_minutes=0.0)
        assert "CxT_transition_per_90" in lb.columns

    def test_empty_scored_df_returns_empty_df(self):
        from src.dashboards.cxt_leaderboard import build_cxt_player_leaderboard
        lb = build_cxt_player_leaderboard(pd.DataFrame(), pd.DataFrame(), min_minutes=0.0)
        assert isinstance(lb, pd.DataFrame)
        assert len(lb) == 0


class TestCxTTeamLeaderboard:
    def test_returns_dataframe(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_team_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="team_id")
        lb = build_cxt_team_leaderboard(scored_df, mins_df, min_minutes=0.0)
        assert isinstance(lb, pd.DataFrame)

    def test_has_cxt_per_90_column(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_team_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="team_id")
        lb = build_cxt_team_leaderboard(scored_df, mins_df, min_minutes=0.0)
        assert "CxT_per_90" in lb.columns

    def test_team_leaderboard_uses_team_id(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_team_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="team_id")
        lb = build_cxt_team_leaderboard(scored_df, mins_df, min_minutes=0.0)
        assert "team_id" in lb.columns

    def test_sorted_descending(self, scored_df):
        from src.dashboards.cxt_leaderboard import build_cxt_team_leaderboard
        mins_df = _make_minutes_df(scored_df, id_col="team_id")
        lb = build_cxt_team_leaderboard(scored_df, mins_df, min_minutes=0.0)
        vals = lb["CxT_per_90"].to_list()
        assert vals == sorted(vals, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CXT_ACTION_TYPES constant
# ═══════════════════════════════════════════════════════════════════════════════

class TestCxtActionTypes:
    def test_cxt_action_types_contains_expected(self):
        from src.models.cxt.cxt_pipeline import CXT_ACTION_TYPES
        for action in ("pass", "carry", "cross", "cutback"):
            assert action in CXT_ACTION_TYPES

    def test_cxt_action_types_is_frozenset(self):
        from src.models.cxt.cxt_pipeline import CXT_ACTION_TYPES
        assert isinstance(CXT_ACTION_TYPES, frozenset)


# ═══════════════════════════════════════════════════════════════════════════════
# StateValueMetrics dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateValueMetrics:
    def test_construction(self):
        from src.models.cxt.state_value_model import StateValueMetrics
        m = StateValueMetrics(mae=0.05, rmse=0.08, spearman=0.6)
        assert m.mae == 0.05
        assert m.rmse == 0.08
        assert m.spearman == 0.6

    def test_calibration_by_zone_defaults_empty(self):
        from src.models.cxt.state_value_model import StateValueMetrics
        m = StateValueMetrics(mae=0.1, rmse=0.15, spearman=None)
        assert isinstance(m.calibration_by_zone, dict)
