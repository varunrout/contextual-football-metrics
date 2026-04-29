"""
tests/models/test_cxg_phase5.py

Phase 5 unit tests — feature sets, GLM contextual, XGBoost, LightGBM, ladder.
All tests use a small synthetic fixture so no StatsBomb data is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.cxg.feature_sets import (
    CONTEXTUAL,
    FULL_360,
    TRADITIONAL,
    FeatureSetSpec,
    get_feature_set,
)
from src.models.cxg.glm_contextual import GlmContextualCxG, GlmContextualMetrics
from src.models.cxg.xgboost_model import XGBoostCxGModel, XGBCxGMetrics
from src.models.cxg.lightgbm_model import LightGBMCxGModel, LGBMCxGMetrics
from src.models.cxg.ladder import CxGLadder, LadderResult


# ── Fixture ───────────────────────────────────────────────────────────────────

def _shots_contextual(n: int = 120, n_matches: int = 6, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic shot DataFrame containing all FULL_360 features.
    Goal labels are driven realistically by logistic(distance) + noise.
    """
    rng = np.random.default_rng(seed)
    match_id = [f"match_{i % n_matches}" for i in range(n)]
    distance = rng.uniform(5.0, 35.0, n)
    angle = rng.uniform(5.0, 60.0, n)

    # Realistic goal probability via logistic
    log_odds = 2.5 - 0.12 * distance + 0.03 * angle
    p_goal = 1 / (1 + np.exp(-log_odds))
    goal = (rng.random(n) < p_goal).astype(int)

    data: dict = {
        "match_id": match_id,
        "goal": goal,
        # Traditional numeric
        "distance_to_goal": distance,
        "shot_angle": angle,
        "x_location": rng.uniform(60.0, 105.0, n),
        "y_location": rng.uniform(10.0, 58.0, n),
        # Traditional boolean
        "header": rng.integers(0, 2, n).astype(float),
        "volley": rng.integers(0, 2, n).astype(float),
        "first_time_shot": rng.integers(0, 2, n).astype(float),
        "open_play": rng.integers(0, 2, n).astype(float),
        "under_pressure": rng.integers(0, 2, n).astype(float),
        # Traditional categorical
        "body_part": rng.choice(["foot", "head", "other"], n),
        "shot_type": rng.choice(["open_play", "free_kick", "corner"], n),
        "set_piece_type": rng.choice(["none", "corner", "free_kick"], n),
        # Contextual numeric
        "opponent_xg_conceded_rolling_5": rng.uniform(0.5, 3.0, n),
        "opponent_shots_conceded_rolling_5": rng.uniform(2.0, 15.0, n),
        "opponent_defensive_rating": rng.uniform(60.0, 90.0, n),
        "opponent_keeper_shot_stopping_rating": rng.uniform(60.0, 90.0, n),
        "opponent_team_strength": rng.uniform(60.0, 90.0, n),
        "minute": rng.integers(1, 90, n).astype(float),
        "score_differential": rng.integers(-3, 4, n).astype(float),
        "events_before_action": rng.integers(0, 20, n).astype(float),
        "passes_before_action": rng.integers(0, 15, n).astype(float),
        "carries_before_action": rng.integers(0, 10, n).astype(float),
        "time_from_possession_start": rng.uniform(0.0, 30.0, n),
        "vertical_progression_speed": rng.uniform(0.0, 3.0, n),
        "directness": rng.uniform(0.0, 1.0, n),
        # Contextual boolean
        "knockout_or_group": rng.integers(0, 2, n).astype(float),
        "set_piece_flag": rng.integers(0, 2, n).astype(float),
        "counterpress_regain_flag": rng.integers(0, 2, n).astype(float),
        # Contextual categorical
        "score_state": rng.choice(["winning", "drawing", "losing"], n),
        "home_or_away": rng.choice(["home", "away"], n),
        "sequence_type": rng.choice(["open", "set_piece", "transition"], n),
        "possession_start_zone": rng.choice(["defensive", "middle", "attacking"], n),
        "transition_or_settled": rng.choice(["transition", "settled"], n),
        # 360 numeric
        "nearest_defender_distance": rng.uniform(0.5, 8.0, n),
        "second_nearest_defender_distance": rng.uniform(1.0, 12.0, n),
        "defenders_within_5m": rng.integers(0, 5, n).astype(float),
        "defenders_between_ball_and_goal": rng.integers(0, 4, n).astype(float),
        "keeper_distance_to_goal": rng.uniform(0.5, 5.0, n),
        "keeper_distance_to_shooter": rng.uniform(2.0, 20.0, n),
        "keeper_angle_coverage": rng.uniform(0.0, 1.0, n),
        "shot_lane_blockage_proxy": rng.uniform(0.0, 1.0, n),
        "defensive_density_in_box": rng.uniform(0.0, 1.0, n),
        # 360 boolean
        "has_360": rng.integers(0, 2, n).astype(float),
    }
    return pd.DataFrame(data)


@pytest.fixture
def shots_df() -> pd.DataFrame:
    return _shots_contextual(n=120, n_matches=6)


@pytest.fixture
def shots_small() -> pd.DataFrame:
    """Smaller fixture for tests that need positives guaranteed."""
    df = _shots_contextual(n=200, n_matches=8, seed=1)
    # Ensure there are positive and negative examples
    if df["goal"].sum() == 0:
        df.loc[0, "goal"] = 1
    if df["goal"].sum() == len(df):
        df.loc[1, "goal"] = 0
    return df


# ── Feature set tests ─────────────────────────────────────────────────────────

class TestFeatureSets:
    def test_registry_accessible(self):
        for name in ("traditional", "contextual", "full_360"):
            fs = get_feature_set(name)
            assert isinstance(fs, FeatureSetSpec)
            assert fs.name == name

    def test_unknown_name_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_feature_set("nonexistent")

    def test_all_features_unique(self):
        for fs in (TRADITIONAL, CONTEXTUAL, FULL_360):
            all_feats = fs.all_features
            assert len(all_feats) == len(set(all_feats)), f"Duplicates in {fs.name}"

    def test_feature_set_ordering(self):
        """Contextual should have more features than traditional."""
        assert len(CONTEXTUAL.all_features) > len(TRADITIONAL.all_features)
        assert len(FULL_360.all_features) > len(CONTEXTUAL.all_features)

    def test_numeric_all_includes_booleans(self):
        for fs in (TRADITIONAL, CONTEXTUAL, FULL_360):
            for b in fs.boolean:
                assert b in fs.numeric_all, f"{b!r} not in numeric_all of {fs.name}"

    def test_requires_360_flag(self):
        assert not TRADITIONAL.requires_360
        assert not CONTEXTUAL.requires_360
        assert FULL_360.requires_360

    def test_frozen_dataclass(self):
        with pytest.raises((TypeError, AttributeError)):
            TRADITIONAL.name = "modified"  # type: ignore[misc]


# ── GLM contextual tests ──────────────────────────────────────────────────────

class TestGlmContextualCxG:
    def test_fit_predict_proba_in_range(self, shots_small):
        model = GlmContextualCxG(feature_set="contextual")
        model.fit(shots_small)
        p = model.predict_proba(shots_small)
        assert p.shape == (len(shots_small),)
        assert (p >= 0).all() and (p <= 1).all()

    def test_evaluate_returns_metrics(self, shots_small):
        model = GlmContextualCxG().fit(shots_small)
        m = model.evaluate(shots_small)
        assert isinstance(m, GlmContextualMetrics)
        assert np.isfinite(m.log_loss)
        assert np.isfinite(m.brier)
        assert m.log_loss > 0
        assert 0 <= m.brier <= 1

    def test_handles_partial_features(self, shots_small):
        """Model must work when some contextual features are absent."""
        drop_cols = ["opponent_xg_conceded_rolling_5", "nearest_defender_distance"]
        partial = shots_small.drop(columns=[c for c in drop_cols if c in shots_small.columns])
        model = GlmContextualCxG().fit(partial)
        p = model.predict_proba(partial)
        assert (p >= 0).all() and (p <= 1).all()

    def test_traditional_feature_set(self, shots_small):
        """GLM can also be used with the TRADITIONAL feature set."""
        model = GlmContextualCxG(feature_set="traditional").fit(shots_small)
        p = model.predict_proba(shots_small)
        assert p.shape == (len(shots_small),)

    def test_empty_df_raises(self):
        model = GlmContextualCxG()
        with pytest.raises(ValueError, match="empty"):
            model.fit(pd.DataFrame())

    def test_missing_target_raises(self, shots_small):
        model = GlmContextualCxG()
        with pytest.raises(ValueError, match="target"):
            model.fit(shots_small.drop(columns=["goal"]))

    def test_predict_before_fit_raises(self, shots_small):
        model = GlmContextualCxG()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(shots_small)

    def test_save_and_load(self, shots_small, tmp_path):
        model = GlmContextualCxG().fit(shots_small)
        path = tmp_path / "glm.pkl"
        model.save(path)
        loaded = GlmContextualCxG.load(path)
        p_orig = model.predict_proba(shots_small)
        p_load = loaded.predict_proba(shots_small)
        np.testing.assert_array_almost_equal(p_orig, p_load)


# ── XGBoost tests ─────────────────────────────────────────────────────────────

class TestXGBoostCxGModel:
    def test_fit_predict_proba_in_range(self, shots_small):
        model = XGBoostCxGModel(feature_set="contextual", n_estimators=30)
        model.fit(shots_small)
        p = model.predict_proba(shots_small)
        assert p.shape == (len(shots_small),)
        assert (p >= 0).all() and (p <= 1).all()

    def test_evaluate_metrics(self, shots_small):
        model = XGBoostCxGModel(n_estimators=30).fit(shots_small)
        m = model.evaluate(shots_small)
        assert isinstance(m, XGBCxGMetrics)
        assert np.isfinite(m.log_loss)
        assert np.isfinite(m.brier)

    def test_traditional_feature_set(self, shots_small):
        model = XGBoostCxGModel(feature_set="traditional", n_estimators=30).fit(shots_small)
        p = model.predict_proba(shots_small)
        assert p.shape == (len(shots_small),)

    def test_handles_partial_features(self, shots_small):
        partial = shots_small.drop(columns=["opponent_xg_conceded_rolling_5"])
        model = XGBoostCxGModel(n_estimators=30).fit(partial)
        p = model.predict_proba(partial)
        assert (p >= 0).all() and (p <= 1).all()

    def test_empty_df_raises(self):
        model = XGBoostCxGModel(n_estimators=30)
        with pytest.raises(ValueError, match="empty"):
            model.fit(pd.DataFrame())

    def test_predict_before_fit_raises(self, shots_small):
        model = XGBoostCxGModel(n_estimators=30)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(shots_small)

    def test_save_and_load(self, shots_small, tmp_path):
        model = XGBoostCxGModel(n_estimators=30).fit(shots_small)
        path = tmp_path / "xgb.pkl"
        model.save(path)
        loaded = XGBoostCxGModel.load(path)
        p_orig = model.predict_proba(shots_small)
        p_load = loaded.predict_proba(shots_small)
        np.testing.assert_array_almost_equal(p_orig, p_load)


# ── LightGBM tests ────────────────────────────────────────────────────────────

class TestLightGBMCxGModel:
    def test_fit_predict_proba_in_range(self, shots_small):
        model = LightGBMCxGModel(feature_set="contextual", n_estimators=30)
        model.fit(shots_small)
        p = model.predict_proba(shots_small)
        assert p.shape == (len(shots_small),)
        assert (p >= 0).all() and (p <= 1).all()

    def test_evaluate_metrics(self, shots_small):
        model = LightGBMCxGModel(n_estimators=30).fit(shots_small)
        m = model.evaluate(shots_small)
        assert isinstance(m, LGBMCxGMetrics)
        assert np.isfinite(m.log_loss)
        assert np.isfinite(m.brier)

    def test_traditional_feature_set(self, shots_small):
        model = LightGBMCxGModel(feature_set="traditional", n_estimators=30).fit(shots_small)
        p = model.predict_proba(shots_small)
        assert p.shape == (len(shots_small),)

    def test_handles_partial_features(self, shots_small):
        partial = shots_small.drop(columns=["score_differential"])
        model = LightGBMCxGModel(n_estimators=30).fit(partial)
        p = model.predict_proba(partial)
        assert (p >= 0).all() and (p <= 1).all()

    def test_empty_df_raises(self):
        model = LightGBMCxGModel(n_estimators=30)
        with pytest.raises(ValueError, match="empty"):
            model.fit(pd.DataFrame())

    def test_predict_before_fit_raises(self, shots_small):
        model = LightGBMCxGModel(n_estimators=30)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(shots_small)

    def test_save_and_load(self, shots_small, tmp_path):
        model = LightGBMCxGModel(n_estimators=30).fit(shots_small)
        path = tmp_path / "lgbm.pkl"
        model.save(path)
        loaded = LightGBMCxGModel.load(path)
        p_orig = model.predict_proba(shots_small)
        p_load = loaded.predict_proba(shots_small)
        np.testing.assert_array_almost_equal(p_orig, p_load)


# ── CxG Ladder tests ──────────────────────────────────────────────────────────

@pytest.fixture
def ladder_shots() -> pd.DataFrame:
    """Larger fixture for ladder tests (needs enough data per fold)."""
    df = _shots_contextual(n=360, n_matches=18, seed=2)
    # Guarantee both classes present
    if df["goal"].sum() == 0:
        df.loc[0, "goal"] = 1
    if df["goal"].sum() == len(df):
        df.loc[1, "goal"] = 0
    return df


class TestCxGLadder:
    def test_end_to_end(self, ladder_shots):
        """Ladder runs all 6 standard candidates and returns sorted results."""
        ladder = CxGLadder()
        results = ladder.run(
            ladder_shots,
            n_folds=3,
            n_estimators=30,
        )
        assert len(results) == 6
        # Results must be sorted ascending by cv_log_loss
        ll_values = [r.cv_log_loss for r in results]
        assert ll_values == sorted(ll_values)

    def test_leaderboard_has_all_candidates(self, ladder_shots):
        ladder = CxGLadder()
        ladder.run(ladder_shots, n_folds=3, n_estimators=30)
        lb = ladder.leaderboard()
        expected_names = {
            "baseline_logit", "glm_contextual",
            "xgb_traditional", "xgb_contextual",
            "lgbm_traditional", "lgbm_contextual",
        }
        assert set(lb["name"]) == expected_names

    def test_leaderboard_sorted_by_log_loss(self, ladder_shots):
        ladder = CxGLadder()
        ladder.run(ladder_shots, n_folds=3, n_estimators=30)
        lb = ladder.leaderboard()
        ll = lb["cv_log_loss"].tolist()
        assert ll == sorted(ll)

    def test_best_is_first_in_leaderboard(self, ladder_shots):
        ladder = CxGLadder()
        ladder.run(ladder_shots, n_folds=3, n_estimators=30)
        best = ladder.best()
        lb = ladder.leaderboard()
        assert best.name == lb.iloc[0]["name"]
        assert best.rank == 1

    def test_include_360_adds_candidates(self, ladder_shots):
        ladder = CxGLadder()
        results = ladder.run(
            ladder_shots,
            n_folds=3,
            n_estimators=30,
            include_360=True,
        )
        assert len(results) == 8
        names = {r.name for r in results}
        assert "xgb_full_360" in names
        assert "lgbm_full_360" in names

    def test_run_before_leaderboard_raises(self):
        ladder = CxGLadder()
        with pytest.raises(RuntimeError, match="run"):
            ladder.leaderboard()

    def test_run_before_best_raises(self):
        ladder = CxGLadder()
        with pytest.raises(RuntimeError, match="run"):
            ladder.best()

    def test_model_instances_are_fitted(self, ladder_shots):
        """Each result's model must be able to generate predictions."""
        ladder = CxGLadder()
        results = ladder.run(ladder_shots, n_folds=3, n_estimators=30)
        for r in results:
            p = r.model.predict_proba(ladder_shots)
            assert p.shape == (len(ladder_shots),)
            assert (p >= 0).all() and (p <= 1).all()

    def test_metrics_are_finite(self, ladder_shots):
        ladder = CxGLadder()
        results = ladder.run(ladder_shots, n_folds=3, n_estimators=30)
        for r in results:
            assert np.isfinite(r.cv_log_loss), f"{r.name}: non-finite log_loss"
            assert np.isfinite(r.cv_brier), f"{r.name}: non-finite brier"
