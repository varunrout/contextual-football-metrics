"""
Tests for src/features/sequence_features.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.sequence_features import (
    _ATK_THIRD_MIN_X,
    _DEF_THIRD_MAX_X,
    _compute_single_possession_features,
    compute_sequence_features,
    is_central,
    is_in_box,
    is_wide_byline,
    zone_from_x,
)

# ── Zone helpers ─────────────────────────────────────────────────────────────


class TestZoneHelpers:
    def test_defensive_third(self):
        assert zone_from_x(10.0) == "defensive_third"

    def test_mid_third(self):
        assert zone_from_x(52.5) == "mid_third"

    def test_attacking_third(self):
        assert zone_from_x(90.0) == "attacking_third"

    def test_boundary_def(self):
        assert zone_from_x(_DEF_THIRD_MAX_X) == "defensive_third"

    def test_boundary_atk(self):
        assert zone_from_x(_ATK_THIRD_MIN_X) == "attacking_third"

    def test_is_central_true(self):
        assert is_central(34.0) is True

    def test_is_central_false(self):
        assert is_central(5.0) is False

    def test_is_in_box_true(self):
        assert is_in_box(95.0, 34.0) is True

    def test_is_in_box_x_too_small(self):
        assert is_in_box(70.0, 34.0) is False

    def test_is_wide_byline_true(self):
        assert is_wide_byline(90.0, 4.0) is True

    def test_is_wide_byline_not_byline(self):
        assert is_wide_byline(90.0, 34.0) is False


# ── Single possession feature computation ─────────────────────────────────────


def _make_poss(
    *,
    start_x: float = 30.0,
    start_y: float = 34.0,
    vertical_progression: float = 20.0,
    n_events: int = 5,
    n_passes: int = 3,
    n_carries: int = 2,
    start_timestamp: float = 0.0,
    end_timestamp: float = 10.0,
    set_piece_flag: bool = False,
    counterpress_regain_flag: bool = False,
    distance_progressed: float = 25.0,
    regain_zone: str = "mid_third",
    **kwargs,
) -> pd.Series:
    data = dict(
        start_x=start_x,
        start_y=start_y,
        vertical_progression=vertical_progression,
        n_events=n_events,
        n_passes=n_passes,
        n_carries=n_carries,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        set_piece_flag=set_piece_flag,
        counterpress_regain_flag=counterpress_regain_flag,
        distance_progressed=distance_progressed,
        regain_zone=regain_zone,
        **kwargs,
    )
    return pd.Series(data)


class TestComputeSinglePossessionFeatures:
    def test_duration_computed(self):
        poss = _make_poss(start_timestamp=0.0, end_timestamp=8.0)
        feats = _compute_single_possession_features(poss, pd.DataFrame())
        assert feats["time_from_possession_start"] == pytest.approx(8.0)

    def test_directness_capped_at_one(self):
        poss = _make_poss(vertical_progression=30.0, distance_progressed=5.0)
        feats = _compute_single_possession_features(poss, pd.DataFrame())
        assert feats["directness"] <= 1.0

    def test_directness_positive(self):
        poss = _make_poss(vertical_progression=20.0, distance_progressed=40.0)
        feats = _compute_single_possession_features(poss, pd.DataFrame())
        assert feats["directness"] > 0.0

    def test_possession_speed_nonzero(self):
        poss = _make_poss(n_events=10, start_timestamp=0.0, end_timestamp=10.0)
        feats = _compute_single_possession_features(poss, pd.DataFrame())
        assert feats["possession_speed"] == pytest.approx(1.0)

    def test_start_zone_defensive(self):
        poss = _make_poss(start_x=10.0)
        feats = _compute_single_possession_features(poss, pd.DataFrame())
        assert feats["possession_start_zone"] == "defensive_third"

    def test_transition_flag_counterpress(self):
        poss = _make_poss(counterpress_regain_flag=True)
        feats = _compute_single_possession_features(poss, pd.DataFrame())
        assert feats["transition_or_settled"] == "transition"

    def test_settled_when_low_speed(self):
        poss = _make_poss(
            vertical_progression=5.0,
            start_timestamp=0.0,
            end_timestamp=30.0,
            counterpress_regain_flag=False,
        )
        feats = _compute_single_possession_features(poss, pd.DataFrame())
        assert feats["transition_or_settled"] == "settled"

    def test_switches_zero_empty_events(self):
        poss = _make_poss()
        feats = _compute_single_possession_features(poss, pd.DataFrame())
        assert feats["number_of_switches"] == 0

    def test_final_action_unknown_empty_events(self):
        poss = _make_poss()
        feats = _compute_single_possession_features(poss, pd.DataFrame())
        assert feats["final_action_type"] == "unknown"


# ── compute_sequence_features (DataFrame-level) ───────────────────────────────


class TestComputeSequenceFeatures:
    def test_empty_possessions_returns_empty(self):
        result = compute_sequence_features(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_columns_added(self):
        poss_df = pd.DataFrame(
            [
                {
                    "match_internal_id": "abc",
                    "possession_index": 1,
                    "start_x": 30.0,
                    "start_y": 34.0,
                    "vertical_progression": 15.0,
                    "n_events": 4,
                    "n_passes": 2,
                    "n_carries": 2,
                    "start_timestamp": 0.0,
                    "end_timestamp": 12.0,
                    "set_piece_flag": False,
                    "counterpress_regain_flag": False,
                    "distance_progressed": 20.0,
                    "regain_zone": "mid_third",
                }
            ]
        )
        events_df = pd.DataFrame()
        result = compute_sequence_features(poss_df, events_df)
        expected_cols = [
            "time_from_possession_start",
            "directness",
            "possession_speed",
            "number_of_switches",
            "possession_start_zone",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self):
        poss_df = pd.DataFrame(
            [
                {
                    "match_internal_id": "abc",
                    "possession_index": 1,
                    "start_x": 50.0,
                    "start_y": 34.0,
                    "vertical_progression": 10.0,
                    "n_events": 3,
                    "n_passes": 1,
                    "n_carries": 2,
                    "start_timestamp": 0.0,
                    "end_timestamp": 8.0,
                    "set_piece_flag": False,
                    "counterpress_regain_flag": False,
                    "distance_progressed": 15.0,
                    "regain_zone": "mid_third",
                }
            ]
        )
        result = compute_sequence_features(poss_df, pd.DataFrame())
        assert "match_internal_id" in result.columns
        assert "possession_index" in result.columns
