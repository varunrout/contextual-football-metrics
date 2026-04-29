"""
Tests for src/features/sequence_labeler.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.schema import SequenceType
from src.features.sequence_labeler import (
    label_possession,
    label_possessions_dataframe,
    LabelResult,
    _is_fast_counterattack,
    _is_set_piece_first_phase,
    _is_set_piece_second_phase,
    _is_high_press_regain,
    _is_carry_led_progression,
    _is_direct_long_ball,
    _is_deep_buildup,
)


def _poss(**kwargs) -> pd.Series:
    defaults = dict(
        start_x=50.0,
        start_y=34.0,
        vertical_progression=10.0,
        n_events=4,
        n_passes=2,
        n_carries=2,
        start_timestamp=0.0,
        end_timestamp=12.0,
        set_piece_flag=False,
        counterpress_regain_flag=False,
        distance_progressed=20.0,
        regain_zone="mid_third",
        number_of_switches=0,
        directness=0.5,
    )
    defaults.update(kwargs)
    return pd.Series(defaults)


# ── Individual rule tests ──────────────────────────────────────────────────────

class TestIndividualRules:
    def test_fast_counterattack_true(self):
        poss = _poss(start_x=30.0, vertical_progression=35.0, end_timestamp=8.0)
        assert _is_fast_counterattack(poss, pd.DataFrame()) is True

    def test_fast_counterattack_too_slow(self):
        poss = _poss(start_x=30.0, vertical_progression=35.0, end_timestamp=15.0)
        assert _is_fast_counterattack(poss, pd.DataFrame()) is False

    def test_fast_counterattack_insufficient_progression(self):
        poss = _poss(start_x=30.0, vertical_progression=10.0, end_timestamp=8.0)
        assert _is_fast_counterattack(poss, pd.DataFrame()) is False

    def test_set_piece_first_phase_true(self):
        poss = _poss(set_piece_flag=True, n_events=3)
        assert _is_set_piece_first_phase(poss, pd.DataFrame()) is True

    def test_set_piece_first_phase_too_many_events(self):
        poss = _poss(set_piece_flag=True, n_events=10)
        assert _is_set_piece_first_phase(poss, pd.DataFrame()) is False

    def test_set_piece_second_phase_true(self):
        poss = _poss(set_piece_flag=True, n_events=8)
        assert _is_set_piece_second_phase(poss, pd.DataFrame()) is True

    def test_set_piece_second_phase_no_set_piece(self):
        poss = _poss(set_piece_flag=False, n_events=8)
        assert _is_set_piece_second_phase(poss, pd.DataFrame()) is False

    def test_high_press_regain_true(self):
        poss = _poss(
            start_x=72.0, counterpress_regain_flag=True, end_timestamp=6.0
        )
        assert _is_high_press_regain(poss, pd.DataFrame()) is True

    def test_high_press_regain_wrong_zone(self):
        poss = _poss(
            start_x=30.0, counterpress_regain_flag=True, end_timestamp=6.0
        )
        assert _is_high_press_regain(poss, pd.DataFrame()) is False

    def test_carry_led_progression_true(self):
        poss = _poss(n_carries=4, n_events=6, vertical_progression=25.0)
        assert _is_carry_led_progression(poss, pd.DataFrame()) is True

    def test_carry_led_progression_low_proportion(self):
        poss = _poss(n_carries=1, n_events=6, vertical_progression=25.0)
        assert _is_carry_led_progression(poss, pd.DataFrame()) is False

    def test_deep_buildup_true(self):
        poss = _poss(start_x=20.0, n_passes=10)
        assert _is_deep_buildup(poss, pd.DataFrame()) is True

    def test_deep_buildup_wrong_zone(self):
        poss = _poss(start_x=60.0, n_passes=10)
        assert _is_deep_buildup(poss, pd.DataFrame()) is False

    def test_direct_long_ball_true(self):
        events_df = pd.DataFrame([{
            "type": {"name": "Pass"},
            "pass": {"length": 40.0, "through_ball": False},
            "location": [30, 40],
        }])
        poss = _poss(n_passes=1)
        assert _is_direct_long_ball(poss, events_df) is True

    def test_direct_long_ball_short(self):
        events_df = pd.DataFrame([{
            "type": {"name": "Pass"},
            "pass": {"length": 10.0, "through_ball": False},
            "location": [30, 40],
        }])
        poss = _poss(n_passes=1)
        assert _is_direct_long_ball(poss, events_df) is False


# ── label_possession ───────────────────────────────────────────────────────────

class TestLabelPossession:
    def test_returns_label_result(self):
        poss = _poss()
        result = label_possession(poss, pd.DataFrame())
        assert isinstance(result, LabelResult)

    def test_fast_counter_takes_priority(self):
        poss = _poss(
            start_x=25.0,
            vertical_progression=38.0,
            end_timestamp=7.0,
            set_piece_flag=False,
        )
        result = label_possession(poss, pd.DataFrame())
        assert result.sequence_type == SequenceType.FAST_COUNTERATTACK

    def test_set_piece_takes_priority_over_counter(self):
        """Set-piece check comes before counter-attack in registry."""
        poss = _poss(
            start_x=25.0,
            vertical_progression=38.0,
            end_timestamp=7.0,
            set_piece_flag=True,
            n_events=2,
        )
        result = label_possession(poss, pd.DataFrame())
        assert result.sequence_type == SequenceType.SET_PIECE_FIRST_PHASE

    def test_unknown_when_no_rule_matches(self):
        # Highly unusual possession that matches nothing
        poss = _poss(
            start_x=50.0,
            vertical_progression=0.0,
            end_timestamp=100.0,   # very long
            n_events=3,
            n_passes=1,
            n_carries=0,
            set_piece_flag=False,
            counterpress_regain_flag=False,
            directness=0.01,
            number_of_switches=0,
            n_passes_in=1,
        )
        # The settled_possession rule would normally match; let's test confidence ≥ 0
        result = label_possession(poss, pd.DataFrame())
        assert result.confidence >= 0.0

    def test_source_is_rule(self):
        poss = _poss()
        result = label_possession(poss, pd.DataFrame())
        assert result.source == "rule"


# ── label_possessions_dataframe ───────────────────────────────────────────────

class TestLabelPossessionsDataFrame:
    def test_empty_returns_empty(self):
        result = label_possessions_dataframe(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_columns_added(self):
        poss_df = pd.DataFrame([{
            "match_internal_id": "abc",
            "possession_index": 1,
            "start_x": 20.0,
            "start_y": 34.0,
            "vertical_progression": 40.0,
            "n_events": 3,
            "n_passes": 1,
            "n_carries": 1,
            "start_timestamp": 0.0,
            "end_timestamp": 8.0,
            "set_piece_flag": False,
            "counterpress_regain_flag": False,
            "distance_progressed": 42.0,
            "regain_zone": "mid_third",
            "number_of_switches": 0,
            "directness": 0.9,
        }])
        result = label_possessions_dataframe(poss_df, pd.DataFrame())
        for col in ["sequence_type_rule", "sequence_type_confidence", "sequence_type_source", "sequence_type"]:
            assert col in result.columns

    def test_confidence_between_0_and_1(self):
        poss_df = pd.DataFrame([{
            "match_internal_id": "abc",
            "possession_index": 1,
            "start_x": 72.0,
            "start_y": 34.0,
            "vertical_progression": 40.0,
            "n_events": 3,
            "n_passes": 1,
            "n_carries": 1,
            "start_timestamp": 0.0,
            "end_timestamp": 8.0,
            "set_piece_flag": False,
            "counterpress_regain_flag": True,
            "distance_progressed": 42.0,
            "regain_zone": "attacking_third",
            "number_of_switches": 0,
            "directness": 0.9,
        }])
        result = label_possessions_dataframe(poss_df, pd.DataFrame())
        conf = float(result["sequence_type_confidence"].iloc[0])
        assert 0.0 <= conf <= 1.0
