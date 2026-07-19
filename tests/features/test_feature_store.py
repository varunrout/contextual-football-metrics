from __future__ import annotations

import pandas as pd

from src.features.feature_store import build_feature_store


def _events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "internal_id": "e1",
                "match_internal_id": "m1",
                "possession_internal_id": "p1",
                "team_internal_id": "t1",
                "player_internal_id": "pl1",
                "event_type": "pass",
                "index": 1,
                "period": 1,
                "timestamp": 10.0,
                "x": 60.0,
                "y": 30.0,
                "end_x": 90.0,
                "end_y": 34.0,
                "length": 30.0,
                "angle": 0.2,
                "under_pressure": False,
                "cross": False,
                "cutback": False,
                "through_ball": True,
                "switch": False,
                "height": "ground",
                "set_piece_type": "none",
                "has_360": True,
            },
            {
                "internal_id": "e2",
                "match_internal_id": "m1",
                "possession_internal_id": "p1",
                "team_internal_id": "t2",
                "player_internal_id": "pl2",
                "event_type": "shot",
                "index": 2,
                "period": 1,
                "timestamp": 20.0,
                "x": 95.0,
                "y": 34.0,
                "goal": True,
                "statsbomb_xg": 0.3,
                "under_pressure": True,
                "has_360": True,
            },
        ]
    )


def _possessions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "internal_id": "p1",
                "match_internal_id": "m1",
                "team_internal_id": "t1",
                "possession_index": 1,
                "start_event_internal_id": "e1",
                "end_event_internal_id": "e2",
                "start_timestamp": 10.0,
                "end_timestamp": 20.0,
                "start_x": 60.0,
                "start_y": 30.0,
                "regain_zone": "mid_third",
                "n_events": 2,
                "n_passes": 1,
                "n_carries": 0,
                "n_shots": 1,
                "vertical_progression": 35.0,
                "distance_progressed": 35.0,
                "set_piece_flag": False,
                "counterpress_regain_flag": False,
                "sequence_type": "through_ball_sequence",
                "sequence_type_confidence": 0.8,
            }
        ]
    )


def _frames_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_internal_id": "e1",
                "match_internal_id": "m1",
                "teammate": False,
                "keeper": False,
                "x": 70.0,
                "y": 30.0,
            },
            {
                "event_internal_id": "e1",
                "match_internal_id": "m1",
                "teammate": False,
                "keeper": True,
                "x": 102.0,
                "y": 34.0,
            },
            {
                "event_internal_id": "e1",
                "match_internal_id": "m1",
                "teammate": True,
                "keeper": False,
                "x": 80.0,
                "y": 32.0,
            },
        ]
    )


def _matches_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "internal_id": "m1",
                "home_team_internal_id": "t1",
                "away_team_internal_id": "t2",
                "match_date": "2024-06-15",
                "stage": "Group Stage",
            }
        ]
    )


def test_build_feature_store_contains_identifier_columns() -> None:
    out = build_feature_store(_events_df(), _possessions_df(), _frames_df(), _matches_df())
    for col in ["event_id", "match_id", "possession_id", "team_id", "player_id", "opponent_id"]:
        assert col in out.columns


def test_build_feature_store_applies_flag_and_zero_policy() -> None:
    out = build_feature_store(_events_df(), _possessions_df(), _frames_df(), _matches_df())
    # These are freeze-frame features with flag_and_zero policy.
    assert "has_nearest_defender_distance" in out.columns
    assert "nearest_defender_distance" in out.columns


def test_build_feature_store_has_sequence_features() -> None:
    out = build_feature_store(_events_df(), _possessions_df(), _frames_df(), _matches_df())
    assert "sequence_type" in out.columns
    assert "time_from_possession_start" in out.columns
