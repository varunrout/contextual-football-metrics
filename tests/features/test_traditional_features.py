from __future__ import annotations

import pandas as pd

from src.features.traditional_features import build_traditional_features


def test_build_traditional_features_has_expected_columns() -> None:
    events = pd.DataFrame(
        [
            {
                "internal_id": "e1",
                "match_internal_id": "m1",
                "possession_internal_id": "p1",
                "team_internal_id": "t1",
                "player_internal_id": "pl1",
                "event_type": "shot",
                "x": 100.0,
                "y": 34.0,
                "under_pressure": True,
                "body_part": "foot",
                "shot_type": "none",
                "first_time": False,
                "header": False,
                "volley": False,
                "open_play": True,
            },
            {
                "internal_id": "e2",
                "match_internal_id": "m1",
                "possession_internal_id": "p1",
                "team_internal_id": "t1",
                "player_internal_id": "pl2",
                "event_type": "pass",
                "x": 70.0,
                "y": 30.0,
                "end_x": 90.0,
                "end_y": 34.0,
                "length": 22.0,
                "angle": 0.1,
                "cross": False,
                "cutback": True,
                "through_ball": False,
                "switch": False,
                "height": "ground",
                "set_piece_type": "none",
            },
        ]
    )

    out = build_traditional_features(events)
    assert len(out) == 2
    assert "distance_to_goal" in out.columns
    assert "shot_angle" in out.columns
    assert "assist_type" in out.columns
    assert "open_play_or_set_piece" in out.columns


def test_pass_cutback_assist_type() -> None:
    events = pd.DataFrame(
        [
            {
                "internal_id": "e2",
                "event_type": "pass",
                "x": 70.0,
                "y": 30.0,
                "end_x": 90.0,
                "end_y": 34.0,
                "cutback": True,
                "cross": False,
                "through_ball": False,
                "set_piece_type": "none",
            }
        ]
    )
    out = build_traditional_features(events)
    assert out.iloc[0]["assist_type"] == "cutback"


def test_shot_distance_lower_near_goal() -> None:
    events = pd.DataFrame(
        [
            {"internal_id": "e1", "event_type": "shot", "x": 103.0, "y": 34.0},
            {"internal_id": "e2", "event_type": "shot", "x": 70.0, "y": 34.0},
        ]
    )
    out = build_traditional_features(events)
    assert out.iloc[0]["distance_to_goal"] < out.iloc[1]["distance_to_goal"]
