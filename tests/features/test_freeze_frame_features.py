from __future__ import annotations

import pandas as pd

from src.features.freeze_frame_features import build_freeze_frame_features


def test_freeze_frame_features_basic_counts() -> None:
    events = pd.DataFrame(
        [
            {
                "internal_id": "e1",
                "match_internal_id": "m1",
                "possession_internal_id": "p1",
                "event_type": "shot",
                "index": 1,
                "x": 90.0,
                "y": 34.0,
                "has_360": True,
            }
        ]
    )
    frames = pd.DataFrame(
        [
            {"event_internal_id": "e1", "teammate": False, "keeper": False, "x": 91.0, "y": 34.0},
            {"event_internal_id": "e1", "teammate": False, "keeper": True, "x": 102.0, "y": 34.0},
            {"event_internal_id": "e1", "teammate": True, "keeper": False, "x": 95.0, "y": 30.0},
        ]
    )

    out = build_freeze_frame_features(events, frames)
    assert len(out) == 1
    assert out.iloc[0]["defenders_within_3m"] >= 1
    assert (
        out.iloc[0]["nearest_defender_distance"] <= out.iloc[0]["second_nearest_defender_distance"]
    )


def test_freeze_frame_features_empty_frames_keeps_row() -> None:
    events = pd.DataFrame(
        [
            {
                "internal_id": "e1",
                "match_internal_id": "m1",
                "possession_internal_id": "p1",
                "event_type": "pass",
                "index": 1,
                "x": 60.0,
                "y": 20.0,
                "has_360": False,
            }
        ]
    )
    out = build_freeze_frame_features(events, pd.DataFrame())
    assert len(out) == 1
    assert bool(out.iloc[0]["has_360"]) is False
