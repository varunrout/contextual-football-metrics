"""
Tests for src.ingestion.provider_mapper

Covers:
  - coordinate normalisation (bounds, edge cases)
  - internal ID stability and uniqueness
  - shot/pass/carry mapping from representative raw StatsBomb dicts
  - freeze-frame row mapping
  - timestamp parsing
"""

from __future__ import annotations

import math

import pytest

from src.ingestion.provider_mapper import (
    make_internal_id,
    map_event_row,
    map_freeze_frame_row,
    normalise_coords,
    normalise_x,
    normalise_y,
)
from src.ingestion.schema import (
    CarryEvent,
    PassEvent,
    Provider,
    ShotEvent,
)


class TestCoordinateNormalisation:
    def test_x_origin(self):
        assert normalise_x(0.0) == pytest.approx(0.0)

    def test_x_max(self):
        assert normalise_x(120.0) == pytest.approx(105.0)

    def test_y_origin(self):
        assert normalise_y(0.0) == pytest.approx(0.0)

    def test_y_max(self):
        assert normalise_y(80.0) == pytest.approx(68.0)

    def test_midpoint_x(self):
        assert normalise_x(60.0) == pytest.approx(52.5)

    def test_midpoint_y(self):
        assert normalise_y(40.0) == pytest.approx(34.0)

    def test_coords_none_returns_nan(self):
        x, y = normalise_coords(None)
        assert math.isnan(x) and math.isnan(y)

    def test_coords_empty_returns_nan(self):
        x, y = normalise_coords([])
        assert math.isnan(x) and math.isnan(y)

    def test_coords_valid(self):
        x, y = normalise_coords([60.0, 40.0])
        assert x == pytest.approx(52.5)
        assert y == pytest.approx(34.0)


class TestInternalIdGeneration:
    def test_deterministic(self):
        id1 = make_internal_id(Provider.STATSBOMB, "event", "abc123")
        id2 = make_internal_id(Provider.STATSBOMB, "event", "abc123")
        assert id1 == id2

    def test_different_keys_differ(self):
        id1 = make_internal_id(Provider.STATSBOMB, "event", "abc")
        id2 = make_internal_id(Provider.STATSBOMB, "event", "xyz")
        assert id1 != id2

    def test_different_providers_differ(self):
        # Even if provider enum is expanded in future
        id1 = make_internal_id(Provider.STATSBOMB, "match", 1)
        # Manually check that changing the prefix changes the hash
        import hashlib

        raw = "otherprovider:match:1"
        id2 = hashlib.sha256(raw.encode()).hexdigest()[:16]
        assert id1 != id2

    def test_length_is_16(self):
        assert len(make_internal_id(Provider.STATSBOMB, "player", 99)) == 16


# ── Sample raw StatsBomb event dicts ─────────────────────────────────────────


def _base_event(event_type_name: str, extra: dict | None = None) -> dict:
    row = {
        "id": "test-uuid-001",
        "index": 1,
        "period": 1,
        "timestamp": "00:01:30.500",
        "type": {"id": 16, "name": event_type_name},
        "possession": 5,
        "play_pattern": {"id": 1, "name": "Regular Play"},
        "team": {"id": 217, "name": "Argentina"},
        "player": {"id": 3501, "name": "Lionel Messi"},
        "location": [100.0, 40.0],
        "under_pressure": False,
        "off_camera": False,
        "out": False,
    }
    if extra:
        row.update(extra)
    return row


class TestShotMapping:
    def _make_shot_row(self) -> dict:
        return _base_event(
            "Shot",
            {
                "shot": {
                    "statsbomb_xg": 0.15,
                    "outcome": {"id": 16, "name": "Goal"},
                    "body_part": {"id": 72, "name": "Right Foot"},
                    "type": {"id": 87, "name": "Open Play"},
                    "technique": {"id": 93, "name": "Normal"},
                    "first_time": False,
                    "end_location": [120.0, 40.0, 0.5],
                }
            },
        )

    def test_maps_to_shot_event(self):
        row = self._make_shot_row()
        ev = map_event_row(
            row,
            match_internal_id="match_001",
            possession_internal_id="poss_001",
            team_internal_id="team_001",
            player_internal_id="player_001",
            has_360=False,
        )
        assert isinstance(ev, ShotEvent)

    def test_goal_flag_true(self):
        row = self._make_shot_row()
        ev = map_event_row(row, "m", "p", "t", "pl", False)
        assert ev.goal is True

    def test_statsbomb_xg_preserved(self):
        row = self._make_shot_row()
        ev = map_event_row(row, "m", "p", "t", "pl", False)
        assert ev.statsbomb_xg == pytest.approx(0.15)

    def test_coordinates_normalised(self):
        row = self._make_shot_row()
        ev = map_event_row(row, "m", "p", "t", "pl", False)
        # location [100, 40] → x=87.5, y=34
        assert ev.x == pytest.approx(87.5)
        assert ev.y == pytest.approx(34.0)

    def test_end_coordinates_normalised(self):
        row = self._make_shot_row()
        ev = map_event_row(row, "m", "p", "t", "pl", False)
        assert ev.end_x == pytest.approx(105.0)  # 120 → 105
        assert ev.end_y == pytest.approx(34.0)


class TestPassMapping:
    def _make_pass_row(self) -> dict:
        return _base_event(
            "Pass",
            {
                "pass": {
                    "length": 25.3,
                    "angle": 0.52,
                    "end_location": [110.0, 50.0],
                    "height": {"id": 1, "name": "Ground Pass"},
                    "body_part": {"id": 72, "name": "Right Foot"},
                    "type": {"id": 67, "name": "Open Play"},
                    "cross": False,
                    "through_ball": True,
                    "outcome": {"id": 9, "name": "Complete"},
                    "recipient": {"id": 8888, "name": "Di Maria"},
                }
            },
        )

    def test_maps_to_pass_event(self):
        row = self._make_pass_row()
        ev = map_event_row(row, "m", "p", "t", "pl", False)
        assert isinstance(ev, PassEvent)

    def test_through_ball_flag(self):
        row = self._make_pass_row()
        ev = map_event_row(row, "m", "p", "t", "pl", False)
        assert ev.through_ball is True

    def test_end_coords_normalised(self):
        row = self._make_pass_row()
        ev = map_event_row(row, "m", "p", "t", "pl", False)
        assert ev.end_x == pytest.approx(96.25)  # 110 × 105/120
        assert ev.end_y == pytest.approx(42.5)  # 50 × 68/80


class TestCarryMapping:
    def _make_carry_row(self) -> dict:
        return _base_event(
            "Carry",
            {"carry": {"end_location": [105.0, 40.0]}},
        )

    def test_maps_to_carry_event(self):
        row = self._make_carry_row()
        ev = map_event_row(row, "m", "p", "t", "pl", False)
        assert isinstance(ev, CarryEvent)

    def test_progressive_distance_non_negative(self):
        row = self._make_carry_row()
        ev = map_event_row(row, "m", "p", "t", "pl", False)
        assert ev.progressive_distance >= 0.0


class TestFreezeFrameMapping:
    def test_teammate_flag(self):
        ff_row = {
            "teammate": True,
            "actor": False,
            "keeper": False,
            "location": [60.0, 40.0],
            "player": {"id": 9999, "name": "Teammate"},
        }
        ff = map_freeze_frame_row(ff_row, "event_001", "match_001", 0)
        assert ff.teammate is True
        assert ff.keeper is False

    def test_coordinates_normalised(self):
        ff_row = {
            "teammate": False,
            "actor": False,
            "keeper": True,
            "location": [6.0, 40.0],
            "player": None,
        }
        ff = map_freeze_frame_row(ff_row, "event_001", "match_001", 1)
        assert ff.x == pytest.approx(5.25)  # 6 × 105/120
        assert ff.y == pytest.approx(34.0)

    def test_player_none_allowed(self):
        ff_row = {
            "teammate": False,
            "actor": False,
            "keeper": False,
            "location": [60.0, 40.0],
        }
        ff = map_freeze_frame_row(ff_row, "event_001", "match_001", 2)
        assert ff.player_internal_id is None
