"""
Tests for src.ingestion.possession_builder
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.possession_builder import build_possessions, possessions_to_dataframe
from src.ingestion.schema import SequenceType


def _make_events(rows: list[dict]) -> pd.DataFrame:
    """Helper to build a minimal StatsBomb-style events DataFrame."""
    return pd.DataFrame(rows)


def _team(team_id: int, name: str) -> dict:
    return {"id": team_id, "name": name}


def _type(name: str) -> dict:
    return {"id": 1, "name": name}


def _play_pattern(name: str = "Regular Play") -> dict:
    return {"id": 1, "name": name}


def _event(
    idx: int,
    poss_id: int,
    event_type: str,
    team_id: int,
    loc: list | None = None,
    timestamp: str = "00:00:30.000",
    period: int = 1,
    play_pattern: str = "Regular Play",
) -> dict:
    return {
        "id": f"evt-{idx}",
        "index": idx,
        "period": period,
        "timestamp": timestamp,
        "possession": poss_id,
        "type": _type(event_type),
        "team": _team(team_id, f"Team{team_id}"),
        "location": loc or [60.0, 40.0],
        "play_pattern": _play_pattern(play_pattern),
        "under_pressure": False,
        "off_camera": False,
        "out": False,
    }


TEAM_ID_MAP = {217: "internal_team_217", 999: "internal_team_999"}


class TestBuildPossessions:
    def test_empty_events_returns_empty(self):
        result = build_possessions(pd.DataFrame(), "match_001", TEAM_ID_MAP)
        assert result == []

    def test_single_possession_built(self):
        events = _make_events([
            _event(1, 1, "Pass", 217, loc=[60.0, 40.0]),
            _event(2, 1, "Carry", 217, loc=[70.0, 40.0]),
            _event(3, 1, "Shot", 217, loc=[100.0, 40.0]),
        ])
        possessions = build_possessions(events, "match_001", TEAM_ID_MAP)
        assert len(possessions) == 1
        p = possessions[0]
        assert p.n_passes == 1
        assert p.n_carries == 1
        assert p.n_shots == 1
        assert p.n_events == 3

    def test_two_separate_possessions(self):
        events = _make_events([
            _event(1, 1, "Pass", 217),
            _event(2, 2, "Pass", 999),
        ])
        possessions = build_possessions(events, "match_001", TEAM_ID_MAP)
        assert len(possessions) == 2

    def test_vertical_progression_positive(self):
        events = _make_events([
            _event(1, 1, "Carry", 217, loc=[60.0, 40.0]),
            _event(2, 1, "Pass", 217, loc=[90.0, 40.0]),
        ])
        possessions = build_possessions(events, "match_001", TEAM_ID_MAP)
        p = possessions[0]
        # start_x ≈ 52.5 (60×105/120), max_x ≈ 78.75 (90×105/120)
        assert p.vertical_progression > 0.0

    def test_set_piece_flag_detected(self):
        events = _make_events([
            _event(1, 1, "Pass", 217, play_pattern="From Corner"),
        ])
        possessions = build_possessions(events, "match_001", TEAM_ID_MAP)
        assert possessions[0].set_piece_flag is True

    def test_counterpress_flag_detected(self):
        events = _make_events([
            _event(1, 1, "Pass", 217, play_pattern="From Counter Press"),
        ])
        possessions = build_possessions(events, "match_001", TEAM_ID_MAP)
        assert possessions[0].counterpress_regain_flag is True

    def test_sequence_type_defaults_unknown(self):
        events = _make_events([_event(1, 1, "Pass", 217)])
        possessions = build_possessions(events, "match_001", TEAM_ID_MAP)
        assert possessions[0].sequence_type == SequenceType.UNKNOWN

    def test_internal_id_is_string(self):
        events = _make_events([_event(1, 1, "Pass", 217)])
        possessions = build_possessions(events, "match_001", TEAM_ID_MAP)
        assert isinstance(possessions[0].internal_id, str)
        assert len(possessions[0].internal_id) == 16


class TestPossessionsToDataFrame:
    def test_returns_dataframe(self):
        events = _make_events([_event(1, 1, "Pass", 217)])
        possessions = build_possessions(events, "match_001", TEAM_ID_MAP)
        df = possessions_to_dataframe(possessions)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "sequence_type" in df.columns
