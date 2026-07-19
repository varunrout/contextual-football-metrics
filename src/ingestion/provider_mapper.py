"""
Provider mapper: StatsBomb raw JSON → internal schema objects.

Responsibilities:
  1. Generate stable internal IDs (sha256-based, provider-scoped).
  2. Normalise coordinates to the internal 105×68m system.
  3. Map provider-specific categorical values to internal Enum values.
  4. Construct typed schema objects (Event, ShotEvent, PassEvent, etc.).

StatsBomb coordinate system:
  x ∈ [0, 120] yards   (origin = own goal line, 120 = opponent goal)
  y ∈ [0, 80]  yards   (origin = left touchline from attacking direction)

Internal coordinate system:
  x ∈ [0, 105] metres
  y ∈ [0, 68]  metres

Conversion: metres = yards × 0.9144
  x_internal = x_sb × (105 / 120)
  y_internal = y_sb × (68 / 80)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from src.ingestion.schema import (
    BodyPart,
    CarryEvent,
    Event,
    EventType,
    FreezeFrame360,
    Match,
    PassEvent,
    PassHeight,
    Player,
    Provider,
    SetPieceType,
    ShotEvent,
    Team,
)

logger = logging.getLogger(__name__)

# ── Coordinate normalisation ──────────────────────────────────────────────────

_SB_X_MAX = 120.0
_SB_Y_MAX = 80.0
_INT_X_MAX = 105.0
_INT_Y_MAX = 68.0


def normalise_x(x_sb: float) -> float:
    """StatsBomb x (0-120 yards) → internal x (0-105 metres)."""
    return x_sb * (_INT_X_MAX / _SB_X_MAX)


def normalise_y(y_sb: float) -> float:
    """StatsBomb y (0-80 yards) → internal y (0-68 metres)."""
    return y_sb * (_INT_Y_MAX / _SB_Y_MAX)


def normalise_coords(loc: list[float] | None) -> tuple[float, float]:
    """
    Convert a StatsBomb [x, y] location list to (x_internal, y_internal).
    Returns (float('nan'), float('nan')) when loc is None/missing.
    """
    if loc is None or len(loc) < 2:
        return float("nan"), float("nan")
    return normalise_x(loc[0]), normalise_y(loc[1])


# ── ID generation ─────────────────────────────────────────────────────────────


def make_internal_id(provider: Provider, *keys: Any) -> str:
    """Stable 16-hex-char ID: sha256(provider:key1:key2:…)[:16]."""
    raw = ":".join([provider.value] + [str(k) for k in keys])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Category mappings ─────────────────────────────────────────────────────────

_EVENT_TYPE_MAP: dict[str, EventType] = {
    "Shot": EventType.SHOT,
    "Pass": EventType.PASS,
    "Carry": EventType.CARRY,
    "Dribble": EventType.DRIBBLE,
    "Dribbled Past": EventType.DRIBBLED_PAST,
    "Ball Receipt*": EventType.BALL_RECEIPT,
    "Pressure": EventType.PRESSURE,
    "Block": EventType.BLOCK,
    "Clearance": EventType.CLEARANCE,
    "Interception": EventType.INTERCEPTION,
    "Ball Recovery": EventType.BALL_RECOVERY,
    "Foul Committed": EventType.FOUL_COMMITTED,
    "Foul Won": EventType.FOUL_WON,
    "Goal Keeper": EventType.GOAL_KEEPER,
    "Miscontrol": EventType.MISCONTROL,
    "Dispossession": EventType.DISPOSSESSION,
}

_BODY_PART_MAP: dict[str, BodyPart] = {
    "Head": BodyPart.HEAD,
    "Right Foot": BodyPart.FOOT,
    "Left Foot": BodyPart.FOOT,
    "No Touch": BodyPart.NO_TOUCH,
    "Chest": BodyPart.CHEST,
}

_SET_PIECE_MAP: dict[str, SetPieceType] = {
    "Corner": SetPieceType.CORNER,
    "Free Kick": SetPieceType.FREE_KICK,
    "Throw-in": SetPieceType.THROW_IN,
    "Kick Off": SetPieceType.KICK_OFF,
    "Penalty": SetPieceType.PENALTY,
    "Goal Kick": SetPieceType.FREE_KICK,
    "No Touch": SetPieceType.NONE,
    "Open Play": SetPieceType.NONE,
}

_PASS_HEIGHT_MAP: dict[str, PassHeight] = {
    "Ground Pass": PassHeight.GROUND,
    "Low Pass": PassHeight.LOW,
    "High Pass": PassHeight.HIGH,
}


def _map_event_type(raw: str) -> EventType:
    return _EVENT_TYPE_MAP.get(raw, EventType.OTHER)


def _map_body_part(raw: str | None) -> BodyPart:
    return _BODY_PART_MAP.get(raw or "", BodyPart.FOOT)


def _map_set_piece(raw: str | None) -> SetPieceType:
    return _SET_PIECE_MAP.get(raw or "", SetPieceType.NONE)


def _map_pass_height(raw: str | None) -> PassHeight:
    return _PASS_HEIGHT_MAP.get(raw or "", PassHeight.GROUND)


def _safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict, returning default on any missing key."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)  # type: ignore[assignment]
        if d is None:
            return default
    return d


# ── Row-level mappers ─────────────────────────────────────────────────────────


def map_event_row(
    row: dict[str, Any],
    match_internal_id: str,
    possession_internal_id: str,
    team_internal_id: str,
    player_internal_id: str | None,
    has_360: bool,
) -> Event | ShotEvent | PassEvent | CarryEvent:
    """
    Convert one raw StatsBomb event dict to the appropriate internal schema object.
    Returns the most specific type available.
    """
    event_type_raw = _safe_get(row, "type", "name", default="")
    event_type = _map_event_type(event_type_raw)
    internal_id = make_internal_id(Provider.STATSBOMB, "event", row.get("id", ""))
    x, y = normalise_coords(row.get("location"))

    base_kwargs = dict(
        internal_id=internal_id,
        provider_source=Provider.STATSBOMB,
        provider_event_id=str(row.get("id", "")),
        match_internal_id=match_internal_id,
        possession_internal_id=possession_internal_id,
        team_internal_id=team_internal_id,
        player_internal_id=player_internal_id,
        event_type=event_type,
        index=row.get("index", 0),
        period=row.get("period", 1),
        timestamp=_parse_timestamp(row.get("timestamp", "00:00:00.000")),
        x=x,
        y=y,
        under_pressure=bool(row.get("under_pressure", False)),
        off_camera=bool(row.get("off_camera", False)),
        out=bool(row.get("out", False)),
        has_360=has_360,
    )

    if event_type == EventType.SHOT:
        return _map_shot(row, base_kwargs)
    if event_type == EventType.PASS:
        return _map_pass(row, base_kwargs)
    if event_type == EventType.CARRY:
        return _map_carry(row, base_kwargs)

    return Event(**base_kwargs)


def _parse_timestamp(ts: str) -> float:
    """'HH:MM:SS.mmm' → seconds as float."""
    try:
        parts = ts.split(":")
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    except Exception:  # noqa: BLE001
        return 0.0


def _map_shot(row: dict, base: dict) -> ShotEvent:
    shot = row.get("shot", {}) or {}
    outcome_raw = _safe_get(shot, "outcome", "name", default="off_target")
    end_loc = shot.get("end_location") or []
    end_x = normalise_x(end_loc[0]) if len(end_loc) >= 1 else None
    end_y = normalise_y(end_loc[1]) if len(end_loc) >= 2 else None
    end_z = end_loc[2] if len(end_loc) >= 3 else None

    play_pattern = _safe_get(row, "play_pattern", "name", default="Regular Play")
    set_piece_type = _map_set_piece(_safe_get(shot, "type", "name") or play_pattern)

    return ShotEvent(
        **base,
        body_part=_map_body_part(_safe_get(shot, "body_part", "name")),
        shot_type=set_piece_type,
        first_time=bool(shot.get("first_time", False)),
        header=_safe_get(shot, "body_part", "name") == "Head",
        volley=bool(shot.get("technique", {}).get("name") == "Volley"),
        open_play=set_piece_type == SetPieceType.NONE,
        outcome=outcome_raw.lower().replace(" ", "_"),
        goal=outcome_raw == "Goal",
        statsbomb_xg=shot.get("statsbomb_xg"),
        end_x=end_x,
        end_y=end_y,
        end_z=end_z,
    )


def _map_pass(row: dict, base: dict) -> PassEvent:
    p = row.get("pass", {}) or {}
    end_loc = p.get("end_location") or []
    end_x = normalise_x(end_loc[0]) if len(end_loc) >= 1 else 0.0
    end_y = normalise_y(end_loc[1]) if len(end_loc) >= 2 else 0.0
    outcome_raw = _safe_get(p, "outcome", "name", default="Complete") or "Complete"
    recipient_pid = _safe_get(p, "recipient", "id")
    recipient_internal = (
        make_internal_id(Provider.STATSBOMB, "player", recipient_pid) if recipient_pid else None
    )
    set_piece_raw = _safe_get(p, "type", "name")
    return PassEvent(
        **base,
        end_x=end_x,
        end_y=end_y,
        length=float(p.get("length", 0.0)),
        angle=float(p.get("angle", 0.0)),
        height=_map_pass_height(_safe_get(p, "height", "name")),
        body_part=_map_body_part(_safe_get(p, "body_part", "name")),
        set_piece_type=_map_set_piece(set_piece_raw),
        cross=bool(p.get("cross", False)),
        cutback=bool(p.get("cut_back", False)),
        through_ball=bool(p.get("through_ball", False)),
        switch=bool(p.get("switch", False)),
        goal_assist=bool(p.get("goal_assist", False)),
        shot_assist=bool(p.get("shot_assist", False)),
        outcome=outcome_raw.lower().replace(" ", "_"),
        recipient_internal_id=recipient_internal,
    )


def _map_carry(row: dict, base: dict) -> CarryEvent:
    c = row.get("carry", {}) or {}
    end_loc = c.get("end_location") or []
    end_x = normalise_x(end_loc[0]) if len(end_loc) >= 1 else base["x"]
    end_y = normalise_y(end_loc[1]) if len(end_loc) >= 2 else base["y"]
    import math

    dist = math.hypot(end_x - base["x"], end_y - base["y"])
    prog = max(0.0, end_x - base["x"])  # positive = toward opponent goal
    return CarryEvent(**base, end_x=end_x, end_y=end_y, distance=dist, progressive_distance=prog)


# ── Match-level mappers ───────────────────────────────────────────────────────


def map_match_row(row: dict[str, Any], competition_internal_id: str) -> Match:
    match_id = row["match_id"]
    home_id = make_internal_id(Provider.STATSBOMB, "team", row["home_team"]["home_team_id"])
    away_id = make_internal_id(Provider.STATSBOMB, "team", row["away_team"]["away_team_id"])
    return Match(
        internal_id=make_internal_id(Provider.STATSBOMB, "match", match_id),
        provider_source=Provider.STATSBOMB,
        provider_match_id=match_id,
        competition_internal_id=competition_internal_id,
        home_team_internal_id=home_id,
        away_team_internal_id=away_id,
        match_date=str(row.get("match_date", "")),
        home_score=int(row.get("home_score", 0)),
        away_score=int(row.get("away_score", 0)),
        has_360=bool(row.get("has_360", False)),
        stage=str(_safe_get(row, "competition_stage", "name", default="")),
        match_week=row.get("match_week"),
        referee=str(_safe_get(row, "referee", "name", default="")),
    )


def map_team_row(raw_team: dict[str, Any]) -> Team:
    team_id = raw_team.get("home_team_id") or raw_team.get("away_team_id") or raw_team.get("id", 0)
    team_name = (
        raw_team.get("home_team_name") or raw_team.get("away_team_name") or raw_team.get("name", "")
    )
    country = str(_safe_get(raw_team, "country", "name", default=""))
    return Team(
        internal_id=make_internal_id(Provider.STATSBOMB, "team", team_id),
        provider_source=Provider.STATSBOMB,
        provider_team_id=int(team_id),
        team_name=team_name,
        country=country,
    )


def map_player_row(raw: dict[str, Any], team_internal_id: str) -> Player:
    return Player(
        internal_id=make_internal_id(Provider.STATSBOMB, "player", raw["player_id"]),
        provider_source=Provider.STATSBOMB,
        provider_player_id=int(raw["player_id"]),
        player_name=raw.get("player_name", ""),
        team_internal_id=team_internal_id,
        position=str(_safe_get(raw, "positions", default=[{}])[0].get("position", "")),
    )


def map_freeze_frame_row(
    ff_row: dict[str, Any],
    event_internal_id: str,
    match_internal_id: str,
    index: int,
) -> FreezeFrame360:
    player_id = _safe_get(ff_row, "player", "id")
    player_internal = (
        make_internal_id(Provider.STATSBOMB, "player", player_id) if player_id else None
    )
    loc = ff_row.get("location") or []
    x = normalise_x(loc[0]) if len(loc) >= 1 else float("nan")
    y = normalise_y(loc[1]) if len(loc) >= 2 else float("nan")
    return FreezeFrame360(
        internal_id=make_internal_id(Provider.STATSBOMB, "ff", event_internal_id, index),
        event_internal_id=event_internal_id,
        match_internal_id=match_internal_id,
        player_internal_id=player_internal,
        teammate=bool(ff_row.get("teammate", False)),
        actor=bool(ff_row.get("actor", False)),
        keeper=bool(ff_row.get("keeper", False)),
        x=x,
        y=y,
    )
