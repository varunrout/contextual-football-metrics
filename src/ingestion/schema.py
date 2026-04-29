"""
Internal canonical schema for the contextual football metric suite.

All data providers map their raw data into these dataclasses before any
feature engineering or modelling. This ensures CxG, CxA and CxT are not
StatsBomb-specific in design.

Coordinate convention (enforced by provider_mapper.py):
  x ∈ [0, 105] m  — 0 = own goal line, 105 = opponent goal line
  y ∈ [0, 68]  m  — 0 = left touchline (from attacking direction), 68 = right

Internal IDs are stable UUIDs constructed as:
  sha256(f"{provider_source}:{provider_id}")[:16]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enumerations ──────────────────────────────────────────────────────────────

class Provider(str, Enum):
    STATSBOMB = "statsbomb"
    # Future: OPTA = "opta", TRACAB = "tracab"


class EventType(str, Enum):
    SHOT = "shot"
    PASS = "pass"
    CARRY = "carry"
    DRIBBLE = "dribble"
    DRIBBLED_PAST = "dribbled_past"
    BALL_RECEIPT = "ball_receipt"
    PRESSURE = "pressure"
    BLOCK = "block"
    CLEARANCE = "clearance"
    INTERCEPTION = "interception"
    BALL_RECOVERY = "ball_recovery"
    FOUL_COMMITTED = "foul_committed"
    FOUL_WON = "foul_won"
    GOAL_KEEPER = "goal_keeper"
    MISCONTROL = "miscontrol"
    DISPOSSESSION = "dispossession"
    KICK_OFF = "kick_off"
    THROW_IN = "throw_in"
    FREE_KICK = "free_kick"
    CORNER = "corner"
    OTHER = "other"


class BodyPart(str, Enum):
    HEAD = "head"
    FOOT = "foot"
    CHEST = "chest"
    NO_TOUCH = "no_touch"


class SetPieceType(str, Enum):
    NONE = "none"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    THROW_IN = "throw_in"
    KICK_OFF = "kick_off"
    PENALTY = "penalty"


class PassHeight(str, Enum):
    GROUND = "ground"
    LOW = "low"
    HIGH = "high"


class Domain(str, Enum):
    INTERNATIONAL = "international"
    CONTINENTAL = "continental"
    CLUB_DOMESTIC = "club_domestic"
    CLUB_CONTINENTAL = "club_continental"


class Region(str, Enum):
    EUROPE = "europe"
    GLOBAL = "global"
    SOUTH_AMERICA = "south_america"
    NORTH_AMERICA = "north_america"
    AFRICA = "africa"
    ASIA = "asia"


class SplitRole(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TRAIN_VAL = "train_val"
    VAL_TEST = "val_test"


class SequenceType(str, Enum):
    SETTLED_POSSESSION = "settled_possession"
    FAST_COUNTERATTACK = "fast_counterattack"
    HIGH_PRESS_REGAIN = "high_press_regain"
    MID_BLOCK_REGAIN = "mid_block_regain"
    LOW_BLOCK_REGAIN = "low_block_regain"
    DIRECT_LONG_BALL = "direct_long_ball"
    WIDE_CROSSING_SEQUENCE = "wide_crossing_sequence"
    CUTBACK_SEQUENCE = "cutback_sequence"
    CENTRAL_COMBINATION = "central_combination"
    THROUGH_BALL_SEQUENCE = "through_ball_sequence"
    CARRY_LED_PROGRESSION = "carry_led_progression"
    SWITCH_OF_PLAY_ATTACK = "switch_of_play_attack"
    SET_PIECE_FIRST_PHASE = "set_piece_first_phase"
    SET_PIECE_SECOND_PHASE = "set_piece_second_phase"
    CHAOTIC_LOOSE_BALL = "chaotic_loose_ball_sequence"
    RECYCLED_ATTACK = "recycled_attack"
    DEEP_BUILDUP = "deep_buildup_sequence"
    UNKNOWN = "unknown"


# ── Core entity dataclasses ───────────────────────────────────────────────────

@dataclass
class Competition:
    internal_id: str
    provider_source: Provider
    provider_competition_id: int
    provider_season_id: int
    competition_name: str
    season_name: str
    has_360: bool
    domain: Domain
    region: Region
    split_role: SplitRole
    competition_weight_cap: Optional[float] = None
    notes: str = ""


@dataclass
class Team:
    internal_id: str
    provider_source: Provider
    provider_team_id: int
    team_name: str
    country: str = ""


@dataclass
class Player:
    internal_id: str
    provider_source: Provider
    provider_player_id: int
    player_name: str
    team_internal_id: str
    position: str = ""


@dataclass
class Match:
    internal_id: str
    provider_source: Provider
    provider_match_id: int
    competition_internal_id: str
    home_team_internal_id: str
    away_team_internal_id: str
    match_date: str          # ISO 8601
    home_score: int
    away_score: int
    has_360: bool
    stage: str = ""          # "group" | "round_of_16" | "quarter_final" | …
    match_week: Optional[int] = None
    referee: str = ""


@dataclass
class Lineup:
    internal_id: str
    match_internal_id: str
    team_internal_id: str
    player_internal_id: str
    jersey_number: int
    position: str
    starting: bool


@dataclass
class Possession:
    """
    A reconstructed possession — consecutive events under control of one team.
    Sequence classification fields are populated by Phase 2.
    """
    internal_id: str              # "{match_id}:{possession_index}"
    match_internal_id: str
    team_internal_id: str
    possession_index: int         # StatsBomb possession_id
    start_event_internal_id: str
    end_event_internal_id: str
    start_timestamp: float        # seconds from kick-off
    end_timestamp: float
    start_x: float
    start_y: float
    regain_zone: str              # "defensive_third" | "mid_third" | "attacking_third"
    n_events: int
    n_passes: int
    n_carries: int
    n_shots: int
    vertical_progression: float
    distance_progressed: float
    set_piece_flag: bool
    counterpress_regain_flag: bool
    # Populated by Phase 2
    sequence_type: SequenceType = SequenceType.UNKNOWN
    sequence_type_confidence: float = 0.0
    sequence_type_source: str = "none"   # "rule" | "classifier" | "none"


@dataclass
class Event:
    """
    Base event record. Shot, Pass and Carry subclass this for type-specific fields.
    All coordinates are in the internal 105×68m system.
    """
    internal_id: str
    provider_source: Provider
    provider_event_id: str
    match_internal_id: str
    possession_internal_id: str
    team_internal_id: str
    player_internal_id: Optional[str]
    event_type: EventType
    index: int               # ordering index within match
    period: int
    timestamp: float         # seconds from period start
    x: float
    y: float
    under_pressure: bool
    off_camera: bool
    out: bool
    has_360: bool


@dataclass
class ShotEvent(Event):
    body_part: BodyPart = BodyPart.FOOT
    shot_type: SetPieceType = SetPieceType.NONE
    first_time: bool = False
    header: bool = False
    volley: bool = False
    open_play: bool = True
    outcome: str = "off_target"   # "goal" | "saved" | "off_target" | "blocked" | "post"
    goal: bool = False
    statsbomb_xg: Optional[float] = None    # raw provider xG for reference
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    end_z: Optional[float] = None


@dataclass
class PassEvent(Event):
    end_x: float = 0.0
    end_y: float = 0.0
    length: float = 0.0
    angle: float = 0.0
    height: PassHeight = PassHeight.GROUND
    body_part: BodyPart = BodyPart.FOOT
    set_piece_type: SetPieceType = SetPieceType.NONE
    cross: bool = False
    cutback: bool = False
    through_ball: bool = False
    switch: bool = False
    goal_assist: bool = False
    shot_assist: bool = False
    outcome: str = "complete"    # "complete" | "incomplete" | "out" | "pass_offside"
    recipient_internal_id: Optional[str] = None


@dataclass
class CarryEvent(Event):
    end_x: float = 0.0
    end_y: float = 0.0
    distance: float = 0.0
    progressive_distance: float = 0.0


@dataclass
class FreezeFrame360:
    """
    A single player entry from a StatsBomb 360 freeze frame, linked to one event.
    Multiple rows per event (one per visible player).
    """
    internal_id: str
    event_internal_id: str
    match_internal_id: str
    player_internal_id: Optional[str]   # None if player not identified
    teammate: bool
    actor: bool       # True if this is the player performing the action
    keeper: bool
    x: float
    y: float
