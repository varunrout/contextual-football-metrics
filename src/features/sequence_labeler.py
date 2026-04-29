"""
Rule-based sequence labeller.

Classifies every possession into one of 17 canonical sequence types using
deterministic rule functions. Labels are used as weak supervision targets
for the learned sequence classifier in Phase 2c.

Resolution order matters — more specific rules take priority.
Each rule function returns True if the possession matches that type.

Tie-breaking: rules are checked in priority order (list below).
First matching rule wins. Possessions matching no rule → SequenceType.UNKNOWN.
"""

from __future__ import annotations

import logging
from typing import Callable, NamedTuple

import pandas as pd

from src.ingestion.schema import SequenceType
from src.features.sequence_features import (
    is_in_box,
    is_wide_byline,
    is_central,
    normalise_x,
    normalise_y,
    _BOX_X_MIN,
    _ATK_THIRD_MIN_X,
    _DEF_THIRD_MAX_X,
)

logger = logging.getLogger(__name__)


def _event_type_series(events: pd.DataFrame) -> pd.Series:
    if events.empty or "type" not in events.columns:
        return pd.Series([], dtype=str)
    col = events["type"]
    if col.dtype == object and len(col) > 0 and isinstance(col.iloc[0], dict):
        return col.apply(lambda t: t.get("name", "") if isinstance(t, dict) else str(t))
    return col.astype(str)


def _pass_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    types = _event_type_series(events)
    return events[types == "Pass"]


class LabelResult(NamedTuple):
    sequence_type: SequenceType
    confidence: float     # 1.0 = hard rule, 0.5 = softer heuristic
    source: str           # "rule"


# ── Individual rule functions ─────────────────────────────────────────────────
# Each function receives a possession row (pd.Series) and an events sub-DataFrame.

def _is_set_piece_first_phase(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Possession begins from a set-piece and has ≤ 3 events before a shot."""
    if not poss.get("set_piece_flag", False):
        return False
    n_before = int(poss.get("n_events", 99))
    return n_before <= 4


def _is_set_piece_second_phase(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Possession begins from a set-piece but has > 3 events (recycled)."""
    if not poss.get("set_piece_flag", False):
        return False
    return int(poss.get("n_events", 0)) > 4


def _is_fast_counterattack(poss: pd.Series, events: pd.DataFrame) -> bool:
    """
    Regain in own half + shot/box entry within 10 seconds + ≥ 30m vertical progress.
    """
    duration = float(poss.get("end_timestamp", 0)) - float(poss.get("start_timestamp", 0))
    vert = float(poss.get("vertical_progression", 0.0))
    start_x = float(poss.get("start_x", 60.0))
    return duration <= 10.0 and vert >= 30.0 and start_x < 60.0


def _is_high_press_regain(poss: pd.Series, events: pd.DataFrame) -> bool:
    """
    Possession starts in attacking third after opponent turnover,
    counterpress flag, within 10 seconds.
    """
    start_x = float(poss.get("start_x", 0.0))
    duration = float(poss.get("end_timestamp", 0)) - float(poss.get("start_timestamp", 0))
    return (
        start_x >= _ATK_THIRD_MIN_X
        and poss.get("counterpress_regain_flag", False)
        and duration <= 10.0
    )


def _is_mid_block_regain(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Regain in middle third, relatively quick (≤ 15s), moderate progression."""
    start_x = float(poss.get("start_x", 0.0))
    duration = float(poss.get("end_timestamp", 0)) - float(poss.get("start_timestamp", 0))
    vert = float(poss.get("vertical_progression", 0.0))
    return (
        _DEF_THIRD_MAX_X < start_x < _ATK_THIRD_MIN_X
        and duration <= 15.0
        and vert >= 15.0
        and not poss.get("set_piece_flag", False)
    )


def _is_low_block_regain(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Regain in defensive third, slow build."""
    start_x = float(poss.get("start_x", 0.0))
    return start_x <= _DEF_THIRD_MAX_X and not poss.get("set_piece_flag", False)


def _is_cutback_sequence(poss: pd.Series, events: pd.DataFrame) -> bool:
    """
    Final pass starts wide/byline, ends centrally inside box, shot within 2 actions.
    """
    if events.empty:
        return False
    pass_events = _pass_events(events)
    if pass_events.empty:
        return False
    last_pass = pass_events.iloc[-1]
    sx = pd.to_numeric(last_pass.get("x", float("nan")), errors="coerce")
    sy = pd.to_numeric(last_pass.get("y", float("nan")), errors="coerce")
    ex = pd.to_numeric(last_pass.get("end_x", float("nan")), errors="coerce")
    ey = pd.to_numeric(last_pass.get("end_y", float("nan")), errors="coerce")

    if pd.isna(sx) or pd.isna(sy) or pd.isna(ex) or pd.isna(ey):
        pass_data = last_pass.get("pass", {}) or {}
        start_loc = last_pass.get("location")
        end_loc = pass_data.get("end_location") if isinstance(pass_data, dict) else None
        if not (isinstance(start_loc, list) and isinstance(end_loc, list)):
            return False
        if len(start_loc) < 2 or len(end_loc) < 2:
            return False
        sx, sy = normalise_x(start_loc[0]), normalise_y(start_loc[1])
        ex, ey = normalise_x(end_loc[0]), normalise_y(end_loc[1])
    else:
        sx, sy, ex, ey = float(sx), float(sy), float(ex), float(ey)

    wide_byline_start = is_wide_byline(sx, sy)
    central_end = is_central(ey) and is_in_box(ex, ey)
    pass_pos = -1
    if last_pass.name in pass_events.index:
        loc = pass_events.index.get_loc(last_pass.name)
        pass_pos = int(loc) if isinstance(loc, int) else -1
    events_after_pass = int(poss.get("n_events", 0)) - (pass_pos + 1 if pass_pos >= 0 else 0)
    return bool(wide_byline_start and central_end and events_after_pass <= 2)


def _is_wide_crossing_sequence(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Possession ends in a cross from a wide area."""
    if events.empty:
        return False
    pass_events = _pass_events(events)
    for _, row in pass_events.iterrows():
        if "pass_cross" in row.index and bool(row.get("pass_cross", False)):
            y = pd.to_numeric(row.get("y", float("nan")), errors="coerce")
            if pd.notna(y) and (float(y) <= 15.0 or float(y) >= 53.0):
                return True
        pass_data = row.get("pass", {}) or {}
        if isinstance(pass_data, dict) and pass_data.get("cross", False):
            loc = row.get("location")
            if isinstance(loc, list) and len(loc) >= 2:
                y = normalise_y(loc[1])
                if y <= 15.0 or y >= 53.0:
                    return True
    return False


def _is_through_ball_sequence(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Final pass is a through ball."""
    if events.empty:
        return False
    pass_events = _pass_events(events)
    if pass_events.empty:
        return False
    last = pass_events.iloc[-1]
    if "pass_through_ball" in last.index:
        return bool(last.get("pass_through_ball", False))
    pass_data = last.get("pass", {}) or {}
    return bool(isinstance(pass_data, dict) and pass_data.get("through_ball", False))


def _is_direct_long_ball(poss: pd.Series, events: pd.DataFrame) -> bool:
    """First pass is long (> 35m) and only 1–2 passes in possession."""
    if poss.get("n_passes", 0) > 2:
        return False
    if events.empty:
        return False
    pass_events = _pass_events(events)
    if pass_events.empty:
        return False
    first_pass = pass_events.iloc[0]
    if "pass_length" in first_pass.index:
        length = pd.to_numeric(first_pass.get("pass_length", 0.0), errors="coerce")
        return float(length) >= 35.0 if pd.notna(length) else False
    pass_data = first_pass.get("pass", {}) or {}
    length = float(pass_data.get("length", 0.0)) if isinstance(pass_data, dict) else 0.0
    # StatsBomb length is in yards; 35 yards ≈ 32m
    return length >= 35.0


def _is_carry_led_progression(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Carries account for > 50% of total events and > 20m vertical progress."""
    n_carries = int(poss.get("n_carries", 0))
    n_total = max(int(poss.get("n_events", 1)), 1)
    vert = float(poss.get("vertical_progression", 0.0))
    return (n_carries / n_total) > 0.50 and vert >= 20.0


def _is_central_combination(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Multiple passes through central corridor, ending centrally."""
    if events.empty:
        return False
    pass_events = _pass_events(events)
    central_passes = 0
    for _, row in pass_events.iterrows():
        y = pd.to_numeric(row.get("y", float("nan")), errors="coerce")
        if pd.notna(y) and is_central(float(y)):
            central_passes += 1
            continue
        loc = row.get("location")
        if isinstance(loc, list) and len(loc) >= 2 and is_central(normalise_y(loc[1])):
            central_passes += 1
    return central_passes >= 3


def _is_switch_of_play_attack(poss: pd.Series, events: pd.DataFrame) -> bool:
    return int(poss.get("number_of_switches", 0)) >= 2


def _is_chaotic_loose_ball(poss: pd.Series, events: pd.DataFrame) -> bool:
    """High event count, low directness, starts in mid-transition."""
    directness = float(poss.get("directness", 1.0))
    n_events = int(poss.get("n_events", 0))
    return directness < 0.2 and n_events >= 6


def _is_recycled_attack(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Starts in attacking third (recycled from previous phase), no counterpress."""
    start_x = float(poss.get("start_x", 0.0))
    return start_x >= _ATK_THIRD_MIN_X and not poss.get("counterpress_regain_flag", False)


def _is_deep_buildup(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Long possession (≥ 8 passes), starts from defensive third."""
    start_x = float(poss.get("start_x", 0.0))
    n_passes = int(poss.get("n_passes", 0))
    return start_x <= _DEF_THIRD_MAX_X and n_passes >= 8


def _is_settled_possession(poss: pd.Series, events: pd.DataFrame) -> bool:
    """Catch-all: moderate possession in mid-third, no strong markers."""
    start_x = float(poss.get("start_x", 0.0))
    return _DEF_THIRD_MAX_X < start_x < _ATK_THIRD_MIN_X


# ── Priority-ordered rule registry ────────────────────────────────────────────
# Earlier entries take priority over later ones when multiple rules match.

_RULE_REGISTRY: list[tuple[SequenceType, Callable, float]] = [
    (SequenceType.SET_PIECE_FIRST_PHASE,   _is_set_piece_first_phase,   1.0),
    (SequenceType.SET_PIECE_SECOND_PHASE,  _is_set_piece_second_phase,  1.0),
    (SequenceType.FAST_COUNTERATTACK,      _is_fast_counterattack,      1.0),
    (SequenceType.HIGH_PRESS_REGAIN,       _is_high_press_regain,       1.0),
    (SequenceType.CUTBACK_SEQUENCE,        _is_cutback_sequence,        0.9),
    (SequenceType.THROUGH_BALL_SEQUENCE,   _is_through_ball_sequence,   0.85),
    (SequenceType.DIRECT_LONG_BALL,        _is_direct_long_ball,        0.85),
    (SequenceType.CARRY_LED_PROGRESSION,   _is_carry_led_progression,   0.85),
    (SequenceType.WIDE_CROSSING_SEQUENCE,  _is_wide_crossing_sequence,  0.80),
    (SequenceType.SWITCH_OF_PLAY_ATTACK,   _is_switch_of_play_attack,   0.80),
    (SequenceType.CENTRAL_COMBINATION,     _is_central_combination,     0.75),
    (SequenceType.CHAOTIC_LOOSE_BALL,      _is_chaotic_loose_ball,      0.70),
    (SequenceType.HIGH_PRESS_REGAIN,       _is_high_press_regain,       0.70),   # second pass
    (SequenceType.MID_BLOCK_REGAIN,        _is_mid_block_regain,        0.70),
    (SequenceType.LOW_BLOCK_REGAIN,        _is_low_block_regain,        0.65),
    (SequenceType.RECYCLED_ATTACK,         _is_recycled_attack,         0.65),
    (SequenceType.DEEP_BUILDUP,            _is_deep_buildup,            0.65),
    (SequenceType.SETTLED_POSSESSION,      _is_settled_possession,      0.60),
]


# ── Public labeller ───────────────────────────────────────────────────────────

def label_possession(poss: pd.Series, events: pd.DataFrame) -> LabelResult:
    """
    Apply rules in priority order and return the first matching label.

    Parameters
    ----------
    poss   : one possession row (pd.Series)
    events : events sub-DataFrame for this possession

    Returns
    -------
    LabelResult(sequence_type, confidence, source="rule")
    """
    for seq_type, rule_fn, confidence in _RULE_REGISTRY:
        try:
            if rule_fn(poss, events):
                return LabelResult(seq_type, confidence, "rule")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Rule %s raised: %s", rule_fn.__name__, exc)
    return LabelResult(SequenceType.UNKNOWN, 0.0, "rule")


def label_possessions_dataframe(
    possessions_df: pd.DataFrame,
    events_df: pd.DataFrame,
    match_id_col: str = "match_internal_id",
    poss_idx_col: str = "possession_index",
) -> pd.DataFrame:
    """
    Apply rule-based labelling to all possessions in a DataFrame.

    Adds / overwrites three columns:
      sequence_type_rule        — SequenceType enum value (as string)
      sequence_type_confidence  — confidence of the rule match
      sequence_type_source      — always "rule" here

    Returns
    -------
    possessions_df with the three new columns.
    """
    if possessions_df.empty:
        return possessions_df

    results = []
    for _, poss in possessions_df.iterrows():
        match_id = poss[match_id_col]
        poss_idx = int(poss[poss_idx_col])
        poss_events = events_df[
            (events_df.get("match_internal_id", pd.Series()) == match_id)
            & (events_df["possession"] == poss_idx)
        ] if not events_df.empty else pd.DataFrame()

        result = label_possession(poss, poss_events)
        results.append(
            {
                "sequence_type_rule": result.sequence_type.value,
                "sequence_type_confidence": result.confidence,
                "sequence_type_source": result.source,
            }
        )

    label_df = pd.DataFrame(results, index=possessions_df.index)
    for col in ["sequence_type_rule", "sequence_type_confidence", "sequence_type_source"]:
        possessions_df[col] = label_df[col]

    # Set final sequence_type = rule label (will be overridden by classifier in Phase 2c)
    possessions_df["sequence_type"] = possessions_df["sequence_type_rule"]

    label_dist = possessions_df["sequence_type_rule"].value_counts()
    logger.info("Sequence label distribution:\n%s", label_dist.to_string())

    return possessions_df
