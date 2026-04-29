"""
Sequence feature engineering.

Computes per-possession sequence-level features used both for rule-based
labelling (Phase 2a) and as model inputs across CxG, CxA and CxT.

Input  : enriched possessions DataFrame + events DataFrame
Output : possessions DataFrame with sequence feature columns appended

All features are computed from the possession record and the raw event stream.
No 360 data is required here — sequence features are available for all competitions.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from src.ingestion.provider_mapper import normalise_x, normalise_y

logger = logging.getLogger(__name__)

# Pitch zone boundaries (internal coordinates, metres)
_DEF_THIRD_MAX_X = 35.0
_ATK_THIRD_MIN_X = 70.0
_CENTRAL_Y_MIN = 27.0
_CENTRAL_Y_MAX = 41.0
_BOX_X_MIN = 88.5        # ~18-yard box start
_BOX_Y_MIN = 13.84
_BOX_Y_MAX = 54.16
_WIDE_Y_BYLINE_RANGE = 10.0   # within 10m of touchline counts as wide/byline


def zone_from_x(x: float) -> str:
    if x <= _DEF_THIRD_MAX_X:
        return "defensive_third"
    if x >= _ATK_THIRD_MIN_X:
        return "attacking_third"
    return "mid_third"


def is_central(y: float) -> bool:
    return _CENTRAL_Y_MIN <= y <= _CENTRAL_Y_MAX


def is_in_box(x: float, y: float) -> bool:
    return x >= _BOX_X_MIN and _BOX_Y_MIN <= y <= _BOX_Y_MAX


def is_wide_byline(x: float, y: float) -> bool:
    """True if position is near the byline (x > 85m) and close to a touchline."""
    return x >= 85.0 and (y <= _WIDE_Y_BYLINE_RANGE or y >= (68.0 - _WIDE_Y_BYLINE_RANGE))


# ── Per-possession feature computation ───────────────────────────────────────

def compute_sequence_features(
    possessions_df: pd.DataFrame,
    events_df: pd.DataFrame,
    match_id_col: str = "match_internal_id",
    poss_idx_col: str = "possession_index",
) -> pd.DataFrame:
    """
    Compute sequence features for every possession.

    Parameters
    ----------
    possessions_df : possession records (output of possession_builder)
    events_df      : raw StatsBomb events with normalised coordinates

    Returns
    -------
    possessions_df with additional sequence feature columns.
    """
    if possessions_df.empty:
        return possessions_df

    # Index events by (match_id, possession_id) for fast lookup
    events_indexed = events_df.copy() if not events_df.empty else pd.DataFrame()
    # Normalise coordinates if raw StatsBomb columns present
    if not events_indexed.empty and "location" in events_indexed.columns:
        events_indexed[["x", "y"]] = pd.DataFrame(
            events_indexed["location"].apply(
                lambda loc: (
                    (normalise_x(loc[0]), normalise_y(loc[1]))
                    if isinstance(loc, list) and len(loc) >= 2
                    else (float("nan"), float("nan"))
                )
            ).tolist(),
            index=events_indexed.index,
        )

    # Pre-group by (match_internal_id, possession) for O(1) lookup per possession
    poss_event_groups: dict[tuple, pd.DataFrame] = {}
    if not events_indexed.empty and "match_internal_id" in events_indexed.columns and "possession" in events_indexed.columns:
        for key, grp in events_indexed.sort_values("index").groupby(
            ["match_internal_id", "possession"], sort=False
        ):
            poss_event_groups[key] = grp

    feature_rows = []
    for _, poss in possessions_df.iterrows():
        match_id = poss[match_id_col]
        poss_idx = int(poss[poss_idx_col])

        poss_events = poss_event_groups.get((match_id, poss_idx), pd.DataFrame())

        features = _compute_single_possession_features(poss, poss_events)
        feature_rows.append(features)

    feat_df = pd.DataFrame(feature_rows, index=possessions_df.index)

    # Drop columns that already exist in possessions_df to avoid conflicts
    new_cols = [c for c in feat_df.columns if c not in possessions_df.columns]
    return pd.concat([possessions_df, feat_df[new_cols]], axis=1)


def _compute_single_possession_features(poss: pd.Series, events: pd.DataFrame) -> dict:
    """Compute sequence features for one possession."""
    feats: dict = {}

    # Use event count from events DF when available; fall back to pre-computed poss value
    n_events = len(events) if not events.empty else int(poss.get("n_events", 0))
    feats["time_from_possession_start"] = float(
        poss.get("end_timestamp", 0) - poss.get("start_timestamp", 0)
    )
    feats["events_before_action"] = n_events

    type_names = _type_series(events)
    feats["passes_before_action"] = int((type_names == "Pass").sum())
    feats["carries_before_action"] = int((type_names == "Carry").sum())

    # Vertical progression speed
    duration = max(feats["time_from_possession_start"], 0.1)
    vert_prog = float(poss.get("vertical_progression", 0.0))
    feats["vertical_progression_speed"] = vert_prog / duration

    # Possession start zone
    start_x = float(poss.get("start_x", 0.0))
    start_y = float(poss.get("start_y", 0.0))
    feats["possession_start_zone"] = zone_from_x(start_x)
    feats["regain_zone"] = poss.get("regain_zone", "mid_third")

    # Directness: straight-line / total path
    dist_prog = float(poss.get("distance_progressed", 0.01))
    straight = float(poss.get("vertical_progression", 0.0))
    feats["directness"] = min(1.0, straight / max(dist_prog, 0.01))

    # Possession speed (events per second)
    feats["possession_speed"] = n_events / duration

    # Number of side-to-side switches (large lateral pass changes)
    feats["number_of_switches"] = _count_switches(events)

    # Final action type
    feats["final_action_type"] = _final_action_type(events)

    # Final pass zone
    feats["final_pass_zone"] = _final_pass_zone(events)

    # Phase of play
    feats["phase_of_play"] = _phase_of_play(start_x, feats["directness"], feats["vertical_progression_speed"])

    # Transition vs settled
    feats["transition_or_settled"] = (
        "transition"
        if feats["vertical_progression_speed"] > 5.0 or poss.get("counterpress_regain_flag", False)
        else "settled"
    )

    return feats


def _type_series(events: pd.DataFrame) -> pd.Series:
    if events.empty:
        return pd.Series([], dtype=str)
    col = events["type"]
    # Handle both flat strings and legacy nested dicts
    if col.dtype == object and len(col) > 0 and isinstance(col.iloc[0], dict):
        return col.apply(lambda t: t.get("name", "") if isinstance(t, dict) else str(t))
    return col.astype(str)


def _count_switches(events: pd.DataFrame) -> int:
    """Count passes that move the ball >25m laterally (side-to-side)."""
    if events.empty:
        return 0
    passes = events[_type_series(events) == "Pass"]
    if passes.empty:
        return 0
    # Use flat end_y / y columns when available
    if "end_y" in passes.columns and "y" in passes.columns:
        ey = pd.to_numeric(passes["end_y"], errors="coerce")
        sy = pd.to_numeric(passes["y"], errors="coerce")
        return int((ey.sub(sy).abs() > 25.0).sum())
    # Fallback: nested dict format
    count = 0
    for _, row in passes.iterrows():
        pass_data = row.get("pass", {}) or {}
        end_loc = pass_data.get("end_location") if isinstance(pass_data, dict) else None
        loc = row.get("location")
        if isinstance(loc, list) and len(loc) >= 2 and isinstance(end_loc, list) and len(end_loc) >= 2:
            dy = abs(normalise_y(end_loc[1]) - normalise_y(loc[1]))
            if dy > 25.0:
                count += 1
    return count


def _final_action_type(events: pd.DataFrame) -> str:
    if events.empty:
        return "unknown"
    types = _type_series(events)
    non_pressure = types[~types.isin(["Pressure", "Ball Receipt*"])]
    return non_pressure.iloc[-1] if len(non_pressure) > 0 else types.iloc[-1]


def _final_pass_zone(events: pd.DataFrame) -> str:
    """Zone of the last pass's end location."""
    if events.empty:
        return "unknown"
    passes = events[_type_series(events) == "Pass"]
    if passes.empty:
        return "unknown"
    last_pass = passes.iloc[-1]
    # Use flat columns first
    if "end_x" in last_pass.index and "end_y" in last_pass.index:
        ex = pd.to_numeric(last_pass["end_x"], errors="coerce")
        ey = pd.to_numeric(last_pass["end_y"], errors="coerce")
        if pd.notna(ex) and pd.notna(ey):
            if is_in_box(float(ex), float(ey)):
                return "inside_box"
            if is_central(float(ey)):
                return "central_channel"
            if float(ex) >= _ATK_THIRD_MIN_X:
                return "wide_attacking"
            return zone_from_x(float(ex))
    # Fallback: nested dict format
    pass_data = last_pass.get("pass", {}) or {}
    end_loc = pass_data.get("end_location") if isinstance(pass_data, dict) else None
    if isinstance(end_loc, list) and len(end_loc) >= 2:
        ex = normalise_x(end_loc[0])
        ey = normalise_y(end_loc[1])
        if is_in_box(ex, ey):
            return "inside_box"
        if is_central(ey):
            return "central_channel"
        if ex >= _ATK_THIRD_MIN_X:
            return "wide_attacking"
        return zone_from_x(ex)
    return "unknown"


def _phase_of_play(start_x: float, directness: float, vert_speed: float) -> str:
    if start_x < _DEF_THIRD_MAX_X and directness < 0.4:
        return "buildup"
    if vert_speed > 5.0:
        return "transition"
    if start_x >= _ATK_THIRD_MIN_X:
        return "final_third"
    return "progression"
