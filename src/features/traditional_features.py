"""Traditional (non-360) event-level feature extraction."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

GOAL_X = 105.0
GOAL_Y = 34.0
LEFT_POST_Y = GOAL_Y - 3.66
RIGHT_POST_Y = GOAL_Y + 3.66
BOX_X_MIN = 88.5
BOX_Y_MIN = 13.84
BOX_Y_MAX = 54.16
CENTRAL_Y_MIN = 27.0
CENTRAL_Y_MAX = 41.0


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _event_type_name(value: Any) -> str:
    raw = _enum_to_value(value)
    if raw is None:
        return "other"
    return str(raw)


def _col(df: pd.DataFrame, name: str, default: Any) -> pd.Series:
    """Return a column as Series; broadcast scalar default when missing."""
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _shot_angle(x: float, y: float) -> float:
    """Angle between vectors from shot location to each goalpost in radians."""
    if np.isnan(x) or np.isnan(y):
        return float("nan")
    ax, ay = GOAL_X - x, LEFT_POST_Y - y
    bx, by = GOAL_X - x, RIGHT_POST_Y - y
    dot = (ax * bx) + (ay * by)
    det = abs((ax * by) - (ay * bx))
    return float(math.atan2(det, dot))


def _assist_type(row: pd.Series) -> str:
    if _event_type_name(row.get("action_type", "")) != "pass":
        return "none"
    if bool(row.get("cutback", False)):
        return "cutback"
    if bool(row.get("through_ball", False)):
        return "through_ball"
    if bool(row.get("cross", False)):
        return "cross"
    return "pass"


def _is_box_entry(end_x: float, end_y: float) -> bool:
    if np.isnan(end_x) or np.isnan(end_y):
        return False
    return end_x >= BOX_X_MIN and BOX_Y_MIN <= end_y <= BOX_Y_MAX


def _is_central(y: float) -> bool:
    return not np.isnan(y) and CENTRAL_Y_MIN <= y <= CENTRAL_Y_MAX


def build_traditional_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Build traditional event-level features from canonical events DataFrame."""
    if events_df.empty:
        return pd.DataFrame()

    df = events_df.copy()
    out = pd.DataFrame(index=df.index)

    out["event_internal_id"] = df.get("internal_id")
    out["match_internal_id"] = df.get("match_internal_id")
    out["possession_internal_id"] = df.get("possession_internal_id")
    out["team_internal_id"] = df.get("team_internal_id")
    out["player_internal_id"] = df.get("player_internal_id")

    out["x_location"] = pd.to_numeric(
        df.get("x_location") if "x_location" in df.columns else df.get("x"), errors="coerce"
    )
    out["y_location"] = pd.to_numeric(
        df.get("y_location") if "y_location" in df.columns else df.get("y"), errors="coerce"
    )

    out["distance_to_goal"] = np.hypot(
        GOAL_X - out["x_location"], GOAL_Y - out["y_location"]
    ).astype(float)
    out["shot_angle"] = [
        _shot_angle(float(x), float(y))
        for x, y in zip(out["x_location"], out["y_location"], strict=False)
    ]

    out["event_type"] = _col(df, "action_type", "other").apply(_event_type_name)

    out["body_part"] = _col(df, "shot_body_part", "unknown").apply(
        lambda v: str(_enum_to_value(v) or "unknown")
    )
    out["shot_type"] = _col(df, "shot_type", "none").apply(
        lambda v: str(_enum_to_value(v) or "none")
    )
    out["first_time_shot"] = _col(df, "shot_first_time", False).fillna(False).astype(bool)
    out["volley"] = _col(df, "shot_technique", "none").apply(lambda v: str(v).lower() == "volley")
    out["header"] = _col(df, "shot_body_part", "none").apply(lambda v: str(v).lower() == "head")
    out["open_play"] = _col(df, "shot_type", "none").apply(lambda v: str(v).lower() == "open play")

    out["set_piece_type"] = _col(df, "play_pattern", "none").apply(
        lambda v: str(_enum_to_value(v) or "none")
    )

    out["pass_length"] = pd.to_numeric(_col(df, "pass_length", np.nan), errors="coerce")
    out["pass_angle"] = pd.to_numeric(_col(df, "pass_angle", np.nan), errors="coerce")
    out["cross"] = _col(df, "pass_cross", False).fillna(False).astype(bool)
    out["cutback"] = _col(df, "pass_cut_back", False).fillna(False).astype(bool)
    out["through_ball"] = _col(df, "pass_through_ball", False).fillna(False).astype(bool)
    out["switch"] = _col(df, "pass_switch", False).fillna(False).astype(bool)

    out["pass_height"] = _col(df, "pass_height", "ground").apply(
        lambda v: str(_enum_to_value(v) or "ground")
    )
    out["pass_body_part"] = _col(df, "pass_body_part", "foot").apply(
        lambda v: str(_enum_to_value(v) or "foot")
    )

    end_x = pd.to_numeric(_col(df, "end_x", np.nan), errors="coerce")
    end_y = pd.to_numeric(_col(df, "end_y", np.nan), errors="coerce")

    carry_distance = pd.to_numeric(_col(df, "carry_length", np.nan), errors="coerce")
    carry_progressive_distance = pd.to_numeric(
        _col(df, "carry_progressive_length", np.nan), errors="coerce"
    )
    is_carry = out["event_type"].astype(str).str.lower().eq("carry")

    # Fallback for flattened event schemas: infer carry lengths from start/end coordinates.
    inferred_carry_dist = np.hypot(end_x - out["x_location"], end_y - out["y_location"])
    inferred_carry_prog = (end_x - out["x_location"]).clip(lower=0)

    out["carry_distance"] = carry_distance.where(carry_distance.notna(), inferred_carry_dist)
    out["carry_progressive_distance"] = carry_progressive_distance.where(
        carry_progressive_distance.notna(), inferred_carry_prog
    )
    out.loc[~is_carry, ["carry_distance", "carry_progressive_distance"]] = np.nan

    out["progressive_distance"] = (end_x - out["x_location"]).clip(lower=0).fillna(0.0)
    out["central_progression"] = [bool(_is_central(float(v))) for v in end_y.fillna(float("nan"))]
    out["box_entry"] = [
        bool(_is_box_entry(float(ex), float(ey)))
        for ex, ey in zip(end_x.fillna(float("nan")), end_y.fillna(float("nan")), strict=False)
    ]

    out["assist_type"] = out.apply(_assist_type, axis=1)
    out["under_pressure"] = _col(df, "under_pressure", False).fillna(False).astype(bool)

    out["open_play_or_set_piece"] = np.where(
        out["set_piece_type"].eq("none"),
        "open_play",
        "set_piece",
    )

    return out
