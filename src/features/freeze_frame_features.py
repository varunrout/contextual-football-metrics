"""360 freeze-frame feature extraction at event level."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

BOX_X_MIN = 88.5
BOX_Y_MIN = 13.84
BOX_Y_MAX = 54.16
CENTRAL_Y_MIN = 27.0
CENTRAL_Y_MAX = 41.0
GOAL_X = 105.0
GOAL_Y = 34.0
BOX_AREA_M2 = 16.5 * 40.32
CENTRAL_AREA_M2 = 35.0 * (CENTRAL_Y_MAX - CENTRAL_Y_MIN)


def _distance(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
    return float(math.hypot(a_x - b_x, a_y - b_y))


def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Shortest distance from point p to segment a-b."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = (abx * abx) + (aby * aby)
    if denom == 0:
        return _distance(px, py, ax, ay)
    t = max(0.0, min(1.0, ((apx * abx) + (apy * aby)) / denom))
    cx, cy = ax + (t * abx), ay + (t * aby)
    return _distance(px, py, cx, cy)


def _polygon_area(poly: Any) -> float | None:
    """Supports nested [[x,y], ...] or flat [x1,y1,...] polygons."""
    if poly is None:
        return None
    points: list[tuple[float, float]] = []
    if isinstance(poly, list) and poly and isinstance(poly[0], list):
        for point in poly:
            if isinstance(point, list) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    elif isinstance(poly, list) and len(poly) >= 6:
        for i in range(0, len(poly) - 1, 2):
            points.append((float(poly[i]), float(poly[i + 1])))
    if len(points) < 3:
        return None

    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def _is_pass_event(row: pd.Series) -> bool:
    return str(row.get("action_type", "")).lower() == "pass"


def build_freeze_frame_features(events_df: pd.DataFrame, frames_df: pd.DataFrame) -> pd.DataFrame:
    """Build 360 feature block at event level.

    Returns one row per event with all freeze-frame features in configs/features.yaml.
    For non-360 events, fields are kept as NaN and handled by flag_and_zero downstream.
    """
    if events_df.empty:
        return pd.DataFrame()

    out_rows: list[dict[str, Any]] = []

    frames_grouped = (
        frames_df.groupby("event_internal_id") if not frames_df.empty and "event_internal_id" in frames_df.columns else None
    )

    event_sort = events_df.sort_values(["match_internal_id", "index"]).reset_index(drop=True)

    for i, row in event_sort.iterrows():
        event_id = row.get("internal_id")
        match_id = row.get("match_internal_id")
        has_360 = bool(row.get("has_360", False))
        ball_x = pd.to_numeric(row.get("x_location", row.get("x", float("nan"))), errors="coerce")
        ball_y = pd.to_numeric(row.get("y_location", row.get("y", float("nan"))), errors="coerce")
        ball_x = float(ball_x) if pd.notna(ball_x) else float("nan")
        ball_y = float(ball_y) if pd.notna(ball_y) else float("nan")

        out: dict[str, Any] = {
            "event_internal_id": event_id,
            "match_internal_id": match_id,
            "has_360": has_360,
            "nearest_defender_distance": np.nan,
            "second_nearest_defender_distance": np.nan,
            "defenders_within_3m": np.nan,
            "defenders_within_5m": np.nan,
            "defenders_within_10m": np.nan,
            "defenders_between_ball_and_goal": np.nan,
            "teammates_ahead_of_ball": np.nan,
            "opponents_ahead_of_ball": np.nan,
            "defensive_density_in_box": np.nan,
            "defensive_density_central": np.nan,
            "keeper_distance_to_goal": np.nan,
            "keeper_distance_to_shooter": np.nan,
            "keeper_angle_coverage": np.nan,
            "shot_lane_blockage_proxy": np.nan,
            "visible_area_size": np.nan,
            "defenders_between_ball_and_goal_before_action": np.nan,
            "defenders_between_ball_and_goal_after_action": np.nan,
            "pressure_before_action": np.nan,
            "pressure_after_action": np.nan,
            "central_access_before_action": np.nan,
            "central_access_after_action": np.nan,
            "box_access_before_action": np.nan,
            "box_access_after_action": np.nan,
            "nearest_defender_to_receiver": np.nan,
            "defenders_between_receiver_and_goal": np.nan,
            "pressure_on_passer": np.nan,
            "pressure_on_receiver": np.nan,
            "passing_lane_blockage_proxy": np.nan,
            "defensive_line_height": np.nan,
            "receiver_inside_box": np.nan,
            "receiver_in_half_space": np.nan,
        }

        if not has_360 or frames_grouped is None or event_id not in frames_grouped.groups:
            out_rows.append(out)
            continue

        ff = frames_grouped.get_group(event_id)
        opponents = ff[ff["teammate"] == False]
        teammates = ff[ff["teammate"] == True]

        opp_distances = [
            _distance(ball_x, ball_y, float(x), float(y)) for x, y in zip(opponents["x"], opponents["y"])
        ]
        opp_distances_sorted = sorted(opp_distances)
        out["nearest_defender_distance"] = opp_distances_sorted[0] if opp_distances_sorted else np.nan
        out["second_nearest_defender_distance"] = opp_distances_sorted[1] if len(opp_distances_sorted) > 1 else np.nan

        out["defenders_within_3m"] = int(sum(d <= 3.0 for d in opp_distances))
        out["defenders_within_5m"] = int(sum(d <= 5.0 for d in opp_distances))
        out["defenders_within_10m"] = int(sum(d <= 10.0 for d in opp_distances))

        out["defenders_between_ball_and_goal"] = int((opponents["x"] > ball_x).sum())
        out["teammates_ahead_of_ball"] = int((teammates["x"] > ball_x).sum())
        out["opponents_ahead_of_ball"] = int((opponents["x"] > ball_x).sum())

        in_box = opponents[(opponents["x"] >= BOX_X_MIN) & (opponents["y"].between(BOX_Y_MIN, BOX_Y_MAX))]
        in_central = opponents[(opponents["x"] >= 70.0) & (opponents["y"].between(CENTRAL_Y_MIN, CENTRAL_Y_MAX))]
        out["defensive_density_in_box"] = float(len(in_box) / BOX_AREA_M2)
        out["defensive_density_central"] = float(len(in_central) / CENTRAL_AREA_M2)

        gk = opponents[opponents["keeper"] == True]
        if not gk.empty:
            kx = float(gk.iloc[0]["x"])
            ky = float(gk.iloc[0]["y"])
            out["keeper_distance_to_goal"] = _distance(kx, ky, GOAL_X, GOAL_Y)
            out["keeper_distance_to_shooter"] = _distance(kx, ky, ball_x, ball_y)
            out["keeper_angle_coverage"] = max(0.0, min(1.0, 1.0 - (out["keeper_distance_to_shooter"] / 30.0)))

        lane_blocks = 0
        for _, opp in opponents.iterrows():
            d = _point_to_segment_distance(float(opp["x"]), float(opp["y"]), ball_x, ball_y, GOAL_X, GOAL_Y)
            if d <= 1.0:
                lane_blocks += 1
        out["shot_lane_blockage_proxy"] = lane_blocks

        out["visible_area_size"] = _polygon_area(row.get("visible_area"))

        out["defenders_between_ball_and_goal_before_action"] = out["defenders_between_ball_and_goal"]
        out["pressure_before_action"] = float(out["defenders_within_3m"])
        out["central_access_before_action"] = 1.0 / (1.0 + float(out["defenders_between_ball_and_goal"]))
        out["box_access_before_action"] = 1.0 / (1.0 + float(len(in_box)))

        if i + 1 < len(event_sort):
            nxt = event_sort.iloc[i + 1]
            same_possession = (
                nxt.get("match_internal_id") == row.get("match_internal_id")
                and nxt.get("possession_internal_id") == row.get("possession_internal_id")
            )
            if same_possession:
                next_event_id = nxt.get("internal_id")
                if frames_grouped is not None and next_event_id in frames_grouped.groups:
                    ff_next = frames_grouped.get_group(next_event_id)
                    opp_next = ff_next[ff_next["teammate"] == False]
                    next_x = pd.to_numeric(nxt.get("x_location", nxt.get("x", float("nan"))), errors="coerce")
                    next_y = pd.to_numeric(nxt.get("y_location", nxt.get("y", float("nan"))), errors="coerce")
                    next_x = float(next_x) if pd.notna(next_x) else float("nan")
                    next_y = float(next_y) if pd.notna(next_y) else float("nan")
                    out["defenders_between_ball_and_goal_after_action"] = int((opp_next["x"] > next_x).sum())
                    next_d3 = [
                        _distance(next_x, next_y, float(x), float(y))
                        for x, y in zip(opp_next["x"], opp_next["y"])
                    ]
                    out["pressure_after_action"] = float(sum(d <= 3.0 for d in next_d3))
                    out["central_access_after_action"] = 1.0 / (1.0 + float(out["defenders_between_ball_and_goal_after_action"]))
                    next_box = opp_next[(opp_next["x"] >= BOX_X_MIN) & (opp_next["y"].between(BOX_Y_MIN, BOX_Y_MAX))]
                    out["box_access_after_action"] = 1.0 / (1.0 + float(len(next_box)))

        if _is_pass_event(row):
            rx = float(row.get("end_x", float("nan")))
            ry = float(row.get("end_y", float("nan")))
            if not np.isnan(rx) and not np.isnan(ry):
                receiver_opp_dist = [
                    _distance(rx, ry, float(x), float(y)) for x, y in zip(opponents["x"], opponents["y"])
                ]
                out["nearest_defender_to_receiver"] = min(receiver_opp_dist) if receiver_opp_dist else np.nan
                out["defenders_between_receiver_and_goal"] = int((opponents["x"] > rx).sum())
                out["pressure_on_passer"] = float(out["defenders_within_3m"])
                out["pressure_on_receiver"] = float(sum(d <= 3.0 for d in receiver_opp_dist))

                lane_blocks_pass = 0
                for _, opp in opponents.iterrows():
                    d = _point_to_segment_distance(float(opp["x"]), float(opp["y"]), ball_x, ball_y, rx, ry)
                    if d <= 1.0:
                        lane_blocks_pass += 1
                out["passing_lane_blockage_proxy"] = lane_blocks_pass
                out["defensive_line_height"] = float(opponents["x"].max()) if not opponents.empty else np.nan
                out["receiver_inside_box"] = bool(rx >= BOX_X_MIN and BOX_Y_MIN <= ry <= BOX_Y_MAX)
                out["receiver_in_half_space"] = bool((18.0 <= ry <= 27.0) or (41.0 <= ry <= 50.0))

        out_rows.append(out)

    return pd.DataFrame(out_rows)
