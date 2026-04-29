"""
Possession builder.

Groups consecutive same-team events in a match into Possession objects,
using StatsBomb's built-in possession_id field. Enriches each possession
with aggregate statistics used by Phase 2 sequence classification.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from src.ingestion.provider_mapper import make_internal_id
from src.ingestion.schema import EventType, Possession, Provider, SequenceType

logger = logging.getLogger(__name__)

# Pitch thirds (internal x coordinates, 0–105m)
_DEF_THIRD_MAX = 35.0
_ATK_THIRD_MIN = 70.0


def _regain_zone(start_x: float) -> str:
    if start_x <= _DEF_THIRD_MAX:
        return "defensive_third"
    if start_x >= _ATK_THIRD_MIN:
        return "attacking_third"
    return "mid_third"


def build_possessions(
    events_df: pd.DataFrame,
    match_internal_id: str,
    team_id_map: dict[int, str],
) -> list[Possession]:
    """
    Build Possession objects from a match events DataFrame.

    Parameters
    ----------
    events_df       : raw StatsBomb events DataFrame for one match
                      (coordinates already normalised by provider_mapper)
    match_internal_id : internal match ID
    team_id_map       : {statsbomb_team_id: internal_team_id}

    Returns
    -------
    List of Possession objects, one per unique (possession_id, team) pair.
    Sequence type fields are left at defaults (UNKNOWN) — populated in Phase 2.
    """
    if events_df.empty:
        return []

    possessions: list[Possession] = []

    for poss_idx, group in events_df.groupby("possession", sort=True):
        group = group.sort_values("index")

        # Team for this possession — statsbombpy provides flat team_id / possession_team_id
        team_row = group.iloc[0]
        raw_tid = team_row.get("team_id") if hasattr(team_row, "get") else None
        if raw_tid is None or (hasattr(raw_tid, "__float__") and pd.isna(raw_tid)):
            # Fallback: possession_team_id column
            raw_tid = team_row.get("possession_team_id") if "possession_team_id" in group.columns else None
        if raw_tid is None or (hasattr(raw_tid, "__float__") and pd.isna(raw_tid)):
            logger.warning("possession %s: cannot resolve team; skipping", poss_idx)
            continue
        sb_team_id = int(raw_tid)
        team_internal_id = team_id_map.get(sb_team_id, str(sb_team_id))

        # Timestamps (seconds from period start — convert to absolute via period)
        period = int(group.iloc[0].get("period", 1))
        period_offset = (period - 1) * 45 * 60  # rough; exact for AET not needed here

        def _ts(row: Any) -> float:
            # row is a pandas Series; .get() works for both Series and dict
            ts = row.get("timestamp", "00:00:00.000")
            try:
                parts = str(ts).split(":")
                return period_offset + int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except Exception:  # noqa: BLE001
                return float(period_offset)

        start_ts = _ts(group.iloc[0])
        end_ts = _ts(group.iloc[-1])

        # Start location — use location of first event that has one
        start_x, start_y = float("nan"), float("nan")
        for _, row in group.iterrows():
            loc = row.get("location")
            if isinstance(loc, list) and len(loc) >= 2:
                from src.ingestion.provider_mapper import normalise_x, normalise_y
                start_x = normalise_x(float(loc[0]))
                start_y = normalise_y(float(loc[1]))
                break
            x_loc = pd.to_numeric(row.get("x_location", float("nan")), errors="coerce")
            y_loc = pd.to_numeric(row.get("y_location", float("nan")), errors="coerce")
            if pd.notna(x_loc) and pd.notna(y_loc):
                start_x = float(x_loc)
                start_y = float(y_loc)
                break

        # Aggregate counts — statsbombpy "type" column is a flat string
        type_names = (
            group["type"].fillna("") if "type" in group.columns
            else pd.Series("", index=group.index)
        )
        n_passes = int((type_names == "Pass").sum())
        n_carries = int((type_names == "Carry").sum())
        n_shots = int((type_names == "Shot").sum())

        # Vertical progression: max x reached minus start x
        x_values = []
        for _, row in group.iterrows():
            x_loc = pd.to_numeric(row.get("x_location", float("nan")), errors="coerce")
            if pd.notna(x_loc):
                x_values.append(float(x_loc))
                continue
            loc = row.get("location")
            if isinstance(loc, list) and len(loc) >= 1:
                from src.ingestion.provider_mapper import normalise_x
                x_values.append(normalise_x(float(loc[0])))
        max_x = max(x_values) if x_values else start_x
        vertical_progression = max(0.0, max_x - start_x) if not math.isnan(start_x) else 0.0

        # Total distance progressed (sum of absolute x gains per event)
        distance_progressed = 0.0
        prev_x = start_x
        for loc_x in x_values[1:]:
            distance_progressed += abs(loc_x - prev_x)
            prev_x = loc_x

        # Flags — statsbombpy "play_pattern" column is a flat string
        play_patterns = (
            group["play_pattern"].fillna("") if "play_pattern" in group.columns
            else pd.Series("", index=group.index)
        )
        set_piece_flag = bool(
            play_patterns.str.contains("Free Kick|Corner|Penalty|Throw|Kick Off", na=False).any()
        )
        counterpress_flag = bool(play_patterns.str.contains("From Counter Press", na=False).any())

        internal_id = make_internal_id(
            Provider.STATSBOMB, "possession", match_internal_id, poss_idx
        )
        start_event_id = make_internal_id(
            Provider.STATSBOMB, "event", str(group.iloc[0].get("id", ""))
        )
        end_event_id = make_internal_id(
            Provider.STATSBOMB, "event", str(group.iloc[-1].get("id", ""))
        )

        possessions.append(
            Possession(
                internal_id=internal_id,
                match_internal_id=match_internal_id,
                team_internal_id=team_internal_id,
                possession_index=int(poss_idx),
                start_event_internal_id=start_event_id,
                end_event_internal_id=end_event_id,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                start_x=start_x if not math.isnan(start_x) else 0.0,
                start_y=start_y if not math.isnan(start_y) else 0.0,
                regain_zone=_regain_zone(start_x if not math.isnan(start_x) else 0.0),
                n_events=len(group),
                n_passes=n_passes,
                n_carries=n_carries,
                n_shots=n_shots,
                vertical_progression=vertical_progression,
                distance_progressed=distance_progressed,
                set_piece_flag=set_piece_flag,
                counterpress_regain_flag=counterpress_flag,
                sequence_type=SequenceType.UNKNOWN,
                sequence_type_confidence=0.0,
                sequence_type_source="none",
            )
        )

    logger.debug("match %s: built %d possessions", match_internal_id, len(possessions))
    return possessions


def possessions_to_dataframe(possessions: list[Possession]) -> pd.DataFrame:
    """Convert a list of Possession objects to a flat DataFrame."""
    return pd.DataFrame([vars(p) for p in possessions])
