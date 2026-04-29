"""Opponent-adjustment feature engineering (pre-match, rolling)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_sort_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    out = matches_df.copy()
    if "match_date" in out.columns:
        out["_match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
        return out.sort_values(["_match_date", "internal_id"]).drop(columns=["_match_date"])
    return out.sort_values(["internal_id"])


def _build_match_team_table(events_df: pd.DataFrame, matches_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return rows: (match_internal_id, team_internal_id, opponent_team_internal_id)."""
    if matches_df is not None and not matches_df.empty and {
        "internal_id", "home_team_internal_id", "away_team_internal_id"
    }.issubset(matches_df.columns):
        rows = []
        for _, row in matches_df.iterrows():
            mid = row["internal_id"]
            home = row["home_team_internal_id"]
            away = row["away_team_internal_id"]
            rows.append({"match_internal_id": mid, "team_internal_id": home, "opponent_team_internal_id": away})
            rows.append({"match_internal_id": mid, "team_internal_id": away, "opponent_team_internal_id": home})
        return pd.DataFrame(rows)

    pairs = []
    for mid, grp in events_df.groupby("match_internal_id"):
        teams = list(grp["team_internal_id"].dropna().unique())
        if len(teams) < 2:
            continue
        a, b = teams[0], teams[1]
        pairs.append({"match_internal_id": mid, "team_internal_id": a, "opponent_team_internal_id": b})
        pairs.append({"match_internal_id": mid, "team_internal_id": b, "opponent_team_internal_id": a})
    return pd.DataFrame(pairs)


def _compute_match_level_stats(events_df: pd.DataFrame, team_map_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team-in-match defensive and context stats (vectorised)."""
    # Pre-group events by (match, team) once — avoids O(n * matches) row scans
    ev = events_df[["match_internal_id", "team_internal_id", "action_type",
                     "end_x", "shot_statsbomb_xg"]].copy() if "shot_statsbomb_xg" in events_df.columns \
        else events_df[["match_internal_id", "team_internal_id", "action_type", "end_x"]].copy()

    ev["_is_shot"] = ev["action_type"].astype(str) == "shot"
    ev["_is_pass"] = ev["action_type"].astype(str) == "pass"
    ev["_is_carry"] = ev["action_type"].astype(str) == "carry"
    ev["_is_pressure"] = ev["action_type"].astype(str) == "pressure"
    ev["_xg"] = pd.to_numeric(ev.get("shot_statsbomb_xg"), errors="coerce").fillna(0.0) \
        if "shot_statsbomb_xg" in ev.columns else 0.0
    ev["_end_x"] = pd.to_numeric(ev.get("end_x"), errors="coerce")
    ev["_box_entry"] = (ev["_is_pass"] | ev["_is_carry"]) & (ev["_end_x"] >= 88.5)

    agg = ev.groupby(["match_internal_id", "team_internal_id"], sort=False).agg(
        shots=("_is_shot", "sum"),
        xg=("_xg", "sum"),
        passes=("_is_pass", "sum"),
        pressures=("_is_pressure", "sum"),
        box_entries=("_box_entry", "sum"),
    ).reset_index()

    # Merge team_map to get opponent side
    merged = team_map_df.merge(
        agg.rename(columns={
            "team_internal_id": "opponent_team_internal_id",
            "shots": "opp_shots", "xg": "opp_xg",
            "passes": "opp_passes", "box_entries": "opp_box_entries",
            "pressures": "opp_pressures",
        }),
        on=["match_internal_id", "opponent_team_internal_id"],
        how="left",
    ).merge(
        agg[["match_internal_id", "team_internal_id", "pressures"]],
        on=["match_internal_id", "team_internal_id"],
        how="left",
    )

    merged["opp_shots"] = merged["opp_shots"].fillna(0).astype(int)
    merged["opp_xg"] = merged["opp_xg"].fillna(0.0)
    merged["opp_passes"] = merged["opp_passes"].fillna(0).astype(int)
    merged["opp_box_entries"] = merged["opp_box_entries"].fillna(0).astype(int)
    merged["pressures"] = merged["pressures"].fillna(0).astype(int)

    rows = []
    for _, r in merged.iterrows():
        pressures = int(r["pressures"])
        opp_passes = int(r["opp_passes"])
        rows.append({
            "match_internal_id": r["match_internal_id"],
            "team_internal_id": r["team_internal_id"],
            "opponent_team_internal_id": r["opponent_team_internal_id"],
            "xg_conceded": float(r["opp_xg"]),
            "shots_conceded": int(r["opp_shots"]),
            "box_entries_conceded": int(r["opp_box_entries"]),
            "pressing_intensity": float(opp_passes / max(1, pressures)),
        })

    return pd.DataFrame(rows)


def _rolling_pre_match(match_stats_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Create rolling pre-match features by team (shifted by 1)."""
    out_rows = []
    if match_stats_df.empty:
        return pd.DataFrame()

    for team_id, grp in match_stats_df.groupby("team_internal_id", sort=False):
        grp = grp.copy()
        grp = grp.sort_values(["match_order"]) if "match_order" in grp.columns else grp

        grp["opponent_xg_conceded_rolling_5"] = grp["xg_conceded"].shift(1).rolling(window, min_periods=1).mean()
        grp["opponent_shots_conceded_rolling_5"] = grp["shots_conceded"].shift(1).rolling(window, min_periods=1).mean()
        grp["opponent_box_entries_conceded_rolling_5"] = grp["box_entries_conceded"].shift(1).rolling(window, min_periods=1).mean()
        grp["opponent_pressing_intensity"] = grp["pressing_intensity"].shift(1).rolling(window, min_periods=1).mean()

        # Composite ratings (simple proxies, intentionally transparent)
        grp["opponent_defensive_rating"] = (
            grp["opponent_xg_conceded_rolling_5"].fillna(grp["xg_conceded"].median())
            + 0.1 * grp["opponent_shots_conceded_rolling_5"].fillna(grp["shots_conceded"].median())
            + 0.1 * grp["opponent_pressing_intensity"].fillna(grp["pressing_intensity"].median())
        )

        grp["opponent_transition_defence_strength"] = 1.0 / (1.0 + grp["opponent_xg_conceded_rolling_5"].fillna(1.0))
        grp["opponent_set_piece_defence_strength"] = 1.0 / (1.0 + grp["opponent_box_entries_conceded_rolling_5"].fillna(1.0))
        grp["opponent_keeper_shot_stopping_rating"] = -grp["opponent_xg_conceded_rolling_5"].fillna(0.0)

        grp["opponent_team_elo"] = 1500.0 - (50.0 * grp["opponent_xg_conceded_rolling_5"].fillna(0.0))
        grp["opponent_chance_suppression_rating"] = 1.0 / (1.0 + grp["opponent_shots_conceded_rolling_5"].fillna(1.0))
        grp["opponent_box_defence_rating"] = 1.0 / (1.0 + grp["opponent_box_entries_conceded_rolling_5"].fillna(1.0))
        grp["opponent_cross_defence_rating"] = grp["opponent_box_defence_rating"]
        grp["opponent_pressing_rating"] = 1.0 / (1.0 + grp["opponent_pressing_intensity"].fillna(1.0))
        grp["opponent_block_compactness_rating"] = grp["opponent_box_defence_rating"]
        grp["opponent_box_entry_prevention_rating"] = grp["opponent_box_defence_rating"]
        grp["opponent_team_strength"] = grp["opponent_team_elo"]

        out_rows.append(grp)

    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def build_opponent_features(
    events_df: pd.DataFrame,
    matches_df: pd.DataFrame | None = None,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Build event-level opponent features using pre-match rolling defensive form."""
    if events_df.empty:
        return pd.DataFrame()

    base = pd.DataFrame(
        {
            "event_internal_id": events_df.get("internal_id"),
            "match_internal_id": events_df.get("match_internal_id"),
            "team_internal_id": events_df.get("team_internal_id"),
        }
    )

    team_map = _build_match_team_table(events_df, matches_df)
    if team_map.empty:
        return base

    if matches_df is not None and not matches_df.empty and "internal_id" in matches_df.columns:
        match_order = _safe_sort_matches(matches_df).reset_index(drop=True)
        match_order["match_order"] = np.arange(len(match_order))
        team_map = team_map.merge(
            match_order[["internal_id", "match_order"]],
            left_on="match_internal_id",
            right_on="internal_id",
            how="left",
        ).drop(columns=["internal_id"])
    else:
        ordered = sorted(team_map["match_internal_id"].unique())
        order_map = {mid: i for i, mid in enumerate(ordered)}
        team_map["match_order"] = team_map["match_internal_id"].map(order_map)

    match_stats = _compute_match_level_stats(events_df, team_map)
    match_stats = match_stats.merge(
        team_map[["match_internal_id", "team_internal_id", "match_order"]],
        on=["match_internal_id", "team_internal_id"],
        how="left",
    )

    rolling = _rolling_pre_match(match_stats, window=rolling_window)
    if rolling.empty:
        return base

    per_match_team = rolling[
        [
            "match_internal_id",
            "team_internal_id",
            "opponent_team_internal_id",
            "opponent_xg_conceded_rolling_5",
            "opponent_shots_conceded_rolling_5",
            "opponent_box_entries_conceded_rolling_5",
            "opponent_pressing_intensity",
            "opponent_defensive_rating",
            "opponent_transition_defence_strength",
            "opponent_set_piece_defence_strength",
            "opponent_keeper_shot_stopping_rating",
            "opponent_team_elo",
            "opponent_chance_suppression_rating",
            "opponent_box_defence_rating",
            "opponent_cross_defence_rating",
            "opponent_pressing_rating",
            "opponent_block_compactness_rating",
            "opponent_box_entry_prevention_rating",
            "opponent_team_strength",
        ]
    ].drop_duplicates(["match_internal_id", "team_internal_id"])

    merged = base.merge(
        per_match_team,
        on=["match_internal_id", "team_internal_id"],
        how="left",
    )
    return merged
