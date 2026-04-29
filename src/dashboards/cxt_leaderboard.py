"""
CxT leaderboard — Phase 7e.

Per-90 aggregation of CxT per player / team, broken down by action type,
pitch zone, phase of play, and opponent strength.

Functions
---------
build_cxt_player_leaderboard(scored_df, minutes_df, min_minutes, elite_teams)
build_cxt_team_leaderboard(scored_df, minutes_df, min_minutes)

scored_df columns expected:
  player_id, team_id, match_id, action_type, cxt, v_before, v_after,
  sequence_type (optional), is_central (optional), transition_or_settled (optional),
  opponent_team_id (optional)

minutes_df columns expected:
  player_id (or team_id), match_id, minutes_played

Output leaderboard columns (player):
  player_id, total_actions, total_minutes, CxT_per_90,
  CxT_carries_per_90, CxT_passes_per_90,
  CxT_under_pressure_per_90 (if under_pressure col available),
  CxT_progressive_per_90, CxT_transition_per_90, CxT_settled_per_90,
  CxT_central_per_90, CxT_wide_per_90,
  CxT_vs_strong_opponents_per_90,
  CxT_minus_xT_per_90  (contextual CxT vs zone-xT baseline; uses opponent_adjustment_delta)
  avg_v_before, avg_v_after
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_PER_90 = 90.0


def _safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if denom > 0 else default


def _build_leaderboard_base(
    scored_df: pd.DataFrame,
    minutes_df: pd.DataFrame,
    id_col: str,
    min_minutes: float,
) -> pd.DataFrame:
    """Shared aggregation helper for both player and team leaderboards."""
    if scored_df.empty:
        return pd.DataFrame()

    cxt_df = scored_df.dropna(subset=["cxt"]).copy()

    # ── Core aggregations ─────────────────────────────────────────────────────
    agg: dict[str, pd.Series] = {}

    agg["total_actions"] = cxt_df.groupby(id_col)["cxt"].count()
    agg["total_cxt"]     = cxt_df.groupby(id_col)["cxt"].sum()

    # By action type
    for action in ("carry", "pass", "cross", "cutback"):
        sub = cxt_df[cxt_df["action_type"] == action] if "action_type" in cxt_df.columns else cxt_df
        agg[f"cxt_{action}"] = sub.groupby(id_col)["cxt"].sum()

    # Under pressure
    if "under_pressure" in cxt_df.columns:
        press_df = cxt_df[cxt_df["under_pressure"].astype(bool)]
        agg["cxt_under_pressure"] = press_df.groupby(id_col)["cxt"].sum()

    # Progressive (positive CxT ≡ territory gained)
    prog_df = cxt_df[cxt_df["cxt"] > 0]
    agg["cxt_progressive"] = prog_df.groupby(id_col)["cxt"].sum()

    # Transition vs settled
    if "transition_or_settled" in cxt_df.columns:
        trans = cxt_df[cxt_df["transition_or_settled"].isin(["transition", "counter-attack"])]
        settled = cxt_df[cxt_df["transition_or_settled"] == "settled"]
        agg["cxt_transition"] = trans.groupby(id_col)["cxt"].sum()
        agg["cxt_settled"]    = settled.groupby(id_col)["cxt"].sum()

    if "sequence_type" in cxt_df.columns:
        trans2 = cxt_df[cxt_df["sequence_type"].str.contains("transition|counter", case=False, na=False)]
        if "cxt_transition" not in agg:
            agg["cxt_transition"] = trans2.groupby(id_col)["cxt"].sum()

    # Central vs wide
    if "is_central" in cxt_df.columns:
        central = cxt_df[cxt_df["is_central"].astype(bool)]
        wide    = cxt_df[~cxt_df["is_central"].astype(bool)]
        agg["cxt_central"] = central.groupby(id_col)["cxt"].sum()
        agg["cxt_wide"]    = wide.groupby(id_col)["cxt"].sum()

    # vs strong opponents
    if "opponent_team_id" in cxt_df.columns and "elite_teams" in cxt_df.columns:
        strong = cxt_df[cxt_df["opponent_team_id"].isin(cxt_df["elite_teams"])]
        agg["cxt_vs_strong"] = strong.groupby(id_col)["cxt"].sum()

    # Opponent adjustment delta (CxT contextual uplift over zone xT)
    if "opponent_adjustment_delta" in cxt_df.columns:
        agg["total_oad"] = cxt_df.groupby(id_col)["opponent_adjustment_delta"].sum()

    # Average V values
    if "v_before" in cxt_df.columns:
        agg["avg_v_before"] = cxt_df.groupby(id_col)["v_before"].mean()
    if "v_after" in cxt_df.columns:
        agg["avg_v_after"] = cxt_df.groupby(id_col)["v_after"].mean()

    # ── Merge aggregations ────────────────────────────────────────────────────
    df = pd.DataFrame(agg)
    df.index.name = id_col
    df = df.reset_index()

    # ── Minutes played ────────────────────────────────────────────────────────
    if not minutes_df.empty and id_col in minutes_df.columns and "minutes_played" in minutes_df.columns:
        mins = minutes_df.groupby(id_col)["minutes_played"].sum().reset_index()
        df = df.merge(mins, on=id_col, how="left")
    else:
        df["minutes_played"] = _PER_90  # fallback

    df["minutes_played"] = df["minutes_played"].fillna(_PER_90)
    df = df[df["minutes_played"] >= min_minutes]

    # ── Per-90 conversions ────────────────────────────────────────────────────
    for raw_col, per90_col in [
        ("total_cxt",           "CxT_per_90"),
        ("cxt_carry",           "CxT_carries_per_90"),
        ("cxt_pass",            "CxT_passes_per_90"),
        ("cxt_cross",           "CxT_crosses_per_90"),
        ("cxt_cutback",         "CxT_cutbacks_per_90"),
        ("cxt_under_pressure",  "CxT_under_pressure_per_90"),
        ("cxt_progressive",     "CxT_progressive_per_90"),
        ("cxt_transition",      "CxT_transition_per_90"),
        ("cxt_settled",         "CxT_settled_per_90"),
        ("cxt_central",         "CxT_central_per_90"),
        ("cxt_wide",            "CxT_wide_per_90"),
        ("cxt_vs_strong",       "CxT_vs_strong_opponents_per_90"),
        ("total_oad",           "CxT_minus_xT_per_90"),
    ]:
        if raw_col in df.columns:
            df[per90_col] = (df[raw_col].fillna(0.0) / df["minutes_played"]) * _PER_90
        else:
            df[per90_col] = 0.0

    df = df.sort_values("CxT_per_90", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"

    return df


def build_cxt_player_leaderboard(
    scored_df: pd.DataFrame,
    minutes_df: pd.DataFrame,
    min_minutes: float = 270.0,
    elite_teams: frozenset | None = None,
) -> pd.DataFrame:
    """
    Build a per-90 player CxT leaderboard.

    Parameters
    ----------
    scored_df:
        Output of CxTPipeline.score(), must contain player_id, cxt, action_type.
    minutes_df:
        DataFrame with columns player_id, match_id, minutes_played.
    min_minutes:
        Minimum minutes played for inclusion (default 270 = 3 full matches).
    elite_teams:
        Optional frozenset of team_ids considered elite opponents.

    Returns
    -------
    DataFrame ranked by CxT_per_90.
    """
    df = scored_df.copy()
    if elite_teams is not None and "opponent_team_id" in df.columns:
        df["elite_teams"] = df["opponent_team_id"].map(lambda t: elite_teams)

    return _build_leaderboard_base(df, minutes_df, id_col="player_id", min_minutes=min_minutes)


def build_cxt_team_leaderboard(
    scored_df: pd.DataFrame,
    minutes_df: pd.DataFrame,
    min_minutes: float = 270.0,
) -> pd.DataFrame:
    """
    Build a per-90 team CxT leaderboard.

    Parameters
    ----------
    scored_df:
        Output of CxTPipeline.score(), must contain team_id, cxt, action_type.
    minutes_df:
        DataFrame with columns team_id, match_id, minutes_played.
    min_minutes:
        Minimum minutes played for inclusion.

    Returns
    -------
    DataFrame ranked by CxT_per_90.
    """
    return _build_leaderboard_base(scored_df, minutes_df, id_col="team_id", min_minutes=min_minutes)
