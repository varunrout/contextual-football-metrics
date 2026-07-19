"""
CxA player and team leaderboards — Phase 6d.

Aggregates per-action CxA scores (from CxAPipeline.score()) into
per-90-minute player and team leaderboard tables.

Available aggregation dimensions
---------------------------------
CxA_per_90              : all creative actions per 90
CxA_open_play           : non-set-piece actions only
CxA_cutbacks            : cutback actions only
CxA_transition          : transition sequences only
CxA_settled             : settled possession sequences only
CxA_against_elite       : actions against top-ranked opponents
CxA_minus_xA            : contextual uplift over baseline xA
shot_creation_prob_per_90 : mean p_shot_created per 90
avg_resulting_CxG       : mean expected_cxg_if_shot (non-zero actions)

Usage
-----
    from src.dashboards.cxa_leaderboard import build_player_leaderboard, build_team_leaderboard

    player_lb = build_player_leaderboard(scored_df, minutes_df)
    team_lb   = build_team_leaderboard(scored_df, minutes_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum minutes to appear on the leaderboard
_MIN_MINUTES = 90.0
# Columns required in scored_df
_REQUIRED_COLS = {"cxa", "p_shot_created", "expected_cxg_if_shot"}


def _per_90(total: float, minutes: float) -> float:
    if minutes < 1.0:
        return np.nan
    return total / minutes * 90.0


def _safe_get(row: pd.Series, col: str, default=None):
    return row[col] if col in row.index and pd.notna(row[col]) else default


# ── Player leaderboard ────────────────────────────────────────────────────────


def build_player_leaderboard(
    scored_df: pd.DataFrame,
    minutes_df: pd.DataFrame | None = None,
    baseline_xa_df: pd.DataFrame | None = None,
    elite_teams: set[str] | None = None,
    min_minutes: float = _MIN_MINUTES,
) -> pd.DataFrame:
    """
    Build a per-player CxA leaderboard.

    Parameters
    ----------
    scored_df       : output of CxAPipeline.score() — one row per creative action
    minutes_df      : DataFrame with columns [player_id, minutes_played].
                      If None, minutes are estimated from the scored_df match count
                      (1 match ≈ 90 min; rough estimate only).
    baseline_xa_df  : optional DataFrame [player_id, xA_total] for CxA_minus_xA.
    elite_teams     : set of team_ids considered elite for CxA_against_elite.
    min_minutes     : minimum minutes threshold for leaderboard inclusion.

    Returns
    -------
    DataFrame indexed by player_id, sorted by CxA_per_90 descending.
    """
    missing = _REQUIRED_COLS - set(scored_df.columns)
    if missing:
        raise ValueError(f"scored_df missing required columns: {missing}")

    df = scored_df.copy()

    # Infer minutes if not provided
    if minutes_df is not None:
        mins = minutes_df.set_index("player_id")["minutes_played"].to_dict()
    else:
        # Approximate: count distinct matches per player, × 90
        if "player_id" in df.columns and "match_id" in df.columns:
            match_counts = df.groupby("player_id")["match_id"].nunique()
            mins = (match_counts * 90.0).to_dict()
        else:
            mins = {}

    grouped = df.groupby("player_id") if "player_id" in df.columns else None
    if grouped is None:
        logger.warning("No player_id column in scored_df; leaderboard will be empty.")
        return pd.DataFrame()

    rows = []
    for player_id, grp in grouped:
        m = float(mins.get(player_id, len(grp) * 1.5))  # fallback: very rough
        if m < min_minutes:
            continue

        # ── Core metrics ──────────────────────────────────────────────────
        total_cxa = float(grp["cxa"].sum())
        n_actions = len(grp)

        # Open play (no set-piece flag)
        if "set_piece_flag" in grp.columns:
            op = grp[grp["set_piece_flag"].astype(float) == 0]
        elif "set_piece_type" in grp.columns:
            op = grp[grp["set_piece_type"].isin(["none", "None", np.nan, ""])]
        else:
            op = grp
        cxa_open_play = float(op["cxa"].sum())

        # Cutbacks
        if "action_type" in grp.columns:
            cb = grp[grp["action_type"] == "cutback"]
        else:
            cb = pd.DataFrame()
        cxa_cutbacks = float(cb["cxa"].sum()) if not cb.empty else 0.0

        # Transition
        trans_mask = (
            (grp.get("transition_or_settled", pd.Series("", index=grp.index)) == "transition")
            if "transition_or_settled" in grp.columns
            else pd.Series(False, index=grp.index)
        )
        cxa_transition = float(grp.loc[trans_mask, "cxa"].sum())

        # Settled
        settled_mask = (
            (grp.get("transition_or_settled", pd.Series("", index=grp.index)) == "settled")
            if "transition_or_settled" in grp.columns
            else pd.Series(False, index=grp.index)
        )
        cxa_settled = float(grp.loc[settled_mask, "cxa"].sum())

        # Against elite
        if elite_teams and "opponent_team_id" in grp.columns:
            elite_grp = grp[grp["opponent_team_id"].isin(elite_teams)]
            cxa_elite = float(elite_grp["cxa"].sum())
        else:
            cxa_elite = np.nan

        # Shot-creation probability per 90
        sc_prob_per_90 = _per_90(float(grp["p_shot_created"].sum()), m)

        # Avg resulting CxG (only for actions where p_shot_created > 0)
        positive = grp[grp["p_shot_created"] > 0]
        avg_resulting_cxg = (
            float(positive["expected_cxg_if_shot"].mean()) if not positive.empty else np.nan
        )

        # CxA_minus_xA
        if baseline_xa_df is not None:
            xa_lookup = baseline_xa_df.set_index("player_id")
            xA_total = (
                float(xa_lookup.loc[player_id, "xA_total"]) if player_id in xa_lookup.index else 0.0
            )
            cxa_minus_xa = _per_90(total_cxa - xA_total, m)
        else:
            cxa_minus_xa = np.nan

        rows.append(
            {
                "player_id": player_id,
                "minutes_played": m,
                "n_creative_actions": n_actions,
                "CxA_total": round(total_cxa, 4),
                "CxA_per_90": round(_per_90(total_cxa, m), 4),
                "CxA_open_play": round(cxa_open_play, 4),
                "CxA_open_play_per_90": round(_per_90(cxa_open_play, m), 4),
                "CxA_cutbacks": round(cxa_cutbacks, 4),
                "CxA_cutbacks_per_90": round(_per_90(cxa_cutbacks, m), 4),
                "CxA_transition": round(cxa_transition, 4),
                "CxA_transition_per_90": round(_per_90(cxa_transition, m), 4),
                "CxA_settled": round(cxa_settled, 4),
                "CxA_settled_per_90": round(_per_90(cxa_settled, m), 4),
                "CxA_against_elite": round(cxa_elite, 4) if not np.isnan(cxa_elite) else np.nan,
                "CxA_minus_xA_per_90": round(cxa_minus_xa, 4)
                if not np.isnan(cxa_minus_xa)
                else np.nan,
                "shot_creation_prob_per_90": round(sc_prob_per_90, 4),
                "avg_resulting_CxG": round(avg_resulting_cxg, 4)
                if not np.isnan(avg_resulting_cxg)
                else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()

    lb = pd.DataFrame(rows).sort_values("CxA_per_90", ascending=False)
    lb["rank"] = range(1, len(lb) + 1)
    return lb.set_index("rank")


# ── Team leaderboard ──────────────────────────────────────────────────────────


def build_team_leaderboard(
    scored_df: pd.DataFrame,
    minutes_df: pd.DataFrame | None = None,
    min_minutes: float = _MIN_MINUTES,
) -> pd.DataFrame:
    """
    Build a per-team CxA leaderboard.

    Parameters
    ----------
    scored_df  : output of CxAPipeline.score() — one row per creative action
    minutes_df : DataFrame with columns [team_id, minutes_played].
    min_minutes: minimum minutes threshold for inclusion.

    Returns
    -------
    DataFrame indexed by team_id, sorted by CxA_per_90 descending.
    """
    missing = _REQUIRED_COLS - set(scored_df.columns)
    if missing:
        raise ValueError(f"scored_df missing required columns: {missing}")

    df = scored_df.copy()
    if "team_id" not in df.columns:
        logger.warning("No team_id column in scored_df; leaderboard will be empty.")
        return pd.DataFrame()

    if minutes_df is not None:
        mins = minutes_df.set_index("team_id")["minutes_played"].to_dict()
    else:
        if "match_id" in df.columns:
            match_counts = df.groupby("team_id")["match_id"].nunique()
            mins = (match_counts * 90.0).to_dict()
        else:
            mins = {}

    rows = []
    for team_id, grp in df.groupby("team_id"):
        m = float(mins.get(team_id, len(grp) * 1.5))
        if m < min_minutes:
            continue

        total_cxa = float(grp["cxa"].sum())

        # Segment by action type
        by_type: dict[str, float] = {}
        if "action_type" in grp.columns:
            for atype, sub in grp.groupby("action_type"):
                by_type[str(atype)] = float(sub["cxa"].sum())

        # Transition vs settled
        trans_mask = (
            (grp.get("transition_or_settled", pd.Series("", index=grp.index)) == "transition")
            if "transition_or_settled" in grp.columns
            else pd.Series(False, index=grp.index)
        )
        cxa_trans = float(grp.loc[trans_mask, "cxa"].sum())
        cxa_settled = float(grp.loc[~trans_mask, "cxa"].sum())

        rows.append(
            {
                "team_id": team_id,
                "minutes_played": m,
                "n_creative_actions": len(grp),
                "CxA_total": round(total_cxa, 4),
                "CxA_per_90": round(_per_90(total_cxa, m), 4),
                "CxA_passes_per_90": round(
                    _per_90(by_type.get("pass", 0.0) + by_type.get("cross", 0.0), m), 4
                ),
                "CxA_carries_per_90": round(_per_90(by_type.get("carry", 0.0), m), 4),
                "CxA_cutbacks_per_90": round(_per_90(by_type.get("cutback", 0.0), m), 4),
                "CxA_transition_per_90": round(_per_90(cxa_trans, m), 4),
                "CxA_settled_per_90": round(_per_90(cxa_settled, m), 4),
                "avg_resulting_CxG": round(
                    float(grp[grp["p_shot_created"] > 0]["expected_cxg_if_shot"].mean()), 4
                )
                if (grp["p_shot_created"] > 0).any()
                else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()

    lb = pd.DataFrame(rows).sort_values("CxA_per_90", ascending=False)
    lb["rank"] = range(1, len(lb) + 1)
    return lb.set_index("rank")
