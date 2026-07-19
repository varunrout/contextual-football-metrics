"""Match-context feature engineering at event level."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def _home_away_lookup(matches_df: pd.DataFrame | None) -> pd.DataFrame:
    if matches_df is None or matches_df.empty:
        return pd.DataFrame(
            columns=["match_internal_id", "team_internal_id", "home_or_away", "knockout_or_group"]
        )

    rows = []
    for _, row in matches_df.iterrows():
        mid = row.get("internal_id")
        stage = str(row.get("stage", "")).lower()
        knockout = any(k in stage for k in ["round", "quarter", "semi", "final", "knockout"])
        home = row.get("home_team_internal_id")
        away = row.get("away_team_internal_id")
        if pd.notna(home):
            rows.append(
                {
                    "match_internal_id": mid,
                    "team_internal_id": home,
                    "home_or_away": "home",
                    "knockout_or_group": bool(knockout),
                }
            )
        if pd.notna(away):
            rows.append(
                {
                    "match_internal_id": mid,
                    "team_internal_id": away,
                    "home_or_away": "away",
                    "knockout_or_group": bool(knockout),
                }
            )
    return pd.DataFrame(rows)


def _rest_days_lookup(matches_df: pd.DataFrame | None) -> pd.DataFrame:
    if matches_df is None or matches_df.empty or "match_date" not in matches_df.columns:
        return pd.DataFrame(columns=["match_internal_id", "team_internal_id", "rest_days"])

    rows = []
    md = matches_df.copy()
    md["_dt"] = pd.to_datetime(md["match_date"], errors="coerce")
    md = md.sort_values(["_dt", "internal_id"])

    for side_col in ["home_team_internal_id", "away_team_internal_id"]:
        temp = md[["internal_id", side_col, "_dt"]].rename(
            columns={"internal_id": "match_internal_id", side_col: "team_internal_id"}
        )
        temp = temp.dropna(subset=["team_internal_id"]).copy()
        temp["rest_days"] = temp.groupby("team_internal_id")["_dt"].diff().dt.days
        rows.append(temp[["match_internal_id", "team_internal_id", "rest_days"]])

    if not rows:
        return pd.DataFrame(columns=["match_internal_id", "team_internal_id", "rest_days"])

    return pd.concat(rows, ignore_index=True)


def build_match_context_features(
    events_df: pd.DataFrame,
    matches_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build event-level match context features (fully vectorised)."""
    if events_df.empty:
        return pd.DataFrame()

    cols = [
        "internal_id",
        "match_internal_id",
        "team_internal_id",
        "timestamp",
        "period",
        "index",
        "goal",
    ]
    available = [c for c in cols if c in events_df.columns]
    work = events_df[available].copy()
    work = work.sort_values(["match_internal_id", "index"]).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "event_internal_id": work["internal_id"],
            "match_internal_id": work["match_internal_id"],
            "team_internal_id": work["team_internal_id"],
        }
    )

    ts = pd.to_numeric(work.get("timestamp"), errors="coerce").fillna(0.0)
    out["minute"] = np.floor(ts / 60.0).astype(int)
    out["second"] = np.floor(ts % 60.0).astype(int)
    out["period"] = pd.to_numeric(work.get("period"), errors="coerce").fillna(1).astype(int)

    # ── Vectorised score state (zero joins) ───────────────────────────────────
    # opp_goals_before_event = match_goals_before_event - my_goals_before_event
    work["_goal"] = pd.to_numeric(work.get("goal"), errors="coerce").fillna(0).astype(int)

    # Goals before this event for my team
    work["_team_cum"] = (
        work.groupby(["match_internal_id", "team_internal_id"])["_goal"].cumsum().sub(work["_goal"])
    )
    # Goals before this event in the whole match (all teams)
    work["_match_cum"] = work.groupby("match_internal_id")["_goal"].cumsum().sub(work["_goal"])
    # Opponent goals = match total - my total
    work["_opp_cum"] = work["_match_cum"] - work["_team_cum"]

    diff = work["_team_cum"] - work["_opp_cum"]
    out["score_differential"] = diff.values
    out["score_state"] = np.where(diff > 0, "winning", np.where(diff < 0, "losing", "level"))
    out["red_card_state"] = "none"

    # ── Home/away & knockout ──────────────────────────────────────────────────
    home_away = _home_away_lookup(matches_df)
    out = out.merge(home_away, on=["match_internal_id", "team_internal_id"], how="left")
    out["home_or_away"] = out["home_or_away"].fillna("neutral")
    out["knockout_or_group"] = out["knockout_or_group"].fillna(False).astype(bool)

    # ── Rest days ─────────────────────────────────────────────────────────────
    rest = _rest_days_lookup(matches_df)
    out = out.merge(rest, on=["match_internal_id", "team_internal_id"], how="left")

    # Opponent rest days — match-level lookup (873 rows, not 3M)
    if not rest.empty:
        # Build (match, team) → opponent_team from matches_df if available
        if (
            matches_df is not None
            and not matches_df.empty
            and {"internal_id", "home_team_internal_id", "away_team_internal_id"}.issubset(
                matches_df.columns
            )
        ):
            opp_map = pd.concat(
                [
                    matches_df[
                        ["internal_id", "home_team_internal_id", "away_team_internal_id"]
                    ].rename(
                        columns={
                            "internal_id": "match_internal_id",
                            "home_team_internal_id": "team_internal_id",
                            "away_team_internal_id": "_opp_team",
                        }
                    ),
                    matches_df[
                        ["internal_id", "away_team_internal_id", "home_team_internal_id"]
                    ].rename(
                        columns={
                            "internal_id": "match_internal_id",
                            "away_team_internal_id": "team_internal_id",
                            "home_team_internal_id": "_opp_team",
                        }
                    ),
                ],
                ignore_index=True,
            )
        else:
            # Derive from events unique (match, team) pairs — only 873*2 rows
            pairs = (
                work[["match_internal_id", "team_internal_id"]]
                .drop_duplicates()
                .groupby("match_internal_id")["team_internal_id"]
                .apply(list)
                .reset_index()
            )
            opp_rows = []
            for _, r in pairs.iterrows():
                teams = r["team_internal_id"]
                if len(teams) >= 2:
                    opp_rows += [
                        {
                            "match_internal_id": r["match_internal_id"],
                            "team_internal_id": teams[0],
                            "_opp_team": teams[1],
                        },
                        {
                            "match_internal_id": r["match_internal_id"],
                            "team_internal_id": teams[1],
                            "_opp_team": teams[0],
                        },
                    ]
            opp_map = pd.DataFrame(opp_rows)

        opp_rest = opp_map.merge(
            rest.rename(
                columns={"team_internal_id": "_opp_team", "rest_days": "opponent_rest_days"}
            ),
            on=["match_internal_id", "_opp_team"],
            how="left",
        )[["match_internal_id", "team_internal_id", "opponent_rest_days"]]

        out = out.merge(opp_rest, on=["match_internal_id", "team_internal_id"], how="left")
    else:
        out["opponent_rest_days"] = np.nan

    return out
