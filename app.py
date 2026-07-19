from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SCORED_PATH = ROOT / "outputs" / "scores" / "scored.parquet"
EVENTS_PATH = ROOT / "data" / "processed" / "events.parquet"

BARCA_BLUE = "#004D98"
BARCA_RED = "#A50044"
ACCENT = "#2A9D8F"


# ── Loaders ──────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading scored events…")
def load_data() -> pd.DataFrame:
    scored = pd.read_parquet(SCORED_PATH)
    events = pd.read_parquet(EVENTS_PATH)

    extra = [
        c for c in ["internal_id", "player", "team", "minute", "second"] if c in events.columns
    ]
    if "event_id" in scored.columns and "internal_id" in extra:
        ev_small = events[extra].rename(
            columns={
                "internal_id": "event_id",
                "player": "player_name",
                "team": "team_name",
            }
        )
        scored = scored.merge(ev_small, on="event_id", how="left")

    def _clean(series: pd.Series, fallback: pd.Series) -> pd.Series:
        s = series.astype("object")
        s = s.where(~s.isin(["None", "nan", "NaN", ""]), other=np.nan)
        return s.fillna(fallback.astype(str))

    if "player_name" not in scored.columns:
        scored["player_name"] = scored["player_id"].astype(str)
    else:
        scored["player_name"] = _clean(scored["player_name"], scored["player_id"])

    if "team_name" not in scored.columns:
        scored["team_name"] = scored["team_id"].astype(str)
    else:
        scored["team_name"] = _clean(scored["team_name"], scored["team_id"])

    return scored


@st.cache_data(show_spinner=False)
def player_minutes(scored: pd.DataFrame) -> pd.DataFrame:
    """Approximate minutes: 90 per (player, match)."""
    pm = (
        scored[["player_id", "player_name", "match_id"]]
        .dropna()
        .drop_duplicates()
        .groupby(["player_id", "player_name"], as_index=False)["match_id"]
        .nunique()
        .rename(columns={"match_id": "matches"})
    )
    pm["minutes"] = pm["matches"] * 90.0
    return pm


# ── Aggregations ─────────────────────────────────────────────────────────────


def cxg_player_table(scored: pd.DataFrame, top_n: int) -> pd.DataFrame:
    shots = scored[scored["event_type"] == "shot"].copy()
    if shots.empty:
        return pd.DataFrame()
    g = (
        shots.groupby(["player_id", "player_name"], as_index=False)
        .agg(shots=("event_id", "count"), cxg=("cxg", "sum"))
        .sort_values("cxg", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    g.index = np.arange(1, len(g) + 1)
    g.index.name = "rank"
    return g[["player_name", "shots", "cxg"]]


def _per_90_table(
    scored: pd.DataFrame,
    mins: pd.DataFrame,
    value_col: str,
    top_n: int,
    min_minutes: float,
) -> pd.DataFrame:
    actions = scored[scored[value_col].notna()].copy()
    if actions.empty:
        return pd.DataFrame()
    g = actions.groupby(["player_id", "player_name"], as_index=False).agg(
        actions=("event_id", "count"), value=(value_col, "sum")
    )
    g = g.merge(mins[["player_id", "minutes"]], on="player_id", how="left")
    g = g[g["minutes"] >= min_minutes]
    g["per_90"] = g["value"] / g["minutes"].replace(0, np.nan) * 90.0
    g = g.sort_values("per_90", ascending=False).head(top_n).reset_index(drop=True)
    g.index = np.arange(1, len(g) + 1)
    g.index.name = "rank"
    g = g.rename(columns={"value": value_col, "per_90": f"{value_col}_per_90"})
    return g[["player_name", "minutes", "actions", value_col, f"{value_col}_per_90"]]


def team_summary(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["team_id", "team_name"], as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            actions=("event_id", "count"),
            cxg=("cxg", "sum"),
            cxa=("cxa", "sum"),
            cxt=("cxt", "sum"),
        )
        .sort_values("cxt", ascending=False)
        .reset_index(drop=True)
    )


# ── UI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="Contextual Football Metrics", page_icon="⚽", layout="wide")

    st.title("Contextual Football Metrics")
    st.caption("CxG, CxA, and CxT leaderboards plus player and match drill-downs.")

    scored = load_data()
    mins = player_minutes(scored)

    st.sidebar.header("Filters")
    min_minutes = st.sidebar.slider("Min minutes (per-90 tabs)", 0, 1800, 270, 90)
    top_n = st.sidebar.slider("Top N", 5, 50, 15, 5)

    teams_available = sorted(scored["team_name"].dropna().unique().tolist())
    team_filter = st.sidebar.multiselect("Teams", options=teams_available, default=teams_available)
    if team_filter and len(team_filter) != len(teams_available):
        scored = scored[scored["team_name"].isin(team_filter)].copy()
        mins = player_minutes(scored)

    tabs = st.tabs(
        [
            "CxG Leaderboard",
            "CxA Leaderboard",
            "CxT Leaderboard",
            "Player Profile",
            "Match Report",
        ]
    )

    # ── CxG ──
    with tabs[0]:
        st.subheader("CxG — Expected Goals (per shot)")
        df = cxg_player_table(scored, top_n)
        if df.empty:
            st.info("No shots in the current filter.")
        else:
            c1, c2 = st.columns([2, 3])
            with c1:
                st.dataframe(df, use_container_width=True)
            with c2:
                st.bar_chart(df.set_index("player_name")[["cxg"]], color=BARCA_BLUE)

    # ── CxA ──
    with tabs[1]:
        st.subheader("CxA — Contextual Expected Assists (per 90)")
        df = _per_90_table(scored, mins, "cxa", top_n, min_minutes)
        if df.empty:
            st.info("No CxA-scored actions in the current filter.")
        else:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.dataframe(df, use_container_width=True)
            with c2:
                st.bar_chart(df.set_index("player_name")[["cxa_per_90"]], color=BARCA_RED)

    # ── CxT ──
    with tabs[2]:
        st.subheader("CxT — Contextual Threat (per 90)")
        df = _per_90_table(scored, mins, "cxt", top_n, min_minutes)
        if df.empty:
            st.info("No CxT-scored actions in the current filter.")
        else:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.dataframe(df, use_container_width=True)
            with c2:
                st.bar_chart(df.set_index("player_name")[["cxt_per_90"]], color=ACCENT)

        st.markdown("##### Team summary")
        st.dataframe(team_summary(scored), use_container_width=True)

    # ── Player profile ──
    with tabs[3]:
        st.subheader("Player Profile")
        name_to_id = (
            mins.sort_values("minutes", ascending=False)
            .drop_duplicates("player_name")
            .set_index("player_name")["player_id"]
            .to_dict()
        )
        if not name_to_id:
            st.info("No players available.")
        else:
            player_name = st.selectbox("Player", list(name_to_id.keys()))
            pid = name_to_id[player_name]
            p = scored[scored["player_id"] == pid].copy()

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Matches", int(p["match_id"].nunique()))
            k2.metric("Actions", int(len(p)))
            k3.metric("CxG", f"{p['cxg'].sum(skipna=True):.2f}")
            k4.metric("CxA", f"{p['cxa'].sum(skipna=True):.2f}")
            k5.metric("CxT", f"{p['cxt'].sum(skipna=True):.2f}")

            trend = (
                p.groupby("match_id", as_index=False)
                .agg(cxg=("cxg", "sum"), cxa=("cxa", "sum"), cxt=("cxt", "sum"))
                .sort_values("match_id")
                .reset_index(drop=True)
            )
            trend["match"] = np.arange(1, len(trend) + 1)
            st.markdown("##### Per-match contribution")
            st.line_chart(
                trend.set_index("match")[["cxg", "cxa", "cxt"]],
                color=[BARCA_BLUE, BARCA_RED, ACCENT],
            )
            st.dataframe(trend, use_container_width=True)

    # ── Match report ──
    with tabs[4]:
        st.subheader("Match Report")
        matches = sorted(scored["match_id"].dropna().unique().tolist())
        if not matches:
            st.info("No matches in the current filter.")
        else:
            match = st.selectbox("Match ID", matches)
            m = scored[scored["match_id"] == match].copy()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Team summary")
                team_view = (
                    m.groupby("team_name", as_index=False)
                    .agg(
                        actions=("event_id", "count"),
                        cxg=("cxg", "sum"),
                        cxa=("cxa", "sum"),
                        cxt=("cxt", "sum"),
                    )
                    .sort_values("cxt", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(team_view, use_container_width=True)
            with c2:
                st.markdown("##### Top 15 players by CxT")
                top_players = (
                    m.groupby("player_name", as_index=False)
                    .agg(
                        actions=("event_id", "count"),
                        cxg=("cxg", "sum"),
                        cxa=("cxa", "sum"),
                        cxt=("cxt", "sum"),
                    )
                    .sort_values("cxt", ascending=False)
                    .head(15)
                    .reset_index(drop=True)
                )
                st.dataframe(top_players, use_container_width=True)


if __name__ == "__main__":
    main()
