"""
scripts/ingest.py
=================
Download and cache StatsBomb Open Data for all configured competitions,
map raw events to the internal schema, and save processed parquet tables.

Output layout (under data/processed/):
  events.parquet       — all events across all competitions
  matches.parquet      — match metadata
  possessions.parquet  — possession records

Usage
-----
    python scripts/ingest.py [--force-reload] [--competitions 43/106 55/282]
    python scripts/ingest.py --force-reload
    python scripts/ingest.py --competitions 43/106
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.ingestion.statsbomb_loader import (
    load_competitions,
    load_match_complete,
    load_matches,
)
from src.ingestion.provider_mapper import (
    make_internal_id,
    normalise_x,
    normalise_y,
)
from src.ingestion.possession_builder import build_possessions
from src.ingestion.schema import Provider
from src.features.sequence_labeler import label_possessions_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
COMPETITIONS_CFG = PROJECT_ROOT / "configs" / "competitions.yaml"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_competition_cfg() -> list[dict]:
    with open(COMPETITIONS_CFG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("competitions", [])


def _normalise_events_df(raw_events: pd.DataFrame, match_internal_id: str) -> pd.DataFrame:
    """
    Add internal IDs and normalised coordinates to a raw StatsBomb events DataFrame.
    This is a lightweight normalisation layer; full schema conversion is in provider_mapper.
    """
    df = raw_events.copy()
    # Internal IDs
    df["match_internal_id"] = match_internal_id
    df["internal_id"] = df.apply(
        lambda r: make_internal_id(Provider.STATSBOMB, match_internal_id, r.get("id", r.name)),
        axis=1,
    )
    # Normalise coordinates from location column
    if "location" in df.columns:
        df[["x_location", "y_location"]] = pd.DataFrame(
            df["location"].apply(
                lambda loc: list(loc[:2]) if isinstance(loc, list) and len(loc) >= 2
                else [float("nan"), float("nan")]
            ).tolist(),
            index=df.index,
        )
        df["x_location"] = df["x_location"].apply(
            lambda x: normalise_x(x) if pd.notna(x) else float("nan")
        )
        df["y_location"] = df["y_location"].apply(
            lambda y: normalise_y(y) if pd.notna(y) else float("nan")
        )

    # End location — statsbombpy provides flat pass_end_location / carry_end_location columns
    if "pass_end_location" in df.columns:
        ends = pd.DataFrame(
            df["pass_end_location"].apply(
                lambda loc: [normalise_x(loc[0]), normalise_y(loc[1])]
                if isinstance(loc, list) and len(loc) >= 2
                else [float("nan"), float("nan")]
            ).tolist(),
            columns=["end_x", "end_y"],
            index=df.index,
        )
        df["end_x"] = ends["end_x"]
        df["end_y"] = ends["end_y"]

    if "carry_end_location" in df.columns:
        carry_ends = pd.DataFrame(
            df["carry_end_location"].apply(
                lambda loc: [normalise_x(loc[0]), normalise_y(loc[1])]
                if isinstance(loc, list) and len(loc) >= 2
                else [float("nan"), float("nan")]
            ).tolist(),
            columns=["end_x", "end_y"],
            index=df.index,
        )
        if "end_x" not in df.columns:
            df["end_x"] = float("nan")
            df["end_y"] = float("nan")
        df["end_x"] = df["end_x"].fillna(carry_ends["end_x"])
        df["end_y"] = df["end_y"].fillna(carry_ends["end_y"])

    # Action type — statsbombpy "type" column is a flat string
    if "type" in df.columns:
        df["action_type"] = df["type"].str.lower().fillna("unknown")

    # Team / player internal IDs — statsbombpy provides flat team_id / player_id columns
    if "team_id" in df.columns:
        df["team_internal_id"] = df["team_id"].apply(
            lambda t: make_internal_id(Provider.STATSBOMB, "team", int(t)) if pd.notna(t) else None
        )
    if "player_id" in df.columns:
        df["player_internal_id"] = df["player_id"].apply(
            lambda p: make_internal_id(Provider.STATSBOMB, "player", int(p)) if pd.notna(p) else None
        )

    # Possession internal ID
    if "possession" in df.columns:
        df["possession_internal_id"] = df.apply(
            lambda r: make_internal_id(Provider.STATSBOMB, match_internal_id, r.get("possession", "")),
            axis=1,
        )

    # Goal indicator — statsbombpy provides flat shot_outcome column
    if "shot_outcome" in df.columns:
        df["goal"] = (df["shot_outcome"] == "Goal").astype(int)
    else:
        df["goal"] = 0

    return df


def _normalise_frames_df(raw_frames: pd.DataFrame, events_df: pd.DataFrame, match_internal_id: str) -> pd.DataFrame:
    """Normalise raw 360 frame rows to event-linked internal schema."""
    if raw_frames is None or raw_frames.empty:
        return pd.DataFrame()

    frames = raw_frames.copy()
    event_key = next((c for c in ["event_uuid", "event_id", "id"] if c in frames.columns), None)
    if event_key is None or "id" not in events_df.columns or "internal_id" not in events_df.columns:
        return pd.DataFrame()

    id_map = events_df[["id", "internal_id"]].dropna().drop_duplicates("id")
    id_map = id_map.rename(columns={"id": "raw_event_id", "internal_id": "event_internal_id"})

    frames["raw_event_id"] = frames[event_key].astype(str)
    id_map["raw_event_id"] = id_map["raw_event_id"].astype(str)
    frames = frames.merge(id_map, on="raw_event_id", how="left")

    if "location" in frames.columns:
        xy = frames["location"].apply(
            lambda loc: [normalise_x(loc[0]), normalise_y(loc[1])]
            if isinstance(loc, list) and len(loc) >= 2 else [float("nan"), float("nan")]
        )
        frames[["x", "y"]] = pd.DataFrame(xy.tolist(), index=frames.index)
    else:
        frames["x"] = pd.to_numeric(frames.get("x", float("nan")), errors="coerce")
        frames["y"] = pd.to_numeric(frames.get("y", float("nan")), errors="coerce")

    if "teammate" not in frames.columns:
        frames["teammate"] = False
    if "keeper" not in frames.columns:
        frames["keeper"] = False

    out = frames[["event_internal_id", "x", "y", "teammate", "keeper"]].copy()
    out["match_internal_id"] = match_internal_id
    out = out.dropna(subset=["event_internal_id", "x", "y"])
    out["teammate"] = out["teammate"].astype(bool)
    out["keeper"] = out["keeper"].astype(bool)
    return out


def _normalise_match_row(
    match_row: pd.Series,
    competition_id: int,
    season_id: int,
    has_360: bool,
    split_role: str,
    home_team_id: int | None = None,
    away_team_id: int | None = None,
) -> dict:
    mid = int(match_row.get("match_id", match_row.name))
    match_internal_id = make_internal_id(Provider.STATSBOMB, "match", mid)
    return {
        "internal_id": match_internal_id,
        "statsbomb_match_id": mid,
        "competition_id": competition_id,
        "season_id": season_id,
        "has_360": has_360,
        "split_role": split_role,
        "home_team_name": match_row.get("home_team"),
        "away_team_name": match_row.get("away_team"),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team_internal_id": make_internal_id(Provider.STATSBOMB, "team", home_team_id) if home_team_id else None,
        "away_team_internal_id": make_internal_id(Provider.STATSBOMB, "team", away_team_id) if away_team_id else None,
        "match_date": match_row.get("match_date"),
        "home_score": match_row.get("home_score"),
        "away_score": match_row.get("away_score"),
    }


# ── Main ingestion logic ──────────────────────────────────────────────────────

def ingest(
    competition_filter: list[str] | None = None,
    force_reload: bool = False,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    comp_cfgs = _load_competition_cfg()
    if competition_filter:
        allowed = {tuple(c.split("/")) for c in competition_filter}
        comp_cfgs = [
            c for c in comp_cfgs
            if (str(c["competition_id"]), str(c["season_id"])) in allowed
        ]
        if not comp_cfgs:
            logger.error("No competitions matched filter: %s", competition_filter)
            sys.exit(1)

    all_events: list[pd.DataFrame] = []
    all_matches: list[dict] = []
    all_possessions: list[pd.DataFrame] = []
    all_frames: list[pd.DataFrame] = []

    for comp in comp_cfgs:
        cid = comp["competition_id"]
        sid = comp["season_id"]
        has_360 = comp.get("has_360", False)
        split_role = comp.get("split_role", "train")
        logger.info("Ingesting competition_id=%s season_id=%s (has_360=%s)", cid, sid, has_360)

        matches_df = load_matches(cid, sid, force_reload=force_reload)
        if matches_df.empty:
            logger.warning("  No matches found for %s/%s", cid, sid)
            continue

        logger.info("  Found %d matches", len(matches_df))

        for _, match_row in matches_df.iterrows():
            mid = int(match_row.get("match_id", match_row.name))
            match_internal_id = make_internal_id(Provider.STATSBOMB, "match", mid)

            try:
                data = load_match_complete(mid, has_360=has_360, force_reload=force_reload)
            except Exception as exc:
                logger.warning("  match %s: failed to load (%s)", mid, exc)
                continue

            raw_events = data["events"]
            if raw_events.empty:
                logger.warning("  match %s: no events", mid)
                continue

            # Build team_id_map from flat team_id column in events
            team_id_map: dict[int, str] = {}
            if "team_id" in raw_events.columns:
                for sb_tid in raw_events["team_id"].dropna().unique():
                    tid = int(sb_tid)
                    team_id_map[tid] = make_internal_id(Provider.STATSBOMB, "team", tid)

            # Derive home/away team IDs from events for match metadata
            home_tid: int | None = None
            away_tid: int | None = None
            home_name = match_row.get("home_team")
            away_name = match_row.get("away_team")
            if "team" in raw_events.columns and "team_id" in raw_events.columns:
                ht = raw_events[raw_events["team"] == home_name]["team_id"].dropna()
                at = raw_events[raw_events["team"] == away_name]["team_id"].dropna()
                home_tid = int(ht.iloc[0]) if not ht.empty else None
                away_tid = int(at.iloc[0]) if not at.empty else None

            match_meta = _normalise_match_row(
                match_row, cid, sid, has_360, split_role,
                home_team_id=home_tid, away_team_id=away_tid,
            )
            all_matches.append(match_meta)

            events_df = _normalise_events_df(raw_events, match_internal_id)
            events_df["competition_internal_id"] = make_internal_id(Provider.STATSBOMB, "comp", cid, sid)
            events_df["has_360"] = bool(has_360)
            all_events.append(events_df)

            frames_df = _normalise_frames_df(data.get("frames", pd.DataFrame()), events_df, match_internal_id)
            if not frames_df.empty:
                all_frames.append(frames_df)

            possessions = build_possessions(raw_events, match_internal_id, team_id_map)
            if possessions:
                poss_df = pd.DataFrame([vars(p) for p in possessions])
                if "sequence_type" in poss_df.columns:
                    poss_df["sequence_type"] = poss_df["sequence_type"].apply(
                        lambda v: v.value if hasattr(v, "value") else str(v)
                    )

                events_for_label = raw_events.copy()
                events_for_label["match_internal_id"] = match_internal_id
                poss_df = label_possessions_dataframe(poss_df, events_for_label)
                all_possessions.append(poss_df)

        logger.info("  Done: %d matches processed", len(all_matches))

    # ── Save processed tables ──────────────────────────────────────────────────
    if all_events:
        events_out = pd.concat(all_events, ignore_index=True)
        path = PROCESSED_DIR / "events.parquet"
        events_out.to_parquet(path, index=False)
        logger.info("Saved events.parquet  (%d rows)", len(events_out))
    else:
        logger.warning("No events to save.")

    if all_matches:
        matches_out = pd.DataFrame(all_matches)
        path = PROCESSED_DIR / "matches.parquet"
        matches_out.to_parquet(path, index=False)
        logger.info("Saved matches.parquet (%d rows)", len(matches_out))

    if all_possessions:
        poss_out = pd.concat(all_possessions, ignore_index=True)
        path = PROCESSED_DIR / "possessions.parquet"
        poss_out.to_parquet(path, index=False)
        logger.info("Saved possessions.parquet (%d rows)", len(poss_out))

    if all_frames:
        frames_out = pd.concat(all_frames, ignore_index=True)
        path = PROCESSED_DIR / "frames.parquet"
        frames_out.to_parquet(path, index=False)
        logger.info("Saved frames.parquet (%d rows)", len(frames_out))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest StatsBomb data for configured competitions.")
    p.add_argument(
        "--competitions",
        nargs="+",
        metavar="CID/SID",
        help="Subset of competitions to ingest, e.g. --competitions 43/106 55/282. "
             "Default: all competitions in configs/competitions.yaml.",
    )
    p.add_argument(
        "--force-reload",
        action="store_true",
        help="Bypass local cache and re-fetch from StatsBomb API.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    ingest(competition_filter=args.competitions, force_reload=args.force_reload)
