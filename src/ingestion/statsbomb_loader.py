"""
StatsBomb Open Data loader.

Fetches raw JSON from statsbombpy and caches to data/raw/statsbomb/ as
gzip-compressed JSON files. Subsequent calls read from cache unless
force_reload=True.

All returned data is raw StatsBomb format — conversion to internal schema
happens in provider_mapper.py.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


def _get_sb():
    """Lazy import of statsbombpy.sb — fails only when actually called."""
    try:
        from statsbombpy import sb  # noqa: PLC0415

        return sb
    except ImportError as exc:
        raise ImportError("statsbombpy is required. Install with: poetry install") from exc


logger = logging.getLogger(__name__)

# Root data directory — resolved relative to project root
_RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "statsbomb"


def _cache_path(resource: str, *keys: str | int) -> Path:
    """Return a deterministic cache file path for a given resource + keys."""
    suffix = "_".join(str(k) for k in keys)
    return _RAW_ROOT / resource / f"{suffix}.json.gz"


def _load_cache(path: Path) -> Any | None:
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


# ── Public API ────────────────────────────────────────────────────────────────


def load_competitions(force_reload: bool = False) -> pd.DataFrame:
    """Return all StatsBomb Open Data competitions as a DataFrame."""
    cache = _cache_path("competitions", "all")
    if not force_reload:
        raw = _load_cache(cache)
        if raw is not None:
            logger.debug("competitions: loaded from cache")
            return pd.DataFrame(raw)
    logger.info("competitions: fetching from statsbombpy")
    sb = _get_sb()
    df = sb.competitions()
    _save_cache(cache, df.to_dict(orient="records"))
    return df


def load_matches(
    competition_id: int,
    season_id: int,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Return all matches for a competition/season."""
    cache = _cache_path("matches", competition_id, season_id)
    if not force_reload:
        raw = _load_cache(cache)
        if raw is not None:
            logger.debug("matches %s/%s: loaded from cache", competition_id, season_id)
            return pd.DataFrame(raw)
    logger.info("matches %s/%s: fetching from statsbombpy", competition_id, season_id)
    sb = _get_sb()
    df = sb.matches(competition_id=competition_id, season_id=season_id)
    _save_cache(cache, df.to_dict(orient="records"))
    return df


def load_lineups(match_id: int, force_reload: bool = False) -> pd.DataFrame:
    """Return both-team lineups for a match."""
    cache = _cache_path("lineups", match_id)
    if not force_reload:
        raw = _load_cache(cache)
        if raw is not None:
            logger.debug("lineups %s: loaded from cache", match_id)
            return pd.DataFrame(raw)
    logger.info("lineups %s: fetching from statsbombpy", match_id)
    sb = _get_sb()
    df = sb.lineups(match_id=match_id)
    # lineups() returns a dict {team_name: DataFrame}; flatten to single df
    frames = []
    for team_name, team_df in df.items():
        team_df = team_df.copy()
        team_df["team_name"] = team_name
        frames.append(team_df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _save_cache(cache, combined.to_dict(orient="records"))
    return combined


def load_events(match_id: int, force_reload: bool = False) -> pd.DataFrame:
    """Return all events for a match (no 360 data)."""
    cache = _cache_path("events", match_id)
    if not force_reload:
        raw = _load_cache(cache)
        if raw is not None:
            logger.debug("events %s: loaded from cache", match_id)
            return pd.DataFrame(raw)
    logger.info("events %s: fetching from statsbombpy", match_id)
    sb = _get_sb()
    df = sb.events(match_id=match_id)
    _save_cache(cache, df.to_dict(orient="records"))
    return df


def load_frames(match_id: int, force_reload: bool = False) -> pd.DataFrame:
    """
    Return 360 freeze-frame data for a match.

    Returns an empty DataFrame for competitions without 360 data —
    callers must check df.empty or has_360 flag rather than raising.
    """
    cache = _cache_path("frames", match_id)
    if not force_reload:
        raw = _load_cache(cache)
        if raw is not None:
            logger.debug("frames %s: loaded from cache", match_id)
            return pd.DataFrame(raw)
    logger.info("frames %s: fetching from statsbombpy", match_id)
    try:
        sb = _get_sb()
        df = sb.frames(match_id=match_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("frames %s: not available (%s)", match_id, exc)
        return pd.DataFrame()
    _save_cache(cache, df.to_dict(orient="records"))
    return df


def load_match_complete(
    match_id: int,
    has_360: bool,
    force_reload: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load events + lineups + (optionally) 360 frames for a single match.

    Returns a dict with keys: "events", "lineups", "frames".
    "frames" is an empty DataFrame when has_360 is False.
    """
    events = load_events(match_id, force_reload=force_reload)
    lineups = load_lineups(match_id, force_reload=force_reload)
    frames = load_frames(match_id, force_reload=force_reload) if has_360 else pd.DataFrame()
    return {"events": events, "lineups": lineups, "frames": frames}
