"""
analysis/_utils.py
==================
Shared helpers used by every analysis script.

Loaders
-------
load_features()  -> features.parquet  (data/features/)
load_shots()     -> shots.parquet     (data/features/)
load_actions()   -> actions.parquet   (data/features/)
load_events()    -> events.parquet    (data/processed/)   ← contains shot_statsbomb_xg
load_matches()   -> matches.parquet   (data/processed/)

Label helpers
-------------
derive_shot_created(df)        -> adds int column 'shot_created'  (CxA target)
derive_shot_in_possession(df)  -> adds int column 'shot_in_possession' (CxT proxy)

Persistence
-----------
save_fig(name, subfolder)  -> reports/figures/{subfolder}/{name}.png
save_json(data, name)      -> reports/{name}.json

Metadata
--------
competition_labels(df)  -> {competition_id: "Name (Season)"}
feature_groups()        -> {group_name: [col_name, ...]}
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
_FEATURES_DIR = _ROOT / "data" / "features"
_PROCESSED_DIR = _ROOT / "data" / "processed"
_REPORTS_DIR = _ROOT / "reports"
_FIGURES_DIR = _REPORTS_DIR / "figures"
_FEATURE_REGISTRY = _ROOT / "configs" / "features.yaml"


# ── Cached data loaders ────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def load_features() -> pd.DataFrame:
    path = _FEATURES_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {path}. Run scripts/build_features.py first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded features.parquet: %d rows × %d cols", len(df), len(df.columns))
    return df


@lru_cache(maxsize=1)
def load_shots() -> pd.DataFrame:
    path = _FEATURES_DIR / "shots.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"shots.parquet not found at {path}. Run scripts/build_features.py first."
        )
    df = pd.read_parquet(path)

    def _best_key_pair(shots_df: pd.DataFrame, events_df: pd.DataFrame) -> tuple[str, str] | None:
        """Pick the join pair with highest key overlap between shots and events."""
        candidates = [
            ("event_id", "event_id"),
            ("event_id", "internal_id"),
            ("internal_id", "internal_id"),
            ("id", "id"),
            ("id", "internal_id"),
            ("possession_internal_id", "possession_internal_id"),
        ]

        best_pair: tuple[str, str] | None = None
        best_overlap = -1
        for lcol, rcol in candidates:
            if lcol not in shots_df.columns or rcol not in events_df.columns:
                continue
            left_keys = shots_df[lcol].dropna().drop_duplicates()
            right_keys = events_df[rcol].dropna().drop_duplicates()
            if left_keys.empty or right_keys.empty:
                continue
            overlap = int(left_keys.isin(set(right_keys)).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_pair = (lcol, rcol)

        return best_pair if best_overlap > 0 else None

    # shots.parquet may not store goal/xG; attach both from events.parquet.
    if "goal" not in df.columns or "shot_statsbomb_xg" not in df.columns:
        try:
            events = load_events()
            key_pair = _best_key_pair(df, events)

            if key_pair is None:
                logger.warning(
                    "Could not find a valid shots↔events key mapping; "
                    "falling back to goal=0 and leaving xG as-is."
                )
                if "goal" not in df.columns:
                    df = df.copy()
                    df["goal"] = 0
            else:
                lcol, rcol = key_pair
                cols = [rcol]
                if "goal" in events.columns:
                    cols.append("goal")
                if "shot_statsbomb_xg" in events.columns:
                    cols.append("shot_statsbomb_xg")

                lookup = events[cols].drop_duplicates(rcol)
                merged = df.merge(
                    lookup, left_on=lcol, right_on=rcol, how="left", suffixes=("", "_ev")
                )

                if "goal" not in df.columns and "goal" in merged.columns:
                    merged["goal"] = (
                        pd.to_numeric(merged["goal"], errors="coerce").fillna(0).astype(int)
                    )

                if rcol != lcol and rcol in merged.columns:
                    merged = merged.drop(columns=[rcol])

                df = merged
                logger.info(
                    "Attached shots labels via %s -> %s from events.parquet",
                    lcol,
                    rcol,
                )

        except FileNotFoundError:
            logger.warning("events.parquet unavailable; adding fallback goal=0 to shots.")
            if "goal" not in df.columns:
                df = df.copy()
                df["goal"] = 0

    logger.info("Loaded shots.parquet: %d rows × %d cols", len(df), len(df.columns))
    return df


@lru_cache(maxsize=1)
def load_actions() -> pd.DataFrame:
    path = _FEATURES_DIR / "actions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"actions.parquet not found at {path}. Run scripts/build_features.py first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded actions.parquet: %d rows × %d cols", len(df), len(df.columns))
    return df


@lru_cache(maxsize=1)
def load_events() -> pd.DataFrame:
    """Raw processed events — contains shot_statsbomb_xg."""
    path = _PROCESSED_DIR / "events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"events.parquet not found at {path}. Run scripts/ingest.py first.")
    df = pd.read_parquet(path)

    # Normalise event identifier naming across datasets.
    # processed/events.parquet typically uses `id`, while feature tables use `event_id`.
    if "event_id" not in df.columns and "id" in df.columns:
        df = df.copy()
        df["event_id"] = df["id"]

    logger.info("Loaded events.parquet: %d rows × %d cols", len(df), len(df.columns))
    return df


@lru_cache(maxsize=1)
def load_matches() -> pd.DataFrame:
    path = _PROCESSED_DIR / "matches.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"matches.parquet not found at {path}. Run scripts/ingest.py first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded matches.parquet: %d rows", len(df))
    return df


# ── Held-out split resolution ───────────────────────────────────────────────────

# Final external held-out competition/season: UEFA Euro 2024 (StatsBomb ids).
HELDOUT_COMPETITION_ID = 55
HELDOUT_SEASON_ID = 282


def heldout_match_ids(
    competition_id: int = HELDOUT_COMPETITION_ID,
    season_id: int = HELDOUT_SEASON_ID,
) -> set[str]:
    """Return the set of match internal_ids for the held-out competition/season.

    The feature tables (shots.parquet, actions.parquet) carry a hashed
    competition_id and drop season_id, so they cannot be filtered on the raw
    StatsBomb ids directly. matches.parquet keeps the real integer ids, so we
    resolve the held-out matches there and return their internal_ids for joining.
    """
    matches = load_matches()
    for col in ("competition_id", "season_id", "internal_id"):
        if col not in matches.columns:
            raise KeyError(f"matches.parquet is missing required column {col!r}.")
    sel = matches[
        (matches["competition_id"] == competition_id) & (matches["season_id"] == season_id)
    ]
    if sel.empty:
        raise ValueError(
            f"No matches for competition {competition_id}, season {season_id} in matches.parquet."
        )
    return set(sel["internal_id"].astype(str))


def heldout_mask(
    df: pd.DataFrame,
    competition_id: int = HELDOUT_COMPETITION_ID,
    season_id: int = HELDOUT_SEASON_ID,
) -> pd.Series:
    """Boolean mask selecting rows of a shots/actions frame in the held-out split.

    Joins the frame to matches.parquet via match_internal_id (falling back to
    match_id) rather than relying on a competition_id/season_id column in the
    frame itself.
    """
    ids = heldout_match_ids(competition_id, season_id)
    key = "match_internal_id" if "match_internal_id" in df.columns else "match_id"
    if key not in df.columns:
        raise KeyError(
            "Frame has no match_internal_id/match_id column to resolve the held-out split."
        )
    return df[key].astype(str).isin(ids)


# ── Label derivation ───────────────────────────────────────────────────────────


def derive_shot_created(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Attach 'shot_created' binary label to actions DataFrame.

    shot_created = 1 if the action's possession contains at least one shot
    in features.parquet. Replicates the logic in scripts/train_cxa.attach_labels().
    """
    if "shot_created" in actions.columns:
        return actions

    features = load_features()
    poss_col = _find_col(features, ["possession_id"])
    act_poss_col = _find_col(actions, ["possession_id"])
    type_col = _find_col(features, ["event_type", "action_type"])

    if poss_col and act_poss_col and type_col:
        shot_poss_ids = set(
            features.loc[features[type_col].astype(str) == "shot", poss_col].dropna()
        )
        out = actions.copy()
        out["shot_created"] = out[act_poss_col].isin(shot_poss_ids).astype(int)
    else:
        logger.warning(
            "Cannot derive shot_created — missing possession_id or event_type columns. "
            "Returning column filled with 0."
        )
        out = actions.copy()
        out["shot_created"] = 0

    logger.info(
        "shot_created derived: %.2f%% positive (%d / %d rows)",
        100 * out["shot_created"].mean(),
        out["shot_created"].sum(),
        len(out),
    )
    return out


def derive_shot_in_possession(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach 'shot_in_possession' binary label.

    Identical logic to derive_shot_created but named differently to signal
    this is the CxT pre-modelling proxy.
    NOTE: The real CxT training target is 'possession_cxg', computed by
          compute_possession_cxg() in src/models/cxt/state_value_model.py
          during train_cxt.py — that requires fitted CxG predictions.
    """
    if "shot_in_possession" in df.columns:
        return df

    features = load_features()
    poss_col = _find_col(features, ["possession_id"])
    df_poss_col = _find_col(df, ["possession_id"])
    type_col = _find_col(features, ["event_type", "action_type"])

    if poss_col and df_poss_col and type_col:
        shot_poss_ids = set(
            features.loc[features[type_col].astype(str) == "shot", poss_col].dropna()
        )
        out = df.copy()
        out["shot_in_possession"] = out[df_poss_col].isin(shot_poss_ids).astype(int)
    else:
        logger.warning("Cannot derive shot_in_possession — filling with 0.")
        out = df.copy()
        out["shot_in_possession"] = 0

    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ── Figure / JSON persistence ──────────────────────────────────────────────────


def save_fig(name: str, subfolder: str, dpi: int = 150) -> Path:
    """Save current matplotlib figure to reports/figures/{subfolder}/{name}.png."""
    out_dir = _FIGURES_DIR / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved figure: %s", path.relative_to(_ROOT))
    return path


def save_json(data: dict, name: str) -> Path:
    """Save data to reports/{name}.json."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    logger.info("Saved JSON: %s", path.relative_to(_ROOT))
    return path


def _json_default(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


# ── Metadata helpers ───────────────────────────────────────────────────────────


def competition_labels(df: pd.DataFrame | None = None) -> dict[str, str]:
    """
    Return {competition_id_str: "Name (Season)"} from matches.parquet.
    Falls back to configs/competitions.yaml if parquet not available.
    """
    try:
        matches = load_matches()
        if "competition_id" in matches.columns and "competition_name" in matches.columns:
            labels: dict[str, str] = {}
            for _, row in matches.drop_duplicates("competition_id").iterrows():
                cid = str(row["competition_id"])
                cname = str(row.get("competition_name", cid))
                season = str(row.get("season_name", row.get("season_id", "")))
                labels[cid] = f"{cname} ({season})" if season else cname
            return labels
    except FileNotFoundError:
        pass

    # Fallback: parse competitions.yaml
    comp_yaml = _ROOT / "configs" / "competitions.yaml"
    if comp_yaml.exists():
        with open(comp_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        labels = {}
        for c in cfg.get("competitions", []):
            cid = str(c["competition_id"])
            name = c.get("competition_name", cid)
            season = c.get("season_name", str(c.get("season_id", "")))
            labels[cid] = f"{name} ({season})"
        return labels

    return {}


def feature_groups() -> dict[str, list[str]]:
    """Parse configs/features.yaml → {group_name: [column_names]}."""
    with open(_FEATURE_REGISTRY, encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    groups: dict[str, list[str]] = {}
    skip = {"identifiers"}
    for group_name, entries in registry.items():
        if group_name in skip:
            continue
        if isinstance(entries, list):
            groups[group_name] = [e["name"] for e in entries if isinstance(e, dict) and "name" in e]
    return groups


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """All float32/int columns that are actual feature values (not IDs)."""
    id_cols = {
        "player_id",
        "team_id",
        "opponent_id",
        "competition_id",
        "match_id",
        "possession_id",
        "event_id",
    }
    return [c for c in df.select_dtypes(include=["number"]).columns if c not in id_cols]


def categorical_feature_cols(df: pd.DataFrame) -> list[str]:
    """All category/object columns that are actual feature values (not IDs)."""
    id_cols = {
        "player_id",
        "team_id",
        "opponent_id",
        "competition_id",
        "match_id",
        "possession_id",
        "event_id",
    }
    return [c for c in df.select_dtypes(include=["category", "object"]).columns if c not in id_cols]
