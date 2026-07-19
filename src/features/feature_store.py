"""Unified feature store builder for CxG/CxA/CxT."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.features.freeze_frame_features import build_freeze_frame_features
from src.features.match_context_features import build_match_context_features
from src.features.opponent_features import build_opponent_features
from src.features.sequence_features import compute_sequence_features
from src.features.sequence_labeler import label_possessions_dataframe
from src.features.traditional_features import build_traditional_features

logger = logging.getLogger("feature_store")


def load_feature_registry(
    path: str | Path = "configs/features.yaml",
) -> dict[str, list[dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def _feature_specs(registry: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for group in ["traditional", "freeze_frame", "opponent", "match_context", "sequence"]:
        specs.extend(registry.get(group, []))
    return specs


def _opponent_lookup(events_df: pd.DataFrame, matches_df: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    if (
        matches_df is not None
        and not matches_df.empty
        and {"internal_id", "home_team_internal_id", "away_team_internal_id"}.issubset(
            matches_df.columns
        )
    ):
        for _, row in matches_df.iterrows():
            mid = row["internal_id"]
            home = row["home_team_internal_id"]
            away = row["away_team_internal_id"]
            rows.append({"match_internal_id": mid, "team_internal_id": home, "opponent_id": away})
            rows.append({"match_internal_id": mid, "team_internal_id": away, "opponent_id": home})
        return pd.DataFrame(rows)

    for mid, grp in events_df.groupby("match_internal_id"):
        teams = list(grp["team_internal_id"].dropna().unique())
        if len(teams) >= 2:
            a, b = teams[0], teams[1]
            rows.append({"match_internal_id": mid, "team_internal_id": a, "opponent_id": b})
            rows.append({"match_internal_id": mid, "team_internal_id": b, "opponent_id": a})
    return pd.DataFrame(rows)


def _apply_missing_policy(df: pd.DataFrame, col: str, policy: str, dtype: str) -> pd.DataFrame:
    if col not in df.columns:
        return df

    if policy == "allowed":
        return df

    if policy == "drop_row":
        return df[df[col].notna()].copy()

    if policy == "impute_zero":
        if dtype == "bool":
            df[col] = df[col].fillna(False)
        else:
            df[col] = df[col].fillna(0)
        return df

    if policy == "impute_median":
        med = pd.to_numeric(df[col], errors="coerce").median()
        if np.isnan(med):
            med = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(float(med))
        return df

    if policy == "impute_mode":
        mode = df[col].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "unknown"
        df[col] = df[col].fillna(fill)
        return df

    if policy == "impute_constant":
        fill = False if dtype == "bool" else "unknown"
        df[col] = df[col].fillna(fill)
        return df

    if policy == "flag_and_zero":
        flag_col = f"has_{col}"
        df[flag_col] = df[col].notna().astype(bool)
        if dtype == "bool":
            df[col] = df[col].fillna(False)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    return df


def _cast_dtype(df: pd.DataFrame, col: str, dtype: str) -> pd.DataFrame:
    if col not in df.columns:
        return df

    if dtype in {"float32", "float64"}:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    elif dtype in {"int8", "int16", "int32"}:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)
    elif dtype == "bool":
        df[col] = df[col].astype(bool)
    elif dtype == "category":
        df[col] = df[col].astype("category")
    elif dtype == "str":
        df[col] = df[col].astype(str)
    return df


def build_feature_store(
    events_df: pd.DataFrame,
    possessions_df: pd.DataFrame,
    frames_df: pd.DataFrame | None = None,
    matches_df: pd.DataFrame | None = None,
    feature_registry_path: str | Path = "configs/features.yaml",
) -> pd.DataFrame:
    """Build the full event-level feature table for all three metrics."""
    if events_df.empty:
        return pd.DataFrame()

    frames_df = frames_df if frames_df is not None else pd.DataFrame()

    base = pd.DataFrame(
        {
            "event_id": events_df.get("internal_id"),
            "match_id": events_df.get("match_internal_id"),
            "possession_id": events_df.get("possession_internal_id"),
            "team_id": events_df.get("team_internal_id"),
            "player_id": events_df.get("player_internal_id"),
            "competition_id": events_df.get("competition_internal_id", np.nan),
        }
    )

    opp_lookup = _opponent_lookup(events_df, matches_df)
    if not opp_lookup.empty:
        base = base.merge(
            opp_lookup,
            left_on=["match_id", "team_id"],
            right_on=["match_internal_id", "team_internal_id"],
            how="left",
        ).drop(columns=["match_internal_id", "team_internal_id"])
    else:
        base["opponent_id"] = np.nan

    t0 = time.time()
    logger.info("  Step 1/5 — traditional features …")
    traditional = build_traditional_features(events_df)
    logger.info("  Step 1/5 — done (%.1fs, %d rows)", time.time() - t0, len(traditional))

    t0 = time.time()
    logger.info("  Step 2/5 — freeze-frame features …")
    freeze_frame = build_freeze_frame_features(events_df, frames_df)
    logger.info("  Step 2/5 — done (%.1fs, %d rows)", time.time() - t0, len(freeze_frame))

    t0 = time.time()
    logger.info("  Step 3/5 — opponent features …")
    opponent = build_opponent_features(events_df, matches_df)
    logger.info("  Step 3/5 — done (%.1fs, %d rows)", time.time() - t0, len(opponent))

    t0 = time.time()
    logger.info("  Step 4/5 — match context features …")
    match_context = build_match_context_features(events_df, matches_df)
    logger.info("  Step 4/5 — done (%.1fs, %d rows)", time.time() - t0, len(match_context))

    # Sequence features are possession-level, then projected to event-level.
    # compute_sequence_features expects raw-style columns (`type`, `location`, etc.).
    events_for_sequence = events_df.copy()

    if "type" not in events_for_sequence.columns and "action_type" in events_for_sequence.columns:
        _type_map = {
            "pass": "Pass",
            "carry": "Carry",
            "shot": "Shot",
            "pressure": "Pressure",
            "ball_receipt": "Ball Receipt*",
            "dribble": "Dribble",
            "clearance": "Clearance",
            "interception": "Interception",
            "foul_committed": "Foul Committed",
        }
        _at_lower = events_for_sequence["action_type"].str.lower().fillna("")
        events_for_sequence["type"] = _at_lower.map(_type_map).fillna(
            events_for_sequence["action_type"].str.replace("_", " ", regex=False).str.title()
        )

    # location and pass dicts not needed — sequence helpers use flat x/y/end_x/end_y directly

    if "possession" not in events_for_sequence.columns and not possessions_df.empty:
        poss_map = possessions_df[["internal_id", "possession_index"]].rename(
            columns={"internal_id": "possession_internal_id", "possession_index": "possession"}
        )
        events_for_sequence = events_for_sequence.merge(
            poss_map,
            on="possession_internal_id",
            how="left",
        )

    t0 = time.time()
    logger.info("  Step 5/5 — sequence features …")
    seq_poss = compute_sequence_features(possessions_df.copy(), events_for_sequence)
    seq_poss = label_possessions_dataframe(seq_poss, events_for_sequence)
    seq_cols = [
        "internal_id",
        "sequence_type",
        "sequence_type_confidence",
        "time_from_possession_start",
        "events_before_action",
        "passes_before_action",
        "carries_before_action",
        "vertical_progression_speed",
        "possession_start_zone",
        "regain_zone",
        "final_action_type",
        "final_pass_zone",
        "set_piece_flag",
        "counterpress_regain_flag",
        "number_of_switches",
        "directness",
        "possession_speed",
        "transition_or_settled",
        "phase_of_play",
    ]
    seq_available = [c for c in seq_cols if c in seq_poss.columns]
    # Join via (match_internal_id, possession_index) — possession_internal_id hashes
    # in events do NOT match possessions.internal_id, so we use the integer index.
    seq_feature_cols = [c for c in seq_available if c != "internal_id"]
    seq_join = ["match_internal_id", "possession_index"] + seq_feature_cols
    seq_join = [c for c in seq_join if c in seq_poss.columns]
    ev_poss_col = "possession" if "possession" in events_df.columns else None
    if (
        ev_poss_col
        and "match_internal_id" in events_df.columns
        and "possession_index" in seq_poss.columns
    ):
        seq_event = (
            events_df[["internal_id", "match_internal_id", ev_poss_col]]
            .merge(
                seq_poss[seq_join],
                left_on=["match_internal_id", ev_poss_col],
                right_on=["match_internal_id", "possession_index"],
                how="left",
            )
            .drop(columns=["possession_index", "match_internal_id", ev_poss_col], errors="ignore")
        )
    else:
        # Fallback: try the old hash-based join
        seq_event = (
            events_df[["internal_id", "possession_internal_id"]]
            .merge(
                seq_poss[[c for c in seq_available]],
                left_on="possession_internal_id",
                right_on="internal_id",
                how="left",
                suffixes=("", "_poss"),
            )
            .drop(columns=["internal_id_poss"], errors="ignore")
        )
    logger.info("  Step 5/5 — done (%.1fs, %d rows)", time.time() - t0, len(seq_event))

    logger.info("  Merging all feature blocks …")
    t0 = time.time()
    merged = base.copy()
    for block in [traditional, freeze_frame, opponent, match_context, seq_event]:
        if block.empty:
            continue
        join_col = "event_internal_id" if "event_internal_id" in block.columns else "internal_id"
        overlap = [c for c in block.columns if c in merged.columns and c != join_col]
        if overlap:
            block = block.drop(columns=overlap)
        merged = merged.merge(block, left_on="event_id", right_on=join_col, how="left")
        merged = merged.drop(columns=[join_col], errors="ignore")

    logger.info(
        "  Merging done (%.1fs) — %d rows, %d cols",
        time.time() - t0,
        len(merged),
        len(merged.columns),
    )

    logger.info("  Applying missing policies and dtype casts …")
    t0 = time.time()
    registry = load_feature_registry(feature_registry_path)
    specs = _feature_specs(registry)

    for spec in specs:
        col = spec["name"]
        dtype = spec.get("dtype", "float32")
        policy = spec.get("missing_policy", "allowed")

        if col not in merged.columns:
            merged[col] = np.nan

        merged = _apply_missing_policy(merged, col, policy, dtype)
        merged = _cast_dtype(merged, col, dtype)

    logger.info("  Policies applied (%.1fs)", time.time() - t0)

    # Keep identifiers from config.
    id_cols = [
        "player_id",
        "team_id",
        "opponent_id",
        "competition_id",
        "match_id",
        "possession_id",
        "event_id",
    ]
    front = [c for c in id_cols if c in merged.columns]
    rest = [c for c in merged.columns if c not in front]
    return merged[front + rest]
