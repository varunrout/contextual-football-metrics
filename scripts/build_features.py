"""
scripts/build_features.py
==========================
Read the processed parquet tables written by ingest.py and build the
full event-level feature store that all three metric models consume.

Outputs (under data/features/):
  features.parquet     — event-level feature table for all competitions
  shots.parquet        — shot rows only (used by CxG training)
  actions.parquet      — creative actions (passes + carries + cutbacks,
                         used by CxA training)

Usage
-----
    python scripts/build_features.py
    python scripts/build_features.py --split-role train
    python scripts/build_features.py --input-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.feature_store import build_feature_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_features")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURE_REGISTRY = PROJECT_ROOT / "configs" / "features.yaml"

# Action types treated as creative (eligible for CxA scoring)
_CREATIVE_TYPES = {"pass", "cross", "carry", "cutback"}
# Action types included in CxT
_CXT_TYPES = {"pass", "cross", "carry", "cutback"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        logger.error(
            "%s not found at %s — run scripts/ingest.py first.", name, path
        )
        sys.exit(1)
    df = pd.read_parquet(path)
    logger.info("Loaded %-20s  %d rows", name, len(df))
    return df


def _filter_split_role(events_df: pd.DataFrame, matches_df: pd.DataFrame, split_role: str | None) -> pd.DataFrame:
    """Filter events to matches with the requested split_role (or all if None)."""
    if split_role is None or "split_role" not in matches_df.columns:
        return events_df

    allowed_roles = set(split_role.split(","))
    # split_role can be compound, e.g. "train_val" means usable for both
    valid_matches = matches_df[
        matches_df["split_role"].apply(
            lambda r: bool(set(str(r).split("_")) & allowed_roles)
        )
    ]["internal_id"]
    logger.info("Filtering to split_role=%r → %d matches", split_role, len(valid_matches))
    return events_df[events_df["match_internal_id"].isin(valid_matches)]


# ── Main ──────────────────────────────────────────────────────────────────────

def build_features(
    input_dir: Path = PROCESSED_DIR,
    output_dir: Path = FEATURES_DIR,
    split_role: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    events_raw = _load_table(input_dir / "events.parquet", "events.parquet")
    matches_df = _load_table(input_dir / "matches.parquet", "matches.parquet")

    poss_path = input_dir / "possessions.parquet"
    possessions_df = pd.read_parquet(poss_path) if poss_path.exists() else pd.DataFrame()
    if possessions_df.empty:
        logger.warning("possessions.parquet not found or empty — possession context may be incomplete.")

    frames_path = input_dir / "frames.parquet"
    frames_df = pd.read_parquet(frames_path) if frames_path.exists() else pd.DataFrame()
    if frames_df.empty:
        logger.warning("frames.parquet not found or empty — 360 freeze-frame features will be sparse.")
    else:
        logger.info("Loaded %-20s  %d rows", "frames.parquet", len(frames_df))

    # Optionally restrict to a specific split role
    events_raw = _filter_split_role(events_raw, matches_df, split_role)
    if events_raw.empty:
        logger.error("No events remain after split_role filter. Aborting.")
        sys.exit(1)

    logger.info("Building feature store for %d events …", len(events_raw))
    features_df = build_feature_store(
        events_df=events_raw,
        possessions_df=possessions_df,
        frames_df=frames_df,
        matches_df=matches_df,
        feature_registry_path=str(FEATURE_REGISTRY),
    )

    if features_df.empty:
        logger.error("Feature store returned empty DataFrame. Check input data.")
        sys.exit(1)

    logger.info("Feature store built: %d rows × %d columns", len(features_df), len(features_df.columns))

    # ── Save full feature table ───────────────────────────────────────────────
    out_path = output_dir / "features.parquet"
    features_df.to_parquet(out_path, index=False)
    logger.info("Saved features.parquet  (%d rows)", len(features_df))

    # ── Shot subset (for CxG) ─────────────────────────────────────────────────
    _type_col = "action_type" if "action_type" in features_df.columns else "event_type"
    if _type_col in features_df.columns:
        shots_df = features_df[features_df[_type_col] == "shot"].copy()

        # Attach outcome labels from events_raw — they are not computed by the
        # feature store (which only produces context features, not targets).
        _outcome_cols = [c for c in ("goal", "shot_outcome") if c in events_raw.columns]
        if _outcome_cols:
            _ev_labels = events_raw[["internal_id"] + _outcome_cols].rename(
                columns={"internal_id": "event_id"}
            )
            shots_df = shots_df.merge(_ev_labels, on="event_id", how="left")
            logger.info("Attached outcome columns to shots: %s", _outcome_cols)
        else:
            logger.warning("No outcome columns (goal, shot_outcome) found in events — shots.parquet will lack targets.")

        shots_path = output_dir / "shots.parquet"
        shots_df.to_parquet(shots_path, index=False)
        logger.info("Saved shots.parquet     (%d rows)", len(shots_df))
    else:
        logger.warning("Cannot identify shot rows — shots.parquet not written.")

    # ── Creative actions subset (for CxA) ─────────────────────────────────────
    if _type_col in features_df.columns:
        actions_df = features_df[features_df[_type_col].isin(_CREATIVE_TYPES)].copy()
        actions_path = output_dir / "actions.parquet"
        actions_df.to_parquet(actions_path, index=False)
        logger.info("Saved actions.parquet   (%d rows)", len(actions_df))
    else:
        logger.warning("action_type column not present — actions.parquet not written.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the feature store from processed parquet tables."
    )
    p.add_argument(
        "--input-dir",
        default=str(PROCESSED_DIR),
        help="Directory containing events.parquet / matches.parquet (default: data/processed).",
    )
    p.add_argument(
        "--output-dir",
        default=str(FEATURES_DIR),
        help="Directory to write feature parquet files (default: data/features).",
    )
    p.add_argument(
        "--split-role",
        default=None,
        metavar="ROLE",
        help="Only include events from matches whose split_role contains ROLE. "
             "Comma-separate multiple roles, e.g. 'train' or 'train,train_val'.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    build_features(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        split_role=args.split_role,
    )
