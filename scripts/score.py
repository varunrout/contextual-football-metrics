"""
scripts/score.py
================
Score a parquet of match events with the trained InferencePipeline,
writing CxG / CxA / CxT columns to an output parquet file.

The pipeline is loaded from configs/models.yaml production pointers
(or overridden with explicit --cxg / --cxa / --cxt flags).

Outputs (under outputs/scores/):
  <competition_id>_<season_id>_scored.parquet — full events with metric columns added

Usage
-----
    python scripts/score.py --events data/features/features.parquet
    python scripts/score.py --competition 55 --season 282
    python scripts/score.py --events data/features/features.parquet \
        --cxg models/cxg/cxg_lgbm_contextual_abc123.pkl \
        --output outputs/scores/my_scores.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.inference_pipeline import InferencePipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("score")

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "scores"
MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Pitch constants (normalised: 105×68m)
_BOX_X_MIN = 88.5
_BOX_Y_MIN = 13.84
_BOX_Y_MAX = 54.16
_CENTRAL_Y_MIN = 27.0
_CENTRAL_Y_MAX = 41.0


# ── Feature enrichment ────────────────────────────────────────────────────────


def _enrich_for_scoring(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns required by CxG and CxT models that are absent from features.parquet:

    - ``action_type``  : alias of ``event_type`` (CxG shot-filter, CxT action-filter)
    - ``in_box``       : bool — ball location inside penalty box
    - ``is_central``   : bool — ball location in central channel
    - ``end_x``/``end_y`` : end-of-action location (from processed events, needed by CxT)
    """
    df = events_df.copy()

    # action_type alias
    if "action_type" not in df.columns and "event_type" in df.columns:
        df["action_type"] = df["event_type"]

    # Spatial booleans derived from ball location
    if "in_box" not in df.columns and "x_location" in df.columns and "y_location" in df.columns:
        df["in_box"] = (
            (df["x_location"] >= _BOX_X_MIN)
            & (df["y_location"] >= _BOX_Y_MIN)
            & (df["y_location"] <= _BOX_Y_MAX)
        ).astype(float)

    if "is_central" not in df.columns and "y_location" in df.columns:
        df["is_central"] = (
            (df["y_location"] >= _CENTRAL_Y_MIN) & (df["y_location"] <= _CENTRAL_Y_MAX)
        ).astype(float)

    # Merge end_x / end_y from processed events (needed for CxT V(after) computation)
    if "end_x" not in df.columns or "end_y" not in df.columns:
        raw_events_path = PROCESSED_DIR / "events.parquet"
        if raw_events_path.exists() and "event_id" in df.columns:
            try:
                raw = pd.read_parquet(
                    raw_events_path,
                    columns=["internal_id", "end_x", "end_y"],
                )
                df = df.merge(
                    raw.rename(columns={"internal_id": "event_id"}),
                    on="event_id",
                    how="left",
                    suffixes=("", "_raw"),
                )
                # Prefer already-present columns; drop _raw duplicates if any
                for col in ("end_x", "end_y"):
                    raw_col = f"{col}_raw"
                    if raw_col in df.columns:
                        df[col] = df[col].combine_first(df[raw_col])
                        df = df.drop(columns=[raw_col])
                logger.info(
                    "Merged end_x/end_y: %.0f%% of rows have end_x",
                    df["end_x"].notna().mean() * 100,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not merge end_x/end_y from processed events: %s", exc)

    return df


def _load_pipeline(
    config_path: Path,
    cxg_override: Path | None,
    cxa_override: Path | None,
    cxt_override: Path | None,
) -> InferencePipeline:
    """
    Load InferencePipeline.

    If any override is supplied, load just that model and leave the others
    at whatever the config specifies.  If all three are None, fall back to
    from_config() which reads production pointers from models.yaml.
    """
    import pickle

    def _load_pkl(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301

    # Start from config
    try:
        pipeline = InferencePipeline.from_config(config_path, models_dir=PROJECT_ROOT)
    except Exception as exc:
        logger.warning("from_config() failed (%s) — building empty pipeline.", exc)
        pipeline = InferencePipeline()

    # Apply overrides
    if cxg_override:
        pipeline.cxg_model = _load_pkl(cxg_override)
        logger.info("CxG override loaded from %s", cxg_override)
    if cxa_override:
        pipeline.cxa_pipeline = _load_pkl(cxa_override)
        logger.info("CxA override loaded from %s", cxa_override)
    if cxt_override:
        pipeline.cxt_pipeline = _load_pkl(cxt_override)
        logger.info("CxT override loaded from %s", cxt_override)

    return pipeline


# ── Main ──────────────────────────────────────────────────────────────────────


def score(
    events_path: Path,
    output_path: Path,
    config_path: Path = MODELS_YAML,
    cxg_override: Path | None = None,
    cxa_override: Path | None = None,
    cxt_override: Path | None = None,
    competition_filter: tuple[int, int] | None = None,
) -> None:
    if not events_path.exists():
        logger.error("Events file not found: %s", events_path)
        sys.exit(1)

    events_df = pd.read_parquet(events_path)
    logger.info("Loaded events: %d rows × %d columns", len(events_df), len(events_df.columns))

    # Optional competition filter
    if competition_filter is not None:
        cid, sid = competition_filter
        for col in ("competition_id", "competition_internal_id"):
            if col in events_df.columns:
                events_df = events_df[events_df[col].astype(str).str.contains(str(cid))]
                break
        logger.info("After competition filter (%s/%s): %d rows", cid, sid, len(events_df))

    if events_df.empty:
        logger.error("No events to score.")
        sys.exit(1)

    pipeline = _load_pipeline(config_path, cxg_override, cxa_override, cxt_override)
    logger.info("Pipeline: %s", pipeline)

    if (
        pipeline.cxg_model is None
        and pipeline.cxa_pipeline is None
        and pipeline.cxt_pipeline is None
    ):
        logger.warning(
            "All production model pointers are null in configs/models.yaml. "
            "Train models first with train_cxg.py / train_cxa.py / train_cxt.py, "
            "or supply explicit --cxg / --cxa / --cxt overrides."
        )

    logger.info("Scoring %d events …", len(events_df))
    events_df = _enrich_for_scoring(events_df)
    scored_df = pipeline.score(events_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(output_path, index=False)
    logger.info("Scored output saved to %s  (%d rows)", output_path, len(scored_df))

    # Summary stats
    for col in ("cxg", "cxa", "cxt"):
        if col in scored_df.columns:
            valid = scored_df[col].dropna()
            logger.info(
                "  %s: n=%d  mean=%.4f  max=%.4f",
                col,
                len(valid),
                valid.mean() if len(valid) else float("nan"),
                valid.max() if len(valid) else float("nan"),
            )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score events with the trained InferencePipeline.")
    p.add_argument(
        "--events",
        default=str(FEATURES_DIR / "features.parquet"),
        help="Feature parquet to score (default: data/features/features.parquet).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output parquet path. Default: outputs/scores/scored.parquet",
    )
    p.add_argument(
        "--config",
        default=str(MODELS_YAML),
        help="Path to models.yaml (default: configs/models.yaml).",
    )
    p.add_argument("--cxg", default=None, help="Override CxG model pickle path.")
    p.add_argument("--cxa", default=None, help="Override CxA pipeline pickle path.")
    p.add_argument("--cxt", default=None, help="Override CxT pipeline pickle path.")
    p.add_argument(
        "--competition",
        type=int,
        default=None,
        help="Filter to a specific competition ID.",
    )
    p.add_argument(
        "--season",
        type=int,
        default=None,
        help="Filter to a specific season ID (used with --competition).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    output = Path(args.output) if args.output else OUTPUTS_DIR / "scored.parquet"

    comp_filter = (args.competition, args.season) if args.competition is not None else None

    score(
        events_path=Path(args.events),
        output_path=output,
        config_path=Path(args.config),
        cxg_override=Path(args.cxg) if args.cxg else None,
        cxa_override=Path(args.cxa) if args.cxa else None,
        cxt_override=Path(args.cxt) if args.cxt else None,
        competition_filter=comp_filter,
    )
