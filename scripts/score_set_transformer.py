"""
scripts/score_set_transformer.py
================================
Score shots with a trained ``SetTransformerCxGModel`` and write a parquet of
per-shot probabilities.

Usage
-----
    python scripts/score_set_transformer.py
    python scripts/score_set_transformer.py \
        --model models/cxg/set_transformer_360.joblib \
        --shots data/features/shots.parquet \
        --frames data/processed/frames.parquet \
        --output outputs/scores/cxg_set_transformer.parquet

Notes
-----
* Runs entirely on CPU when ``CFM_PROFILE=cpu`` (default if no GPU).
* Shots whose ``event_internal_id`` (or ``event_id``) has no matching frame
  in the freeze-frame parquet are scored using the tabular branch alone
  (set tokens are fully padded → masked out by the SetTransformer).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.models.cxg.set_transformer_model import SetTransformerCxGModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("score_set_transformer")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "models" / "cxg" / "set_transformer_360.joblib")
    parser.add_argument("--shots", type=Path,
                        default=PROJECT_ROOT / "data" / "features" / "shots.parquet")
    parser.add_argument("--frames", type=Path, default=None,
                        help="Override frames parquet path (defaults to model's saved path).")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs" / "scores" / "cxg_set_transformer.parquet")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    args = parser.parse_args()

    if not args.model.exists():
        logger.error("Model file not found: %s", args.model)
        logger.error("Train it first:  python scripts/train_cxg.py --include-neural")
        return 1
    if not args.shots.exists():
        logger.error("Shots file not found: %s", args.shots)
        return 1

    logger.info("Loading model: %s", args.model)
    model = SetTransformerCxGModel.load(args.model)
    model.device = args.device  # honour CLI override at predict time

    if args.frames is not None:
        model.frames_path = str(args.frames)
        model._frames_cache = None  # force reload from new path

    logger.info("Loading shots: %s", args.shots)
    shots = pd.read_parquet(args.shots)
    logger.info("Scoring %d shots …", len(shots))
    proba = model.predict_proba(shots)

    out = shots.copy()
    out["cxg_set_transformer"] = proba
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    logger.info("Wrote %s  (mean prob = %.4f)", args.output, float(proba.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
