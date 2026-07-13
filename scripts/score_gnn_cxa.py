"""
scripts/score_gnn_cxa.py
========================
Score creative actions with the GNN passing-network creation model paired
with a LightGBM shot-quality regressor and write a per-action parquet of
``cxa = p_shot_created * expected_cxg``.

Usage
-----
    python scripts/score_gnn_cxa.py
    python scripts/score_gnn_cxa.py \
        --creation models/cxa/shot_creation_gnn_passing_360.pkl \
        --quality  models/cxa/shot_quality_gnn_passing_360.pkl \
        --actions  data/features/actions.parquet \
        --output   outputs/scores/cxa_gnn_passing.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.models.cxa.gnn_passing_network import GNNPassingNetworkCxAModel
from src.models.cxa.shot_quality_model import ShotQualityModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("score_gnn_cxa")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--creation", type=Path,
                        default=PROJECT_ROOT / "models" / "cxa" / "shot_creation_gnn_passing_360.pkl")
    parser.add_argument("--quality", type=Path,
                        default=PROJECT_ROOT / "models" / "cxa" / "shot_quality_gnn_passing_360.pkl")
    parser.add_argument("--actions", type=Path,
                        default=PROJECT_ROOT / "data" / "features" / "actions.parquet")
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs" / "scores" / "cxa_gnn_passing.parquet")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    for f in (args.creation, args.quality, args.actions):
        if not f.exists():
            logger.error("Required file missing: %s", f)
            logger.error(
                "Train first:  python scripts/train_cxa.py --include-neural "
                "--frames data/processed/frames.parquet --no-promote"
            )
            return 1

    logger.info("Loading creation model: %s", args.creation)
    creation = GNNPassingNetworkCxAModel.load(args.creation)
    creation.device = args.device
    if args.frames is not None:
        creation.frames_path = str(args.frames)
        creation._frames_cache = None

    logger.info("Loading quality model: %s", args.quality)
    quality = ShotQualityModel.load(args.quality)

    logger.info("Loading actions: %s", args.actions)
    actions = pd.read_parquet(args.actions)
    logger.info("Scoring %d actions …", len(actions))

    p_creation = creation.predict_proba(actions)
    expected_cxg = quality.predict(actions)

    out = actions.copy()
    out["p_shot_created_gnn"] = p_creation
    out["expected_cxg"] = expected_cxg
    out["cxa_gnn"] = p_creation * expected_cxg

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    logger.info(
        "Wrote %s  (mean p_creation=%.4f, mean E[cxg]=%.4f, mean cxa=%.5f)",
        args.output,
        float(p_creation.mean()), float(expected_cxg.mean()), float((p_creation * expected_cxg).mean()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
