"""
scripts/score_state_value_gnn.py
=================================
Score CxT-eligible actions (pass/carry/cross/cutback) with a trained
freeze-frame-aware CxT state-value model — either the GNN (graph-attention)
or SetTransformer variant — and write a per-action parquet of predicted
``possession_cxg``.

Mirrors the pattern of scripts/score_set_transformer.py (CxG) and
scripts/score_gnn_cxa.py (CxA), which previously had no CxT counterpart.

Usage
-----
    python scripts/score_state_value_gnn.py
    python scripts/score_state_value_gnn.py --model-class set_transformer
    python scripts/score_state_value_gnn.py \
        --model models/cxt/gnn_contextual.joblib \
        --actions data/features/actions.parquet \
        --frames data/processed/frames.parquet \
        --output outputs/scores/cxt_gnn.parquet

Notes
-----
* Runs entirely on CPU when ``CFM_PROFILE=cpu`` (default if no GPU).
* Actions whose event has no matching freeze-frame are scored using the
  tabular branch alone (set/graph tokens are fully padded → masked out).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("score_state_value_gnn")

_MODEL_CLASSES = {"gnn": "gnn_contextual", "set_transformer": "set_transformer_contextual"}


def _load_model_class(model_class: str):
    if model_class == "gnn":
        from src.models.cxt.state_value_gnn import GNNStateValueModel

        return GNNStateValueModel
    from src.models.cxt.state_value_set_transformer import SetTransformerStateValueModel

    return SetTransformerStateValueModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-class", choices=sorted(_MODEL_CLASSES), default="gnn",
                        help="Which freeze-frame CxT state-value variant to load (default: gnn).")
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to the trained model .joblib "
                             "(default: models/cxt/<gnn|set_transformer>_contextual.joblib).")
    parser.add_argument("--actions", type=Path,
                        default=PROJECT_ROOT / "data" / "features" / "actions.parquet")
    parser.add_argument("--frames", type=Path, default=None,
                        help="Override frames parquet path (defaults to model's saved path).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: outputs/scores/cxt_<model-class>.parquet")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    args = parser.parse_args()

    model_path = args.model or (
        PROJECT_ROOT / "models" / "cxt" / f"{_MODEL_CLASSES[args.model_class]}.joblib"
    )
    output_path = args.output or (
        PROJECT_ROOT / "outputs" / "scores" / f"cxt_{args.model_class}.parquet"
    )

    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        logger.error(
            "Train it first:  python scripts/train_cxt.py --include-neural "
            "--frames data/processed/frames.parquet --no-promote"
        )
        return 1
    if not args.actions.exists():
        logger.error("Actions file not found: %s", args.actions)
        return 1

    model_cls = _load_model_class(args.model_class)
    logger.info("Loading %s model: %s", args.model_class, model_path)
    model = model_cls.load(model_path)
    model.device = args.device  # honour CLI override at predict time

    if args.frames is not None:
        model.frames_path = str(args.frames)
        model._frames_cache = None  # force reload from new path

    logger.info("Loading actions: %s", args.actions)
    actions = pd.read_parquet(args.actions)
    logger.info("Scoring %d actions …", len(actions))
    predicted = model.predict(actions)

    out = actions.copy()
    out[f"cxt_{args.model_class}"] = predicted
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    logger.info("Wrote %s  (mean predicted possession_cxg = %.5f)", output_path, float(predicted.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
