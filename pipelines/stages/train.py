"""Prefect tasks wrapping the training scripts.

Each task delegates to the existing ``train_<family>`` function in
``scripts/train_<family>.py``. The task layer adds:

* MLflow nested-run wrapping (via :mod:`pipelines.mlflow_helpers`)
* Profile-driven defaults (the active profile's ``data_root`` /
  ``models_root`` are used unless explicit paths are supplied)
* Caching keyed on (task name, profile, input hash)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
from prefect import task
from prefect.tasks import task_input_hash

from pipelines.mlflow_helpers import log_path, stage_run
from src.runtime import require_profile

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────


def _maybe_log_summary(summary_path: Path) -> None:
    """If a JSON training summary exists, log key scalar metrics + the file."""
    if not summary_path.exists():
        logger.warning("Summary not found at %s; skipping MLflow scalar log", summary_path)
        return
    try:
        payload = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Could not parse %s as JSON; logging as artifact only", summary_path)
        log_path(summary_path)
        return
    # Best-effort: log any top-level scalar fields as metrics.
    for k, v in payload.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            mlflow.log_metric(k, float(v))
    log_path(summary_path)


# ── tasks ────────────────────────────────────────────────────────────────────


@task(
    name="train_cxg",
    retries=0,
    cache_key_fn=task_input_hash,
    cache_expiration=None,
    tags=["modeling", "cxg"],
)
def train_cxg_task(
    parent_run_id: str | None = None,
    *,
    n_folds: int = 5,
    n_optuna_trials: int = 0,
    include_360: bool = False,
    promote: bool = True,
    random_state: int = 42,
    n_estimators: int = 300,
) -> dict[str, Any]:
    """Train all CxG candidates and promote the best."""
    prof = require_profile()
    shots_path = prof.data_root / "features" / "shots.parquet"

    from scripts.train_cxg import train_cxg

    with stage_run("train_cxg", parent_run_id=parent_run_id) as run:
        mlflow.log_params(
            {
                "n_folds": n_folds,
                "n_optuna_trials": n_optuna_trials,
                "include_360": include_360,
                "n_estimators": n_estimators,
                "random_state": random_state,
                "shots_path": str(shots_path),
            }
        )
        train_cxg(
            shots_path=shots_path,
            n_folds=n_folds,
            n_optuna_trials=n_optuna_trials,
            include_360=include_360,
            promote=promote,
            random_state=random_state,
            n_estimators=n_estimators,
        )
        _maybe_log_summary(prof.reports_root / "cxg_training_summary.json")
        return {
            "run_id": run.info.run_id,
            "summary": str(prof.reports_root / "cxg_training_summary.json"),
        }


@task(
    name="train_cxa",
    retries=0,
    cache_key_fn=task_input_hash,
    tags=["modeling", "cxa"],
)
def train_cxa_task(
    parent_run_id: str | None = None,
    *,
    feature_set: str = "contextual",
    n_folds: int = 5,
    n_estimators: int = 300,
    promote: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    prof = require_profile()
    actions_path = prof.data_root / "features" / "actions.parquet"
    features_path = prof.data_root / "features" / "features.parquet"

    from scripts.train_cxa import train_cxa

    with stage_run("train_cxa", parent_run_id=parent_run_id) as run:
        mlflow.log_params(
            {
                "feature_set": feature_set,
                "n_folds": n_folds,
                "n_estimators": n_estimators,
                "random_state": random_state,
                "actions_path": str(actions_path),
                "features_path": str(features_path),
            }
        )
        train_cxa(
            actions_path=actions_path,
            features_path=features_path,
            feature_set=feature_set,
            n_folds=n_folds,
            n_estimators=n_estimators,
            promote=promote,
            random_state=random_state,
        )
        _maybe_log_summary(prof.reports_root / "cxa_training_summary.json")
        return {
            "run_id": run.info.run_id,
            "summary": str(prof.reports_root / "cxa_training_summary.json"),
        }


@task(
    name="train_cxt",
    retries=0,
    cache_key_fn=task_input_hash,
    tags=["modeling", "cxt"],
)
def train_cxt_task(
    parent_run_id: str | None = None,
    *,
    n_folds: int = 5,
    n_estimators: int = 400,
    promote: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    prof = require_profile()
    features_path = prof.data_root / "features" / "features.parquet"

    from scripts.train_cxt import train_cxt

    with stage_run("train_cxt", parent_run_id=parent_run_id) as run:
        mlflow.log_params(
            {
                "n_folds": n_folds,
                "n_estimators": n_estimators,
                "random_state": random_state,
                "features_path": str(features_path),
            }
        )
        train_cxt(
            features_path=features_path,
            n_folds=n_folds,
            n_estimators=n_estimators,
            promote=promote,
            random_state=random_state,
        )
        _maybe_log_summary(prof.reports_root / "cxt_training_summary.json")
        return {
            "run_id": run.info.run_id,
            "summary": str(prof.reports_root / "cxt_training_summary.json"),
        }
