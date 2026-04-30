"""Top-level Prefect flow + CLI for the contextual-football-metrics pipeline.

Stages currently wired (Phase 4 of the rollout):

    train_cxg → train_cxa → train_cxt

Future phases will add ingest / features / pre-modelling analysis / scoring /
post-modelling analysis. Stages are independent Prefect tasks; new ones can
be added without touching this file's CLI.

Examples
--------
Run the whole training pipeline on the active profile (auto-detected)::

    python -m pipelines.flow

Run with an explicit profile and only the CxG stage::

    python -m pipelines.flow --profile gpu --only train_cxg

Run on a fresh tracking URI (no caching)::

    python -m pipelines.flow --profile cpu --no-cache
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable

from prefect import flow

from pipelines.mlflow_helpers import pipeline_run
from pipelines.stages.train import train_cxa_task, train_cxg_task, train_cxt_task
from src.runtime import load_profile

logger = logging.getLogger(__name__)

# Stage registry: name → (task callable, kwargs key in CLI overrides)
TRAIN_STAGES: dict[str, callable] = {
    "train_cxg": train_cxg_task,
    "train_cxa": train_cxa_task,
    "train_cxt": train_cxt_task,
}

ALL_STAGES = list(TRAIN_STAGES.keys())


def _select_stages(only: list[str] | None, skip: list[str] | None) -> list[str]:
    if only:
        bad = set(only) - set(ALL_STAGES)
        if bad:
            raise ValueError(f"Unknown stage(s) in --only: {sorted(bad)}. Valid: {ALL_STAGES}")
        return [s for s in ALL_STAGES if s in only]
    selected = list(ALL_STAGES)
    if skip:
        bad = set(skip) - set(ALL_STAGES)
        if bad:
            raise ValueError(f"Unknown stage(s) in --skip: {sorted(bad)}. Valid: {ALL_STAGES}")
        selected = [s for s in selected if s not in skip]
    return selected


@flow(name="contextual-football-metrics")
def cfm_pipeline(
    *,
    profile: str = "auto",
    stages: list[str] | None = None,
    promote: bool = True,
    n_folds: int = 5,
    n_optuna_trials: int = 0,
    include_360: bool = False,
    n_estimators_cxg: int = 300,
    n_estimators_cxa: int = 300,
    n_estimators_cxt: int = 400,
    feature_set_cxa: str = "contextual",
    random_state: int = 42,
) -> dict[str, str]:
    """Run the configured pipeline stages under a single MLflow parent run."""
    cfg = load_profile(profile)
    stages = stages or ALL_STAGES
    logger.info("Running stages %s on profile=%s", stages, cfg.name)

    results: dict[str, str] = {}
    with pipeline_run("cfm", stages=",".join(stages)) as parent:
        parent_id = parent.info.run_id
        if "train_cxg" in stages:
            results["train_cxg"] = train_cxg_task(
                parent_run_id=parent_id,
                n_folds=n_folds,
                n_optuna_trials=n_optuna_trials,
                include_360=include_360,
                promote=promote,
                random_state=random_state,
                n_estimators=n_estimators_cxg,
            )
        if "train_cxa" in stages:
            results["train_cxa"] = train_cxa_task(
                parent_run_id=parent_id,
                feature_set=feature_set_cxa,
                n_folds=n_folds,
                n_estimators=n_estimators_cxa,
                promote=promote,
                random_state=random_state,
            )
        if "train_cxt" in stages:
            results["train_cxt"] = train_cxt_task(
                parent_run_id=parent_id,
                n_folds=n_folds,
                n_estimators=n_estimators_cxt,
                promote=promote,
                random_state=random_state,
            )
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the contextual-football-metrics pipeline.")
    p.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "cpu", "gpu", "cloud"],
        help="Runtime profile (default: auto).",
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=ALL_STAGES,
        help="Run only the listed stages.",
    )
    p.add_argument(
        "--skip",
        nargs="+",
        choices=ALL_STAGES,
        help="Skip the listed stages.",
    )
    p.add_argument("--no-promote", action="store_true", help="Skip writing production pointers.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-optuna-trials", type=int, default=0)
    p.add_argument("--include-360", action="store_true")
    p.add_argument("--n-estimators-cxg", type=int, default=300)
    p.add_argument("--n-estimators-cxa", type=int, default=300)
    p.add_argument("--n-estimators-cxt", type=int, default=400)
    p.add_argument("--feature-set-cxa", default="contextual")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Prefect task caching for this run.",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    stages = _select_stages(args.only, args.skip)

    if args.no_cache:
        # Disable Prefect caching globally for this process.
        import os as _os

        _os.environ["PREFECT_TASKS_REFRESH_CACHE"] = "true"

    cfm_pipeline(
        profile=args.profile,
        stages=stages,
        promote=not args.no_promote,
        n_folds=args.n_folds,
        n_optuna_trials=args.n_optuna_trials,
        include_360=args.include_360,
        n_estimators_cxg=args.n_estimators_cxg,
        n_estimators_cxa=args.n_estimators_cxa,
        n_estimators_cxt=args.n_estimators_cxt,
        feature_set_cxa=args.feature_set_cxa,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
