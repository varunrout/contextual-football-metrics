"""Top-level Prefect flow + CLI for the contextual-football-metrics pipeline.

Stage groups (in execution order):

    pre   — pre-modelling analysis (data quality, stability, EDA, hypotheses, …)
    gate  — quality gates that hard-stop the run on threshold breaches
    train — model training (cxg, cxa, cxt)

Examples
--------
Run the full pipeline on the active profile (auto-detected)::

    python -m pipelines.flow

Skip the long pre-analysis stage and only train::

    python -m pipelines.flow --only-group train

Run with an explicit profile and only the CxG stage::

    python -m pipelines.flow --profile gpu --only train_cxg

Run on a fresh tracking URI (no Prefect cache)::

    python -m pipelines.flow --profile cpu --no-cache
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable

from prefect import flow

from pipelines.gates import data_quality_gate_task, feature_stability_gate_task
from pipelines.mlflow_helpers import pipeline_run
from pipelines.stages.analysis import ANALYSIS_STAGE_NAMES, ANALYSIS_TASKS
from pipelines.stages.post import POST_STAGE_NAMES, POST_TASKS, drift_monitor_task, score_task
from pipelines.stages.train import train_cxa_task, train_cxg_task, train_cxt_task
from src.runtime import load_profile

logger = logging.getLogger(__name__)

# ── Stage registry ───────────────────────────────────────────────────────────
PRE_STAGES: list[str] = list(ANALYSIS_STAGE_NAMES)
GATE_STAGES: list[str] = ["gate.data_quality", "gate.feature_stability"]
TRAIN_STAGES: dict[str, callable] = {
    "train_cxg": train_cxg_task,
    "train_cxa": train_cxa_task,
    "train_cxt": train_cxt_task,
}
POST_STAGES: list[str] = list(POST_STAGE_NAMES)
# score + drift_monitor are opt-in (need real parquet inputs); excluded from default run.
OPTIONAL_STAGES: list[str] = ["score", "drift_monitor"]

GATE_TASKS = {
    "gate.data_quality": data_quality_gate_task,
    "gate.feature_stability": feature_stability_gate_task,
}

GROUPS: dict[str, list[str]] = {
    "pre": PRE_STAGES,
    "gate": GATE_STAGES,
    "train": list(TRAIN_STAGES.keys()),
    "post": POST_STAGES,
    "optional": OPTIONAL_STAGES,
}

# Default flow: pre → gate → train → post (optional stages must be opted-in).
DEFAULT_STAGES: list[str] = PRE_STAGES + GATE_STAGES + list(TRAIN_STAGES.keys()) + POST_STAGES
ALL_STAGES: list[str] = DEFAULT_STAGES + OPTIONAL_STAGES


def _select_stages(
    only: list[str] | None,
    skip: list[str] | None,
    only_group: list[str] | None = None,
    skip_group: list[str] | None = None,
) -> list[str]:
    """Resolve the final ordered list of stages to run."""
    if only_group:
        bad = set(only_group) - set(GROUPS)
        if bad:
            raise ValueError(
                f"Unknown group(s) in --only-group: {sorted(bad)}. Valid: {list(GROUPS)}"
            )
        selected = [s for g in only_group for s in GROUPS[g]]
    elif only:
        bad = set(only) - set(ALL_STAGES)
        if bad:
            raise ValueError(f"Unknown stage(s) in --only: {sorted(bad)}. Valid: {ALL_STAGES}")
        selected = [s for s in ALL_STAGES if s in only]
    else:
        selected = list(DEFAULT_STAGES)

    if skip_group:
        bad = set(skip_group) - set(GROUPS)
        if bad:
            raise ValueError(
                f"Unknown group(s) in --skip-group: {sorted(bad)}. Valid: {list(GROUPS)}"
            )
        skip_set = {s for g in skip_group for s in GROUPS[g]}
        selected = [s for s in selected if s not in skip_set]
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
    gate_max_missing: float = 0.40,
    gate_max_dtype: int = 5,
    gate_max_psi: float = 0.25,
    score_events_path: str | None = None,
    score_output_path: str | None = None,
    drift_reference_path: str | None = None,
    drift_current_path: str | None = None,
) -> dict[str, str]:
    """Run the configured pipeline stages under a single MLflow parent run."""
    cfg = load_profile(profile)
    stages = stages or DEFAULT_STAGES
    logger.info("Running %d stages on profile=%s", len(stages), cfg.name)

    results: dict[str, str] = {}
    with pipeline_run("cfm", stages=",".join(stages)) as parent:
        parent_id = parent.info.run_id

        # ── pre-analysis ────────────────────────────────────────────────────
        for stage in PRE_STAGES:
            if stage in stages:
                results[stage] = ANALYSIS_TASKS[stage](parent_run_id=parent_id)

        # ── gates ───────────────────────────────────────────────────────────
        if "gate.data_quality" in stages:
            data_quality_gate_task(
                parent_run_id=parent_id,
                max_overall_missing_rate=gate_max_missing,
                max_dtype_mismatches=gate_max_dtype,
            )
            results["gate.data_quality"] = "passed"
        if "gate.feature_stability" in stages:
            feature_stability_gate_task(
                parent_run_id=parent_id,
                max_psi=gate_max_psi,
            )
            results["gate.feature_stability"] = "passed"

        # ── training ────────────────────────────────────────────────────────
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

        # ── post-modelling analysis ─────────────────────────────────────────
        for stage in POST_STAGES:
            if stage in stages:
                results[stage] = POST_TASKS[stage](parent_run_id=parent_id)

        # ── optional: scoring + drift monitor ───────────────────────────────
        if "score" in stages:
            results["score"] = score_task(
                parent_run_id=parent_id,
                events_path=score_events_path,
                output_path=score_output_path,
            )
        if "drift_monitor" in stages:
            drift_monitor_task(
                parent_run_id=parent_id,
                reference_path=drift_reference_path,
                current_path=drift_current_path,
                fail_on_drift=False,
            )
            results["drift_monitor"] = "completed"
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
    p.add_argument("--only", nargs="+", choices=ALL_STAGES, help="Run only the listed stages.")
    p.add_argument("--skip", nargs="+", choices=ALL_STAGES, help="Skip the listed stages.")
    p.add_argument(
        "--only-group",
        nargs="+",
        choices=list(GROUPS),
        help="Run only the listed stage groups (pre, gate, train).",
    )
    p.add_argument(
        "--skip-group", nargs="+", choices=list(GROUPS), help="Skip the listed stage groups."
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
    p.add_argument("--gate-max-missing", type=float, default=0.40)
    p.add_argument("--gate-max-dtype", type=int, default=5)
    p.add_argument("--gate-max-psi", type=float, default=0.25)
    p.add_argument("--score-events", default=None, help="Events parquet for scoring stage.")
    p.add_argument("--score-output", default=None, help="Output parquet for scoring stage.")
    p.add_argument("--drift-reference", default=None, help="Reference parquet for drift monitor.")
    p.add_argument("--drift-current", default=None, help="Current parquet for drift monitor.")
    p.add_argument(
        "--no-cache", action="store_true", help="Disable Prefect task caching for this run."
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    stages = _select_stages(args.only, args.skip, args.only_group, args.skip_group)

    if args.no_cache:
        import os as _os

        _os.environ["PREFECT_TASKS_REFRESH_CACHE"] = "true"

    # Resolve the profile up front (idempotent — cfm_pipeline() re-resolves the
    # same name internally) so its prefect.task_runner setting can be applied
    # to the flow via with_options() before the flow body runs.
    cfg = load_profile(args.profile)
    task_runner = cfg.build_task_runner()
    logger.info(
        "Using task runner %s (max_workers=%s) for profile '%s'",
        cfg.prefect_task_runner,
        cfg.prefect_max_workers,
        cfg.name,
    )

    cfm_pipeline.with_options(task_runner=task_runner)(
        profile=cfg.name,
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
        gate_max_missing=args.gate_max_missing,
        gate_max_dtype=args.gate_max_dtype,
        gate_max_psi=args.gate_max_psi,
        score_events_path=args.score_events,
        score_output_path=args.score_output,
        drift_reference_path=args.drift_reference,
        drift_current_path=args.drift_current,
    )


if __name__ == "__main__":
    main()
