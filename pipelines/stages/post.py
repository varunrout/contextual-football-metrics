"""Prefect tasks for the post-modelling analysis stage.

Wraps:

    analysis/17_scoring_validation.py   → ``scoring_validation``
    analysis/18_interpretability.py     → ``interpretability``
    analysis/19_model_comparison.py     → ``model_comparison``
    scripts/score.py::score             → ``score`` (optional, opt-in)
    scripts/monitor.py::monitor         → ``drift_monitor`` (optional, opt-in)

The first three follow the same dynamic-load pattern as ``stages.analysis``
(the source filenames begin with digits). The latter two call importable
functions in ``scripts/`` directly.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from prefect import task
from prefect.tasks import task_input_hash

from pipelines.mlflow_helpers import log_path, stage_run
from pipelines.stages.analysis import AnalysisStep, _maybe_log_report
from src.runtime import require_profile

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Reused dynamic-loader (analysis scripts with digit prefixes) ─────────────
def _load_script_main(script_path: Path):
    spec = importlib.util.spec_from_file_location(f"_post_{script_path.stem}", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError(f"{script_path.name} has no `main()` function")
    return module.main


@dataclass(frozen=True)
class PostStep:
    stage_name: str
    script: str
    report: str | None  # JSON or HTML to log as artifact (relative to reports_root)


POST_REGISTRY: tuple[PostStep, ...] = (
    PostStep("scoring_validation", "17_scoring_validation.py", None),
    PostStep("interpretability", "18_interpretability.py", "interpretability_report.html"),
    PostStep("model_comparison", "19_model_comparison.py", "model_comparison_cxg.json"),
)

POST_BY_NAME: dict[str, PostStep] = {s.stage_name: s for s in POST_REGISTRY}


def _maybe_log_post_report(stage: PostStep) -> None:
    if stage.report is None:
        return
    prof = require_profile()
    report_path = prof.reports_root / stage.report
    if not report_path.exists():
        logger.warning("Report not found for stage %s: %s", stage.stage_name, report_path)
        return
    if stage.report.endswith(".json"):
        # Reuse analysis._maybe_log_report (it lifts scalars to metrics).
        _maybe_log_report(AnalysisStep(stage.stage_name, stage.script, stage.report))
    else:
        log_path(report_path, artifact_path="reports")


def _run_post_step(stage: PostStep, parent_run_id: str | None) -> str:
    script_path = ANALYSIS_DIR / stage.script
    if not script_path.exists():
        raise FileNotFoundError(f"Post-analysis script missing: {script_path}")
    with stage_run(stage.stage_name, parent_run_id=parent_run_id) as run:
        mlflow.log_param("post_script", stage.script)
        main_fn = _load_script_main(script_path)
        logger.info("[post] running %s …", stage.script)
        main_fn()
        _maybe_log_post_report(stage)
        return run.info.run_id


def _make_post_task(stage: PostStep):
    @task(name=stage.stage_name, tags=["analysis", "post"], cache_key_fn=task_input_hash)
    def _t(parent_run_id: str | None = None) -> str:
        return _run_post_step(stage, parent_run_id)

    _t.__doc__ = f"Run analysis/{stage.script} as a Prefect post-task."
    return _t


POST_TASKS: dict[str, Any] = {s.stage_name: _make_post_task(s) for s in POST_REGISTRY}
POST_STAGE_NAMES: list[str] = [s.stage_name for s in POST_REGISTRY]


# ── Optional: scoring on a real events parquet ──────────────────────────────
@task(name="score", tags=["score", "post"], cache_key_fn=task_input_hash)
def score_task(
    parent_run_id: str | None = None,
    *,
    events_path: str | None = None,
    output_path: str | None = None,
) -> str:
    """Score an events parquet through the production pipeline."""
    from scripts.score import MODELS_YAML, score

    prof = require_profile()
    events = Path(events_path) if events_path else (prof.data_root / "processed" / "events.parquet")
    out = Path(output_path) if output_path else (prof.outputs_root / "scores" / "scored.parquet")

    with stage_run("score", parent_run_id=parent_run_id) as run:
        mlflow.log_param("events_path", str(events))
        mlflow.log_param("output_path", str(out))
        score(
            events_path=events,
            output_path=out,
            config_path=Path(MODELS_YAML),
        )
        if out.exists():
            log_path(out, artifact_path="scores")
        return run.info.run_id


# ── Optional: drift monitor between reference + current parquet ──────────────
@task(name="drift_monitor", tags=["monitor", "post"], cache_key_fn=task_input_hash)
def drift_monitor_task(
    parent_run_id: str | None = None,
    *,
    reference_path: str | None = None,
    current_path: str | None = None,
    psi_threshold: float | None = None,
    kl_threshold: float | None = None,
    fail_on_drift: bool = True,
) -> bool:
    """Run feature drift monitor and log the JSON report as an MLflow artifact."""
    from scripts.monitor import monitor

    prof = require_profile()
    ref = (
        Path(reference_path) if reference_path else (prof.data_root / "features" / "shots.parquet")
    )
    cur = (
        Path(current_path) if current_path else ref
    )  # pragma: no cover — caller normally overrides
    report_path = prof.outputs_root / "drift_report.json"

    with stage_run("drift_monitor", parent_run_id=parent_run_id):
        mlflow.log_param("reference_path", str(ref))
        mlflow.log_param("current_path", str(cur))
        has_drift = monitor(
            reference_path=ref,
            current_path=cur,
            psi_threshold=psi_threshold,
            kl_threshold=kl_threshold,
            report_path=report_path,
            fail_on_drift=False,  # never sys.exit inside Prefect
        )
        if report_path.exists():
            log_path(report_path, artifact_path="monitoring")
        mlflow.set_tag("drift_detected", str(has_drift))
        if fail_on_drift and has_drift:
            raise RuntimeError(f"Feature drift detected (see {report_path})")
        return has_drift
