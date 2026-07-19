"""Pipeline gates: hard-stop tasks that fail the run when quality thresholds are breached.

Gates read JSON summaries already produced by the pre-analysis stage and
raise a ``PipelineGateError`` when thresholds are exceeded. The flow puts
gates between the pre-analysis stage and training so that broken inputs
never reach the modelling step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
from prefect import task

from pipelines.mlflow_helpers import stage_run
from src.runtime import require_profile

logger = logging.getLogger(__name__)


class PipelineGateError(RuntimeError):
    """Raised when a quality gate fails."""


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PipelineGateError(f"Gate input missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@task(name="gate.data_quality", tags=["gate"])
def data_quality_gate_task(
    parent_run_id: str | None = None,
    *,
    max_overall_missing_rate: float = 0.40,
    max_dtype_mismatches: int = 5,
) -> None:
    """Fail the pipeline if data-quality thresholds are exceeded."""
    prof = require_profile()
    report = _load(prof.reports_root / "data_quality_summary.json")
    overall = float(report.get("overall_missing_rate", 0.0))
    n_dtype = int(report.get("n_dtype_mismatches", 0))
    failures: list[str] = []
    if overall > max_overall_missing_rate:
        failures.append(f"overall_missing_rate={overall:.3f} > {max_overall_missing_rate}")
    if n_dtype > max_dtype_mismatches:
        failures.append(f"n_dtype_mismatches={n_dtype} > {max_dtype_mismatches}")

    with stage_run("gate.data_quality", parent_run_id=parent_run_id):
        mlflow.log_metric("overall_missing_rate", overall)
        mlflow.log_metric("n_dtype_mismatches", float(n_dtype))
        mlflow.log_param("threshold_missing", max_overall_missing_rate)
        mlflow.log_param("threshold_dtype", max_dtype_mismatches)
        if failures:
            mlflow.set_tag("gate_status", "failed")
            raise PipelineGateError("Data-quality gate failed: " + "; ".join(failures))
        mlflow.set_tag("gate_status", "passed")
    logger.info("[gate] data quality OK (missing=%.3f, dtype=%d)", overall, n_dtype)


@task(name="gate.feature_stability", tags=["gate"])
def feature_stability_gate_task(
    parent_run_id: str | None = None,
    *,
    max_psi: float = 0.25,
) -> None:
    """Fail if any feature's PSI exceeds ``max_psi``."""
    prof = require_profile()
    report = _load(prof.reports_root / "feature_stability_summary.json")
    # The exact schema may evolve; tolerate either {"max_psi": float} or
    # {"features": [{"feature": ..., "psi": ...}, ...]} representations.
    max_observed = 0.0
    if isinstance(report, dict):
        if "max_psi" in report and isinstance(report["max_psi"], (int, float)):
            max_observed = float(report["max_psi"])
        elif "features" in report and isinstance(report["features"], list):
            for entry in report["features"]:
                psi = entry.get("psi") if isinstance(entry, dict) else None
                if isinstance(psi, (int, float)):
                    max_observed = max(max_observed, float(psi))

    with stage_run("gate.feature_stability", parent_run_id=parent_run_id):
        mlflow.log_metric("max_psi", max_observed)
        mlflow.log_param("threshold_psi", max_psi)
        if max_observed > max_psi:
            mlflow.set_tag("gate_status", "failed")
            raise PipelineGateError(
                f"Feature stability gate failed: max_psi={max_observed:.3f} > {max_psi}"
            )
        mlflow.set_tag("gate_status", "passed")
    logger.info("[gate] feature stability OK (max_psi=%.3f)", max_observed)


GATE_TASKS = {
    "gate.data_quality": data_quality_gate_task,
    "gate.feature_stability": feature_stability_gate_task,
}
