"""Prefect tasks for the pre-modelling analysis stage.

Each `analysis/NN_*.py` script already exposes `def main() -> None` and writes
its summary JSON under ``<reports_root>/<name>.json``. We load the script
dynamically (filenames start with digits, so they aren't normal Python
modules), call ``main()``, and log the resulting summary as an MLflow
artifact (with scalar fields lifted to MLflow metrics where possible).

The generic worker keeps this file short — adding a new analysis step is
a one-line entry in ``ANALYSIS_REGISTRY``.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from prefect import task
from prefect.tasks import task_input_hash

from pipelines.mlflow_helpers import log_path, stage_run
from src.runtime import require_profile

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# Make the project root importable so analysis scripts that rely on
# top-level packages (src.*, etc.) work when loaded dynamically.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class AnalysisStep:
    """One pre-analysis stage."""

    stage_name: str        # Prefect task / MLflow stage tag
    script: str            # filename inside analysis/
    report: str | None     # JSON filename in <reports_root>/ to log; None = skip


ANALYSIS_REGISTRY: tuple[AnalysisStep, ...] = (
    AnalysisStep("data_quality",      "01_data_quality.py",      "data_quality_summary.json"),
    AnalysisStep("feature_stability", "02_feature_stability.py", "feature_stability_summary.json"),
    AnalysisStep("univariate",        "03_univariate.py",        "univariate_stats.json"),
    AnalysisStep("bivariate_cxg",     "04_bivariate_cxg.py",     None),
    AnalysisStep("bivariate_cxa",     "05_bivariate_cxa.py",     "bivariate_cxa_sequence_type_summary.json"),
    AnalysisStep("bivariate_cxt",     "06_bivariate_cxt.py",     None),
    AnalysisStep("correlations",      "07_correlations.py",      "correlation_summary.json"),
    AnalysisStep("eda_shots",         "08_eda_shots.py",         None),
    AnalysisStep("eda_sequences",     "09_eda_sequences.py",     None),
    AnalysisStep("eda_opponents",     "10_eda_opponents.py",     None),
    AnalysisStep("hypothesis_cxg",    "11_hypothesis_cxg.py",    "hypothesis_cxg.json"),
    AnalysisStep("hypothesis_cxa",    "12_hypothesis_cxa.py",    "hypothesis_cxa.json"),
    AnalysisStep("hypothesis_cxt",    "13_hypothesis_cxt.py",    "hypothesis_cxt.json"),
    AnalysisStep("statsbomb_baseline", "14_statsbomb_baseline.py", "statsbomb_baseline_metrics.json"),
    AnalysisStep("zone_xt_priors",    "15_zone_xt_priors.py",    None),
    AnalysisStep("deep_eda",          "16_deep_eda.py",          "deep_eda_summary.json"),
)

ANALYSIS_BY_NAME: dict[str, AnalysisStep] = {s.stage_name: s for s in ANALYSIS_REGISTRY}


def _load_script_main(script_path: Path):
    """Dynamic-import the script (filename starts with a digit) and return ``main``."""
    spec = importlib.util.spec_from_file_location(f"_analysis_{script_path.stem}", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load analysis script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError(f"{script_path.name} has no `main()` function")
    return module.main


def _flatten_scalars(payload: Any, prefix: str = "", out: dict[str, float] | None = None) -> dict[str, float]:
    """Lift scalar (int/float/bool) leaf values from a nested dict to a flat metric dict."""
    if out is None:
        out = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_scalars(v, key, out)
    elif isinstance(payload, (int, float, bool)) and not isinstance(payload, bool):
        # MLflow keys: only [a-zA-Z0-9_\-./ ] allowed; sanitise lightly.
        safe = "".join(c if c.isalnum() or c in "_-./ " else "_" for c in prefix)[:240]
        if safe:
            out[safe] = float(payload)
    return out


def _maybe_log_report(stage: AnalysisStep) -> None:
    if stage.report is None:
        return
    prof = require_profile()
    report_path = prof.reports_root / stage.report
    if not report_path.exists():
        logger.warning("Report not found for stage %s: %s", stage.stage_name, report_path)
        return
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = _flatten_scalars(payload)
        if metrics:
            # Cap at 200 metrics to avoid spamming MLflow.
            for k, v in list(metrics.items())[:200]:
                try:
                    mlflow.log_metric(k, v)
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse report %s as scalars: %s", report_path, exc)
    log_path(report_path, artifact_path="reports")


def _run_step(stage: AnalysisStep, parent_run_id: str | None) -> str:
    script_path = ANALYSIS_DIR / stage.script
    if not script_path.exists():
        raise FileNotFoundError(f"Analysis script missing: {script_path}")
    with stage_run(stage.stage_name, parent_run_id=parent_run_id) as run:
        mlflow.log_param("analysis_script", stage.script)
        main_fn = _load_script_main(script_path)
        logger.info("[analysis] running %s …", stage.script)
        main_fn()
        _maybe_log_report(stage)
        return run.info.run_id


def _make_task(stage: AnalysisStep):
    """Factory: build a Prefect task for one analysis step."""

    @task(name=stage.stage_name, tags=["analysis", "pre"], cache_key_fn=task_input_hash)
    def _t(parent_run_id: str | None = None) -> str:
        return _run_step(stage, parent_run_id)

    _t.__doc__ = f"Run analysis/{stage.script} as a Prefect task."
    return _t


# Build one task per registered analysis step.
ANALYSIS_TASKS: dict[str, Any] = {s.stage_name: _make_task(s) for s in ANALYSIS_REGISTRY}
ANALYSIS_STAGE_NAMES: list[str] = [s.stage_name for s in ANALYSIS_REGISTRY]
