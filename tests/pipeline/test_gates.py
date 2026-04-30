"""Tests for pipelines.gates — quality gates that hard-stop the pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipelines.mlflow_helpers as mh
import src.runtime.profile as profile_mod
from pipelines.gates import (
    PipelineGateError,
    data_quality_gate_task,
    feature_stability_gate_task,
)


@pytest.fixture
def isolated_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide a stand-in profile + isolated MLflow tracking store."""
    from types import SimpleNamespace

    fake_reports = tmp_path / "reports"
    fake_reports.mkdir()
    fake_cfg = SimpleNamespace(
        name="cpu",
        reports_root=fake_reports,
        mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        mlflow_experiment="cfm-test",
        mlflow_artifact_root=None,
        accelerator_type="cpu",
        gbm_device="cpu",
        to_dict=lambda: {"name": "cpu"},
    )
    # Reset MLflow helper state so configure_mlflow re-reads.
    mh._configured = False  # type: ignore[attr-defined]
    profile_mod._active = None  # type: ignore[attr-defined]
    monkeypatch.setattr("pipelines.gates.require_profile", lambda: fake_cfg)
    monkeypatch.setattr("pipelines.mlflow_helpers.require_profile", lambda: fake_cfg)
    return fake_cfg


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_data_quality_gate_passes(isolated_profile, monkeypatch):
    _write_json(
        isolated_profile.reports_root / "data_quality_summary.json",
        {"overall_missing_rate": 0.05, "n_dtype_mismatches": 0},
    )
    # Run the underlying function (not via Prefect engine) for unit-testing.
    data_quality_gate_task.fn(parent_run_id=None)


def test_data_quality_gate_fails_on_high_missing(isolated_profile):
    _write_json(
        isolated_profile.reports_root / "data_quality_summary.json",
        {"overall_missing_rate": 0.99, "n_dtype_mismatches": 0},
    )
    with pytest.raises(PipelineGateError, match="overall_missing_rate"):
        data_quality_gate_task.fn(parent_run_id=None)


def test_data_quality_gate_fails_on_dtype(isolated_profile):
    _write_json(
        isolated_profile.reports_root / "data_quality_summary.json",
        {"overall_missing_rate": 0.0, "n_dtype_mismatches": 99},
    )
    with pytest.raises(PipelineGateError, match="n_dtype_mismatches"):
        data_quality_gate_task.fn(parent_run_id=None)


def test_data_quality_gate_missing_report_raises(isolated_profile):
    with pytest.raises(PipelineGateError, match="Gate input missing"):
        data_quality_gate_task.fn(parent_run_id=None)


def test_feature_stability_gate_passes_max_psi_field(isolated_profile):
    _write_json(
        isolated_profile.reports_root / "feature_stability_summary.json",
        {"max_psi": 0.10},
    )
    feature_stability_gate_task.fn(parent_run_id=None)


def test_feature_stability_gate_passes_features_list(isolated_profile):
    _write_json(
        isolated_profile.reports_root / "feature_stability_summary.json",
        {"features": [{"feature": "a", "psi": 0.05}, {"feature": "b", "psi": 0.20}]},
    )
    feature_stability_gate_task.fn(parent_run_id=None)


def test_feature_stability_gate_fails(isolated_profile):
    _write_json(
        isolated_profile.reports_root / "feature_stability_summary.json",
        {"max_psi": 0.99},
    )
    with pytest.raises(PipelineGateError, match="max_psi"):
        feature_stability_gate_task.fn(parent_run_id=None)
