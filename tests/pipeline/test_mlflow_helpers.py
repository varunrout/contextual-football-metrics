"""Tests for pipelines.mlflow_helpers."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from pipelines import mlflow_helpers
from pipelines.mlflow_helpers import (
    _standard_tags,
    configure_mlflow,
    pipeline_run,
    stage_run,
)
from src.runtime import load_profile


@pytest.fixture(autouse=True)
def _isolated_mlflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Per-test MLflow store + profile reset."""
    # Force every test to use an isolated file store.
    tracking = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)

    # Reset module state so configure_mlflow runs fresh.
    monkeypatch.setattr(mlflow_helpers, "_configured", False)
    import src.runtime.profile as p

    monkeypatch.setattr(p, "_active", None)
    load_profile("cpu", validate=False)


def test_configure_mlflow_sets_tracking_uri(tmp_path: Path) -> None:
    configure_mlflow()
    assert mlflow.get_tracking_uri().startswith("file:")
    assert "mlruns" in mlflow.get_tracking_uri()


def test_standard_tags_include_profile_and_stage() -> None:
    tags = _standard_tags(stage="train_cxg")
    assert tags["stage"] == "train_cxg"
    assert tags["profile"] == "cpu"
    assert tags["accelerator"] == "cpu"
    assert tags["gbm_device"] == "cpu"


def test_pipeline_run_creates_run_with_tags() -> None:
    with pipeline_run("smoke") as run:
        assert run.info.run_name.startswith("pipeline:")
        # Profile snapshot is logged as artifact
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id)
        assert any(a.path == "profile.json" for a in artifacts)


def test_stage_run_nests_under_parent() -> None:
    with pipeline_run("smoke") as parent:
        parent_id = parent.info.run_id
        with stage_run("train_cxg", parent_run_id=parent_id) as child:
            child_id = child.info.run_id
            mlflow.log_metric("log_loss", 0.42)
    client = mlflow.MlflowClient()
    parent_run = client.get_run(parent_id)
    child_run = client.get_run(child_id)
    assert child_run.data.tags.get("mlflow.parentRunId") == parent_id
    assert child_run.data.tags.get("stage") == "train_cxg"
    assert child_run.data.metrics["log_loss"] == 0.42
    # Parent itself has stage="pipeline"
    assert parent_run.data.tags["stage"] == "pipeline"


def test_stage_run_standalone_without_parent() -> None:
    """Stages can also be run outside a pipeline (useful for ad-hoc dev)."""
    with stage_run("data_quality") as run:
        mlflow.log_metric("violations", 0)
    client = mlflow.MlflowClient()
    fetched = client.get_run(run.info.run_id)
    assert fetched.data.metrics["violations"] == 0
    assert "mlflow.parentRunId" not in fetched.data.tags


def test_extra_tags_are_merged() -> None:
    with pipeline_run("smoke", run_id_external="abc123") as run:
        client = mlflow.MlflowClient()
        fetched = client.get_run(run.info.run_id)
    assert fetched.data.tags.get("run_id_external") == "abc123"


def test_active_profile_drives_tracking_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Switching profile + reconfiguring should update tracking destination."""
    custom = f"file:{tmp_path / 'alt_store'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", custom)
    load_profile("cpu", validate=False)
    configure_mlflow(force=True)
    assert mlflow.get_tracking_uri() == custom
