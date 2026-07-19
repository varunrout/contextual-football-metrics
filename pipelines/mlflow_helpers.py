"""MLflow integration helpers for the pipeline.

Provides a single entry point — :func:`stage_run` — that opens a (possibly
nested) MLflow run, applies standard tags (profile, git SHA, stage name),
and yields the run object so the caller can ``log_metrics`` / ``log_artifact``
freely.

Typical usage from a Prefect task::

    from pipelines.mlflow_helpers import pipeline_run, stage_run

    with pipeline_run("full") as parent:
        with stage_run("train_cxg", parent_run_id=parent.info.run_id) as run:
            mlflow.log_metric("log_loss", 0.31)

The helper reads the tracking URI / experiment from the active runtime
profile (``src.runtime.get_profile()``), so a single ``--profile`` flag at
the CLI controls where runs land.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from collections.abc import Iterator
from typing import Any

import mlflow

from src.runtime import require_profile

logger = logging.getLogger(__name__)


# ── git helpers ──────────────────────────────────────────────────────────────


def _git_sha(short: bool = True) -> str | None:
    """Return the current git commit SHA, or None if not in a repo."""
    cmd = ["git", "rev-parse"] + (["--short", "HEAD"] if short else ["HEAD"])
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_branch() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ── MLflow setup ─────────────────────────────────────────────────────────────


_configured = False


def configure_mlflow(force: bool = False) -> None:
    """Apply tracking URI + experiment from the active profile.

    Idempotent — second call is a no-op unless ``force=True``.
    """
    global _configured
    if _configured and not force:
        return
    prof = require_profile()
    mlflow.set_tracking_uri(prof.mlflow_tracking_uri)
    mlflow.set_experiment(prof.mlflow_experiment)
    _configured = True
    logger.info(
        "MLflow configured: tracking_uri=%s experiment=%s",
        prof.mlflow_tracking_uri,
        prof.mlflow_experiment,
    )


def _standard_tags(stage: str) -> dict[str, str]:
    """Tags attached to every run for filterability in the UI."""
    prof = require_profile()
    tags: dict[str, str] = {
        "stage": stage,
        "profile": prof.name,
        "accelerator": prof.accelerator_type,
        "gbm_device": prof.gbm_device,
    }
    sha = _git_sha()
    if sha:
        tags["git_sha"] = sha
    branch = _git_branch()
    if branch:
        tags["git_branch"] = branch
    dirty = _git_dirty()
    if dirty is not None:
        tags["git_dirty"] = "true" if dirty else "false"
    user = os.environ.get("USER") or os.environ.get("USERNAME")
    if user:
        tags["user"] = user
    return tags


# ── Context managers ─────────────────────────────────────────────────────────


@contextlib.contextmanager
def pipeline_run(name: str, **extra_tags: Any) -> Iterator[mlflow.ActiveRun]:
    """Open the *parent* run for an end-to-end pipeline invocation.

    All :func:`stage_run` calls made inside this context become nested runs
    of the parent.

    Parameters
    ----------
    name
        Logical pipeline name (e.g. ``"full"``, ``"train-only"``).
    extra_tags
        Free-form tags merged on top of the standard ones.
    """
    configure_mlflow()
    tags = {
        **_standard_tags(stage="pipeline"),
        "pipeline_name": name,
        **{k: str(v) for k, v in extra_tags.items()},
    }
    with mlflow.start_run(run_name=f"pipeline:{name}", tags=tags) as run:
        prof = require_profile()
        mlflow.log_dict(prof.to_dict(), "profile.json")
        yield run


@contextlib.contextmanager
def stage_run(
    stage: str,
    *,
    parent_run_id: str | None = None,
    run_name: str | None = None,
    **extra_tags: Any,
) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run for one pipeline stage.

    If ``parent_run_id`` is provided the run is nested underneath that parent;
    otherwise a top-level run is opened (useful when a stage is run standalone
    outside the pipeline).
    """
    configure_mlflow()
    tags = {**_standard_tags(stage=stage), **{k: str(v) for k, v in extra_tags.items()}}
    if parent_run_id:
        tags["mlflow.parentRunId"] = parent_run_id
    nested = parent_run_id is not None
    with mlflow.start_run(
        run_name=run_name or stage,
        nested=nested,
        tags=tags,
    ) as run:
        yield run


# ── Convenience helpers ──────────────────────────────────────────────────────


def log_dict_artifact(payload: dict[str, Any], filename: str) -> None:
    """Log a dict as a JSON artifact attached to the active run."""
    mlflow.log_dict(payload, filename)


def log_path(path: str | os.PathLike, artifact_path: str | None = None) -> None:
    """Log a file or directory as artifact(s) on the active run."""
    p = os.fspath(path)
    if os.path.isdir(p):
        mlflow.log_artifacts(p, artifact_path=artifact_path)
    else:
        mlflow.log_artifact(p, artifact_path=artifact_path)
