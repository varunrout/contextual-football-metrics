"""Smoke tests for the Prefect flow CLI/registry layer.

We don't run the actual training tasks here (that requires real data and
takes minutes); this just validates wiring, stage selection, and CLI parsing.
"""

from __future__ import annotations

import pytest

from pipelines.flow import (
    ALL_STAGES,
    DEFAULT_STAGES,
    GATE_STAGES,
    GROUPS,
    OPTIONAL_STAGES,
    POST_STAGES,
    PRE_STAGES,
    TRAIN_STAGES,
    _parse_args,
    _select_stages,
)


def test_groups_registered() -> None:
    assert set(GROUPS) == {"pre", "gate", "train", "post", "optional"}
    assert "data_quality" in PRE_STAGES
    assert "gate.data_quality" in GATE_STAGES
    assert set(TRAIN_STAGES) == {"train_cxg", "train_cxa", "train_cxt"}
    assert "interpretability" in POST_STAGES
    assert OPTIONAL_STAGES == ["score", "drift_monitor"]
    # Default stages do NOT include optional (they need real parquet inputs).
    assert "score" not in DEFAULT_STAGES
    assert "drift_monitor" not in DEFAULT_STAGES
    assert ALL_STAGES == DEFAULT_STAGES + OPTIONAL_STAGES


def test_select_stages_default_returns_default_set() -> None:
    assert _select_stages(None, None) == DEFAULT_STAGES


def test_select_stages_only() -> None:
    assert _select_stages(["train_cxg"], None) == ["train_cxg"]
    assert _select_stages(["train_cxt", "train_cxg"], None) == ["train_cxg", "train_cxt"]


def test_select_stages_skip() -> None:
    sel = _select_stages(None, ["train_cxa"])
    assert "train_cxa" not in sel
    assert "train_cxg" in sel and "train_cxt" in sel


def test_select_stages_only_group_train() -> None:
    sel = _select_stages(None, None, only_group=["train"])
    assert sel == ["train_cxg", "train_cxa", "train_cxt"]


def test_select_stages_skip_group_pre() -> None:
    sel = _select_stages(None, None, skip_group=["pre"])
    for s in PRE_STAGES:
        assert s not in sel
    assert "train_cxg" in sel


def test_select_stages_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        _select_stages(["nope"], None)
    with pytest.raises(ValueError, match="Unknown stage"):
        _select_stages(None, ["also-nope"])
    with pytest.raises(ValueError, match="Unknown group"):
        _select_stages(None, None, only_group=["nope"])


def test_cli_parses_minimal() -> None:
    args = _parse_args([])
    assert args.profile == "auto"
    assert args.only is None
    assert args.skip is None
    assert args.only_group is None
    assert args.no_promote is False
    assert args.n_folds == 5


def test_cli_parses_full() -> None:
    args = _parse_args(
        [
            "--profile",
            "gpu",
            "--only-group",
            "train",
            "--no-promote",
            "--n-folds",
            "3",
            "--include-360",
            "--seed",
            "7",
            "--gate-max-psi",
            "0.5",
        ]
    )
    assert args.profile == "gpu"
    assert args.only_group == ["train"]
    assert args.no_promote is True
    assert args.n_folds == 3
    assert args.include_360 is True
    assert args.seed == 7
    assert args.gate_max_psi == 0.5


def test_train_tasks_importable() -> None:
    from pipelines.stages.train import train_cxa_task, train_cxg_task, train_cxt_task

    assert train_cxg_task.name == "train_cxg"
    assert train_cxa_task.name == "train_cxa"
    assert train_cxt_task.name == "train_cxt"


def test_analysis_tasks_importable() -> None:
    from pipelines.stages.analysis import ANALYSIS_REGISTRY, ANALYSIS_TASKS

    assert len(ANALYSIS_REGISTRY) == 16
    assert "data_quality" in ANALYSIS_TASKS
    assert ANALYSIS_TASKS["data_quality"].name == "data_quality"


def test_gate_tasks_importable() -> None:
    from pipelines.gates import data_quality_gate_task, feature_stability_gate_task

    assert data_quality_gate_task.name == "gate.data_quality"
    assert feature_stability_gate_task.name == "gate.feature_stability"


def test_post_tasks_importable() -> None:
    from pipelines.stages.post import POST_REGISTRY, POST_TASKS, drift_monitor_task, score_task

    assert len(POST_REGISTRY) == 3
    assert {"scoring_validation", "interpretability", "model_comparison"} <= set(POST_TASKS)
    assert score_task.name == "score"
    assert drift_monitor_task.name == "drift_monitor"


def test_select_optional_stage_via_only() -> None:
    assert _select_stages(["score"], None) == ["score"]
    assert _select_stages(["drift_monitor"], None) == ["drift_monitor"]
