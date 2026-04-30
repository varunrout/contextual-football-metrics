"""Smoke tests for the Prefect flow CLI/registry layer.

We don't run the actual training tasks here (that requires real data and
takes minutes); this just validates wiring, stage selection, and CLI parsing.
"""

from __future__ import annotations

import pytest

from pipelines.flow import ALL_STAGES, _parse_args, _select_stages


def test_all_stages_registered() -> None:
    assert ALL_STAGES == ["train_cxg", "train_cxa", "train_cxt"]


def test_select_stages_default_returns_all() -> None:
    assert _select_stages(None, None) == ALL_STAGES


def test_select_stages_only() -> None:
    assert _select_stages(["train_cxg"], None) == ["train_cxg"]
    # Order is preserved from ALL_STAGES, not the input list.
    assert _select_stages(["train_cxt", "train_cxg"], None) == ["train_cxg", "train_cxt"]


def test_select_stages_skip() -> None:
    assert _select_stages(None, ["train_cxa"]) == ["train_cxg", "train_cxt"]


def test_select_stages_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        _select_stages(["nope"], None)
    with pytest.raises(ValueError, match="Unknown stage"):
        _select_stages(None, ["also-nope"])


def test_cli_parses_minimal() -> None:
    args = _parse_args([])
    assert args.profile == "auto"
    assert args.only is None
    assert args.skip is None
    assert args.no_promote is False
    assert args.n_folds == 5


def test_cli_parses_full() -> None:
    args = _parse_args([
        "--profile", "gpu",
        "--only", "train_cxg",
        "--no-promote",
        "--n-folds", "3",
        "--include-360",
        "--seed", "7",
    ])
    assert args.profile == "gpu"
    assert args.only == ["train_cxg"]
    assert args.no_promote is True
    assert args.n_folds == 3
    assert args.include_360 is True
    assert args.seed == 7


def test_train_tasks_importable() -> None:
    """Sanity check: tasks import without side-effects."""
    from pipelines.stages.train import train_cxa_task, train_cxg_task, train_cxt_task

    assert train_cxg_task.name == "train_cxg"
    assert train_cxa_task.name == "train_cxa"
    assert train_cxt_task.name == "train_cxt"
