"""
Tests for the SQLite results store: scripts/build_results_db.py + src/results_store.

Builds the DB from the committed JSON reports into a temp path and checks the
tables populate and the read helpers work.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_results_db", _ROOT / "scripts" / "build_results_db.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def results_db(tmp_path_factory):
    if not (_ROOT / "reports" / "cxg_training_summary.json").exists():
        pytest.skip("reports/*.json not present")
    db = tmp_path_factory.mktemp("results") / "results.db"
    _load_builder().build(db)
    return db


def test_tables_populated(results_db):
    import sqlite3

    con = sqlite3.connect(results_db)
    try:
        for table in ("model_run", "model_metric", "incremental_lift", "calibration"):
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
            assert n > 0, f"{table} is empty"
    finally:
        con.close()


def test_exactly_one_promoted_per_metric(results_db):
    import sqlite3

    con = sqlite3.connect(results_db)
    try:
        rows = con.execute(
            "SELECT metric, SUM(is_promoted) FROM model_run GROUP BY metric"
        ).fetchall()
    finally:
        con.close()
    assert rows, "no model runs"
    for metric, promoted in rows:
        assert promoted == 1, f"{metric} should have exactly one promoted model, has {promoted}"


def test_leaderboard_helper(results_db):
    from src import results_store

    lb = results_store.leaderboard("cxg", db_path=results_db)
    assert not lb.empty
    assert "is_promoted" in lb.columns
    assert "log_loss" in lb.columns  # held-out log-loss pivoted in


def test_incremental_lift_helper(results_db):
    from src import results_store

    lift = results_store.incremental_lift("cxt", db_path=results_db)
    assert not lift.empty
    assert {"delta_mean", "ci_low", "ci_high", "excludes_zero"} <= set(lift.columns)
