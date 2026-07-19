"""
src/results_store.py
====================
Read helpers for the SQLite results store (`results.db`).

The store is built from the JSON reports by ``scripts/build_results_db.py``; see
docs/data.md. These helpers return tidy pandas DataFrames so the Streamlit app
(and notebooks / reviewers) can read leaderboards and lift results without
touching the parquet layer or re-running training.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "results.db"


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else _DEFAULT_DB
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: python scripts/build_results_db.py"
        )
    return sqlite3.connect(str(path))


def leaderboard(
    metric: str, split: str = "holdout", db_path: str | Path | None = None
) -> pd.DataFrame:
    """One row per model for ``metric``, with a column per metric_name at ``split``.

    Promoted (production) model flagged by ``is_promoted``.
    """
    con = _connect(db_path)
    try:
        runs = pd.read_sql_query(
            "SELECT model_name, family, feature_set, is_promoted FROM model_run WHERE metric = ?",
            con,
            params=(metric,),
        )
        long = pd.read_sql_query(
            "SELECT model_name, metric_name, value "
            "FROM model_metric WHERE metric = ? AND split = ?",
            con,
            params=(metric, split),
        )
    finally:
        con.close()

    if long.empty:
        return runs
    wide = long.pivot_table(index="model_name", columns="metric_name", values="value").reset_index()
    return runs.merge(wide, on="model_name", how="left")


def incremental_lift(metric: str | None = None, db_path: str | Path | None = None) -> pd.DataFrame:
    """Incremental-lift deltas (candidate vs baseline) with bootstrap CIs."""
    con = _connect(db_path)
    try:
        query = "SELECT * FROM incremental_lift"
        params: tuple = ()
        if metric:
            query += " WHERE metric = ?"
            params = (metric,)
        return pd.read_sql_query(query, con, params=params)
    finally:
        con.close()


def calibration(db_path: str | Path | None = None) -> pd.DataFrame:
    """Calibration summary rows (e.g. CxA composite ECE and Spearman)."""
    con = _connect(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM calibration", con)
    finally:
        con.close()
