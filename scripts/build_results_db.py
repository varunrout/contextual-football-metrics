"""
scripts/build_results_db.py
===========================
Build a small SQLite results store (`results.db`) from the committed JSON
reports (CONT-05).

The parquet layer stays the primary analytics store (see docs/data.md). This
script distils the derived, relational, dashboard-facing numbers - model
leaderboards, held-out metrics, incremental-lift deltas and calibration - into
one queryable database so the Streamlit app and any reviewer can read them
without scanning 754k events or re-running training.

It reads only files under `reports/` (all committed), so it is fast and
deterministic. `results.db` itself is gitignored; rebuild it any time with:

    python scripts/build_results_db.py

Tables
------
model_run(metric, model_name, family, feature_set, is_promoted, holdout_comp, created_at)
model_metric(metric, model_name, metric_name, split, value)          split in {cv, train, holdout}
incremental_lift(metric, candidate, baseline, delta_name, delta_mean, ci_low, ci_high, excludes_zero, verdict)
calibration(metric, kind, value)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"
DEFAULT_DB = PROJECT_ROOT / "results.db"
HOLDOUT_COMP = "Euro 2024"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("build_results_db")

SCHEMA = """
DROP TABLE IF EXISTS model_run;
DROP TABLE IF EXISTS model_metric;
DROP TABLE IF EXISTS incremental_lift;
DROP TABLE IF EXISTS calibration;

CREATE TABLE model_run (
    metric       TEXT NOT NULL,
    model_name   TEXT NOT NULL,
    family       TEXT,
    feature_set  TEXT,
    is_promoted  INTEGER NOT NULL DEFAULT 0,
    holdout_comp TEXT,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (metric, model_name)
);

CREATE TABLE model_metric (
    metric      TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    split       TEXT NOT NULL,
    value       REAL NOT NULL,
    PRIMARY KEY (metric, model_name, metric_name, split)
);

CREATE TABLE incremental_lift (
    metric        TEXT NOT NULL,
    candidate     TEXT NOT NULL,
    baseline      TEXT NOT NULL,
    delta_name    TEXT NOT NULL,
    delta_mean    REAL NOT NULL,
    ci_low        REAL NOT NULL,
    ci_high       REAL NOT NULL,
    excludes_zero INTEGER NOT NULL,
    verdict       TEXT,
    PRIMARY KEY (metric, candidate, baseline, delta_name)
);

CREATE TABLE calibration (
    metric TEXT NOT NULL,
    kind   TEXT NOT NULL,
    value  REAL NOT NULL,
    PRIMARY KEY (metric, kind)
);
"""


def _load(name: str) -> dict | None:
    path = REPORTS / name
    if not path.exists():
        logger.warning("report not found, skipping: %s", name)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _production_models() -> dict[str, str]:
    """metric -> production model stem, from configs/models.yaml."""
    cfg = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    prod = (cfg or {}).get("production", {}) or {}
    return {metric: Path(str(path)).stem for metric, path in prod.items()}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _is_promoted(metric: str, name: str, promoted_stem: str) -> int:
    """Match a leaderboard row to the production pointer stem.

    The CxA production file is ``cxa_<name>.pkl`` while its ladder rows are named
    ``<name>`` (e.g. logistic_contextual), so allow the ``<metric>_`` prefix.
    """
    return int(name == promoted_stem or f"{metric}_{name}" == promoted_stem)


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _ingest_cv_holdout_leaderboard(cur, metric: str, summary: dict, promoted: str) -> None:
    """cxg / cxt style: rows with cv_* metrics and a nested `heldout` block."""
    for row in summary.get("leaderboard", []):
        name = row.get("name")
        if not name:
            continue
        cur.execute(
            "INSERT OR REPLACE INTO model_run VALUES (?,?,?,?,?,?,?)",
            (
                metric,
                name,
                row.get("family"),
                row.get("feature_set"),
                _is_promoted(metric, name, promoted),
                HOLDOUT_COMP,
                _now(),
            ),
        )
        for key, value in row.items():
            if key.startswith("cv_") and (v := _num(value)) is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO model_metric VALUES (?,?,?,?,?)",
                    (metric, name, key[3:], "cv", v),
                )
        for key, value in (row.get("heldout") or {}).items():
            if (v := _num(value)) is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO model_metric VALUES (?,?,?,?,?)",
                    (metric, name, key.replace("heldout_", ""), "holdout", v),
                )


def _ingest_cxa_ladder(cur, summary: dict, promoted: str) -> None:
    """cxa style: rows with train_* (train split) and bare creation_*/quality_* (holdout)."""
    for row in summary.get("ladder", []):
        name = row.get("name")
        if not name:
            continue
        family = f"{row.get('creation_family')}+{row.get('quality_family')}"
        cur.execute(
            "INSERT OR REPLACE INTO model_run VALUES (?,?,?,?,?,?,?)",
            (
                "cxa",
                name,
                family,
                "contextual",
                _is_promoted("cxa", name, promoted),
                HOLDOUT_COMP,
                _now(),
            ),
        )
        for key, value in row.items():
            v = _num(value)
            if v is None or key in ("name",):
                continue
            if key.startswith("train_"):
                cur.execute(
                    "INSERT OR REPLACE INTO model_metric VALUES (?,?,?,?,?)",
                    ("cxa", name, key[6:], "train", v),
                )
            elif key.startswith(("creation_", "quality_")):
                cur.execute(
                    "INSERT OR REPLACE INTO model_metric VALUES (?,?,?,?,?)",
                    ("cxa", name, key, "holdout", v),
                )


def _ingest_incremental_lift(cur, metric: str, report: dict) -> None:
    if report is None:
        return
    candidate = report.get("candidate", "candidate")
    verdict = report.get("verdict")
    # cxg nests deltas per baseline; cxa/cxt have a single baseline.
    if "deltas_vs_baselines" in report:
        blocks = report["deltas_vs_baselines"]
    else:
        blocks = {report.get("baseline", "baseline"): report.get("delta_vs_baseline", {})}
    for baseline, deltas in blocks.items():
        for delta_name, d in deltas.items():
            lo, hi = _num(d.get("ci_low")), _num(d.get("ci_high"))
            if lo is None or hi is None:
                continue
            excludes_zero = int((lo > 0 and hi > 0) or (lo < 0 and hi < 0))
            cur.execute(
                "INSERT OR REPLACE INTO incremental_lift VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    metric,
                    candidate,
                    baseline,
                    delta_name,
                    _num(d.get("delta_mean")),
                    lo,
                    hi,
                    excludes_zero,
                    verdict,
                ),
            )


def _ingest_calibration(cur) -> None:
    report = _load("cxa_composite_calibration.json")
    if report is None:
        return
    ece = _num(report.get("expected_calibration_error"))
    sp = _num(report.get("spearman_pred_vs_realised"))
    if ece is not None:
        cur.execute(
            "INSERT OR REPLACE INTO calibration VALUES (?,?,?)", ("cxa", "composite_ece", ece)
        )
    if sp is not None:
        cur.execute(
            "INSERT OR REPLACE INTO calibration VALUES (?,?,?)", ("cxa", "composite_spearman", sp)
        )


def build(db_path: Path) -> None:
    promoted = _production_models()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        cur = conn.cursor()

        for metric, fname in (
            ("cxg", "cxg_training_summary.json"),
            ("cxt", "cxt_training_summary.json"),
        ):
            summary = _load(fname)
            if summary:
                _ingest_cv_holdout_leaderboard(cur, metric, summary, promoted.get(metric, ""))

        cxa_summary = _load("cxa_training_summary.json")
        if cxa_summary:
            _ingest_cxa_ladder(cur, cxa_summary, promoted.get("cxa", ""))

        for metric in ("cxg", "cxa", "cxt"):
            _ingest_incremental_lift(cur, metric, _load(f"incremental_lift_{metric}.json"))

        _ingest_calibration(cur)
        conn.commit()

        n_runs = cur.execute("SELECT COUNT(*) FROM model_run").fetchone()[0]
        n_metrics = cur.execute("SELECT COUNT(*) FROM model_metric").fetchone()[0]
        n_lift = cur.execute("SELECT COUNT(*) FROM incremental_lift").fetchone()[0]
        logger.info(
            "Built %s: %d model runs, %d metrics, %d lift rows.",
            db_path.name,
            n_runs,
            n_metrics,
            n_lift,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build results.db from reports/*.json.")
    ap.add_argument(
        "--db", default=str(DEFAULT_DB), help="Output SQLite path (default: results.db)."
    )
    build(Path(ap.parse_args().db))
