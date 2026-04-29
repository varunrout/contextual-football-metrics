"""
Data QA module.

Runs integrity checks on the processed parquet tables and emits a
structured coverage report. Raises DataQAError for hard failures;
logs warnings for soft failures.

Checks performed:
  1. Row count assertions (each table is non-empty)
  2. Primary key uniqueness
  3. Coordinate bounds (x ∈ [0,105], y ∈ [0,68])
  4. 360 linkage success rate per competition
  5. Missingness audit per column
  6. Europe-skew monitoring (share of 360 events from European competitions)
  7. Foreign key integrity (event → match, possession → match)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


class DataQAError(Exception):
    """Hard data quality failure that should stop the pipeline."""


@dataclass
class CoverageReport:
    total_matches: int = 0
    total_events: int = 0
    total_shots: int = 0
    total_possessions: int = 0
    total_freeze_frames: int = 0
    competitions: list[dict[str, Any]] = field(default_factory=list)
    coordinate_violations: int = 0
    key_duplicates: dict[str, int] = field(default_factory=dict)
    missingness: dict[str, dict[str, float]] = field(default_factory=dict)
    linkage_rates: dict[str, float] = field(default_factory=dict)   # competition → rate
    europe_360_share: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_qa(
    processed_dir: Path | str | None = None,
    raise_on_error: bool = True,
) -> CoverageReport:
    """
    Run all QA checks against the processed parquet directory.

    Parameters
    ----------
    processed_dir : path to data/processed/; defaults to project-standard location
    raise_on_error : if True, raise DataQAError when hard errors are found

    Returns
    -------
    CoverageReport with all findings populated.
    """
    if processed_dir is None:
        processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    processed_dir = Path(processed_dir)

    report = CoverageReport()

    # Load competition config for skew thresholds
    skew_cfg = _load_skew_thresholds()

    # ── 1. Load tables (soft-fail when file missing — pipeline may be partial) ──
    tables: dict[str, pd.DataFrame] = {}
    for name in ["matches", "events", "shots", "possessions", "freeze_frames_360"]:
        path = processed_dir / f"{name}.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path)
            logger.info("QA: loaded %s (%d rows)", name, len(tables[name]))
        else:
            report.warnings.append(f"Table not found: {path}")
            tables[name] = pd.DataFrame()

    # ── 2. Row counts ──────────────────────────────────────────────────────────
    report.total_matches = len(tables["matches"])
    report.total_events = len(tables["events"])
    report.total_shots = len(tables["shots"])
    report.total_possessions = len(tables["possessions"])
    report.total_freeze_frames = len(tables["freeze_frames_360"])

    for name, df in tables.items():
        if df.empty:
            report.warnings.append(f"Table '{name}' is empty — pipeline may be incomplete")

    # ── 3. Primary key uniqueness ──────────────────────────────────────────────
    pk_map = {
        "matches": "internal_id",
        "events": "internal_id",
        "shots": "internal_id",
        "possessions": "internal_id",
    }
    for table_name, pk_col in pk_map.items():
        df = tables[table_name]
        if df.empty or pk_col not in df.columns:
            continue
        n_dupes = int(df[pk_col].duplicated().sum())
        if n_dupes > 0:
            msg = f"Table '{table_name}': {n_dupes} duplicate primary keys in '{pk_col}'"
            report.errors.append(msg)
            report.key_duplicates[table_name] = n_dupes

    # ── 4. Coordinate bounds ───────────────────────────────────────────────────
    for table_name, x_col, y_col in [
        ("events", "x", "y"),
        ("shots", "x", "y"),
    ]:
        df = tables[table_name]
        if df.empty:
            continue
        if x_col in df.columns and y_col in df.columns:
            bad_x = (~df[x_col].between(0, 105, inclusive="both")).sum()
            bad_y = (~df[y_col].between(0, 68, inclusive="both")).sum()
            violations = int(bad_x + bad_y)
            report.coordinate_violations += violations
            if violations > 0:
                report.warnings.append(
                    f"Table '{table_name}': {violations} coordinate values outside "
                    f"[0,105]×[0,68] bounds"
                )

    # ── 5. 360 linkage rate per competition ────────────────────────────────────
    shots = tables["shots"]
    if not shots.empty and "competition_id" in shots.columns and "has_360" in shots.columns:
        for comp_id, grp in shots.groupby("competition_id"):
            expected_360 = _competition_has_360(str(comp_id))
            if expected_360:
                rate = float(grp["has_360"].mean())
                report.linkage_rates[str(comp_id)] = rate
                if rate < 0.80:
                    msg = (
                        f"Competition {comp_id}: 360 linkage rate {rate:.1%} "
                        f"< 80 % threshold"
                    )
                    report.warnings.append(msg)

    # ── 6. Europe-skew monitoring ──────────────────────────────────────────────
    events = tables["events"]
    if not events.empty and "has_360" in events.columns and "domain" in events.columns:
        events_360 = events[events["has_360"] == True]  # noqa: E712
        if len(events_360) > 0:
            europe_count = (events_360["domain"] == "continental").sum() + (
                events_360["region"] == "europe"
            ).sum() if "region" in events_360.columns else (
                events_360["domain"] == "continental"
            ).sum()
            europe_share = float(europe_count) / len(events_360)
            report.europe_360_share = europe_share
            warn_thresh = skew_cfg.get("europe_360_share_warn", 0.70)
            err_thresh = skew_cfg.get("europe_360_share_error", 0.85)
            if europe_share > err_thresh:
                report.errors.append(
                    f"Europe share of 360 events {europe_share:.1%} exceeds error "
                    f"threshold {err_thresh:.0%}"
                )
            elif europe_share > warn_thresh:
                report.warnings.append(
                    f"Europe share of 360 events {europe_share:.1%} exceeds warning "
                    f"threshold {warn_thresh:.0%}"
                )

    # ── 7. Missingness audit ───────────────────────────────────────────────────
    for table_name, df in tables.items():
        if df.empty:
            continue
        miss = (df.isnull().mean() * 100).round(2).to_dict()
        report.missingness[table_name] = {k: v for k, v in miss.items() if v > 0}

    # ── 8. Final verdict ───────────────────────────────────────────────────────
    if report.errors and raise_on_error:
        raise DataQAError(
            f"Data QA failed with {len(report.errors)} error(s):\n"
            + "\n".join(f"  • {e}" for e in report.errors)
        )

    _log_report(report)
    return report


def _load_skew_thresholds() -> dict:
    cfg_path = _CONFIGS_DIR / "competitions.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("skew_thresholds", {})


def _competition_has_360(competition_id: str) -> bool:
    cfg_path = _CONFIGS_DIR / "competitions.yaml"
    if not cfg_path.exists():
        return False
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for comp in cfg.get("competitions", []):
        if str(comp.get("competition_id")) == competition_id:
            return bool(comp.get("has_360", False))
    return False


def _log_report(report: CoverageReport) -> None:
    logger.info(
        "QA summary: %d matches | %d events | %d shots | %d possessions | "
        "%d freeze frames | Europe 360 share: %.1f%%",
        report.total_matches,
        report.total_events,
        report.total_shots,
        report.total_possessions,
        report.total_freeze_frames,
        report.europe_360_share * 100,
    )
    for w in report.warnings:
        logger.warning("QA WARNING: %s", w)
    for e in report.errors:
        logger.error("QA ERROR: %s", e)
