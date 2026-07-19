"""
scripts/monitor.py
==================
Run feature/score drift detection between a reference dataset and a current
(live / new-competition) dataset using DriftDetector.

Exits with code 1 if significant drift is detected (for CI integration).

Usage
-----
    python scripts/monitor.py \
        --reference data/features/features.parquet \
        --current outputs/scores/scored.parquet

    # Customise thresholds (override configs/models.yaml values)
    python scripts/monitor.py \
        --reference data/features/features.parquet \
        --current outputs/scores/scored.parquet \
        --psi-threshold 0.15 \
        --kl-threshold 0.08

    # Write report to disk
    python scripts/monitor.py \
        --reference data/features/features.parquet \
        --current outputs/scores/scored.parquet \
        --report-path outputs/drift_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.drift_detector import DriftDetector  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("monitor")

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"

# Default numeric columns to monitor
_DEFAULT_NUMERIC_COLS = [
    "x_location",
    "y_location",
    "end_x",
    "end_y",
    "distance_to_goal",
    "angle_to_goal",
    "cxg",
    "cxa",
    "cxt",
]
# Default categorical columns to monitor
_DEFAULT_CATEGORICAL_COLS = [
    "action_type",
    "body_part",
    "play_pattern",
]


# ── Config helpers ────────────────────────────────────────────────────────────


def _read_monitoring_config() -> dict:
    if not MODELS_YAML.exists():
        return {}
    with open(MODELS_YAML, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg.get("monitoring", {})


# ── Column selection ──────────────────────────────────────────────────────────


def _select_cols(
    df: pd.DataFrame,
    candidates: list[str],
) -> list[str]:
    return [c for c in candidates if c in df.columns]


# ── Report serialisation ──────────────────────────────────────────────────────


def _report_to_dict(report) -> dict:
    """Convert a DriftReport to a JSON-serialisable dict."""
    from dataclasses import asdict, is_dataclass

    def _jsonify(value):
        if is_dataclass(value):
            return _jsonify(asdict(value))
        if isinstance(value, dict):
            return {k: _jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonify(v) for v in value]
        if hasattr(value, "__float__"):
            try:
                return float(value)
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            return {k: _jsonify(v) for k, v in value.__dict__.items()}
        return value

    return _jsonify(report)


# ── Main ──────────────────────────────────────────────────────────────────────


def monitor(
    reference_path: Path,
    current_path: Path,
    psi_threshold: float | None = None,
    kl_threshold: float | None = None,
    n_bins: int | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    report_path: Path | None = None,
    fail_on_drift: bool = True,
) -> bool:
    """Return True if significant drift was detected, False otherwise."""
    for path in (reference_path, current_path):
        if not path.exists():
            logger.error("File not found: %s", path)
            sys.exit(1)

    ref_df = pd.read_parquet(reference_path)
    cur_df = pd.read_parquet(current_path)
    logger.info(
        "Reference: %d rows × %d cols | Current: %d rows × %d cols",
        len(ref_df),
        len(ref_df.columns),
        len(cur_df),
        len(cur_df.columns),
    )

    # Resolve thresholds from config (then CLI, then default)
    mon_cfg = _read_monitoring_config()
    resolved_psi = (
        psi_threshold if psi_threshold is not None else float(mon_cfg.get("psi_threshold", 0.2))
    )
    resolved_kl = (
        kl_threshold if kl_threshold is not None else float(mon_cfg.get("kl_threshold", 0.1))
    )
    resolved_bins = n_bins if n_bins is not None else int(mon_cfg.get("n_bins", 10))
    resolved_numeric = (
        numeric_cols or mon_cfg.get("numeric_monitor_cols", None) or _DEFAULT_NUMERIC_COLS
    )
    resolved_categorical = (
        categorical_cols
        or mon_cfg.get("categorical_monitor_cols", None)
        or _DEFAULT_CATEGORICAL_COLS
    )

    # Restrict to columns present in both datasets
    both = set(ref_df.columns) & set(cur_df.columns)
    numeric_present = [c for c in resolved_numeric if c in both]
    categorical_present = [c for c in resolved_categorical if c in both]

    logger.info(
        "Monitoring %d numeric cols, %d categorical cols  "
        "(psi_threshold=%.2f, kl_threshold=%.2f, n_bins=%d)",
        len(numeric_present),
        len(categorical_present),
        resolved_psi,
        resolved_kl,
        resolved_bins,
    )

    if not numeric_present and not categorical_present:
        logger.warning(
            "No monitorable columns found in both datasets. "
            "Monitoring will be skipped. Check that features.parquet and "
            "scored.parquet share columns: %s / %s",
            resolved_numeric,
            resolved_categorical,
        )
        return False

    detector = DriftDetector(
        numeric_cols=numeric_present,
        categorical_cols=categorical_present,
        psi_threshold=resolved_psi,
        kl_threshold=resolved_kl,
        n_bins=resolved_bins,
    )
    detector.fit(ref_df)
    report = detector.detect(cur_df)

    # Log per-column results
    drifted: list[str] = []
    for col, psi_val in (getattr(report, "psi_scores", None) or {}).items():
        flag = psi_val > resolved_psi
        if flag:
            drifted.append(col)
        logger.info("PSI  %-35s %.4f  %s", col, psi_val, "DRIFT" if flag else "ok")
    for col, kl_val in (getattr(report, "kl_scores", None) or {}).items():
        flag = kl_val > resolved_kl
        if flag and col not in drifted:
            drifted.append(col)
        logger.info("KL   %-35s %.4f  %s", col, kl_val, "DRIFT" if flag else "ok")

    has_drift = getattr(report, "has_drift", bool(drifted))

    if has_drift:
        logger.warning("Significant drift detected in %d column(s): %s", len(drifted), drifted)
    else:
        logger.info("No significant drift detected.")

    # Persist report
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_dict = _report_to_dict(report)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2)
        logger.info("Drift report saved to %s", report_path)

    return has_drift


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect feature drift between reference and current datasets."
    )
    p.add_argument(
        "--reference",
        default=str(FEATURES_DIR / "features.parquet"),
        help="Reference (training) parquet (default: data/features/features.parquet).",
    )
    p.add_argument(
        "--current",
        required=True,
        help="Current (production) parquet to evaluate.",
    )
    p.add_argument(
        "--psi-threshold",
        type=float,
        default=None,
        help="PSI drift threshold (overrides configs/models.yaml).",
    )
    p.add_argument(
        "--kl-threshold",
        type=float,
        default=None,
        help="KL-divergence drift threshold (overrides configs/models.yaml).",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Number of bins for numeric PSI (overrides configs/models.yaml).",
    )
    p.add_argument(
        "--report-path",
        default=None,
        help="Path to write the JSON drift report (optional).",
    )
    p.add_argument(
        "--no-fail",
        action="store_true",
        help="Exit with code 0 even if drift is detected (disables CI gate).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    has_drift = monitor(
        reference_path=Path(args.reference),
        current_path=Path(args.current),
        psi_threshold=args.psi_threshold,
        kl_threshold=args.kl_threshold,
        n_bins=args.n_bins,
        report_path=Path(args.report_path) if args.report_path else None,
        fail_on_drift=not args.no_fail,
    )
    if has_drift and not args.no_fail:
        sys.exit(1)
