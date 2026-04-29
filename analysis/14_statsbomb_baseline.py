"""
analysis/14_statsbomb_baseline.py
===================================
Part 8 — StatsBomb xG as CxG Baseline Evaluation.

Uses sklearn metrics directly (log_loss, brier_score_loss, roc_auc_score,
average_precision_score). ECE computed manually. Evaluates on 5-fold
match-grouped CV splits via match_kfold().

Outputs
-------
reports/statsbomb_baseline_metrics.json
reports/figures/baselines/statsbomb_xg_calibration.png
reports/figures/baselines/statsbomb_xg_roc.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import load_events, load_shots, save_fig, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("14_statsbomb_baseline")

_XG_COL = "shot_statsbomb_xg"
_TARGET = "goal"


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        frac = mask.sum() / n
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += frac * abs(acc - conf)
    return float(ece)


def _build_dataset(shots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Join xG from events to shots and drop rows without xG."""
    if _XG_COL in shots.columns:
        df = shots.copy()
    elif not events.empty:
        id_col = next((c for c in ["event_id", "internal_id"] if c in shots.columns and c in events.columns), None)
        if id_col and _XG_COL in events.columns:
            xg_lookup = events[[id_col, _XG_COL]].drop_duplicates(id_col)
            df = shots.merge(xg_lookup, on=id_col, how="left")
        else:
            raise ValueError("Cannot join shot_statsbomb_xg from events.parquet.")
    else:
        raise ValueError("shot_statsbomb_xg not available.")

    df[_XG_COL] = pd.to_numeric(df[_XG_COL], errors="coerce")
    df[_TARGET] = pd.to_numeric(df[_TARGET], errors="coerce")
    df = df.dropna(subset=[_XG_COL, _TARGET])
    logger.info("Baseline dataset: %d shots with xG", len(df))
    return df


def evaluate_overall(df: pd.DataFrame) -> dict:
    """Compute all metrics on the full dataset."""
    y_true = df[_TARGET].values.astype(int)
    y_prob = df[_XG_COL].values.clip(1e-7, 1 - 1e-7)

    return {
        "log_loss": round(log_loss(y_true, y_prob), 5),
        "brier_score": round(brier_score_loss(y_true, y_prob), 5),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 5),
        "average_precision": round(average_precision_score(y_true, y_prob), 5),
        "ece": round(_ece(y_true, y_prob), 5),
        "n_shots": int(len(y_true)),
        "n_goals": int(y_true.sum()),
        "goal_rate": round(float(y_true.mean()), 5),
        "mean_xg": round(float(y_prob.mean()), 5),
    }


def evaluate_cv(df: pd.DataFrame) -> list[dict]:
    """Cross-validated metrics using match_kfold."""
    try:
        from src.evaluation.validation_splits import match_kfold
    except ImportError:
        logger.warning("match_kfold not importable — skipping CV evaluation.")
        return []

    if "match_id" not in df.columns:
        logger.warning("match_id not in dataset — skipping CV evaluation.")
        return []

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        match_kfold(df, n_splits=5, match_id_col="match_id", random_state=42)
    ):
        val = df.iloc[val_idx]
        y_true = val[_TARGET].values.astype(int)
        y_prob = val[_XG_COL].values.clip(1e-7, 1 - 1e-7)

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            logger.warning("Fold %d: no class variation — skipping.", fold_idx)
            continue

        fold_results.append({
            "fold": fold_idx,
            "n_val": len(y_true),
            "log_loss": round(log_loss(y_true, y_prob), 5),
            "brier_score": round(brier_score_loss(y_true, y_prob), 5),
            "roc_auc": round(roc_auc_score(y_true, y_prob), 5),
            "average_precision": round(average_precision_score(y_true, y_prob), 5),
            "ece": round(_ece(y_true, y_prob), 5),
        })

    return fold_results


def plot_calibration(df: pd.DataFrame) -> None:
    y_true = df[_TARGET].values.astype(int)
    y_prob = df[_XG_COL].values.clip(1e-7, 1 - 1e-7)

    n_bins = 20
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means_xg, bin_rates, ci_lo, ci_hi = [], [], [], []

    z = 1.96
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() < 5:
            continue
        n_bin = mask.sum()
        mean_xg = float(y_prob[mask].mean())
        rate = float(y_true[mask].mean())
        # Wilson CI
        lo = (rate + z**2 / (2 * n_bin) - z * np.sqrt(rate * (1 - rate) / n_bin + z**2 / (4 * n_bin**2))) / (1 + z**2 / n_bin)
        hi = (rate + z**2 / (2 * n_bin) + z * np.sqrt(rate * (1 - rate) / n_bin + z**2 / (4 * n_bin**2))) / (1 + z**2 / n_bin)
        bin_means_xg.append(mean_xg)
        bin_rates.append(rate)
        ci_lo.append(lo)
        ci_hi.append(hi)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(bin_means_xg, bin_rates, "o-", color="#1f77b4", label="Actual goal rate")
    ax.fill_between(bin_means_xg, ci_lo, ci_hi, alpha=0.2, color="#1f77b4", label="95% CI")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlabel("Mean StatsBomb xG (bin)")
    ax.set_ylabel("Actual Goal Rate")
    ax.set_title("StatsBomb xG Calibration  (20 bins, 95% CI)")
    ax.legend()
    plt.tight_layout()
    save_fig("statsbomb_xg_calibration", "baselines")


def plot_roc(df: pd.DataFrame) -> None:
    y_true = df[_TARGET].values.astype(int)
    y_prob = df[_XG_COL].values.clip(1e-7, 1 - 1e-7)

    if y_true.sum() == 0:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("StatsBomb xG ROC Curve")
    ax.legend()
    plt.tight_layout()
    save_fig("statsbomb_xg_roc", "baselines")


def main() -> None:
    logger.info("Loading shots.parquet …")
    shots = load_shots()

    logger.info("Loading events.parquet for xG …")
    try:
        events = load_events()
    except FileNotFoundError:
        logger.warning("events.parquet not found.")
        events = pd.DataFrame()

    try:
        df = _build_dataset(shots, events)
    except ValueError as e:
        logger.error("Cannot build baseline dataset: %s", e)
        save_json({"error": str(e)}, "statsbomb_baseline_metrics")
        return

    logger.info("Overall metrics …")
    overall = evaluate_overall(df)
    logger.info("  log_loss=%.4f  brier=%.4f  AUC=%.4f  ECE=%.4f",
                overall["log_loss"], overall["brier_score"],
                overall["roc_auc"], overall["ece"])

    logger.info("Cross-validated metrics …")
    cv_folds = evaluate_cv(df)
    cv_summary: dict = {}
    if cv_folds:
        for metric in ["log_loss", "brier_score", "roc_auc", "average_precision", "ece"]:
            vals = [f[metric] for f in cv_folds]
            cv_summary[metric] = {
                "mean": round(float(np.mean(vals)), 5),
                "std": round(float(np.std(vals)), 5),
            }

    logger.info("Calibration plot …")
    plot_calibration(df)

    logger.info("ROC plot …")
    plot_roc(df)

    save_json({
        "overall": overall,
        "cv_5fold": cv_folds,
        "cv_summary": cv_summary,
    }, "statsbomb_baseline_metrics")

    logger.info("14_statsbomb_baseline.py complete.")


if __name__ == "__main__":
    main()
