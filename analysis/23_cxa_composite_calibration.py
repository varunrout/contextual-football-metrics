"""
analysis/23_cxa_composite_calibration.py
========================================
CONT-F08 - reliability / calibration of the COMPOSITE CxA on held-out data.

WHY THIS EXISTS
---------------
CxA is sold as an expected shot-creation value: cxa = P(shot_created) x
E(CxG | shot). Training already reports the creation stage (ROC/PR) and the
quality stage (predicted vs actual CxG on shot-creating rows). What was never
checked is whether the COMPOSITE number is CALIBRATED: if you bucket actions by
predicted cxa, does the realised same-possession shot value match?

This script produces that missing reliability check on the identical Euro 2024
held-out actions:
  * predicted    = cxa from the production composite model (P(shot) x E[CxG]).
  * realised      = resulting_shot_cxg, the actual same-possession shot value
                    (0 when no shot followed, mean shot CxG when one did).
Predicted cxa is split into deciles; per decile we compare mean predicted vs
mean realised, report an expected-calibration-error (ECE), and state the usable
resolution of the metric (rank correlation of predicted vs realised).

Outputs
-------
reports/cxa_composite_calibration.json
reports/figures/incremental_lift/cxa_composite_reliability.png

Run
---
    python analysis/23_cxa_composite_calibration.py
    python analysis/23_cxa_composite_calibration.py --smoke   # no data / models needed
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("23_cxa_composite_calibration")

# Creative action types the composite model scores (matches CxAPipeline).
CREATIVE_ACTION_TYPES = {"pass", "cross", "carry", "cutback"}
COMPOSITE_MODEL = _ROOT / "models" / "cxa" / "cxa_logistic_contextual.pkl"
N_BINS = 10


def _score_shots(features_df: pd.DataFrame) -> pd.DataFrame:
    """Score shot rows in the feature store with the production CxG model.

    Shots live in the feature store, not in actions.parquet (which is
    passes/carries only), so the CxG scoring and the label linkage both source
    shots from here, exactly like scripts/train_cxa.py.
    """
    import joblib
    import yaml

    cfg = yaml.safe_load((_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    cxg_rel = (cfg.get("production") or {}).get("cxg")
    if not cxg_rel:
        raise SystemExit("No production CxG model in configs/models.yaml.")
    model = joblib.load(_ROOT / cxg_rel)

    df = features_df.copy()
    df["cxg"] = 0.0
    shot_mask = df["event_type"].astype(str) == "shot"
    if shot_mask.any():
        proba = np.asarray(model.predict_proba(df.loc[shot_mask]))
        scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        df.loc[shot_mask, "cxg"] = scores.astype(float)
    return df


def _attach_labels(actions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Attach shot_created and resulting_shot_cxg, mirroring scripts/train_cxa.py.

    Shots are linked to actions by possession: resulting_shot_cxg = mean CxG of
    the possession's shots (0 if the possession produced no shot).
    """
    poss = "possession_internal_id"
    shots = features[features["event_type"].astype(str) == "shot"]
    poss_with_shots = set(shots[poss].dropna())
    poss_mean_cxg = shots.groupby(poss)["cxg"].mean()

    df = actions.copy()
    df["shot_created"] = df[poss].isin(poss_with_shots).astype(int)
    df["resulting_shot_cxg"] = df[poss].map(poss_mean_cxg).fillna(0.0)
    return df


def _load_heldout() -> pd.DataFrame:
    from analysis._utils import load_actions, load_features, load_matches

    features = _score_shots(load_features())
    actions = _attach_labels(load_actions(), features)
    matches = load_matches()[["internal_id", "split_role"]].rename(
        columns={"internal_id": "_match_id"}
    )
    actions = actions.merge(matches, left_on="match_internal_id", right_on="_match_id", how="left")
    ho = actions[
        (actions["split_role"] == "val_test") & (actions["event_type"].isin(CREATIVE_ACTION_TYPES))
    ].copy()
    if ho.empty:
        raise SystemExit("CxA composite held-out set is empty. Check ids / dvc pull.")
    logger.info(
        "CxA composite held-out: %d creative actions (%.1f%% created a shot)",
        len(ho),
        100 * ho["shot_created"].mean(),
    )
    return ho


def _reliability(pred: np.ndarray, realised: np.ndarray, n_bins: int = N_BINS):
    """Quantile-bin on predicted cxa; return per-bin mean predicted/realised + ECE."""
    order = np.argsort(pred, kind="stable")
    xs, ys, ns = [], [], []
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        xs.append(float(pred[idx].mean()))
        ys.append(float(realised[idx].mean()))
        ns.append(int(len(idx)))
    xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)
    ece = float(np.sum(ns / ns.sum() * np.abs(xs - ys)))
    return xs, ys, ns, ece


def _smoke():
    rng = np.random.default_rng(0)
    n = 5000
    pred = np.abs(rng.normal(0.03, 0.02, n))
    realised = np.clip(pred + rng.normal(0, 0.02, n), 0, None)
    return pred, realised


def main(smoke: bool = False) -> dict:
    if smoke:
        logger.info("SMOKE MODE - synthetic data, no repo data required.")
        pred, realised = _smoke()
    else:
        import joblib

        from src.models.cxa.cxa_pipeline import CxAPipeline  # noqa: F401  (needed to unpickle)

        if not COMPOSITE_MODEL.exists():
            raise SystemExit(f"Composite CxA model not found: {COMPOSITE_MODEL}")
        ho = _load_heldout()
        model = joblib.load(COMPOSITE_MODEL)
        scored = model.score(ho, filter_creative=False)
        pred = scored["cxa"].to_numpy(dtype=float)
        realised = ho["resulting_shot_cxg"].to_numpy(dtype=float)

    from scipy.stats import spearmanr

    xs, ys, ns, ece = _reliability(pred, realised)
    sp = spearmanr(pred, realised).statistic if len(pred) > 2 else float("nan")
    mean_pred, mean_real = float(pred.mean()), float(realised.mean())

    if ece < 0.005 and abs(mean_pred - mean_real) < 0.005:
        verdict = "WELL CALIBRATED: composite cxa tracks realised shot value across deciles."
    elif ece < 0.02:
        verdict = (
            "PARTIALLY CALIBRATED: composite cxa is right on average but drifts in the tails; "
            "usable for ranking, not as a precise expected-value."
        )
    else:
        verdict = (
            "MISCALIBRATED: composite cxa does not match realised shot value across deciles; "
            "treat it as an ordinal creation score, not a calibrated expected value."
        )

    result = {
        "metric": "cxa_composite",
        "evaluation": "Euro 2024 held-out creative actions; predicted cxa vs realised resulting_shot_cxg",
        "n_actions": int(len(pred)),
        "n_bins": N_BINS,
        "mean_predicted_cxa": mean_pred,
        "mean_realised_shot_cxg": mean_real,
        "expected_calibration_error": ece,
        "spearman_pred_vs_realised": float(sp) if sp == sp else None,
        "reliability_curve": {
            "mean_predicted": [float(v) for v in xs],
            "mean_realised": [float(v) for v in ys],
            "n_per_bin": [int(v) for v in ns],
        },
        "verdict": verdict,
        "usable_resolution": (
            "The composite discriminates high- from low-value creative actions "
            f"(Spearman {float(sp):.3f} vs realised shot value) but its ECE of {ece:.4f} "
            "means the absolute cxa value is only approximately an expected shot value."
        ),
        "smoke": smoke,
    }

    try:
        from analysis._utils import save_json

        save_json(result, "cxa_composite_calibration")
        _plot(xs, ys, ece, sp)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not persist via repo utils (%s).", e)

    logger.info("VERDICT: %s", verdict)
    logger.info(
        "  ECE=%.4f  mean_pred=%.4f  mean_realised=%.4f  spearman=%.3f",
        ece,
        mean_pred,
        mean_real,
        float(sp),
    )
    return result


def _plot(xs, ys, ece, sp):
    from analysis._utils import save_fig

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(float(np.max(xs)), float(np.max(ys)), 1e-6) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="#999", lw=1, label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#A50044", lw=1.6, ms=5, label="composite cxa (deciles)")
    ax.set_xlabel("Mean predicted cxa")
    ax.set_ylabel("Mean realised same-possession shot value")
    ax.set_title(f"CxA composite reliability (ECE={ece:.4f}, Spearman={float(sp):.3f})")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    save_fig("cxa_composite_reliability", "incremental_lift")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run on synthetic data")
    main(smoke=ap.parse_args().smoke)
