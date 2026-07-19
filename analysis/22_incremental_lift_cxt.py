"""
analysis/22_incremental_lift_cxt.py
===================================
Part 9c - Honest incremental-lift evaluation for CxT (contextual threat / state value).

CxT is a REGRESSION problem: predict the value of a game state (target
`possession_cxg`). The right baseline is not an off-the-shelf xG but the
project's own STATIC zone-based threat surface (ZoneXTBaseline, a Karun-Singh
style grid). The honest question: does the contextual state-value model reduce
error versus a static zone lookup, on the same held-out actions, by more than
noise?

This scores, on the identical Euro 2024 held-out actions:
  * zone_baseline  - static ZoneXT value looked up by pitch zone
  * <contextual>   - the best contextual state-value model (predict(df))
on MAE / RMSE / Spearman, with a paired bootstrap 95% CI on the MAE reduction.
Contextual is credited only if it lowers MAE with a CI excluding zero.

>>> ONE THING TO CONFIRM before your first real run: the CONFIG block below
    (target column and x/y column names). They match the repo's conventions as
    read on 2026-07-17; if your schema differs, fix them here in one place.

Outputs
-------
reports/incremental_lift_cxt.json
reports/figures/incremental_lift/cxt_error_by_value_bin.png
reports/figures/incremental_lift/cxt_delta_forest.png

Run
---
    python analysis/22_incremental_lift_cxt.py
    python analysis/22_incremental_lift_cxt.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("22_incremental_lift_cxt")

# ---- CONFIG (confirm against your schema) --------------------------------
HELDOUT_COMPETITION_ID = "55"
HELDOUT_SEASON_ID = "282"
TARGET = "possession_cxg"  # continuous state-value target used by CxT training
X_COL, Y_COL = "x_location", "y_location"
ZONES_X, ZONES_Y = 16, 12  # ZoneXTBaseline grid (see src/models/cxt/baseline.py)
PITCH_X, PITCH_Y = 105.0, 68.0
CANDIDATE_MODEL = _ROOT / "models" / "cxt" / "lgbm_contextual.joblib"  # rank-1 contextual model
ZONE_PRIORS = _ROOT / "data" / "features" / "zone_xt_priors.parquet"
# --------------------------------------------------------------------------
N_BOOTSTRAP = 2000
RNG_SEED = 0


def _reg_metrics(y_true, y_pred):
    corr, _ = spearmanr(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(corr) if not np.isnan(corr) else None,
    }


def _paired_bootstrap(y_true, preds, candidate, baseline):
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_true)
    dmae, drmse = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        dmae.append(
            mean_absolute_error(yt, preds[candidate][idx])
            - mean_absolute_error(yt, preds[baseline][idx])
        )
        drmse.append(
            np.sqrt(mean_squared_error(yt, preds[candidate][idx]))
            - np.sqrt(mean_squared_error(yt, preds[baseline][idx]))
        )

    def ci(a):
        a = np.asarray(a)
        return {
            "delta_mean": float(a.mean()),
            "ci_low": float(np.percentile(a, 2.5)),
            "ci_high": float(np.percentile(a, 97.5)),
        }

    return {"mae": ci(dmae), "rmse": ci(drmse)}


def _zone_index(x, y):
    bx = np.clip((x / PITCH_X * ZONES_X).astype(int), 0, ZONES_X - 1)
    by = np.clip((y / PITCH_Y * ZONES_Y).astype(int), 0, ZONES_Y - 1)
    return (by * ZONES_X) + bx


def _zone_mean_lookup(train: pd.DataFrame, ho: pd.DataFrame) -> np.ndarray:
    """Per-zone mean possession_cxg learned on TRAIN, looked up by zone for held-out.

    This is the honest static baseline: the best a predictor can do knowing only
    the pitch zone of the action, calibrated to the same target the contextual
    model predicts (so MAE is a fair comparison).
    """
    tx = np.nan_to_num(train[X_COL].to_numpy(dtype=float))
    ty = np.nan_to_num(train[Y_COL].to_numpy(dtype=float))
    zone_mean = pd.Series(train[TARGET].to_numpy(dtype=float)).groupby(_zone_index(tx, ty)).mean()
    global_mean = float(np.nanmean(train[TARGET].to_numpy(dtype=float)))

    hx = np.nan_to_num(ho[X_COL].to_numpy(dtype=float))
    hy = np.nan_to_num(ho[Y_COL].to_numpy(dtype=float))
    ho_zone = _zone_index(hx, hy)
    return np.array([zone_mean.get(z, global_mean) for z in ho_zone], dtype=float)


def _zone_baseline_pred(ho: pd.DataFrame) -> np.ndarray | None:
    """Return the precomputed static zone-baseline prediction, or None."""
    if "zone_baseline_pred" in ho.columns:
        return ho["zone_baseline_pred"].to_numpy(dtype=float)
    logger.warning("No zone baseline available - reporting contextual metrics only.")
    return None


# CxT-eligible action types — must match scripts/train_cxt.py so the held-out
# evaluation set is identical to what the model was trained and validated on.
_CXT_ACTION_TYPES = {"pass", "carry", "cross", "cutback"}


def _attach_possession_cxg(features_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the possession_cxg target, mirroring scripts/train_cxt.py.

    The feature tables do not persist this target, so we rebuild it from the same
    production CxG model the trainer uses: score shot rows, then take the
    discounted sum of shot CxG within each possession.
    """
    import joblib
    import yaml

    from src.models.cxt.state_value_model import compute_possession_cxg

    cfg = yaml.safe_load((_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    cxg_rel = (cfg.get("production") or {}).get("cxg")
    if not cxg_rel:
        raise SystemExit(
            "No production CxG model in configs/models.yaml; cannot derive CxT target."
        )
    cxg_model = joblib.load(_ROOT / cxg_rel)

    df = features_df.copy()
    df["cxg"] = 0.0
    shot_mask = df["event_type"].astype(str) == "shot"
    if shot_mask.any():
        proba = np.asarray(cxg_model.predict_proba(df.loc[shot_mask]))
        scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        df.loc[shot_mask, "cxg"] = scores.astype(float)

    df[TARGET] = compute_possession_cxg(
        df,
        cxg_col="cxg",
        possession_id_col="possession_internal_id",
        match_id_col="match_internal_id",
    )
    return df


def _load_heldout():
    from analysis._utils import load_features, load_matches

    features = _attach_possession_cxg(load_features())

    # Held-out = Euro 2024 (split_role == "val_test"), resolved via matches.parquet.
    matches = load_matches()[["internal_id", "split_role"]].rename(
        columns={"internal_id": "_match_id"}
    )
    features = features.merge(
        matches, left_on="match_internal_id", right_on="_match_id", how="left"
    )

    # CxT-eligible actions only, matching the trainer, then split into the same
    # train / held-out (Euro 2024 = val_test) partitions train_cxt uses.
    cxt = features[features["event_type"].isin(_CXT_ACTION_TYPES)].copy()
    ho = cxt[cxt["split_role"] == "val_test"].copy()
    train = cxt[~cxt["split_role"].isin({"val_test", "test"})].copy()

    if ho.empty or TARGET not in ho.columns:
        raise SystemExit(
            f"CxT held-out empty or missing target {TARGET!r}. Check CONFIG / dvc pull."
        )

    ho["zone_baseline_pred"] = _zone_mean_lookup(train, ho)
    logger.info(
        "CxT held-out: %d actions (train %d), target mean=%.5f",
        len(ho),
        len(train),
        ho[TARGET].mean(),
    )
    return ho


def _smoke():
    rng = np.random.default_rng(3)
    n = 5000
    truth = np.abs(rng.normal(scale=0.05, size=n))
    preds = {
        "zone_baseline": truth + rng.normal(scale=0.05, size=n),  # coarse
        "contextual": truth + rng.normal(scale=0.035, size=n),  # finer
    }
    return truth, preds, "contextual", "zone_baseline"


def main(smoke=False):
    if smoke:
        logger.info("SMOKE MODE - synthetic data.")
        y, preds, cand, base = _smoke()
    else:
        import joblib

        ho = _load_heldout()
        y = ho[TARGET].astype(float).to_numpy()
        preds = {}
        if CANDIDATE_MODEL.exists():
            preds["contextual"] = np.asarray(joblib.load(CANDIDATE_MODEL).predict(ho), dtype=float)
        else:
            raise SystemExit(f"Contextual CxT model not found: {CANDIDATE_MODEL}")
        zb = _zone_baseline_pred(ho)
        if zb is not None:
            preds["zone_baseline"] = zb
        cand, base = "contextual", ("zone_baseline" if "zone_baseline" in preds else None)

    per_model = {k: _reg_metrics(y, v) for k, v in preds.items()}
    if base is None:
        verdict = "NO BASELINE AVAILABLE: contextual metrics reported, but the zone-baseline comparison could not be built (see CONFIG / priors). Lift is UNPROVEN until the baseline is wired."
        deltas = {}
    else:
        deltas = _paired_bootstrap(y, preds, cand, base)
        beats_mae = deltas["mae"]["ci_high"] < 0  # lower MAE is better
        if beats_mae:
            verdict = f"CONTEXTUAL ADDS VALUE over {base}: lower MAE, CI excludes zero."
        else:
            verdict = f"NO DEMONSTRABLE LIFT over {base}: MAE reduction CI includes zero. Report contextual CxT as not yet proven vs a static zone surface."

    result = {
        "metric": "cxt_state_value",
        "evaluation": "identical Euro 2024 held-out actions; regression; paired bootstrap",
        "baseline_note": "Baseline is the project's own static ZoneXT surface, the honest thing a contextual state-value model must beat.",
        "n_actions": int(len(y)),
        "n_bootstrap": N_BOOTSTRAP,
        "per_model": per_model,
        "candidate": cand,
        "baseline": base,
        "delta_vs_baseline": deltas,
        "verdict": verdict,
        "smoke": smoke,
    }
    try:
        from analysis._utils import save_json

        save_json(result, "incremental_lift_cxt")
        if base is not None:
            _plot_forest(deltas, cand, base)
        _plot_error_bins(y, preds)
    except Exception as e:
        logger.warning("Could not persist via repo utils (%s).", e)
    logger.info("VERDICT: %s", verdict)
    for k, m in per_model.items():
        logger.info("  %-14s mae=%.5f rmse=%.5f spearman=%s", k, m["mae"], m["rmse"], m["spearman"])
    return result


def _plot_forest(deltas, cand, base):
    from analysis._utils import save_fig

    rows = [
        (f"{cand} - {base}\nMAE (lower better)", deltas["mae"]),
        (f"{cand} - {base}\nRMSE (lower better)", deltas["rmse"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for i, (_label, d) in enumerate(rows):
        ax.plot([d["ci_low"], d["ci_high"]], [i, i], color="#004D98", lw=2)
        ax.plot(d["delta_mean"], i, "o", color="#A50044", ms=6)
    ax.axvline(0, color="#999", ls="--", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Delta (candidate - baseline), 95% paired-bootstrap CI")
    ax.set_title("CxT: does the contextual model beat the static zone surface?")
    fig.tight_layout()
    save_fig("cxt_delta_forest", "incremental_lift")
    plt.close(fig)


def _plot_error_bins(y, preds):
    from analysis._utils import save_fig

    fig, ax = plt.subplots(figsize=(6.5, 4))
    q = np.quantile(y, np.linspace(0, 1, 9))
    centres = 0.5 * (q[:-1] + q[1:])
    for name, p in preds.items():
        errs = []
        for i in range(len(q) - 1):
            m = (y >= q[i]) & (y <= q[i + 1])
            errs.append(float(np.mean(np.abs(p[m] - y[m]))) if m.any() else np.nan)
        ax.plot(centres, errs, "o-", lw=1.6, ms=4, label=name)
    ax.set_xlabel("True state value (bin centre)")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("CxT error by true-value bin (held-out)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig("cxt_error_by_value_bin", "incremental_lift")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(smoke=ap.parse_args().smoke)
