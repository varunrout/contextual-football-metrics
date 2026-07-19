"""
analysis/20_incremental_lift_vs_baselines.py
============================================
Part 9 - Honest incremental-lift evaluation for CxG.

WHY THIS EXISTS
---------------
The repo currently reports three CxG evaluations that are NOT comparable and
disagree with each other:

  * reports/statsbomb_baseline_metrics.json  - StatsBomb published xG, scored on
    5-fold CV over ~4,123 shots (AUC 0.839, ECE 0.013).
  * reports/cxg_training_summary.json        - model leaderboard, scored on the
    Euro 2024 held-out set (1,340 shots); here glm_contextual wins.
  * reports/model_comparison_cxg.json        - model-comparison suite, scored on
    CV; here baseline_logit wins.

You cannot claim "contextual features add value" from three numbers measured on
three different samples with two different ECE definitions. This script produces
the ONE apples-to-apples comparison: StatsBomb xG, the traditional-feature
baseline, and the contextual model, all scored on the IDENTICAL Euro 2024
held-out rows, with one ECE definition, and with a PAIRED bootstrap so every
"contextual minus baseline" delta carries a 95% confidence interval.

The headline question is deliberately strict: does the contextual model beat the
STRONGER of the two baselines (StatsBomb xG), and does the delta's CI exclude
zero? If not, the honest finding is that contextual context does not yet add
demonstrable value over an off-the-shelf xG, and we say so.

Outputs
-------
reports/incremental_lift_cxg.json
reports/figures/incremental_lift/cxg_calibration_overlay.png
reports/figures/incremental_lift/cxg_delta_forest.png

Run
---
    python analysis/20_incremental_lift_vs_baselines.py
    python analysis/20_incremental_lift_vs_baselines.py --smoke   # no data needed
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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("20_incremental_lift")

# Euro 2024 = the final external held-out set (see configs/competitions.yaml).
HELDOUT_COMPETITION_ID = "55"
HELDOUT_SEASON_ID = "282"

XG_COL = "shot_statsbomb_xg"
TARGET = "goal"
N_BOOTSTRAP = 2000
N_CAL_BINS = 10
RNG_SEED = 0

# Models to compare against the StatsBomb xG benchmark. Paths relative to repo root.
MODELS = [
    ("baseline_logit", "traditional", _ROOT / "models" / "cxg" / "baseline_logit.joblib"),
    ("glm_contextual", "contextual", _ROOT / "models" / "cxg" / "glm_contextual.joblib"),
]
# The model whose incremental value we are testing, and the two baselines it must beat.
CANDIDATE = "glm_contextual"
STRONG_BASELINE = "statsbomb_xg"  # off-the-shelf xG - the strict benchmark
WEAK_BASELINE = "baseline_logit"  # our own traditional-feature model


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_CAL_BINS) -> float:
    """Expected Calibration Error - ONE definition, applied to every predictor."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        frac = mask.sum() / n
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += frac * abs(acc - conf)
    return float(ece)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "ece": _ece(y_true, y_prob),
        "mean_pred": float(y_prob.mean()),
    }


def _reliability(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_CAL_BINS):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() < 5:
            continue
        xs.append(float(y_prob[mask].mean()))
        ys.append(float(y_true[mask].mean()))
    return xs, ys


def _paired_bootstrap(y_true, preds: dict, candidate: str, baselines: list[str]):
    """
    Paired bootstrap: resample rows ONCE per iteration and score every predictor
    on the same resample, so the delta (candidate - baseline) is a proper paired
    statistic. Returns percentile 95% CIs for the delta on log_loss and roc_auc.
    """
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_true)
    deltas = {b: {"log_loss": [], "roc_auc": []} for b in baselines}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:  # degenerate resample - skip
            continue
        scored = {}
        for name, p in preds.items():
            pp = np.clip(p[idx], 1e-6, 1 - 1e-6)
            scored[name] = (log_loss(yt, pp), roc_auc_score(yt, pp))
        for b in baselines:
            deltas[b]["log_loss"].append(scored[candidate][0] - scored[b][0])
            deltas[b]["roc_auc"].append(scored[candidate][1] - scored[b][1])
    out = {}
    for b in baselines:
        out[b] = {}
        for metric, arr in deltas[b].items():
            a = np.asarray(arr)
            out[b][metric] = {
                "delta_mean": float(a.mean()),
                "ci_low": float(np.percentile(a, 2.5)),
                "ci_high": float(np.percentile(a, 97.5)),
                # for log_loss lower is better -> candidate wins if CI high < 0
                # for roc_auc higher is better -> candidate wins if CI low > 0
            }
    return out


# ---------------------------------------------------------------------------
# data / predictions
# ---------------------------------------------------------------------------
def _prob(model, df: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(df)
    p = np.asarray(p)
    return p[:, 1] if p.ndim == 2 else p.ravel()


def _load_heldout():
    from analysis._utils import heldout_mask, load_shots  # noqa: E402

    shots = load_shots()
    # Resolve the Euro 2024 held-out split via matches.parquet: the feature
    # tables carry a hashed competition_id and no season_id, so a direct
    # competition/season filter on shots does not work.
    ho = shots.loc[heldout_mask(shots, int(HELDOUT_COMPETITION_ID), int(HELDOUT_SEASON_ID))].copy()
    if ho.empty:
        raise SystemExit(
            "Held-out set is empty. Check competition/season ids and that data is pulled (dvc pull)."
        )
    if XG_COL not in ho.columns or TARGET not in ho.columns:
        raise SystemExit(f"Held-out set is missing required columns {XG_COL!r}/{TARGET!r}.")
    logger.info(
        "Held-out (Euro 2024): %d shots, %d goals (%.1f%%)",
        len(ho),
        int(ho[TARGET].sum()),
        100 * ho[TARGET].mean(),
    )
    return ho


def _build_predictions(ho: pd.DataFrame) -> dict:
    import joblib  # noqa: E402

    y = ho[TARGET].astype(int).to_numpy()
    preds = {STRONG_BASELINE: np.clip(ho[XG_COL].astype(float).to_numpy(), 1e-6, 1 - 1e-6)}
    for name, _fs, path in MODELS:
        if not path.exists():
            logger.warning("Model not found, skipping: %s", path)
            continue
        model = joblib.load(path)
        preds[name] = _prob(model, ho)
    return y, preds


# ---------------------------------------------------------------------------
# smoke mode (no data / no models required) - proves the code path runs
# ---------------------------------------------------------------------------
def _smoke():
    rng = np.random.default_rng(1)
    n = 1200
    latent = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(latent - 2.0)))).astype(int)

    # statsbomb xg: well-calibrated-ish; baseline: noisier; contextual: slightly better
    def sig(z):
        return 1 / (1 + np.exp(-z))

    preds = {
        STRONG_BASELINE: sig(latent - 2.0 + rng.normal(scale=0.15, size=n)),
        WEAK_BASELINE: sig(latent - 2.0 + rng.normal(scale=0.55, size=n)),
        CANDIDATE: sig(latent - 2.0 + rng.normal(scale=0.35, size=n)),
    }
    return y, preds


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(smoke: bool = False) -> dict:
    if smoke:
        logger.info("SMOKE MODE - synthetic data, no repo data required.")
        y, preds = _smoke()
    else:
        ho = _load_heldout()
        y, preds = _build_predictions(ho)

    if CANDIDATE not in preds:
        raise SystemExit(f"Candidate model {CANDIDATE!r} unavailable - cannot evaluate lift.")

    per_model = {name: _metrics(y, p) for name, p in preds.items()}
    baselines = [b for b in (STRONG_BASELINE, WEAK_BASELINE) if b in preds]
    deltas = _paired_bootstrap(y, preds, CANDIDATE, baselines)

    # honest verdict against the STRONGER available baseline
    strict = STRONG_BASELINE if STRONG_BASELINE in preds else WEAK_BASELINE
    d_ll = deltas[strict]["log_loss"]
    d_auc = deltas[strict]["roc_auc"]
    beats_ll = d_ll["ci_high"] < 0  # log-loss lower is better
    beats_auc = d_auc["ci_low"] > 0  # auc higher is better
    if beats_ll and beats_auc:
        verdict = f"CONTEXTUAL ADDS VALUE over {strict}: lower log-loss and higher AUC, both CIs exclude zero."
    elif beats_ll or beats_auc:
        verdict = (
            f"MIXED: contextual improves one metric over {strict} but not both with confidence. "
            f"Not a clean win."
        )
    else:
        verdict = (
            f"NO DEMONSTRABLE LIFT over {strict}: neither log-loss nor AUC improvement has a CI "
            f"excluding zero. Honest conclusion - contextual context does not yet beat off-the-shelf xG."
        )

    result = {
        "evaluation": "identical Euro 2024 held-out rows; single ECE definition; paired bootstrap",
        "n_shots": int(len(y)),
        "n_goals": int(y.sum()),
        "n_bootstrap": N_BOOTSTRAP,
        "per_model": per_model,
        "deltas_vs_baselines": deltas,
        "candidate": CANDIDATE,
        "strict_baseline": strict,
        "verdict": verdict,
        "smoke": smoke,
    }

    # ---- persist ----
    try:
        from analysis._utils import save_json

        save_json(result, "incremental_lift_cxg")
        _plot_calibration(y, preds)
        _plot_delta_forest(deltas, baselines)
        logger.info("Wrote reports/incremental_lift_cxg.json + figures.")
    except Exception as e:  # smoke test outside repo, or figures dir missing
        logger.warning("Could not persist via repo utils (%s). Result returned only.", e)

    logger.info("VERDICT: %s", verdict)
    for name, m in per_model.items():
        logger.info(
            "  %-16s log_loss=%.4f brier=%.4f auc=%.4f ece=%.4f",
            name,
            m["log_loss"],
            m["brier"],
            m["roc_auc"],
            m["ece"],
        )
    return result


def _plot_calibration(y, preds):
    from analysis._utils import save_fig

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect")
    for name, p in preds.items():
        xs, ys = _reliability(y, np.clip(p, 1e-6, 1 - 1e-6))
        ax.plot(xs, ys, "o-", lw=1.6, ms=4, label=name)
    ax.set_xlabel("Mean predicted P(goal)")
    ax.set_ylabel("Observed goal rate")
    ax.set_title("CxG calibration on identical Euro 2024 held-out set")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, max(0.6, ax.get_xlim()[1]))
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    save_fig("cxg_calibration_overlay", "incremental_lift")
    plt.close(fig)


def _plot_delta_forest(deltas, baselines):
    from analysis._utils import save_fig

    rows = []
    for b in baselines:
        rows.append((f"{CANDIDATE} - {b}\nlog-loss (lower better)", deltas[b]["log_loss"]))
        rows.append((f"{CANDIDATE} - {b}\nAUC (higher better)", deltas[b]["roc_auc"]))
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(rows) + 1))
    for i, (_label, d) in enumerate(rows):
        ax.plot([d["ci_low"], d["ci_high"]], [i, i], color="#004D98", lw=2)
        ax.plot(d["delta_mean"], i, "o", color="#A50044", ms=6)
    ax.axvline(0, color="#999", ls="--", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Delta (candidate - baseline), 95% paired-bootstrap CI")
    ax.set_title("Does contextual beat each baseline? CI crossing 0 = no clean win")
    fig.tight_layout()
    save_fig("cxg_delta_forest", "incremental_lift")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke", action="store_true", help="run on synthetic data, no repo data needed"
    )
    args = ap.parse_args()
    main(smoke=args.smoke)
