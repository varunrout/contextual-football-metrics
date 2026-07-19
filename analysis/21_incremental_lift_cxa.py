"""
analysis/21_incremental_lift_cxa.py
===================================
Part 9b - Honest incremental-lift evaluation for CxA (contextual expected assists).

Unlike CxG there is NO off-the-shelf external benchmark (StatsBomb does not ship
a public xA the way it ships xG). So the honest question is narrower and must be
stated as such: do CONTEXTUAL features beat the TRADITIONAL-feature model at
predicting shot creation, on the same held-out rows, by more than noise?

The shot-creation stage (target `shot_created`, ~15.9% positive) is the
classification core of CxA and the part where a baseline comparison is
meaningful. This script scores, on the identical Euro 2024 held-out actions:

  * naive_prevalence  - predict the constant training positive rate (the floor)
  * <traditional>     - the traditional-feature creation model, if present
  * <contextual>      - the contextual-feature creation model

with one ECE definition and a paired bootstrap giving a 95% CI on the
contextual-minus-traditional delta. Contextual is credited only if it beats the
traditional model (not merely the naive floor) with a CI excluding zero.

Outputs
-------
reports/incremental_lift_cxa.json
reports/figures/incremental_lift/cxa_calibration_overlay.png
reports/figures/incremental_lift/cxa_delta_forest.png

Run
---
    python analysis/21_incremental_lift_cxa.py
    python analysis/21_incremental_lift_cxa.py --smoke
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("21_incremental_lift_cxa")

HELDOUT_COMPETITION_ID = "55"
HELDOUT_SEASON_ID = "282"
TARGET = "shot_created"
N_BOOTSTRAP = 2000
N_CAL_BINS = 10
RNG_SEED = 0
MODELS_DIR = _ROOT / "models" / "cxa"


def _ece(y_true, y_prob, n_bins=N_CAL_BINS):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def _metrics(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ece": _ece(y_true, y_prob),
    }


def _reliability(y_true, y_prob, n_bins=N_CAL_BINS):
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


def _paired_bootstrap(y_true, preds, candidate, baseline):
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_true)
    dll, dauc = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        pc = np.clip(preds[candidate][idx], 1e-6, 1 - 1e-6)
        pb = np.clip(preds[baseline][idx], 1e-6, 1 - 1e-6)
        dll.append(log_loss(yt, pc) - log_loss(yt, pb))
        dauc.append(roc_auc_score(yt, pc) - roc_auc_score(yt, pb))

    def ci(a):
        a = np.asarray(a)
        return {
            "delta_mean": float(a.mean()),
            "ci_low": float(np.percentile(a, 2.5)),
            "ci_high": float(np.percentile(a, 97.5)),
        }

    return {"log_loss": ci(dll), "roc_auc": ci(dauc)}


def _prob(model, df):
    p = np.asarray(model.predict_proba(df))
    return p[:, 1] if p.ndim == 2 else p.ravel()


def _discover_models():
    """Return {'traditional': path|None, 'contextual': path|None} by filename."""
    found = {"traditional": None, "contextual": None}
    if not MODELS_DIR.exists():
        return found
    paths = sorted(MODELS_DIR.glob("*.joblib")) + sorted(MODELS_DIR.glob("*.pkl"))

    # The target here is shot_created, so the correct predictor is a
    # shot-creation-stage model. Prefer files whose name marks the creation
    # stage; only fall back to a generic contextual model if none is found. This
    # avoids picking the composite CxA pipeline, which has no predict_proba.
    def _pick(kind_terms: tuple[str, ...]) -> Path | None:
        creation = None
        generic = None
        for p in paths:
            name = p.name.lower()
            if not any(t in name for t in kind_terms):
                continue
            if "quality" in name:  # the shot-quality stage is not the creation target
                continue
            if "creation" in name or "shot_created" in name:
                creation = creation or p
            else:
                generic = generic or p
        return creation or generic

    found["contextual"] = _pick(("contextual",))
    found["traditional"] = _pick(("traditional", "baseline"))
    return found


def _load_heldout():
    from analysis._utils import derive_shot_created, heldout_mask, load_actions

    actions = load_actions()
    if TARGET not in actions.columns:
        actions = derive_shot_created(actions)
    # Resolve Euro 2024 held-out via matches.parquet (feature tables carry a
    # hashed competition_id and no season_id).
    ho = actions.loc[
        heldout_mask(actions, int(HELDOUT_COMPETITION_ID), int(HELDOUT_SEASON_ID))
    ].copy()
    if ho.empty or TARGET not in ho.columns:
        raise SystemExit("CxA held-out set empty or missing target. Check ids / dvc pull.")
    logger.info("CxA held-out: %d actions, %.2f%% created", len(ho), 100 * ho[TARGET].mean())
    return ho


def _smoke():
    rng = np.random.default_rng(2)
    n = 4000
    latent = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(latent - 1.7)))).astype(int)

    def sig(z):
        return 1 / (1 + np.exp(-z))

    prev = y.mean()
    preds = {
        "naive_prevalence": np.full(n, prev),
        "traditional": sig(latent - 1.7 + rng.normal(scale=0.5, size=n)),
        "contextual": sig(latent - 1.7 + rng.normal(scale=0.35, size=n)),
    }
    return y, preds, "contextual", "traditional"


def main(smoke=False):
    if smoke:
        logger.info("SMOKE MODE - synthetic data.")
        y, preds, cand, base = _smoke()
    else:
        import joblib

        ho = _load_heldout()
        y = ho[TARGET].astype(int).to_numpy()
        preds = {"naive_prevalence": np.full(len(y), float(y.mean()))}
        found = _discover_models()
        for kind in ("traditional", "contextual"):
            if found[kind] is not None:
                preds[kind] = _prob(joblib.load(found[kind]), ho)
                logger.info("loaded %s creation model: %s", kind, found[kind].name)
        cand = "contextual" if "contextual" in preds else None
        base = "traditional" if "traditional" in preds else "naive_prevalence"
        if cand is None:
            raise SystemExit(
                "No contextual CxA creation model found in models/cxa - cannot evaluate lift."
            )

    per_model = {k: _metrics(y, v) for k, v in preds.items()}
    deltas = _paired_bootstrap(y, preds, cand, base)
    beats_ll = deltas["log_loss"]["ci_high"] < 0
    beats_auc = deltas["roc_auc"]["ci_low"] > 0
    if beats_ll and beats_auc:
        verdict = f"CONTEXTUAL ADDS VALUE over {base}: lower log-loss and higher AUC, both CIs exclude zero."
    elif beats_ll or beats_auc:
        verdict = (
            f"MIXED vs {base}: improves one metric with confidence, not both. Not a clean win."
        )
    else:
        verdict = f"NO DEMONSTRABLE LIFT over {base}: no CI excludes zero. Report contextual CxA as not yet proven."

    result = {
        "metric": "cxa_shot_creation",
        "evaluation": "identical Euro 2024 held-out actions; one ECE; paired bootstrap",
        "note": "No external xA benchmark exists; the meaningful test is contextual vs traditional, not vs an off-the-shelf model.",
        "n_actions": int(len(y)),
        "n_created": int(y.sum()),
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

        save_json(result, "incremental_lift_cxa")
        _plot_cal(y, preds)
        _plot_forest(deltas, cand, base)
    except Exception as e:
        logger.warning("Could not persist via repo utils (%s).", e)
    logger.info("VERDICT: %s", verdict)
    for k, m in per_model.items():
        logger.info(
            "  %-16s log_loss=%.4f auc=%.4f pr_auc=%.4f ece=%.4f",
            k,
            m["log_loss"],
            m["roc_auc"],
            m["pr_auc"],
            m["ece"],
        )
    return result


def _plot_cal(y, preds):
    from analysis._utils import save_fig

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect")
    for name, p in preds.items():
        if name == "naive_prevalence":
            continue
        xs, ys = _reliability(y, np.clip(p, 1e-6, 1 - 1e-6))
        ax.plot(xs, ys, "o-", lw=1.6, ms=4, label=name)
    ax.set_xlabel("Mean predicted P(shot created)")
    ax.set_ylabel("Observed creation rate")
    ax.set_title("CxA calibration on identical Euro 2024 held-out set")
    ax.legend(frameon=False, fontsize=8)
    save_fig("cxa_calibration_overlay", "incremental_lift")
    plt.close(fig)


def _plot_forest(deltas, cand, base):
    from analysis._utils import save_fig

    rows = [
        (f"{cand} - {base}\nlog-loss (lower better)", deltas["log_loss"]),
        (f"{cand} - {base}\nAUC (higher better)", deltas["roc_auc"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for i, (_label, d) in enumerate(rows):
        ax.plot([d["ci_low"], d["ci_high"]], [i, i], color="#004D98", lw=2)
        ax.plot(d["delta_mean"], i, "o", color="#A50044", ms=6)
    ax.axvline(0, color="#999", ls="--", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Delta (candidate - baseline), 95% paired-bootstrap CI")
    ax.set_title("CxA: does contextual beat the traditional model?")
    fig.tight_layout()
    save_fig("cxa_delta_forest", "incremental_lift")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(smoke=ap.parse_args().smoke)
