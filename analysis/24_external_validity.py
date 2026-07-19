"""
analysis/24_external_validity.py
================================
CONT-F07 - external validity: do the metrics rank players like reality?

A metric that fits well in-sample is not useful unless it ranks the right
players. This script aggregates the production metrics to the player level and
correlates them with the outcomes they are supposed to predict:

  * CxG:  season total CxG per player  vs  actual goals.
  * CxA:  season total CxA per player  vs  actual assists (pass_goal_assist).

Both are pooled over every ingested match (the metrics are scored out of their
own model; goals/assists come from the raw events). Players are filtered to a
minimum activity so a handful of shots/passes do not dominate the ranking. We
report Spearman (rank) and Pearson correlation, and state the scouting decision
each metric can and cannot support.

Outputs
-------
reports/external_validity.json
reports/figures/external_validity/cxg_vs_goals.png
reports/figures/external_validity/cxa_vs_assists.png

Run
---
    python analysis/24_external_validity.py
    python analysis/24_external_validity.py --smoke   # synthetic, no data/models
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
logger = logging.getLogger("24_external_validity")

MIN_SHOTS = 5  # players below this are excluded from the CxG check
MIN_ACTIONS = 50  # players below this are excluded from the CxA check
PLAYER_KEY = "player_internal_id"


def _production_model(metric: str):
    import joblib
    import yaml

    cfg = yaml.safe_load((_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    rel = (cfg.get("production") or {}).get(metric)
    if not rel:
        raise SystemExit(f"No production {metric} model in configs/models.yaml.")
    path = _ROOT / rel
    if not path.exists():
        raise SystemExit(f"Production {metric} model not found: {path}")
    return joblib.load(path)


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr

    return {
        "spearman": float(spearmanr(x, y).statistic),
        "pearson": float(pearsonr(x, y).statistic),
        "n_players": int(len(x)),
    }


# ---------------------------------------------------------------------------
# CxG vs goals
# ---------------------------------------------------------------------------
def _cxg_vs_goals() -> tuple[pd.DataFrame, dict]:
    from analysis._utils import load_shots

    shots = load_shots()
    model = _production_model("cxg")
    proba = np.asarray(model.predict_proba(shots))
    shots = shots.copy()
    shots["cxg"] = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
    shots["goal"] = pd.to_numeric(shots.get("goal"), errors="coerce").fillna(0).astype(int)

    agg = (
        shots.groupby(PLAYER_KEY)
        .agg(cxg_sum=("cxg", "sum"), goals=("goal", "sum"), n_shots=("cxg", "size"))
        .reset_index()
    )
    agg = agg[agg["n_shots"] >= MIN_SHOTS]
    result = _corr(agg["cxg_sum"].to_numpy(), agg["goals"].to_numpy())
    result["min_shots"] = MIN_SHOTS
    logger.info(
        "CxG vs goals: %d players (>=%d shots), Spearman=%.3f Pearson=%.3f",
        result["n_players"],
        MIN_SHOTS,
        result["spearman"],
        result["pearson"],
    )
    return agg, result


# ---------------------------------------------------------------------------
# CxA vs assists
# ---------------------------------------------------------------------------
def _actual_assists() -> pd.DataFrame:
    """Actual assists per player from raw events (pass_goal_assist)."""
    from analysis._utils import load_events

    events = load_events()
    key = PLAYER_KEY if PLAYER_KEY in events.columns else "player_id"
    a = pd.to_numeric(events.get("pass_goal_assist"), errors="coerce").fillna(0)
    events = events.assign(_assist=(a > 0).astype(int))
    return (
        events.groupby(key)["_assist"]
        .sum()
        .rename("assists")
        .reset_index()
        .rename(columns={key: PLAYER_KEY})
    )


def _cxa_vs_assists() -> tuple[pd.DataFrame, dict]:
    from analysis._utils import load_actions
    from src.models.cxa.cxa_pipeline import CxAPipeline  # noqa: F401  (needed to unpickle)

    actions = load_actions()
    model = _production_model("cxa")
    scored = model.score(actions, filter_creative=False)
    actions = actions.copy()
    actions["cxa"] = pd.to_numeric(scored["cxa"], errors="coerce").fillna(0.0).to_numpy()

    agg = (
        actions.groupby(PLAYER_KEY)
        .agg(cxa_sum=("cxa", "sum"), n_actions=("cxa", "size"))
        .reset_index()
    )
    agg = agg.merge(_actual_assists(), on=PLAYER_KEY, how="left")
    agg["assists"] = agg["assists"].fillna(0).astype(int)
    agg = agg[agg["n_actions"] >= MIN_ACTIONS]
    result = _corr(agg["cxa_sum"].to_numpy(), agg["assists"].to_numpy())
    result["min_actions"] = MIN_ACTIONS
    logger.info(
        "CxA vs assists: %d players (>=%d actions), Spearman=%.3f Pearson=%.3f",
        result["n_players"],
        MIN_ACTIONS,
        result["spearman"],
        result["pearson"],
    )
    return agg, result


# ---------------------------------------------------------------------------
def _verdict(name: str, spearman: float, outcome: str) -> str:
    if spearman >= 0.6:
        return (
            f"{name} ranks players by {outcome} reliably (Spearman {spearman:.2f}): "
            f"usable to shortlist {outcome} on expected rather than realised output."
        )
    if spearman >= 0.4:
        return (
            f"{name} tracks {outcome} moderately (Spearman {spearman:.2f}): usable as a "
            f"noisy prior, best combined with other evidence, not as a sole ranking."
        )
    return (
        f"{name} does not rank {outcome} well (Spearman {spearman:.2f}): treat as weak "
        f"external validity; do not scout on it alone."
    )


def _plot(agg, xcol, ycol, title, fname):
    from analysis._utils import save_fig

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(agg[xcol], agg[ycol], alpha=0.5, s=18, color="#004D98", edgecolor="none")
    ax.set_xlabel(f"Season {xcol}")
    ax.set_ylabel(f"Actual {ycol}")
    ax.set_title(title)
    save_fig(fname, "external_validity")
    plt.close(fig)


def _smoke():
    rng = np.random.default_rng(0)
    n = 200
    latent = rng.gamma(2.0, 1.0, n)
    cxg = pd.DataFrame({"cxg_sum": latent, "goals": rng.poisson(latent)})
    cxa = pd.DataFrame({"cxa_sum": latent * 0.5, "assists": rng.poisson(latent * 0.5)})
    return cxg, cxa


def main(smoke: bool = False) -> dict:
    if smoke:
        logger.info("SMOKE MODE - synthetic data, no repo data required.")
        cxg_agg, cxa_agg = _smoke()
        cxg_res = _corr(cxg_agg["cxg_sum"].to_numpy(), cxg_agg["goals"].to_numpy())
        cxa_res = _corr(cxa_agg["cxa_sum"].to_numpy(), cxa_agg["assists"].to_numpy())
    else:
        cxg_agg, cxg_res = _cxg_vs_goals()
        cxa_agg, cxa_res = _cxa_vs_assists()

    cxg_verdict = _verdict("CxG", cxg_res["spearman"], "goals")
    cxa_verdict = _verdict("CxA", cxa_res["spearman"], "assists")

    result = {
        "evaluation": "player-level, pooled over all ingested matches; metrics scored by the "
        "production model, goals/assists from raw events",
        "cxg_vs_goals": {**cxg_res, "decision": cxg_verdict},
        "cxa_vs_assists": {**cxa_res, "decision": cxa_verdict},
        "smoke": smoke,
    }

    try:
        from analysis._utils import save_json

        save_json(result, "external_validity")
        if not smoke:
            _plot(
                cxg_agg,
                "cxg_sum",
                "goals",
                f"CxG vs goals (Spearman {cxg_res['spearman']:.2f})",
                "cxg_vs_goals",
            )
            _plot(
                cxa_agg,
                "cxa_sum",
                "assists",
                f"CxA vs assists (Spearman {cxa_res['spearman']:.2f})",
                "cxa_vs_assists",
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not persist via repo utils (%s).", e)

    logger.info("CxG decision: %s", cxg_verdict)
    logger.info("CxA decision: %s", cxa_verdict)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run on synthetic data")
    main(smoke=ap.parse_args().smoke)
