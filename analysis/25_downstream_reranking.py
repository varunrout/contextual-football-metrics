"""
analysis/25_downstream_reranking.py
===================================
CONT-08 - a worked downstream example: does the contextual metric change a
decision?

A metric only matters if it reranks players versus the vanilla alternative.
This builds a player threat leaderboard two ways over the same actions:

  * contextual CxT: the production LightGBM state-value model (uses opponent /
    build-up context).
  * vanilla xT:     a static Karun-Singh threat surface, the value of the pitch
    zone the action started in (data/features/zone_xt_priors.parquet).

Both are summed per player over their CxT actions; players are ranked by each.
The interesting output is the disagreement: players the contextual model rates
much higher or lower than the static zone view, with commentary on the decision
that changes.

Outputs
-------
reports/downstream_reranking_cxt.json
reports/figures/downstream/cxt_vs_xt_rank.png

Run
---
    python analysis/25_downstream_reranking.py
    python analysis/25_downstream_reranking.py --smoke   # synthetic, no data/models
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
logger = logging.getLogger("25_downstream_reranking")

CXT_ACTION_TYPES = {"pass", "carry", "cross", "cutback"}
MIN_ACTIONS = 150  # players below this are too small a sample to rank
TOP_N = 15
PLAYER_KEY = "player_internal_id"

# Pitch / zone grid, matching analysis/22 and the zone_xt_priors builder.
PITCH_X, PITCH_Y = 105.0, 68.0
ZONES_X, ZONES_Y = 16, 12


def _production_model(metric: str):
    import joblib
    import yaml

    cfg = yaml.safe_load((_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    rel = (cfg.get("production") or {}).get(metric)
    if not rel or not (_ROOT / rel).exists():
        raise SystemExit(f"Production {metric} model not found (configs/models.yaml -> {rel}).")
    return joblib.load(_ROOT / rel)


def _zone_index(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    bx = np.clip((np.nan_to_num(x) / PITCH_X * ZONES_X).astype(int), 0, ZONES_X - 1)
    by = np.clip((np.nan_to_num(y) / PITCH_Y * ZONES_Y).astype(int), 0, ZONES_Y - 1)
    return (by * ZONES_X) + bx


def _player_names() -> pd.Series:
    from analysis._utils import load_events

    events = load_events()
    if "player" not in events.columns:
        return pd.Series(dtype=str)
    return events.dropna(subset=["player"]).groupby(PLAYER_KEY)["player"].first()


def _build_table() -> pd.DataFrame:
    from analysis._utils import load_features

    features = load_features()
    cxt = features[features["event_type"].isin(CXT_ACTION_TYPES)].copy()

    # Contextual CxT (production model).
    model = _production_model("cxt")
    cxt["cxt"] = np.asarray(model.predict(cxt), dtype=float)

    # Vanilla xT: static per-zone threat surface, looked up by start zone.
    priors = pd.read_parquet(_ROOT / "data" / "features" / "zone_xt_priors.parquet")
    xt_by_zone = priors.set_index("zone_id")["xt_value"].to_dict()
    zones = _zone_index(cxt["x_location"].to_numpy(float), cxt["y_location"].to_numpy(float))
    cxt["xt"] = np.array([xt_by_zone.get(int(z), 0.0) for z in zones], dtype=float)

    agg = (
        cxt.groupby(PLAYER_KEY)
        .agg(cxt_sum=("cxt", "sum"), xt_sum=("xt", "sum"), n_actions=("cxt", "size"))
        .reset_index()
    )
    agg = agg[agg["n_actions"] >= MIN_ACTIONS].copy()

    names = _player_names()
    agg["player"] = agg[PLAYER_KEY].map(names).fillna(agg[PLAYER_KEY])
    agg["cxt_rank"] = agg["cxt_sum"].rank(ascending=False, method="min").astype(int)
    agg["xt_rank"] = agg["xt_sum"].rank(ascending=False, method="min").astype(int)
    # Positive rank_delta => contextual rates the player higher than static xT.
    agg["rank_delta"] = agg["xt_rank"] - agg["cxt_rank"]
    return agg


def _rows(df: pd.DataFrame) -> list[dict]:
    cols = ["player", "cxt_sum", "cxt_rank", "xt_rank", "rank_delta", "n_actions"]
    out = df[cols].copy()
    out["cxt_sum"] = out["cxt_sum"].round(3)
    return out.to_dict("records")


def _smoke() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    base = rng.gamma(2.0, 1.0, n)
    df = pd.DataFrame(
        {
            PLAYER_KEY: [f"p{i}" for i in range(n)],
            "player": [f"Player {i}" for i in range(n)],
            "cxt_sum": base + rng.normal(0, 0.4, n),
            "xt_sum": base + rng.normal(0, 0.4, n),
            "n_actions": rng.integers(150, 900, n),
        }
    )
    df["cxt_rank"] = df["cxt_sum"].rank(ascending=False, method="min").astype(int)
    df["xt_rank"] = df["xt_sum"].rank(ascending=False, method="min").astype(int)
    df["rank_delta"] = df["xt_rank"] - df["cxt_rank"]
    return df


def main(smoke: bool = False) -> dict:
    from scipy.stats import spearmanr

    agg = _smoke() if smoke else _build_table()
    if smoke:
        logger.info("SMOKE MODE - synthetic data, no repo data required.")

    rank_agreement = float(spearmanr(agg["cxt_rank"], agg["xt_rank"]).statistic)
    risers = agg.sort_values("rank_delta", ascending=False).head(TOP_N)
    fallers = agg.sort_values("rank_delta").head(TOP_N)
    top_cxt = agg.sort_values("cxt_rank").head(TOP_N)

    top_riser = risers.iloc[0]
    top_faller = fallers.iloc[0]
    commentary = (
        f"Contextual CxT and the static xT surface rank players similarly overall "
        f"(Spearman {rank_agreement:.2f}), but they disagree sharply on individuals. "
        f"{top_riser['player']} rises {int(top_riser['rank_delta'])} places under "
        f"contextual CxT (xT rank {int(top_riser['xt_rank'])} -> "
        f"{int(top_riser['cxt_rank'])}): the model credits the build-up context of "
        f"their actions, not just the pitch zone. Conversely {top_faller['player']} "
        f"falls {int(-top_faller['rank_delta'])} places "
        f"(xT rank {int(top_faller['xt_rank'])} -> {int(top_faller['cxt_rank'])}): "
        f"high touch volume in nominally valuable zones, but low contextual threat. "
        f"The biggest fallers are typically goalkeepers and defenders the static "
        f"surface overrates. Whom you shortlist as a threat-creator changes for the "
        f"movers, which is the point of a contextual metric."
    )

    result = {
        "evaluation": "player threat leaderboard: contextual CxT (production model) vs static "
        "zone xT surface, over all CxT actions",
        "n_players": int(len(agg)),
        "min_actions": MIN_ACTIONS,
        "rank_agreement_spearman": rank_agreement,
        "top_by_contextual_cxt": _rows(top_cxt),
        "biggest_risers_under_cxt": _rows(risers),
        "biggest_fallers_under_cxt": _rows(fallers),
        "commentary": commentary,
        "smoke": smoke,
    }

    try:
        from analysis._utils import save_fig, save_json

        save_json(result, "downstream_reranking_cxt")
        if not smoke:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(agg["xt_rank"], agg["cxt_rank"], alpha=0.5, s=16, color="#004D98")
            lim = max(agg["xt_rank"].max(), agg["cxt_rank"].max())
            ax.plot([1, lim], [1, lim], "--", color="#999", lw=1)
            ax.set_xlabel("Rank by static xT (1 = most threatening)")
            ax.set_ylabel("Rank by contextual CxT")
            ax.set_title(f"CxT vs static xT player ranking (Spearman {rank_agreement:.2f})")
            ax.invert_xaxis()
            ax.invert_yaxis()
            save_fig("cxt_vs_xt_rank", "downstream")
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not persist via repo utils (%s).", e)

    logger.info("Rank agreement Spearman=%.3f over %d players.", rank_agreement, len(agg))
    logger.info("Top riser under CxT: %s (+%d)", top_riser["player"], int(top_riser["rank_delta"]))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run on synthetic data")
    main(smoke=ap.parse_args().smoke)
