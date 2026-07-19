"""
analysis/11_hypothesis_cxg.py
==============================
Part 7a — Hypothesis Testing for CxG (goal target, shots.parquet).

8 hypotheses with Bonferroni correction.

Outputs
-------
reports/hypothesis_cxg.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import load_events, load_shots, save_json  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("11_hypothesis_cxg")


def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")


def _odds_ratio(df: pd.DataFrame, group_col: str, group_a, group_b, target: str) -> float:
    """Odds ratio: P(target=1|group_a) / P(target=0|group_a) vs group_b."""
    a = df[df[group_col] == group_a][target].dropna()
    b = df[df[group_col] == group_b][target].dropna()
    if a.empty or b.empty:
        return float("nan")
    p_a = a.mean()
    p_b = b.mean()
    if p_a in (0, 1) or p_b in (0, 1):
        return float("nan")
    return float((p_a / (1 - p_a)) / (p_b / (1 - p_b)))


def _record(
    name: str,
    statistic: float,
    pval: float,
    n_hypotheses: int,
    effect_size: float,
    effect_label: str = "effect_size",
    extra: dict | None = None,
) -> dict:
    bonferroni_pval = min(pval * n_hypotheses, 1.0)
    result: dict = {
        "hypothesis": name,
        "statistic": round(float(statistic), 4),
        "pval": round(float(pval), 6),
        "bonferroni_pval": round(float(bonferroni_pval), 6),
        effect_label: round(float(effect_size), 4) if not np.isnan(effect_size) else None,
        "reject_H0": bool(bonferroni_pval < 0.05),
    }
    if extra:
        result.update(extra)
    return result


def run_hypotheses(shots: pd.DataFrame, events: pd.DataFrame) -> list[dict]:
    N = 8  # total hypotheses for Bonferroni
    results = []

    # Ensure xG is available from events
    if "shot_statsbomb_xg" not in shots.columns and not events.empty:
        id_col = next(
            (c for c in ["event_id", "internal_id"] if c in shots.columns and c in events.columns),
            None,
        )
        if id_col and "shot_statsbomb_xg" in events.columns:
            xg_lookup = events[[id_col, "shot_statsbomb_xg"]].drop_duplicates(id_col)
            shots = shots.merge(xg_lookup, on=id_col, how="left")

    def _col(candidates: list[str]) -> str | None:
        return next((c for c in candidates if c in shots.columns), None)

    # H1: Headed shots have lower goal rate than foot shots
    bp_col = _col(["body_part", "bodypart"])
    if bp_col:
        bp = shots[[bp_col, "goal"]].dropna()
        bp[bp_col] = bp[bp_col].astype(str).str.lower()
        head = bp[bp[bp_col].str.contains("head")]["goal"]
        foot = bp[~bp[bp_col].str.contains("head")]["goal"]
        stat, pval = stats.ttest_ind(head, foot, equal_var=False)
        cohen = _cohen_d(head, foot)
        results.append(
            _record(
                "H1_header_vs_foot_goal_rate",
                stat,
                pval,
                N,
                cohen,
                "cohen_d",
                {"mean_head": round(head.mean(), 4), "mean_foot": round(foot.mean(), 4)},
            )
        )
    else:
        logger.warning("body_part column not found — skipping H1.")

    # H2: Shots inside 18-yard box have higher xG than outside
    x_col = _col(["x_location", "x"])
    y_col = _col(["y_location", "y"])
    xg_col = _col(["shot_statsbomb_xg"])
    if x_col and y_col and xg_col:
        shots_valid = shots[[x_col, y_col, xg_col]].copy()
        shots_valid[xg_col] = pd.to_numeric(shots_valid[xg_col], errors="coerce")
        shots_valid[x_col] = pd.to_numeric(shots_valid[x_col], errors="coerce")
        shots_valid[y_col] = pd.to_numeric(shots_valid[y_col], errors="coerce")
        shots_valid = shots_valid.dropna()
        inside = shots_valid[
            (shots_valid[x_col] >= 83) & (shots_valid[y_col].between(13.85, 54.15))
        ][xg_col]
        outside = shots_valid[
            ~((shots_valid[x_col] >= 83) & (shots_valid[y_col].between(13.85, 54.15)))
        ][xg_col]
        if len(inside) > 5 and len(outside) > 5:
            stat, pval = stats.mannwhitneyu(inside, outside, alternative="greater")
            effect = (
                float((inside.mean() - outside.mean()) / shots_valid[xg_col].std())
                if shots_valid[xg_col].std() > 0
                else float("nan")
            )
            results.append(
                _record(
                    "H2_inside_box_higher_xg",
                    stat,
                    pval,
                    N,
                    effect,
                    "cohen_d_approx",
                    {
                        "mean_inside": round(inside.mean(), 4),
                        "mean_outside": round(outside.mean(), 4),
                    },
                )
            )
    else:
        logger.warning("x_location, y_location, or shot_statsbomb_xg not found — skipping H2.")

    # H3: Set-piece shots have different goal rate than open-play
    sp_col = _col(["set_piece_type", "is_set_piece"])
    if sp_col and "goal" in shots.columns:
        sp = shots[[sp_col, "goal"]].dropna()
        sp_str = sp[sp_col].astype(str).str.lower()
        is_sp = sp_str.isin(
            [
                "corner",
                "free_kick",
                "penalty",
                "direct_free_kick",
                "indirect_free_kick",
                "true",
                "1",
            ]
        )
        a = sp[is_sp]["goal"]
        b = sp[~is_sp]["goal"]
        if len(a) > 5 and len(b) > 5:
            stat, pval = stats.ttest_ind(a, b, equal_var=False)
            results.append(
                _record(
                    "H3_setpiece_vs_openplay_goal_rate",
                    stat,
                    pval,
                    N,
                    _cohen_d(a, b),
                    "cohen_d",
                    {"mean_sp": round(a.mean(), 4), "mean_op": round(b.mean(), 4)},
                )
            )

    # H4: Shots preceded by sequence ≥ 3 actions have higher xG
    seq_col = _col(["events_before_action", "events_in_possession", "possession_length"])
    if seq_col and xg_col and xg_col in shots.columns:
        seq_shots = shots[[seq_col, xg_col]].copy()
        seq_shots[seq_col] = pd.to_numeric(seq_shots[seq_col], errors="coerce")
        seq_shots[xg_col] = pd.to_numeric(seq_shots[xg_col], errors="coerce")
        seq_shots = seq_shots.dropna()
        long_seq = seq_shots[seq_shots[seq_col] >= 3][xg_col]
        short_seq = seq_shots[seq_shots[seq_col] < 3][xg_col]
        if len(long_seq) > 5 and len(short_seq) > 5:
            stat, pval = stats.mannwhitneyu(long_seq, short_seq, alternative="two-sided")
            results.append(
                _record(
                    "H4_long_sequence_xg",
                    stat,
                    pval,
                    N,
                    _cohen_d(long_seq, short_seq),
                    "cohen_d",
                    {
                        "median_long": round(long_seq.median(), 4),
                        "median_short": round(short_seq.median(), 4),
                    },
                )
            )

    # H5: Home shots have higher goal rate than away shots
    ha_col = _col(["home_or_away", "is_home"])
    if ha_col and "goal" in shots.columns:
        ha = shots[[ha_col, "goal"]].dropna()
        ha_str = ha[ha_col].astype(str).str.lower()
        home = ha[ha_str.isin(["home", "1", "true"])]["goal"]
        away = ha[ha_str.isin(["away", "0", "false"])]["goal"]
        if len(home) > 5 and len(away) > 5:
            stat, pval = stats.ttest_ind(home, away, equal_var=False)
            results.append(
                _record(
                    "H5_home_vs_away_goal_rate",
                    stat,
                    pval,
                    N,
                    _cohen_d(home, away),
                    "cohen_d",
                    {"mean_home": round(home.mean(), 4), "mean_away": round(away.mean(), 4)},
                )
            )

    # H6: Shots with 360 data available have different xG distribution
    if "has_360" in shots.columns and xg_col and xg_col in shots.columns:
        with360 = shots[shots["has_360"].astype(bool)][xg_col].dropna()
        without360 = shots[~shots["has_360"].astype(bool)][xg_col].dropna()
        if len(with360) > 5 and len(without360) > 5:
            stat, pval = stats.mannwhitneyu(with360, without360, alternative="two-sided")
            results.append(
                _record(
                    "H6_360_vs_non360_xg",
                    stat,
                    pval,
                    N,
                    _cohen_d(with360, without360),
                    "cohen_d",
                    {
                        "median_360": round(with360.median(), 4),
                        "median_non360": round(without360.median(), 4),
                    },
                )
            )

    # H7: Score state affects goal rate (leading vs drawing vs trailing)
    ss_col = _col(["score_state", "score_differential"])
    if ss_col and "goal" in shots.columns:
        ss = shots[[ss_col, "goal"]].dropna()
        if "score_state" in ss.columns:
            groups_ss = {k: grp["goal"] for k, grp in ss.groupby("score_state")}
            if len(groups_ss) >= 2:
                vals = list(groups_ss.values())
                combined = pd.concat(vals, ignore_index=True)
                if combined.nunique() > 1:
                    try:
                        stat, pval = stats.kruskal(*vals)
                    except ValueError:
                        stat, pval = np.nan, np.nan
                    if not np.isnan(stat) and not np.isnan(pval):
                        n_groups = len(vals)
                        # Eta-squared approximation
                        eta2 = float((stat - n_groups + 1) / (len(ss) - n_groups))
                        results.append(
                            _record(
                                "H7_score_state_goal_rate",
                                stat,
                                pval,
                                N,
                                eta2,
                                "eta_squared",
                                {
                                    "n_groups": n_groups,
                                    "group_means": {
                                        str(k): round(float(v.mean()), 4)
                                        for k, v in groups_ss.items()
                                    },
                                },
                            )
                        )

    # H8: Nearest defender distance (360) correlates with goal outcome
    def_col = _col(["nearest_defender_distance", "nearest_defender_dist"])
    if def_col and "goal" in shots.columns:
        def_shots = shots[[def_col, "goal"]].copy()
        def_shots[def_col] = pd.to_numeric(def_shots[def_col], errors="coerce")
        def_shots = def_shots.dropna()
        goal_def = def_shots[def_shots["goal"] == 1][def_col]
        no_goal_def = def_shots[def_shots["goal"] == 0][def_col]
        if len(goal_def) > 5 and len(no_goal_def) > 5:
            stat, pval = stats.mannwhitneyu(goal_def, no_goal_def, alternative="greater")
            results.append(
                _record(
                    "H8_defender_distance_vs_goal",
                    stat,
                    pval,
                    N,
                    _cohen_d(goal_def, no_goal_def),
                    "cohen_d",
                    {
                        "median_goal": round(goal_def.median(), 4),
                        "median_no_goal": round(no_goal_def.median(), 4),
                    },
                )
            )

    return results


def main() -> None:
    logger.info("Loading shots.parquet …")
    shots = load_shots()

    logger.info("Loading events.parquet for xG …")
    try:
        events = load_events()
    except FileNotFoundError:
        logger.warning("events.parquet not found — xG join unavailable.")
        events = pd.DataFrame()

    logger.info("Running 8 CxG hypotheses …")
    results = run_hypotheses(shots, events)

    n_rejected = sum(1 for r in results if r.get("reject_H0"))
    logger.info("Hypotheses run: %d — rejected (Bonferroni): %d", len(results), n_rejected)

    save_json(
        {"n_hypotheses": len(results), "n_rejected_bonferroni": n_rejected, "results": results},
        "hypothesis_cxg",
    )
    logger.info("11_hypothesis_cxg.py complete.")


if __name__ == "__main__":
    main()
