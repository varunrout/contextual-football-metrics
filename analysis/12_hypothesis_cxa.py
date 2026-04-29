"""
analysis/12_hypothesis_cxa.py
==============================
Part 7b — Hypothesis Testing for CxA (shot_created target, actions.parquet).

7 hypotheses with Bonferroni correction.

Outputs
-------
reports/hypothesis_cxa.json
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

from analysis._utils import derive_shot_created, load_actions, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("12_hypothesis_cxa")

N_HYPOTHESES = 7


def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")


def _record(name: str, statistic: float, pval: float, effect_size: float,
            effect_label: str = "cohen_d", extra: dict | None = None) -> dict:
    bonferroni_pval = min(pval * N_HYPOTHESES, 1.0)
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


def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def run_hypotheses(actions: pd.DataFrame) -> list[dict]:
    results = []

    # H1: Progressive actions (positive vertical progression) have higher shot_created rate
    vp_col = _col(actions, ["vertical_progression_speed", "progressive_distance"])
    if vp_col and "shot_created" in actions.columns:
        vp = actions[[vp_col, "shot_created"]].copy()
        vp[vp_col] = pd.to_numeric(vp[vp_col], errors="coerce")
        vp = vp.dropna()
        progressive = vp[vp[vp_col] > 0]["shot_created"]
        not_progressive = vp[vp[vp_col] <= 0]["shot_created"]
        if len(progressive) > 5 and len(not_progressive) > 5:
            stat, pval = stats.mannwhitneyu(progressive, not_progressive, alternative="greater")
            results.append(_record("H1_progressive_actions_shot_created", stat, pval,
                                    _cohen_d(progressive, not_progressive),
                                    extra={"mean_progressive": round(progressive.mean(), 4),
                                           "mean_not_progressive": round(not_progressive.mean(), 4)}))

    # H2: Actions in the final third have higher shot_created rate
    x_col = _col(actions, ["x_location", "end_x", "x"])
    if x_col and "shot_created" in actions.columns:
        ax = actions[[x_col, "shot_created"]].copy()
        ax[x_col] = pd.to_numeric(ax[x_col], errors="coerce")
        ax = ax.dropna()
        final_third = ax[ax[x_col] >= 70]["shot_created"]
        not_final = ax[ax[x_col] < 70]["shot_created"]
        if len(final_third) > 5 and len(not_final) > 5:
            stat, pval = stats.ttest_ind(final_third, not_final, equal_var=False)
            results.append(_record("H2_final_third_shot_created", stat, pval,
                                    _cohen_d(final_third, not_final),
                                    extra={"mean_final_third": round(final_third.mean(), 4),
                                           "mean_other": round(not_final.mean(), 4)}))

    # H3: Carries have different shot_created rate than passes
    etype_col = _col(actions, ["event_type", "action_type"])
    if etype_col and "shot_created" in actions.columns:
        et = actions[[etype_col, "shot_created"]].dropna()
        et_str = et[etype_col].astype(str).str.lower()
        carries = et[et_str == "carry"]["shot_created"]
        passes = et[et_str == "pass"]["shot_created"]
        if len(carries) > 5 and len(passes) > 5:
            stat, pval = stats.ttest_ind(carries, passes, equal_var=False)
            results.append(_record("H3_carry_vs_pass_shot_created", stat, pval,
                                    _cohen_d(carries, passes),
                                    extra={"mean_carry": round(carries.mean(), 4),
                                           "mean_pass": round(passes.mean(), 4)}))

    # H4: Actions in transition sequences have higher shot_created rate
    trans_col = _col(actions, ["transition_or_settled", "is_transition", "phase_of_play"])
    if trans_col and "shot_created" in actions.columns:
        tr = actions[[trans_col, "shot_created"]].dropna()
        tr_str = tr[trans_col].astype(str).str.lower()
        transition = tr[tr_str.isin(["transition", "1", "true"])]["shot_created"]
        settled = tr[tr_str.isin(["settled", "0", "false"])]["shot_created"]
        if len(transition) > 5 and len(settled) > 5:
            stat, pval = stats.ttest_ind(transition, settled, equal_var=False)
            results.append(_record("H4_transition_vs_settled_shot_created", stat, pval,
                                    _cohen_d(transition, settled),
                                    extra={"mean_transition": round(transition.mean(), 4),
                                           "mean_settled": round(settled.mean(), 4)}))

    # H5: High opponent pressing intensity reduces shot_created rate
    pressing_col = _col(actions, ["opponent_pressing_intensity", "opp_press_intensity", "pressing_intensity"])
    if pressing_col and "shot_created" in actions.columns:
        press = actions[[pressing_col, "shot_created"]].copy()
        press[pressing_col] = pd.to_numeric(press[pressing_col], errors="coerce")
        press = press.dropna()
        if len(press) > 20:
            median_press = press[pressing_col].median()
            high_press = press[press[pressing_col] >= median_press]["shot_created"]
            low_press = press[press[pressing_col] < median_press]["shot_created"]
            stat, pval = stats.ttest_ind(high_press, low_press, equal_var=False)
            results.append(_record("H5_high_pressing_reduces_shot_created", stat, pval,
                                    _cohen_d(high_press, low_press),
                                    extra={"mean_high_press": round(high_press.mean(), 4),
                                           "mean_low_press": round(low_press.mean(), 4)}))

    # H6: Score state affects shot creation rate (leading teams defend more)
    ss_col = _col(actions, ["score_state", "score_differential"])
    if ss_col and "shot_created" in actions.columns:
        ss = actions[[ss_col, "shot_created"]].dropna()
        if "score_state" == ss_col:
            groups = {k: grp["shot_created"] for k, grp in ss.groupby("score_state")}
        else:
            # Bin score differential
            ss["score_bin"] = pd.cut(
                pd.to_numeric(ss[ss_col], errors="coerce"),
                bins=[-10, -1, 0, 1, 10],
                labels=["trailing", "drawing_neg", "drawing_pos", "leading"],
            )
            groups = {k: grp["shot_created"] for k, grp in ss.groupby("score_bin") if not grp.empty}

        if len(groups) >= 2:
            vals = [v for v in groups.values() if len(v) > 5]
            if len(vals) >= 2:
                combined = pd.concat(vals, ignore_index=True)
                if combined.nunique() > 1:
                    try:
                        stat, pval = stats.kruskal(*vals)
                    except ValueError:
                        stat, pval = np.nan, np.nan
                    if not np.isnan(stat) and not np.isnan(pval):
                        eta2 = float((stat - len(vals) + 1) / (len(ss) - len(vals)))
                        results.append(_record("H6_score_state_shot_created", stat, pval, eta2,
                                                effect_label="eta_squared",
                                                extra={"group_means": {str(k): round(float(v.mean()), 4)
                                                                         for k, v in groups.items()}}))

    # H7: Directness positively correlates with shot_created
    dir_col = _col(actions, ["directness"])
    if dir_col and "shot_created" in actions.columns:
        d = actions[[dir_col, "shot_created"]].copy()
        d[dir_col] = pd.to_numeric(d[dir_col], errors="coerce")
        d = d.dropna()
        if len(d) > 30:
            r, pval = stats.pearsonr(d[dir_col], d["shot_created"])
            results.append(_record("H7_directness_shot_created_correlation", r, pval, r,
                                    effect_label="pearson_r",
                                    extra={"n": len(d)}))

    return results


def main() -> None:
    logger.info("Loading actions.parquet …")
    actions = load_actions()

    logger.info("Deriving shot_created …")
    actions = derive_shot_created(actions)

    logger.info("Running %d CxA hypotheses …", N_HYPOTHESES)
    results = run_hypotheses(actions)

    n_rejected = sum(1 for r in results if r.get("reject_H0"))
    logger.info("Hypotheses run: %d — rejected (Bonferroni): %d", len(results), n_rejected)

    save_json({"n_hypotheses": len(results), "n_rejected_bonferroni": n_rejected, "results": results},
              "hypothesis_cxa")
    logger.info("12_hypothesis_cxa.py complete.")


if __name__ == "__main__":
    main()
