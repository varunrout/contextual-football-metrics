"""
analysis/13_hypothesis_cxt.py
==============================
Part 7c — Hypothesis Testing for CxT (shot_in_possession proxy, passes/carries).

6 hypotheses with Bonferroni correction.

Outputs
-------
reports/hypothesis_cxt.json
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

from analysis._utils import derive_shot_in_possession, load_features, save_json  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("13_hypothesis_cxt")

N_HYPOTHESES = 6
_CXT_TYPES = {"pass", "carry", "cross", "cutback"}


def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")


def _record(
    name: str,
    statistic: float,
    pval: float,
    effect_size: float,
    effect_label: str = "cohen_d",
    extra: dict | None = None,
) -> dict:
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


def run_hypotheses(df: pd.DataFrame) -> list[dict]:
    results = []

    # H1: Higher vertical progression speed → higher shot_in_possession rate
    vp_col = _col(df, ["vertical_progression_speed"])
    if vp_col and "shot_in_possession" in df.columns:
        vp = df[[vp_col, "shot_in_possession"]].copy()
        vp[vp_col] = pd.to_numeric(vp[vp_col], errors="coerce")
        vp = vp.dropna()
        if len(vp) > 30:
            r, pval = stats.pearsonr(vp[vp_col], vp["shot_in_possession"])
            results.append(
                _record(
                    "H1_vertical_progression_vs_sip",
                    r,
                    pval,
                    r,
                    effect_label="pearson_r",
                    extra={"n": len(vp)},
                )
            )

    # H2: Transition possessions have higher shot_in_possession rate than settled
    trans_col = _col(df, ["transition_or_settled", "is_transition", "phase_of_play"])
    if trans_col and "shot_in_possession" in df.columns:
        tr = df[[trans_col, "shot_in_possession"]].dropna()
        tr_str = tr[trans_col].astype(str).str.lower()
        transition = tr[tr_str.isin(["transition", "1", "true"])]["shot_in_possession"]
        settled = tr[tr_str.isin(["settled", "0", "false"])]["shot_in_possession"]
        if len(transition) > 5 and len(settled) > 5:
            stat, pval = stats.ttest_ind(transition, settled, equal_var=False)
            results.append(
                _record(
                    "H2_transition_higher_sip",
                    stat,
                    pval,
                    _cohen_d(transition, settled),
                    extra={
                        "mean_transition": round(transition.mean(), 4),
                        "mean_settled": round(settled.mean(), 4),
                    },
                )
            )

    # H3: Directness positively predicts shot_in_possession
    dir_col = _col(df, ["directness"])
    if dir_col and "shot_in_possession" in df.columns:
        d = df[[dir_col, "shot_in_possession"]].copy()
        d[dir_col] = pd.to_numeric(d[dir_col], errors="coerce")
        d = d.dropna()
        if len(d) > 30:
            r, pval = stats.pearsonr(d[dir_col], d["shot_in_possession"])
            results.append(
                _record(
                    "H3_directness_vs_sip",
                    r,
                    pval,
                    r,
                    effect_label="pearson_r",
                    extra={"n": len(d)},
                )
            )

    # H4: Possessions starting in own half have lower shot_in_possession than attacking half
    pstart_col = _col(df, ["possession_start_zone", "possession_start_x"])
    if pstart_col and "shot_in_possession" in df.columns:
        ps = df[[pstart_col, "shot_in_possession"]].dropna()
        if "zone" in pstart_col:
            # Categorical: any zone labelled with "attacking" vs "defensive"
            ps_str = ps[pstart_col].astype(str).str.lower()
            attacking = ps[ps_str.str.contains("attack|final|offensive")]["shot_in_possession"]
            defensive = ps[ps_str.str.contains("def|own|back")]["shot_in_possession"]
        else:
            ps[pstart_col] = pd.to_numeric(ps[pstart_col], errors="coerce")
            ps = ps.dropna()
            attacking = ps[ps[pstart_col] >= 52.5]["shot_in_possession"]
            defensive = ps[ps[pstart_col] < 52.5]["shot_in_possession"]

        if len(attacking) > 5 and len(defensive) > 5:
            stat, pval = stats.ttest_ind(attacking, defensive, equal_var=False)
            results.append(
                _record(
                    "H4_possession_start_zone_sip",
                    stat,
                    pval,
                    _cohen_d(attacking, defensive),
                    extra={
                        "mean_attacking": round(attacking.mean(), 4),
                        "mean_defensive": round(defensive.mean(), 4),
                    },
                )
            )

    # H5: Lower opponent pressing → higher shot_in_possession
    pressing_col = _col(df, ["opponent_pressing_intensity", "opp_press_intensity"])
    if pressing_col and "shot_in_possession" in df.columns:
        press = df[[pressing_col, "shot_in_possession"]].copy()
        press[pressing_col] = pd.to_numeric(press[pressing_col], errors="coerce")
        press = press.dropna()
        if len(press) > 20:
            median_press = press[pressing_col].median()
            high_press = press[press[pressing_col] >= median_press]["shot_in_possession"]
            low_press = press[press[pressing_col] < median_press]["shot_in_possession"]
            stat, pval = stats.ttest_ind(high_press, low_press, equal_var=False)
            results.append(
                _record(
                    "H5_low_pressing_higher_sip",
                    stat,
                    pval,
                    _cohen_d(low_press, high_press),
                    extra={
                        "mean_high_press": round(high_press.mean(), 4),
                        "mean_low_press": round(low_press.mean(), 4),
                    },
                )
            )

    # H6: Possessions with more events have higher shot_in_possession rate
    len_col = _col(df, ["events_in_possession", "events_before_action", "possession_length"])
    if len_col and "shot_in_possession" in df.columns:
        pl = df[[len_col, "shot_in_possession"]].copy()
        pl[len_col] = pd.to_numeric(pl[len_col], errors="coerce")
        pl = pl.dropna()
        if len(pl) > 30:
            r, pval = stats.spearmanr(pl[len_col], pl["shot_in_possession"])
            results.append(
                _record(
                    "H6_possession_length_vs_sip",
                    r,
                    pval,
                    r,
                    effect_label="spearman_rho",
                    extra={"n": len(pl)},
                )
            )

    return results


def main() -> None:
    logger.info("Loading features.parquet …")
    features = load_features()

    # Filter to CxT action types
    type_col = next((c for c in ["event_type", "action_type"] if c in features.columns), None)
    if type_col:
        df = features[features[type_col].astype(str).isin(_CXT_TYPES)].copy()
        logger.info("Filtered to CxT types: %d rows", len(df))
    else:
        df = features.copy()

    logger.info("Deriving shot_in_possession …")
    df = derive_shot_in_possession(df)

    logger.info("Running %d CxT hypotheses …", N_HYPOTHESES)
    results = run_hypotheses(df)

    n_rejected = sum(1 for r in results if r.get("reject_H0"))
    logger.info("Hypotheses run: %d — rejected (Bonferroni): %d", len(results), n_rejected)

    save_json(
        {"n_hypotheses": len(results), "n_rejected_bonferroni": n_rejected, "results": results},
        "hypothesis_cxt",
    )
    logger.info("13_hypothesis_cxt.py complete.")


if __name__ == "__main__":
    main()
