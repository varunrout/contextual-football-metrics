"""
analysis/18_interpretability.py
================================
Model interpretability for CxG and CxT models.

Outputs (reports/figures/interpretability/):
  01_logit_coefficients.png   — signed logistic regression coefficients (baseline CxG)
  02_lgbm_feature_importance.png — LightGBM gain-based feature importance (contextual CxG)
  03_ablation_study.png       — leave-one-group-out ablation on lgbm_contextual
  04_direction_check.png      — coefficient sign plausibility heatmap
  05_cxt_feature_importance.png — LightGBM gain importance for CxT model

Also writes:
  reports/interpretability_report.html
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import save_fig
from src.evaluation.interpretability import (
    check_coefficient_directions,
    run_ablation_study,
    build_interpretability_html,
    EXPECTED_SIGNS_CXG,
    EXPECTED_SIGNS_CXT,
)

import joblib

# ── Compatibility wrapper ─────────────────────────────────────────────────────

class _Sklearn2DWrapper:
    """Wraps a model whose predict_proba returns 1D → returns 2D (n,2) for sklearn compat."""
    def __init__(self, model):
        self._model = model
    def predict_proba(self, X):
        p = self._model.predict_proba(X)
        if p.ndim == 1:
            import numpy as _np
            return _np.column_stack([1 - p, p])
        return p
    def predict(self, X):
        return self._model.predict(X)
    def __getattr__(self, name):
        return getattr(self._model, name)

# ── Style ─────────────────────────────────────────────────────────────────────
BARCA_BLUE   = "#004D98"
BARCA_RED    = "#A50044"
NEUTRAL_GREY = "#666666"
ACCENT       = "#F5A623"
FIGURE_DIR   = "interpretability"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F8F8",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = _ROOT / "data" / "features" / "features.parquet"
MODELS_CXG = {
    "baseline_logit":    _ROOT / "models" / "cxg" / "baseline_logit.joblib",
    "lgbm_contextual":   _ROOT / "models" / "cxg" / "lgbm_contextual.joblib",
}
MODELS_CXT = {
    "lgbm_contextual": _ROOT / "models" / "cxt" / "lgbm_contextual.joblib",
}

# Pitch / enrichment constants
_BOX_X_MIN = 88.5; _BOX_Y_MIN = 13.84; _BOX_Y_MAX = 54.16
_CENTRAL_Y_MIN = 27.0; _CENTRAL_Y_MAX = 41.0


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_shots() -> pd.DataFrame:
    f = pd.read_parquet(FEATURES_PATH)
    shots = f[f["event_type"] == "shot"].copy()
    # Enrich
    shots["action_type"] = shots["event_type"]
    shots["in_box"] = (
        (shots["x_location"] >= _BOX_X_MIN)
        & (shots["y_location"] >= _BOX_Y_MIN)
        & (shots["y_location"] <= _BOX_Y_MAX)
    )
    shots["is_central"] = shots["y_location"].between(_CENTRAL_Y_MIN, _CENTRAL_Y_MAX)
    # Attach actual outcome from events.parquet
    events = pd.read_parquet(_ROOT / "data" / "processed" / "events.parquet")
    outcome = events[["internal_id", "shot_outcome"]].rename(columns={"internal_id": "event_id"})
    shots = shots.merge(outcome, on="event_id", how="left")
    shots["goal"] = (shots["shot_outcome"] == "Goal").astype(int)
    return shots


def _load_actions() -> pd.DataFrame:
    """Load all actions scored by CxT (pass, carry, cross, cutback)."""
    f = pd.read_parquet(FEATURES_PATH)
    actions = f[f["event_type"].isin(["pass", "carry", "cross", "cutback"])].copy()
    actions["action_type"] = actions["event_type"]
    actions["in_box"] = (
        (actions["x_location"] >= _BOX_X_MIN)
        & (actions["y_location"] >= _BOX_Y_MIN)
        & (actions["y_location"] <= _BOX_Y_MAX)
    )
    actions["is_central"] = actions["y_location"].between(_CENTRAL_Y_MIN, _CENTRAL_Y_MAX)
    # Merge end_x/end_y
    events = pd.read_parquet(_ROOT / "data" / "processed" / "events.parquet")
    end_loc = events[["internal_id", "end_x", "end_y"]].rename(columns={"internal_id": "event_id"})
    actions = actions.merge(end_loc, on="event_id", how="left")
    return actions


# ── Helper: extract LightGBM native feature importance ────────────────────────

def _lgbm_gain_importance(model) -> pd.DataFrame:
    """Walk model.pipeline to find a LightGBM booster and return gain importance."""
    from sklearn.pipeline import Pipeline
    pipe = model.pipeline if hasattr(model, "pipeline") else model
    est = pipe
    # Unwrap sklearn Pipeline
    while isinstance(est, Pipeline):
        est = est.steps[-1][1]
    # LightGBM native
    if hasattr(est, "booster_"):
        booster = est.booster_
        names = booster.feature_name()
        gain  = booster.feature_importance(importance_type="gain")
    elif hasattr(est, "get_booster"):
        booster = est.get_booster()
        # XGBoost
        scores = booster.get_score(importance_type="gain")
        names  = list(scores.keys())
        gain   = np.array([scores[n] for n in names])
    else:
        raise ValueError(f"Cannot extract importances from {type(est)}")

    # LightGBM can persist generic names (Column_0, Column_1, ...). When that
    # happens, recover semantic names from the fitted preprocessing step.
    if all(str(n).startswith("Column_") for n in names):
        pre = None
        if isinstance(pipe, Pipeline) and "pre" in pipe.named_steps:
            pre = pipe.named_steps["pre"]
        elif hasattr(model, "pipeline") and isinstance(model.pipeline, Pipeline):
            pre = model.pipeline.named_steps.get("pre")
        if pre is not None:
            try:
                transformed = list(pre.get_feature_names_out())
                if len(transformed) == len(names):
                    names = [
                        t.replace("num__", "").replace("bool__", "").replace("cat__", "")
                        for t in transformed
                    ]
            except Exception:
                pass

    df = pd.DataFrame({"feature": names, "gain": gain})
    df = df[df["gain"] > 0].sort_values("gain", ascending=False)
    return df


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _short(name: str) -> str:
    """Replace underscores, title-case, max 28 chars."""
    s = name.replace("_", " ").title()
    return s[:28] + ".." if len(s) > 28 else s


# ── Chart 1 — Logistic regression coefficients ───────────────────────────────

def plot_logit_coefficients(model) -> None:
    from sklearn.pipeline import Pipeline
    pipeline = model.pipeline
    pre  = pipeline.named_steps["pre"]
    clf  = pipeline.named_steps["clf"]

    try:
        feat_names = list(pre.get_feature_names_out())
        # Strip ColumnTransformer prefixes
        feat_names = [
            n.split("__", 1)[-1] if "__" in n else n
            for n in feat_names
        ]
    except Exception:
        feat_names = [f"f{i}" for i in range(len(clf.coef_.ravel()))]

    coefs = clf.coef_.ravel()
    df = pd.DataFrame({"feature": feat_names, "coef": coefs})
    df = df.reindex(df["coef"].abs().sort_values(ascending=False).index).head(20)

    colors = [BARCA_BLUE if c >= 0 else BARCA_RED for c in df["coef"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(df))
    ax.barh(y, df["coef"].values, color=colors, alpha=0.85, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(n) for n in df["feature"]], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=NEUTRAL_GREY, lw=1.0)
    ax.set_xlabel("Logistic regression coefficient (L2-regularised)")
    ax.set_title(
        "Baseline CxG Model — Logistic Regression Coefficients\n"
        "Blue = increases xG | Red = decreases xG",
        pad=10,
    )
    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=BARCA_BLUE, label="Positive (increases xG)"),
        Patch(facecolor=BARCA_RED,  label="Negative (decreases xG)"),
    ], fontsize=9, loc="lower right")

    plt.tight_layout()
    save_fig("01_logit_coefficients", FIGURE_DIR)
    print("  [OK] 01_logit_coefficients.png")


# ── Chart 2 — LightGBM feature importance (CxG contextual) ───────────────────

def plot_lgbm_importance(model, title_suffix: str, fname: str) -> None:
    df = _lgbm_gain_importance(model).head(20)
    df["gain_pct"] = df["gain"] / df["gain"].sum() * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(df))
    bars = ax.barh(y, df["gain_pct"].values, color=BARCA_BLUE, alpha=0.85, zorder=3)
    for bar, v in zip(bars, df["gain_pct"]):
        ax.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", ha="left", fontsize=8, color="#444")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(n) for n in df["feature"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of total gain importance")
    ax.set_title(f"LightGBM Feature Importance (Gain) — {title_suffix}", pad=10)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    save_fig(fname, FIGURE_DIR)
    print(f"  [OK] {fname}.png")


# ── Chart 3 — Ablation study ──────────────────────────────────────────────────

def plot_ablation(model, shots: pd.DataFrame) -> None:
    from sklearn.metrics import log_loss

    feature_groups = {
        "Shot geometry\n(distance, angle, location)": [
            "distance_to_goal", "shot_angle", "x_location", "y_location", "in_box", "is_central",
        ],
        "Shot technique\n(header, volley, foot)": [
            "header", "volley", "body_part", "first_time_shot",
        ],
        "Game context\n(score state, minute, home/away)": [
            "score_state", "score_differential", "minute", "home_or_away",
        ],
        "Sequence context\n(possession build-up)": [
            "sequence_type", "transition_or_settled", "events_before_action",
            "passes_before_action", "carries_before_action", "time_from_possession_start",
            "possession_start_zone", "directness", "vertical_progression_speed",
        ],
        "Opponent quality\n(defensive strength)": [
            "opponent_defensive_rating", "opponent_keeper_shot_stopping_rating",
            "opponent_team_strength", "opponent_xg_conceded_rolling_5",
            "opponent_shots_conceded_rolling_5",
        ],
        "Pressure &\nset-piece flags": [
            "under_pressure", "open_play", "set_piece_type", "set_piece_flag",
            "counterpress_regain_flag",
        ],
    }

    result = run_ablation_study(
        model=_Sklearn2DWrapper(model),
        df=shots,
        target_col="goal",
        task_type="classification",
        feature_groups=feature_groups,
    )

    abl_df = result.to_dataframe()
    if abl_df.empty:
        print("  ⚠  Ablation returned no results — skipping chart 3.")
        return

    # Keep only groups with at least one present feature
    abl_df = abl_df[abl_df["features_removed"] > 0]

    colors = [BARCA_RED if d > 0 else BARCA_BLUE for d in abl_df["degradation"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(abl_df))
    bars = ax.barh(y, abl_df["degradation"].values * 100, color=colors, alpha=0.85, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(abl_df["group"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=NEUTRAL_GREY, lw=1.0)
    for bar, v in zip(bars, abl_df["degradation"].values * 100):
        offset = 0.05 if v >= 0 else -0.1
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color="#444")
    ax.set_xlabel("Log-loss change when feature group is ablated\n(positive = model gets worse, group is important)")
    ax.set_title(
        "Leave-One-Group-Out Ablation — Contextual CxG (LightGBM)\n"
        "Baseline log-loss: {:.4f}".format(result.baseline_metric),
        pad=10,
    )
    plt.tight_layout()
    save_fig("03_ablation_study", FIGURE_DIR)
    print("  [OK] 03_ablation_study.png")


# ── Chart 4 — Coefficient direction plausibility ─────────────────────────────

def plot_direction_check(model, feature_names: list[str]) -> None:
    result = check_coefficient_directions(model, feature_names, EXPECTED_SIGNS_CXG)

    features_checked = list(EXPECTED_SIGNS_CXG.keys())
    violations = {v.feature for v in result.violations}

    fig, ax = plt.subplots(figsize=(8, max(2.5, len(features_checked) * 0.45 + 1.5)))
    y = np.arange(len(features_checked))

    for i, feat in enumerate(features_checked):
        ok = feat not in violations
        color = "#28a745" if ok else BARCA_RED
        label = "PASS" if ok else "FAIL"
        ax.barh(i, 1, color=color, alpha=0.7, zorder=3, height=0.6)
        ax.text(0.5, i, label, va="center", ha="center", fontsize=9,
                color="white", fontweight="bold")
        if not ok:
            v = next(vv for vv in result.violations if vv.feature == feat)
            ax.text(1.05, i,
                    f"coef={v.coefficient:+.3f}  (expected {'+' if v.expected_sign==1 else '-'})",
                    va="center", ha="left", fontsize=8, color=BARCA_RED)

    ax.set_yticks(y)
    ax.set_yticklabels([_short(f) for f in features_checked], fontsize=9)
    ax.set_xticks([])
    ax.set_xlim(0, 2.8)
    ax.set_title(
        f"Coefficient Direction Plausibility — Baseline Logit CxG\n"
        f"Pass rate: {result.pass_rate*100:.0f}%  "
        f"({result.n_checked - result.n_violations}/{result.n_checked} features correct)",
        pad=10,
    )
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#28a745", label="Correct sign (pass)"),
        Patch(facecolor=BARCA_RED, label="Wrong sign (fail)"),
    ], fontsize=9, loc="lower right")

    plt.tight_layout()
    save_fig("04_direction_check", FIGURE_DIR)
    print("  [OK] 04_direction_check.png")


# ── HTML report ───────────────────────────────────────────────────────────────

def write_html_report(direction_check_result, ablation_result) -> None:
    html = build_interpretability_html(
        shap_result=None,
        direction_check=direction_check_result,
        ablation_result=ablation_result,
        output_path=str(_ROOT / "reports" / "interpretability_report.html"),
    )
    print("  [OK] interpretability_report.html")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
                        datefmt="%H:%M:%S")

    print("Loading data...")
    shots   = _load_shots()
    print(f"  shots: {len(shots):,}")
    actions = _load_actions()
    print(f"  CxT actions: {len(actions):,}")

    print("\nLoading models...")
    logit_model = joblib.load(MODELS_CXG["baseline_logit"])
    lgbm_cxg    = joblib.load(MODELS_CXG["lgbm_contextual"])
    lgbm_cxt    = joblib.load(MODELS_CXT["lgbm_contextual"])
    print("  OK")

    print("\nGenerating charts -> reports/figures/interpretability/")
    plot_logit_coefficients(logit_model)
    plot_lgbm_importance(lgbm_cxg, "Contextual CxG", "02_lgbm_cxg_importance")
    plot_ablation(lgbm_cxg, shots)
    dir_result = check_coefficient_directions(
        logit_model.pipeline, logit_model.feature_columns, EXPECTED_SIGNS_CXG
    )
    plot_direction_check(logit_model.pipeline, logit_model.feature_columns)
    plot_lgbm_importance(lgbm_cxt, "Contextual CxT", "05_cxt_feature_importance")

    print("\nBuilding HTML report...")
    abl_result = run_ablation_study(
        model=_Sklearn2DWrapper(lgbm_cxg),
        df=shots,
        target_col="goal",
        task_type="classification",
        feature_groups={
            "Shot geometry": ["distance_to_goal", "shot_angle", "x_location", "y_location"],
            "Shot technique": ["header", "volley", "body_part", "first_time_shot"],
            "Game context": ["score_state", "score_differential", "minute"],
            "Sequence context": ["sequence_type", "events_before_action"],
            "Opponent quality": ["opponent_defensive_rating", "opponent_team_strength"],
        },
    )
    write_html_report(dir_result, abl_result)

    print("\nDone.")


if __name__ == "__main__":
    main()
