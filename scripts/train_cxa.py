"""
scripts/train_cxa.py
====================
Fit the two-stage CxA pipeline (shot-creation + shot-quality) using
LightGBM contextual models as defaults (configurable).

Logged to MLflow under experiment  cfm/cxa.
Saved to  models/cxa/<run_id>/cxa_pipeline.pkl

The script:
  1. Loads actions.parquet (all creative actions — passes, carries, cutbacks).
  2. Attaches shot_created labels derived from possession-level shot linkage.
  3. Attaches resulting_shot_cxg labels from the CxG-scored shot rows.
  4. Fits ShotCreationModel + ShotQualityModel.
  5. Assembles CxAPipeline and saves it.

Usage
-----
    python scripts/train_cxa.py
    python scripts/train_cxa.py --n-folds 5 --n-estimators 300
    python scripts/train_cxa.py --features data/features/actions.parquet --no-promote
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.models.cxa.cxa_pipeline import CxAPipeline
from src.models.cxa.shot_creation_model import ShotCreationModel
from src.models.cxa.shot_quality_model import ShotQualityModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s \u2014 %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_cxa")

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "cxa"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "cxa"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"


# ── MLflow helpers ────────────────────────────────────────────────────────────

def _get_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        logger.warning("mlflow not installed — skipping experiment tracking.")
        return None


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *_): pass


def _start_run(mlflow, experiment: str, run_name: str):
    if mlflow is None:
        return None
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)


# ── Production pointer ────────────────────────────────────────────────────────

def _update_production_pointer(model_filename: str) -> None:
    with open(MODELS_YAML, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("production", {})["cxa"] = model_filename
    with open(MODELS_YAML, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
    logger.info("Updated production.cxa pointer → %s", model_filename)


# ── CxG inline scoring ────────────────────────────────────────────────────────

def _attach_cxg_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the production CxG model and score shot rows in features_df,
    adding a 'cxg' column.  Returns features_df unchanged if the model
    is not available or shots cannot be identified.
    """
    try:
        with open(MODELS_YAML, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        cxg_path_rel = cfg.get("production", {}).get("cxg")
        if not cxg_path_rel:
            logger.warning("No production CxG model in configs/models.yaml — skipping CxG scoring.")
            return features_df
        cxg_path = PROJECT_ROOT / cxg_path_rel
        if not cxg_path.exists():
            logger.warning("CxG model file not found: %s — skipping CxG scoring.", cxg_path)
            return features_df
        import joblib
        cxg_model = joblib.load(cxg_path)
    except Exception as exc:
        logger.warning("Could not load CxG model (%s) — skipping CxG scoring.", exc)
        return features_df

    _type_col = next(
        (c for c in ("event_type", "action_type") if c in features_df.columns), None
    )
    if _type_col is None:
        logger.warning("No event_type/action_type column in features — cannot identify shots.")
        return features_df

    shot_mask = features_df[_type_col].astype(str) == "shot"
    shot_df = features_df[shot_mask]
    if shot_df.empty:
        logger.warning("No shot rows found in features — skipping CxG scoring.")
        return features_df

    try:
        proba = cxg_model.predict_proba(shot_df)
        scores = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] >= 2) else proba.ravel()
        features_df = features_df.copy()
        features_df.loc[shot_mask, "cxg"] = scores.astype(float)
        logger.info(
            "CxG scores attached to %d shot rows (mean=%.4f, max=%.4f)",
            len(shot_df), float(scores.mean()), float(scores.max()),
        )
    except Exception as exc:
        logger.warning("CxG scoring failed (%s) — quality model will use binary fallback.", exc)

    return features_df


# ── Diagnostic charts ─────────────────────────────────────────────────────────

def _chart_pr_calibration(
    actions: pd.DataFrame,
    heldout: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """PR curve and reliability diagram for the shot-creation classifier."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.calibration import calibration_curve

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, df, label, color in [
        (axes[0], actions, "Train", "steelblue"),
        (axes[1], heldout, "Held-out", "darkorange"),
    ]:
        if "shot_created" not in df.columns or "p_shot_created" not in df.columns:
            ax.set_title(f"PR Curve — {label}\n(no data)")
            continue
        y = df["shot_created"].astype(int).to_numpy()
        p = df["p_shot_created"].to_numpy()
        if len(np.unique(y)) < 2:
            ax.set_title(f"PR Curve — {label}\n(single class)")
            continue
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, color=color, lw=1.5)
        ax.axhline(y.mean(), color="gray", ls="--", lw=1, label=f"Baseline={y.mean():.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {label}\nAP={ap:.4f}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Shot-Creation Model: Precision–Recall Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = figures_dir / "pr_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", out)

    # Calibration / reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, df, label in [
        (axes[0], actions, "Train"),
        (axes[1], heldout, "Held-out"),
    ]:
        if "shot_created" not in df.columns or "p_shot_created" not in df.columns:
            ax.set_title(f"Calibration — {label}\n(no data)")
            continue
        y = df["shot_created"].astype(int).to_numpy()
        p = df["p_shot_created"].to_numpy()
        if len(np.unique(y)) < 2:
            ax.set_title(f"Calibration — {label}")
            continue
        try:
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10)
        except Exception:
            ax.set_title(f"Calibration — {label}")
            continue
        ax.plot(mean_pred, frac_pos, "o-", color="steelblue", lw=1.5, label="Model")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positives")
        ax.set_title(f"Reliability Diagram — {label}")
        ax.legend(fontsize=8)

    fig.suptitle("Shot-Creation Model: Calibration", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = figures_dir / "creation_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", out)


def _chart_cxa_by_action_type(
    heldout_scored: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """CxA score distributions split by action type (pass vs carry)."""
    from scipy.stats import gaussian_kde

    figures_dir.mkdir(parents=True, exist_ok=True)
    if "cxa" not in heldout_scored.columns:
        logger.warning("No cxa column — skipping CxA-by-action-type chart.")
        return

    type_col = next(
        (c for c in ("action_type", "event_type") if c in heldout_scored.columns), None
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"pass": "steelblue", "carry": "darkorange", "cross": "seagreen", "cutback": "crimson"}
    any_plotted = False
    for atype, color in colors.items():
        subset = heldout_scored[
            heldout_scored.get(type_col, pd.Series(dtype=str)) == atype
            if type_col else pd.Series(False, index=heldout_scored.index)
        ]["cxa"].dropna()
        if len(subset) < 20:
            continue
        vals = subset.to_numpy()
        xs = np.linspace(0, vals.max() + 0.01, 300)
        try:
            kde = gaussian_kde(vals, bw_method="scott")
            ax.plot(xs, kde(xs), color=color, lw=2, label=f"{atype} (n={len(vals):,})")
            any_plotted = True
        except Exception:
            pass
    if not any_plotted:
        plt.close(fig)
        return
    ax.set_xlabel("CxA score")
    ax.set_ylabel("Density")
    ax.set_title("CxA Score Distribution by Action Type (Held-out)", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = figures_dir / "cxa_by_action_type.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", out)


def _draw_pitch(ax, *, lw: float = 1.5, color: str = "white", alpha: float = 0.9) -> None:
    """Draw StatsBomb 120×80 pitch markings on *ax* (call AFTER data layers)."""
    kw = dict(color=color, lw=lw, alpha=alpha, zorder=5)
    # Pitch outline
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], **kw)
    # Halfway line
    ax.plot([60, 60], [0, 80], ls="--", **{**kw, "lw": lw * 0.7, "alpha": alpha * 0.6})
    # Centre circle (approx radius 10)
    theta = np.linspace(0, 2 * np.pi, 120)
    ax.plot(60 + 10 * np.cos(theta), 40 + 10 * np.sin(theta), **{**kw, "lw": lw * 0.7, "alpha": alpha * 0.6})
    # Penalty areas
    for x0, w in [(0, 18), (102, 18)]:
        ax.plot([x0, x0 + w, x0 + w, x0, x0], [18, 18, 62, 62, 18], **kw)
    # 6-yard boxes
    for x0, w in [(0, 6), (114, 6)]:
        ax.plot([x0, x0 + w, x0 + w, x0, x0], [30, 30, 50, 50, 30], **kw)
    # Goals
    for x0, w in [(0, 0), (120, 0)]:
        ax.plot([x0, x0 - 2.4 if x0 == 0 else x0 + 2.4], [36, 36], **{**kw, "lw": lw * 2})
        ax.plot([x0, x0 - 2.4 if x0 == 0 else x0 + 2.4], [44, 44], **{**kw, "lw": lw * 2})
        ax.plot(
            [x0 - 2.4 if x0 == 0 else x0 + 2.4, x0 - 2.4 if x0 == 0 else x0 + 2.4],
            [36, 44], **{**kw, "lw": lw * 2},
        )
    # Penalty spots
    for px in [12, 108]:
        ax.plot(px, 40, "o", color=color, ms=3, alpha=alpha, zorder=6)


def _chart_cxa_pitch_heatmap(
    heldout_scored: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Mean CxA by pitch zone for pass, carry, and dribble actions."""
    from mplsoccer import Pitch

    figures_dir.mkdir(parents=True, exist_ok=True)
    x_col = next((c for c in ("x_location", "x", "location_x") if c in heldout_scored.columns), None)
    y_col = next((c for c in ("y_location", "y", "location_y") if c in heldout_scored.columns), None)
    type_col = next((c for c in ("event_type", "action_type") if c in heldout_scored.columns), None)
    if x_col is None or y_col is None or type_col is None or "cxa" not in heldout_scored.columns:
        logger.warning("Missing x/y/type/cxa columns — skipping pitch zone chart.")
        return

    df = heldout_scored[[x_col, y_col, type_col, "cxa"]].dropna().copy()
    if len(df) < 50:
        logger.warning("Too few scored actions (%d) for pitch zone chart.", len(df))
        return

    df[x_col] = df[x_col].astype(float)
    df[y_col] = df[y_col].astype(float)

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        pitch_color="#1a1a2e",
        line_color="white",
        line_zorder=2,
        linewidth=1.8,
    )
    fig, axs = pitch.draw(nrows=1, ncols=3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")

    axes = np.atleast_1d(axs).ravel()
    action_order = ["pass", "carry", "dribble"]
    action_titles = {
        "pass": "Pass",
        "carry": "Carry",
        "dribble": "Dribble",
    }

    panel_frames: dict[str, pd.DataFrame | None] = {}
    valid_values: list[np.ndarray] = []
    for action in action_order:
        action_df = df[df[type_col].astype(str) == action]
        if action_df.empty:
            panel_frames[action] = None
            continue
        panel_frames[action] = action_df
        vals = action_df["cxa"].to_numpy()
        valid = vals[np.isfinite(vals)]
        if valid.size:
            valid_values.append(valid)

    if not valid_values:
        logger.warning("No valid CxA zone values available — skipping pitch zone chart.")
        plt.close(fig)
        return

    vmin = float(min(np.nanmin(vals) for vals in valid_values))
    vmax = float(max(np.nanmax(vals) for vals in valid_values))
    hm = None
    for ax, action in zip(axes, action_order):
        action_df = panel_frames[action]
        if action_df is None:
            ax.text(
                52.5,
                34,
                "No dribble data\navailable",
                ha="center",
                va="center",
                color="white",
                fontsize=13,
                fontweight="bold",
                zorder=3,
            )
            ax.set_title(action_titles[action], color="white", fontsize=12, fontweight="bold", pad=10)
            continue
        hm = pitch.hexbin(
            action_df[x_col],
            action_df[y_col],
            ax=ax,
            C=action_df["cxa"],
            reduce_C_function=np.nanmean,
            gridsize=(24, 16),
            mincnt=1,
            cmap="YlOrRd",
            alpha=0.92,
            edgecolors="none",
            zorder=1,
            vmin=vmin,
            vmax=vmax,
        )
        n_rows = int(len(action_df))
        ax.set_title(
            f"{action_titles[action]}\nmean CxA · n={n_rows:,}",
            color="white",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

    fig.subplots_adjust(right=0.92, wspace=0.12)
    cax = fig.add_axes([0.935, 0.16, 0.012, 0.68])
    cb = fig.colorbar(hm, cax=cax)
    cb.set_label("Mean CxA", fontsize=11, color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        "CxA Best Zones by Action Type\n(Held-out · internal 105x68 · attacking direction left to right)",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.93))
    out = figures_dir / "cxa_pitch_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Chart saved: %s", out)


def _chart_creation_rate_by_zone(
    heldout_scored: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Empirical shot-creation rate vs predicted p_shot_created by distance band."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    dist_col = next(
        (c for c in ("distance_to_goal", "dist_to_goal", "distance") if c in heldout_scored.columns),
        None,
    )
    if dist_col is None or "shot_created" not in heldout_scored.columns or "p_shot_created" not in heldout_scored.columns:
        logger.warning("Missing distance/shot_created columns — skipping creation-rate-by-zone chart.")
        return

    df = heldout_scored[[dist_col, "shot_created", "p_shot_created"]].dropna()
    if len(df) < 100:
        return

    df["dist_band"] = pd.cut(df[dist_col], bins=10)
    grouped = df.groupby("dist_band", observed=True).agg(
        empirical=("shot_created", "mean"),
        predicted=("p_shot_created", "mean"),
        n=("shot_created", "count"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(grouped))
    w = 0.35
    ax.bar(xs - w/2, grouped["empirical"], w, label="Empirical rate", color="steelblue", alpha=0.8)
    ax.bar(xs + w/2, grouped["predicted"], w, label="Predicted p_shot_created", color="darkorange", alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [str(b) for b in grouped.index], rotation=45, ha="right", fontsize=8
    )
    ax.set_xlabel("Distance to goal band")
    ax.set_ylabel("Shot-creation rate")
    ax.set_title("Empirical vs Predicted Shot-Creation Rate by Distance (Held-out)", fontweight="bold")
    ax.legend(fontsize=9)
    # Annotate n
    for i, n in enumerate(grouped["n"]):
        ax.text(xs[i], max(grouped["empirical"].iloc[i], grouped["predicted"].iloc[i]) + 0.005,
                f"n={n}", ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    out = figures_dir / "creation_rate_by_distance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", out)


def _chart_quality_scatter(
    heldout_scored: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Predicted vs actual CxG for shot-creating actions (quality model)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    if "resulting_shot_cxg" not in heldout_scored.columns or "expected_cxg_if_shot" not in heldout_scored.columns:
        logger.warning("Missing quality columns — skipping quality scatter chart.")
        return

    df = heldout_scored[
        ["resulting_shot_cxg", "expected_cxg_if_shot"]
    ].dropna()
    # Only rows where a shot actually followed (resulting_shot_cxg > 0)
    df = df[df["resulting_shot_cxg"] > 0]
    if len(df) < 20:
        logger.warning("Too few shot-creating rows (%d) for quality scatter.", len(df))
        return

    from scipy.stats import spearmanr
    spearman_r, _ = spearmanr(df["resulting_shot_cxg"], df["expected_cxg_if_shot"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter
    ax = axes[0]
    ax.scatter(df["resulting_shot_cxg"], df["expected_cxg_if_shot"],
               alpha=0.3, s=10, color="steelblue", rasterized=True)
    lim = max(df["resulting_shot_cxg"].max(), df["expected_cxg_if_shot"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("Actual resulting CxG")
    ax.set_ylabel("Predicted CxG (quality model)")
    ax.set_title(f"Quality Model: Actual vs Predicted\nSpearman r={spearman_r:.3f}")

    # Binned calibration
    ax = axes[1]
    df["actual_bin"] = pd.cut(df["resulting_shot_cxg"], bins=8)
    binned = df.groupby("actual_bin", observed=True).agg(
        mean_actual=("resulting_shot_cxg", "mean"),
        mean_pred=("expected_cxg_if_shot", "mean"),
        n=("resulting_shot_cxg", "count"),
    ).dropna()
    ax.plot(binned["mean_actual"], binned["mean_pred"], "o-", color="steelblue", lw=1.5)
    lim2 = max(binned["mean_actual"].max(), binned["mean_pred"].max()) * 1.1
    ax.plot([0, lim2], [0, lim2], "k--", lw=1, label="Perfect calibration")
    for _, row in binned.iterrows():
        ax.text(row["mean_actual"], row["mean_pred"] + 0.002,
                f"n={int(row['n'])}", ha="center", fontsize=7, color="gray")
    ax.set_xlabel("Mean actual CxG (bin)")
    ax.set_ylabel("Mean predicted CxG")
    ax.set_title("Quality Model: Binned Calibration")
    ax.legend(fontsize=8)

    fig.suptitle("Shot-Quality Model (CxA Stage 2)\n(Held-out, shot-creating actions only)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = figures_dir / "quality_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", out)


# ── Label linkage ─────────────────────────────────────────────────────────────

def _attach_labels(
    actions_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach shot_created and resulting_shot_cxg labels to the actions DataFrame.

    shot_created = 1 if a shot occurred in the same possession.
    resulting_shot_cxg = mean CxG of shots in that possession (used by quality model).

    If 'shot_created' / 'resulting_shot_cxg' already present, nothing is added.
    """
    actions = actions_df.copy()

    # If already present, skip
    if "shot_created" in actions.columns and "resulting_shot_cxg" in actions.columns:
        return actions

    # Identify shot rows in the full feature store
    # Probe both column names — feature store uses event_type, raw events use action_type
    _type_col = next(
        (c for c in ("event_type", "action_type") if c in features_df.columns),
        None,
    )
    if _type_col:
        shot_mask = features_df[_type_col].astype(str) == "shot"
    else:
        shot_mask = pd.Series(False, index=features_df.index)
    shots = features_df[shot_mask].copy()
    logger.info("Label linkage: found %d shot rows via column '%s'", len(shots), _type_col)

    # Prioritise globally-unique possession IDs; never use match_id alone
    # (every match contains shots, so match_id would label 100% as shot_created)
    _poss_candidates = ["possession_internal_id", "possession_id"]
    poss_col = next(
        (c for c in _poss_candidates if c in shots.columns and c in actions.columns),
        None,
    )

    if poss_col is None:
        logger.warning(
            "Cannot link shots to actions — no possession column found in both datasets. "
            "Falling back to zero labels."
        )
        actions["shot_created"] = 0
        actions["resulting_shot_cxg"] = 0.0
        return actions

    logger.info("Label linkage using possession column: '%s'", poss_col)

    # shot_created: possession has ≥1 shot
    poss_with_shots = set(shots[poss_col].dropna().unique())
    actions["shot_created"] = actions[poss_col].apply(
        lambda p: 1 if p in poss_with_shots else 0
    )

    # resulting_shot_cxg: mean CxG of shots in the possession (if cxg col present)
    cxg_col = next((c for c in ("cxg", "event_cxg", "predicted_cxg") if c in shots.columns), None)
    if cxg_col:
        poss_mean_cxg = shots.groupby(poss_col)[cxg_col].mean().to_dict()
        actions["resulting_shot_cxg"] = actions[poss_col].map(poss_mean_cxg).fillna(0.0)
    else:
        # No CxG column yet — use indicator (binary proxy)
        actions["resulting_shot_cxg"] = actions["shot_created"].astype(float)

    logger.info(
        "Label linkage: shot_created=%.2f%% mean / resulting_cxg_nonzero=%d rows",
        100 * actions["shot_created"].mean(),
        (actions["resulting_shot_cxg"] > 0).sum(),
    )
    return actions


# ── Main ──────────────────────────────────────────────────────────────────────

def _eval_creation_heldout(
    creation_model,
    heldout: pd.DataFrame,
) -> dict:
    """Return held-out classification metrics for a creation model."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    )
    p = creation_model.predict_proba(heldout)
    y = heldout["shot_created"].astype(int).to_numpy()
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    ap  = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None
    ll  = float(log_loss(y, p, labels=[0, 1]))
    br  = float(brier_score_loss(y, p))
    return {"auc": auc, "average_precision": ap, "log_loss": ll, "brier": br}


def _eval_quality_heldout(
    quality_model,
    heldout: pd.DataFrame,
    quality_col: str = "resulting_shot_cxg",
) -> dict:
    """Return held-out regression metrics for a quality model."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import spearmanr
    df = heldout[heldout.get(quality_col, pd.Series(0.0, index=heldout.index)) > 0]
    if len(df) < 10:
        return {}
    y = df[quality_col].astype(float).to_numpy()
    p = quality_model.predict(df)
    r, _ = spearmanr(y, p)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "spearman": float(r),
        "n_rows": int(len(df)),
    }


def _chart_leaderboard(
    ladder_results: list[dict],
    figures_dir: Path,
) -> None:
    """Grouped bar chart comparing all CxA model candidates by creation AUC and AP."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not ladder_results:
        return

    names  = [r["name"] for r in ladder_results]
    # Prefer held-out metrics; fall back to train-set metrics when the
    # held-out columns are absent (no Euro 2024 split available).
    def _val(r: dict, primary: str, fallback: str) -> float:
        v = r.get(primary)
        return float(v) if v is not None else float(r.get(fallback) or 0.0)
    aucs = [_val(r, "creation_auc",   "train_creation_auc")   for r in ladder_results]
    aps  = [_val(r, "creation_ap",    "train_creation_ap")    for r in ladder_results]
    lls  = [_val(r, "creation_ll",    "train_creation_ll")    for r in ladder_results]
    using_train = all(r.get("creation_auc") is None for r in ladder_results)

    palette = plt.cm.Set2(np.linspace(0, 1, len(names)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, vals, ylabel, title, fmt in [
        (axes[0], aucs, "ROC AUC",        "Creation AUC (↑)",         ".4f"),
        (axes[1], aps,  "Avg Precision",  "Creation AP (↑)",          ".4f"),
        (axes[2], lls,  "Log-loss",       "Creation Log-loss (↓)",    ".4f"),
    ]:
        bars = ax.bar(names, vals, color=palette, edgecolor="white", linewidth=0.6)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=8,
            )
        ax.set_ylim(0, max(vals) * 1.12 if max(vals) > 0 else 1)

    # Star best models
    best_auc = names[int(np.argmax(aucs))]
    best_ll  = names[int(np.argmin(lls))]
    axes[0].set_title(f"Creation AUC (↑)\n★ best: {best_auc}", fontsize=10, fontweight="bold")
    axes[2].set_title(f"Creation Log-loss (↓)\n★ best: {best_ll}", fontsize=10, fontweight="bold")

    fig.suptitle(
        "CxA Model Ladder — "
        + ("Train-set Creation Metrics" if using_train
           else "Held-out Creation Metrics"),
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_dir / "cxa_leaderboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", out)


# ── Main ──────────────────────────────────────────────────────────────────────

# Model ladder: (name, creation_family, quality_family)
_LADDER: list[tuple[str, str, str]] = [
    ("logistic_contextual",  "logistic", "glm"),
    ("xgb_contextual",       "xgboost",  "xgboost"),
    ("lgbm_contextual",      "lgbm",     "lgbm"),
]

def train_cxa(
    actions_path: Path,
    features_path: Path,
    feature_set: str = "contextual",
    n_folds: int = 5,
    n_estimators: int = 300,
    promote: bool = True,
    random_state: int = 42,
    include_neural: bool = False,
    frames_path: Path | None = None,
) -> None:
    if not actions_path.exists():
        logger.error("actions.parquet not found at %s — run build_features.py first.", actions_path)
        sys.exit(1)

    actions_df = pd.read_parquet(actions_path)
    logger.info("Loaded actions: %d rows × %d columns", len(actions_df), len(actions_df.columns))

    # Load full features for label linkage
    features_df = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
    if features_df.empty:
        logger.warning("features.parquet not found — will use actions_df for label linkage.")
        features_df = actions_df

    # ── Attach CxG scores to features so quality model uses real CxG ──────────
    features_df = _attach_cxg_scores(features_df)

    # ── Exclude val_test (Euro 2024) actions from training ────────────────────
    matches_path = PROJECT_ROOT / "data" / "processed" / "matches.parquet"
    if matches_path.exists() and "match_internal_id" in actions_df.columns:
        _matches = pd.read_parquet(matches_path)[["internal_id", "split_role"]].rename(
            columns={"internal_id": "match_internal_id"}
        )
        actions_df = actions_df.merge(_matches, on="match_internal_id", how="left")
        n_before = len(actions_df)
        # val_test = Euro 2024 (held-out for evaluation).
        # test = La Liga (reserved for downstream scoring only — NOT used for
        # training or evaluation).
        heldout_mask = actions_df["split_role"] == "val_test"
        scoring_mask = actions_df["split_role"] == "test"
        train_mask = ~heldout_mask & ~scoring_mask
        train_actions = actions_df[train_mask].copy()
        heldout_actions = actions_df[heldout_mask].copy()
        n_scoring = int(scoring_mask.sum())
        logger.info(
            "Split: %d total → %d train, %d held-out (val_test/Euro 2024), "
            "%d reserved for scoring only (test/La Liga, excluded).",
            n_before, len(train_actions), len(heldout_actions), n_scoring,
        )

        # Fallback: if every match shares the same split_role (i.e. no real
        # train/holdout partition exists in matches.parquet), construct a
        # deterministic 80/20 match-level split so we still produce honest
        # generalization metrics instead of training on everything.
        if train_actions.empty and not heldout_actions.empty:
            import hashlib
            all_match_ids = sorted(actions_df["match_internal_id"].dropna().unique().tolist())
            def _hash01(mid: str) -> float:
                h = hashlib.md5(str(mid).encode("utf-8")).hexdigest()
                return int(h[:8], 16) / 0xFFFFFFFF
            heldout_ids = {mid for mid in all_match_ids if _hash01(mid) < 0.20}
            train_ids = [mid for mid in all_match_ids if mid not in heldout_ids]
            train_actions = actions_df[actions_df["match_internal_id"].isin(train_ids)].copy()
            heldout_actions = actions_df[actions_df["match_internal_id"].isin(heldout_ids)].copy()
            logger.warning(
                "No real train/holdout partition in matches.parquet. "
                "Falling back to deterministic 80/20 match-level split: "
                "%d train matches (%d actions) / %d held-out matches (%d actions).",
                len(train_ids), len(train_actions),
                len(heldout_ids), len(heldout_actions),
            )
    else:
        logger.warning(
            "Could not filter val_test split — matches.parquet missing or "
            "match_internal_id not in actions. Training on all %d actions.",
            len(actions_df),
        )
        train_actions = actions_df.copy()
        heldout_actions = pd.DataFrame()

    train_actions = _attach_labels(train_actions, features_df)
    if not heldout_actions.empty:
        heldout_actions = _attach_labels(heldout_actions, features_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow = _get_mlflow()
    ladder_records: list[dict] = []
    best_pipeline = None
    best_name = ""
    best_auc = -1.0

    # ── Model ladder ──────────────────────────────────────────────────────────
    ladder = list(_LADDER)
    if include_neural:
        ladder.append(("gnn_passing_360", "gnn", "lgbm"))
    for name, c_family, q_family in ladder:
        logger.info("── Training %s (creation=%s, quality=%s) ──", name, c_family, q_family)
        with (_start_run(mlflow, "cfm/cxa", f"cxa_{name}") or _NullContext()):
            try:
                if c_family == "gnn":
                    from src.models.cxa.gnn_passing_network import GNNPassingNetworkCxAModel
                    creation_model = GNNPassingNetworkCxAModel(
                        feature_set=feature_set,
                        frames_path=str(frames_path) if frames_path else None,
                        random_state=random_state,
                    )
                else:
                    creation_model = ShotCreationModel(
                        family=c_family,
                        feature_set=feature_set,
                        n_estimators=n_estimators,
                        random_state=random_state,
                    )
                quality_model = ShotQualityModel(
                    family=q_family,
                    feature_set=feature_set,
                    n_estimators=n_estimators,
                    random_state=random_state,
                )
                pipeline = CxAPipeline(creation_model=creation_model, quality_model=quality_model)
                pipeline.fit(train_actions, creation_target="shot_created", quality_target="resulting_shot_cxg")

                # Save individual stage models
                creation_path = MODELS_DIR / f"shot_creation_{name}.pkl"
                quality_path  = MODELS_DIR / f"shot_quality_{name}.pkl"
                pipeline_path = MODELS_DIR / f"cxa_{name}.pkl"
                pipeline.creation_model.save(creation_path)
                pipeline.quality_model.save(quality_path)
                # Neural creation models embed locally-defined nn.Modules that
                # can't be pickled as part of a CxAPipeline. Skip the combined
                # pickle in that case — score script reconstructs from the two
                # stage files instead.
                if c_family == "gnn":
                    logger.info(
                        "Skipping combined pipeline pickle for %s (neural creation "
                        "model — score via scripts/score_gnn_cxa.py).", name,
                    )
                else:
                    with open(pipeline_path, "wb") as fh:
                        pickle.dump(pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("Saved: %s", pipeline_path.name)

                record: dict = {
                    "name": name,
                    "creation_family": c_family,
                    "quality_family": q_family,
                    "pipeline_path": str(pipeline_path.relative_to(PROJECT_ROOT)),
                }

                # Always compute metrics on the training set so the leaderboard
                # is populated even when there is no held-out (Euro 2024) split.
                try:
                    train_c = _eval_creation_heldout(pipeline.creation_model, train_actions)
                    train_q = _eval_quality_heldout(pipeline.quality_model, train_actions)
                    record.update({
                        "train_creation_auc":   train_c.get("auc"),
                        "train_creation_ap":    train_c.get("average_precision"),
                        "train_creation_ll":    train_c.get("log_loss"),
                        "train_creation_brier": train_c.get("brier"),
                        "train_quality_mae":      train_q.get("mae"),
                        "train_quality_rmse":     train_q.get("rmse"),
                        "train_quality_spearman": train_q.get("spearman"),
                        "train_quality_n_rows":   train_q.get("n_rows"),
                    })
                    logger.info(
                        "  %s [train]  AUC=%.4f  AP=%.4f  ll=%.5f | MAE=%.4f  Spearman=%.4f",
                        name,
                        train_c.get("auc") or 0,
                        train_c.get("average_precision") or 0,
                        train_c.get("log_loss") or 0,
                        train_q.get("mae") or 0,
                        train_q.get("spearman") or 0,
                    )
                except Exception as exc:
                    logger.warning("Train-set metrics failed for %s: %s", name, exc)

                # Held-out evaluation
                if not heldout_actions.empty:
                    c_metrics = _eval_creation_heldout(pipeline.creation_model, heldout_actions)
                    q_metrics = _eval_quality_heldout(pipeline.quality_model, heldout_actions)
                    record.update({
                        "creation_auc": c_metrics.get("auc"),
                        "creation_ap":  c_metrics.get("average_precision"),
                        "creation_ll":  c_metrics.get("log_loss"),
                        "creation_brier": c_metrics.get("brier"),
                        "quality_mae":    q_metrics.get("mae"),
                        "quality_rmse":   q_metrics.get("rmse"),
                        "quality_spearman": q_metrics.get("spearman"),
                    })
                    logger.info(
                        "  %s  AUC=%.4f  AP=%.4f  ll=%.5f | MAE=%.4f  Spearman=%.4f",
                        name,
                        c_metrics.get("auc") or 0,
                        c_metrics.get("average_precision") or 0,
                        c_metrics.get("log_loss") or 0,
                        q_metrics.get("mae") or 0,
                        q_metrics.get("spearman") or 0,
                    )
                    # Track best by creation AUC
                    auc_val = c_metrics.get("auc") or 0.0
                    if auc_val > best_auc:
                        best_auc = auc_val
                        best_pipeline = pipeline
                        best_name = name
                else:
                    # No held-out set → fall back to train-AUC ranking.
                    auc_val = float(record.get("train_creation_auc") or 0.0)
                    if auc_val > best_auc:
                        best_auc = auc_val
                        best_pipeline = pipeline
                        best_name = name

                ladder_records.append(record)

                # Fallback: when there is no held-out set we still want a
                # ``best_pipeline`` to drive charts and the production pointer.
                # Use the last-trained pipeline as the best in that case.
                if heldout_actions.empty and best_pipeline is None:
                    best_pipeline = pipeline
                    best_name = name

            except Exception as exc:
                logger.error("Training %s failed: %s", name, exc, exc_info=True)
                continue

    if best_pipeline is None:
        logger.error("All models failed — nothing to promote.")
        sys.exit(1)

    logger.info("Best model: %s (creation AUC=%.4f)", best_name, best_auc)

    # ── Charts for best model ─────────────────────────────────────────────────
    logger.info("Generating diagnostic charts for best model: %s …", best_name)

    heldout_scored: pd.DataFrame = pd.DataFrame()
    # Use the held-out set for charts when available; otherwise fall back to
    # the training set so the pitch / action-type charts still render.
    chart_actions = heldout_actions if not heldout_actions.empty else train_actions
    if not chart_actions.empty:
        try:
            chart_actions = chart_actions.copy()
            chart_actions["p_shot_created"] = best_pipeline.creation_model.predict_proba(chart_actions)
            # Mirror the scored chart frame back into heldout_actions so
            # downstream chart helpers receive p_shot_created for the
            # held-out side. Without this assignment the held-out panels
            # render as "no data" because p_shot_created lives only on
            # the local copy.
            if not heldout_actions.empty:
                heldout_actions = chart_actions
            train_actions = train_actions.copy()
            if "p_shot_created" not in train_actions.columns:
                train_actions["p_shot_created"] = best_pipeline.creation_model.predict_proba(train_actions)

            heldout_scored = best_pipeline.score(chart_actions)

            quality_col = "resulting_shot_cxg"
            if not heldout_scored.empty and quality_col in chart_actions.columns:
                try:
                    heldout_scored = heldout_scored.merge(
                        chart_actions[["possession_internal_id", quality_col]].drop_duplicates(),
                        on="possession_internal_id", how="left",
                    )
                except Exception:
                    pass
            # Merge event_type for action-type chart
            if not heldout_scored.empty and "event_type" in chart_actions.columns:
                _mc = next((c for c in ("event_id", "possession_internal_id")
                            if c in chart_actions.columns and c in heldout_scored.columns), None)
                if _mc:
                    try:
                        heldout_scored = heldout_scored.merge(
                            chart_actions[[_mc, "event_type"]].drop_duplicates(_mc),
                            on=_mc, how="left",
                        )
                    except Exception:
                        pass
            # Keep the held-out variable name for downstream charts that
            # accept either (chart_pr_calibration treats empty as train-only).
            if heldout_actions.empty:
                heldout_actions = chart_actions
        except Exception as exc:
            logger.warning("Chart data prep failed: %s", exc)

    _chart_leaderboard(ladder_records, FIGURES_DIR)
    _chart_pr_calibration(train_actions, heldout_actions, FIGURES_DIR)
    _chart_cxa_by_action_type(heldout_scored, FIGURES_DIR)
    _chart_cxa_pitch_heatmap(heldout_scored, FIGURES_DIR)
    _chart_creation_rate_by_zone(
        heldout_actions if not heldout_actions.empty else train_actions,
        FIGURES_DIR,
    )
    _chart_quality_scatter(heldout_scored, FIGURES_DIR)

    # ── Save training summary JSON ────────────────────────────────────────────
    summary = {
        "feature_set": feature_set,
        "n_estimators": n_estimators,
        "n_train_actions": int(len(train_actions)),
        "train_shot_created_rate": float(train_actions["shot_created"].mean()),
        "best_model": best_name,
        "best_creation_auc": best_auc,
        "ladder": ladder_records,
    }
    summary_path = REPORTS_DIR / "cxa_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Training summary saved to %s", summary_path)

    if promote:
        best_path = MODELS_DIR / f"cxa_{best_name}.pkl"
        if not best_path.exists():
            logger.warning(
                "Best model %s has no combined pickle (likely a neural model) "
                "\u2014 skipping production promotion.", best_name,
            )
        else:
            _update_production_pointer(str(best_path.relative_to(PROJECT_ROOT)))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the two-stage CxA pipeline.")
    p.add_argument(
        "--features",
        default=str(FEATURES_DIR / "actions.parquet"),
        help="Creative actions parquet (default: data/features/actions.parquet).",
    )
    p.add_argument(
        "--full-features",
        default=str(FEATURES_DIR / "features.parquet"),
        help="Full feature store parquet for label linkage (default: data/features/features.parquet).",
    )
    p.add_argument(
        "--feature-set",
        default="contextual",
        choices=["traditional", "contextual", "full_360"],
        help="Feature set name (default: contextual).",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--no-promote", action="store_true")
    p.add_argument("--include-neural", action="store_true",
                   help="Also train the GNN passing-network creation model. "
                        "Requires PyTorch and freeze frames.")
    p.add_argument("--frames", type=Path, default=None,
                   help="Path to frames parquet for the neural model "
                        "(default: data/processed/frames.parquet).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train_cxa(
        actions_path=Path(args.features),
        features_path=Path(args.full_features),
        feature_set=args.feature_set,
        n_folds=args.n_folds,
        n_estimators=args.n_estimators,
        promote=not args.no_promote,
        random_state=args.seed,
        include_neural=args.include_neural,
        frames_path=args.frames,
    )
