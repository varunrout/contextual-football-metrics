"""
scripts/train_cxg.py
====================
Train all CxG model candidates via CxGLadder, rank by CV log-loss,
and promote the best model to the production pointer in configs/models.yaml.

Outputs
-------
  models/cxg/<name>.joblib          — every candidate model (joblib)
  reports/cxg_training_summary.json — CV metrics for all candidates + held-out eval
  reports/figures/cxg/
    leaderboard.png                 — CV log-loss / AUC bar chart
    roc_curves.png                  — ROC curves on held-out Euro 2024 set
    calibration.png                 — reliability diagram on held-out set
    binned_empirical_distance.png   — empirical vs predicted goal rate by distance bin
    pitch_heatmap.png               — 2-D goal probability surface (GLM vs XGB)
    residuals.png                   — residual diagnostics vs distance and angle
    coef_forest.png                 — glm_contextual coefficient forest plot
    calibration_by_shot_type.png    — calibration split by shot_type

Usage
-----
    python scripts/train_cxg.py
    python scripts/train_cxg.py --n-folds 5 --n-optuna-trials 30
    python scripts/train_cxg.py --include-360 --shots data/features/shots.parquet
    python scripts/train_cxg.py --no-promote   # skip writing production pointer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc as sk_auc, brier_score_loss, log_loss, roc_auc_score, roc_curve

from src.models.cxg.ladder import CxGLadder, LadderResult
from src.models.neural import is_neural_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_cxg")

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "cxg"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_YAML = CONFIGS_DIR / "models.yaml"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "cxg"


# ── MLflow helper (optional dep) ──────────────────────────────────────────────

def _get_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        logger.warning("mlflow not installed — skipping experiment tracking.")
        return None


def _start_run(mlflow, experiment: str, run_name: str):
    if mlflow is None:
        return None
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)


# ── Promote to production pointer ─────────────────────────────────────────────

def _update_production_pointer(model_filename: str) -> None:
    with open(MODELS_YAML, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("production", {})["cxg"] = model_filename
    with open(MODELS_YAML, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
    logger.info("Updated production.cxg pointer → %s", model_filename)


# ── Save all candidate models as joblib ───────────────────────────────────────

def _save_all_models(results: list[LadderResult], models_dir: Path) -> dict[str, str]:
    """Save every candidate model as <name>.joblib. Returns {name: path_str}.

    Models that expose a custom ``.save(path)`` (e.g. neural models with
    locally-scoped ``nn.Module`` subclasses that vanilla pickle can't handle)
    use that path; everything else falls back to ``joblib.dump``.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for r in results:
        path = models_dir / f"{r.name}.joblib"
        if is_neural_model(r.model):
            r.model.save(path)
        else:
            joblib.dump(r.model, path)
        saved[r.name] = str(path.relative_to(PROJECT_ROOT))
        logger.info("Saved %s → %s", r.name, path.name)
    return saved


# ── Build held-out evaluation set ─────────────────────────────────────────────

def _build_heldout(shots_path: Path) -> pd.DataFrame | None:
    """Return Euro 2024 (val_test) shots for held-out evaluation, or None."""
    matches_path = PROCESSED_DIR / "matches.parquet"
    if not matches_path.exists():
        logger.warning("matches.parquet not found — skipping held-out evaluation.")
        return None
    matches = pd.read_parquet(matches_path)[["internal_id", "split_role"]]
    shots = pd.read_parquet(shots_path)
    merged = shots.merge(
        matches.rename(columns={"internal_id": "match_internal_id"}),
        on="match_internal_id",
        how="left",
    )
    heldout = merged[merged["split_role"] == "val_test"].copy()
    if heldout.empty or "goal" not in heldout.columns:
        logger.warning("No held-out (val_test) shots or missing 'goal' column — skipping held-out eval.")
        return None
    logger.info("Held-out set: %d shots (val_test = Euro 2024)", len(heldout))
    return heldout


# ── Per-model held-out evaluation ─────────────────────────────────────────────

def _eval_heldout(results: list[LadderResult], heldout: pd.DataFrame) -> dict[str, dict]:
    """Run each fitted model against held-out set. Returns {name: metrics_dict}."""
    y_true = heldout["goal"].astype(int).to_numpy()
    evals: dict[str, dict] = {}
    for r in results:
        try:
            p = r.model.predict_proba(heldout)
            evals[r.name] = {
                "heldout_log_loss": round(float(log_loss(y_true, p, labels=[0, 1])), 5),
                "heldout_brier": round(float(brier_score_loss(y_true, p)), 5),
                "heldout_auc": round(float(roc_auc_score(y_true, p)), 4),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Held-out eval failed for %s: %s", r.name, exc)
            evals[r.name] = {}
    return evals


# ── Save JSON report ──────────────────────────────────────────────────────────

def _save_report(
    results: list[LadderResult],
    heldout_evals: dict[str, dict],
    saved_paths: dict[str, str],
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "leaderboard": [
            {
                "rank": r.rank,
                "name": r.name,
                "family": r.family,
                "feature_set": r.feature_set,
                "cv_log_loss": round(r.cv_log_loss, 5),
                "cv_brier": round(r.cv_brier, 5),
                "cv_auc": round(r.cv_auc, 4) if r.cv_auc is not None else None,
                "n_cv_folds_used": r.n_cv_folds_used,
                "heldout": heldout_evals.get(r.name, {}),
                "model_path": saved_paths.get(r.name, ""),
            }
            for r in results
        ]
    }
    out_path = reports_dir / "cxg_training_summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Metrics report → %s", out_path)


# ── Charts ────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#2196F3", "#4CAF50", "#FF5722", "#9C27B0",
    "#FF9800", "#00BCD4", "#E91E63", "#607D8B",
]


def _chart_leaderboard(results: list[LadderResult], figures_dir: Path) -> None:
    names = [r.name for r in results]
    ll_vals = [r.cv_log_loss for r in results]
    auc_vals = [r.cv_auc if r.cv_auc is not None else 0.0 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.6 * len(names) + 1)))
    colors = _PALETTE[: len(names)]

    # Log-loss (lower is better — sort ascending already done by ladder)
    axes[0].barh(names[::-1], ll_vals[::-1], color=colors[::-1])
    axes[0].set_xlabel("CV Log-loss (↓ better)")
    axes[0].set_title("CxG — CV Log-loss by Candidate")
    for i, v in enumerate(ll_vals[::-1]):
        axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)

    # AUC (higher is better)
    auc_order = sorted(range(len(auc_vals)), key=lambda i: auc_vals[i])
    axes[1].barh(
        [names[i] for i in auc_order],
        [auc_vals[i] for i in auc_order],
        color=[colors[i] for i in auc_order],
    )
    axes[1].set_xlabel("CV ROC-AUC (↑ better)")
    axes[1].set_title("CxG — CV AUC by Candidate")
    for j, idx in enumerate(auc_order):
        axes[1].text(auc_vals[idx] + 0.001, j, f"{auc_vals[idx]:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    out = figures_dir / "leaderboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_roc(results: list[LadderResult], heldout: pd.DataFrame, figures_dir: Path) -> None:
    y_true = heldout["goal"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")

    for i, r in enumerate(results):
        try:
            p = r.model.predict_proba(heldout)
            fpr, tpr, _ = roc_curve(y_true, p)
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=_PALETTE[i % len(_PALETTE)], lw=1.5,
                    label=f"{r.name}  AUC={roc_auc:.3f}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ROC chart skipped for %s: %s", r.name, exc)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("CxG — ROC Curves (held-out Euro 2024)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = figures_dir / "roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_calibration(results: list[LadderResult], heldout: pd.DataFrame, figures_dir: Path) -> None:
    y_true = heldout["goal"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")

    for i, r in enumerate(results):
        try:
            p = r.model.predict_proba(heldout)
            frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, "o-", color=_PALETTE[i % len(_PALETTE)],
                    lw=1.5, ms=4, label=r.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Calibration chart skipped for %s: %s", r.name, exc)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (goals)")
    ax.set_title("CxG — Calibration (held-out Euro 2024)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = figures_dir / "calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


# ── Diagnostic charts ────────────────────────────────────────────────────────


def _chart_binned_empirical(
    results: list[LadderResult], heldout: pd.DataFrame, figures_dir: Path
) -> None:
    """Empirical goal rate vs mean model prediction, binned by distance."""
    if "distance_to_goal" not in heldout.columns or "goal" not in heldout.columns:
        logger.warning("Skipping binned empirical chart — missing required columns.")
        return

    y_true = heldout["goal"].astype(int).to_numpy()
    dist = heldout["distance_to_goal"].to_numpy()

    # 14 quantile bins → roughly equal shot counts per bin
    n_bins = 14
    bin_edges = np.unique(np.quantile(dist, np.linspace(0, 1, n_bins + 1)))
    bin_labels = np.digitize(dist, bin_edges, right=True).clip(1, len(bin_edges) - 1)
    n_actual = len(bin_edges) - 1
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    empirical = np.array([y_true[bin_labels == b].mean() for b in range(1, n_actual + 1)])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(bin_centres, empirical, s=70, zorder=5, color="black",
               label="Empirical goal rate", marker="D")

    highlight = {"glm_contextual", "xgb_contextual", "baseline_logit"}
    for i, r in enumerate(results):
        if r.name not in highlight:
            continue
        try:
            p = r.model.predict_proba(heldout)
            pred = np.array([p[bin_labels == b].mean() for b in range(1, n_actual + 1)])
            ax.plot(bin_centres, pred, "o-", color=_PALETTE[i % len(_PALETTE)],
                    lw=1.8, ms=5, label=r.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Binned empirical skipped for %s: %s", r.name, exc)

    ax.set_xlabel("Distance to goal (m)")
    ax.set_ylabel("Goal probability")
    ax.set_title("CxG — Empirical vs Predicted Goal Rate by Distance\n"
                 "(held-out Euro 2024, quantile bins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = figures_dir / "binned_empirical_distance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_pitch_heatmap(
    results: list[LadderResult], heldout: pd.DataFrame, figures_dir: Path
) -> None:
    """2-D goal probability surface on the internal 105x68 pitch (GLM vs XGB)."""
    from mplsoccer import Pitch

    if "x_location" not in heldout.columns or "y_location" not in heldout.columns:
        logger.warning("Skipping pitch heatmap — x_location / y_location not in data.")
        return

    model_map = {r.name: r.model for r in results}
    to_plot = [(n, model_map[n]) for n in ("glm_contextual", "xgb_contextual") if n in model_map]
    if not to_plot:
        logger.warning("Skipping pitch heatmap — required models not found.")
        return

    # Build a grid over the attacking half of the internal 105x68 pitch.
    x_range = np.linspace(52.5, 104.5, 60)
    y_range = np.linspace(0.5, 67.5, 40)
    xx, yy = np.meshgrid(x_range, y_range)
    n_pts = xx.size

    # Synthetic DataFrame: all other features set to their median / mode from heldout
    num_medians = heldout.select_dtypes(include="number").median().to_dict()
    grid_df = pd.DataFrame({col: np.full(n_pts, val) for col, val in num_medians.items()})
    for col in heldout.select_dtypes(include=["object", "category"]).columns:
        mode_val = heldout[col].mode()
        grid_df[col] = mode_val.iloc[0] if not mode_val.empty else "unknown"

    grid_df["x_location"] = xx.ravel()
    grid_df["y_location"] = yy.ravel()

    # Recompute geometry from grid positions on the internal pitch.
    goal_x, near_y, far_y = 105.0, 30.34, 37.66
    dx = goal_x - grid_df["x_location"]
    grid_df["distance_to_goal"] = np.sqrt(dx ** 2 + (grid_df["y_location"] - 34.0) ** 2)
    a1 = np.arctan2(near_y - grid_df["y_location"], dx)
    a2 = np.arctan2(far_y - grid_df["y_location"], dx)
    grid_df["shot_angle"] = np.abs(a2 - a1)

    n_cols = len(to_plot)
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        pitch_color="#1a1a2e",
        line_color="white",
        line_zorder=2,
        linewidth=1.8,
    )
    fig, axes = pitch.draw(nrows=1, ncols=n_cols, figsize=(7 * n_cols, 5.5))
    fig.patch.set_facecolor("#1a1a2e")
    if n_cols == 1:
        axes = [axes]

    surfaces: list[tuple[str, np.ndarray]] = []
    all_probs: list[np.ndarray] = []
    for name, model in to_plot:
        try:
            probs = model.predict_proba(grid_df)
            surfaces.append((name, probs))
            all_probs.append(probs[np.isfinite(probs)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pitch heatmap failed for %s: %s", name, exc)
            surfaces.append((name, np.array([])))

    finite_probs = [vals for vals in all_probs if vals.size]
    if not finite_probs:
        logger.warning("Skipping pitch heatmap — no model probabilities available.")
        plt.close(fig)
        return
    vmin = 0.0
    vmax = float(max(np.nanmax(vals) for vals in finite_probs))

    hm = None
    for ax, (name, probs) in zip(axes, surfaces):
        if probs.size == 0:
            ax.set_title(f"{name} — error")
            continue
        hm = pitch.hexbin(
            grid_df["x_location"],
            grid_df["y_location"],
            ax=ax,
            C=probs,
            reduce_C_function=np.nanmean,
            gridsize=(24, 16),
            mincnt=1,
            cmap="RdYlGn_r",
            edgecolors="none",
            alpha=0.95,
            zorder=1,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{name}")

    if hm is not None:
        fig.subplots_adjust(right=0.92, wspace=0.14)
        cax = fig.add_axes([0.935, 0.17, 0.012, 0.66])
        cb = fig.colorbar(hm, cax=cax, label="P(goal)")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        cb.set_label("P(goal)", color="white")

    fig.suptitle(
        "CxG — Goal Probability Surface\n"
        "(internal 105x68 pitch; all other features held at held-out medians)",
        fontsize=11,
        color="white",
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    out = figures_dir / "pitch_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_residuals(
    results: list[LadderResult], heldout: pd.DataFrame, figures_dir: Path
) -> None:
    """Residual (y − p̂) vs distance and angle for the best model, with binned smoother."""
    if "distance_to_goal" not in heldout.columns:
        logger.warning("Skipping residuals chart — distance_to_goal not in held-out data.")
        return

    # Use production model (glm_contextual) if present, otherwise rank-1
    best = next((r for r in results if r.name == "glm_contextual"), results[0])

    y_true = heldout["goal"].astype(int).to_numpy()
    try:
        p_hat = best.model.predict_proba(heldout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Residuals chart skipped: %s", exc)
        return

    residuals = y_true - p_hat
    feats = [f for f in ("distance_to_goal", "shot_angle") if f in heldout.columns]

    fig, axes = plt.subplots(1, len(feats), figsize=(7 * len(feats), 5))
    if len(feats) == 1:
        axes = [axes]

    for ax, feat in zip(axes, feats):
        x_vals = heldout[feat].to_numpy()
        ax.scatter(x_vals, residuals, alpha=0.12, s=7, color="#607D8B", rasterized=True)
        ax.axhline(0, color="black", lw=1)

        # Binned mean (approximate LOWESS) — 30 equal-width bins
        edges = np.linspace(x_vals.min(), x_vals.max(), 31)
        centres, means = [], []
        for j in range(30):
            mask = (x_vals >= edges[j]) & (x_vals < edges[j + 1])
            if mask.sum() >= 5:
                centres.append(0.5 * (edges[j] + edges[j + 1]))
                means.append(residuals[mask].mean())
        ax.plot(centres, means, color="#E91E63", lw=2.2, label="Binned mean residual")

        ax.set_xlabel(feat.replace("_", " ").title())
        ax.set_ylabel("Residual  (y − p̂)")
        ax.set_title(f"{best.name}\nResiduals vs {feat.replace('_', ' ')}")
        ax.legend(fontsize=8)

    fig.suptitle("CxG — Residual Diagnostics (held-out Euro 2024)", fontsize=11)
    fig.tight_layout()
    out = figures_dir / "residuals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_coef_forest(results: list[LadderResult], figures_dir: Path) -> None:
    """Horizontal bar chart of glm_contextual coefficients (log-odds, features standardised)."""
    model_result = next((r for r in results if r.name == "glm_contextual"), None)
    if model_result is None:
        logger.warning("Skipping coef forest — glm_contextual not in results.")
        return

    model = model_result.model
    if not hasattr(model, "pipeline") or model.pipeline is None:
        logger.warning("Skipping coef forest — pipeline not fitted.")
        return

    try:
        pre = model.pipeline.named_steps["pre"]
        clf = model.pipeline.named_steps["clf"]
        raw_names = list(pre.get_feature_names_out())
        coefs = clf.coef_[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping coef forest — could not extract coefficients: %s", exc)
        return

    # Strip ColumnTransformer prefixes produced by get_feature_names_out
    clean_names = [
        n.replace("num__", "").replace("cat__", "") for n in raw_names
    ]

    n_top = min(30, len(coefs))
    order = np.argsort(np.abs(coefs))[-n_top:]
    c_sorted = coefs[order]
    names_sorted = [clean_names[i] for i in order]
    colors = ["#E53935" if c > 0 else "#1E88E5" for c in c_sorted]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.38 * n_top + 1.2)))
    y_pos = np.arange(n_top)
    ax.barh(y_pos, c_sorted, color=colors, alpha=0.78, height=0.65)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("Coefficient (log-odds, inputs standardised)")
    ax.set_title(
        f"glm_contextual — Top {n_top} Coefficients\n"
        "Red = increases goal probability  |  Blue = decreases"
    )
    fig.tight_layout()
    out = figures_dir / "coef_forest.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_calibration_by_shot_type(
    results: list[LadderResult], heldout: pd.DataFrame, figures_dir: Path
) -> None:
    """Reliability diagram split by shot_type for glm_contextual."""
    seg_col = next((c for c in ("shot_type", "sequence_type") if c in heldout.columns), None)
    if seg_col is None:
        logger.warning("Skipping calibration-by-type — shot_type / sequence_type not found.")
        return

    model = next((r.model for r in results if r.name == "glm_contextual"), None)
    if model is None:
        model = results[0].model

    y_true = heldout["goal"].astype(int).to_numpy()
    try:
        p_hat = model.predict_proba(heldout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Calibration-by-type skipped: %s", exc)
        return

    seg_vals = heldout[seg_col].astype(str).replace("nan", "unknown").replace("", "unknown").to_numpy()
    segments = sorted(np.unique(seg_vals))
    n_seg = len(segments)
    n_cols = min(3, n_seg)
    n_rows = (n_seg + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes_flat = axes.ravel()

    for i, seg in enumerate(segments):
        ax = axes_flat[i]
        mask = seg_vals == seg
        n_shots = mask.sum()
        n_goals = int(y_true[mask].sum())
        if n_shots < 20:
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_title(f"{seg}\n(n={n_shots} — too few)")
            continue
        try:
            frac_pos, mean_pred = calibration_curve(
                y_true[mask], p_hat[mask], n_bins=8, strategy="quantile"
            )
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
            ax.plot(mean_pred, frac_pos, "o-", color="#2196F3", lw=1.8, ms=5)
            ax.set_title(f"{seg}\n(n={n_shots}, {n_goals} goals)")
        except Exception:  # noqa: BLE001
            ax.set_title(f"{seg} — error")
        ax.set_xlabel("Predicted p")
        ax.set_ylabel("Empirical rate")

    for ax in axes_flat[n_seg:]:
        ax.set_visible(False)

    fig.suptitle(
        f"glm_contextual — Calibration by {seg_col.replace('_', ' ')}\n"
        "(held-out Euro 2024)",
        fontsize=11,
    )
    fig.tight_layout()
    out = figures_dir / f"calibration_by_{seg_col}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


# ── Main ──────────────────────────────────────────────────────────────────────

def train_cxg(
    shots_path: Path,
    n_folds: int = 5,
    n_optuna_trials: int = 0,
    include_360: bool = False,
    include_neural: bool = False,
    frames_path: Path | None = None,
    promote: bool = True,
    random_state: int = 42,
    n_estimators: int = 300,
) -> None:
    if not shots_path.exists():
        logger.error("shots.parquet not found at %s — run build_features.py first.", shots_path)
        sys.exit(1)

    shots_df = pd.read_parquet(shots_path)
    logger.info("Loaded shots: %d rows × %d columns", len(shots_df), len(shots_df.columns))

    if "goal" not in shots_df.columns:
        logger.error("shots.parquet is missing 'goal' target column.")
        sys.exit(1)

    # ── Exclude val_test (Euro 2024) shots from training ──────────────────────
    matches_path = PROCESSED_DIR / "matches.parquet"
    if matches_path.exists() and "match_internal_id" in shots_df.columns:
        _matches = pd.read_parquet(matches_path)[["internal_id", "split_role"]].rename(
            columns={"internal_id": "match_internal_id"}
        )
        shots_df = shots_df.merge(_matches, on="match_internal_id", how="left")
        n_before = len(shots_df)
        # val_test = Euro 2024 (held-out for evaluation).
        # test = La Liga (reserved for downstream scoring only).
        shots_df = shots_df[~shots_df["split_role"].isin({"val_test", "test"})].copy()
        logger.info(
            "Excluded val_test (Euro 2024) + test (La Liga) shots from training: "
            "%d → %d rows for training",
            n_before, len(shots_df),
        )
    else:
        logger.warning(
            "Could not filter val_test split — matches.parquet missing or "
            "match_internal_id not in shots. Training on all %d shots.",
            len(shots_df),
        )

    match_id_col = next(
        (c for c in ("match_id", "match_internal_id") if c in shots_df.columns),
        "match_id",
    )

    mlflow = _get_mlflow()

    with (_start_run(mlflow, "cfm/cxg", "ladder_run") or _NullContext()):
        ladder = CxGLadder()
        logger.info(
            "Running CxGLadder: n_folds=%d n_optuna=%d include_360=%s include_neural=%s n_estimators=%d",
            n_folds, n_optuna_trials, include_360, include_neural, n_estimators,
        )
        results = ladder.run(
            shots_df,
            target_col="goal",
            match_id_col=match_id_col,
            n_folds=n_folds,
            n_optuna_trials=n_optuna_trials,
            include_360=include_360,
            include_neural=include_neural,
            frames_path=str(frames_path) if frames_path else None,
            random_state=random_state,
            n_estimators=n_estimators,
        )

        # Print leaderboard
        lb = ladder.leaderboard()
        logger.info("\n%s", lb.to_string(index=False))

        best = ladder.best()
        logger.info(
            "Best model: %s  cv_log_loss=%.4f  cv_brier=%.4f",
            best.name, best.cv_log_loss, best.cv_brier,
        )

        # ── Save all candidate models as joblib ───────────────────────────────
        saved_paths = _save_all_models(results, MODELS_DIR)

        # ── Held-out evaluation (Euro 2024 val_test) ──────────────────────────
        heldout_df = _build_heldout(shots_path)
        heldout_evals: dict[str, dict] = {}
        if heldout_df is not None:
            heldout_evals = _eval_heldout(results, heldout_df)
            logger.info("Held-out evaluation complete for %d models.", len(heldout_evals))

        # ── Metrics report ────────────────────────────────────────────────────
        _save_report(results, heldout_evals, saved_paths, REPORTS_DIR)

        # ── Charts ────────────────────────────────────────────────────────────
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        _chart_leaderboard(results, FIGURES_DIR)
        if heldout_df is not None:
            _chart_roc(results, heldout_df, FIGURES_DIR)
            _chart_calibration(results, heldout_df, FIGURES_DIR)
            _chart_binned_empirical(results, heldout_df, FIGURES_DIR)
            _chart_pitch_heatmap(results, heldout_df, FIGURES_DIR)
            _chart_residuals(results, heldout_df, FIGURES_DIR)
            _chart_calibration_by_shot_type(results, heldout_df, FIGURES_DIR)
        else:
            logger.warning("Skipping held-out charts — no held-out set available.")
        _chart_coef_forest(results, FIGURES_DIR)

        # ── MLflow logging ────────────────────────────────────────────────────
        if mlflow is not None:
            mlflow.log_param("n_folds", n_folds)
            mlflow.log_param("best_model", best.name)
            mlflow.log_param("include_360", include_360)
            mlflow.log_metric("cv_log_loss", best.cv_log_loss)
            mlflow.log_metric("cv_brier", best.cv_brier)
            if best.cv_auc is not None:
                mlflow.log_metric("cv_auc", best.cv_auc)
            for fig_path in FIGURES_DIR.glob("*.png"):
                mlflow.log_artifact(str(fig_path))

    # Promote best model in configs/models.yaml
    best_model_path = MODELS_DIR / f"{best.name}.joblib"
    if promote:
        _update_production_pointer(str(best_model_path.relative_to(PROJECT_ROOT)))


class _NullContext:
    """No-op context manager used when MLflow is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CxG models and select the best via CV.")
    p.add_argument(
        "--shots",
        default=str(FEATURES_DIR / "shots.parquet"),
        help="Path to shots feature parquet (default: data/features/shots.parquet).",
    )
    p.add_argument("--n-folds", type=int, default=5, help="CV folds (default: 5).")
    p.add_argument(
        "--n-optuna-trials",
        type=int,
        default=0,
        help="Optuna hyperparameter trials for the best tree model (default: 0 = skip).",
    )
    p.add_argument(
        "--include-360",
        action="store_true",
        help="Include full_360 feature set candidates (requires 360 data).",
    )
    p.add_argument(
        "--include-neural",
        action="store_true",
        help="Include the SetTransformer-over-freeze-frames neural CxG model. "
             "Requires PyTorch and freeze_frames_360.parquet.",
    )
    p.add_argument(
        "--frames",
        default=None,
        help="Path to the freeze-frame parquet for the neural model "
             "(default: data/processed/frames.parquet, falling back to "
             "data/processed/freeze_frames_360.parquet if that doesn't exist).",
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees for tree models (default: 300).",
    )
    p.add_argument(
        "--no-promote",
        action="store_true",
        help="Skip updating the production pointer in configs/models.yaml.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train_cxg(
        shots_path=Path(args.shots),
        n_folds=args.n_folds,
        n_optuna_trials=args.n_optuna_trials,
        include_360=args.include_360,
        include_neural=args.include_neural,
        frames_path=Path(args.frames) if args.frames else None,
        promote=not args.no_promote,
        random_state=args.seed,
        n_estimators=args.n_estimators,
    )
