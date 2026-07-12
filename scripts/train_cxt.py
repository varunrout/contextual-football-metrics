"""
scripts/train_cxt.py
====================
Fit the CxT state-value ladder (Gamma GLM + XGBoost + LightGBM) using
discounted possession CxG as the regression target.

Logged to MLflow under experiment  cfm/cxt.
All candidate models saved to  models/cxt/<name>.joblib.
Best model promoted to configs/models.yaml production.cxt pointer.

The script:
  1. Loads features.parquet (all events with contextual features).
  2. Attaches per-shot CxG scores from the production CxG model (inline).
  3. Computes discounted possession_cxg target per action row.
  4. Excludes val_test (Euro 2024) rows from training.
  5. Filters to CxT-eligible action types (pass, carry, cross, cutback).
  6. Runs StateValueLadder — CV across 3 candidates ranked by MAE.
  7. Evaluates all candidates on held-out Euro 2024 set.
  8. Saves all candidates as joblib, generates figures, writes JSON report.

Usage
-----
    python scripts/train_cxt.py
    python scripts/train_cxt.py --n-folds 5 --n-estimators 400
    python scripts/train_cxt.py --features data/features/features.parquet --no-promote
"""

from __future__ import annotations

import argparse
import json
import logging
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

from src.models.cxt.state_value_model import (
    StateValueLadder,
    StateValueLadderResult,
    compute_possession_cxg,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s \u2014 %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_cxt")

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "cxt"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "cxt"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"

# Action types eligible for CxT (require end_x / end_y before/after state)
_CXT_ACTION_TYPES = {"pass", "carry", "cross", "cutback"}

_PALETTE = [
    "#2196F3", "#4CAF50", "#FF5722", "#9C27B0",
    "#FF9800", "#00BCD4", "#E91E63", "#607D8B",
]


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
    cfg.setdefault("production", {})["cxt"] = model_filename
    with open(MODELS_YAML, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
    logger.info("Updated production.cxt pointer → %s", model_filename)


# ── Inline CxG scoring ────────────────────────────────────────────────────────

def _attach_cxg_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the production CxG model and score shot rows, adding a 'cxg' column.
    Returns features_df unchanged if the model is unavailable or no shots exist.
    """
    try:
        import joblib
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
        cxg_model = joblib.load(cxg_path)
    except Exception as exc:
        logger.warning("Could not load CxG model (%s) — skipping CxG scoring.", exc)
        return features_df

    _type_col = next(
        (c for c in ("event_type", "action_type") if c in features_df.columns), None
    )
    if _type_col is None:
        logger.warning("No event_type/action_type column — cannot identify shots for CxG scoring.")
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
        logger.warning("CxG scoring failed (%s) — possession_cxg target unavailable.", exc)

    return features_df


# ── Compute discounted possession_cxg target ─────────────────────────────────

def _build_target(features_df: pd.DataFrame) -> pd.DataFrame:
    """Attach possession_cxg column. Raises SystemExit if no CxG source found."""
    cxg_col = next((c for c in ("cxg", "event_cxg", "predicted_cxg") if c in features_df.columns), None)

    if cxg_col is None:
        logger.error(
            "No CxG column found in features (tried: cxg, event_cxg, predicted_cxg). "
            "Ensure the production CxG model is available in configs/models.yaml "
            "so it can be used to score shots inline."
        )
        sys.exit(1)

    logger.info("Computing discounted possession_cxg from column '%s' …", cxg_col)
    poss_id_col = next(
        (c for c in ("possession_internal_id", "possession_id") if c in features_df.columns), None
    )
    match_id_col = next(
        (c for c in ("match_internal_id", "match_id") if c in features_df.columns), None
    )
    if poss_id_col and match_id_col:
        features_df["possession_cxg"] = compute_possession_cxg(
            features_df,
            cxg_col=cxg_col,
            possession_id_col=poss_id_col,
            match_id_col=match_id_col,
        )
    else:
        logger.warning("Cannot find possession/match ID columns — using %s directly as target.", cxg_col)
        features_df["possession_cxg"] = features_df[cxg_col].fillna(0.0)

    logger.info(
        "possession_cxg  mean=%.5f  >0 rows=%d / %d",
        features_df["possession_cxg"].mean(),
        (features_df["possession_cxg"] > 0).sum(),
        len(features_df),
    )
    return features_df


# ── Split helpers ─────────────────────────────────────────────────────────────

def _split_train_heldout(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Exclude val_test (Euro 2024) rows from training and return them as
    the held-out evaluation set.  Returns (train_df, heldout_df | None).
    """
    matches_path = PROCESSED_DIR / "matches.parquet"
    match_id_col = next(
        (c for c in ("match_internal_id", "match_id") if c in features_df.columns), None
    )
    if not matches_path.exists() or match_id_col is None:
        logger.warning(
            "matches.parquet not found or match ID column missing — "
            "training on all rows, no held-out evaluation."
        )
        return features_df, None

    matches = pd.read_parquet(matches_path)[["internal_id", "split_role"]].rename(
        columns={"internal_id": match_id_col}
    )
    features_df = features_df.merge(matches, on=match_id_col, how="left")
    # val_test = Euro 2024 (held-out for evaluation).
    # test = La Liga (reserved for downstream scoring only — NOT used for
    # training or evaluation).
    heldout = features_df[features_df["split_role"] == "val_test"].copy()
    scoring = features_df[features_df["split_role"] == "test"].copy()
    train = features_df[~features_df["split_role"].isin({"val_test", "test"})].copy()
    logger.info(
        "Split: %d train rows, %d held-out (val_test/Euro 2024) rows, "
        "%d reserved for scoring only (test/La Liga, excluded).",
        len(train), len(heldout), len(scoring),
    )
    return train, heldout if not heldout.empty else None


def _filter_cxt_actions(df: pd.DataFrame, label: str = "dataset") -> pd.DataFrame:
    """Keep only CxT-eligible action types (pass, carry, cross, cutback)."""
    type_col = next(
        (c for c in ("action_type", "event_type") if c in df.columns), None
    )
    if type_col is None:
        logger.warning("No action_type/event_type column in %s — keeping all rows.", label)
        return df
    filtered = df[df[type_col].isin(_CXT_ACTION_TYPES)].copy()
    logger.info("Filtered %s to CxT action types via '%s': %d → %d rows",
                label, type_col, len(df), len(filtered))
    return filtered


# ── Save all candidate models ─────────────────────────────────────────────────

def _save_all_models(results: list[StateValueLadderResult], models_dir: Path) -> dict[str, str]:
    """Save every candidate model. Tree/GLM/GAM models use joblib;
    neural models (FFNN/SetTransformer/GNN) use their own ``.save()``
    pickle (they hold locally-defined ``nn.Module`` classes that joblib
    cannot round-trip). Returns ``{name: path_str}``."""
    import joblib
    models_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    neural_families = {"ffnn", "set_transformer", "gnn"}
    for r in results:
        path = models_dir / f"{r.name}.joblib"
        if r.family in neural_families and hasattr(r.model, "save"):
            r.model.save(path)
        else:
            joblib.dump(r.model, path)
        saved[r.name] = str(path.relative_to(PROJECT_ROOT))
        logger.info("Saved %s → %s", r.name, path.name)
    return saved


# ── Held-out evaluation ───────────────────────────────────────────────────────

def _eval_heldout(
    results: list[StateValueLadderResult], heldout: pd.DataFrame
) -> dict[str, dict]:
    """Evaluate each fitted model on the held-out set. Returns {name: metrics}."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import spearmanr

    y_true = heldout["possession_cxg"].astype(float).to_numpy()
    evals: dict[str, dict] = {}
    for r in results:
        try:
            p = r.model.predict(heldout)
            mae = float(mean_absolute_error(y_true, p))
            rmse = float(np.sqrt(mean_squared_error(y_true, p)))
            corr, _ = spearmanr(y_true, p)
            sp = float(corr) if not np.isnan(corr) else None
            evals[r.name] = {
                "heldout_mae": round(mae, 5),
                "heldout_rmse": round(rmse, 5),
                "heldout_spearman": round(sp, 4) if sp is not None else None,
            }
        except Exception as exc:
            logger.warning("Held-out eval failed for %s: %s", r.name, exc)
            evals[r.name] = {}
    return evals


# ── JSON report ───────────────────────────────────────────────────────────────

def _save_report(
    results: list[StateValueLadderResult],
    heldout_evals: dict[str, dict],
    saved_paths: dict[str, str],
    n_train: int,
    n_heldout: int,
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_train_rows": n_train,
        "n_heldout_rows": n_heldout,
        "leaderboard": [
            {
                "rank": r.rank,
                "name": r.name,
                "family": r.family,
                "feature_set": r.feature_set,
                "cv_mae": round(r.cv_mae, 5),
                "cv_rmse": round(r.cv_rmse, 5),
                "cv_spearman": round(r.cv_spearman, 4) if r.cv_spearman is not None else None,
                "n_cv_folds_used": r.n_cv_folds_used,
                "heldout": heldout_evals.get(r.name, {}),
                "model_path": saved_paths.get(r.name, ""),
            }
            for r in results
        ],
    }
    out_path = reports_dir / "cxt_training_summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Metrics report → %s", out_path)


# ── Charts ────────────────────────────────────────────────────────────────────

def _chart_leaderboard(results: list[StateValueLadderResult], figures_dir: Path) -> None:
    names = [r.name for r in results]
    mae_vals = [r.cv_mae for r in results]
    sp_vals = [r.cv_spearman if r.cv_spearman is not None else 0.0 for r in results]
    colors = _PALETTE[: len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.6 * len(names) + 1)))

    # MAE — lower is better; ladder already sorted ascending
    axes[0].barh(names[::-1], mae_vals[::-1], color=colors[::-1])
    axes[0].set_xlabel("CV MAE (↓ better)")
    axes[0].set_title("CxT — CV MAE by Candidate")
    for i, v in enumerate(mae_vals[::-1]):
        axes[0].text(v + 1e-5, i, f"{v:.5f}", va="center", fontsize=8)

    # Spearman — higher is better
    sp_order = sorted(range(len(sp_vals)), key=lambda i: sp_vals[i])
    axes[1].barh(
        [names[i] for i in sp_order],
        [sp_vals[i] for i in sp_order],
        color=[colors[i] for i in sp_order],
    )
    axes[1].set_xlabel("CV Spearman ρ (↑ better)")
    axes[1].set_title("CxT — CV Spearman by Candidate")
    for j, idx in enumerate(sp_order):
        axes[1].text(sp_vals[idx] + 1e-4, j, f"{sp_vals[idx]:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    out = figures_dir / "leaderboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_target_distribution(train_df: pd.DataFrame, figures_dir: Path) -> None:
    """Distribution of the possession_cxg target showing the zero-inflated structure."""
    vals = train_df["possession_cxg"].to_numpy()
    pos_vals = vals[vals > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # All values (mostly 0)
    axes[0].hist(vals, bins=60, color="#2196F3", alpha=0.8, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("possession_cxg")
    axes[0].set_ylabel("Count")
    axes[0].set_title(
        f"CxT Target Distribution (all)\n"
        f"n={len(vals):,}  zero={100*(vals==0).mean():.1f}%  mean={vals.mean():.4f}"
    )

    # Positive values only
    if pos_vals.size:
        axes[1].hist(pos_vals, bins=50, color="#4CAF50", alpha=0.8, edgecolor="white", linewidth=0.3)
        axes[1].set_xlabel("possession_cxg")
        axes[1].set_ylabel("Count")
        axes[1].set_title(
            f"CxT Target — Positive Values Only\n"
            f"n={len(pos_vals):,}  mean={pos_vals.mean():.4f}  p95={np.quantile(pos_vals, 0.95):.4f}"
        )
    else:
        axes[1].set_title("No positive values")

    fig.tight_layout()
    out = figures_dir / "target_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_predicted_vs_actual(
    results: list[StateValueLadderResult], heldout: pd.DataFrame, figures_dir: Path
) -> None:
    """Scatter of predicted vs actual possession_cxg on held-out set (positive target only)."""
    y_true = heldout["possession_cxg"].astype(float).to_numpy()
    pos_mask = y_true > 0

    if pos_mask.sum() < 20:
        logger.warning("Skipping predicted-vs-actual chart — fewer than 20 positive held-out targets.")
        return

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    axes = axes[0]

    for ax, r, color in zip(axes, results, _PALETTE):
        try:
            p = r.model.predict(heldout)
            ax.scatter(
                y_true[pos_mask], p[pos_mask],
                alpha=0.25, s=8, color=color, rasterized=True,
            )
            lim = max(y_true[pos_mask].max(), p[pos_mask].max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=1, label="Perfect")
            ax.set_xlabel("Actual possession_cxg")
            ax.set_ylabel("Predicted possession_cxg")
            ax.set_title(f"{r.name}\n(positive targets only, n={pos_mask.sum():,})")
            ax.legend(fontsize=7)
        except Exception as exc:
            logger.warning("predicted-vs-actual skipped for %s: %s", r.name, exc)
            ax.set_title(f"{r.name} — error")

    fig.suptitle("CxT — Predicted vs Actual (held-out Euro 2024)", fontsize=11)
    fig.tight_layout()
    out = figures_dir / "predicted_vs_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_residuals(
    results: list[StateValueLadderResult], heldout: pd.DataFrame, figures_dir: Path
) -> None:
    """Residual (y − ŷ) vs x_location for each candidate with binned mean smoother."""
    if "x_location" not in heldout.columns:
        logger.warning("Skipping residuals chart — x_location not in held-out data.")
        return

    y_true = heldout["possession_cxg"].astype(float).to_numpy()
    x_vals = heldout["x_location"].to_numpy()

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), squeeze=False)
    axes = axes[0]

    for ax, r, color in zip(axes, results, _PALETTE):
        try:
            p = r.model.predict(heldout)
            residuals = y_true - p
            ax.scatter(x_vals, residuals, alpha=0.12, s=6, color=color, rasterized=True)
            ax.axhline(0, color="black", lw=1)

            # Binned mean smoother
            edges = np.linspace(x_vals.min(), x_vals.max(), 31)
            centres, means = [], []
            for j in range(30):
                mask = (x_vals >= edges[j]) & (x_vals < edges[j + 1])
                if mask.sum() >= 5:
                    centres.append(0.5 * (edges[j] + edges[j + 1]))
                    means.append(residuals[mask].mean())
            ax.plot(centres, means, color="#E91E63", lw=2.2, label="Binned mean")

            ax.set_xlabel("x_location (pitch position)")
            ax.set_ylabel("Residual  (y − ŷ)")
            ax.set_title(f"{r.name}\nResiduals vs pitch x")
            ax.legend(fontsize=7)
        except Exception as exc:
            logger.warning("Residuals chart skipped for %s: %s", r.name, exc)
            ax.set_title(f"{r.name} — error")

    fig.suptitle("CxT — Residual Diagnostics (held-out Euro 2024)", fontsize=11)
    fig.tight_layout()
    out = figures_dir / "residuals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_pitch_value_surface(
    results: list[StateValueLadderResult], heldout: pd.DataFrame, figures_dir: Path
) -> None:
    """
    2-D V(s) surface on the internal 105×68 pitch — expected possession_cxg
    as a function of ball position, all other features held at held-out medians.
    """
    try:
        from mplsoccer import Pitch
    except ImportError:
        logger.warning("Skipping pitch value surface — mplsoccer not installed.")
        return

    if "x_location" not in heldout.columns or "y_location" not in heldout.columns:
        logger.warning("Skipping pitch value surface — x_location / y_location not in data.")
        return

    # Pick best two models for the plot
    model_names = [r.name for r in results]
    to_plot = [(r.name, r.model) for r in results[:2]]

    # Build a grid over the full pitch
    x_range = np.linspace(0.5, 104.5, 80)
    y_range = np.linspace(0.5, 67.5, 50)
    xx, yy = np.meshgrid(x_range, y_range)
    n_pts = xx.size

    # Grid DataFrame: numeric/bool features at median/mode, categorical at mode
    num_medians = heldout.select_dtypes(include=["number", "bool"]).median().to_dict()
    grid_df = pd.DataFrame({col: np.full(n_pts, val) for col, val in num_medians.items()})
    for col in heldout.select_dtypes(include=["object", "category"]).columns:
        mode_val = heldout[col].mode()
        grid_df[col] = mode_val.iloc[0] if not mode_val.empty else "unknown"

    # Override position columns with grid values
    grid_df["x_location"] = xx.ravel()
    grid_df["y_location"] = yy.ravel()

    # Recompute distance geometry from grid positions
    goal_x = 105.0
    dx = goal_x - grid_df["x_location"]
    grid_df["distance_to_goal"] = np.sqrt(dx**2 + (grid_df["y_location"] - 34.0) ** 2)

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
    all_vals: list[np.ndarray] = []
    for name, model in to_plot:
        try:
            v = model.predict(grid_df)
            surfaces.append((name, v))
            all_vals.append(v[np.isfinite(v)])
        except Exception as exc:
            logger.warning("Pitch value surface failed for %s: %s", name, exc)
            surfaces.append((name, np.array([])))

    finite_vals = [a for a in all_vals if a.size]
    if not finite_vals:
        logger.warning("Skipping pitch value surface — no predictions available.")
        plt.close(fig)
        return
    vmin = 0.0
    vmax = float(max(np.nanpercentile(a, 98) for a in finite_vals))

    hm = None
    for ax, (name, v) in zip(axes, surfaces):
        if v.size == 0:
            ax.set_title(f"{name} — error")
            continue
        hm = pitch.hexbin(
            grid_df["x_location"],
            grid_df["y_location"],
            ax=ax,
            C=v,
            reduce_C_function=np.nanmean,
            gridsize=(30, 20),
            mincnt=1,
            cmap="RdYlGn",
            edgecolors="none",
            alpha=0.95,
            zorder=1,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(name, color="white")

    if hm is not None:
        fig.subplots_adjust(right=0.92, wspace=0.14)
        cax = fig.add_axes([0.935, 0.17, 0.012, 0.66])
        cb = fig.colorbar(hm, cax=cax, label="V(s)")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        cb.set_label("V(s) = E[possession_cxg]", color="white")

    fig.suptitle(
        "CxT — State-Value Surface V(s)\n"
        "(internal 105×68 pitch; all other features at held-out medians)",
        fontsize=11,
        color="white",
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    out = figures_dir / "pitch_value_surface.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def _chart_calibration_by_zone(
    results: list[StateValueLadderResult], heldout: pd.DataFrame, figures_dir: Path
) -> None:
    """
    Mean predicted vs mean actual possession_cxg, binned by pitch zone.
    Uses x_location quantile bins as zones.
    """
    if "x_location" not in heldout.columns:
        logger.warning("Skipping calibration-by-zone chart — x_location not available.")
        return

    y_true = heldout["possession_cxg"].astype(float).to_numpy()
    x_vals = heldout["x_location"].to_numpy()
    n_bins = 10
    bin_edges = np.unique(np.quantile(x_vals, np.linspace(0, 1, n_bins + 1)))
    bin_labels = np.digitize(x_vals, bin_edges, right=True).clip(1, len(bin_edges) - 1)
    n_actual = len(bin_edges) - 1
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    empirical = np.array([y_true[bin_labels == b].mean() for b in range(1, n_actual + 1)])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(bin_centres, empirical, s=80, zorder=5, color="black",
               label="Empirical mean", marker="D")

    for i, r in enumerate(results):
        try:
            p = r.model.predict(heldout)
            pred_mean = np.array([p[bin_labels == b].mean() for b in range(1, n_actual + 1)])
            ax.plot(bin_centres, pred_mean, "o-", color=_PALETTE[i % len(_PALETTE)],
                    lw=1.8, ms=5, label=r.name)
        except Exception as exc:
            logger.warning("Calibration-by-zone skipped for %s: %s", r.name, exc)

    ax.set_xlabel("x_location (pitch position, quantile bins)")
    ax.set_ylabel("Mean possession_cxg")
    ax.set_title("CxT — Calibration by Pitch Zone\n(held-out Euro 2024)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = figures_dir / "calibration_by_zone.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


# ── Main ──────────────────────────────────────────────────────────────────────

def train_cxt(
    features_path: Path,
    n_folds: int = 5,
    n_estimators: int = 400,
    promote: bool = True,
    random_state: int = 42,
    include_neural: bool = False,
    frames_path: Path | None = None,
    nn_max_epochs: int = 30,
) -> None:
    if not features_path.exists():
        logger.error("features.parquet not found at %s — run build_features.py first.", features_path)
        sys.exit(1)

    features_df = pd.read_parquet(features_path)
    logger.info("Loaded features: %d rows × %d columns", len(features_df), len(features_df.columns))

    # ── Attach CxG scores inline (needed to build possession_cxg target) ──────
    features_df = _attach_cxg_scores(features_df)

    # ── Compute discounted possession_cxg target ──────────────────────────────
    features_df = _build_target(features_df)

    # ── Split train / held-out ────────────────────────────────────────────────
    all_df, heldout_all = _split_train_heldout(features_df)

    # ── Filter to CxT-eligible action types ──────────────────────────────────
    train_df = _filter_cxt_actions(all_df, label="train")
    heldout_df: pd.DataFrame | None = None
    if heldout_all is not None:
        heldout_df = _filter_cxt_actions(heldout_all, label="held-out")
        if heldout_df.empty:
            logger.warning("Held-out set has no CxT actions — skipping held-out evaluation.")
            heldout_df = None

    logger.info(
        "Training on %d CxT actions (target mean=%.5f, zero=%.1f%%)",
        len(train_df),
        train_df["possession_cxg"].mean(),
        100 * (train_df["possession_cxg"] == 0).mean(),
    )

    match_id_col = next(
        (c for c in ("match_internal_id", "match_id") if c in train_df.columns), "match_id"
    )

    mlflow = _get_mlflow()

    with (_start_run(mlflow, "cfm/cxt", "ladder_run") or _NullContext()):
        ladder = StateValueLadder()
        logger.info(
            "Running StateValueLadder: n_folds=%d, n_estimators=%d",
            n_folds, n_estimators,
        )
        results = ladder.run(
            train_df,
            target_col="possession_cxg",
            match_id_col=match_id_col,
            n_folds=n_folds,
            n_estimators=n_estimators,
            random_state=random_state,
            include_neural=include_neural,
            frames_path=str(frames_path) if frames_path is not None else None,
            nn_max_epochs=nn_max_epochs,
        )

        lb = ladder.leaderboard()
        logger.info("\n%s", lb.to_string(index=False))

        best = ladder.best()
        logger.info(
            "Best model: %s  cv_mae=%.5f  cv_rmse=%.5f  cv_spearman=%s",
            best.name, best.cv_mae, best.cv_rmse,
            f"{best.cv_spearman:.4f}" if best.cv_spearman is not None else "N/A",
        )

        # ── Save all candidate models ──────────────────────────────────────────
        saved_paths = _save_all_models(results, MODELS_DIR)

        # ── Held-out evaluation ────────────────────────────────────────────────
        heldout_evals: dict[str, dict] = {}
        if heldout_df is not None:
            heldout_evals = _eval_heldout(results, heldout_df)
            logger.info("Held-out evaluation complete for %d models.", len(heldout_evals))

        # ── JSON report ────────────────────────────────────────────────────────
        _save_report(
            results, heldout_evals, saved_paths,
            n_train=len(train_df),
            n_heldout=len(heldout_df) if heldout_df is not None else 0,
            reports_dir=REPORTS_DIR,
        )

        # ── Charts ─────────────────────────────────────────────────────────────
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        _chart_leaderboard(results, FIGURES_DIR)
        _chart_target_distribution(train_df, FIGURES_DIR)
        if heldout_df is not None:
            _chart_predicted_vs_actual(results, heldout_df, FIGURES_DIR)
            _chart_residuals(results, heldout_df, FIGURES_DIR)
            _chart_calibration_by_zone(results, heldout_df, FIGURES_DIR)
            _chart_pitch_value_surface(results, heldout_df, FIGURES_DIR)
        else:
            logger.warning("Skipping held-out charts — no held-out set available.")

        # ── MLflow logging ─────────────────────────────────────────────────────
        if mlflow is not None:
            mlflow.log_param("n_folds", n_folds)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("best_model", best.name)
            mlflow.log_metric("cv_mae", best.cv_mae)
            mlflow.log_metric("cv_rmse", best.cv_rmse)
            if best.cv_spearman is not None:
                mlflow.log_metric("cv_spearman", best.cv_spearman)
            for fig_path in FIGURES_DIR.glob("*.png"):
                mlflow.log_artifact(str(fig_path))

    # ── Promote best model to production pointer ───────────────────────────────
    best_model_path = MODELS_DIR / f"{best.name}.joblib"
    if promote:
        _update_production_pointer(str(best_model_path.relative_to(PROJECT_ROOT)))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the CxT state-value ladder.")
    p.add_argument(
        "--features",
        default=str(FEATURES_DIR / "features.parquet"),
        help="Full feature store parquet (default: data/features/features.parquet).",
    )
    p.add_argument("--n-folds", type=int, default=5, help="CV folds (default: 5).")
    p.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of trees for tree models (default: 400).",
    )
    p.add_argument(
        "--no-promote",
        action="store_true",
        help="Skip updating the production pointer in configs/models.yaml.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument(
        "--include-neural",
        action="store_true",
        help="Also train neural candidates (FFNN; SetTransformer/GNN if --frames given).",
    )
    p.add_argument(
        "--frames",
        default=None,
        help="Path to the freeze-frame parquet (enables SetTransformer + GNN). "
             "Default: data/processed/frames.parquet, falling back to "
             "data/processed/freeze_frames_360.parquet if that doesn't exist.",
    )
    p.add_argument(
        "--nn-max-epochs",
        type=int,
        default=30,
        help="Max training epochs for neural candidates (default: 30).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train_cxt(
        features_path=Path(args.features),
        n_folds=args.n_folds,
        n_estimators=args.n_estimators,
        promote=not args.no_promote,
        random_state=args.seed,
        include_neural=args.include_neural,
        frames_path=Path(args.frames) if args.frames else None,
        nn_max_epochs=args.nn_max_epochs,
    )
