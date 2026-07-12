"""
CxG Model Ladder — Phase 5

Trains and cross-validates all CxG model candidates, ranks them by CV log-loss,
and produces a leaderboard table ready for MLflow logging.

Candidates in order of complexity:
  1. baseline_logit      (traditional features, logistic regression — Phase 4)
  2. glm_contextual      (contextual features, logistic regression)
  3. xgb_traditional     (traditional features, XGBoost)
  4. xgb_contextual      (contextual features, XGBoost)
  5. lgbm_traditional    (traditional features, LightGBM)
  6. lgbm_contextual     (contextual features, LightGBM)
  7. xgb_full_360        (full 360 features, XGBoost)     — only when include_360=True
  8. lgbm_full_360       (full 360 features, LightGBM)    — only when include_360=True

Cross-validation is always at match level (whole matches stay together) to
prevent match-context leakage. Models are evaluated with default hyperparameters
during CV; Optuna tuning runs once on the full training set for the winner.

Usage
-----
    from src.models.cxg.ladder import CxGLadder

    ladder = CxGLadder()
    results = ladder.run(shots_df, match_id_col="match_id", n_folds=5)
    print(ladder.leaderboard())
    best = ladder.best()  # fitted model with lowest CV log-loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.evaluation.validation_splits import match_kfold
from src.models.cxg.baseline import BaselineCxGModel
from src.models.cxg.glm_contextual import GlmContextualCxG
from src.models.cxg.lightgbm_model import LightGBMCxGModel
from src.models.cxg.xgboost_model import XGBoostCxGModel

logger = logging.getLogger(__name__)


def _build_set_transformer(frames_path: str | None, random_state: int):
    """Lazy SetTransformer factory — torch import deferred until called."""
    from src.models.cxg.set_transformer_model import SetTransformerCxGModel
    return SetTransformerCxGModel(
        feature_set="contextual",
        frames_path=frames_path,
        random_state=random_state,
    )


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class LadderResult:
    """Evaluation record for one model candidate."""

    name: str                           # e.g. "lgbm_contextual"
    family: str                         # "logistic" | "xgboost" | "lightgbm"
    feature_set: str                    # "traditional" | "contextual" | "full_360"
    cv_log_loss: float
    cv_brier: float
    cv_auc: float | None
    n_cv_folds_used: int                # folds that had ≥1 positive
    model: object = field(repr=False)   # fitted model instance (on full data)
    rank: int = 0


# ── CV helper ─────────────────────────────────────────────────────────────────

def _cross_validate(
    model_factory: Callable[[], object],
    shots_df: pd.DataFrame,
    target_col: str,
    match_id_col: str,
    n_folds: int,
    random_state: int,
) -> tuple[float, float, float | None, int]:
    """
    Match-level k-fold cross-validation.

    Returns
    -------
    (mean_log_loss, mean_brier, mean_auc, n_valid_folds)
    """
    df = shots_df.reset_index(drop=True)

    if match_id_col in df.columns:
        folds = list(
            match_kfold(df, n_splits=n_folds, match_id_col=match_id_col, random_state=random_state)
        )
    else:
        logger.warning("match_id_col %r not in shots_df; using random %d-fold CV", match_id_col, n_folds)
        from sklearn.model_selection import KFold
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=random_state).split(df))

    log_losses: list[float] = []
    briers: list[float] = []
    aucs: list[float] = []

    for tr_idx, va_idx in folds:
        tr_df = df.loc[tr_idx]
        va_df = df.loc[va_idx]

        # Skip fold if not enough data or no positive class in train/val
        if len(tr_df) < 10 or tr_df[target_col].nunique() < 2:
            continue
        if va_df[target_col].nunique() < 2:
            continue

        model = model_factory()
        model.fit(tr_df, target_col)
        p = model.predict_proba(va_df)
        y = va_df[target_col].astype(int).to_numpy()

        log_losses.append(log_loss(y, p, labels=[0, 1]))
        briers.append(brier_score_loss(y, p))
        if len(np.unique(y)) > 1:
            aucs.append(roc_auc_score(y, p))

    if not log_losses:
        logger.warning("No valid CV folds — returning sentinel metrics")
        return float("inf"), float("inf"), None, 0

    return (
        float(mean(log_losses)),
        float(mean(briers)),
        float(mean(aucs)) if aucs else None,
        len(log_losses),
    )


# ── Ladder ────────────────────────────────────────────────────────────────────

class CxGLadder:
    """
    Trains and ranks all CxG model candidates.

    After calling run(), access results via leaderboard() and best().
    """

    def __init__(self) -> None:
        self._results: list[LadderResult] = []

    # ── Core ──────────────────────────────────────────────────────────────────

    def run(
        self,
        shots_df: pd.DataFrame,
        target_col: str = "goal",
        match_id_col: str = "match_id",
        n_folds: int = 5,
        n_optuna_trials: int = 0,
        include_360: bool = False,
        include_neural: bool = False,
        frames_path: str | None = None,
        random_state: int = 42,
        n_estimators: int = 300,
    ) -> list[LadderResult]:
        """
        Evaluate all candidates with match-level k-fold CV, then refit on full data.

        Parameters
        ----------
        shots_df         : shot events from the feature store (must contain target_col)
        target_col       : binary goal indicator column name
        match_id_col     : column used for match-level CV splits
        n_folds          : number of CV folds
        n_optuna_trials  : Optuna trials for the best tree model (0 = skip tuning)
        include_360      : if True, adds full_360 candidates (only useful when 360 data present)
        random_state     : reproducibility seed
        n_estimators     : number of trees for tree models (lower = faster; override for tests)
        """
        if shots_df.empty:
            raise ValueError("shots_df is empty")
        if target_col not in shots_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")

        candidates: list[tuple[str, str, str, Callable[[], object]]] = [
            # (name, family, feature_set, factory)
            (
                "baseline_logit", "logistic", "traditional",
                lambda: BaselineCxGModel(random_state=random_state),
            ),
            (
                "glm_contextual", "logistic", "contextual",
                lambda: GlmContextualCxG(feature_set="contextual", random_state=random_state),
            ),
            (
                "xgb_traditional", "xgboost", "traditional",
                lambda ne=n_estimators: XGBoostCxGModel(
                    feature_set="traditional", n_estimators=ne, random_state=random_state
                ),
            ),
            (
                "xgb_contextual", "xgboost", "contextual",
                lambda ne=n_estimators: XGBoostCxGModel(
                    feature_set="contextual", n_estimators=ne, random_state=random_state
                ),
            ),
            (
                "lgbm_traditional", "lightgbm", "traditional",
                lambda ne=n_estimators: LightGBMCxGModel(
                    feature_set="traditional", n_estimators=ne, random_state=random_state
                ),
            ),
            (
                "lgbm_contextual", "lightgbm", "contextual",
                lambda ne=n_estimators: LightGBMCxGModel(
                    feature_set="contextual", n_estimators=ne, random_state=random_state
                ),
            ),
        ]

        if include_360:
            candidates += [
                (
                    "xgb_full_360", "xgboost", "full_360",
                    lambda ne=n_estimators: XGBoostCxGModel(
                        feature_set="full_360", n_estimators=ne, random_state=random_state
                    ),
                ),
                (
                    "lgbm_full_360", "lightgbm", "full_360",
                    lambda ne=n_estimators: LightGBMCxGModel(
                        feature_set="full_360", n_estimators=ne, random_state=random_state
                    ),
                ),
            ]

        if include_neural:
            candidates.append((
                "set_transformer_360", "neural", "contextual+360_set",
                lambda fp=frames_path: _build_set_transformer(fp, random_state),
            ))

        results: list[LadderResult] = []

        for name, family, fset, factory in candidates:
            logger.info("CxGLadder: evaluating %s …", name)

            try:
                # Probe importability once before CV
                factory()
            except ImportError as exc:
                logger.warning("CxGLadder: skipping %s — %s", name, exc)
                continue

            cv_ll, cv_brier, cv_auc, n_valid = _cross_validate(
                factory, shots_df, target_col, match_id_col, n_folds, random_state
            )
            logger.info(
                "  %s — cv_log_loss=%.4f  cv_brier=%.4f  auc=%s  folds_used=%d",
                name, cv_ll, cv_brier,
                f"{cv_auc:.4f}" if cv_auc is not None else "N/A",
                n_valid,
            )

            # Fit final model on full data
            final_model = factory()
            final_model.fit(shots_df, target_col)

            results.append(LadderResult(
                name=name,
                family=family,
                feature_set=fset,
                cv_log_loss=cv_ll,
                cv_brier=cv_brier,
                cv_auc=cv_auc,
                n_cv_folds_used=n_valid,
                model=final_model,
            ))

        # Rank by CV log-loss (lower is better)
        results.sort(key=lambda r: r.cv_log_loss)
        for i, r in enumerate(results):
            r.rank = i + 1

        self._results = results

        # Optional: tune the best tree model with Optuna on full data
        if n_optuna_trials > 0:
            best = results[0]
            if best.family in ("xgboost", "lightgbm"):
                logger.info(
                    "CxGLadder: running Optuna on best model (%s) with %d trials",
                    best.name, n_optuna_trials,
                )
                best.model.fit(shots_df, target_col, n_trials=n_optuna_trials, match_id_col=match_id_col)

        return results

    # ── Outputs ───────────────────────────────────────────────────────────────

    def leaderboard(self) -> pd.DataFrame:
        """Return a ranked DataFrame of CV metrics for all candidates."""
        if not self._results:
            raise RuntimeError("No results yet. Call run() first.")
        rows = []
        for r in self._results:
            rows.append({
                "rank": r.rank,
                "name": r.name,
                "family": r.family,
                "feature_set": r.feature_set,
                "cv_log_loss": round(r.cv_log_loss, 5),
                "cv_brier": round(r.cv_brier, 5),
                "cv_auc": round(r.cv_auc, 4) if r.cv_auc is not None else None,
                "n_cv_folds_used": r.n_cv_folds_used,
            })
        return pd.DataFrame(rows).set_index("rank")

    def best(self) -> LadderResult:
        """Return the LadderResult with the lowest CV log-loss."""
        if not self._results:
            raise RuntimeError("No results yet. Call run() first.")
        return self._results[0]

    def get(self, name: str) -> LadderResult:
        """Return a specific result by model name."""
        for r in self._results:
            if r.name == name:
                return r
        raise KeyError(f"No result for model {name!r}")

    # ── MLflow logging ────────────────────────────────────────────────────────

    def log_to_mlflow(
        self,
        experiment_name: str = "cfm/cxg",
        run_name_prefix: str = "cxg.ladder",
    ) -> None:
        """
        Log every ladder result as a separate MLflow run.

        Requires MLflow to be configured (mlflow.set_tracking_uri called before).
        """
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("mlflow not installed") from exc

        mlflow.set_experiment(experiment_name)
        for r in self._results:
            run_name = f"{run_name_prefix}.{r.name}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "name": r.name,
                    "family": r.family,
                    "feature_set": r.feature_set,
                    "rank": r.rank,
                })
                mlflow.log_metrics({
                    "cv_log_loss": r.cv_log_loss,
                    "cv_brier": r.cv_brier,
                    **({} if r.cv_auc is None else {"cv_auc": r.cv_auc}),
                })
                logger.info("MLflow run logged: %s", run_name)
