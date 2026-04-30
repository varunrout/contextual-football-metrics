"""
XGBoost CxG model — Step 3 of the CxG ladder.

Supports all three feature sets (traditional / contextual / full_360).
Hyperparameter search via Optuna (optional — pass n_trials > 0 to fit()).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.evaluation.validation_splits import match_kfold
from src.models.cxg.feature_sets import CONTEXTUAL, FeatureSetSpec, get_feature_set

logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class XGBCxGMetrics:
    log_loss: float
    brier: float
    auc: float | None
    best_params: dict = field(default_factory=dict)


# ── Shared pipeline builder (used by XGBoost and re-used by ladder) ───────────

def _build_tree_pipeline(
    estimator,
    numeric_all: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """
    Shared sklearn Pipeline constructor for all tree-based CxG models.

    numeric_all : already includes bool columns cast to float
    cat_cols    : string categoricals → OrdinalEncoder → float
    """
    transformers: list = [
        ("num", SimpleImputer(strategy="median"), numeric_all),
    ]
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("enc", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=float,
                )),
            ]),
            cat_cols,
        ))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("clf", estimator)])


def _make_X(
    df: pd.DataFrame,
    numeric_all: list[str],
    cat_cols: list[str],
    bool_set: frozenset[str],
) -> pd.DataFrame:
    """Build input matrix, filling missing columns with appropriate defaults."""
    X = pd.DataFrame(index=df.index)
    for col in numeric_all:
        raw = df.get(col, pd.Series(np.nan, index=df.index))
        s = pd.to_numeric(raw, errors="coerce")
        X[col] = s.astype(float) if col in bool_set else s
    for col in cat_cols:
        X[col] = (
            df.get(col, pd.Series("unknown", index=df.index))
            .astype(str)
            .replace("nan", "unknown")
            .replace("", "unknown")
        )
    return X


# ── XGBoost model ─────────────────────────────────────────────────────────────

class XGBoostCxGModel:
    """
    XGBoost binary classifier for shot-goal probability.

    Parameters
    ----------
    feature_set : FeatureSetSpec | str
        Which feature group to use ("traditional" | "contextual" | "full_360").
    n_estimators, learning_rate, max_depth, … : XGBoost hyperparameters.
    random_state : int
    """

    def __init__(
        self,
        feature_set: FeatureSetSpec | str = "contextual",
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise ImportError("xgboost not installed. Run: poetry install --with models") from exc

        self.device = device
        self.feature_set = (
            get_feature_set(feature_set) if isinstance(feature_set, str) else feature_set
        )
        self.params: dict = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.best_params_: dict = {}
        self._numeric_all: list[str] = []
        self._cat_cols: list[str] = []
        self._bool_set: frozenset[str] = frozenset()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_cols(
        self, df: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        numeric_all = [c for c in self.feature_set.numeric_all if c in df.columns]
        cat_cols = [c for c in self.feature_set.categorical if c in df.columns]
        return numeric_all, cat_cols

    def _make_estimator(self, params: dict):
        import xgboost as xgb
        from src.runtime.gbm_device import xgboost_kwargs
        return xgb.XGBClassifier(
            **params,
            **xgboost_kwargs(self.device),
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            random_state=self.random_state,
        )

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return _make_X(df, self._numeric_all, self._cat_cols, self._bool_set)

    # ── Optuna hyperparameter search ──────────────────────────────────────────

    def tune(
        self,
        shots_df: pd.DataFrame,
        target_col: str = "goal",
        n_trials: int = 40,
        match_id_col: str = "match_id",
        n_folds: int = 3,
    ) -> dict:
        """
        Optuna TPE hyperparameter search with match-level k-fold cross-validation.
        Returns the best params dict and stores them in self.best_params_.
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("optuna not installed. Run: poetry install --with models") from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Reset index so match_kfold label indices == positional indices
        df = shots_df.reset_index(drop=True)
        numeric_all, cat_cols = self._resolve_cols(df)
        bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)
        X_all = _make_X(df, numeric_all, cat_cols, bool_set)
        y_all = df[target_col].astype(int).to_numpy()

        if match_id_col not in df.columns:
            logger.warning("match_id_col %r not found; using 3-fold stratified CV", match_id_col)
            from sklearn.model_selection import StratifiedKFold
            folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state).split(X_all, y_all))
        else:
            folds = list(match_kfold(df, n_splits=n_folds, match_id_col=match_id_col, random_state=self.random_state))

        def objective(trial) -> float:
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 800),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 3.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0),
            )
            cv_scores: list[float] = []
            for tr_idx, va_idx in folds:
                pipe = _build_tree_pipeline(self._make_estimator(params), numeric_all, cat_cols)
                pipe.fit(X_all.loc[tr_idx], y_all[tr_idx])
                p = pipe.predict_proba(X_all.loc[va_idx])[:, 1]
                y_va = y_all[va_idx]
                if len(np.unique(y_va)) > 1:
                    cv_scores.append(log_loss(y_va, p, labels=[0, 1]))
            return float(np.mean(cv_scores)) if cv_scores else 1.0

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params_ = study.best_params
        logger.info(
            "XGBoostCxGModel: best_params=%s (cv_log_loss=%.4f)",
            study.best_params, study.best_value,
        )
        return study.best_params

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        shots_df: pd.DataFrame,
        target_col: str = "goal",
        n_trials: int = 0,
        match_id_col: str = "match_id",
    ) -> "XGBoostCxGModel":
        """
        Fit the model.

        Parameters
        ----------
        shots_df   : shot events with required feature columns
        target_col : binary goal indicator
        n_trials   : Optuna trials to run before final fit (0 = skip tuning)
        match_id_col : column used for match-level CV inside Optuna
        """
        if shots_df.empty:
            raise ValueError("shots_df is empty")
        if target_col not in shots_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")

        df = shots_df.reset_index(drop=True)
        numeric_all, cat_cols = self._resolve_cols(df)
        if not numeric_all:
            raise ValueError("No feature columns found in shots_df for this feature set")

        self._numeric_all = numeric_all
        self._cat_cols = cat_cols
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)

        if n_trials > 0 and match_id_col in df.columns:
            best = self.tune(df, target_col, n_trials, match_id_col)
            self.params.update(best)

        self.pipeline = _build_tree_pipeline(
            self._make_estimator(self.params), numeric_all, cat_cols
        )
        X = self._X(df)
        y = df[target_col].astype(int).to_numpy()
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, shots_df: pd.DataFrame) -> np.ndarray:
        """Return P(goal=1) for each row."""
        if self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict_proba(self._X(shots_df))[:, 1]

    def evaluate(
        self, shots_df: pd.DataFrame, target_col: str = "goal"
    ) -> XGBCxGMetrics:
        y = shots_df[target_col].astype(int).to_numpy()
        p = self.predict_proba(shots_df)
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
        return XGBCxGMetrics(
            log_loss=float(log_loss(y, p, labels=[0, 1])),
            brier=float(brier_score_loss(y, p)),
            auc=auc,
            best_params=self.best_params_,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostCxGModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
