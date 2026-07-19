"""
GLM Contextual CxG model — Step 2 of the CxG ladder.

Extends the Phase 4 baseline logistic regression to the full CONTEXTUAL
feature set (opponent adjustment + match context + sequence context).

Optional L2 regularisation constant C is selected by k-fold grid search
when tune=True is passed to fit().
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.cxg.feature_sets import FeatureSetSpec, get_feature_set

logger = logging.getLogger(__name__)

_DEFAULT_C_SEARCH: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)


# ── Metrics ───────────────────────────────────────────────────────────────────


@dataclass
class GlmContextualMetrics:
    log_loss: float
    brier: float
    auc: float | None
    best_c: float | None = None


# ── Model ─────────────────────────────────────────────────────────────────────


class GlmContextualCxG:
    """
    Logistic regression CxG model using the contextual (or custom) feature set.

    Parameters
    ----------
    feature_set : FeatureSetSpec | str
        Feature group to use. Defaults to CONTEXTUAL.
    C : float
        Inverse regularisation strength (larger = less regularisation).
    max_iter : int
        Maximum solver iterations.
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        feature_set: FeatureSetSpec | str = "contextual",
        C: float = 1.0,
        max_iter: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.feature_set = (
            get_feature_set(feature_set) if isinstance(feature_set, str) else feature_set
        )
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.best_C_: float | None = None
        # Set during fit — defines the exact columns the pipeline was trained on.
        self._numeric_all: list[str] = []
        self._cat_cols: list[str] = []
        self._bool_set: frozenset[str] = frozenset()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_cols(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Return (numeric_all, cat_cols) filtered to columns present in df.
        numeric_all = numeric + boolean (both handled by the numeric transformer).
        """
        numeric_all = [c for c in self.feature_set.numeric_all if c in df.columns]
        cat_cols = [c for c in self.feature_set.categorical if c in df.columns]
        return numeric_all, cat_cols

    def _build_pipeline(
        self,
        C: float,
        numeric_all: list[str],
        cat_cols: list[str],
    ) -> Pipeline:
        transformers: list = [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_all,
            ),
        ]
        if cat_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    cat_cols,
                )
            )
        pre = ColumnTransformer(transformers, remainder="drop")
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=self.max_iter,
            C=C,
            random_state=self.random_state,
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build input matrix with ALL trained columns.
        Columns missing from df are filled with NaN (numeric) or 'unknown' (cat).
        """
        X = pd.DataFrame(index=df.index)
        for col in self._numeric_all:
            raw = df.get(col, pd.Series(np.nan, index=df.index))
            s = pd.to_numeric(raw, errors="coerce")
            X[col] = s.astype(float) if col in self._bool_set else s
        for col in self._cat_cols:
            X[col] = (
                df.get(col, pd.Series("unknown", index=df.index))
                .astype(str)
                .replace("nan", "unknown")
                .replace("", "unknown")
            )
        return X

    @staticmethod
    def _make_x_from(
        df: pd.DataFrame,
        numeric_all: list[str],
        cat_cols: list[str],
        bool_set: frozenset[str],
    ) -> pd.DataFrame:
        """Static version of _x used during tune_c (before instance vars are set)."""
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

    # ── Public API ────────────────────────────────────────────────────────────

    def tune_c(
        self,
        shots_df: pd.DataFrame,
        target_col: str = "goal",
        c_search: Sequence[float] = _DEFAULT_C_SEARCH,
        cv: int = 5,
    ) -> float:
        """
        Grid-search over c_search values using stratified k-fold cross-validation.
        Updates self.best_C_ and returns the best C value found.
        """
        numeric_all, cat_cols = self._resolve_cols(shots_df)
        bool_set = frozenset(self.feature_set.boolean)
        X = self._make_x_from(shots_df, numeric_all, cat_cols, bool_set)
        y = shots_df[target_col].astype(int).to_numpy()

        best_C, best_score = self.C, float("inf")
        for c in c_search:
            pipe = self._build_pipeline(c, numeric_all, cat_cols)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_log_loss", n_jobs=1)
            mean_ll = -float(np.mean(scores))
            logger.debug("GlmContextualCxG: C=%.4f → cv_log_loss=%.4f", c, mean_ll)
            if mean_ll < best_score:
                best_score = mean_ll
                best_C = c

        logger.info("GlmContextualCxG: best_C=%.4f (cv_log_loss=%.4f)", best_C, best_score)
        self.best_C_ = best_C
        return best_C

    def fit(
        self,
        shots_df: pd.DataFrame,
        target_col: str = "goal",
        tune: bool = False,
        c_search: Sequence[float] = _DEFAULT_C_SEARCH,
        cv: int = 5,
    ) -> GlmContextualCxG:
        """
        Fit the model.

        Parameters
        ----------
        shots_df   : shot events from the feature store (must contain target_col)
        target_col : binary goal indicator column
        tune       : if True, grid-search C before fitting
        c_search   : C values to search when tune=True
        cv         : number of folds for C tuning
        """
        if shots_df.empty:
            raise ValueError("shots_df is empty")
        if target_col not in shots_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")

        numeric_all, cat_cols = self._resolve_cols(shots_df)
        if not numeric_all:
            raise ValueError("No numeric features found in shots_df for this feature set")

        self._numeric_all = numeric_all
        self._cat_cols = cat_cols
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)

        C_to_use = self.tune_c(shots_df, target_col, c_search, cv) if tune else self.C
        self.pipeline = self._build_pipeline(C_to_use, numeric_all, cat_cols)

        X = self._x(shots_df)
        y = shots_df[target_col].astype(int).to_numpy()
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, shots_df: pd.DataFrame) -> np.ndarray:
        """Return P(goal=1) for each row."""
        if self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict_proba(self._x(shots_df))[:, 1]

    def evaluate(self, shots_df: pd.DataFrame, target_col: str = "goal") -> GlmContextualMetrics:
        """Compute log-loss, Brier score, AUC on shots_df."""
        y = shots_df[target_col].astype(int).to_numpy()
        p = self.predict_proba(shots_df)
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
        return GlmContextualMetrics(
            log_loss=float(log_loss(y, p, labels=[0, 1])),
            brier=float(brier_score_loss(y, p)),
            auc=auc,
            best_c=self.best_C_,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> GlmContextualCxG:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
