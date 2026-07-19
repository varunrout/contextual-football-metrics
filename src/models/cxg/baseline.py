"""Phase 4 baseline CxG model (logistic regression)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

_DEFAULT_NUMERIC = ["distance_to_goal", "shot_angle", "x_location", "y_location"]
_DEFAULT_BOOL = ["header", "volley", "under_pressure", "open_play"]
_DEFAULT_CAT = ["body_part", "shot_type", "set_piece_type"]


@dataclass
class BaselineCxGMetrics:
    log_loss: float
    brier: float
    auc: float | None


class BaselineCxGModel:
    """Baseline xG model trained on shot events only."""

    def __init__(self, random_state: int = 42, C: float = 1.0, max_iter: int = 1000) -> None:
        self.random_state = random_state
        self.C = C
        self.max_iter = max_iter
        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = []

    def _resolve_features(self, df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        numeric = [c for c in _DEFAULT_NUMERIC if c in df.columns]
        booleans = [c for c in _DEFAULT_BOOL if c in df.columns]
        categorical = [c for c in _DEFAULT_CAT if c in df.columns]
        if not (numeric or booleans or categorical):
            raise ValueError("No usable baseline CxG feature columns found")
        return numeric, booleans, categorical

    def fit(self, shots_df: pd.DataFrame, target_col: str = "goal") -> BaselineCxGModel:
        if shots_df.empty:
            raise ValueError("shots_df is empty")
        if target_col not in shots_df.columns:
            raise ValueError(f"Missing target column: {target_col}")

        numeric, booleans, categorical = self._resolve_features(shots_df)
        self.feature_columns = numeric + booleans + categorical

        numeric_all = numeric + booleans
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    numeric_all,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical,
                ),
            ]
        )

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=self.max_iter,
            C=self.C,
            random_state=self.random_state,
        )

        self.pipeline = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("clf", clf),
            ]
        )

        X = shots_df[self.feature_columns].copy()
        for col in booleans:
            if col in X.columns:
                X[col] = X[col].astype(float)
        y = shots_df[target_col].astype(int).to_numpy()
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, shots_df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted")
        X = shots_df[self.feature_columns].copy()
        for col in _DEFAULT_BOOL:
            if col in X.columns:
                X[col] = X[col].astype(float)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, shots_df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(shots_df) >= threshold).astype(int)

    def evaluate(self, shots_df: pd.DataFrame, target_col: str = "goal") -> BaselineCxGMetrics:
        y_true = shots_df[target_col].astype(int).to_numpy()
        p = self.predict_proba(shots_df)
        auc = None
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, p))
        return BaselineCxGMetrics(
            log_loss=float(log_loss(y_true, p, labels=[0, 1])),
            brier=float(brier_score_loss(y_true, p)),
            auc=auc,
        )


def filter_shot_events(features_df: pd.DataFrame) -> pd.DataFrame:
    """Utility: keep shot rows from unified feature store."""
    if "event_type" not in features_df.columns:
        return features_df.copy()
    return features_df[features_df["event_type"].astype(str) == "shot"].copy()
