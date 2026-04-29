"""Phase 4 baseline xA model (two-stage expected assisted xG)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


_NUMERIC = [
    "x_location",
    "y_location",
    "pass_length",
    "pass_angle",
    "progressive_distance",
]
_BOOL = [
    "under_pressure",
    "cross",
    "cutback",
    "through_ball",
    "switch",
    "central_progression",
    "box_entry",
]
_CAT = ["pass_height", "pass_body_part", "set_piece_type", "phase_of_play", "sequence_type"]


@dataclass
class BaselineCxAMetrics:
    creation_log_loss: float
    quality_rmse: float
    total_rmse: float
    total_r2: float | None


class BaselineCxAModel:
    """Two-stage baseline:
    1) P(shot created) classifier
    2) E(xG | shot created) regressor
    Final xA = p_create * quality
    """

    def __init__(self, random_state: int = 42, C: float = 1.0, alpha: float = 1.0) -> None:
        self.random_state = random_state
        self.C = C
        self.alpha = alpha
        self.feature_columns: list[str] = []
        self.creation_model: Pipeline | None = None
        self.quality_model: Pipeline | None = None

    def _resolve_feature_cols(self, df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        num = [c for c in _NUMERIC if c in df.columns]
        boo = [c for c in _BOOL if c in df.columns]
        cat = [c for c in _CAT if c in df.columns]
        if not (num or boo or cat):
            raise ValueError("No usable baseline CxA feature columns found")
        return num, boo, cat

    def _build_preprocessor(self, num: list[str], boo: list[str], cat: list[str]) -> ColumnTransformer:
        num_all = num + boo
        return ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_all),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat,
                ),
            ]
        )

    def fit(
        self,
        pass_df: pd.DataFrame,
        creation_target_col: str = "leads_to_shot",
        quality_target_col: str = "resulting_shot_xg",
    ) -> "BaselineCxAModel":
        if pass_df.empty:
            raise ValueError("pass_df is empty")

        df = pass_df.copy()
        if creation_target_col not in df.columns:
            if "shot_assist" in df.columns:
                df[creation_target_col] = df["shot_assist"].astype(bool).astype(int)
            else:
                raise ValueError(f"Missing creation target column: {creation_target_col}")

        if quality_target_col not in df.columns:
            if "next_shot_xg" in df.columns:
                df[quality_target_col] = pd.to_numeric(df["next_shot_xg"], errors="coerce").fillna(0.0)
            else:
                raise ValueError(f"Missing quality target column: {quality_target_col}")

        num, boo, cat = self._resolve_feature_cols(df)
        self.feature_columns = num + boo + cat
        pre = self._build_preprocessor(num, boo, cat)

        self.creation_model = Pipeline(
            steps=[
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=1000,
                        C=self.C,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        self.quality_model = Pipeline(
            steps=[
                ("pre", pre),
                ("reg", Ridge(alpha=self.alpha, random_state=self.random_state)),
            ]
        )

        X = df[self.feature_columns].copy()
        for col in boo:
            if col in X.columns:
                X[col] = X[col].astype(float)
        y_create = df[creation_target_col].astype(int).to_numpy()
        y_quality = pd.to_numeric(df[quality_target_col], errors="coerce").fillna(0.0).to_numpy()

        self.creation_model.fit(X, y_create)

        # Quality stage: only rows that actually led to shots.
        mask = y_create == 1
        if mask.sum() == 0:
            raise ValueError("No positive shot-creation examples for quality model")
        self.quality_model.fit(X.loc[mask], y_quality[mask])

        return self

    def predict_components(self, pass_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.creation_model is None or self.quality_model is None:
            raise RuntimeError("Model not fitted")
        X = pass_df[self.feature_columns].copy()
        for col in _BOOL:
            if col in X.columns:
                X[col] = X[col].astype(float)
        p_create = self.creation_model.predict_proba(X)[:, 1]
        q = self.quality_model.predict(X)
        q = np.clip(q, 0.0, 1.0)
        return p_create, q

    def predict_xa(self, pass_df: pd.DataFrame) -> np.ndarray:
        p_create, q = self.predict_components(pass_df)
        return p_create * q

    def evaluate(
        self,
        pass_df: pd.DataFrame,
        creation_target_col: str = "leads_to_shot",
        quality_target_col: str = "resulting_shot_xg",
        total_target_col: str = "xa_target",
    ) -> BaselineCxAMetrics:
        if creation_target_col not in pass_df.columns:
            raise ValueError(f"Missing creation target column: {creation_target_col}")
        if quality_target_col not in pass_df.columns:
            raise ValueError(f"Missing quality target column: {quality_target_col}")

        y_create = pass_df[creation_target_col].astype(int).to_numpy()
        y_quality = pd.to_numeric(pass_df[quality_target_col], errors="coerce").fillna(0.0).to_numpy()

        p_create, q = self.predict_components(pass_df)
        xa = p_create * q

        y_total = (
            pd.to_numeric(pass_df[total_target_col], errors="coerce").fillna(0.0).to_numpy()
            if total_target_col in pass_df.columns
            else y_create * y_quality
        )

        total_r2 = None
        if len(np.unique(y_total)) > 1:
            total_r2 = float(r2_score(y_total, xa))

        return BaselineCxAMetrics(
            creation_log_loss=float(log_loss(y_create, p_create, labels=[0, 1])),
            quality_rmse=float(mean_squared_error(y_quality, q) ** 0.5),
            total_rmse=float(mean_squared_error(y_total, xa) ** 0.5),
            total_r2=total_r2,
        )


def filter_pass_events(features_df: pd.DataFrame) -> pd.DataFrame:
    if "event_type" not in features_df.columns:
        return features_df.copy()
    return features_df[features_df["event_type"].astype(str) == "pass"].copy()
