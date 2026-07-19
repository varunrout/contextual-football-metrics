"""
Weakly-supervised multi-class sequence type classifier (Phase 2c).

Architecture:
  - LightGBM multi-class (num_class = len(SequenceType) – 1 for UNKNOWN)
  - Training labels come from the rule-based labeller (Phase 2b)
  - Sample weights = rule confidence scores, clipped to [min_label_weight, 1.0]
  - Excludes UNKNOWN-labelled possessions from training
  - Outputs soft probabilities; assigns final label only when max_prob ≥ threshold

MLflow integration mirrors the pattern from configs/models.yaml:
  experiment = "cfm/sequence"
  run name   = "sequence.lgbm.v1"

The model is intentionally lightweight — sequence typing is a pre-processing
step, not a scored metric. Production calibration is deferred to Phase 5+.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

from src.ingestion.schema import SequenceType

logger = logging.getLogger(__name__)

# All SequenceType values we will predict (exclude UNKNOWN from target classes)
_LABEL_CLASSES: list[str] = [st.value for st in SequenceType if st != SequenceType.UNKNOWN]
_CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(_LABEL_CLASSES)}

# Features consumed by the classifier (must exist in possessions_df after Phase 2b)
_SEQUENCE_FEATURE_COLS = [
    "n_events",
    "n_passes",
    "n_carries",
    "vertical_progression",
    "passes_before_action",
    "carries_before_action",
    "time_from_possession_start",
    "vertical_progression_speed",
    "directness",
    "possession_speed",
    "number_of_switches",
    "set_piece_flag",
    "counterpress_regain_flag",
    "start_x",
    "start_y",
]

# Categorical feature columns (encoded as category dtype for LGBM)
_CAT_COLS = ["possession_start_zone", "regain_zone", "phase_of_play", "transition_or_settled"]

_ALL_FEATURE_COLS = _SEQUENCE_FEATURE_COLS + _CAT_COLS

_MIN_LABEL_WEIGHT = 0.5  # from models.yaml: weak_label_weight
_MIN_CONFIDENCE = 0.70  # only train on possessions where rule confidence ≥ 0.70
_PREDICT_THRESHOLD = 0.40  # assign label only if max class prob ≥ threshold


class SequenceClassifier:
    """
    Wrapper around a LightGBM multi-class model for sequence typing.

    Parameters
    ----------
    lgb_params : override default LightGBM hyperparameters
    min_confidence : rule confidence threshold below which possessions are
                     excluded from training (default 0.70)
    predict_threshold : minimum max-class probability to assign a label;
                        below this UNKNOWN is returned (default 0.40)
    """

    def __init__(
        self,
        lgb_params: dict[str, Any] | None = None,
        min_confidence: float = _MIN_CONFIDENCE,
        predict_threshold: float = _PREDICT_THRESHOLD,
    ) -> None:
        if not _HAS_LGB:
            raise ImportError("lightgbm is required. Install with: poetry install --with models")
        self.min_confidence = min_confidence
        self.predict_threshold = predict_threshold
        self.lgb_params = {
            "objective": "multiclass",
            "num_class": len(_LABEL_CLASSES),
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }
        if lgb_params:
            self.lgb_params.update(lgb_params)
        self._model: lgb.LGBMClassifier | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, possessions_df: pd.DataFrame) -> SequenceClassifier:
        """
        Fit on possessions that have rule labels with sufficient confidence.

        Parameters
        ----------
        possessions_df : must contain columns from _ALL_FEATURE_COLS plus
                         'sequence_type_rule' and 'sequence_type_confidence'
        """
        train_df = self._prepare_train(possessions_df)
        if train_df.empty:
            raise ValueError(
                "No possessions with rule confidence ≥ "
                f"{self.min_confidence} and known sequence type"
            )

        X, y, w = self._build_xyw(train_df)
        from src.runtime.gbm_device import lightgbm_kwargs

        params = {**self.lgb_params, **lightgbm_kwargs()}
        self._model = lgb.LGBMClassifier(**params)
        self._model.fit(X, y, sample_weight=w)
        logger.info(
            "SequenceClassifier trained on %d possessions (%d classes)",
            len(train_df),
            len(_LABEL_CLASSES),
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, possessions_df: pd.DataFrame) -> np.ndarray:
        """Return (n_possessions, n_classes) probability matrix."""
        if self._model is None:
            raise RuntimeError("Model not fitted — call fit() first")
        X = self._prepare_features(possessions_df)
        return self._model.predict_proba(X)

    def predict(self, possessions_df: pd.DataFrame) -> list[str]:
        """Return predicted SequenceType.value strings (UNKNOWN when uncertain)."""
        proba = self.predict_proba(possessions_df)
        labels = []
        for row in proba:
            max_idx = int(np.argmax(row))
            if row[max_idx] >= self.predict_threshold:
                labels.append(_LABEL_CLASSES[max_idx])
            else:
                labels.append(SequenceType.UNKNOWN.value)
        return labels

    def label_dataframe(self, possessions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add / overwrite sequence_type, sequence_type_confidence,
        sequence_type_source columns using the trained classifier.
        """
        proba = self.predict_proba(possessions_df)
        max_probs = proba.max(axis=1)
        max_idxs = proba.argmax(axis=1)
        labels = [
            _LABEL_CLASSES[idx] if prob >= self.predict_threshold else SequenceType.UNKNOWN.value
            for idx, prob in zip(max_idxs, max_probs, strict=False)
        ]
        possessions_df = possessions_df.copy()
        possessions_df["sequence_type"] = labels
        possessions_df["sequence_type_confidence"] = max_probs
        possessions_df["sequence_type_source"] = "classifier"
        return possessions_df

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Save the underlying LightGBM model to disk."""
        if self._model is None:
            raise RuntimeError("Nothing to save — model not fitted")
        import joblib

        joblib.dump(self._model, path)
        logger.info("SequenceClassifier saved to %s", path)

    @classmethod
    def load(cls, path: Path | str, **kwargs: Any) -> SequenceClassifier:
        """Load a previously saved classifier."""
        import joblib

        instance = cls(**kwargs)
        instance._model = joblib.load(path)
        return instance

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare_train(self, df: pd.DataFrame) -> pd.DataFrame:
        known = df[
            (df["sequence_type_rule"] != SequenceType.UNKNOWN.value)
            & (df["sequence_type_confidence"] >= self.min_confidence)
        ]
        return known

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in _ALL_FEATURE_COLS if c in df.columns]
        X = df[available].copy()
        # Encode booleans as int
        for col in ["set_piece_flag", "counterpress_regain_flag"]:
            if col in X.columns:
                X[col] = X[col].astype(int)
        # Categorical columns
        for col in _CAT_COLS:
            if col in X.columns:
                X[col] = X[col].astype("category")
        # Fill missing numeric features with 0
        num_cols = [c for c in available if c not in _CAT_COLS]
        X[num_cols] = X[num_cols].fillna(0.0)
        return X

    def _build_xyw(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        X = self._prepare_features(df)
        y = np.array([_CLASS_TO_IDX.get(v, 0) for v in df["sequence_type_rule"]], dtype=np.int32)
        w = np.clip(
            df["sequence_type_confidence"].fillna(_MIN_LABEL_WEIGHT).values,
            _MIN_LABEL_WEIGHT,
            1.0,
        )
        return X, y, w

    # ── MLflow logging ─────────────────────────────────────────────────────────

    def log_to_mlflow(self, experiment_name: str = "cfm/sequence") -> None:
        """Log model params + feature importances to MLflow (best-effort)."""
        if not _HAS_MLFLOW or self._model is None:
            return
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="sequence.lgbm.v1"):
            mlflow.log_params(self.lgb_params)
            fi = self._model.feature_importances_
            feature_names = getattr(self._model, "feature_name_", [f"f{i}" for i in range(len(fi))])
            for name, imp in zip(feature_names, fi, strict=False):
                mlflow.log_metric(f"feature_importance_{name}", float(imp))
            logger.info("Logged SequenceClassifier run to MLflow experiment '%s'", experiment_name)
