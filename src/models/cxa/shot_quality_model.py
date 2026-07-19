"""
Resulting shot quality model — Stage 2 of the CxA two-stage pipeline.

Target: resulting_shot_cxg (from Phase 5's CxG model, linked by possession_id).
Unit of analysis: actions that are followed by a shot within the same possession.

This is a **regression** task (predicting a value in (0, 1]).

Model ladder:
  1. gamma_glm         — Gamma GLM (canonical link = log; appropriate for
                         positive-bounded targets like CxG)
  2. xgb_contextual    — XGBoost regressor
  3. lgbm_contextual   — LightGBM regressor
  4. mlp_regressor     — Feed-forward neural network with Huber loss
                         (optionally Beta NLL after scaling CxG to (0,1))

Evaluation: MAE, RMSE, Spearman rank correlation, calibration by value bucket.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.evaluation.validation_splits import match_kfold
from src.models.cxa.feature_sets import CxAFeatureSetSpec, get_feature_set
from src.models.cxa.shot_creation_model import _make_x

logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────


@dataclass
class ShotQualityMetrics:
    mae: float
    rmse: float
    spearman: float | None
    calibration_by_bucket: dict[str, float] = field(default_factory=dict)


@dataclass
class ShotQualityLadderResult:
    name: str
    family: str
    feature_set: str
    cv_mae: float
    cv_rmse: float
    cv_spearman: float | None
    n_cv_folds_used: int
    model: object = field(repr=False)
    rank: int = 0


# ── Shared pipeline builders ──────────────────────────────────────────────────


def _build_gamma_pipeline(
    numeric_all: list[str],
    cat_cols: list[str],
    alpha: float = 1.0,
) -> Pipeline:
    """Gamma GLM with log link via TweedieRegressor(power=2, link='log')."""
    from sklearn.preprocessing import OneHotEncoder

    transformers: list = [
        (
            "num",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
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
                        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            )
        )
    pre = ColumnTransformer(transformers, remainder="drop")
    # power=2 → Gamma family, link='log' (default for power>1)
    glm = TweedieRegressor(power=2, alpha=alpha, max_iter=2000, link="log")
    return Pipeline([("pre", pre), ("reg", glm)])


def _build_tree_reg_pipeline(
    estimator,
    numeric_all: list[str],
    cat_cols: list[str],
) -> Pipeline:
    transformers: list = [("num", SimpleImputer(strategy="median"), numeric_all)]
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                        (
                            "enc",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1, dtype=float
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("reg", estimator)])


# ── Shared evaluation logic ───────────────────────────────────────────────────


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> ShotQualityMetrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    corr, _ = spearmanr(y_true, y_pred)
    spearman = float(corr) if not np.isnan(corr) else None

    # Calibration by value bucket (quartiles)
    buckets: dict[str, float] = {}
    try:
        quantiles = np.quantile(y_true, [0.25, 0.5, 0.75])
        edges = [0.0] + quantiles.tolist() + [float("inf")]
        for i in range(len(edges) - 1):
            mask = (y_true >= edges[i]) & (y_true < edges[i + 1])
            if mask.sum() > 0:
                bucket_name = f"q{i + 1}"
                buckets[bucket_name] = float(np.mean(y_pred[mask]) - np.mean(y_true[mask]))
    except Exception:
        pass
    return ShotQualityMetrics(mae=mae, rmse=rmse, spearman=spearman, calibration_by_bucket=buckets)


# ── Base class ────────────────────────────────────────────────────────────────


class _BaseShotQualityModel:
    feature_set: CxAFeatureSetSpec
    _numeric_all: list[str]
    _cat_cols: list[str]
    _bool_set: frozenset[str]
    pipeline: Pipeline | None

    def _resolve_cols(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        numeric_all = [c for c in self.feature_set.numeric_all if c in df.columns]
        cat_cols = [c for c in self.feature_set.categorical if c in df.columns]
        return numeric_all, cat_cols

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        return _make_x(df, self._numeric_all, self._cat_cols, self._bool_set)

    def predict(self, actions_df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return np.clip(self.pipeline.predict(self._x(actions_df)), 1e-6, None)

    def evaluate(
        self, actions_df: pd.DataFrame, target_col: str = "resulting_shot_cxg"
    ) -> ShotQualityMetrics:
        y = actions_df[target_col].astype(float).to_numpy()
        p = self.predict(actions_df)
        return _evaluate_regression(y, p)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Gamma GLM ─────────────────────────────────────────────────────────────────


class GammaShotQualityModel(_BaseShotQualityModel):
    """
    Gamma GLM with log link for resulting shot CxG prediction.

    Gamma family is appropriate because CxG is strictly positive and
    right-skewed (most shots have low probability, a few are very high).
    """

    def __init__(
        self,
        feature_set: str | CxAFeatureSetSpec = "contextual",
        alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.feature_set = (
            get_feature_set(feature_set) if isinstance(feature_set, str) else feature_set
        )
        self.alpha = alpha
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self._numeric_all: list[str] = []
        self._cat_cols: list[str] = []
        self._bool_set: frozenset[str] = frozenset()

    def fit(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "resulting_shot_cxg",
    ) -> GammaShotQualityModel:
        if actions_df.empty:
            raise ValueError("actions_df is empty")
        if target_col not in actions_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")
        y = actions_df[target_col].astype(float)
        if (y <= 0).any():
            raise ValueError(
                "Gamma GLM requires strictly positive target values. "
                "Filter out rows with resulting_shot_cxg <= 0 before fitting."
            )
        numeric_all, cat_cols = self._resolve_cols(actions_df)
        if not numeric_all:
            raise ValueError("No numeric feature columns found for this feature set")
        self._numeric_all = numeric_all
        self._cat_cols = cat_cols
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)
        self.pipeline = _build_gamma_pipeline(numeric_all, cat_cols, self.alpha)
        self.pipeline.fit(self._x(actions_df), y.to_numpy())
        return self


# ── XGBoost regressor ─────────────────────────────────────────────────────────


class XGBoostShotQualityModel(_BaseShotQualityModel):
    """XGBoost regressor for resulting shot CxG."""

    def __init__(
        self,
        feature_set: str | CxAFeatureSetSpec = "contextual",
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise ImportError("xgboost not installed. Run: poetry install --with models") from exc
        self.feature_set = (
            get_feature_set(feature_set) if isinstance(feature_set, str) else feature_set
        )
        self.params = dict(
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
        self._numeric_all: list[str] = []
        self._cat_cols: list[str] = []
        self._bool_set: frozenset[str] = frozenset()

    def _make_estimator(self, params: dict):
        import xgboost as xgb

        from src.runtime.gbm_device import xgboost_kwargs

        return xgb.XGBRegressor(
            **params,
            **xgboost_kwargs(getattr(self, "device", None)),
            objective="reg:squarederror",
            verbosity=0,
            random_state=self.random_state,
        )

    def fit(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "resulting_shot_cxg",
    ) -> XGBoostShotQualityModel:
        if actions_df.empty:
            raise ValueError("actions_df is empty")
        if target_col not in actions_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")
        df = actions_df.reset_index(drop=True)
        numeric_all, cat_cols = self._resolve_cols(df)
        if not numeric_all:
            raise ValueError("No feature columns found")
        self._numeric_all = numeric_all
        self._cat_cols = cat_cols
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)
        self.pipeline = _build_tree_reg_pipeline(
            self._make_estimator(self.params), numeric_all, cat_cols
        )
        self.pipeline.fit(self._x(df), df[target_col].astype(float).to_numpy())
        return self


# ── LightGBM regressor ────────────────────────────────────────────────────────


class LightGBMShotQualityModel(_BaseShotQualityModel):
    """LightGBM regressor for resulting shot CxG."""

    def __init__(
        self,
        feature_set: str | CxAFeatureSetSpec = "contextual",
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        random_state: int = 42,
    ) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError as exc:
            raise ImportError("lightgbm not installed. Run: poetry install --with models") from exc
        self.feature_set = (
            get_feature_set(feature_set) if isinstance(feature_set, str) else feature_set
        )
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
        )
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self._numeric_all: list[str] = []
        self._cat_cols: list[str] = []
        self._bool_set: frozenset[str] = frozenset()

    def _make_estimator(self, params: dict):
        import lightgbm as lgb

        from src.runtime.gbm_device import lightgbm_kwargs

        return lgb.LGBMRegressor(
            **params,
            **lightgbm_kwargs(getattr(self, "device", None)),
            objective="regression",
            metric="rmse",
            verbose=-1,
            random_state=self.random_state,
        )

    def fit(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "resulting_shot_cxg",
    ) -> LightGBMShotQualityModel:
        if actions_df.empty:
            raise ValueError("actions_df is empty")
        if target_col not in actions_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")
        df = actions_df.reset_index(drop=True)
        numeric_all, cat_cols = self._resolve_cols(df)
        if not numeric_all:
            raise ValueError("No feature columns found")
        self._numeric_all = numeric_all
        self._cat_cols = cat_cols
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)
        self.pipeline = _build_tree_reg_pipeline(
            self._make_estimator(self.params), numeric_all, cat_cols
        )
        self.pipeline.fit(self._x(df), df[target_col].astype(float).to_numpy())
        return self


# ── MLP regressor ─────────────────────────────────────────────────────────────


class MLPShotQualityModel(_BaseShotQualityModel):
    """
    Feed-forward neural MLP regressor with Huber loss.

    Architecture: Linear(d_in → 256) → ReLU → Dropout(0.1) →
                  Linear(256 → 128) → ReLU → Dropout(0.1) → Linear(128 → 1) → Softplus
    Loss: HuberLoss(delta=0.1) — robust to outlier high-CxG shots.
    """

    def __init__(
        self,
        feature_set: str | CxAFeatureSetSpec = "contextual",
        hidden_dims: tuple[int, ...] = (256, 128),
        lr: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int | None = None,
        huber_delta: float = 0.1,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.feature_set = (
            get_feature_set(feature_set) if isinstance(feature_set, str) else feature_set
        )
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.huber_delta = huber_delta
        self.device = device
        self._resolved_device: str | None = None
        self.random_state = random_state
        self.pipeline: Pipeline | None = None  # scaler
        self._torch_model = None
        self._numeric_all: list[str] = []
        self._cat_cols: list[str] = []
        self._bool_set: frozenset[str] = frozenset()

    def _torch_device(self) -> str:
        if self._resolved_device is None:
            from src.models.neural import resolve_device

            self._resolved_device = resolve_device(self.device)
            logger.info("MLPShotQuality: using torch device %s", self._resolved_device)
        return self._resolved_device

    def _build_torch_model(self, in_dim: int):
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("PyTorch not installed.") from exc

        layers: list = []
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Softplus()]  # ensures positive output

        class _MLP(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return _MLP(layers)

    def fit(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "resulting_shot_cxg",
    ) -> MLPShotQualityModel:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError("PyTorch not installed. Run: pip install torch") from exc

        if actions_df.empty:
            raise ValueError("actions_df is empty")
        if target_col not in actions_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")

        torch.manual_seed(self.random_state)
        df = actions_df.reset_index(drop=True)
        numeric_all, cat_cols = self._resolve_cols(df)
        if not numeric_all:
            raise ValueError("No feature columns found")
        self._numeric_all = numeric_all
        self._cat_cols = []  # MLP uses numeric only
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)

        # Fit scaler
        self.pipeline = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]
        )
        X_raw = _make_x(df, self._numeric_all, [], self._bool_set)[self._numeric_all]
        X_np = self.pipeline.fit_transform(X_raw).astype(np.float32)
        y_np = df[target_col].astype(float).to_numpy(dtype=np.float32)

        X_t = torch.tensor(X_np)
        y_t = torch.tensor(y_np)
        dataset = TensorDataset(X_t, y_t)
        from src.models.neural import resolve_batch_size

        bs = resolve_batch_size("ffnn", self.batch_size)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)

        device = self._torch_device()
        model = self._build_torch_model(X_np.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=self.huber_delta)
        model.train()

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                avg = epoch_loss / max(len(loader), 1)
                logger.info("MLPShotQuality epoch %d/%d loss=%.4f", epoch + 1, self.max_epochs, avg)

        model.eval()
        self._torch_model = model
        return self

    def predict(self, actions_df: pd.DataFrame) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch not installed") from exc
        if self._torch_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        df = actions_df.reset_index(drop=True)
        X_raw = _make_x(df, self._numeric_all, [], self._bool_set)[self._numeric_all]
        X_np = self.pipeline.transform(X_raw).astype(np.float32)
        device = self._torch_device()
        self._torch_model.eval()
        with torch.no_grad():
            out = self._torch_model(torch.tensor(X_np).to(device)).cpu().numpy()
        return np.clip(out, 1e-6, None)


# ── Shot-Quality Ladder ───────────────────────────────────────────────────────


def _cv_shot_quality(
    factory: Callable[[], _BaseShotQualityModel],
    actions_df: pd.DataFrame,
    target_col: str,
    match_id_col: str,
    n_folds: int,
    random_state: int,
) -> tuple[float, float, float | None, int]:
    df = actions_df.reset_index(drop=True)
    folds = (
        list(
            match_kfold(df, n_splits=n_folds, match_id_col=match_id_col, random_state=random_state)
        )
        if match_id_col in df.columns
        else list(
            __import__("sklearn.model_selection", fromlist=["KFold"])
            .KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            .split(df)
        )
    )
    maes, rmses, spearmans = [], [], []
    for tr_idx, va_idx in folds:
        tr_df, va_df = df.loc[tr_idx], df.loc[va_idx]
        if len(tr_df) < 10:
            continue
        try:
            m = factory()
            m.fit(tr_df, target_col)
            p = m.predict(va_df)
            y = va_df[target_col].astype(float).to_numpy()
            maes.append(float(mean_absolute_error(y, p)))
            rmses.append(float(np.sqrt(mean_squared_error(y, p))))
            corr, _ = spearmanr(y, p)
            if not np.isnan(corr):
                spearmans.append(float(corr))
        except Exception as exc:
            logger.warning("ShotQuality CV fold failed: %s", exc)
    if not maes:
        return float("inf"), float("inf"), None, 0
    return (
        float(np.mean(maes)),
        float(np.mean(rmses)),
        float(np.mean(spearmans)) if spearmans else None,
        len(maes),
    )


class ShotQualityLadder:
    """Trains and ranks all shot-quality regression model candidates."""

    def __init__(self) -> None:
        self._results: list[ShotQualityLadderResult] = []

    def run(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "resulting_shot_cxg",
        match_id_col: str = "match_id",
        n_folds: int = 5,
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> list[ShotQualityLadderResult]:
        if actions_df.empty:
            raise ValueError("actions_df is empty")
        if target_col not in actions_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")

        # Gamma GLM requires positive targets — filter once
        pos_df = actions_df[actions_df[target_col] > 0].copy()

        ne, rs = n_estimators, random_state
        candidates: list[tuple[str, str, str, Callable, pd.DataFrame]] = [
            (
                "gamma_glm",
                "glm",
                "contextual",
                lambda: GammaShotQualityModel(feature_set="contextual", random_state=rs),
                pos_df,
            ),
            (
                "xgb_contextual",
                "xgboost",
                "contextual",
                lambda: XGBoostShotQualityModel(
                    feature_set="contextual", n_estimators=ne, random_state=rs
                ),
                actions_df,
            ),
            (
                "lgbm_contextual",
                "lightgbm",
                "contextual",
                lambda: LightGBMShotQualityModel(
                    feature_set="contextual", n_estimators=ne, random_state=rs
                ),
                actions_df,
            ),
        ]

        results: list[ShotQualityLadderResult] = []
        for name, family, fset, factory, df_for_fit in candidates:
            logger.info("ShotQualityLadder: evaluating %s …", name)
            try:
                factory()  # probe availability
            except ImportError as exc:
                logger.warning("ShotQualityLadder: skipping %s — %s", name, exc)
                continue
            cv_mae, cv_rmse, cv_sp, n_valid = _cv_shot_quality(
                factory, df_for_fit, target_col, match_id_col, n_folds, random_state
            )
            final = factory()
            final.fit(df_for_fit, target_col)
            results.append(
                ShotQualityLadderResult(
                    name=name,
                    family=family,
                    feature_set=fset,
                    cv_mae=cv_mae,
                    cv_rmse=cv_rmse,
                    cv_spearman=cv_sp,
                    n_cv_folds_used=n_valid,
                    model=final,
                )
            )

        results.sort(key=lambda r: r.cv_mae)
        for i, r in enumerate(results):
            r.rank = i + 1
        self._results = results
        return results

    def leaderboard(self) -> pd.DataFrame:
        if not self._results:
            raise RuntimeError("No results yet. Call run() first.")
        rows = [
            {
                "rank": r.rank,
                "name": r.name,
                "family": r.family,
                "feature_set": r.feature_set,
                "cv_mae": round(r.cv_mae, 5),
                "cv_rmse": round(r.cv_rmse, 5),
                "cv_spearman": round(r.cv_spearman, 4) if r.cv_spearman is not None else None,
            }
            for r in self._results
        ]
        return pd.DataFrame(rows).set_index("rank")

    def best(self) -> ShotQualityLadderResult:
        if not self._results:
            raise RuntimeError("No results yet. Call run() first.")
        return self._results[0]


# ── Factory ───────────────────────────────────────────────────────────────────


class ShotQualityModel(_BaseShotQualityModel):
    """
    Family-dispatching factory for shot-quality regression models.

    Used by train_cxa.py::

        model = ShotQualityModel(family="lgbm", feature_set="contextual", n_estimators=300)
        model.fit(actions_df, target_col="resulting_shot_cxg")
    """

    def __init__(
        self,
        family: str = "lgbm",
        feature_set: str = "contextual",
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> None:
        _family = family.lower()
        if _family in {"lgbm", "lightgbm"}:
            self._delegate = LightGBMShotQualityModel(
                feature_set=feature_set,
                n_estimators=n_estimators,
                random_state=random_state,
            )
        elif _family in {"xgboost", "xgb"}:
            self._delegate = XGBoostShotQualityModel(
                feature_set=feature_set,
                n_estimators=n_estimators,
                random_state=random_state,
            )
        elif _family in {"glm", "gamma"}:
            self._delegate = GammaShotQualityModel(
                feature_set=feature_set,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown family {family!r}. Choose from: lgbm, xgboost, glm.")
        self.feature_set = self._delegate.feature_set
        self._numeric_all = self._delegate._numeric_all
        self._cat_cols = self._delegate._cat_cols
        self._bool_set = self._delegate._bool_set
        self.pipeline = self._delegate.pipeline

    def fit(
        self, actions_df: pd.DataFrame, target_col: str = "resulting_shot_cxg", **kwargs
    ) -> ShotQualityModel:
        self._delegate.fit(actions_df, target_col, **kwargs)
        self._numeric_all = self._delegate._numeric_all
        self._cat_cols = self._delegate._cat_cols
        self._bool_set = self._delegate._bool_set
        self.pipeline = self._delegate.pipeline
        return self

    def predict(self, actions_df: pd.DataFrame) -> np.ndarray:
        return self._delegate.predict(actions_df)

    def evaluate(
        self, actions_df: pd.DataFrame, target_col: str = "resulting_shot_cxg"
    ) -> ShotQualityMetrics:
        return self._delegate.evaluate(actions_df, target_col)
