"""
Shot-creation probability model — Stage 1 of the CxA two-stage pipeline.

Target: shot_created = 1 if a shot is taken within the same possession
after this action. Window variants (within_5_actions, within_10s,
within_15s, same_possession) can be evaluated via stability analysis.

Unit of analysis: every pass, cross, carry, and cutback.

Model ladder (in order of complexity):
  1. baseline_logit      (traditional features, logistic regression — reuses Phase 4)
  2. glm_contextual      (contextual features, logistic regression)
  3. xgb_traditional     (traditional features, XGBoost)
  4. xgb_contextual      (contextual features, XGBoost)
  5. lgbm_traditional    (traditional features, LightGBM)
  6. lgbm_contextual     (contextual features, LightGBM)
  7. transformer_contextual  (Transformer encoder + MLP, contextual + sequence)
  8. xgb_full_360 / lgbm_full_360  (when include_360=True)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.evaluation.validation_splits import match_kfold
from src.models.cxa.feature_sets import CxAFeatureSetSpec, get_feature_set

logger = logging.getLogger(__name__)

# Window variants used for stability analysis
WINDOW_VARIANTS: tuple[str, ...] = (
    "shot_created",           # same possession (primary)
    "shot_within_5_actions",  # within 5 actions
    "shot_within_10s",        # within 10 seconds
    "shot_within_15s",        # within 15 seconds
)


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class ShotCreationMetrics:
    log_loss: float
    brier: float
    auc: float | None
    pr_auc: float | None
    window: str = "shot_created"


@dataclass
class WindowStabilityReport:
    """Cross-window stability of classifier metrics."""
    window_metrics: dict[str, ShotCreationMetrics] = field(default_factory=dict)

    def best_window(self) -> str:
        """Return the window variant with the lowest log-loss."""
        return min(self.window_metrics, key=lambda k: self.window_metrics[k].log_loss)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for w, m in self.window_metrics.items():
            rows.append({
                "window": w,
                "log_loss": m.log_loss,
                "brier": m.brier,
                "auc": m.auc,
                "pr_auc": m.pr_auc,
            })
        return pd.DataFrame(rows).set_index("window")


# ── Shared pipeline builders ───────────────────────────────────────────────────

def _make_X(
    df: pd.DataFrame,
    numeric_all: list[str],
    cat_cols: list[str],
    bool_set: frozenset[str],
) -> pd.DataFrame:
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


def _build_logistic_pipeline(
    C: float,
    numeric_all: list[str],
    cat_cols: list[str],
    random_state: int,
) -> Pipeline:
    transformers: list = [
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), numeric_all),
    ]
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", __import__("sklearn.preprocessing", fromlist=["OneHotEncoder"]).OneHotEncoder(
                handle_unknown="ignore", sparse_output=False,
            )),
        ]), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("clf", LogisticRegression(
        solver="lbfgs", max_iter=2000, C=C, random_state=random_state,
    ))])


def _build_tree_pipeline(
    estimator,
    numeric_all: list[str],
    cat_cols: list[str],
) -> Pipeline:
    from sklearn.preprocessing import OrdinalEncoder
    transformers: list = [("num", SimpleImputer(strategy="median"), numeric_all)]
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=float)),
        ]), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("clf", estimator)])


# ── Base class ────────────────────────────────────────────────────────────────

class _BaseShotCreationModel:
    """Shared persistence, column resolution and X-building logic."""

    feature_set: CxAFeatureSetSpec
    _numeric_all: list[str]
    _cat_cols: list[str]
    _bool_set: frozenset[str]
    pipeline: Pipeline | None

    def _resolve_cols(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        numeric_all = [c for c in self.feature_set.numeric_all if c in df.columns]
        cat_cols = [c for c in self.feature_set.categorical if c in df.columns]
        return numeric_all, cat_cols

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return _make_X(df, self._numeric_all, self._cat_cols, self._bool_set)

    def predict_proba(self, actions_df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict_proba(self._X(actions_df))[:, 1]

    def evaluate(
        self, actions_df: pd.DataFrame, target_col: str = "shot_created"
    ) -> ShotCreationMetrics:
        y = actions_df[target_col].astype(int).to_numpy()
        p = self.predict_proba(actions_df)
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
        pr_auc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None
        return ShotCreationMetrics(
            log_loss=float(log_loss(y, p, labels=[0, 1])),
            brier=float(brier_score_loss(y, p)),
            auc=auc,
            pr_auc=pr_auc,
            window=target_col,
        )

    def stability_analysis(
        self, actions_df: pd.DataFrame
    ) -> WindowStabilityReport:
        """Evaluate on each available window variant column."""
        report = WindowStabilityReport()
        for w in WINDOW_VARIANTS:
            if w in actions_df.columns:
                report.window_metrics[w] = self.evaluate(actions_df, target_col=w)
        return report

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj


# ── GLM (logistic contextual) ─────────────────────────────────────────────────

class GlmShotCreationModel(_BaseShotCreationModel):
    """
    Logistic regression shot-creation model.

    Parameters
    ----------
    feature_set : str | CxAFeatureSetSpec
    C           : inverse regularisation (tunable)
    """

    def __init__(
        self,
        feature_set: str | CxAFeatureSetSpec = "contextual",
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.feature_set = (
            get_feature_set(feature_set) if isinstance(feature_set, str) else feature_set
        )
        self.C = C
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self._numeric_all: list[str] = []
        self._cat_cols: list[str] = []
        self._bool_set: frozenset[str] = frozenset()

    def fit(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "shot_created",
    ) -> "GlmShotCreationModel":
        if actions_df.empty:
            raise ValueError("actions_df is empty")
        if target_col not in actions_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")
        numeric_all, cat_cols = self._resolve_cols(actions_df)
        if not numeric_all:
            raise ValueError("No numeric feature columns found for this feature set")
        self._numeric_all = numeric_all
        self._cat_cols = cat_cols
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)
        self.pipeline = _build_logistic_pipeline(self.C, numeric_all, cat_cols, self.random_state)
        self.pipeline.fit(self._X(actions_df), actions_df[target_col].astype(int).to_numpy())
        return self


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostShotCreationModel(_BaseShotCreationModel):
    """XGBoost shot-creation binary classifier."""

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
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, subsample=subsample,
            colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        )
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self._numeric_all: list[str] = []
        self._cat_cols: list[str] = []
        self._bool_set: frozenset[str] = frozenset()

    def _make_estimator(self, params: dict):
        import xgboost as xgb
        from src.runtime.gbm_device import xgboost_kwargs
        return xgb.XGBClassifier(
            **params,
            **xgboost_kwargs(getattr(self, "device", None)),
            objective="binary:logistic", eval_metric="logloss",
            verbosity=0, random_state=self.random_state,
        )

    def fit(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "shot_created",
        n_trials: int = 0,
        match_id_col: str = "match_id",
    ) -> "XGBoostShotCreationModel":
        if actions_df.empty:
            raise ValueError("actions_df is empty")
        if target_col not in actions_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")
        df = actions_df.reset_index(drop=True)
        numeric_all, cat_cols = self._resolve_cols(df)
        if not numeric_all:
            raise ValueError("No feature columns found for this feature set")
        self._numeric_all = numeric_all
        self._cat_cols = cat_cols
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)
        if n_trials > 0 and match_id_col in df.columns:
            self._tune(df, target_col, n_trials, match_id_col)
        self.pipeline = _build_tree_pipeline(self._make_estimator(self.params), numeric_all, cat_cols)
        self.pipeline.fit(self._X(df), df[target_col].astype(int).to_numpy())
        return self

    def _tune(self, df: pd.DataFrame, target_col: str, n_trials: int, match_id_col: str) -> None:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        bool_set = frozenset(c for c in self.feature_set.boolean if c in self._numeric_all)
        X_all = _make_X(df, self._numeric_all, self._cat_cols, bool_set)
        y_all = df[target_col].astype(int).to_numpy()
        folds = list(match_kfold(df, n_splits=3, match_id_col=match_id_col))

        def objective(trial) -> float:
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 600),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 3.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0),
            )
            scores = []
            for tr_idx, va_idx in folds:
                pipe = _build_tree_pipeline(self._make_estimator(params), self._numeric_all, self._cat_cols)
                pipe.fit(X_all.loc[tr_idx], y_all[tr_idx])
                p = pipe.predict_proba(X_all.loc[va_idx])[:, 1]
                y_va = y_all[va_idx]
                if len(np.unique(y_va)) > 1:
                    scores.append(log_loss(y_va, p, labels=[0, 1]))
            return float(np.mean(scores)) if scores else 1.0

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.params.update(study.best_params)


# ── LightGBM ──────────────────────────────────────────────────────────────────

class LightGBMShotCreationModel(_BaseShotCreationModel):
    """LightGBM shot-creation binary classifier."""

    def __init__(
        self,
        feature_set: str | CxAFeatureSetSpec = "contextual",
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
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
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, subsample=subsample,
            colsample_bytree=colsample_bytree, min_child_samples=min_child_samples,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        )
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self._numeric_all: list[str] = []
        self._cat_cols: list[str] = []
        self._bool_set: frozenset[str] = frozenset()

    def _make_estimator(self, params: dict):
        import lightgbm as lgb
        from src.runtime.gbm_device import lightgbm_kwargs
        return lgb.LGBMClassifier(
            **params,
            **lightgbm_kwargs(getattr(self, "device", None)),
            objective="binary", metric="binary_logloss",
            verbose=-1, random_state=self.random_state,
        )

    def fit(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "shot_created",
        n_trials: int = 0,
        match_id_col: str = "match_id",
    ) -> "LightGBMShotCreationModel":
        if actions_df.empty:
            raise ValueError("actions_df is empty")
        if target_col not in actions_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")
        df = actions_df.reset_index(drop=True)
        numeric_all, cat_cols = self._resolve_cols(df)
        if not numeric_all:
            raise ValueError("No feature columns found for this feature set")
        self._numeric_all = numeric_all
        self._cat_cols = cat_cols
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)
        self.pipeline = _build_tree_pipeline(self._make_estimator(self.params), numeric_all, cat_cols)
        self.pipeline.fit(self._X(df), df[target_col].astype(int).to_numpy())
        return self


# ── Sequence-aware Transformer model ─────────────────────────────────────────

class TransformerShotCreationModel(_BaseShotCreationModel):
    """
    Transformer encoder over the event sequence preceding this action,
    with a CLS token, concatenated with tabular contextual features and
    passed through a 2-layer MLP for shot-creation probability.

    Architecture
    ------------
    Sequence input  : (B, T, d_token) where d_token = 5
                      [event_type_emb, x_norm, y_norm, under_pressure, body_part_emb]
    Transformer     : n_heads=4, n_layers=2, d_model=64, max_seq_len=15
    CLS token       : prepended to each sequence
    Tabular input   : contextual numeric_all features → Linear → ReLU
    Combined        : concat(cls_output, tabular_enc) → MLP(256, 128) → sigmoid
    Loss            : BCEWithLogitsLoss

    Requires PyTorch. Falls back gracefully to XGBoost contextual if torch
    is not available.
    """

    MAX_SEQ_LEN = 15
    # Sequence token fields (per event step)
    SEQ_NUMERIC = ("seq_x", "seq_y", "seq_under_pressure")
    SEQ_EVENT_TYPES = (
        "pass", "carry", "shot", "pressure", "dribble",
        "ball_receipt", "clearance", "interception", "other",
    )
    SEQ_BODY_PARTS = ("foot", "head", "other")

    def __init__(
        self,
        feature_set: str | CxAFeatureSetSpec = "contextual",
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_hidden: int = 128,
        lr: float = 1e-3,
        max_epochs: int = 30,
        batch_size: int = 256,
        random_state: int = 42,
    ) -> None:
        self.feature_set = (
            get_feature_set(feature_set) if isinstance(feature_set, str) else feature_set
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_hidden = mlp_hidden
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.pipeline: Pipeline | None = None      # tabular scaler only
        self._numeric_all: list[str] = []
        self._cat_cols: list[str] = []
        self._bool_set: frozenset[str] = frozenset()
        self._torch_model = None
        self._tabular_dim: int = 0

    def _build_torch_model(self, tabular_dim: int):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("PyTorch not installed. Run: pip install torch") from exc

        n_event_types = len(self.SEQ_EVENT_TYPES)
        n_body_parts = len(self.SEQ_BODY_PARTS)
        d_model = self.d_model

        class _ShotCreationTransformer(nn.Module):
            def __init__(self, n_et, n_bp, d_model, n_heads, n_layers, mlp_hidden, tab_dim):
                super().__init__()
                # +1 for unknown
                self.event_emb = nn.Embedding(n_et + 1, d_model // 4)
                self.body_emb = nn.Embedding(n_bp + 1, d_model // 4)
                token_dim = d_model // 4 * 2 + 3  # event_emb + body_emb + 3 numeric
                self.token_proj = nn.Linear(token_dim, d_model)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                    dropout=0.1, batch_first=True, norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
                self.tab_proj = nn.Sequential(
                    nn.Linear(tab_dim, d_model), nn.ReLU(), nn.Dropout(0.1),
                )
                combined_dim = d_model * 2
                self.head = nn.Sequential(
                    nn.Linear(combined_dim, mlp_hidden), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(mlp_hidden, mlp_hidden // 2), nn.ReLU(),
                    nn.Linear(mlp_hidden // 2, 1),
                )
                nn.init.zeros_(self.cls_token)

            def forward(self, seq_et, seq_bp, seq_num, seq_mask, tab):
                # seq_et: (B, T)  seq_bp: (B, T)  seq_num: (B, T, 3)  tab: (B, tab_dim)
                et_emb = self.event_emb(seq_et)    # (B, T, d/4)
                bp_emb = self.body_emb(seq_bp)     # (B, T, d/4)
                tok = self.token_proj(torch.cat([et_emb, bp_emb, seq_num], dim=-1))  # (B,T,d)
                B = tok.size(0)
                cls = self.cls_token.expand(B, -1, -1)
                seq = torch.cat([cls, tok], dim=1)  # (B, T+1, d)
                # Extend mask: CLS token is never masked
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=seq_mask.device)
                full_mask = torch.cat([cls_mask, seq_mask], dim=1)  # (B, T+1)
                out = self.transformer(seq, src_key_padding_mask=full_mask)
                cls_out = out[:, 0, :]             # (B, d)
                tab_enc = self.tab_proj(tab)       # (B, d)
                combined = torch.cat([cls_out, tab_enc], dim=-1)
                return self.head(combined).squeeze(-1)

        return _ShotCreationTransformer(
            n_et=n_event_types,
            n_bp=n_body_parts,
            d_model=d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            mlp_hidden=self.mlp_hidden,
            tab_dim=tabular_dim,
        )

    def _build_tabular_scaler(self, df: pd.DataFrame) -> Pipeline:
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ])
        X_tab = _make_X(df, self._numeric_all, [], self._bool_set)[self._numeric_all]
        pipe.fit(X_tab)
        return pipe

    def _encode_sequences(
        self, actions_df: pd.DataFrame
    ) -> tuple:
        """
        Build padded sequence tensors from per-action sequence columns.

        Expects columns: seq_event_types_{0..T-1}, seq_body_parts_{0..T-1},
        seq_x_{0..T-1}, seq_y_{0..T-1}, seq_up_{0..T-1}.
        Falls back to zeros if absent.
        """
        import torch
        T = self.MAX_SEQ_LEN
        B = len(actions_df)
        et_idx = {v: i for i, v in enumerate(self.SEQ_EVENT_TYPES)}
        bp_idx = {v: i for i, v in enumerate(self.SEQ_BODY_PARTS)}
        n_et = len(self.SEQ_EVENT_TYPES)
        n_bp = len(self.SEQ_BODY_PARTS)

        seq_et = torch.full((B, T), n_et, dtype=torch.long)    # unknown = last idx
        seq_bp = torch.full((B, T), n_bp, dtype=torch.long)
        seq_num = torch.zeros(B, T, 3)
        seq_mask = torch.ones(B, T, dtype=torch.bool)           # True = pad

        for t in range(T):
            et_col = f"seq_event_type_{t}"
            bp_col = f"seq_body_part_{t}"
            x_col = f"seq_x_{t}"
            y_col = f"seq_y_{t}"
            up_col = f"seq_under_pressure_{t}"
            has_step = et_col in actions_df.columns

            if has_step:
                et_vals = actions_df[et_col].fillna("other").astype(str).map(
                    lambda v: et_idx.get(v, n_et)
                ).to_numpy()
                bp_vals = actions_df[bp_col].fillna("other").astype(str).map(
                    lambda v: bp_idx.get(v, n_bp)
                ).to_numpy() if bp_col in actions_df.columns else np.full(B, n_bp)
                x_vals = pd.to_numeric(actions_df.get(x_col, pd.Series(0.0, index=actions_df.index)), errors="coerce").fillna(0.0).to_numpy() / 105.0
                y_vals = pd.to_numeric(actions_df.get(y_col, pd.Series(0.0, index=actions_df.index)), errors="coerce").fillna(0.0).to_numpy() / 68.0
                up_vals = pd.to_numeric(actions_df.get(up_col, pd.Series(0.0, index=actions_df.index)), errors="coerce").fillna(0.0).to_numpy()
                seq_et[:, t] = torch.tensor(et_vals, dtype=torch.long)
                seq_bp[:, t] = torch.tensor(bp_vals, dtype=torch.long)
                seq_num[:, t, 0] = torch.tensor(x_vals, dtype=torch.float32)
                seq_num[:, t, 1] = torch.tensor(y_vals, dtype=torch.float32)
                seq_num[:, t, 2] = torch.tensor(up_vals, dtype=torch.float32)
                # Unmask steps where event type is present and valid
                valid = actions_df[et_col].notna().to_numpy()
                seq_mask[torch.tensor(np.where(valid)[0]), t] = False

        return seq_et, seq_bp, seq_num, seq_mask

    def fit(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "shot_created",
    ) -> "TransformerShotCreationModel":
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

        df = actions_df.reset_index(drop=True)
        torch.manual_seed(self.random_state)

        numeric_all, cat_cols = self._resolve_cols(df)
        # Transformer uses numeric only (cats would need embeddings — handled via ordinal below)
        self._numeric_all = numeric_all
        self._cat_cols = []   # tabular side uses numeric_all only for simplicity
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in numeric_all)

        # Build tabular scaler (numeric_all only)
        self.pipeline = self._build_tabular_scaler(df)
        self._tabular_dim = len(self._numeric_all)

        # Build and encode data
        X_tab_raw = _make_X(df, self._numeric_all, [], self._bool_set)[self._numeric_all]
        X_tab = torch.tensor(self.pipeline.transform(X_tab_raw), dtype=torch.float32)
        seq_et, seq_bp, seq_num, seq_mask = self._encode_sequences(df)
        y = torch.tensor(df[target_col].astype(float).to_numpy(), dtype=torch.float32)

        dataset = TensorDataset(seq_et, seq_bp, seq_num, seq_mask, X_tab, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = self._build_torch_model(self._tabular_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        model.train()

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for batch in loader:
                b_et, b_bp, b_num, b_mask, b_tab, b_y = batch
                optimizer.zero_grad()
                logits = model(b_et, b_bp, b_num, b_mask, b_tab)
                loss = criterion(logits, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / max(len(loader), 1)
            if (epoch + 1) % 5 == 0:
                logger.info("TransformerShotCreation epoch %d/%d loss=%.4f", epoch + 1, self.max_epochs, avg)

        model.eval()
        self._torch_model = model
        return self

    def predict_proba(self, actions_df: pd.DataFrame) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch not installed") from exc
        if self._torch_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        df = actions_df.reset_index(drop=True)
        X_tab_raw = _make_X(df, self._numeric_all, [], self._bool_set)[self._numeric_all]
        X_tab = torch.tensor(self.pipeline.transform(X_tab_raw), dtype=torch.float32)
        seq_et, seq_bp, seq_num, seq_mask = self._encode_sequences(df)
        self._torch_model.eval()
        with torch.no_grad():
            logits = self._torch_model(seq_et, seq_bp, seq_num, seq_mask, X_tab)
        return torch.sigmoid(logits).numpy()

    def save(self, path: str | Path) -> None:
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Save the whole object; torch model stored within
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "TransformerShotCreationModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj


# ── Shot-Creation Ladder ──────────────────────────────────────────────────────

@dataclass
class ShotCreationLadderResult:
    name: str
    family: str
    feature_set: str
    cv_log_loss: float
    cv_brier: float
    cv_auc: float | None
    cv_pr_auc: float | None
    n_cv_folds_used: int
    model: object = field(repr=False)
    rank: int = 0


def _cv_shot_creation(
    factory: Callable[[], _BaseShotCreationModel],
    actions_df: pd.DataFrame,
    target_col: str,
    match_id_col: str,
    n_folds: int,
    random_state: int,
) -> tuple[float, float, float | None, float | None, int]:
    from statistics import mean
    df = actions_df.reset_index(drop=True)
    folds = list(match_kfold(df, n_splits=n_folds, match_id_col=match_id_col, random_state=random_state)) \
        if match_id_col in df.columns else \
        list(__import__("sklearn.model_selection", fromlist=["KFold"]).KFold(
            n_splits=n_folds, shuffle=True, random_state=random_state).split(df))
    lls, briers, aucs, pr_aucs = [], [], [], []
    for tr_idx, va_idx in folds:
        tr_df, va_df = df.loc[tr_idx], df.loc[va_idx]
        if len(tr_df) < 10 or tr_df[target_col].nunique() < 2:
            continue
        if va_df[target_col].nunique() < 2:
            continue
        m = factory()
        m.fit(tr_df, target_col)
        p = m.predict_proba(va_df)
        y = va_df[target_col].astype(int).to_numpy()
        lls.append(log_loss(y, p, labels=[0, 1]))
        briers.append(brier_score_loss(y, p))
        if len(np.unique(y)) > 1:
            aucs.append(roc_auc_score(y, p))
            pr_aucs.append(average_precision_score(y, p))
    if not lls:
        return float("inf"), float("inf"), None, None, 0
    return (
        float(np.mean(lls)), float(np.mean(briers)),
        float(np.mean(aucs)) if aucs else None,
        float(np.mean(pr_aucs)) if pr_aucs else None,
        len(lls),
    )


class ShotCreationLadder:
    """
    Trains and cross-validates all shot-creation model candidates.
    Mirrors the structure of CxGLadder (Phase 5).
    """

    def __init__(self) -> None:
        self._results: list[ShotCreationLadderResult] = []

    def run(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "shot_created",
        match_id_col: str = "match_id",
        n_folds: int = 5,
        n_estimators: int = 300,
        include_360: bool = False,
        include_transformer: bool = False,
        random_state: int = 42,
    ) -> list[ShotCreationLadderResult]:
        if actions_df.empty:
            raise ValueError("actions_df is empty")
        if target_col not in actions_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")

        ne = n_estimators
        rs = random_state
        candidates: list[tuple[str, str, str, Callable]] = [
            ("glm_traditional", "logistic", "traditional",
             lambda: GlmShotCreationModel(feature_set="traditional", random_state=rs)),
            ("glm_contextual", "logistic", "contextual",
             lambda: GlmShotCreationModel(feature_set="contextual", random_state=rs)),
            ("xgb_traditional", "xgboost", "traditional",
             lambda: XGBoostShotCreationModel(feature_set="traditional", n_estimators=ne, random_state=rs)),
            ("xgb_contextual", "xgboost", "contextual",
             lambda: XGBoostShotCreationModel(feature_set="contextual", n_estimators=ne, random_state=rs)),
            ("lgbm_traditional", "lightgbm", "traditional",
             lambda: LightGBMShotCreationModel(feature_set="traditional", n_estimators=ne, random_state=rs)),
            ("lgbm_contextual", "lightgbm", "contextual",
             lambda: LightGBMShotCreationModel(feature_set="contextual", n_estimators=ne, random_state=rs)),
        ]
        if include_360:
            candidates += [
                ("xgb_full_360", "xgboost", "full_360",
                 lambda: XGBoostShotCreationModel(feature_set="full_360", n_estimators=ne, random_state=rs)),
                ("lgbm_full_360", "lightgbm", "full_360",
                 lambda: LightGBMShotCreationModel(feature_set="full_360", n_estimators=ne, random_state=rs)),
            ]
        if include_transformer:
            candidates.append((
                "transformer_contextual", "transformer", "contextual",
                lambda: TransformerShotCreationModel(feature_set="contextual", random_state=rs),
            ))

        results: list[ShotCreationLadderResult] = []
        for name, family, fset, factory in candidates:
            logger.info("ShotCreationLadder: evaluating %s …", name)
            try:
                # Probe whether the model is importable before CV
                factory()
            except ImportError as exc:
                logger.warning("ShotCreationLadder: skipping %s — %s", name, exc)
                continue
            cv_ll, cv_b, cv_auc, cv_pr, n_valid = _cv_shot_creation(
                factory, actions_df, target_col, match_id_col, n_folds, random_state
            )
            final = factory()
            final.fit(actions_df, target_col)
            results.append(ShotCreationLadderResult(
                name=name, family=family, feature_set=fset,
                cv_log_loss=cv_ll, cv_brier=cv_b, cv_auc=cv_auc, cv_pr_auc=cv_pr,
                n_cv_folds_used=n_valid, model=final,
            ))

        results.sort(key=lambda r: r.cv_log_loss)
        for i, r in enumerate(results):
            r.rank = i + 1
        self._results = results
        return results

    def leaderboard(self) -> pd.DataFrame:
        if not self._results:
            raise RuntimeError("No results yet. Call run() first.")
        rows = [{
            "rank": r.rank, "name": r.name, "family": r.family,
            "feature_set": r.feature_set,
            "cv_log_loss": round(r.cv_log_loss, 5),
            "cv_brier": round(r.cv_brier, 5),
            "cv_auc": round(r.cv_auc, 4) if r.cv_auc is not None else None,
            "cv_pr_auc": round(r.cv_pr_auc, 4) if r.cv_pr_auc is not None else None,
        } for r in self._results]
        return pd.DataFrame(rows).set_index("rank")

    def best(self) -> ShotCreationLadderResult:
        if not self._results:
            raise RuntimeError("No results yet. Call run() first.")
        return self._results[0]


# ── Factory ───────────────────────────────────────────────────────────────────

class ShotCreationModel(_BaseShotCreationModel):
    """
    Family-dispatching factory for shot-creation models.

    Used by train_cxa.py::

        model = ShotCreationModel(family="lgbm", feature_set="contextual", n_estimators=300)
        model.fit(actions_df, target_col="shot_created")
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
            self._delegate = LightGBMShotCreationModel(
                feature_set=feature_set,
                n_estimators=n_estimators,
                random_state=random_state,
            )
        elif _family in {"xgboost", "xgb"}:
            self._delegate = XGBoostShotCreationModel(
                feature_set=feature_set,
                n_estimators=n_estimators,
                random_state=random_state,
            )
        elif _family in {"logistic", "glm"}:
            self._delegate = GlmShotCreationModel(
                feature_set=feature_set,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown family {family!r}. Choose from: lgbm, xgboost, logistic.")
        # Expose the same attributes so isinstance checks on _BaseShotCreationModel still work.
        self.feature_set = self._delegate.feature_set
        self._numeric_all = self._delegate._numeric_all
        self._cat_cols = self._delegate._cat_cols
        self._bool_set = self._delegate._bool_set
        self.pipeline = self._delegate.pipeline

    def fit(self, actions_df: pd.DataFrame, target_col: str = "shot_created", **kwargs) -> "ShotCreationModel":
        self._delegate.fit(actions_df, target_col, **kwargs)
        # Sync public attrs after fit
        self._numeric_all = self._delegate._numeric_all
        self._cat_cols = self._delegate._cat_cols
        self._bool_set = self._delegate._bool_set
        self.pipeline = self._delegate.pipeline
        return self

    def predict_proba(self, actions_df: pd.DataFrame) -> np.ndarray:
        return self._delegate.predict_proba(actions_df)

    def evaluate(self, actions_df: pd.DataFrame, target_col: str = "shot_created") -> ShotCreationMetrics:
        return self._delegate.evaluate(actions_df, target_col)
