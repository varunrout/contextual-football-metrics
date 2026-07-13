"""Sequence-of-players SetTransformer CxG model.

Treats the 360 freeze frame for each shot as an unordered set of player
tokens, encodes it with a small Transformer + CLS pooling, and fuses with
the contextual tabular feature block before producing P(goal).

Same sklearn-style ``fit``/``predict_proba``/``save``/``load`` API as the
LightGBM/XGBoost CxG models, so it slots straight into ``CxGLadder``.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.cxg.feature_sets import FeatureSetSpec, get_feature_set
from src.models.cxg.xgboost_model import _make_X
from src.models.neural import (
    TOKEN_DIM,
    TorchModelMixin,
    build_mlp_head,
    build_set_transformer_encoder,
    build_tabular_encoder,
    encode_frame_tokens,
    load_freeze_frames,
    require_torch,
    resolve_batch_size,
    select_event_id_column,
    shots_with_frames_count,
)

logger = logging.getLogger(__name__)


class SetTransformerCxGModel(TorchModelMixin):
    """SetTransformer-over-freeze-frames + contextual tabular fusion CxG model.

    Parameters
    ----------
    feature_set
        Tabular feature set ("contextual" recommended; "full_360" also works
        but its aggregated 360 columns become redundant with the raw set).
    frames_path
        Path to ``freeze_frames_360.parquet``. ``None`` resolves to the
        canonical ``data/processed/`` location.
    max_players
        Max tokens per shot (closest-to-ball wins on overflow).
    d_model, n_heads, n_layers, mlp_hidden
        Transformer + head widths.
    lr, max_epochs, batch_size, dropout
        Training knobs. ``batch_size=None`` reads from the active profile.
    device
        Torch device override; ``None`` resolves from active profile or CUDA
        availability.
    random_state
        Seed for ``torch.manual_seed`` and DataLoader shuffling.
    """

    def __init__(
        self,
        feature_set: str | FeatureSetSpec = "contextual",
        frames_path: str | Path | None = None,
        max_players: int = 22,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_hidden: int = 128,
        lr: float = 1e-3,
        max_epochs: int = 30,
        batch_size: int | None = None,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.feature_set = (
            get_feature_set(feature_set) if isinstance(feature_set, str) else feature_set
        )
        self.frames_path = Path(frames_path) if frames_path is not None else None
        self.max_players = max_players
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_hidden = mlp_hidden
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state

        # Set lazily during fit
        self.pipeline: Pipeline | None = None
        self._numeric_all: list[str] = []
        self._bool_set: frozenset[str] = frozenset()
        self._tabular_dim: int = 0
        self._torch_model: Any = None
        self._frames_cache: pd.DataFrame | None = None
        self._event_id_col: str | None = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resolve_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.feature_set.numeric_all if c in df.columns]

    def _frames(self) -> pd.DataFrame:
        if self._frames_cache is None:
            self._frames_cache = load_freeze_frames(self.frames_path)
            logger.info(
                "SetTransformerCxG: loaded %d freeze-frame rows",
                len(self._frames_cache),
            )
        return self._frames_cache

    def _build_tabular_pipeline(self, df: pd.DataFrame) -> Pipeline:
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ])
        X_tab = _make_X(df, self._numeric_all, [], self._bool_set)[self._numeric_all]
        pipe.fit(X_tab)
        return pipe

    def _build_torch_model(self, tabular_dim: int):
        torch, nn = require_torch()

        d = self.d_model
        set_encoder = build_set_transformer_encoder(
            token_dim=TOKEN_DIM,
            d_model=d,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )
        tab_encoder = build_tabular_encoder(tabular_dim, d, dropout=self.dropout)
        head = build_mlp_head(
            in_dim=2 * d,
            hidden_dims=(self.mlp_hidden, self.mlp_hidden // 2),
            out_dim=1,
            dropout=self.dropout,
            final_activation=None,  # logits → BCEWithLogitsLoss
        )

        class _SetTransformerCxG(nn.Module):
            def __init__(self):
                super().__init__()
                self.set_encoder = set_encoder
                self.tab_encoder = tab_encoder
                self.head = head

            def forward(self, tokens, mask, tab):
                set_out = self.set_encoder(tokens, mask)
                tab_out = self.tab_encoder(tab)
                fused = torch.cat([set_out, tab_out], dim=-1)
                return self.head(fused).squeeze(-1)

        return _SetTransformerCxG()

    def _encode_batch_inputs(self, df: pd.DataFrame):
        torch, _ = require_torch()
        if self._event_id_col is None:
            raise RuntimeError("event_id column not set — call fit() first.")
        tokens, mask = encode_frame_tokens(
            df, self._frames(),
            max_players=self.max_players,
            event_id_col=self._event_id_col,
            frames_event_id_col="event_internal_id",
        )
        if self.pipeline is None:
            raise RuntimeError("Tabular pipeline not fitted — call fit() first.")
        X_tab_raw = _make_X(df, self._numeric_all, [], self._bool_set)[self._numeric_all]
        X_tab = torch.tensor(self.pipeline.transform(X_tab_raw), dtype=torch.float32)
        return tokens, mask, X_tab

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        shots_df: pd.DataFrame,
        target_col: str = "goal",
        match_id_col: str | None = None,  # accepted for ladder compatibility
        n_trials: int = 0,                # ignored (no Optuna for NN)
    ) -> "SetTransformerCxGModel":
        if shots_df.empty:
            raise ValueError("shots_df is empty")
        if target_col not in shots_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")

        torch, nn = require_torch()
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        df = shots_df.reset_index(drop=True)

        self._event_id_col = select_event_id_column(df)
        if self._event_id_col is None:
            logger.warning(
                "SetTransformerCxG: no event_internal_id/event_id column on "
                "shots_df — every shot will get an empty (fully-masked) frame "
                "and the model degenerates to a tabular MLP."
            )

        self._numeric_all = self._resolve_cols(df)
        if not self._numeric_all:
            raise ValueError("No tabular feature columns found for this feature set")
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in self._numeric_all)
        self._tabular_dim = len(self._numeric_all)

        # Diagnostics
        n_with = (
            shots_with_frames_count(df, self._frames(), self._event_id_col,
                                    frames_event_id_col="event_internal_id")
            if self._event_id_col else 0
        )
        logger.info(
            "SetTransformerCxG: %d / %d shots have ≥1 freeze-frame row",
            n_with, len(df),
        )

        self.pipeline = self._build_tabular_pipeline(df)
        tokens, mask, X_tab = self._encode_batch_inputs(df)
        y = torch.tensor(df[target_col].astype(float).to_numpy(), dtype=torch.float32)

        device = self._torch_device()
        model = self._build_torch_model(self._tabular_dim).to(device)
        self._torch_model = model

        bs = resolve_batch_size("set_transformer", self.batch_size)
        dataset = TensorDataset(tokens, mask, X_tab, y)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        model.train()

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for b_tok, b_mask, b_tab, b_y in loader:
                b_tok = b_tok.to(device)
                b_mask = b_mask.to(device)
                b_tab = b_tab.to(device)
                b_y = b_y.to(device)
                optimizer.zero_grad()
                logits = model(b_tok, b_mask, b_tab)
                loss = criterion(logits, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            avg = epoch_loss / max(1, n_batches)
            logger.info(
                "SetTransformerCxG epoch %d/%d loss=%.4f",
                epoch + 1, self.max_epochs, avg,
            )
        return self

    def predict_proba(self, shots_df: pd.DataFrame) -> np.ndarray:
        if self._torch_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        torch, _ = require_torch()
        df = shots_df.reset_index(drop=True)
        tokens, mask, X_tab = self._encode_batch_inputs(df)
        device = self._torch_device()
        self._torch_model.eval()
        with torch.no_grad():
            logits = self._torch_model(
                tokens.to(device), mask.to(device), X_tab.to(device)
            )
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(float)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch, _ = require_torch()
        state = {
            "init": {
                "feature_set": self.feature_set.name,
                "frames_path": str(self.frames_path) if self.frames_path else None,
                "max_players": self.max_players,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "mlp_hidden": self.mlp_hidden,
                "lr": self.lr,
                "max_epochs": self.max_epochs,
                "batch_size": self.batch_size,
                "dropout": self.dropout,
                "weight_decay": self.weight_decay,
                "device": self.device,
                "random_state": self.random_state,
            },
            "fitted": {
                "_numeric_all": self._numeric_all,
                "_bool_set": list(self._bool_set),
                "_tabular_dim": self._tabular_dim,
                "_event_id_col": self._event_id_col,
                "pipeline": self.pipeline,
            },
            "torch_state": (
                self._torch_model.state_dict() if self._torch_model is not None else None
            ),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "SetTransformerCxGModel":
        torch, _ = require_torch()
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(**state["init"])
        for k, v in state["fitted"].items():
            setattr(obj, k, frozenset(v) if k == "_bool_set" else v)
        if state["torch_state"] is not None:
            model = obj._build_torch_model(obj._tabular_dim)
            model.load_state_dict(state["torch_state"])
            device = obj._torch_device()
            model.to(device)
            obj._torch_model = model
        return obj
