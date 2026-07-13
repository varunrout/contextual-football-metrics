"""Graph-Neural-Network shot-creation model.

Builds a per-action graph from the 360 freeze frame at the moment of the
action: each player is a node, and each teammate node attends to its k
nearest teammates (a soft "passing options" graph). The graph embedding is
fused with the contextual tabular features and passed through an MLP head
producing P(shot_created).

The actual graph attention layer is implemented manually on top of
``nn.MultiheadAttention`` with a per-graph adjacency mask, so this model
has no dependency on ``torch_geometric``.

API mirrors the other CxA models so it slots into ``ShotCreationLadder``:

    GNNPassingNetworkCxAModel().fit(actions_df, target_col="shot_created")
    GNNPassingNetworkCxAModel().predict_proba(actions_df)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.cxa.shot_creation_model import (
    ShotCreationMetrics, _make_X,
)
from src.models.cxa.feature_sets import (
    CxAFeatureSetSpec, get_feature_set,
)
from src.models.neural import (
    TOKEN_DIM,
    TorchModelMixin,
    build_graph_attention_encoder,
    build_knn_adjacency,
    build_mlp_head,
    build_tabular_encoder,
    encode_frame_tokens,
    load_freeze_frames,
    require_torch,
    resolve_batch_size,
    select_event_id_column,
)

logger = logging.getLogger(__name__)


class GNNPassingNetworkCxAModel(TorchModelMixin):
    """Graph-attention encoder over freeze-frame players + tabular fusion."""

    def __init__(
        self,
        feature_set: str | CxAFeatureSetSpec = "contextual",
        frames_path: str | Path | None = None,
        max_players: int = 22,
        k_neighbors: int = 4,
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
        self.k_neighbors = k_neighbors
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
                "GNNPassingNetworkCxA: loaded %d freeze-frame rows",
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
        graph_encoder = build_graph_attention_encoder(
            node_feat_dim=TOKEN_DIM,
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
            final_activation=None,
        )

        class _GNNCxA(nn.Module):
            def __init__(self):
                super().__init__()
                self.graph_encoder = graph_encoder
                self.tab_encoder = tab_encoder
                self.head = head

            def forward(self, tokens, mask, adj_mask, tab):
                g = self.graph_encoder(tokens, adj_mask, mask)
                t = self.tab_encoder(tab)
                fused = torch.cat([g, t], dim=-1)
                return self.head(fused).squeeze(-1)

        return _GNNCxA()

    def _encode_batch_inputs(self, df: pd.DataFrame):
        torch, _ = require_torch()
        tokens, mask = encode_frame_tokens(
            df, self._frames(),
            max_players=self.max_players,
            event_id_col=self._event_id_col or "event_internal_id",
            frames_event_id_col="event_internal_id",
        )
        adj = build_knn_adjacency(tokens, mask, k=self.k_neighbors)
        if self.pipeline is None:
            raise RuntimeError("Tabular pipeline not fitted — call fit() first.")
        X_tab_raw = _make_X(df, self._numeric_all, [], self._bool_set)[self._numeric_all]
        X_tab = torch.tensor(self.pipeline.transform(X_tab_raw), dtype=torch.float32)
        return tokens, mask, adj, X_tab

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "shot_created",
    ) -> "GNNPassingNetworkCxAModel":
        if actions_df.empty:
            raise ValueError("actions_df is empty")
        if target_col not in actions_df.columns:
            raise ValueError(f"Missing target column: {target_col!r}")

        torch, nn = require_torch()
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        df = actions_df.reset_index(drop=True)

        self._event_id_col = select_event_id_column(df)
        self._numeric_all = self._resolve_cols(df)
        if not self._numeric_all:
            raise ValueError("No tabular feature columns found")
        self._bool_set = frozenset(c for c in self.feature_set.boolean if c in self._numeric_all)
        self._tabular_dim = len(self._numeric_all)

        self.pipeline = self._build_tabular_pipeline(df)
        tokens, mask, adj, X_tab = self._encode_batch_inputs(df)
        y = torch.tensor(df[target_col].astype(float).to_numpy(), dtype=torch.float32)

        device = self._torch_device()
        model = self._build_torch_model(self._tabular_dim).to(device)
        self._torch_model = model

        bs = resolve_batch_size("gnn", self.batch_size)
        dataset = TensorDataset(tokens, mask, adj, X_tab, y)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        model.train()

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for b_tok, b_mask, b_adj, b_tab, b_y in loader:
                b_tok = b_tok.to(device)
                b_mask = b_mask.to(device)
                b_adj = b_adj.to(device)
                b_tab = b_tab.to(device)
                b_y = b_y.to(device)
                optimizer.zero_grad()
                logits = model(b_tok, b_mask, b_adj, b_tab)
                loss = criterion(logits, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            logger.info(
                "GNNPassingNetworkCxA epoch %d/%d loss=%.4f",
                epoch + 1, self.max_epochs, epoch_loss / max(1, n_batches),
            )
        return self

    def predict_proba(self, actions_df: pd.DataFrame) -> np.ndarray:
        if self._torch_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        torch, _ = require_torch()
        df = actions_df.reset_index(drop=True)
        tokens, mask, adj, X_tab = self._encode_batch_inputs(df)
        device = self._torch_device()
        self._torch_model.eval()
        with torch.no_grad():
            logits = self._torch_model(
                tokens.to(device), mask.to(device),
                adj.to(device), X_tab.to(device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(float)

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

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "init": {
                "feature_set": self.feature_set.name,
                "frames_path": str(self.frames_path) if self.frames_path else None,
                "max_players": self.max_players,
                "k_neighbors": self.k_neighbors,
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
    def load(cls, path: str | Path) -> "GNNPassingNetworkCxAModel":
        require_torch()
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(**state["init"])
        for k, v in state["fitted"].items():
            setattr(obj, k, frozenset(v) if k == "_bool_set" else v)
        if state["torch_state"] is not None:
            model = obj._build_torch_model(obj._tabular_dim)
            model.load_state_dict(state["torch_state"])
            model.to(obj._torch_device())
            obj._torch_model = model
        return obj
