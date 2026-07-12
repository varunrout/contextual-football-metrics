"""Reusable neural building blocks (heads + encoders).

All classes use the ``require_torch()`` helper so importing this module is
safe on torch-less environments — the ``nn.Module`` subclasses are defined
inside builder functions and only constructed when callers explicitly ask.
"""

from __future__ import annotations

from typing import Sequence

from src.models.neural.base import require_torch


# ── Heads ────────────────────────────────────────────────────────────────────


def build_mlp_head(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int = 1,
    dropout: float = 0.1,
    final_activation: str | None = None,
):
    """Sequential MLP with ReLU + Dropout between layers.

    Parameters
    ----------
    in_dim, hidden_dims, out_dim
        Layer widths.
    dropout
        Dropout probability between hidden layers.
    final_activation
        ``None`` (logits), ``"sigmoid"``, ``"softplus"``.
    """
    _, nn = require_torch()
    layers: list = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if final_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif final_activation == "softplus":
        layers.append(nn.Softplus())
    elif final_activation is not None:
        raise ValueError(f"Unknown final_activation={final_activation!r}")
    return nn.Sequential(*layers)


# ── Tabular encoder ──────────────────────────────────────────────────────────


def build_tabular_encoder(in_dim: int, out_dim: int, dropout: float = 0.1):
    """Linear → ReLU → Dropout block used by every fusion head."""
    _, nn = require_torch()
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)
    )


# ── Set encoder (transformer over an unordered set of tokens) ────────────────


def build_set_transformer_encoder(
    token_dim: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
):
    """Build a SetTransformer-style encoder with a learnable CLS token.

    Returns an ``nn.Module`` whose ``forward(tokens, mask)`` signature is:

        tokens : (B, K, token_dim)   — set elements (order-invariant)
        mask   : (B, K) bool         — True = padding (ignored by attention)
        →       (B, d_model)         — CLS pooled output

    The padded positions are excluded from attention via
    ``src_key_padding_mask``; the CLS token is never masked.
    """
    torch, nn = require_torch()

    class _SetTransformerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_proj = nn.Linear(token_dim, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            nn.init.zeros_(self.cls_token)
            nn.init.xavier_uniform_(self.token_proj.weight)

        def forward(self, tokens, mask):
            tok = self.token_proj(tokens)              # (B, K, d)
            B = tok.size(0)
            cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d)
            seq = torch.cat([cls, tok], dim=1)         # (B, K+1, d)
            cls_mask = torch.zeros(
                B, 1, dtype=torch.bool, device=mask.device
            )
            full_mask = torch.cat([cls_mask, mask], dim=1)
            out = self.transformer(seq, src_key_padding_mask=full_mask)
            return out[:, 0, :]                        # (B, d)

    return _SetTransformerEncoder()


# ── Graph encoder (attention over k-NN player graph) ─────────────────────────


def build_graph_attention_encoder(
    node_feat_dim: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
):
    """Stack of Graph Attention layers operating on a soft adjacency mask.

    ``forward(node_feats, adj_mask)``:
        node_feats : (B, N, node_feat_dim)
        adj_mask   : (B, N, N) bool — True where the edge does NOT exist
                     (so it can be passed straight as ``attn_mask``).
    Returns a graph-level pooling: mean over valid nodes, ``(B, d_model)``.

    This is the lightweight equivalent of a 2-layer GAT and avoids the
    ``torch_geometric`` dependency. For each node we attend to its graph
    neighbours via a multi-head attention block where edges absent from the
    adjacency are masked out.
    """
    torch, nn = require_torch()

    class _GraphAttentionLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x, adj_mask, key_padding_mask):
            # adj_mask: (B, N, N) True=mask out. MultiheadAttention expects
            # attn_mask of shape (B*nhead, N, N) or (N, N). Replicate per head.
            B, N, _ = x.shape
            head_mask = (
                adj_mask.unsqueeze(1)
                .expand(B, n_heads, N, N)
                .reshape(B * n_heads, N, N)
            )
            attn_out, _ = self.attn(
                x, x, x,
                attn_mask=head_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            # When a node has no valid edges (all keys masked) the softmax of
            # all-(-inf) yields NaN. Replace those with zero so the residual
            # connection carries the input feature forward unchanged.
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.ffn(x))
            return x

    class _GraphEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(node_feat_dim, d_model)
            self.layers = nn.ModuleList([_GraphAttentionLayer() for _ in range(n_layers)])

        def forward(self, node_feats, adj_mask, node_pad_mask):
            x = self.in_proj(node_feats)
            for layer in self.layers:
                x = layer(x, adj_mask, node_pad_mask)
            # Masked mean pool over valid nodes
            valid = (~node_pad_mask).unsqueeze(-1).float()
            summed = (x * valid).sum(dim=1)
            count = valid.sum(dim=1).clamp(min=1.0)
            return summed / count

    return _GraphEncoder()
