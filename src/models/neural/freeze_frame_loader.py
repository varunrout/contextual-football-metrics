"""Raw 360 freeze-frame tensorisation for neural models.

Loads ``data/processed/freeze_frames_360.parquet`` (one row per player per
event) and produces fixed-shape tensors suitable for SetTransformer or graph
encoders.

Schema expected on the input:
    event_internal_id : str  — joins to events.internal_id
    x, y              : float — pitch coords in StatsBomb 105×68 frame
    teammate          : bool — True for shooter's team
    keeper            : bool — True for the goalkeeper

The shooter's own action location (``x_location``, ``y_location``) and target
column (``goal``) come from the shots dataframe.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# StatsBomb pitch coordinates
GOAL_X = 105.0
GOAL_Y = 34.0
PITCH_X = 105.0
PITCH_Y = 68.0
BOX_X_MIN = 88.5
BOX_Y_MIN = 13.84
BOX_Y_MAX = 54.16

# Per-token feature dimension. Keep stable — encoders read it.
TOKEN_DIM = 9
TOKEN_FEATURES = (
    "x_norm",                # x / 105
    "y_norm",                # y / 68
    "is_teammate",           # 1.0 / 0.0
    "is_opponent",           # 1.0 / 0.0
    "is_keeper",             # 1.0 / 0.0
    "dist_to_ball_norm",     # / pitch diag
    "dist_to_goal_norm",     # / pitch diag
    "angle_to_goal",         # radians, signed
    "in_box",                # 1.0 / 0.0
)


def default_frames_path() -> Path:
    """Return the canonical processed-data freeze-frame path.

    Prefers ``data/processed/frames.parquet`` (current pipeline output);
    falls back to the legacy ``freeze_frames_360.parquet`` name.
    """
    base = Path(__file__).resolve().parents[3] / "data" / "processed"
    primary = base / "frames.parquet"
    if primary.exists():
        return primary
    return base / "freeze_frames_360.parquet"


def load_freeze_frames(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``freeze_frames_360.parquet`` if it exists, else return empty DF."""
    fp = Path(path) if path is not None else default_frames_path()
    if not fp.exists():
        logger.warning("freeze_frames_360 not found at %s", fp)
        return pd.DataFrame(
            columns=["event_internal_id", "x", "y", "teammate", "keeper"]
        )
    return pd.read_parquet(fp)


def encode_frame_tokens(
    shots_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    *,
    max_players: int = 22,
    event_id_col: str = "event_internal_id",
    frames_event_id_col: str | None = None,
    ball_x_col: str = "x_location",
    ball_y_col: str = "y_location",
):
    """Encode freeze frames into ``(B, K, TOKEN_DIM)`` token tensors.

    Parameters
    ----------
    shots_df
        Per-shot dataframe (one row = one shot). Must contain ``event_id_col``
        and the shooter's ``ball_x_col``/``ball_y_col``.
    frames_df
        Per-player-per-event dataframe. Rows for events not in ``shots_df``
        are ignored. Shots without any matching frame yield a fully-masked
        all-zero token block.
    max_players
        K. Hard cap on tokens per shot. With 22 outfield + 1 keeper = 23
        max, K=22 truncates the furthest defender from the ball when a
        full set is present (rare in practice).

    Returns
    -------
    tokens : torch.FloatTensor (B, K, TOKEN_DIM)
    mask   : torch.BoolTensor  (B, K)         True = padding
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch not installed.") from exc

    B = len(shots_df)
    K = max_players
    tokens = np.zeros((B, K, TOKEN_DIM), dtype=np.float32)
    mask = np.ones((B, K), dtype=bool)  # default everything padded

    pitch_diag = math.hypot(PITCH_X, PITCH_Y)

    f_col = frames_event_id_col or event_id_col
    if frames_df.empty or f_col not in frames_df.columns or event_id_col not in shots_df.columns:
        return torch.from_numpy(tokens), torch.from_numpy(mask)

    # Build a per-event lookup once
    grouped = frames_df.groupby(f_col)

    shots = shots_df.reset_index(drop=True)
    for i, row in shots.iterrows():
        eid = row.get(event_id_col)
        if eid is None or eid not in grouped.groups:
            continue
        ff = grouped.get_group(eid)
        bx = float(row.get(ball_x_col, np.nan))
        by = float(row.get(ball_y_col, np.nan))
        if not (np.isfinite(bx) and np.isfinite(by)):
            continue

        # Sort by distance to ball so the K closest survive truncation
        x = ff["x"].astype(float).to_numpy()
        y = ff["y"].astype(float).to_numpy()
        teammate = ff["teammate"].astype(bool).to_numpy()
        keeper = (
            ff["keeper"].astype(bool).to_numpy()
            if "keeper" in ff.columns
            else np.zeros(len(ff), dtype=bool)
        )
        d_ball = np.hypot(x - bx, y - by)
        order = np.argsort(d_ball)[:K]

        n = len(order)
        for j, idx in enumerate(order):
            px, py = x[idx], y[idx]
            d_goal = math.hypot(GOAL_X - px, GOAL_Y - py)
            angle = math.atan2(GOAL_Y - py, GOAL_X - px)
            in_box = (
                (px >= BOX_X_MIN)
                and (BOX_Y_MIN <= py <= BOX_Y_MAX)
            )
            tokens[i, j] = (
                px / PITCH_X,
                py / PITCH_Y,
                1.0 if teammate[idx] else 0.0,
                0.0 if teammate[idx] else 1.0,
                1.0 if keeper[idx] else 0.0,
                d_ball[idx] / pitch_diag,
                d_goal / pitch_diag,
                float(angle),
                1.0 if in_box else 0.0,
            )
        mask[i, :n] = False  # unmask the n actual players

    return torch.from_numpy(tokens), torch.from_numpy(mask)


def build_knn_adjacency(tokens, mask, k: int = 4):
    """Build a per-graph k-NN adjacency mask over teammate-only edges.

    For the GNN passing-network model: each node attends only to its k
    nearest teammates (by pitch distance). Opponents are kept as nodes but
    don't form outgoing edges (masked).

    Parameters
    ----------
    tokens : (B, N, TOKEN_DIM) — assumes column 0/1 are x_norm/y_norm and
                                  column 2 is is_teammate.
    mask   : (B, N) bool       — True = padding.
    k      : int

    Returns
    -------
    attn_mask : (B, N, N) bool — True where the edge does NOT exist.
                                  True on the diagonal (no self-loop) and
                                  for any (i, j) where j is not in i's
                                  k-NN-teammate set.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch not installed.") from exc

    B, N, _ = tokens.shape
    attn = torch.ones(B, N, N, dtype=torch.bool)  # default: no edge
    coords = tokens[..., :2]                       # (B, N, 2) in [0,1]
    is_teammate = tokens[..., 2] > 0.5             # (B, N)

    # Pairwise distances per graph
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)   # (B, N, N, 2)
    dist = (diff * diff).sum(dim=-1).sqrt()            # (B, N, N)

    # Disallow edges to padded positions or to self
    pad_col = mask.unsqueeze(1).expand(B, N, N)        # mask along columns
    not_teammate_col = (~is_teammate).unsqueeze(1).expand(B, N, N)
    invalid = pad_col | not_teammate_col
    eye = torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
    invalid = invalid | eye

    big = dist.max() + 1.0
    masked_dist = dist.masked_fill(invalid, big)

    # Take k smallest along last dim
    k_eff = min(k, N - 1)
    _, idx = masked_dist.topk(k_eff, dim=-1, largest=False)
    keep = torch.zeros_like(attn)                       # False = will keep edge
    keep.scatter_(2, idx, True)
    # An edge exists where keep AND not invalid.
    edges_present = keep & (~invalid)
    attn = ~edges_present                              # True = mask out
    return attn


def shots_with_frames_count(shots_df: pd.DataFrame, frames_df: pd.DataFrame,
                             event_id_col: str = "event_internal_id",
                             frames_event_id_col: str | None = None) -> int:
    """Diagnostic: how many shots actually have any freeze-frame data."""
    f_col = frames_event_id_col or event_id_col
    if frames_df.empty or event_id_col not in shots_df.columns or f_col not in frames_df.columns:
        return 0
    fids = set(frames_df[f_col].unique())
    return int(shots_df[event_id_col].isin(fids).sum())


def select_event_id_column(shots_df: pd.DataFrame) -> str | None:
    """Return the column to join shots → frames on, or None if absent."""
    for col in ("event_internal_id", "event_id"):
        if col in shots_df.columns:
            return col
    return None


__all__: Iterable[str] = (
    "TOKEN_DIM",
    "TOKEN_FEATURES",
    "default_frames_path",
    "load_freeze_frames",
    "encode_frame_tokens",
    "build_knn_adjacency",
    "shots_with_frames_count",
    "select_event_id_column",
)
