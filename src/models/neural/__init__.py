"""Shared neural-network primitives for cxg/cxa/cxt models.

All public surface is re-exported here. Concrete model wrappers
(``SetTransformerCxGModel``, ``GNNPassingNetworkCxAModel``, ...) live
alongside their non-neural siblings in ``src/models/<family>/``.
"""

from src.models.neural.base import (
    TorchModelMixin,
    require_torch,
    resolve_batch_size,
    resolve_device,
)
from src.models.neural.encoders import (
    build_graph_attention_encoder,
    build_mlp_head,
    build_set_transformer_encoder,
    build_tabular_encoder,
)
from src.models.neural.freeze_frame_loader import (
    TOKEN_DIM,
    TOKEN_FEATURES,
    build_knn_adjacency,
    default_frames_path,
    encode_frame_tokens,
    load_freeze_frames,
    select_event_id_column,
    shots_with_frames_count,
)

__all__ = [
    "TorchModelMixin",
    "require_torch",
    "resolve_batch_size",
    "resolve_device",
    "build_graph_attention_encoder",
    "build_mlp_head",
    "build_set_transformer_encoder",
    "build_tabular_encoder",
    "TOKEN_DIM",
    "TOKEN_FEATURES",
    "build_knn_adjacency",
    "default_frames_path",
    "encode_frame_tokens",
    "load_freeze_frames",
    "select_event_id_column",
    "shots_with_frames_count",
]
