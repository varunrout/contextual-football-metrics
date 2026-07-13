"""CxA models."""

from src.models.cxa.baseline import BaselineCxAModel, BaselineCxAMetrics, filter_pass_events

__all__ = [
    "BaselineCxAModel",
    "BaselineCxAMetrics",
    "filter_pass_events",
    # Lazy: GNNPassingNetworkCxAModel imports torch and is exported on demand
    # via `from src.models.cxa.gnn_passing_network import GNNPassingNetworkCxAModel`
]
