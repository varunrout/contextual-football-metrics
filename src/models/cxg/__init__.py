"""CxG models."""

from src.models.cxg.baseline import BaselineCxGMetrics, BaselineCxGModel, filter_shot_events

__all__ = [
    "BaselineCxGModel",
    "BaselineCxGMetrics",
    "filter_shot_events",
    # Lazy: SetTransformerCxGModel imports torch and is exported on demand
    # via `from src.models.cxg.set_transformer_model import SetTransformerCxGModel`
]
