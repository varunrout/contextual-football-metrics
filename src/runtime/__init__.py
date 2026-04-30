"""Runtime helpers for the contextual-football-metrics pipeline.

Exposes the active execution profile (CPU / GPU / Cloud) and convenience
accessors used by training scripts, model wrappers, and the Prefect flow.

Usage
-----
>>> from src.runtime import load_profile, get_profile
>>> cfg = load_profile("gpu")          # explicit
>>> cfg = load_profile("auto")         # auto-detect

The profile is cached in a module-level global; subsequent ``get_profile()``
calls return the same instance until ``load_profile`` is called again.
"""

from .gbm_device import lightgbm_kwargs, xgboost_kwargs
from .profile import (
    ProfileConfig,
    autodetect,
    get_profile,
    load_profile,
    require_profile,
)

__all__ = [
    "ProfileConfig",
    "autodetect",
    "get_profile",
    "lightgbm_kwargs",
    "load_profile",
    "require_profile",
    "xgboost_kwargs",
]
