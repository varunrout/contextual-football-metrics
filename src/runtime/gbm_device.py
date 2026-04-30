"""Device-aware kwargs for LightGBM and XGBoost.

The two libraries spell GPU support differently. This helper hides those
differences so model code can simply ``**lightgbm_kwargs(device)`` into the
estimator constructor.

Resolution order for the device:
1. Explicit ``device`` argument.
2. The currently-active runtime profile (``src.runtime.get_profile()``).
3. Fallback to ``"cpu"``.

LightGBM accepts ``device``  ∈ {``"cpu"``, ``"gpu"``, ``"cuda"``}. The
default GPU build uses OpenCL (``"gpu"``); only CUDA-compiled wheels accept
``"cuda"``. We pass through ``"cuda"`` if requested and let LightGBM error if
the wheel doesn't support it (clear error is better than silent fallback).

XGBoost ≥2.0 accepts ``device`` ∈ {``"cpu"``, ``"cuda"``, ``"cuda:0"``}. Older
versions used ``tree_method="gpu_hist"``; we target ≥2.0 (already in
``pyproject.toml``).
"""

from __future__ import annotations

import logging
from typing import Any

from .profile import get_profile

logger = logging.getLogger(__name__)


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    prof = get_profile()
    if prof is not None:
        return prof.gbm_device
    return "cpu"


def lightgbm_kwargs(device: str | None = None) -> dict[str, Any]:
    """Return kwargs to splat into ``lgb.LGBMClassifier`` / ``LGBMRegressor``.

    >>> lightgbm_kwargs("cpu")
    {'device': 'cpu'}
    >>> lightgbm_kwargs("cuda")
    {'device': 'cuda'}
    """
    dev = _resolve_device(device)
    if dev not in ("cpu", "gpu", "cuda"):
        logger.warning("Unknown LightGBM device %r; defaulting to cpu", dev)
        dev = "cpu"
    return {"device": dev}


def xgboost_kwargs(device: str | None = None) -> dict[str, Any]:
    """Return kwargs to splat into ``xgb.XGBClassifier`` / ``XGBRegressor``.

    Always uses ``tree_method="hist"`` (the recommended default for both CPU
    and GPU since XGBoost 2.0).

    >>> xgboost_kwargs("cpu")
    {'device': 'cpu', 'tree_method': 'hist'}
    >>> xgboost_kwargs("cuda")
    {'device': 'cuda', 'tree_method': 'hist'}
    """
    dev = _resolve_device(device)
    if dev == "gpu":
        # XGBoost ≥2.0 only accepts "cpu" or "cuda"; map LightGBM-style "gpu"
        # to "cuda" since both signal "use the GPU".
        dev = "cuda"
    if dev not in ("cpu", "cuda") and not dev.startswith("cuda:"):
        logger.warning("Unknown XGBoost device %r; defaulting to cpu", dev)
        dev = "cpu"
    return {"device": dev, "tree_method": "hist"}
