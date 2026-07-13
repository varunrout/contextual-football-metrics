"""Base mixin for neural models.

Centralises:
- Lazy torch import.
- Device resolution from the active runtime profile.
- Standardised pickle (state-dict + reconstruction metadata) for save/load.

Concrete model classes live next to their non-neural siblings (e.g.
``src/models/cxg/set_transformer_model.py``) so the cxg/cxa/cxt taxonomy
stays intact. Shared building blocks (encoders, heads, freeze-frame loader)
live in this package.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def require_torch():
    """Import torch + nn or raise a clean ImportError.

    Returns
    -------
    (torch, nn) module pair.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - environmental
        raise ImportError(
            "PyTorch not installed. Install with: poetry install --with neural "
            "(or `pip install torch`)."
        ) from exc
    return torch, nn


def resolve_device(device: str | None) -> str:
    """Resolve a torch device string.

    Order:
      1. Explicit ``device`` argument.
      2. ``src.runtime.get_profile().device`` if a profile is active.
      3. ``torch.cuda.is_available()`` heuristic.
      4. ``"cpu"``.
    """
    if device is not None:
        return device
    try:
        from src.runtime import get_profile

        prof = get_profile()
    except Exception:  # noqa: BLE001 — runtime is optional
        prof = None
    if prof is not None:
        return prof.device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def resolve_batch_size(family: str, override: int | None = None) -> int:
    """Resolve a batch size for a neural family from the active profile.

    Falls back to ``override`` (or a sensible default) when no profile is active
    or the family is not registered in the profile YAML.
    """
    if override is not None:
        return int(override)
    try:
        from src.runtime import get_profile

        prof = get_profile()
        if prof is not None:
            try:
                return prof.batch_size(family)
            except KeyError:
                logger.debug("Profile has no batch_size for %s; using fallback", family)
    except Exception:  # noqa: BLE001
        pass
    fallback = {"ffnn": 256, "transformer": 64, "set_transformer": 32, "gnn": 64}
    return fallback.get(family, 128)


class TorchModelMixin:
    """Shared utilities for sklearn-style neural model wrappers.

    Concrete classes are still expected to define ``fit``/``predict[_proba]``,
    but they should call ``self._device`` (set lazily by ``_torch_device()``)
    when moving tensors and the model.
    """

    device: str | None  # constructor-set; ``None`` means autodetect at .fit()
    _resolved_device: str | None = None

    def _torch_device(self) -> str:
        """Return the device string, resolving and caching on first call."""
        if self._resolved_device is None:
            self._resolved_device = resolve_device(getattr(self, "device", None))
            logger.info(
                "%s: using torch device %s",
                type(self).__name__,
                self._resolved_device,
            )
        return self._resolved_device

    # ── Persistence helpers ───────────────────────────────────────────────────
    # Concrete models override save/load when they need extra state. The
    # default below stores the wrapper's __dict__ minus the torch module
    # (since locally-defined nn.Module subclasses can't always be re-imported)
    # plus the model's state_dict, and reconstructs by re-running the model's
    # ``_build_torch_model`` factory.

    def _state_for_save(self) -> dict[str, Any]:
        torch, _ = require_torch()
        state = {k: v for k, v in self.__dict__.items() if k != "_torch_model"}
        if getattr(self, "_torch_model", None) is not None:
            state["_torch_state_dict"] = self._torch_model.state_dict()  # type: ignore[union-attr]
        return state


def is_neural_model(model: Any) -> bool:
    """Return True if ``model`` needs its own ``.save()`` instead of joblib.

    All neural model wrappers (``SetTransformerCxGModel``,
    ``GNNPassingNetworkCxAModel``, ``GNNStateValueModel``,
    ``SetTransformerStateValueModel``, ...) subclass :class:`TorchModelMixin`
    and hold locally-scoped ``nn.Module`` state that vanilla ``joblib``/pickle
    cannot round-trip. Training scripts across cxg/cxa/cxt should use this
    single check instead of ad hoc attribute or family-string sniffing, so a
    new neural family only needs to subclass ``TorchModelMixin`` to be picked
    up everywhere.
    """
    return isinstance(model, TorchModelMixin)
