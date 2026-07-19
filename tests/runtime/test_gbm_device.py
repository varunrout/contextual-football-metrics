"""Tests for the GBM device-kwargs helper."""

from __future__ import annotations

import pytest

from src.runtime import load_profile
from src.runtime.gbm_device import lightgbm_kwargs, xgboost_kwargs


@pytest.fixture(autouse=True)
def _reset_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure tests don't bleed an active profile into one another."""
    import src.runtime.profile as p

    monkeypatch.setattr(p, "_active", None)


def test_lightgbm_kwargs_explicit_cpu() -> None:
    assert lightgbm_kwargs("cpu") == {"device": "cpu"}


def test_lightgbm_kwargs_explicit_cuda() -> None:
    assert lightgbm_kwargs("cuda") == {"device": "cuda"}


def test_lightgbm_kwargs_unknown_falls_back_cpu() -> None:
    assert lightgbm_kwargs("rocm")["device"] == "cpu"


def test_xgboost_kwargs_explicit_cpu() -> None:
    assert xgboost_kwargs("cpu") == {"device": "cpu", "tree_method": "hist"}


def test_xgboost_kwargs_explicit_cuda() -> None:
    assert xgboost_kwargs("cuda") == {"device": "cuda", "tree_method": "hist"}


def test_xgboost_kwargs_lightgbm_style_gpu_maps_to_cuda() -> None:
    """`gpu` is a LightGBM-only spelling; XGBoost helper should map → cuda."""
    assert xgboost_kwargs("gpu")["device"] == "cuda"


def test_no_active_profile_defaults_to_cpu() -> None:
    assert lightgbm_kwargs() == {"device": "cpu"}
    assert xgboost_kwargs() == {"device": "cpu", "tree_method": "hist"}


def test_active_cpu_profile_propagates() -> None:
    load_profile("cpu", validate=False)
    assert lightgbm_kwargs() == {"device": "cpu"}
    assert xgboost_kwargs()["device"] == "cpu"


def test_active_gpu_profile_propagates() -> None:
    load_profile("gpu", validate=False)
    assert lightgbm_kwargs()["device"] == "cuda"
    assert xgboost_kwargs()["device"] == "cuda"


def test_explicit_device_overrides_active_profile() -> None:
    load_profile("gpu", validate=False)
    # Explicit "cpu" wins over the active gpu profile.
    assert lightgbm_kwargs("cpu") == {"device": "cpu"}
    assert xgboost_kwargs("cpu")["device"] == "cpu"
