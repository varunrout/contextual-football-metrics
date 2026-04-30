"""Tests for the runtime profile loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.runtime.profile import (
    PROFILES_DIR,
    VALID_PROFILES,
    ProfileConfig,
    autodetect,
    load_profile,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure ambient env vars don't leak into tests."""
    for var in ("CFM_PROFILE", "COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE"):
        monkeypatch.delenv(var, raising=False)


def test_all_profile_yaml_files_exist() -> None:
    for name in VALID_PROFILES:
        assert (PROFILES_DIR / f"{name}.yaml").is_file(), f"missing {name}.yaml"


@pytest.mark.parametrize("name", VALID_PROFILES)
def test_load_profile_valid(name: str) -> None:
    cfg = load_profile(name, validate=False)
    assert isinstance(cfg, ProfileConfig)
    assert cfg.name == name
    assert cfg.accelerator_type in ("cpu", "cuda")
    assert cfg.gbm_device in ("cpu", "cuda")
    assert cfg.batch_size("ffnn") > 0
    assert cfg.batch_size("transformer") > 0


def test_cpu_profile_specifics() -> None:
    cfg = load_profile("cpu", validate=False)
    assert cfg.accelerator_type == "cpu"
    assert cfg.is_gpu is False
    assert cfg.gbm_device == "cpu"
    assert cfg.precision == 32


def test_gpu_profile_specifics() -> None:
    cfg = load_profile("gpu", validate=False)
    assert cfg.accelerator_type == "cuda"
    assert cfg.is_gpu is True
    assert cfg.gbm_device == "cuda"
    assert cfg.device.startswith("cuda")


def test_cloud_profile_paths_use_drive() -> None:
    cfg = load_profile("cloud", validate=False)
    assert cfg.is_gpu
    # Cloud points to a Drive-mounted location (use as_posix for OS-independent check)
    assert "/content/drive" in cfg.data_root.as_posix()


def test_load_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        load_profile("does-not-exist", validate=False)


def test_unknown_batch_family_raises() -> None:
    cfg = load_profile("cpu", validate=False)
    with pytest.raises(KeyError):
        cfg.batch_size("nonexistent-family")


def test_env_var_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFM_PROFILE", "gpu")
    cfg = load_profile(None, validate=False)
    assert cfg.name == "gpu"


def test_auto_falls_through_to_autodetect(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force autodetect to land on cpu by ensuring no GPU env hints + no CUDA torch.
    monkeypatch.delenv("CFM_PROFILE", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    # Patch torch to report no CUDA so the test is deterministic on any host.
    import importlib

    try:
        torch = importlib.import_module("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    except ImportError:
        pass  # No torch installed → autodetect already returns "cpu".

    assert autodetect() == "cpu"
    cfg = load_profile("auto", validate=False)
    assert cfg.name == "cpu"


def test_autodetect_cloud_via_colab_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLAB_GPU", "1")
    assert autodetect() == "cloud"


def test_mlflow_uri_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_uri = f"file:{tmp_path / 'custom_mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", custom_uri)
    cfg = load_profile("cpu", validate=False)
    assert cfg.mlflow_tracking_uri == custom_uri


def test_to_dict_round_trip() -> None:
    cfg = load_profile("cpu", validate=False)
    snapshot = cfg.to_dict()
    assert snapshot["profile_name"] == "cpu"
    assert snapshot["accelerator"]["type"] == "cpu"


def test_paths_are_path_objects() -> None:
    cfg = load_profile("cpu", validate=False)
    for attr in ("data_root", "reports_root", "models_root", "outputs_root", "checkpoint_root"):
        assert isinstance(getattr(cfg, attr), Path)


def test_get_profile_returns_last_loaded() -> None:
    from src.runtime.profile import get_profile

    load_profile("cpu", validate=False)
    assert get_profile().name == "cpu"  # type: ignore[union-attr]
    load_profile("gpu", validate=False)
    assert get_profile().name == "gpu"  # type: ignore[union-attr]


def test_validate_warns_when_cuda_missing(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """Loading 'gpu' on a CPU host should warn (not raise) when torch exists."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level("WARNING"):
        load_profile("gpu", validate=True)
    assert any("CUDA" in rec.message for rec in caplog.records)
