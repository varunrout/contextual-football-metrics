"""Profile loader for the contextual-football-metrics pipeline.

A *profile* bundles every device- and environment-specific setting so the
exact same code path runs on:

* a CPU-only laptop  (``profile=cpu``)
* a workstation with an NVIDIA GPU  (``profile=gpu``)
* a free cloud notebook such as Colab / Kaggle  (``profile=cloud``)

YAML files live in ``configs/profiles/``. Environment-variable interpolation
is supported via OmegaConf's ``${oc.env:VAR,default}`` syntax.

Selection precedence (highest first):
1. Explicit name passed to :func:`load_profile`.
2. ``CFM_PROFILE`` environment variable.
3. :func:`autodetect` based on runtime hints.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).resolve().parents[2] / "configs" / "profiles"
VALID_PROFILES = ("cpu", "gpu", "cloud")

# Module-level cache; populated by load_profile / get_profile.
_active: "ProfileConfig | None" = None


@dataclass(frozen=True)
class ProfileConfig:
    """Validated, immutable view over a profile YAML."""

    name: str
    raw: DictConfig = field(repr=False)

    # ── Accelerator ──────────────────────────────────────────────────────
    @property
    def accelerator_type(self) -> str:
        """Either ``"cpu"`` or ``"cuda"``."""
        return str(self.raw.accelerator.type)

    @property
    def device(self) -> str:
        """Torch device string, e.g. ``"cpu"`` or ``"cuda:0"``."""
        return str(self.raw.accelerator.device)

    @property
    def precision(self) -> str | int:
        """Lightning precision spec; ``32`` / ``"16-mixed"`` / ``"bf16-mixed"``."""
        return self.raw.accelerator.precision

    @property
    def is_gpu(self) -> bool:
        return self.accelerator_type == "cuda"

    # ── Torch DataLoader knobs ───────────────────────────────────────────
    @property
    def num_workers(self) -> int:
        return int(self.raw.torch.num_workers)

    @property
    def pin_memory(self) -> bool:
        return bool(self.raw.torch.pin_memory)

    # ── GBM (LightGBM / XGBoost) ─────────────────────────────────────────
    @property
    def gbm_device(self) -> str:
        """``"cpu"`` or ``"cuda"``. Pass directly to ``LGBM*`` / ``XGB*``."""
        return str(self.raw.gbm.device)

    @property
    def gbm_n_jobs(self) -> int:
        return int(self.raw.gbm.n_jobs)

    # ── Batch sizes ──────────────────────────────────────────────────────
    def batch_size(self, family: str) -> int:
        """Return the batch size for a neural family (ffnn/transformer/...)."""
        sizes = self.raw.batch_size
        if family not in sizes:
            raise KeyError(
                f"No batch_size for family '{family}' in profile '{self.name}'. "
                f"Known: {sorted(sizes.keys())}"
            )
        return int(sizes[family])

    # ── MLflow ───────────────────────────────────────────────────────────
    @property
    def mlflow_tracking_uri(self) -> str:
        return str(self.raw.mlflow.tracking_uri)

    @property
    def mlflow_experiment(self) -> str:
        return str(self.raw.mlflow.experiment)

    @property
    def mlflow_artifact_root(self) -> str | None:
        v = self.raw.mlflow.get("artifact_root", None)
        return None if v in (None, "null", "") else str(v)

    # ── Paths ────────────────────────────────────────────────────────────
    @property
    def data_root(self) -> Path:
        return Path(self.raw.paths.data_root)

    @property
    def reports_root(self) -> Path:
        return Path(self.raw.paths.reports_root)

    @property
    def models_root(self) -> Path:
        return Path(self.raw.paths.models_root)

    @property
    def outputs_root(self) -> Path:
        return Path(self.raw.paths.outputs_root)

    @property
    def checkpoint_root(self) -> Path:
        return Path(self.raw.paths.checkpoint_root)

    # ── Prefect ──────────────────────────────────────────────────────────
    @property
    def prefect_task_runner(self) -> str:
        """Name of a class in ``prefect.task_runners`` (e.g. ``ThreadPoolTaskRunner``)."""
        return str(self.raw.prefect.task_runner)

    @property
    def prefect_max_workers(self) -> int | None:
        """``max_workers`` kwarg for the task runner, or ``None`` for unbounded."""
        v = self.raw.prefect.get("max_workers", None)
        return None if v in (None, "null", "") else int(v)

    @property
    def prefect_log_level(self) -> str:
        return str(self.raw.prefect.log_level)

    def build_task_runner(self):
        """Instantiate the Prefect 3 task runner configured for this profile.

        Resolves ``prefect_task_runner`` against ``prefect.task_runners`` by
        name and applies ``prefect_max_workers`` when the runner accepts it.
        Raises ``ValueError`` if the configured name isn't a real Prefect 3
        task runner (e.g. a pre-Prefect-3 name like ``SequentialTaskRunner``).
        """
        import prefect.task_runners as _task_runners

        name = self.prefect_task_runner
        runner_cls = getattr(_task_runners, name, None)
        if runner_cls is None:
            raise ValueError(
                f"Profile '{self.name}' configures prefect.task_runner={name!r}, "
                f"which is not a valid Prefect 3 task runner in "
                f"prefect.task_runners. Valid options include "
                f"ThreadPoolTaskRunner, ProcessPoolTaskRunner."
            )
        max_workers = self.prefect_max_workers
        if max_workers is not None:
            return runner_cls(max_workers=max_workers)
        return runner_cls()

    # ── Misc ─────────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """Resolved plain-dict snapshot (good for MLflow tagging)."""
        return OmegaConf.to_container(self.raw, resolve=True)  # type: ignore[return-value]


def autodetect() -> str:
    """Best-effort guess of the right profile for the current process.

    Order:
        1. ``COLAB_GPU`` / ``KAGGLE_KERNEL_RUN_TYPE`` env vars → ``cloud``.
        2. ``torch.cuda.is_available()`` → ``gpu``.
        3. Otherwise → ``cpu``.
    """
    if os.environ.get("COLAB_GPU") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "cloud"
    try:  # torch is optional at runtime
        import torch

        if torch.cuda.is_available():
            return "gpu"
    except Exception:  # noqa: BLE001 — torch may not be installed
        pass
    return "cpu"


def _resolve_name(name: str | None) -> str:
    if name and name != "auto":
        if name not in VALID_PROFILES:
            raise ValueError(
                f"Unknown profile '{name}'. Valid: {VALID_PROFILES} or 'auto'."
            )
        return name
    env_name = os.environ.get("CFM_PROFILE")
    if env_name:
        if env_name not in VALID_PROFILES:
            raise ValueError(
                f"CFM_PROFILE='{env_name}' is not one of {VALID_PROFILES}."
            )
        return env_name
    return autodetect()


def _validate(cfg: ProfileConfig) -> None:
    """Sanity-check the loaded profile against the actual environment."""
    if cfg.is_gpu:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                f"Profile '{cfg.name}' requests CUDA but torch is not installed. "
                "Install with: poetry install --with neural"
            ) from e
        if not torch.cuda.is_available():
            logger.warning(
                "Profile '%s' requests CUDA but torch.cuda.is_available()=False. "
                "Neural training will fail; GBM models will silently fall back "
                "to CPU on most builds.",
                cfg.name,
            )


def load_profile(name: str | None = None, *, validate: bool = True) -> ProfileConfig:
    """Load (and cache) a profile.

    Parameters
    ----------
    name
        ``"cpu"`` | ``"gpu"`` | ``"cloud"`` | ``"auto"`` | ``None``.
        ``None``/``"auto"`` falls through to ``CFM_PROFILE`` env var, then
        :func:`autodetect`.
    validate
        Run environment sanity checks (e.g. CUDA availability). Set False in
        tests to load a GPU profile on a CPU-only host.
    """
    global _active
    resolved = _resolve_name(name)
    yaml_path = PROFILES_DIR / f"{resolved}.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Profile YAML not found: {yaml_path}")

    raw = OmegaConf.load(yaml_path)
    OmegaConf.resolve(raw)  # expands ${oc.env:...}
    cfg = ProfileConfig(name=resolved, raw=raw)
    if validate:
        _validate(cfg)
    _active = cfg
    logger.info("Loaded profile '%s' (accelerator=%s, gbm=%s, mlflow=%s)",
                cfg.name, cfg.accelerator_type, cfg.gbm_device, cfg.mlflow_tracking_uri)
    return cfg


def get_profile() -> ProfileConfig | None:
    """Return the currently-loaded profile, or None if none loaded yet."""
    return _active


def require_profile() -> ProfileConfig:
    """Like :func:`get_profile` but raises if nothing is loaded yet.

    Use this from library code that *must* run inside a profiled context.
    """
    if _active is None:
        return load_profile()  # auto
    return _active
