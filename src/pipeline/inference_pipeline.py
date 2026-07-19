"""
Phase 10: End-to-End Inference Pipeline
========================================

Assembles the three fitted metric models (CxG, CxA, CxT) into a single
scoring entry point.  Each model component is optional — if not provided
the corresponding metric column is simply omitted from the output.

Design
------
- ``InferencePipeline.score(events_df)`` → DataFrame with added metric columns
- Framework-agnostic: CxG model only needs ``predict_proba(df)``, the CxA and
  CxT components use their respective ``.score(df)`` pipeline methods.
- ``save`` / ``load`` use pickle for portability.
- ``from_config`` reads production model pointers from ``configs/models.yaml``
  and loads each pickled model from a caller-supplied models directory.

CxG scoring
-----------
The CxG model is a shot-level classifier.  ``score()`` filters ``events_df``
to rows where ``action_type == "shot"`` (configurable), predicts P(goal),
and left-joins the result back to the full DataFrame as column ``cxg``.

CxA scoring
-----------
``cxa_pipeline.score(events_df)`` is called directly; it internally filters
to creative actions and returns a DataFrame with a ``cxa`` column.  The result
is joined back by row-index / event_id.

CxT scoring
-----------
``cxt_pipeline.score(events_df)`` is called directly and returns a DataFrame
with a ``cxt`` column (NaN for non-CxT action types).  Joined back by index.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Column name written by InferencePipeline.score()
COL_CXG = "cxg"
COL_CXA = "cxa"
COL_CXT = "cxt"

# Default shot filter
DEFAULT_SHOT_ACTION_TYPE = "shot"


# ── Score helpers ─────────────────────────────────────────────────────────────


def _score_cxg(
    cxg_model,
    events_df: pd.DataFrame,
    action_type_col: str = "action_type",
    shot_type: str = DEFAULT_SHOT_ACTION_TYPE,
) -> pd.Series:
    """
    Return a Series indexed like ``events_df`` with CxG values.

    Shot rows are predicted; all other rows get NaN.
    """
    out = pd.Series(float("nan"), index=events_df.index, name=COL_CXG, dtype=float)
    if action_type_col in events_df.columns:
        shot_mask = events_df[action_type_col] == shot_type
        shot_df = events_df.loc[shot_mask]
    else:
        # No action_type column — score everything
        shot_df = events_df

    if shot_df.empty:
        return out

    proba = cxg_model.predict_proba(shot_df)
    # Support both single-column and two-column output
    if proba.ndim == 2 and proba.shape[1] >= 2:
        scores = proba[:, 1]
    else:
        scores = proba.ravel()

    out.loc[shot_df.index] = scores
    return out


def _join_pipeline_output(
    events_df: pd.DataFrame,
    scored: pd.DataFrame,
    col: str,
    event_id_col: str = "event_id",
) -> pd.Series:
    """
    Join a metric column from ``scored`` back to ``events_df``.

    Tries to join on ``event_id_col`` first; falls back to positional index.
    """
    out = pd.Series(float("nan"), index=events_df.index, name=col, dtype=float)

    if col not in scored.columns:
        return out

    if event_id_col in events_df.columns and event_id_col in scored.columns:
        mapping = scored.set_index(event_id_col)[col]
        out = events_df[event_id_col].map(mapping)
        out.name = col
        out.index = events_df.index
    else:
        # Positional fallback: align by index
        common_idx = events_df.index.intersection(scored.index)
        out.loc[common_idx] = scored.loc[common_idx, col]

    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────


@dataclass
class InferencePipelineConfig:
    """Lightweight config snapshot stored with the pipeline."""

    cxg_model_name: str | None = None
    cxa_model_name: str | None = None
    cxt_model_name: str | None = None
    action_type_col: str = "action_type"
    event_id_col: str = "event_id"
    shot_action_type: str = DEFAULT_SHOT_ACTION_TYPE
    extra: dict = field(default_factory=dict)


class InferencePipeline:
    """
    Assembled deployment pipeline for CxG / CxA / CxT scoring.

    Parameters
    ----------
    cxg_model : fitted estimator with ``predict_proba(df)`` | None
    cxa_pipeline : fitted CxAPipeline with ``score(df)`` | None
    cxt_pipeline : fitted CxTPipeline with ``score(df)`` | None
    config : InferencePipelineConfig | None
    """

    def __init__(
        self,
        cxg_model=None,
        cxa_pipeline=None,
        cxt_pipeline=None,
        config: InferencePipelineConfig | None = None,
    ) -> None:
        self.cxg_model = cxg_model
        self.cxa_pipeline = cxa_pipeline
        self.cxt_pipeline = cxt_pipeline
        self.config = config or InferencePipelineConfig()

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all events in ``events_df``, returning a copy with added columns.

        Added columns (each present only if the corresponding model is loaded):
          - ``cxg``  : P(goal) for shot rows; NaN elsewhere
          - ``cxa``  : CxA for creative action rows; NaN elsewhere
          - ``cxt``  : CxT value; NaN for non-CxT action types

        Parameters
        ----------
        events_df : DataFrame
            Must contain the columns expected by each sub-model.

        Returns
        -------
        DataFrame
            A copy of ``events_df`` with metric columns appended.

        Raises
        ------
        ValueError
            If ``events_df`` is empty.
        """
        if events_df.empty:
            raise ValueError("events_df is empty — nothing to score.")

        result = events_df.copy()

        # ── CxG ───────────────────────────────────────────────────────────────
        if self.cxg_model is not None:
            try:
                result[COL_CXG] = _score_cxg(
                    self.cxg_model,
                    result,
                    action_type_col=self.config.action_type_col,
                    shot_type=self.config.shot_action_type,
                )
                logger.debug("CxG scored: %d shot rows", result[COL_CXG].notna().sum())
            except Exception as exc:  # noqa: BLE001
                logger.warning("CxG scoring failed: %s", exc)
                result[COL_CXG] = float("nan")

        # ── CxA ───────────────────────────────────────────────────────────────
        if self.cxa_pipeline is not None:
            try:
                cxa_scored = self.cxa_pipeline.score(events_df)
                result[COL_CXA] = _join_pipeline_output(
                    result,
                    cxa_scored,
                    COL_CXA,
                    event_id_col=self.config.event_id_col,
                )
                logger.debug("CxA scored: %d rows", result[COL_CXA].notna().sum())
            except Exception as exc:  # noqa: BLE001
                logger.warning("CxA scoring failed: %s", exc)
                result[COL_CXA] = float("nan")

        # ── CxT ───────────────────────────────────────────────────────────────
        if self.cxt_pipeline is not None:
            try:
                cxt_scored = self.cxt_pipeline.score(events_df)
                result[COL_CXT] = _join_pipeline_output(
                    result,
                    cxt_scored,
                    COL_CXT,
                    event_id_col=self.config.event_id_col,
                )
                logger.debug("CxT scored: %d rows", result[COL_CXT].notna().sum())
            except Exception as exc:  # noqa: BLE001
                logger.warning("CxT scoring failed: %s", exc)
                result[COL_CXT] = float("nan")

        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Pickle the full pipeline to ``path``.

        Parameters
        ----------
        path : str | Path
            Target file path. Parent directories are created if necessary.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("InferencePipeline saved to %s", out)

    @classmethod
    def load(cls, path: str | Path) -> InferencePipeline:
        """
        Load a pickled InferencePipeline from ``path``.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        InferencePipeline

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        out = Path(path)
        if not out.exists():
            raise FileNotFoundError(f"InferencePipeline file not found: {out}")
        with open(out, "rb") as fh:
            obj = pickle.load(fh)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj).__name__}, expected InferencePipeline.")
        logger.info("InferencePipeline loaded from %s", out)
        return obj

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        models_dir: str | Path | None = None,
    ) -> InferencePipeline:
        """
        Build an InferencePipeline from ``configs/models.yaml``.

        Reads ``production.cxg``, ``production.cxa``, ``production.cxt``
        entries.  If a pointer is ``null`` / absent, that metric is skipped.
        Each non-null pointer is treated as a path relative to
        ``models_dir`` (default: the repository root, since the committed
        pointers are repo-root-relative like ``models/cxg/baseline_logit.joblib``).

        Parameters
        ----------
        config_path : str | Path
            Path to the YAML config file (e.g., ``configs/models.yaml``).
        models_dir : str | Path | None
            Base directory the pointers are resolved against.  Defaults to the
            repository root.

        Returns
        -------
        InferencePipeline

        Raises
        ------
        FileNotFoundError
            If a non-null pointer file does not exist in ``models_dir``.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for InferencePipeline.from_config(). "
                "Install with: pip install pyyaml"
            ) from exc

        config_path = Path(config_path)
        # Production pointers in configs/models.yaml are repo-root-relative
        # (e.g. "models/cxg/baseline_logit.joblib"), so resolve against the repo
        # root by default, not the config's own directory. Callers may override.
        repo_root = Path(__file__).resolve().parents[2]
        models_dir = Path(models_dir) if models_dir is not None else repo_root

        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        production = (cfg or {}).get("production", {}) or {}
        pip_config = InferencePipelineConfig()

        def _load_model(pointer: str | None, label: str):
            if not pointer:
                logger.info("Production %s model pointer is null — skipping.", label)
                return None
            model_file = models_dir / pointer
            if not model_file.exists():
                raise FileNotFoundError(f"Production {label} model not found at: {model_file}")
            if model_file.suffix == ".joblib":
                import joblib as _joblib

                return _joblib.load(model_file)
            with open(model_file, "rb") as fh:
                return pickle.load(fh)  # noqa: S301

        cxg_model = _load_model(production.get("cxg"), "CxG")
        cxa_pipeline = _load_model(production.get("cxa"), "CxA")
        cxt_pipeline = _load_model(production.get("cxt"), "CxT")

        # Auto-wrap bare state-value models in CxTPipeline (which provides .score())
        if cxt_pipeline is not None and not hasattr(cxt_pipeline, "score"):
            from src.models.cxt.cxt_pipeline import CxTPipeline as _CxTPipeline

            cxt_pipeline = _CxTPipeline(state_value_model=cxt_pipeline)
            logger.info("Wrapped CxT state-value model in CxTPipeline.")

        return cls(
            cxg_model=cxg_model,
            cxa_pipeline=cxa_pipeline,
            cxt_pipeline=cxt_pipeline,
            config=pip_config,
        )

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        parts = []
        if self.cxg_model is not None:
            parts.append(f"cxg={type(self.cxg_model).__name__}")
        if self.cxa_pipeline is not None:
            parts.append(f"cxa={type(self.cxa_pipeline).__name__}")
        if self.cxt_pipeline is not None:
            parts.append(f"cxt={type(self.cxt_pipeline).__name__}")
        return f"InferencePipeline({', '.join(parts) or 'empty'})"
