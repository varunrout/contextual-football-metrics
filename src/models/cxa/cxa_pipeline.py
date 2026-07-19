"""
CxA pipeline assembly — Phase 6c.

CxA = P(shot_created) × E(resulting_cxg_if_shot)

Combines Stage 1 (shot-creation classifier) and Stage 2 (shot-quality
regressor) into a single scoring pipeline that produces per-action CxA
predictions with a full decomposition record.

Key outputs
-----------
cxa_action  : CxA score for all creative actions (passes + carries + cutbacks)
cxa_pass    : CxA score for passes only

Decomposition record per action
--------------------------------
event_id, player_id, team_id, match_id, possession_id,
p_shot_created, expected_cxg_if_shot, cxa,
realised_cxa (= resulting_shot_cxg where a shot actually followed),
sequence_type, opponent_adjustment_delta,
action_type, x_location, y_location, minute

Usage
-----
    from src.models.cxa.cxa_pipeline import CxAPipeline

    pipeline = CxAPipeline(creation_model, quality_model)
    pipeline.fit(actions_df)
    scores_df = pipeline.score(actions_df)
    # or load from fitted models directly:
    pipeline = CxAPipeline.from_models(creation_model, quality_model)
    scores_df = pipeline.score(new_actions_df)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Action types treated as creative (eligible for CxA)
CREATIVE_ACTION_TYPES = frozenset({"pass", "cross", "carry", "cutback"})
PASS_ACTION_TYPES = frozenset({"pass", "cross"})

# Columns to carry through into the decomposition output
_PASSTHROUGH_COLS = [
    "event_id",
    "player_id",
    "team_id",
    "match_id",
    "possession_id",
    "action_type",
    "sequence_type",
    "x_location",
    "y_location",
    "minute",
    "home_or_away",
    "shot_created",  # observed label (where available)
    "resulting_shot_cxg",  # observed quality  (where available)
    "score_state",
    "transition_or_settled",
]


@dataclass
class CxADecompositionRecord:
    """Structured decomposition for a single creative action."""

    event_id: str | None
    player_id: str | None
    team_id: str | None
    match_id: str | None
    possession_id: str | None
    action_type: str
    p_shot_created: float
    expected_cxg_if_shot: float
    cxa: float
    realised_cxa: float | None  # = resulting_shot_cxg if shot followed, else None
    sequence_type: str | None
    opponent_adjustment_delta: float  # difference vs. opponent-agnostic expected CxG
    x_location: float | None
    y_location: float | None
    minute: float | None


class CxAPipeline:
    """
    Two-stage CxA scoring pipeline.

    Parameters
    ----------
    creation_model : fitted shot-creation classifier (has predict_proba)
    quality_model  : fitted shot-quality regressor   (has predict)
    """

    def __init__(self, creation_model, quality_model) -> None:
        self.creation_model = creation_model
        self.quality_model = quality_model
        self._is_fitted: bool = True  # models are expected pre-fitted

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_models(cls, creation_model, quality_model) -> CxAPipeline:
        """Construct from two already-fitted model objects."""
        return cls(creation_model, quality_model)

    def fit(
        self,
        actions_df: pd.DataFrame,
        creation_target: str = "shot_created",
        quality_target: str = "resulting_shot_cxg",
    ) -> CxAPipeline:
        """
        Fit both stages on the supplied actions DataFrame.

        For the quality model, only rows where a shot followed
        (shot_created == 1 or resulting_shot_cxg > 0) are used.
        """
        if actions_df.empty:
            raise ValueError("actions_df is empty")

        logger.info("CxAPipeline.fit: fitting shot-creation model …")
        self.creation_model.fit(actions_df, creation_target)

        logger.info("CxAPipeline.fit: fitting shot-quality model …")
        shot_rows = actions_df[
            actions_df.get(quality_target, pd.Series(0.0, index=actions_df.index)) > 0
        ]
        if shot_rows.empty:
            raise ValueError(
                f"No rows with {quality_target!r} > 0 found for quality model training"
            )
        self.quality_model.fit(shot_rows, quality_target)
        self._is_fitted = True
        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(
        self,
        actions_df: pd.DataFrame,
        action_type_col: str = "action_type",
        filter_creative: bool = True,
    ) -> pd.DataFrame:
        """
        Score all (creative) actions with CxA.

        Parameters
        ----------
        actions_df      : feature-store actions DataFrame
        action_type_col : column containing action type string
        filter_creative : if True, restrict to CREATIVE_ACTION_TYPES

        Returns
        -------
        DataFrame with one row per scored action:
          event_id, player_id, team_id, match_id, possession_id,
          action_type, p_shot_created, expected_cxg_if_shot, cxa,
          realised_cxa, sequence_type, opponent_adjustment_delta, …
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        df = actions_df.copy()
        if filter_creative and action_type_col in df.columns:
            mask = df[action_type_col].isin(CREATIVE_ACTION_TYPES)
            df = df[mask].copy()
            if df.empty:
                logger.warning("No creative actions found after filtering.")
                return pd.DataFrame()

        # Stage 1: shot-creation probability
        p_creation = self.creation_model.predict_proba(df)

        # Stage 2: expected CxG if a shot is taken
        expected_cxg = self.quality_model.predict(df)

        # CxA = product
        cxa = p_creation * expected_cxg

        # Build output
        out = pd.DataFrame(index=df.index)
        for col in _PASSTHROUGH_COLS:
            if col in df.columns:
                out[col] = df[col]

        out["p_shot_created"] = p_creation
        out["expected_cxg_if_shot"] = expected_cxg
        out["cxa"] = cxa

        # realised_cxa: actual resulting CxG where a shot followed
        if "resulting_shot_cxg" in df.columns:
            out["realised_cxa"] = df["resulting_shot_cxg"].where(
                df.get("shot_created", pd.Series(0, index=df.index)).astype(bool),
                other=np.nan,
            )
        else:
            out["realised_cxa"] = np.nan

        # Opponent adjustment delta: difference between CxA and a
        # hypothetical opponent-agnostic expected CxG (using mean quality).
        # Approximated as cxa - p_creation * mean(expected_cxg_if_shot).
        mean_quality = float(np.nanmean(expected_cxg))
        out["opponent_adjustment_delta"] = cxa - p_creation * mean_quality

        return out.reset_index(drop=True)

    def score_passes(self, actions_df: pd.DataFrame) -> pd.DataFrame:
        """Score passes + crosses only (cxa_pass)."""
        if "action_type" in actions_df.columns:
            pass_df = actions_df[actions_df["action_type"].isin(PASS_ACTION_TYPES)]
        else:
            pass_df = actions_df
        return self.score(pass_df, filter_creative=False)

    # ── Decomposition records ─────────────────────────────────────────────────

    def decompose(self, actions_df: pd.DataFrame) -> list[CxADecompositionRecord]:
        """Return a list of structured decomposition records."""
        scored = self.score(actions_df)
        records = []
        for _, row in scored.iterrows():
            records.append(
                CxADecompositionRecord(
                    event_id=row.get("event_id"),
                    player_id=row.get("player_id"),
                    team_id=row.get("team_id"),
                    match_id=row.get("match_id"),
                    possession_id=row.get("possession_id"),
                    action_type=str(row.get("action_type", "")),
                    p_shot_created=float(row["p_shot_created"]),
                    expected_cxg_if_shot=float(row["expected_cxg_if_shot"]),
                    cxa=float(row["cxa"]),
                    realised_cxa=float(row["realised_cxa"])
                    if not pd.isna(row.get("realised_cxa"))
                    else None,
                    sequence_type=row.get("sequence_type"),
                    opponent_adjustment_delta=float(row.get("opponent_adjustment_delta", 0.0)),
                    x_location=float(row["x_location"])
                    if "x_location" in row and pd.notna(row["x_location"])
                    else None,
                    y_location=float(row["y_location"])
                    if "y_location" in row and pd.notna(row["y_location"])
                    else None,
                    minute=float(row["minute"])
                    if "minute" in row and pd.notna(row["minute"])
                    else None,
                )
            )
        return records

    # ── MLflow logging ────────────────────────────────────────────────────────

    def log_to_mlflow(
        self,
        actions_df: pd.DataFrame,
        experiment_name: str = "cfm/cxa",
        creation_run_name: str = "cxa.shot_creation.v1",
        quality_run_name: str = "cxa.shot_quality.v1",
        creation_target: str = "shot_created",
        quality_target: str = "resulting_shot_cxg",
    ) -> None:
        """Log both stages as separate MLflow runs."""
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("mlflow not installed") from exc

        mlflow.set_experiment(experiment_name)
        # Stage 1
        if hasattr(self.creation_model, "evaluate") and creation_target in actions_df.columns:
            with mlflow.start_run(run_name=creation_run_name):
                m = self.creation_model.evaluate(actions_df, creation_target)
                mlflow.log_metrics(
                    {
                        "log_loss": m.log_loss,
                        "brier": m.brier,
                        **({} if m.auc is None else {"auc": m.auc}),
                        **({} if m.pr_auc is None else {"pr_auc": m.pr_auc}),
                    }
                )
                mlflow.log_param("model_class", type(self.creation_model).__name__)

        # Stage 2
        shot_df = actions_df[
            actions_df.get(quality_target, pd.Series(0.0, index=actions_df.index)) > 0
        ]
        if (
            hasattr(self.quality_model, "evaluate")
            and not shot_df.empty
            and quality_target in shot_df.columns
        ):
            with mlflow.start_run(run_name=quality_run_name):
                m = self.quality_model.evaluate(shot_df, quality_target)
                mlflow.log_metrics(
                    {
                        "mae": m.mae,
                        "rmse": m.rmse,
                        **({} if m.spearman is None else {"spearman": m.spearman}),
                    }
                )
                mlflow.log_param("model_class", type(self.quality_model).__name__)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> CxAPipeline:
        with open(path, "rb") as f:
            return pickle.load(f)
