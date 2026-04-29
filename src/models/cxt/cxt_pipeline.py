"""
CxT pipeline — Phase 7c.

CxT(action) = V(state_after) - V(state_before)

Where V(s) is a fitted StateValueModel (or ZoneValueModel fallback) that
maps a possession state to its expected future CxG.

For every pass and carry:
  - state_before: ball at (x_location, y_location) with full context
  - state_after: ball at (end_x, end_y) — same context columns

For unsuccessful actions (is_successful == False / carry_miscontrol,
incomplete pass), the state transitions to the opponent — CxT is negated.

CxTDecompositionRecord captures full per-action attribution info.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.cxt.feature_sets import AFTER_TO_BEFORE, BEFORE_TO_AFTER, CxTFeatureSetSpec

logger = logging.getLogger(__name__)

# Action types processed by CxT (passes + carries)
CXT_ACTION_TYPES = frozenset({"pass", "carry", "cross", "cutback"})


# ── Decomposition record ──────────────────────────────────────────────────────

@dataclass
class CxTDecompositionRecord:
    event_id: str | int
    player_id: str | int
    team_id: str | int
    match_id: str | int
    possession_id: str | int
    action_type: str
    x_before: float
    y_before: float
    x_after: float
    y_after: float
    v_before: float
    v_after: float
    cxt: float
    sequence_type: str = "unknown"
    opponent_adjustment_delta: float = 0.0
    is_successful: bool = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_after_state_df(actions_df: pd.DataFrame, feature_set: CxTFeatureSetSpec) -> pd.DataFrame:
    """
    Build a DataFrame where the after-state location columns have been
    renamed to the before-state names, so the state_value_model can
    score both states with the same feature set.

    Mapping applied (e.g.): end_x → x_location, end_y → y_location, etc.
    Context columns (opponent, minute, …) are carried through unchanged.
    """
    after_df = actions_df.copy()
    # Apply remapping: after-state col → before-state col name
    rename: dict[str, str] = {}
    for before_col, after_col in BEFORE_TO_AFTER.items():
        if after_col in after_df.columns and before_col in feature_set.all_features:
            rename[after_col] = before_col
    if rename:
        # Drop original before-state columns first to avoid collision
        cols_to_drop = [c for c in rename.values() if c in after_df.columns]
        after_df = after_df.drop(columns=cols_to_drop)
        after_df = after_df.rename(columns=rename)
    return after_df


def _is_successful_action(row: pd.Series) -> bool:
    """Determine if an action successfully transferred the ball to the after state."""
    outcome = str(row.get("outcome", "")).lower()
    action_type = str(row.get("action_type", "")).lower()
    # Carries: miscontrol or dispossession = unsuccessful
    if action_type == "carry":
        return outcome not in {"miscontrol", "dispossessed", "failed"}
    # Passes: incomplete = unsuccessful
    return outcome not in {"incomplete", "out", "failed", "blocked"}


# ── CxT Pipeline ─────────────────────────────────────────────────────────────

class CxTPipeline:
    """
    Two-stage CxT computation:
      1. state_value_model scores V(before) and V(after) for each action.
      2. CxT = V(after) - V(before); negated for unsuccessful actions.
    """

    def __init__(
        self,
        state_value_model,
        zone_value_model=None,
    ) -> None:
        self.state_value_model = state_value_model
        self.zone_value_model = zone_value_model

    @classmethod
    def from_models(cls, state_value_model, zone_value_model=None) -> "CxTPipeline":
        return cls(state_value_model=state_value_model, zone_value_model=zone_value_model)

    def fit(
        self,
        actions_df: pd.DataFrame,
        target_col: str = "possession_cxg",
        match_id_col: str = "match_id",
    ) -> "CxTPipeline":
        """Fit the underlying state_value_model on actions_df."""
        if actions_df.empty:
            raise ValueError("actions_df is empty")
        self.state_value_model.fit(actions_df, target_col)
        return self

    def score(
        self,
        actions_df: pd.DataFrame,
        filter_cxt_actions: bool = True,
    ) -> pd.DataFrame:
        """
        Score each action and return a DataFrame with CxT columns appended.

        Added columns:
          v_before, v_after, cxt, is_successful (if not present)

        Parameters
        ----------
        actions_df:
            Must contain x_location, y_location, end_x, end_y.
        filter_cxt_actions:
            If True, only rows whose action_type is in CXT_ACTION_TYPES
            are returned (others receive NaN CxT).
        """
        if self.state_value_model.pipeline is None and \
                getattr(self.state_value_model, "_torch_model", None) is None:
            raise RuntimeError("state_value_model not fitted. Call fit() first.")

        df = actions_df.copy()

        # ── V(before) ────────────────────────────────────────────────────────
        v_before = self.state_value_model.predict(df)

        # ── V(after) ─────────────────────────────────────────────────────────
        after_df = _build_after_state_df(df, self.state_value_model.feature_set)
        v_after = self.state_value_model.predict(after_df)

        df["v_before"] = v_before
        df["v_after"] = v_after
        df["cxt"] = v_after - v_before

        # ── Negate CxT for unsuccessful actions ───────────────────────────────
        if "is_successful" not in df.columns:
            df["is_successful"] = df.apply(_is_successful_action, axis=1)
        df.loc[~df["is_successful"], "cxt"] = -df.loc[~df["is_successful"], "cxt"].abs()

        # ── Mask out non-CxT action types ─────────────────────────────────────
        if filter_cxt_actions and "action_type" in df.columns:
            non_cxt = ~df["action_type"].isin(CXT_ACTION_TYPES)
            df.loc[non_cxt, ["v_before", "v_after", "cxt"]] = np.nan

        return df

    def decompose(
        self,
        actions_df: pd.DataFrame,
    ) -> list[CxTDecompositionRecord]:
        """
        Score actions and return a list of CxTDecompositionRecord objects.

        Uses zone_value_model baseline to compute opponent_adjustment_delta
        if available: delta = cxt - (v_zone_after - v_zone_before).
        """
        scored = self.score(actions_df, filter_cxt_actions=True)
        scored = scored.dropna(subset=["cxt"])

        records: list[CxTDecompositionRecord] = []
        for _, row in scored.iterrows():
            # Opponent adjustment delta (contextual CxT vs zone xT baseline)
            oad = 0.0
            if self.zone_value_model is not None:
                try:
                    x_b = float(row.get("x_location", 0))
                    y_b = float(row.get("y_location", 0))
                    x_a = float(row.get("end_x", x_b))
                    y_a = float(row.get("end_y", y_b))
                    zone_delta = (
                        self.zone_value_model.get_zone_value(x_a, y_a)
                        - self.zone_value_model.get_zone_value(x_b, y_b)
                    )
                    oad = float(row["cxt"]) - zone_delta
                except Exception:
                    oad = 0.0

            records.append(CxTDecompositionRecord(
                event_id=row.get("event_id", row.name),
                player_id=row.get("player_id", 0),
                team_id=row.get("team_id", 0),
                match_id=row.get("match_id", 0),
                possession_id=row.get("possession_id", 0),
                action_type=str(row.get("action_type", "")),
                x_before=float(row.get("x_location", 0.0)),
                y_before=float(row.get("y_location", 0.0)),
                x_after=float(row.get("end_x", row.get("x_location", 0.0))),
                y_after=float(row.get("end_y", row.get("y_location", 0.0))),
                v_before=float(row["v_before"]),
                v_after=float(row["v_after"]),
                cxt=float(row["cxt"]),
                sequence_type=str(row.get("sequence_type", "unknown")),
                opponent_adjustment_delta=oad,
                is_successful=bool(row.get("is_successful", True)),
            ))

        return records

    def log_to_mlflow(
        self,
        experiment_name: str = "cfm/cxt",
        run_name: str | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Log CxT pipeline params to MLflow (best-effort)."""
        try:
            import mlflow
        except ImportError:
            logger.warning("MLflow not installed; skipping log_to_mlflow")
            return
        mlflow.set_experiment(experiment_name)
        rn = run_name or f"cxt.{getattr(self.state_value_model, '__class__', type(self.state_value_model)).__name__}.v1"
        with mlflow.start_run(run_name=rn):
            mlflow.log_param("state_value_model", type(self.state_value_model).__name__)
            mlflow.log_param("zone_value_model", type(self.zone_value_model).__name__ if self.zone_value_model else "none")
            mlflow.log_param("cxt_action_types", sorted(CXT_ACTION_TYPES))
            if metrics:
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "CxTPipeline":
        with open(path, "rb") as f:
            return pickle.load(f)
