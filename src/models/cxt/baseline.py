"""Phase 4 baseline xT model (zone-based expected threat)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ZoneBaselineConfig:
    pitch_zones_x: int = 16
    pitch_zones_y: int = 12
    laplace_smoothing_alpha: float = 1.0
    max_iterations: int = 200
    tolerance: float = 1e-6


class ZoneXTBaseline:
    """Zone-based xT using move/shot frequencies and transition dynamics."""

    def __init__(self, config: ZoneBaselineConfig | None = None) -> None:
        self.config = config or ZoneBaselineConfig()
        self.values_: np.ndarray | None = None
        self.shot_prob_: np.ndarray | None = None
        self.shot_value_: np.ndarray | None = None
        self.transition_: np.ndarray | None = None

    def _zone_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bx = np.clip((x / 105.0 * self.config.pitch_zones_x).astype(int), 0, self.config.pitch_zones_x - 1)
        by = np.clip((y / 68.0 * self.config.pitch_zones_y).astype(int), 0, self.config.pitch_zones_y - 1)
        return (by * self.config.pitch_zones_x) + bx

    @property
    def n_zones(self) -> int:
        return self.config.pitch_zones_x * self.config.pitch_zones_y

    def fit(
        self,
        events_df: pd.DataFrame,
        goal_col: str = "goal",
        event_type_col: str = "event_type",
    ) -> "ZoneXTBaseline":
        required = {"x_location", "y_location", event_type_col}
        if not required.issubset(events_df.columns):
            missing = required - set(events_df.columns)
            raise ValueError(f"Missing columns for xT baseline: {sorted(missing)}")

        df = events_df.copy()
        start_zone = self._zone_index(
            pd.to_numeric(df["x_location"], errors="coerce").fillna(0.0).to_numpy(),
            pd.to_numeric(df["y_location"], errors="coerce").fillna(0.0).to_numpy(),
        )

        n = self.n_zones
        alpha = self.config.laplace_smoothing_alpha

        zone_actions = np.zeros(n, dtype=float)
        zone_shots = np.zeros(n, dtype=float)
        zone_goals = np.zeros(n, dtype=float)
        transitions = np.full((n, n), alpha, dtype=float)

        etype = df[event_type_col].astype(str).to_numpy()
        goals = df[goal_col].astype(bool).to_numpy() if goal_col in df.columns else np.zeros(len(df), dtype=bool)

        has_end = {"end_x", "end_y"}.issubset(df.columns)
        if has_end:
            end_zone = self._zone_index(
                pd.to_numeric(df["end_x"], errors="coerce").fillna(0.0).to_numpy(),
                pd.to_numeric(df["end_y"], errors="coerce").fillna(0.0).to_numpy(),
            )
        else:
            end_zone = start_zone

        for i in range(len(df)):
            sz = int(start_zone[i])
            zone_actions[sz] += 1.0
            if etype[i] == "shot":
                zone_shots[sz] += 1.0
                zone_goals[sz] += float(goals[i])
            elif etype[i] in {"pass", "carry"}:
                ez = int(end_zone[i])
                transitions[sz, ez] += 1.0

        zone_actions = np.maximum(zone_actions, 1.0)
        shot_prob = zone_shots / zone_actions

        # Empirical goal probability by origin zone, smoothed.
        shot_value = (zone_goals + alpha) / (zone_shots + (2.0 * alpha))

        transition = transitions / transitions.sum(axis=1, keepdims=True)

        values = np.zeros(n, dtype=float)
        for _ in range(self.config.max_iterations):
            updated = shot_prob * shot_value + (1.0 - shot_prob) * (transition @ values)
            if float(np.max(np.abs(updated - values))) < self.config.tolerance:
                values = updated
                break
            values = updated

        self.values_ = values
        self.shot_prob_ = shot_prob
        self.shot_value_ = shot_value
        self.transition_ = transition
        return self

    def predict_state_value(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.values_ is None:
            raise RuntimeError("Model not fitted")
        zone = self._zone_index(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self.values_[zone]

    def predict_action_delta(self, actions_df: pd.DataFrame) -> np.ndarray:
        """For moves: xT(end)-xT(start); for shots: xT(start)."""
        if self.values_ is None:
            raise RuntimeError("Model not fitted")

        start = self.predict_state_value(
            pd.to_numeric(actions_df["x_location"], errors="coerce").fillna(0.0).to_numpy(),
            pd.to_numeric(actions_df["y_location"], errors="coerce").fillna(0.0).to_numpy(),
        )

        etype = actions_df["event_type"].astype(str)
        if {"end_x", "end_y"}.issubset(actions_df.columns):
            end = self.predict_state_value(
                pd.to_numeric(actions_df["end_x"], errors="coerce").fillna(0.0).to_numpy(),
                pd.to_numeric(actions_df["end_y"], errors="coerce").fillna(0.0).to_numpy(),
            )
        else:
            end = start

        move_mask = etype.isin(["pass", "carry"]).to_numpy()
        out = np.where(move_mask, end - start, start)
        return out


def filter_xt_actions(features_df: pd.DataFrame) -> pd.DataFrame:
    if "event_type" not in features_df.columns:
        return features_df.copy()
    return features_df[features_df["event_type"].astype(str).isin(["pass", "carry", "shot"])].copy()
