"""
CxT feature set definitions — state features for the contextual threat-value model.

Three nested tiers:
  TRADITIONAL  — ball location + action type + possession zone (no 360)
  CONTEXTUAL   — traditional + sequence context + opponent adjustment + match state
  FULL_360     — contextual + 360 before/after defensive density features

Unit of analysis: every pass and carry (the events for which before/after
state-value differences are meaningful).

Key design principle
--------------------
The state value model V(s) is trained on the BEFORE-state features.
To compute V(after), the CxT pipeline remaps end-position columns back
into the same feature names (end_x → x_location, etc.) so the same
model can score both states.

Before-state columns → after-state column equivalents (for remapping):
  x_location                → end_x
  y_location                → end_y
  distance_to_goal          → end_distance_to_goal
  in_box                    → end_in_box
  is_central                → end_is_central
  under_pressure            → after_under_pressure
  (360) nearest_defender_distance       → after_nearest_defender_distance
  (360) defenders_within_5m             → after_defenders_within_5m
  (360) defensive_density_in_box        → after_defensive_density_in_box
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CxTFeatureSetSpec:
    """Immutable specification of one CxT state-value feature group."""

    name: str
    numeric: tuple[str, ...]  # continuous before-state features
    boolean: tuple[str, ...]  # binary before-state flags
    categorical: tuple[str, ...]  # categorical context features (shared)
    after_numeric: tuple[
        str, ...
    ]  # after-state numeric equivalents (same order as numeric overlap)
    after_boolean: tuple[str, ...]  # after-state boolean equivalents
    requires_360: bool = False

    @property
    def all_features(self) -> list[str]:
        return list(self.numeric) + list(self.boolean) + list(self.categorical)

    @property
    def numeric_all(self) -> tuple[str, ...]:
        """Numeric + boolean (both processed by the numeric transformer)."""
        return self.numeric + self.boolean

    @property
    def all_after_features(self) -> list[str]:
        return list(self.after_numeric) + list(self.after_boolean)


# Column mapping: before-state → after-state (used by CxTPipeline)
BEFORE_TO_AFTER: dict[str, str] = {
    "x_location": "end_x",
    "y_location": "end_y",
    "distance_to_goal": "end_distance_to_goal",
    "in_box": "end_in_box",
    "is_central": "end_is_central",
    "under_pressure": "after_under_pressure",
    # 360 state columns
    "nearest_defender_distance": "after_nearest_defender_distance",
    "defenders_within_5m": "after_defenders_within_5m",
    "defensive_density_in_box": "after_defensive_density_in_box",
}

# Reverse mapping for convenience
AFTER_TO_BEFORE: dict[str, str] = {v: k for k, v in BEFORE_TO_AFTER.items()}


# ── TRADITIONAL ───────────────────────────────────────────────────────────────

TRADITIONAL = CxTFeatureSetSpec(
    name="traditional",
    numeric=(
        # Before-state location
        "x_location",
        "y_location",
        "distance_to_goal",
        # Action magnitude
        "progressive_distance",
        "pass_length",
    ),
    boolean=(
        "in_box",
        "is_central",
        "under_pressure",
        "box_entry",
        "cross",
        "cutback",
    ),
    categorical=(
        "action_type",  # pass / carry / cross / cutback
        "possession_start_zone",
    ),
    after_numeric=(
        "end_x",
        "end_y",
        "end_distance_to_goal",
        "progressive_distance",  # same value — action already happened
        "pass_length",
    ),
    after_boolean=(
        "end_in_box",
        "end_is_central",
        "after_under_pressure",
        "box_entry",  # box_entry = end is in box (reuse)
        "cross",
        "cutback",
    ),
    requires_360=False,
)

# ── CONTEXTUAL ────────────────────────────────────────────────────────────────

CONTEXTUAL = CxTFeatureSetSpec(
    name="contextual",
    numeric=TRADITIONAL.numeric
    + (
        # Opponent quality
        "opponent_xg_conceded_rolling_5",
        "opponent_shots_conceded_rolling_5",
        "opponent_defensive_rating",
        "opponent_team_strength",
        # Match context
        "minute",
        "score_differential",
        # Possession / sequence context
        "events_before_action",
        "passes_before_action",
        "carries_before_action",
        "time_from_possession_start",
        "vertical_progression_speed",
        "directness",
    ),
    boolean=TRADITIONAL.boolean
    + (
        "knockout_or_group",
        "set_piece_flag",
        "counterpress_regain_flag",
        "central_progression",
        "through_ball",
        "switch",
    ),
    categorical=TRADITIONAL.categorical
    + (
        "score_state",
        "home_or_away",
        "sequence_type",
        "transition_or_settled",
        "phase_of_play",
    ),
    after_numeric=TRADITIONAL.after_numeric
    + (
        # Opponent / match context doesn't change within one action
        "opponent_xg_conceded_rolling_5",
        "opponent_shots_conceded_rolling_5",
        "opponent_defensive_rating",
        "opponent_team_strength",
        "minute",
        "score_differential",
        "events_before_action",
        "passes_before_action",
        "carries_before_action",
        "time_from_possession_start",
        "vertical_progression_speed",
        "directness",
    ),
    after_boolean=TRADITIONAL.after_boolean
    + (
        "knockout_or_group",
        "set_piece_flag",
        "counterpress_regain_flag",
        "central_progression",
        "through_ball",
        "switch",
    ),
    requires_360=False,
)

# ── FULL_360 ──────────────────────────────────────────────────────────────────

FULL_360 = CxTFeatureSetSpec(
    name="full_360",
    numeric=CONTEXTUAL.numeric
    + (
        # Before-state 360 features
        "nearest_defender_distance",
        "defenders_within_5m",
        "defensive_density_in_box",
    ),
    boolean=CONTEXTUAL.boolean + ("has_360",),
    categorical=CONTEXTUAL.categorical,
    after_numeric=CONTEXTUAL.after_numeric
    + (
        # After-state 360 features
        "after_nearest_defender_distance",
        "after_defenders_within_5m",
        "after_defensive_density_in_box",
    ),
    after_boolean=CONTEXTUAL.after_boolean + ("has_360",),
    requires_360=True,
)


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, CxTFeatureSetSpec] = {
    "traditional": TRADITIONAL,
    "contextual": CONTEXTUAL,
    "full_360": FULL_360,
}


def get_feature_set(name: str) -> CxTFeatureSetSpec:
    """Return a CxTFeatureSetSpec by name. Raises ValueError for unknown names."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown CxT feature set {name!r}. Choose from: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
