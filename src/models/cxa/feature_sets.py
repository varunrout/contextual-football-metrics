"""
CxA feature set definitions — action features for the two-stage CxA model.

Three nested tiers, parallel to the CxG feature sets:
  TRADITIONAL  — minimal features available for all competitions
                 (action location + action type flags, no opponent context)
  CONTEXTUAL   — traditional + opponent adjustment + match context +
                 sequence context + receiver context
                 (all competitions, no 360 required)
  FULL_360     — contextual + passer/receiver freeze-frame defensive context
                 (360 competitions only)

Unit of analysis: passes, crosses, carries and cutbacks.
Both stages (shot-creation classifier and resulting-CxG regressor) share
the same feature sets; the appropriate target column differs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CxAFeatureSetSpec:
    """Immutable specification of one CxA feature group."""

    name: str
    numeric: tuple[str, ...]
    boolean: tuple[str, ...]
    categorical: tuple[str, ...]
    requires_360: bool = False

    @property
    def all_features(self) -> list[str]:
        return list(self.numeric) + list(self.boolean) + list(self.categorical)

    @property
    def numeric_all(self) -> tuple[str, ...]:
        """Numeric + boolean (both passed through the numeric transformer)."""
        return self.numeric + self.boolean


# ── TRADITIONAL ───────────────────────────────────────────────────────────────

TRADITIONAL = CxAFeatureSetSpec(
    name="traditional",
    numeric=(
        # Action location
        "x_location",
        "y_location",
        # Pass / carry metrics
        "pass_length",
        "pass_angle",
        "progressive_distance",
        "end_x",
        "end_y",
        # Spatial derived
        "distance_to_goal",
        "end_distance_to_goal",
        "distance_gained",
    ),
    boolean=(
        "cross",
        "cutback",
        "through_ball",
        "switch",
        "central_progression",
        "box_entry",
        "under_pressure",
    ),
    categorical=(
        "action_type",  # pass / carry / cross / cutback
        "pass_height",
        "pass_body_part",
        "set_piece_type",
    ),
    requires_360=False,
)

# ── CONTEXTUAL ────────────────────────────────────────────────────────────────

CONTEXTUAL = CxAFeatureSetSpec(
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
        # Receiver context
        "receiver_distance_to_goal",
        "receiver_x",
        "receiver_y",
    ),
    boolean=TRADITIONAL.boolean
    + (
        "knockout_or_group",
        "set_piece_flag",
        "counterpress_regain_flag",
        "receiver_in_box",
        "receiver_is_central",
    ),
    categorical=TRADITIONAL.categorical
    + (
        "score_state",
        "home_or_away",
        "sequence_type",
        "possession_start_zone",
        "transition_or_settled",
        "phase_of_play",
    ),
    requires_360=False,
)

# ── FULL_360 ──────────────────────────────────────────────────────────────────

FULL_360 = CxAFeatureSetSpec(
    name="full_360",
    numeric=CONTEXTUAL.numeric
    + (
        # Passer defensive context (360)
        "nearest_defender_distance",
        "defenders_within_5m",
        "defensive_density_in_box",
        # Receiver space context (360)
        "receiver_nearest_defender_distance",
        "receiver_defenders_within_5m",
        "open_passing_lanes",
        "passing_lane_blockage_proxy",
    ),
    boolean=CONTEXTUAL.boolean + ("has_360",),
    categorical=CONTEXTUAL.categorical,
    requires_360=True,
)

# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, CxAFeatureSetSpec] = {
    "traditional": TRADITIONAL,
    "contextual": CONTEXTUAL,
    "full_360": FULL_360,
}


def get_feature_set(name: str) -> CxAFeatureSetSpec:
    """Return a CxAFeatureSetSpec by name. Raises ValueError for unknown names."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown CxA feature set {name!r}. Choose from: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
