"""
CxG feature set definitions for the model ladder.

Three nested tiers:
  TRADITIONAL  — minimal features available for all competitions
                 (location, shot type, body part — no opponent context, no 360)
  CONTEXTUAL   — traditional + opponent adjustment + match context + sequence context
                 (all competitions, no 360 required)
  FULL_360     — contextual + freeze-frame 360 features
                 (360 competitions only: WC 2022, Euro 2020/2024)

Usage
-----
    from src.models.cxg.feature_sets import get_feature_set, CONTEXTUAL

    spec = get_feature_set("contextual")
    numeric_cols = spec.numeric        # tuple of pure numeric feature names
    bool_cols    = spec.boolean        # tuple of bool features (cast to float)
    cat_cols     = spec.categorical    # tuple of categorical features
    all_cols     = spec.all_features   # ordered list: numeric + boolean + categorical
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSetSpec:
    """Immutable specification of one CxG feature group."""

    name: str
    numeric: tuple[str, ...]
    boolean: tuple[str, ...]
    categorical: tuple[str, ...]
    requires_360: bool = False

    @property
    def all_features(self) -> list[str]:
        """Ordered list: numeric + boolean + categorical."""
        return list(self.numeric) + list(self.boolean) + list(self.categorical)

    @property
    def numeric_all(self) -> tuple[str, ...]:
        """Numeric + boolean (both handled by the numeric transformer)."""
        return self.numeric + self.boolean


# ── TRADITIONAL ───────────────────────────────────────────────────────────────

TRADITIONAL = FeatureSetSpec(
    name="traditional",
    numeric=(
        "distance_to_goal",
        "shot_angle",
        "x_location",
        "y_location",
    ),
    boolean=(
        "header",
        "volley",
        "first_time_shot",
        "open_play",
        "under_pressure",
    ),
    categorical=(
        "body_part",
        "shot_type",
        "set_piece_type",
    ),
    requires_360=False,
)

# ── CONTEXTUAL ────────────────────────────────────────────────────────────────

CONTEXTUAL = FeatureSetSpec(
    name="contextual",
    numeric=TRADITIONAL.numeric
    + (
        # Opponent quality adjustment
        "opponent_xg_conceded_rolling_5",
        "opponent_shots_conceded_rolling_5",
        "opponent_defensive_rating",
        "opponent_keeper_shot_stopping_rating",
        "opponent_team_strength",
        # Match context
        "minute",
        "score_differential",
        # Sequence / possession context
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
    ),
    categorical=TRADITIONAL.categorical
    + (
        "score_state",
        "home_or_away",
        "sequence_type",
        "possession_start_zone",
        "transition_or_settled",
    ),
    requires_360=False,
)

# ── FULL_360 ──────────────────────────────────────────────────────────────────

FULL_360 = FeatureSetSpec(
    name="full_360",
    numeric=CONTEXTUAL.numeric
    + (
        # Freeze-frame defensive context
        "nearest_defender_distance",
        "second_nearest_defender_distance",
        "defenders_within_5m",
        "defenders_between_ball_and_goal",
        "keeper_distance_to_goal",
        "keeper_distance_to_shooter",
        "keeper_angle_coverage",
        "shot_lane_blockage_proxy",
        "defensive_density_in_box",
    ),
    boolean=CONTEXTUAL.boolean + ("has_360",),
    categorical=CONTEXTUAL.categorical,
    requires_360=True,
)

# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, FeatureSetSpec] = {
    "traditional": TRADITIONAL,
    "contextual": CONTEXTUAL,
    "full_360": FULL_360,
}


def get_feature_set(name: str) -> FeatureSetSpec:
    """Return a FeatureSetSpec by name. Raises ValueError for unknown names."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown feature set {name!r}. Choose from: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
