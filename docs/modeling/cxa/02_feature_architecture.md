# CxA — Feature Architecture

## 1. Overview

Features are organised into three nested tiers of increasing contextual richness,
defined in `src/models/cxa/feature_sets.py` as `CxAFeatureSetSpec` objects.
Each tier is a strict superset of the previous one:

```
TRADITIONAL  ⊂  CONTEXTUAL  ⊂  FULL_360
```

Both stages of the CxA pipeline (Stage 1 — shot-creation classifier; Stage 2 —
shot-quality regressor) consume the same feature set. The feature set is
selected at training time; the current production model uses **CONTEXTUAL**.

---

## 2. Tier 1 — Traditional

The minimal feature set for any competition with basic event data.
Replicates the action-location inputs used in canonical pass-value and
carry-value models.

### 2.1 Action Location and Destination

| Feature | Type | Description |
|---|---|---|
| `x_location` | continuous | Ball x-coordinate at action start (0–105 m) |
| `y_location` | continuous | Ball y-coordinate at action start (0–68 m) |
| `end_x` | continuous | Destination x-coordinate (pass end / carry end) |
| `end_y` | continuous | Destination y-coordinate |
| `distance_to_goal` | continuous | Euclidean distance from action start to goal centre (m) |
| `end_distance_to_goal` | continuous | Distance from action destination to goal centre (m) |
| `distance_gained` | continuous | Net reduction in distance to goal: `distance_to_goal − end_distance_to_goal` |

`distance_gained` captures the spatial threat progression of an action:
a positive value indicates the ball moved toward the goal. This is the
dominant predictor of shot-creation probability in the traditional tier.

### 2.2 Action Metrics

| Feature | Type | Description |
|---|---|---|
| `pass_length` | continuous | Euclidean length of pass (m; 0 for carries) |
| `pass_angle` | continuous | Direction of pass relative to goal (radians) |
| `progressive_distance` | continuous | Metres gained toward goal by the pass |
| `carry_distance` | continuous | Total carry distance (m) |
| `carry_progressive_distance` | continuous | Progressive metres gained during carry |

### 2.3 Action-Type Flags (Boolean)

| Feature | Description |
|---|---|
| `cross` | Delivery from a wide position |
| `cutback` | Pull-back from the byline into the box |
| `through_ball` | Pass played between or behind the defensive line |
| `switch` | Long lateral switch of play |
| `central_progression` | Pass ending in central corridor (y ∈ 27–41 m) |
| `box_entry` | Action whose destination is inside the penalty box |
| `under_pressure` | Passer/carrier pressured by an opponent |

### 2.4 Categorical Descriptors

| Feature | Values | Description |
|---|---|---|
| `action_type` | `pass`, `carry`, `cross`, `cutback` | Type of creative action |
| `pass_height` | `Ground Pass`, `Low Pass`, `High Pass` | Ball height at delivery |
| `pass_body_part` | `Right Foot`, `Left Foot`, `Head`, `No Touch` | Delivery body part |
| `set_piece_type` | `none`, `corner`, `free_kick`, `throw_in`, `kick_off`, `penalty` | Set-piece variant |

---

## 3. Tier 2 — Contextual (Traditional + 3 Adjustment Groups)

The contextual tier adds opponent quality, match state, sequence context, and
receiver context. All features are available without 360 freeze-frame data.

### 3.1 Opponent Quality

The defending team's organisation and quality systematically affects the
probability that any given pass leads to a shot. A through-ball against a
high defensive line is more likely to produce a shot than the same pass
against a well-organised low block.

| Feature | Type | Description |
|---|---|---|
| `opponent_xg_conceded_rolling_5` | continuous | Rolling 5-match mean xG conceded by the defending team (pre-match) |
| `opponent_shots_conceded_rolling_5` | continuous | Rolling 5-match mean shots conceded (pre-match) |
| `opponent_defensive_rating` | continuous | Composite defensive quality rating (normalised 0–1) |
| `opponent_team_strength` | continuous | Overall team ELO / strength signal |

All rolling statistics are computed strictly over completed matches prior to the
current match, satisfying temporal integrity constraint **C2**.

### 3.2 Match State

Score state and match minute alter both the attacking team's urgency and the
defending team's compactness. A team that is losing in the final 10 minutes
will press higher, creating more space behind for through-balls. A team that
is winning will park and deny the central passing lanes.

| Feature | Type | Description |
|---|---|---|
| `minute` | continuous | Match minute at time of action |
| `score_differential` | continuous | Attacking team goals minus opponent goals |
| `score_state` | categorical | `winning`, `drawing`, `losing` |
| `home_or_away` | categorical | `home`, `away`, `neutral` |
| `knockout_or_group` | boolean | True if elimination stage |

### 3.3 Sequence / Possession Context

The structure of the possession containing the action carries information about
attacking intent and defensive positioning at the moment of the action.
A carry that initiates a fast counter-attack is structurally different from the
same carry in a slow 20-pass build-up.

| Feature | Type | Description |
|---|---|---|
| `events_before_action` | continuous | Events in possession chain prior to this action |
| `passes_before_action` | continuous | Passes in possession chain prior to this action |
| `carries_before_action` | continuous | Carries in possession chain prior to this action |
| `time_from_possession_start` | continuous | Seconds elapsed since possession began |
| `vertical_progression_speed` | continuous | Net pitch-vertical metres gained per second in possession |
| `directness` | continuous | Displacement ÷ total path length (1 = perfectly direct) |
| `set_piece_flag` | boolean | Possession originated from a set piece |
| `counterpress_regain_flag` | boolean | Possession started via counterpress regain within 5 s |
| `sequence_type` | categorical | `counter_attack`, `build_up`, `set_piece`, `direct`, `unknown` |
| `possession_start_zone` | categorical | Pitch zone where possession began |
| `transition_or_settled` | categorical | `transition`, `settled` |
| `phase_of_play` | categorical | Broader phase classification |

### 3.4 Receiver Context

Unlike shot models (where there is no receiver), action-value models can
condition on **where the ball arrives** and in what space. A pass that delivers
the ball to a striker with their back to goal 30 m out is qualitatively
different from a pass into the path of a runner in behind the defence.

| Feature | Type | Description |
|---|---|---|
| `receiver_x` | continuous | x-coordinate of the action's destination / receiver position |
| `receiver_y` | continuous | y-coordinate of the action's destination |
| `receiver_distance_to_goal` | continuous | Distance from receiver's position to goal centre (m) |
| `receiver_in_box` | boolean | True if receiver is inside the penalty box |
| `receiver_is_central` | boolean | True if receiver is in the central corridor |

The receiver features bridge Stage 1 and Stage 2 conceptually: `receiver_in_box`
is one of the strongest predictors of both shot-creation probability and
resulting shot quality.

---

## 4. Tier 3 — Full 360 (Contextual + Freeze-Frame)

Available only for 360-data competitions. Extends the contextual tier with
spatial defensive context drawn from the freeze-frame snapshot at the moment
of the action.

### 4.1 Passer / Carrier Defensive Pressure

| Feature | Type | Description |
|---|---|---|
| `nearest_defender_distance` | continuous | Distance to the nearest opponent at action time (m) |
| `defenders_within_5m` | integer | Opponents within 5 m of the ball at action time |
| `defensive_density_in_box` | continuous | Defenders per 100 m² in the penalty box |

### 4.2 Receiver Space (Freeze-Frame)

| Feature | Type | Description |
|---|---|---|
| `receiver_nearest_defender_distance` | continuous | Distance from receiver to the nearest opponent |
| `receiver_defenders_within_5m` | integer | Opponents within 5 m of the receiver |
| `open_passing_lanes` | continuous | Number of unobstructed passing lanes ahead |
| `passing_lane_blockage_proxy` | continuous | Fraction of forward lanes blocked by defenders |
| `has_360` | boolean | Routing flag: True only for 360 competitions |

The Full 360 tier is defined for future use. Current training uses only
**CONTEXTUAL** because the training competitions (WC 2022, Euro 2020) do not
have universal freeze-frame coverage.

---

## 5. Feature Engineering Notes

### 5.1 Location Encoding

Pitch coordinates are stored and consumed in the **internal 105×68 metre
frame**. Features derived from location (`distance_to_goal`, `progressive_distance`,
`distance_gained`) are all computed in metres, making them interpretable as
physical distances on the pitch.

### 5.2 Receiver Features for Carries

For carry events, `receiver_x/receiver_y` are set to the carry destination
(`end_x`, `end_y`). The carrier is themselves the "receiver". This ensures
Stage 1 and Stage 2 both benefit from destination-based receiver features
regardless of action type.

### 5.3 Missing Value Policy

Categorical features use `impute_constant` (unknown class). Continuous features
use `impute_median`. Boolean flags use `impute_zero`. The `has_360` column
routes 360-specific features to `flag_and_zero` when freeze-frame is
unavailable, adding a companion `has_{col}` indicator to preserve missingness
information without dropping rows.
