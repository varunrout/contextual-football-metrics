# CxT — Feature Architecture

## 1. Overview

CxT features describe **game states** rather than discrete event properties.
The model learns $V(s)$ — the expected threat value of a possession state — so
features must capture: where the ball is, how dangerous the field position is,
who is defending, and what the match situation is.

Features are organised into three nested tiers defined in
`src/models/cxt/feature_sets.py` as `CxTFeatureSetSpec` objects:

```
TRADITIONAL  ⊂  CONTEXTUAL  ⊂  FULL_360
```

Each tier is a strict superset of the previous one. The current training run
uses the **CONTEXTUAL** tier for all candidates.

---

## 2. Before-State and After-State Design

A single `CxTFeatureSetSpec` defines **two feature sets**:

- **Before-state** (`numeric`, `boolean`, `categorical`): the game state before
  the action is executed. The model is trained on these.
- **After-state** (`after_numeric`, `after_boolean`): the game state after the
  action lands. Used only at score time, not during training.

At prediction time, to compute $V(s_{\text{after}})$, the pipeline remaps
after-state column names to their before-state equivalents using
`BEFORE_TO_AFTER`:

| Before-state column | After-state column |
|---|---|
| `x_location` | `end_x` |
| `y_location` | `end_y` |
| `distance_to_goal` | `end_distance_to_goal` |
| `in_box` | `end_in_box` |
| `is_central` | `end_is_central` |
| `under_pressure` | `after_under_pressure` |
| `nearest_defender_distance` | `after_nearest_defender_distance` (360 only) |
| `defenders_within_5m` | `after_defenders_within_5m` (360 only) |
| `defensive_density_in_box` | `after_defensive_density_in_box` (360 only) |

This design means a single trained model $V(\cdot)$ can score both states
without any architectural duplication. The CxT score is $V(\text{after}) - V(\text{before})$.

Context features that do not change within a single action (opponent ratings,
match minute, score state) take the same value in both before and after states.

---

## 3. Tier 1 — Traditional

Minimal state features available from any event-data provider.

### 3.1 Location

| Feature | Type | Description |
|---|---|---|
| `x_location` | continuous | Longitudinal pitch coordinate (StatsBomb internal: 0–105) |
| `y_location` | continuous | Lateral pitch coordinate (StatsBomb internal: 0–68) |
| `distance_to_goal` | continuous | Euclidean distance from current position to centre of goal (m) |

These three features are the primary determinants of state value. Distance to
goal has a near-monotonic relationship with `possession_cxg` among positive
values: actions closer to goal precede more dangerous shot opportunities.

### 3.2 Action Magnitude

| Feature | Type | Description |
|---|---|---|
| `progressive_distance` | continuous | Metres gained toward the opponent goal by this action |
| `pass_length` | continuous | Length of the pass (0 for carries) |

### 3.3 Boolean State Flags

| Feature | Type | Description |
|---|---|---|
| `in_box` | boolean | Ball location is within the penalty area |
| `is_central` | boolean | Ball is in the central channel (within 15m of pitch centre) |
| `under_pressure` | boolean | Receiving player is under immediate defensive pressure |
| `box_entry` | boolean | Action ends inside the penalty area (end-state flag) |
| `cross` | boolean | Action is a cross (wide delivery) |
| `cutback` | boolean | Action is a cutback from the byline |

### 3.4 Categorical Context

| Feature | Type | Description |
|---|---|---|
| `action_type` | categorical | `pass`, `carry`, `cross`, `cutback` |
| `possession_start_zone` | categorical | Pitch zone where the possession started |

---

## 4. Tier 2 — Contextual

The contextual tier adds three adjustment groups identical in structure to
those used in CxG and CxA, plus additional sequence-level features.

### 4.1 Opponent Quality

| Feature | Type | Description |
|---|---|---|
| `opponent_xg_conceded_rolling_5` | continuous | Rolling 5-match mean xG conceded by defending team (pre-match) |
| `opponent_shots_conceded_rolling_5` | continuous | Rolling 5-match mean shots conceded (pre-match) |
| `opponent_defensive_rating` | continuous | Composite defensive quality rating (normalised 0–1) |
| `opponent_team_strength` | continuous | Overall team ELO / strength signal |

All rolling statistics are computed strictly over completed prior matches —
no future match data contaminates the training features.

### 4.2 Match State

| Feature | Type | Description |
|---|---|---|
| `minute` | continuous | Match minute of the action |
| `score_differential` | continuous | Possessing team goals minus opponent goals |
| `score_state` | categorical | `winning`, `drawing`, `losing` |
| `home_or_away` | categorical | `home`, `away`, `neutral` |
| `knockout_or_group` | boolean | True if elimination stage |
| `set_piece_flag` | boolean | Possession started from a set piece |

### 4.3 Sequence and Possession Context

| Feature | Type | Description |
|---|---|---|
| `events_before_action` | continuous | Events in possession chain before this action |
| `passes_before_action` | continuous | Passes in possession chain before this action |
| `carries_before_action` | continuous | Carries in possession chain before this action |
| `time_from_possession_start` | continuous | Seconds elapsed since possession began |
| `vertical_progression_speed` | continuous | Pitch-vertical metres gained per second in possession |
| `directness` | continuous | Displacement-to-path-length ratio (1 = perfectly direct) |
| `sequence_type` | categorical | `counter_attack`, `build_up`, `set_piece`, `direct`, `unknown` |
| `transition_or_settled` | categorical | `transition`, `settled` |
| `phase_of_play` | categorical | Broader phase classification |
| `counterpress_regain_flag` | boolean | Possession started via counterpress regain within 5 seconds |
| `central_progression` | boolean | Action moves the ball centrally toward goal |
| `through_ball` | boolean | Pass played between defensive lines |
| `switch` | boolean | Action switches play across the pitch |

---

## 5. Tier 3 — Full 360

Requires StatsBomb 360 freeze-frame data. Adds spatial defensive density
features for both before and after states. Not activated in the current
training run; reserved for future experiments.

| Feature | Type | Description |
|---|---|---|
| `nearest_defender_distance` | continuous | Distance (m) to nearest outfield defender |
| `defenders_within_5m` | continuous | Count of defenders within 5m radius |
| `defensive_density_in_box` | continuous | Defenders per square metre in penalty area |
| `has_360` | boolean | Indicator flag for 360-enriched events |

After-state equivalents: `after_nearest_defender_distance`,
`after_defenders_within_5m`, `after_defensive_density_in_box`.

---

## 6. Feature Count Summary

| Tier | Numeric | Boolean | Categorical | Total before-state |
|---|---|---|---|---|
| Traditional | 5 | 6 | 2 | 13 |
| Contextual | 17 | 13 | 7 | 37 |
| Full 360 | 20 | 14 | 7 | 41 |

After one-hot encoding of categorical variables (7 categorical columns in
the contextual tier), the GLM design matrix contains approximately **55–65
columns** depending on observed cardinality.

---

## 7. Preprocessing Pipeline

Features pass through a sklearn `ColumnTransformer` before reaching the model.
The exact pipeline varies by algorithm family:

### GLM (Gamma, Tweedie)

```
numeric + boolean columns
    → SimpleImputer(strategy="median")
    → StandardScaler()

categorical columns
    → SimpleImputer(strategy="constant", fill_value="unknown")
    → OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

StandardScaler is required for GLMs because `sklearn.linear_model.TweedieRegressor`
uses L-BFGS and is sensitive to feature scale: features with large dynamic
ranges (e.g. `x_location` 0–105) would otherwise dominate the gradient signal
over binary flags.

### Tree models (XGBoost, LightGBM)

```
numeric + boolean columns
    → SimpleImputer(strategy="median")

categorical columns
    → SimpleImputer(strategy="constant", fill_value="unknown")
    → OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=float)
```

Tree models are scale-invariant. `OrdinalEncoder` is used instead of
`OneHotEncoder` because trees handle integer-coded categories directly
without inflating dimensionality.

### GAM (pygam)

```
numeric + boolean columns
    → SimpleImputer(strategy="median")
    → StandardScaler()
```

The GAM uses numeric features only; categorical columns are excluded (pygam
does not support `OrdinalEncoder`-like categorical handling natively).
Spline terms (`s()`) are applied to spatial continuous features; linear terms
(`l()`) are applied to all others.
