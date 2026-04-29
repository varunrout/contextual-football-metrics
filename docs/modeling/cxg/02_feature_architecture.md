# CxG — Feature Architecture

## 1. Overview

Features are organised into three nested tiers of increasing contextual richness,
defined in `src/models/cxg/feature_sets.py` as `FeatureSetSpec` objects.
Each tier is a strict superset of the previous one.

```
TRADITIONAL  ⊂  CONTEXTUAL  ⊂  FULL_360
```

This nesting allows direct ablation: the contribution of each context group is
isolated by comparing models trained on successive tiers.

---

## 2. Tier 1 — Traditional

These features are available for any competition with basic event data and
replicate the inputs used by canonical open-play xG models.

### 2.1 Shot Geometry

| Feature | Type | Description |
|---|---|---|
| `distance_to_goal` | continuous | Euclidean distance from shot location to centre of goal (metres) |
| `shot_angle` | continuous | Angle subtended at the shooter by the goal opening (radians) |
| `x_location` | continuous | Longitudinal pitch coordinate (StatsBomb frame: 0–120) |
| `y_location` | continuous | Lateral pitch coordinate (StatsBomb frame: 0–80) |

The geometry pair $(d, \theta)$ is the dominant signal in any xG model. Their
joint relationship with goal probability is non-linear: marginal difficulty
increases sharply below ~8m and the angle effect saturates beyond ~30m.

### 2.2 Technique

| Feature | Type | Description |
|---|---|---|
| `header` | boolean | Shot taken with the head |
| `volley` | boolean | Struck before the ball bounces |
| `first_time_shot` | boolean | Struck without a controlling touch |
| `open_play` | boolean | Not a set piece |
| `under_pressure` | boolean | Defender within challenging distance at moment of shot |
| `body_part` | categorical | `Head`, `Right Foot`, `Left Foot`, `No Touch` |
| `shot_type` | categorical | `Open Play`, `Free Kick`, `Corner`, `Penalty`, `Kick Off` |
| `set_piece_type` | categorical | Direct vs. indirect set piece variant |

---

## 3. Tier 2 — Contextual (Traditional + 3 adjustment groups)

### 3.1 Opponent Quality Adjustment

The quality of the defending team is a theoretically justified confounder: a
striker facing a weakly organised back-four with a low-rated goalkeeper
faces systematically easier shots than the geometry alone indicates.

| Feature | Type | Description |
|---|---|---|
| `opponent_xg_conceded_rolling_5` | continuous | Rolling 5-match mean xG conceded by the defending team (pre-match) |
| `opponent_shots_conceded_rolling_5` | continuous | Rolling 5-match mean shots conceded (pre-match) |
| `opponent_defensive_rating` | continuous | Composite defensive quality rating (normalised 0–1) |
| `opponent_keeper_shot_stopping_rating` | continuous | Keeper-specific shot-stopping rating (normalised 0–1) |
| `opponent_team_strength` | continuous | Overall team ELO / strength signal |

All rolling features are computed strictly over completed matches prior to the
current match, satisfying temporal integrity constraint **C2**.

### 3.2 Match State

| Feature | Type | Description |
|---|---|---|
| `minute` | continuous | Match minute at time of shot |
| `score_differential` | continuous | Shooting team goals minus opponent goals at time of shot |
| `score_state` | categorical | `winning`, `drawing`, `losing` |
| `home_or_away` | categorical | `home`, `away`, `neutral` |
| `knockout_or_group` | boolean | True if elimination stage (changes pressure dynamics) |
| `set_piece_flag` | boolean | Derived from possession context — shot came via set piece possession |

Score state interacts with shot-taking behaviour: teams that are losing tend
to take lower-quality, more speculative shots in desperation; teams that are
winning often concede higher-quality chances from defensive overextension.
Conditioning on match state partially deconfounds this selection effect.

### 3.3 Sequence / Possession Context

| Feature | Type | Description |
|---|---|---|
| `events_before_action` | continuous | Number of events in possession chain before shot |
| `passes_before_action` | continuous | Number of passes in possession chain before shot |
| `carries_before_action` | continuous | Number of carries in possession chain before shot |
| `time_from_possession_start` | continuous | Seconds from possession start to shot (seconds) |
| `vertical_progression_speed` | continuous | Net pitch-vertical metres gained per second in possession |
| `directness` | continuous | Ratio of displacement to path length (1 = perfectly direct) |
| `sequence_type` | categorical | `counter_attack`, `build_up`, `set_piece`, `direct`, `unknown` |
| `possession_start_zone` | categorical | Pitch zone where possession began |
| `transition_or_settled` | categorical | `transition`, `settled` |
| `counterpress_regain_flag` | boolean | Possession started via counterpress regain within 5 seconds |

A shot at the end of a fast, direct counter-attack (high `directness`,
low `events_before_action`, `transition`) is structurally different from
a shot manufactured through a slow build-up, even at the same $(d, \theta)$.
Defenders and keepers have had less time to organise in the former case.

---

## 4. Tier 3 — Full 360 (Contextual + Spatial Freeze-Frame)

Available only for StatsBomb 360-enriched competitions. Not activated in
this training run; reserved for a future experiment when sufficient 360-only
training data is available.

| Feature | Type | Description |
|---|---|---|
| `nearest_defender_distance` | continuous | Distance (m) to nearest outfield defender |
| `second_nearest_defender_distance` | continuous | Distance to second-nearest defender |
| `defenders_within_5m` | continuous | Count of defenders within 5m radius of shooter |
| `defenders_between_ball_and_goal` | continuous | Defenders in the shot lane |
| `keeper_distance_to_goal` | continuous | Keeper's distance from goal line |
| `keeper_distance_to_shooter` | continuous | Distance from keeper to shooter |
| `keeper_angle_coverage` | continuous | Angle of goal the keeper subtends (radians) |
| `shot_lane_blockage_proxy` | continuous | Estimated proportion of goal mouth obstructed |
| `defensive_density_in_box` | continuous | Defenders per square metre in penalty area |
| `has_360` | boolean | Indicator flag for 360-enriched event |

---

## 5. Feature Count Summary

| Tier | Numeric | Boolean | Categorical | Total |
|---|---|---|---|---|
| Traditional | 4 | 5 | 3 | 12 |
| Contextual | 17 | 8 | 8 | 33 |
| Full 360 | 26 | 9 | 8 | 43 |

After one-hot encoding of categorical variables (8 categorical columns in the
contextual tier, expanded with `handle_unknown="ignore"`), the design matrix
for the contextual model contains approximately **55–60 columns** depending on
the number of observed categories in the training set.

---

## 6. Preprocessing Pipeline

All features pass through a sklearn `ColumnTransformer` before entering the model:

```
numeric + boolean columns
    → SimpleImputer(strategy="median")
    → StandardScaler()                   # GLM only; trees use median imputer only

categorical columns
    → SimpleImputer(strategy="constant", fill_value="unknown")
    → OneHotEncoder(handle_unknown="ignore")   # GLM
       or
       OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)  # Trees
```

**StandardScaler is applied only for logistic regression.** It rescales each
numeric feature to mean 0 and standard deviation 1:

$$\tilde{x}_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}$$

This is necessary because `lbfgs` uses gradient information and is sensitive to
feature scale: without scaling, features with large dynamic ranges (e.g.
`distance_to_goal` in metres) dominate the gradient updates over features with
small ranges (e.g. `score_differential`).

Tree models (XGBoost, LightGBM) are scale-invariant and do not require
standardisation; they use `OrdinalEncoder` rather than `OneHotEncoder` because
tree splits handle integer-coded categories directly without inflating
dimensionality.
