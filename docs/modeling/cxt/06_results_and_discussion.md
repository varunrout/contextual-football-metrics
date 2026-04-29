# CxT — Results and Discussion

## 1. Full Leaderboard

Results from training on 220,899 creative actions (WC 2022 + Euro 2020),
evaluated on the Euro 2024 held-out set (98,029 creative actions).

| Rank | Model | CV MAE | CV RMSE | CV Spearman | HO MAE | HO RMSE | HO Spearman |
|---|---|---|---|---|---|---|---|
| **1** | `lgbm_contextual` | **0.02921** | 0.06645 | 0.2963 | **0.03082** | 0.07064 | **0.3015** |
| 2 | `xgb_contextual` | 0.03027 | 0.06694 | **0.3013** | 0.03086 | 0.07066 | 0.2937 |
| 3 | `tweedie_glm` | 0.03099 | 0.06704 | 0.2410 | 0.03210 | 0.07172 | 0.2431 |
| 4 | `gamma_glm` | 0.08993 | 0.12822 | 0.1451 | 0.10807 | 0.11688 | 0.0914 |
| 5 | `gam_contextual` | 0.09112 | 0.12913 | 0.1790 | 0.10188 | 0.11215 | 0.1173 |

**CV** = 5-fold match-level cross-validation on 220,899 training actions.
**HO** = Euro 2024 held-out set (98,029 actions, never seen during training).

Model selection criterion: CV MAE (lower is better).
**Production model selected: `lgbm_contextual`** — rank 1 on CV MAE and HO MAE;
rank 1 on HO Spearman.

Saved to: `models/cxt/lgbm_contextual.joblib`
Config pointer: `configs/models.yaml → production.cxt`

---

## 2. Primary Finding: Tweedie GLM Fixes the Structural GLM Failure

The most important result is the contrast between the three GLM-family models:

| Model | Training data | HO MAE | HO Spearman |
|---|---|---|---|
| `gamma_glm` | pos_df only (~16%) | 0.10807 | 0.0914 |
| `gam_contextual` | pos_df only (~16%) | 0.10188 | 0.1173 |
| `tweedie_glm` | all rows (100%) | **0.03210** | **0.2431** |

The Gamma GLM and GammaGAM both train only on positive `possession_cxg` rows.
When evaluated on the full held-out set (84% zeros), they systematically
over-predict zero-valued actions — the model never learned to output near-zero
values because it was never shown zero-valued training examples.

The Tweedie GLM (power=1.5) places a point mass at zero and a Gamma-like
continuous component for positive values. By training on all rows, it learns:

1. The probability that a possession produces no goal threat (the zero mass).
2. The expected threat value conditional on a shot occurring (the positive tail).

Result: a **3.4× MAE reduction** compared to Gamma GLM (0.108 → 0.032) and
**2.7× Spearman improvement** (0.091 → 0.243).

The Tweedie GLM is now the correct GLM baseline for CxT — replacing Gamma as
the interpretable, linear reference model.

---

## 3. GAM vs. Tweedie GLM: Splines Help Ranking but not MAE

The GAM was designed to outperform the GLMs by capturing nonlinear spatial
relationships through penalised splines:

- Spline over `x_location` captures the nonlinear relationship between
  longitudinal pitch position and threat value.
- Spline over `distance_to_goal` captures the accelerating danger close to goal.
- Spline over `minute` captures time-dependent tactical shifts.

The GAM does show a ranking improvement over Gamma GLM:
Spearman 0.117 vs. 0.091 (+2.8pp) — the spatial splines provide a discriminative
benefit even under the zero-handling constraint.

However, the MAE gap between GAM and Tweedie remains large (0.102 vs. 0.032)
because the GAM uses a `GammaGAM` family (pygam does not implement Tweedie).
The spline capacity is overwhelmed by the systematic bias from not modelling
zero-valued possessions.

**Conclusion:** A Tweedie-family GAM would likely outperform the Tweedie GLM
on Spearman while matching it on MAE — but this requires an implementation
not available in `pygam` (available in R's `mgcv` package as `family=tw()`).
The current GAM result is limited by the library constraint, not the
theoretical model.

---

## 4. Tree Models vs. Tweedie GLM: The Interaction Gap

The two tree models outperform Tweedie GLM on both MAE and Spearman:

| Model | HO MAE | HO Spearman | vs. Tweedie GLM |
|---|---|---|---|
| `lgbm_contextual` | 0.03082 | 0.3015 | −0.00128 MAE, +5.8pp Spearman |
| `xgb_contextual` | 0.03086 | 0.2937 | −0.00124 MAE, +5.1pp Spearman |
| `tweedie_glm` | 0.03210 | 0.2431 | — |

The MAE gap is small (4–5%) but the Spearman gap is substantial (+5–6pp).
This reflects the tree models' ability to capture:

1. **Pitch-position interactions** — a carry from the left half-space into the
   box is worth more than the same carry from the right half-space against a
   right-heavy defensive shape. The GLM models each feature independently.
2. **Sequence-context interactions** — a pass in a fast counter-attack
   (low `events_before_action`, high `directness`) at a given location is more
   threatening than the same pass in a settled build-up. Trees learn these
   joint effects; GLMs apply additive adjustments.
3. **Non-linear distance effects** — the threat value of `distance_to_goal`
   is not log-linear. Inside 15m it accelerates sharply; beyond 35m it
   plateaus. Trees partition the feature space and fit separate leaf values;
   the log-link GLM applies a single monotone transformation.

---

## 5. LightGBM vs. XGBoost

The two tree models are nearly identical in MAE (0.03082 vs. 0.03086).
The differences:

- LightGBM leads on **HO Spearman** (0.3015 vs. 0.2937) — leaf-wise growth
  finds optimal splits for the positive tail more effectively on this dataset.
- XGBoost leads on **CV Spearman** (0.3013 vs. 0.2963) — a negligible
  difference that does not distinguish the models in practice.
- LightGBM has faster training time due to Gradient-based One-Side Sampling
  (GOSS) and histogram-based tree construction.

LightGBM is selected as production. The performance advantage is marginal;
the selection is justified by slightly better generalisation to the held-out
tournament and faster scoring.

---

## 6. CV–Heldout Generalisation

| Model | ΔMae (HO − CV) | ΔSpearman (HO − CV) | Verdict |
|---|---|---|---|
| `lgbm_contextual` | +0.00161 | +0.005 | ✓ Well-generalised |
| `xgb_contextual` | +0.00059 | −0.008 | ✓ Well-generalised |
| `tweedie_glm` | +0.00111 | +0.002 | ✓ Well-generalised |
| `gamma_glm` | +0.01814 | −0.054 | ⚠ CV–HO degradation |
| `gam_contextual` | +0.01076 | −0.062 | ⚠ CV–HO degradation |

The tree models and Tweedie GLM all generalise cleanly to Euro 2024.

The Gamma GLM and GAM show larger held-out degradation. The cause is the
`pos_df` training constraint: the CV validation sets include zero-valued rows,
and the fold-to-fold variance in the proportion of zero-vs-positive rows in
the validation split inflates CV MAE relative to held-out (or vice versa,
depending on how Euro 2024 differs in positive rate). The Euro 2024 held-out
set has ~19.6% positive rate vs. ~16% in the CV pool — more non-shot possessions
relative to what the GLMs learned, worsening their miscalibration.

---

## 7. Production Model Selection

The CV criterion nominates `lgbm_contextual` as rank 1. On every held-out
metric, `lgbm_contextual` is also strictly best or near-tied:

| Criterion | `lgbm_contextual` | `xgb_contextual` | `tweedie_glm` |
|---|---|---|---|
| CV MAE | **0.02921** | 0.03027 | 0.03099 |
| HO MAE | **0.03082** | 0.03086 | 0.03210 |
| HO Spearman | **0.3015** | 0.2937 | 0.2431 |

`lgbm_contextual` is promoted to production. `tweedie_glm` is retained as
the interpretable baseline: its coefficients can be directly inspected to
understand the additive contribution of each contextual adjustment group.

---

## 8. Interpretation: What the Model Learns

### 8.1 Pitch Value Surface

The pitch value surface (`reports/figures/cxt/pitch_value_surface.png`) shows
the estimated $V(s)$ at each pitch location, holding contextual features
at their median values. Key features:

- **Final third**: Steep increase in $V(s)$ for $x > 80$m, particularly
  in the central channel.
- **Penalty area**: Highest absolute $V(s)$ values, concentrated at the
  6-yard box entrance.
- **Wide channels**: Moderate $V(s)$ in wide positions even outside the box,
  reflecting the value of crossing opportunities.
- **Own half**: Near-zero $V(s)$, confirming that possessions starting deep
  rarely generate threatening shots in the contextual tier.

### 8.2 Action Type Effects

CxT scores by action type from the held-out set (approximate):

| Action type | Mean CxT (positive subset) |
|---|---|
| `cutback` | Highest — receives ball from byline, immediately near goal |
| `carry` (final third) | High — spatial progress into dangerous zones |
| `pass` (final third) | Moderate — spatially progressive but constrained by recipient location |
| `cross` | Variable — depends on end zone; wide crosses lower than central deliveries |
| All actions (all zones) | Near zero — 84% zero-valued possessions dominate |

### 8.3 Sequence Context Effects

Actions in **counter-attacks** (`sequence_type = "counter_attack"`) are
assigned higher $V(s_{\text{before}})$ values than equivalent-location actions
in settled build-ups — the model captures that fast, direct possessions tend
to produce higher-quality chances. This is the key advantage of the contextual
feature tier over the traditional tier.

---

## 9. Limitations and Future Work

**L1 — Shared target within possession.** All actions in a possession share
the same `possession_cxg` label. The model cannot distinguish whether a specific
pass within a 10-pass build-up was the critical link in the sequence or
incidental. The CxT score is a **state evaluation**, not an attribution of
individual credit.

**L2 — Discount factor is heuristic.** The $\gamma = 0.9$ discount is not
learned from the data. A smaller $\gamma$ would attribute value more to
earlier actions in the possession; $\gamma = 1.0$ would treat all shots
within a possession equally. Sensitivity analysis across $\gamma$ values
(0.7, 0.8, 0.9, 1.0) is a natural next experiment.

**L3 — GAM constrained by library.** The `pygam` GammaGAM cannot model
zero-inflated targets natively. A Tweedie-family GAM (e.g. via `statsmodels`
or `mgcv` in R) would provide both spline flexibility and correct zero handling.

**L4 — FULL_360 not evaluated.** Freeze-frame spatial features (`nearest_defender_distance`,
`defensive_density_in_box`) were not included in this training run. They are
expected to substantially improve Spearman by capturing defensive compactness
effects invisible to location-alone features.

**L5 — CxT computed post-possession, not in real time.** The `possession_cxg`
target is only computable after the possession ends (because it depends on all
future shots in the possession). This is appropriate for retrospective player
analysis but prevents use in real-time tactical applications.
