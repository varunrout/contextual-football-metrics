# CxA — Training Methodology

## 1. Pipeline Overview

Training is orchestrated by `scripts/train_cxa.py`. The pipeline has two
distinct phases: label construction (which depends on the pre-trained CxG
model) and model training (which iterates the ladder).

```
actions.parquet (318,928 creative actions)
    │
    ▼
[1] CxG score attachment
    │  Load production CxG model (models/cxg/glm_contextual.joblib)
    │  Score 4,123 shot rows → add 'cxg' column to features_df
    │
    ▼
[2] Train/held-out split
    │  train: WC 2022 + Euro 2020 → 220,899 actions (15.93% shot_created)
    │  held_out: Euro 2024         →  98,029 actions (19.61% shot_created)
    │
    ▼
[3] Label attachment (_attach_labels)
    │  shot_created = 1 if possession_internal_id contains a shot
    │  resulting_shot_cxg = mean CxG of shots in that possession
    │
    ▼
[4] Ladder loop (3 pipelines)
    │
    ├─ logistic_contextual   (Logistic / Gamma GLM)
    ├─ xgb_contextual        (XGBoost clf / XGBoost reg)
    └─ lgbm_contextual       (LightGBM clf / LightGBM reg)
    │
    │  For each pipeline:
    │    [4a] Train Stage 1 on all 220,899 training actions
    │    [4b] Train Stage 2 on the 35,168 shot-creating training actions
    │    [4c] Score held-out set with both stages → compute metrics
    │    [4d] Save pipeline to models/cxa/cxa_{name}.pkl
    │
    ▼
[5] Select best pipeline by held-out creation AUC
    │
    ▼
[6] Update production pointer in configs/models.yaml
    │
    ▼
[7] Generate charts for best pipeline
    │  PR curve + calibration diagram  → reports/figures/cxa/pr_curve.png
    │  Pitch heatmap (pass, carry)     → reports/figures/cxa/cxa_pitch_heatmap.png
    │  Creation rate by distance       → reports/figures/cxa/creation_rate_by_distance.png
    │  Quality scatter + calibration   → reports/figures/cxa/quality_scatter.png
    │  CxA leaderboard bar chart       → reports/figures/cxa/cxa_leaderboard.png
    │
    ▼
[8] Write reports/cxa_training_summary.json
```

---

## 2. Label Construction

### 2.1 Shot-Creation Label

The binary label $S_i \in \{0, 1\}$ is assigned via **possession-level linkage**:

1. Identify all shot events in the full feature store by event type.
2. Collect the set of unique `possession_internal_id` values for those shots:
   $\mathcal{P}^+ = \{p : \exists \text{ shot in possession } p\}$.
3. Assign $S_i = 1$ if `possession_internal_id` for action $i \in \mathcal{P}^+$,
   else $S_i = 0$.

**Why `possession_internal_id`?** Using `match_id` alone would be incorrect —
every match contains shots, so every creative action would receive $S_i = 1$,
producing a 100% positive rate. The possession-level join correctly identifies
whether the **specific possession** containing action $i$ resulted in a shot.

Positive rate in training data: $15.93\%$ (35,168 of 220,899 actions).

### 2.2 Shot-Quality Label

The continuous label $G_i$ (resulting CxG for Stage 2) is assigned as:

$$G_i = \text{mean CxG of all shots in possession}(i)$$

where CxG scores come from the production CxG model
(`glm_contextual`, `models/cxg/glm_contextual.joblib`).

This averaging is a conservative approximation: if a possession contains
two shots (e.g. a strike and a rebound), both are averaged rather than taking
the maximum. Actions without a subsequent shot receive $G_i = 0$.

**CxG dependency.** Stage 2 labels use the output of a separately trained
CxG model. The training order is therefore:

```
train_cxg.py → [glm_contextual.joblib] → train_cxa.py
```

Upgrading the CxG production model requires re-running `train_cxa.py`.

---

## 3. Train/Test Split Design

### 3.1 Competition-Level Partitioning

The data are partitioned at the **competition level** to preserve temporal
and distributional structure:

| Role | Competition | Season | Actions | Positive rate |
|---|---|---|---|---|
| `train` | FIFA World Cup + UEFA Euro | 2022 + 2020 | 220,899 | 15.93% |
| `held_out` | UEFA Euro | 2024 | 98,029 | 19.61% |

Unlike CxG (where train/val split uses match-level CV), CxA trains directly
on the full training set and evaluates on the held-out competition. This is
because CxA has a much larger training set (220,899 actions vs. 2,783 shots),
making cross-validation less critical for variance estimation.

### 3.2 Rationale for Competition-Level Holdout

1. **Opponent quality features** — rolling statistics for a team are computed
   from prior matches. Row-level splitting would allow training to see
   statistics computed from the same period as validation examples.
2. **Possession correlation** — actions within the same possession share
   sequence features, sequence type, score state, and time. Row-level splits
   would allow the model to see near-identical context vectors in train and
   validation.
3. **Distribution shift testing** — holding out Euro 2024 tests whether the
   model generalises to a new cohort of teams and playing styles distinct from
   WC 2022 and Euro 2020. The higher held-out positive rate (19.61% vs. 15.93%)
   is exactly the kind of distributional shift that must be handled in production.

---

## 4. Stage 1 Training Procedure

Stage 1 fits the shot-creation classifier on all $N = 220{,}899$ training actions.

### 4.1 Logistic Regression (logistic_contextual)

1. Build feature matrix $\mathbf{X} \in \mathbb{R}^{N \times p}$ from the
   contextual feature set.
2. Apply `StandardScaler` to numeric features, `OneHotEncoder` to categoricals.
3. Fit `LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000)` to
   $(X, S)$ where $S \in \{0,1\}^N$.
4. Evaluate on held-out set: AUC, AP, log-loss, Brier.

### 4.2 XGBoost Classifier (xgb_contextual)

1. Apply `SimpleImputer(median)` to numeric; `OrdinalEncoder` to categoricals.
2. Fit `XGBClassifier(objective="binary:logistic", n_estimators=300, ...)`.
3. `scale_pos_weight ≈ 5.3` (= $N^{-}$ / $N^{+}$) adjusts for class imbalance.

### 4.3 LightGBM Classifier (lgbm_contextual)

1. Same preprocessing as XGBoost.
2. Fit `LGBMClassifier(objective="binary", is_unbalance=True, ...)`.
3. `is_unbalance=True` reweights samples automatically for the imbalanced target.

---

## 5. Stage 2 Training Procedure

Stage 2 fits the shot-quality regressor on the **subset of actions where a
shot followed** ($S_i = 1$ in training data):

$$\mathcal{D}^{+} = \{({\mathbf{x}_i, G_i}) : S_i = 1\} \quad
|\mathcal{D}^{+}| = 35{,}168$$

### 5.1 Gamma GLM

1. Filter to $\mathcal{D}^+$; drop rows with $G_i \leq 0$ (no CxG attached).
2. Apply `StandardScaler` + `OneHotEncoder`.
3. Fit `TweedieRegressor(power=2, link="log", alpha=1.0)` to $(X^+, G^+)$.

### 5.2 XGBoost / LightGBM Regressors

1. Filter to $\mathcal{D}^+$.
2. Impute and encode as in Stage 1.
3. Fit `XGBRegressor(objective="reg:squarederror", ...)` or
   `LGBMRegressor(objective="regression", ...)`.

---

## 6. Model Persistence

Each pipeline is saved as a single pickle file containing the fitted
`CxAPipeline` (with both stages):

```
models/cxa/cxa_logistic_contextual.pkl
models/cxa/cxa_xgb_contextual.pkl
models/cxa/cxa_lgbm_contextual.pkl
```

The production pointer in `configs/models.yaml` is updated to reference
the best-performing pipeline:

```yaml
production:
  cxa: models/cxa/cxa_logistic_contextual.pkl
```

---

## 7. MLflow Experiment Tracking

Training metrics, parameters, and model artefacts are logged to the MLflow
experiment `cfm/cxa` under `mlruns/`. Each pipeline run is logged as a
separate MLflow run with:

- **Parameters**: feature set, n_estimators, family names
- **Metrics**: creation_auc, creation_ap, creation_ll, creation_brier,
  quality_mae, quality_rmse, quality_spearman
- **Artefacts**: the `.pkl` pipeline file

The training summary is additionally serialised to
`reports/cxa_training_summary.json` for offline inspection.
