# CxA — Model Specifications

## 1. Two-Stage Architecture Overview

The CxA pipeline consists of two independently trained sub-models, assembled
in `src/models/cxa/cxa_pipeline.py`:

| Stage | Task | Class | Output |
|---|---|---|---|
| **Stage 1** | Shot-creation classifier | `ShotCreationModel` | $p_i = P(S_i=1 \mid \mathbf{x}_i) \in [0,1]$ |
| **Stage 2** | Shot-quality regressor | `ShotQualityModel` | $q_i = \mathbb{E}[G_i \mid S_i=1, \mathbf{x}_i] \in (0,1]$ |
| **Combined** | CxA scorer | `CxAPipeline` | $\text{CxA}_i = p_i \times q_i$ |

Both stages share the same contextual feature set. Stage 2 trains only on the
subset of actions where a shot followed ($S_i = 1$).

---

## 2. Model Ladder

Three candidate pipelines were evaluated, each implementing the same Stage 1 +
Stage 2 architecture with a different algorithm family:

| Pipeline name | Stage 1 algorithm | Stage 2 algorithm | Feature tier |
|---|---|---|---|
| `logistic_contextual` | Logistic regression | Gamma GLM | CONTEXTUAL |
| `xgb_contextual` | XGBoost classifier | XGBoost regressor | CONTEXTUAL |
| `lgbm_contextual` | LightGBM classifier | LightGBM regressor | CONTEXTUAL |

---

## 3. Stage 1 — Shot-Creation Classifier

### 3.1 Logistic Regression

The logistic classifier is a GLM with a Bernoulli response and logit link.
The linear predictor $\eta_i$ combines all contextual features:

$$\eta_i = \beta_0
+ \underbrace{\beta_1 x_i + \beta_2 y_i + \beta_3 d_i^{(\text{goal})} + \cdots}_{\text{location}}
+ \underbrace{\beta_k e_i^{(\text{end})} + \beta_l d_i^{(\text{end})} + \beta_m \Delta d_i}_{\text{destination}}
+ \underbrace{\sum_k \beta_k^{(\text{type})} \mathbb{1}[a_i = k]}_{\text{action type}}
+ \underbrace{\beta_n \gamma_i^{(\text{opp})} + \cdots}_{\text{opponent quality}}
+ \underbrace{\beta_p \delta_i + \beta_q m_i + \cdots}_{\text{match state}}
+ \underbrace{\beta_r n_i^{(\text{pass})} + \beta_s \text{dir}_i + \cdots}_{\text{sequence}}
+ \underbrace{\beta_t x_i^{(\text{rcv})} + \beta_u d_i^{(\text{rcv})} + \cdots}_{\text{receiver}}$$

where $\Delta d_i$ = `distance_gained`, $n_i^{(\text{pass})}$ = `passes_before_action`,
$\text{dir}_i$ = `directness`, $x_i^{(\text{rcv})}$ = `receiver_x`,
$d_i^{(\text{rcv})}$ = `receiver_distance_to_goal`.

The shot-creation probability follows via the sigmoid:

$$p_i = \sigma(\eta_i) = \frac{1}{1 + e^{-\eta_i}}$$

**Parameter estimation.** The penalised log-likelihood:

$$\hat{\boldsymbol{\beta}} = \argmax_{\boldsymbol{\beta}} \left[
\sum_{i=1}^{N} \bigl[S_i \log p_i + (1-S_i)\log(1-p_i)\bigr]
- \frac{1}{2C}\|\boldsymbol{\beta}\|_2^2
\right]$$

With $N = 220{,}899$ actions and 15.93% positives ($\approx 35{,}168$ shot-creating
actions), regularisation ($C = 1.0$) provides moderate L2 shrinkage.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `solver` | `lbfgs` |
| `C` | 1.0 |
| `max_iter` | 2000 |
| `random_state` | 42 |

Pre-processing: `StandardScaler` on numeric features; `OneHotEncoder` (handle_unknown=ignore) on categoricals.

### 3.2 XGBoost Classifier

XGBoost builds an additive ensemble of $T$ regression trees fitted to
pseudo-residuals of the **logistic loss**:

$$p_i = \sigma\!\left(\sum_{t=1}^{T} f_t(\mathbf{x}_i)\right)$$

At stage $t$, the model minimises:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{N} \left[
g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t(\mathbf{x}_i)^2
\right] + \Omega(f_t)$$

where $g_i = \partial_{F} \ell(S_i, F_i)$, $h_i = \partial^2_{F} \ell(S_i, F_i)$
are the first and second-order gradient statistics of the binary cross-entropy,
and $\Omega(f_t) = \gamma T_t + \frac{1}{2}\lambda \|w\|^2$ is the structural
regularisation term ($T_t$ = number of leaves, $w$ = leaf weights).

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 10 |
| `scale_pos_weight` | ~5.3 (= neg/pos ratio) |
| `eval_metric` | `logloss` |
| `random_state` | 42 |

`scale_pos_weight` adjusts for the 15.93% positive rate by upweighting positive
samples, equivalent to a prior adjustment. `min_child_weight=10` prevents
leaf splits with fewer than 10 training samples, providing structural regularisation
given the imbalanced target.

Pre-processing: `SimpleImputer(median)` on numeric; `OrdinalEncoder` on categoricals.

### 3.3 LightGBM Classifier

LightGBM uses leaf-wise (best-first) tree growth rather than the level-wise
growth of standard GBDT. This typically achieves lower training error faster
but requires stronger regularisation to prevent overfitting.

The objective is the same binary cross-entropy as XGBoost. Leaf-wise growth
finds the leaf with the maximum gain at each iteration, producing asymmetric
trees.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `num_leaves` | 15 |
| `min_child_samples` | 20 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `is_unbalance` | True |
| `verbosity` | -1 |
| `random_state` | 42 |

`is_unbalance=True` replaces explicit `scale_pos_weight` with LightGBM's
internal class weight balancing, which reweights samples to equalise class
contributions to the loss.

---

## 4. Stage 2 — Shot-Quality Regressor

Stage 2 is trained exclusively on the $N^+ = \{i : S_i = 1\}$ subset of
actions where a shot followed. The target is the CxG score of the resulting
shot, $G_i \in (0, 1]$, attached by the production CxG model
(`glm_contextual`).

### 4.1 Gamma GLM (logistic family)

CxG scores are positive and bounded in $(0, 1]$. The Gamma distribution is
a natural choice for positive continuous responses:

$$G_i \mid \mathbf{x}_i \sim \text{Gamma}(\mu_i, \phi)$$

The canonical log link connects the linear predictor to the mean:

$$\log \mu_i = \eta_i = \boldsymbol{\beta}^\top \mathbf{x}_i$$

so $\mu_i = e^{\eta_i} > 0$ by construction. The Gamma GLM is implemented
via `sklearn.linear_model.TweedieRegressor(power=2, link="log")`.

**Key property:** The Gamma log link prevents the model from predicting
non-positive CxG values regardless of feature input — a physically meaningful
constraint since $G_i > 0$.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `power` | 2 (Gamma family) |
| `link` | `log` |
| `alpha` (L2) | 1.0 |
| `max_iter` | 2000 |

### 4.2 XGBoost Regressor

XGBoost is applied to Stage 2 with `objective="reg:squarederror"` (MSE loss).
The same structural hyperparameters as Stage 1 are used, with `scale_pos_weight`
removed (regression task).

### 4.3 LightGBM Regressor

LightGBM Stage 2 uses `objective="regression"` (MSE). Leaf-wise growth with
`min_child_samples=20` and `num_leaves=15` limits overfitting on the smaller
shot-subset training data (35,168 rows in training; 19,216 in held-out).

---

## 5. Pipeline Assembly

`CxAPipeline` wraps both stages and exposes a unified `.score(actions_df)`
interface:

```python
# Stage 1: shot-creation probability
p_creation = self.creation_model.predict_proba(df)   # shape (N,)

# Stage 2: expected CxG if a shot is taken
expected_cxg = self.quality_model.predict(df)         # shape (N,)

# CxA: product of both stages
cxa = p_creation * expected_cxg
```

The output DataFrame includes the full decomposition for interpretability:
`event_id`, `p_shot_created`, `expected_cxg_if_shot`, `cxa`,
`realised_cxa` (= `resulting_shot_cxg` where a shot actually followed,
else NaN), plus all passthrough context columns.

---

## 6. Worked Example (from Euro 2024 Held-Out Set)

Two representative actions from the held-out evaluation illustrate the
decomposition in practice:

**Top pass** (`event_id=5d38135bb4d76e9e`):
- Location: $x=101.7\,\text{m}$, $y=18.4\,\text{m}$ (right side, opponent box)
- Stage 1: $p_{\text{shot}} = 0.9536$
- Stage 2: $\hat{q} = 0.1513$
- CxA: $0.9536 \times 0.1513 = \mathbf{0.1443}$
- Realised: $G = 0.1594$ (shot did follow)

**Top carry** (`event_id=48d2c442f0203730`):
- Location: $x=87.3\,\text{m}$, $y=23.5\,\text{m}$ (right channel, approaching box)
- Stage 1: $p_{\text{shot}} = 0.9566$
- Stage 2: $\hat{q} = 0.1320$
- CxA: $0.9566 \times 0.1320 = \mathbf{0.1263}$
- Realised: $G = 0.1594$ (same match, shot did follow)

In both cases, Stage 1 is near-certain ($>0.95$) because the actions occur
inside or at the edge of the penalty box. The CxA differences between the
two are driven by Stage 2: the pass scores slightly higher quality because
its destination geometry is marginally more central than the carry's end point.
