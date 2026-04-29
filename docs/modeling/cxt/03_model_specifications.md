# CxT — Model Specifications

## 1. Candidate Model Family

Five candidate models were evaluated, covering three algorithm families:

| Model name | Algorithm | Family | Training data | Class |
|---|---|---|---|---|
| `gamma_glm` | Gamma GLM (power=2) | GLM | Positive targets only (`pos_df`) | `GammaStateValueModel` |
| `tweedie_glm` | Tweedie GLM (power=1.5) | GLM | All rows (`actions_df`) | `TweedieStateValueModel` |
| `gam_contextual` | GAM / GammaGAM | GAM | Positive targets only (`pos_df`) | `GAMStateValueModel` |
| `xgb_contextual` | XGBoost | Boosted tree | All rows (`actions_df`) | `XGBoostStateValueModel` |
| `lgbm_contextual` | LightGBM | Boosted tree | All rows (`actions_df`) | `LightGBMStateValueModel` |

All are **regression models** targeting `possession_cxg` — the discounted
possession-level CxG. All implementations reside in
`src/models/cxt/state_value_model.py`.

---

## 2. Gamma GLM

### 2.1 Model Structure

The Gamma GLM is a Generalised Linear Model with a **Gamma response** and
**log link function**. Under the Gamma family (power=2), the response is
modelled as:

$$Y_i \mid \mathbf{x}_i \sim \text{Gamma}(\mu_i, \phi)$$

The log link connects the linear predictor $\eta_i$ to the mean response:

$$\log \mu_i = \eta_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}$$

Exponentiation gives the mean prediction:

$$\hat{\mu}_i = e^{\eta_i}$$

This guarantees predictions are strictly positive — a natural fit for a
skewed, positive-only continuous target.

### 2.2 Critical Limitation: Zero Handling

The Gamma distribution has support on $(0, \infty)$ only. It is **undefined at
zero**. Since 84% of `possession_cxg` values are exactly zero, the Gamma GLM
cannot be trained on the full dataset.

Solution: the model is trained on `pos_df` — the subset of rows where
`possession_cxg > 0` (~16% of training data). At prediction time, when the
model scores zero-valued rows (during CV or held-out evaluation), it predicts
a positive value and incurs systematic error. This is the primary reason for
the Gamma GLM's poor MAE performance on the full evaluation set.

### 2.3 Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `power` | 2.0 | Gamma family (EDM variance function $V(\mu) = \mu^2$) |
| `alpha` | 1.0 | L2 regularisation strength |
| `link` | `log` | Log link function |
| `max_iter` | 2000 | L-BFGS iteration budget |

Implemented via `sklearn.linear_model.TweedieRegressor(power=2, link="log")`.

---

## 3. Tweedie GLM

### 3.1 Model Structure

The Tweedie GLM uses the **Tweedie distribution** — a member of the exponential
dispersion family (EDF) that unifies several distributions by varying the
variance power parameter $p$:

| Power $p$ | Distribution |
|---|---|
| 0 | Normal |
| 1 | Poisson |
| (1, 2) | **Compound Poisson-Gamma** ← used here |
| 2 | Gamma |
| 3 | Inverse Gaussian |

For $1 < p < 2$, the Tweedie distribution has a **point mass at zero** (from
the Poisson component) combined with a positive continuous tail (from the Gamma
component). This is structurally well-matched to the `possession_cxg` target:
most possessions produce no shots (mass at zero) while shot-producing
possessions have a Gamma-like CxG distribution.

The log link and linear predictor structure are identical to the Gamma GLM:

$$\log \mathbb{E}[Y_i \mid \mathbf{x}_i] = \eta_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}$$

### 3.2 Key Advantage Over Gamma

The Tweedie (power=1.5) model trains on **all rows** including zeros. This is
the decisive structural advantage: the model learns the probability of a zero
outcome from the data rather than discarding that information. When evaluated
on the full test set, zero-valued targets are predicted near zero rather than
at an elevated positive value.

Result: held-out MAE drops from 0.10807 (Gamma) to 0.03210 (Tweedie) — a 3×
improvement attributable entirely to this structural fix.

### 3.3 Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `power` | 1.5 | Compound Poisson-Gamma (zero mass + positive tail) |
| `alpha` | 1.0 | L2 regularisation strength |
| `link` | `log` | Log link function |
| `max_iter` | 2000 | L-BFGS iteration budget |

Implemented via `sklearn.linear_model.TweedieRegressor(power=1.5, link="log")`.

---

## 4. GAM (Generalised Additive Model)

### 4.1 Model Structure

The GAM uses the `pygam` library (`GammaGAM` family) with mixed spline and
linear terms. The additive predictor is:

$$\log \mathbb{E}[Y_i \mid \mathbf{x}_i] = \beta_0 + \sum_{j \in \mathcal{S}} s_j(x_{ij}) + \sum_{k \notin \mathcal{S}} \beta_k x_{ik}$$

where $\mathcal{S}$ is the set of features assigned spline terms and
$s_j(\cdot)$ is a penalised cubic regression spline for feature $j$.

Spline terms are applied to known continuous spatial features:

| Feature | Term type | Rationale |
|---|---|---|
| `x_location` | `s()` | Nonlinear pitch-longitude effect |
| `y_location` | `s()` | Nonlinear pitch-lateral effect |
| `distance_to_goal` | `s()` | Nonlinear distance-threat relationship |
| `end_x`, `end_y` | `s()` | After-state spatial location |
| `end_distance_to_goal` | `s()` | After-state distance |
| `minute` | `s()` | Nonlinear time-in-match dynamics |
| `keeper_distance_to_goal` | `s()` | Spatial 360 feature (if present) |

All remaining numeric and boolean features use linear terms `l()`.

### 4.2 Limitation: GammaGAM Family

`pygam` does not implement a Tweedie family. The `GammaGAM` family requires
strictly positive targets, so `gam_contextual` trains on `pos_df` — the same
subset as `gamma_glm`. This prevents the GAM from benefiting from its
theoretical advantage (spatial splines) because 84% of evaluation rows are
zero-valued and systematically mispredicted.

The Spearman correlation improvement over Gamma GLM (0.117 vs. 0.091) shows
that the spatial splines do provide some ranking benefit, but the MAE gap
relative to Tweedie remains large (0.10188 vs. 0.03210).

### 4.3 Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `n_splines` | 20 | Knots per spline term |
| `lam` | 0.6 | Smoothing penalty (higher = smoother) |
| Family | `GammaGAM` | Gamma family, log link |

---

## 5. XGBoost

### 5.1 Model Structure

XGBoost builds an ensemble of $T$ regression trees in a stage-wise fashion.
Each tree $f_t$ fits the **pseudo-residuals** of the current ensemble.
After $T$ trees:

$$\hat{y}_i^{(T)} = \sum_{t=1}^{T} f_t(\mathbf{x}_i)$$

For regression with squared-error objective, the prediction is used directly
as the target estimate. Non-negativity is enforced by clipping at 0.

### 5.2 Objective

At stage $t$, XGBoost minimises a second-order Taylor expansion of the
squared-error loss plus a structural regularisation term:

$$\mathcal{L}^{(t)} = \sum_{i} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t(\mathbf{x}_i)^2 \right] + \Omega(f_t)$$

where:
- $g_i = \hat{y}_i^{(t-1)} - y_i$ (residual)
- $h_i = 1$ (constant for squared error)
- $\Omega(f_t) = \gamma T_{\text{leaves}} + \frac{1}{2}\lambda \|w\|^2$ (complexity penalty)

### 5.3 Parameters

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` | 400 | Boosting rounds |
| `learning_rate` | 0.05 | Step shrinkage |
| `max_depth` | 6 | Maximum tree depth |
| `subsample` | 0.8 | Row subsampling per tree |
| `colsample_bytree` | 0.8 | Feature subsampling per tree |
| `min_child_weight` | 5 | Minimum Hessian sum per leaf |
| `reg_lambda` | 1.0 | L2 penalty on leaf weights |
| `objective` | `reg:squarederror` | Squared-error regression |
| `tree_method` | `hist` | Histogram-based split finding (faster) |

---

## 6. LightGBM

### 6.1 Differences from XGBoost

LightGBM uses the same gradient boosting framework as XGBoost but with two
algorithmic differences that affect training on large, sparse datasets:

**Leaf-wise growth:** XGBoost grows trees level-wise (all leaves at a given
depth are split before proceeding). LightGBM grows trees **leaf-wise**,
expanding the single leaf with the greatest loss reduction at each step.
This achieves lower training loss with fewer splits but increases overfitting
risk on small datasets (not a concern here with 220,899 rows).

**Gradient-based One-Side Sampling (GOSS):** At each iteration, instances with
small gradients are randomly down-sampled while instances with large gradients
(harder examples) are retained. This reduces computation while concentrating
training on the most informative examples.

### 6.2 Parameters

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` | 400 | Boosting rounds |
| `learning_rate` | 0.05 | Step size |
| `num_leaves` | 63 | Maximum leaves per tree ($\approx 2^6 - 1$) |
| `subsample` | 0.8 | Row sampling |
| `colsample_bytree` | 0.8 | Feature sampling |
| `min_child_samples` | 20 | Minimum instances per leaf |
| `objective` | `regression` | Squared-error regression |
| `metric` | `rmse` | RMSE as internal evaluation metric |

---

## 7. FFNN (Feed-Forward Neural Network)

A fifth candidate `ffnn_contextual` was designed as an optional candidate:

**Architecture:** $d_{\text{in}} \to 256 \xrightarrow{\text{ReLU}} \text{Dropout}(0.1) \to 128 \xrightarrow{\text{ReLU}} \text{Dropout}(0.1) \to 1 \xrightarrow{\text{Softplus}}$

The Softplus activation on the output ($\log(1 + e^x)$) enforces non-negative
predictions without hard clipping. Training uses **Huber loss** (delta=0.1),
which is less sensitive to large outliers than MSE — appropriate given the
heavy-tailed `possession_cxg` distribution.

The FFNN requires PyTorch and was not included in the current training ladder
due to the substantially longer training time and the absence of a meaningful
performance advantage over LightGBM on tabular data at this scale.

---

## 8. Shared Design Decisions

### 8.1 Non-Negativity

All models clip predictions at 0: `np.clip(pred, 0.0, None)`. The
`possession_cxg` target is non-negative by construction; allowing negative
predictions would be physically meaningless.

### 8.2 Training Data Subset by Family

| Model family | Training rows | Reason |
|---|---|---|
| Gamma GLM | `pos_df` (~16% of data) | Gamma requires $y > 0$ |
| Tweedie GLM | `actions_df` (all rows) | Tweedie handles $y = 0$ natively |
| GAM (GammaGAM) | `pos_df` (~16% of data) | GammaGAM requires $y > 0$ |
| XGBoost | `actions_df` (all rows) | Squared-error works at $y = 0$ |
| LightGBM | `actions_df` (all rows) | Squared-error works at $y = 0$ |

This asymmetry is the primary determinant of relative MAE performance. Models
trained on `actions_df` have a structural advantage: they learn to predict near
zero for the 84% of rows where no shot follows.
