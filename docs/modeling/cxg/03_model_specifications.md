# CxG — Model Specifications

## 1. Candidate Model Family

Six candidate models were evaluated, organised into three algorithm families
each applied to two feature tiers:

| Model name | Algorithm | Feature tier | Class |
|---|---|---|---|
| `baseline_logit` | Logistic regression | Traditional | `BaselineCxGModel` |
| `glm_contextual` | Logistic regression | Contextual | `GlmContextualCxG` |
| `xgb_traditional` | XGBoost | Traditional | `XGBoostCxGModel` |
| `xgb_contextual` | XGBoost | Contextual | `XGBoostCxGModel` |
| `lgbm_traditional` | LightGBM | Traditional | `LightGBMCxGModel` |
| `lgbm_contextual` | LightGBM | Contextual | `LightGBMCxGModel` |

---

## 2. Logistic Regression (GLM)

### 2.1 Model Structure

The baseline and contextual logistic regression models are both Generalised
Linear Models (GLMs) with a **Bernoulli response** and **logit link function**.

The linear predictor $\eta_i$ is a scalar combination of input features:

$$\eta_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}$$

where $x_{ij}$ are the (standardised) feature values and $\boldsymbol{\beta}$
are the estimated coefficients.

The logit link connects the linear predictor to the probability scale:

$$\text{logit}(p_i) = \log\frac{p_i}{1-p_i} = \eta_i$$

Inverting gives the **sigmoid** (logistic) function:

$$p_i = \sigma(\eta_i) = \frac{1}{1 + e^{-\eta_i}}$$

### 2.2 Contextual Linear Predictor

Expanding $\eta_i$ for the full contextual model:

$$\eta_i = \beta_0
+ \underbrace{\beta_1 d_i + \beta_2 \theta_i + \beta_3 x_i + \beta_4 y_i}_{\text{geometry}}
+ \underbrace{\sum_{k} \beta_k^{(\text{tech})} t_{ik}}_{\text{technique}}
+ \underbrace{\beta_5 \gamma_i^{(\text{opp})} + \beta_6 r_i^{(\text{keep})} + \cdots}_{\text{opponent quality}}
+ \underbrace{\beta_7 \delta_i + \beta_8 m_i + \cdots}_{\text{match state}}
+ \underbrace{\beta_9 n_i^{(\text{pass})} + \beta_{10} \text{dir}_i + \cdots}_{\text{sequence}}
+ \underbrace{\sum_{k} \beta_k^{(\text{cat})} \mathbb{1}[c_i = k]}_{\text{one-hot categoricals}}$$

where:
- $d_i$ = `distance_to_goal`, $\theta_i$ = `shot_angle`
- $\gamma_i^{(\text{opp})}$ = `opponent_defensive_rating`
- $r_i^{(\text{keep})}$ = `opponent_keeper_shot_stopping_rating`
- $\delta_i$ = `score_differential`, $m_i$ = `minute`
- $n_i^{(\text{pass})}$ = `passes_before_action`, $\text{dir}_i$ = `directness`

Each $\beta_j$ has a direct interpretation: a unit increase in standardised
feature $j$ multiplies the **odds** of a goal by $e^{\beta_j}$.

### 2.3 Parameter Estimation

Parameters are estimated by maximising the penalised log-likelihood:

$$\hat{\boldsymbol{\beta}} = \argmax_{\boldsymbol{\beta}} \left[
\ell(\boldsymbol{\beta}) - \frac{1}{2C} \|\boldsymbol{\beta}\|_2^2
\right]$$

where $\ell(\boldsymbol{\beta}) = \sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$
and $C > 0$ is the inverse regularisation strength (L2 / Ridge penalty).

The $\|\boldsymbol{\beta}\|_2^2$ term shrinks coefficients toward zero,
reducing variance at the cost of small bias — essential when the design matrix
has ~60 columns and only ~320 positive training examples.

The gradient with respect to $\boldsymbol{\beta}$ is:

$$\nabla_{\boldsymbol{\beta}} = \sum_{i=1}^{N} (y_i - p_i)\, \mathbf{x}_i - \frac{1}{C}\boldsymbol{\beta}$$

This has the elegant interpretation: at optimum, each coefficient is pulled
toward zero by the regulariser and pushed away by the weighted sum of residuals
$(y_i - p_i)$ scaled by the corresponding feature value.

Optimisation uses **L-BFGS** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno),
a quasi-Newton method that approximates the inverse Hessian from gradient
history. This converges faster than gradient descent for well-scaled problems
(hence the `StandardScaler` requirement).

Hyperparameters used:

| Parameter | Value | Meaning |
|---|---|---|
| `solver` | `lbfgs` | Quasi-Newton second-order method |
| `C` | 1.0 | Inverse L2 strength (no explicit tuning in this run) |
| `max_iter` | 2000 | Maximum L-BFGS iterations |
| `random_state` | 42 | Reproducibility seed |

---

## 3. XGBoost

### 3.1 Model Structure

XGBoost (Chen & Guestrin, 2016) builds an ensemble of $T$ regression trees
in a stage-wise fashion. Each tree $t$ fits a function $f_t : \mathbb{R}^p \to \mathbb{R}$
to the **pseudo-residuals** of the current ensemble. The prediction for shot $i$
after $T$ trees is:

$$\hat{y}_i^{(T)} = \sum_{t=1}^{T} f_t(\mathbf{x}_i)$$

For binary classification with logistic loss, the model outputs a log-odds
which is converted to probability via the sigmoid:

$$p_i = \sigma\!\left(\hat{y}_i^{(T)}\right) = \frac{1}{1 + e^{-\hat{y}_i^{(T)}}}$$

### 3.2 Objective Function

At each stage $t$, the objective to minimise is a second-order Taylor
approximation of the log-loss plus a regularisation term on the tree structure:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{N} \left[
g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t(\mathbf{x}_i)^2
\right] + \Omega(f_t)$$

where:
- $g_i = \frac{\partial \ell(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}} = p_i^{(t-1)} - y_i$ — first-order gradient (residual)
- $h_i = \frac{\partial^2 \ell}{\partial (\hat{y}_i^{(t-1)})^2} = p_i^{(t-1)}(1 - p_i^{(t-1)})$ — second-order Hessian
- $\Omega(f_t) = \gamma T_{\text{leaves}} + \frac{1}{2}\lambda \|w\|^2$ — complexity penalty on leaves and leaf weights

The optimal leaf weight for a leaf $j$ containing instance set $I_j$ is:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

This is the exact minimum of the second-order approximation within each leaf.

### 3.3 Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` | 300 | Number of boosting rounds |
| `learning_rate` | 0.05 | Step size shrinkage — each tree's contribution is scaled down |
| `max_depth` | 6 | Maximum tree depth — controls model capacity |
| `subsample` | 0.8 | Row subsampling ratio per tree (stochastic boosting) |
| `colsample_bytree` | 0.8 | Feature subsampling ratio per tree |
| `min_child_weight` | 5 | Minimum sum of Hessian in leaf — prevents low-coverage splits |
| `reg_alpha` | 0.0 | L1 penalty on leaf weights |
| `reg_lambda` | 1.0 | L2 penalty on leaf weights |

---

## 4. LightGBM

### 4.1 Differences from XGBoost

LightGBM (Ke et al., 2017) uses the same gradient boosting framework as XGBoost
but with two key algorithmic changes:

**Gradient-based One-Side Sampling (GOSS):** Rather than using all training
instances at each iteration, GOSS retains instances with large gradients
(informative instances) and randomly samples instances with small gradients.
This reduces computation while preserving the most informative gradient signal.

**Exclusive Feature Bundling (EFB):** Sparse, mutually exclusive features
(e.g. one-hot-encoded categories) are bundled into single features. This
reduces the effective number of features and speeds up histogram construction.

LightGBM also grows trees **leaf-wise** rather than depth-wise, expanding the
single leaf with the greatest loss reduction at each step, which can achieve
lower loss with fewer leaves but increases overfitting risk.

### 4.2 Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` | 300 | Boosting rounds |
| `learning_rate` | 0.05 | Step size |
| `num_leaves` | 63 | Maximum leaves per tree ($\approx 2^6-1$) |
| `subsample` | 0.8 | Row sampling |
| `colsample_bytree` | 0.8 | Feature sampling |
| `min_child_samples` | 20 | Minimum instances per leaf |
| `reg_alpha` | 0.0 | L1 |
| `reg_lambda` | 0.0 | L2 |

---

## 5. Design Matrix Construction

All three algorithm families use a shared feature resolution logic:

1. The `FeatureSetSpec` for the selected tier defines the nominal column lists
   (`numeric`, `boolean`, `categorical`).
2. Columns not present in the data are silently handled: numeric columns
   receive `NaN` (imputed by `SimpleImputer`); categorical columns receive
   `"unknown"`.
3. Boolean columns are cast to `float` before entering the numeric transformer
   so imputation and scaling behave correctly.
4. After `ColumnTransformer`, the matrix is dense (GLM) or passed directly
   to the tree estimator (XGBoost/LightGBM support sparse inputs natively).

This ensures all six candidates see the same input matrix for their respective
feature tier, making the comparison clean.
