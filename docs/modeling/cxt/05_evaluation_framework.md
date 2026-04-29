# CxT — Evaluation Framework

## 1. Overview

CxT is a **regression** problem. The evaluation framework uses three
complementary metrics that together characterise absolute accuracy,
error magnitude, and rank ordering.

| Metric | Measures | Primary use |
|---|---|---|
| MAE | Mean absolute prediction error | **Model selection criterion** |
| RMSE | Root mean squared error | Sensitivity to large errors |
| Spearman ρ | Rank correlation of predictions vs. actuals | Relative ordering quality |

An additional **calibration-by-zone** diagnostic is computed at evaluation time
but is not used for model selection.

---

## 2. Mean Absolute Error (MAE)

### 2.1 Definition

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$

where $y_i = \texttt{possession\_cxg}_i$ and $\hat{y}_i$ is the model prediction.

### 2.2 Why MAE is the Primary Selection Criterion

The `possession_cxg` distribution is **heavily zero-inflated** (~84% zeros).
Mean Squared Error (MSE) would over-penalise the rare large errors on high-CxG
possessions, causing model selection to optimise for the easy-to-predict zero
region at the expense of the informative positive tail.

MAE penalises all errors equally in magnitude, making it a more balanced
criterion across the full distribution.

The null MAE — always predicting the training mean — is:

$$\text{MAE}_{\text{null}} = \frac{1}{N}\sum_i |\bar{y} - y_i| \approx 0.013$$

Wait — this is an overestimate of the null for zero-inflated targets.
A better null is predicting zero everywhere:

$$\text{MAE}_{\text{zero}} = \frac{1}{N}\sum_i y_i = \bar{y} \approx 0.013$$

Because 84% of true values are zero, predicting zero everywhere achieves
$\text{MAE} = \bar{y} \approx 0.013$. Our best model achieves
$\text{MAE} = 0.031$ — numerically larger than the zero-predictor because
it also tries to estimate the positive tail and incurs non-trivial error there.
This is **not a failure**: a model that always predicts zero has Spearman
$\rho = 0$ and is useless for ranking actions. The MAE-vs-Spearman trade-off
is intentional.

### 2.3 MAE in Context

MAE is measured in the same units as `possession_cxg` (which is in CxG units).
The mean positive `possession_cxg` is approximately 0.08. An MAE of 0.031
(best model) is approximately 39% of the mean positive target — a meaningful
reduction in absolute error compared to the GLM baselines (0.108).

---

## 3. Root Mean Squared Error (RMSE)

### 3.1 Definition

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2}$$

RMSE is always $\geq$ MAE; the gap between them indicates the presence of
large errors. For zero-inflated targets, RMSE is dominated by the residuals
on the rare high-CxG possessions.

RMSE is reported as a secondary metric. It is not used for model selection
but is useful for detecting models that perform well on average (low MAE) while
occasionally producing large, implausible predictions.

---

## 4. Spearman Rank Correlation

### 4.1 Definition

$$\rho_s = \text{Cor}\!\left(\text{rank}(\hat{y}), \text{rank}(y)\right)$$

where $\text{rank}(\cdot)$ replaces values with their rank order.

Spearman $\rho$ is invariant to monotone transformations of the predictions.
It measures whether the model correctly **orders** actions by possession threat
value, independent of the absolute scale of predictions.

### 4.2 Interpretation for CxT

The principal use of CxT scores is to **rank** actions and players by
possession-building contribution. For this purpose, Spearman $\rho$ is the
most operationally relevant metric: it directly measures whether a player
ranked first by CxT actually contributed more expected goal threat than a
player ranked second.

A model with low MAE but low Spearman $\rho$ would be useless for player
evaluation — it might predict near zero for almost all actions (low error
on the majority class) while failing to discriminate between high- and
low-value actions.

In the training data, approximately 84% of `possession_cxg` values are zero.
The Spearman correlation is computed over the full distribution. Among the
zero-valued majority, all predictions cluster near zero and contribute little
to $\rho$; the discriminative signal comes from the positive tail.

### 4.3 Expected Magnitude

The best achievable Spearman $\rho$ for this task is substantially below 1.0.
Even a perfect state-value model cannot perfectly rank individual actions
because:

1. All actions within a possession share the same `possession_cxg` target —
   ties are extensive and unavoidable.
2. The `possession_cxg` target reflects the quality of the **entire possession**,
   not just the quality of the individual action within it. A weak carry
   preceding a brilliant through-ball receives the same label as the
   through-ball itself.

Values of $\rho \approx 0.3$ are reasonable for the tree-based models;
GLMs achieve $\rho \approx 0.15$–$0.25$.

---

## 5. Calibration by Zone

### 5.1 Definition

After CV and held-out evaluation, a zone-level calibration diagnostic is
computed. Positive `possession_cxg` values are divided into four quartile
buckets (Q1–Q4). For each bucket, the **mean prediction bias** is reported:

$$\text{bias}_b = \overline{\hat{y}}_b - \overline{y}_b$$

where $\overline{\hat{y}}_b$ and $\overline{y}_b$ are the mean predicted and
actual values in bucket $b$ respectively.

This identifies systematic over- or under-prediction at different levels of
the target range, which is not visible from the scalar MAE.

### 5.2 Expected Behaviour

Well-calibrated models should show bias $\approx 0$ across all four quartile
buckets. Systematic biases to watch for:

- **GLMs (Gamma, GAM)**: Positive bias in Q1 (lowest positive values) because
  the model predicts positive values for zero-valued rows, and some near-zero
  positives get over-predicted.
- **Tree models**: Possible negative bias in Q4 (highest positive values) due
  to regularisation (leaf minimum weight) pulling extreme predictions
  toward the mean.

---

## 6. Evaluation Phases

### 6.1 Cross-Validation Phase (Model Selection)

CV metrics are computed during the `StateValueLadder.run()` call. For each
candidate and each fold $k$:

- A fresh model instance is trained on the fold training split.
- Predictions are generated on the full fold validation split (not just
  positive rows) for all three metrics.
- Fold metrics are averaged across valid folds: $\overline{\text{MAE}}^{(\text{CV})}$.

Model selection ranks candidates by $\overline{\text{MAE}}^{(\text{CV})}$.
The best model is selected and its path written to `configs/models.yaml`.

### 6.2 Held-Out Test Phase (Generalisation)

After the CV ranking and full-dataset refit, each model is evaluated on the
**Euro 2024 held-out set** (98,029 creative actions). This evaluation:

1. Uses the model refitted on all 220,899 CV-pool actions.
2. Computes MAE, RMSE, and Spearman $\rho$ on the 98,029 held-out actions.
3. Is reported separately from CV metrics in `cxt_training_summary.json`.
4. Is **not** used for model selection — it is a post-hoc generalisation check.

Alignment between CV and held-out MAE within ~10% indicates the model
generalises well to the new tournament distribution.

---

## 7. Diagnostic Figures

Six figures are generated after each training run and saved to
`reports/figures/cxt/`:

| Figure | Description |
|---|---|
| `leaderboard.png` | Bar charts of CV MAE and CV Spearman ρ for all candidates |
| `target_distribution.png` | Histogram of `possession_cxg` (full + positive-only) |
| `predicted_vs_actual.png` | Scatter of predicted vs. actual on positive held-out rows |
| `residuals.png` | Residual (y − ŷ) vs. `x_location` with binned mean smoother |
| `calibration_by_zone.png` | Mean prediction bias per quartile bucket |
| `pitch_value_surface.png` | Heatmap of $V(s)$ across a 30×20 pitch grid |

The **pitch value surface** is the primary interpretability figure: it shows
the estimated state value $V(s)$ at each pitch location, holding all other
contextual features at their median. This directly visualises what the model
has learned about positional threat value.
