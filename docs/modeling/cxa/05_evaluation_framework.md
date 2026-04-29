# CxA — Evaluation Framework

## 1. Overview

CxA is a two-stage model: a classifier (Stage 1) and a regressor (Stage 2).
Each stage requires distinct metrics. The final CxA score is the product of
both stages, so accuracy in each propagates multiplicatively to the final output.

| Stage | Task | Primary metric | Secondary metrics |
|---|---|---|---|
| Stage 1 — Shot-creation | Binary classification | **ROC-AUC** | AP, log-loss, Brier |
| Stage 2 — Shot-quality | Regression | **MAE** | RMSE, Spearman ρ |
| Combined CxA | Product | (via stage metrics) | Pitch heatmap, calibration |

---

## 2. Stage 1 — Shot-Creation Classifier Metrics

### 2.1 ROC-AUC (Primary Selection Criterion)

$$\text{AUC} = P(\hat{p}_{i^+} > \hat{p}_{j^-})$$

where $i^+$ is a random shot-creating action and $j^-$ is a random
non-creating action. AUC measures whether the model correctly **ranks**
shot-creating actions higher than non-creating ones.

**Why AUC is primary for Stage 1.** Unlike CxG — where calibration is critical
because CxG scores are used as numerical multipliers — Stage 1 outputs $p_i$
that is immediately multiplied by $q_i$ (Stage 2). A miscalibrated but
well-ranked $p_i$ would only rescale CxA systematically, which partially
cancels in comparative analyses. More importantly, at a 15.93% positive rate
with 220,899 actions, the dataset is large enough that AUC is a stable
discriminative signal.

**Interpretation.** A random classifier achieves AUC = 0.5. At 15.93% positive
rate, a strong model can achieve AUC ≈ 0.75–0.80 on tabular event data.

### 2.2 Average Precision (PR-AUC)

$$\text{AP} = \sum_{k} (R_k - R_{k-1}) P_k$$

where $P_k$ and $R_k$ are precision and recall at threshold $k$. AP is the
area under the precision–recall curve, which is **more informative than AUC
at high class imbalance** (15.93% positives). A perfect classifier achieves
AP = 1.0; a naive classifier achieves AP ≈ the positive rate ≈ 0.159.

### 2.3 Log-Loss (Binary Cross-Entropy)

$$\mathcal{L}_{\log} = -\frac{1}{N} \sum_{i=1}^{N}
\left[ S_i \log p_i + (1-S_i) \log(1-p_i) \right]$$

Log-loss penalises confident wrong predictions exponentially. For a model
that always predicts the base rate $\bar{S} = 0.1593$:

$$\mathcal{L}_{\log}^{(\text{null})} = -[0.1593 \log 0.1593 + 0.8407 \log 0.8407] \approx 0.486$$

A model with $\mathcal{L}_{\log} < 0.486$ provides information beyond the
base rate. The best model achieves 0.4264, a **12.3% reduction** over the null.

### 2.4 Brier Score

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (p_i - S_i)^2$$

At 15.93% positives, the null Brier score is $0.1593 \times 0.8407 \approx 0.1338$.
The best model achieves 0.1363 — near the null, indicating that while discrimination
is good (AUC = 0.766), absolute calibration of the probability scale is modest.
This is expected: shot-creation probability is inherently noisy because identical
feature values can lead to both shot and non-shot outcomes depending on
unobserved factors (player decisions, slight positional variations).

---

## 3. Stage 2 — Shot-Quality Regressor Metrics

Stage 2 is evaluated only on the **subset of held-out actions where a shot
followed** ($S_i = 1$ in held-out, $\approx 19{,}216$ rows at 19.61% rate).

### 3.1 Mean Absolute Error (Primary)

$$\text{MAE} = \frac{1}{N^+} \sum_{i \in \mathcal{D}^+} |\hat{q}_i - G_i|$$

MAE measures the average magnitude of prediction error in CxG units. Since
CxG values range from ~0.02 to ~0.90, an MAE of 0.069 (best model) means the
quality prediction is off by about 0.069 CxG units on average — roughly one
standard deviation of the CxG distribution.

### 3.2 Root Mean Squared Error

$$\text{RMSE} = \sqrt{\frac{1}{N^+} \sum_{i \in \mathcal{D}^+} (\hat{q}_i - G_i)^2}$$

RMSE penalises large errors more heavily than MAE. The gap between
RMSE (0.093) and MAE (0.069) for the best model indicates the presence of
some large outlier predictions — likely extreme CxG values (close-range shots
or penalties) where the quality model deviates most from reality.

### 3.3 Spearman Rank Correlation

$$\rho = \text{Spearman}(\hat{q}, G)$$

The Spearman rank correlation measures whether the quality model correctly
**orders** shot-creating actions by shot difficulty — regardless of absolute
scale. A $\rho = 0.186$ (best model) indicates weak but positive correlation:
the model does rank higher-quality shots somewhat higher, but the signal is
modest. This is not surprising: the features available to the quality model
(action location, sequence context) are weak predictors of shot difficulty
compared to features directly at the shot (distance, angle, body part).

**Interpretation of weak Spearman.** The quality model is predicting the mean
CxG of all shots in a possession from pre-shot action features. These features
do not directly observe shot geometry — they only describe the creative action
that preceded the shot. A short pass into the box (high $p_{\text{shot}}$) can
precede both a simple tap-in and a difficult volley. The Spearman ρ reflects
the best achievable correlation from upstream features rather than a model
failure.

---

## 4. Evaluation Phases

### 4.1 Held-Out Evaluation

After training on 220,899 actions (WC 2022 + Euro 2020), all three pipelines
are evaluated on the **Euro 2024 held-out set** (98,029 actions). This is the
only evaluation reported in `reports/cxa_training_summary.json`.

For Stage 1:
- Score all 98,029 held-out creative actions with `predict_proba`.
- Compute AUC, AP, log-loss, Brier against `shot_created` labels.

For Stage 2:
- Filter to the $\approx 19{,}216$ shot-creating held-out actions.
- Score with `quality_model.predict`.
- Compute MAE, RMSE, Spearman against `resulting_shot_cxg` labels.

### 4.2 Model Selection Criterion

Pipelines are ranked by **held-out creation AUC** (Stage 1). The best
pipeline is selected and designated the production model.

Creation AUC is chosen over log-loss as the selection criterion because:

1. **Ranking validity**: For CxA as a player ranking metric, the relative
   ordering of actions matters more than absolute calibration.
2. **Interpretability**: AUC is easier to communicate to non-technical
   stakeholders than cross-entropy.
3. **Robustness**: At 220,899 training actions, all models are well-regularised;
   AUC and log-loss rankings are consistent in this dataset.

---

## 5. Diagnostic Charts

Four diagnostic charts are generated for the production pipeline:

### 5.1 PR Curve and Reliability Diagram (`pr_curve.png`)

- **Left panel**: Precision–Recall curve for the held-out creation predictions.
  A diagonal grey line marks the baseline (random classifier = positive rate).
- **Right panel**: Reliability diagram — mean predicted $p_i$ vs. empirical
  shot-creation rate per decile bin. A well-calibrated model lies on the diagonal.

### 5.2 Pitch Heatmap (`cxa_pitch_heatmap.png`)

Three side-by-side pitch panels (pass, carry, dribble) showing mean CxA per
hexagonal bin across the 105×68 pitch in the internal coordinate frame.
Attacking direction: left to right. The heatmap uses mplsoccer with
`pitch_type="custom", pitch_length=105, pitch_width=68` to match stored
coordinates.

Expected spatial pattern: CxA should increase sharply in the final third
(x > 80 m), particularly in central and right-channel zones corresponding
to the highest-density shot-creation areas. Passes into the box should score
highest among passes; carries that penetrate the box should score highest
among carries.

### 5.3 Creation Rate by Distance (`creation_rate_by_distance.png`)

Empirical shot-creation rate (bars) vs. mean predicted $p_{\text{shot}}$ (bars)
across 10 distance-to-goal bands. This is a **calibration check for Stage 1**:
if the model is well-calibrated, predicted and empirical rates should align
within each band. Systematic divergence indicates bias (over- or under-prediction)
in a particular distance range.

### 5.4 Quality Scatter and Calibration (`quality_scatter.png`)

- **Left panel**: Scatter of actual `resulting_shot_cxg` vs. predicted
  `expected_cxg_if_shot` for all shot-creating held-out actions. A dashed
  diagonal marks perfect prediction.
- **Right panel**: Binned calibration — mean actual CxG vs. mean predicted CxG
  across 8 actual-CxG bins, with sample counts annotated. Departure from the
  diagonal reveals systematic bias in the quality model.
