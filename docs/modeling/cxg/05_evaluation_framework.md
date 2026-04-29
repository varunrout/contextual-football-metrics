# CxG — Evaluation Framework

## 1. Overview

Three complementary metrics evaluate each candidate model. They measure different
properties of the predicted probability distribution and together provide a
complete picture of model quality.

| Metric | Measures | Primary use |
|---|---|---|
| Log-loss (cross-entropy) | Calibration + discrimination jointly | **Model selection criterion** |
| Brier score | Mean squared probability error | Calibration |
| ROC-AUC | Discriminative rank ordering | Discrimination |

---

## 2. Log-Loss (Binary Cross-Entropy)

### 2.1 Definition

$$\mathcal{L}_{\log} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

Log-loss is the negative log-likelihood of the Bernoulli model, normalised by
$N$. It penalises **confident wrong predictions** exponentially: predicting
$\hat{p} = 0.99$ when $y = 0$ contributes $-\log(0.01) \approx 4.6$ to the
sum — approximately 100 times the penalty for predicting $\hat{p} = 0.5$.

### 2.2 Interpretation for Shot Models

A naive model that always predicts the base rate $\bar{y} \approx 0.115$
achieves log-loss:

$$\mathcal{L}_{\log}^{(\text{null})} = -[0.115 \log 0.115 + 0.885 \log 0.885] \approx 0.379$$

A model with $\mathcal{L}_{\log} = 0.241$ (our best) is making substantially
more informative probability estimates. There is no absolute target; the
comparison is among candidates and against the null.

### 2.3 Why Log-Loss is the Primary Selection Criterion

Log-loss is the natural objective for a **probability model**: it is strictly
proper (maximised only by the true probability), and minimising it is equivalent
to maximising the likelihood of the data. AUC is purely ordinal and rewards
ranking but not calibration. A model can achieve high AUC while being
systematically over- or under-confident, which matters when CxG scores are used
as numerical inputs to CxA and CxT.

---

## 3. Brier Score

### 3.1 Definition

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$

The Brier score is the mean squared error between predicted probabilities and
binary outcomes. It ranges from 0 (perfect) to 1 (perfectly wrong).

### 3.2 Decomposition

The Brier score decomposes into three components:

$$\text{BS} = \underbrace{\text{Uncertainty}}_{\bar{y}(1-\bar{y})} - \underbrace{\text{Resolution}}_{\text{variance of conditional means}} + \underbrace{\text{Reliability}}_{\text{calibration error}}$$

- **Uncertainty** is irreducible — it depends only on the base rate.
- **Resolution** measures how much the model separates shots by difficulty.
- **Reliability** measures systematic over- or under-prediction.

At a base rate of 11.5%, uncertainty $= 0.115 \times 0.885 \approx 0.102$.
Our best model achieves BS $= 0.063$, meaning it eliminates about 38% of the
reducible Brier score.

---

## 4. ROC-AUC

### 4.1 Definition

The Area Under the Receiver Operating Characteristic Curve (ROC-AUC) is:

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t))\, dt$$

Equivalently, AUC is the probability that the model assigns a higher score to a
randomly chosen positive (goal) than to a randomly chosen negative (non-goal):

$$\text{AUC} = P(\hat{p}_{i^+} > \hat{p}_{j^-})$$

where $i^+$ is a random goal shot and $j^-$ is a random non-goal shot.

### 4.2 Properties and Limitations

AUC is threshold-independent and invariant to any monotone transformation of
$\hat{p}$. A model can achieve high AUC while being completely miscalibrated
(e.g. all probabilities multiplied by 100). For this reason, AUC is treated as
a secondary metric here — it confirms that the model discriminates correctly,
but calibration (log-loss, Brier) governs model selection.

Random classifier: AUC = 0.5. Perfect classifier: AUC = 1.0.
At 11.5% positives, AUC $\approx 0.81$ is a strong result for a tabular model
on event data.

---

## 5. Calibration Assessment

Beyond the scalar Brier score, model calibration is assessed visually via
**reliability diagrams** (`reports/figures/cxg/calibration.png`).

Shots are binned by predicted probability into $B = 10$ equal-width bins.
For each bin $b$ containing $n_b$ shots, the mean predicted probability
$\bar{p}_b$ is plotted against the empirical fraction of goals $o_b$:

$$o_b = \frac{1}{n_b} \sum_{i \in b} y_i$$

A perfectly calibrated model has $o_b \approx \bar{p}_b$ for all $b$,
lying on the diagonal. Points above the diagonal indicate under-prediction
(the model is too conservative); points below indicate over-prediction.

---

## 6. Evaluation Phases

### 6.1 Cross-Validation Metrics (Model Selection)

CV metrics are computed during the ladder run. For each fold $k$:

$$\mathcal{L}_{\log}^{(k)} = -\frac{1}{|\mathcal{V}_k|} \sum_{i \in \mathcal{V}_k}
\left[ y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \right]$$

The reported CV log-loss is the **arithmetic mean across the 5 valid folds**:

$$\overline{\mathcal{L}}_{\log}^{(\text{CV})} = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}_{\log}^{(k)}$$

Model selection ranks candidates by $\overline{\mathcal{L}}_{\log}^{(\text{CV})}$
(lower is better). The best model's predictions are from models that **never
saw the validation set during fitting**, making this an unbiased estimate of
generalisation performance.

### 6.2 Held-Out Test Metrics (Generalisation)

After the CV-based ranking and final refit, each model is evaluated on the
**Euro 2024 held-out set** (1,340 shots). This evaluation:

1. Uses the model fitted on all 2,783 CV-pool shots.
2. Computes log-loss, Brier score, and AUC on the 1,340 held-out shots.
3. Is reported separately from CV metrics — it is not used for model selection.

The held-out metrics serve as an independent check on whether CV scores
generalised to an unseen tournament. Good generalisation is indicated by
$|\overline{\mathcal{L}}_{\log}^{(\text{heldout})} - \overline{\mathcal{L}}_{\log}^{(\text{CV})}| \lesssim 0.02$.

---

## 7. Leakage Audit

Prior to the current implementation, a data leakage bug was identified and
corrected:

**Symptom:** Tree models showed heldout AUC values of 0.94–1.00 while their
CV AUC was 0.74–0.79. LightGBM contextual achieved AUC = 1.0000 on held-out.

**Root cause:** The `shots_df` passed to `CxGLadder.run()` included the 1,340
Euro 2024 `val_test` shots. The final production models were fitted on all
4,123 shots (including held-out). When the same held-out rows were presented
for evaluation, tree models with sufficient depth had memorised them exactly,
producing artificially perfect metrics. Logistic regression, being a global
model with strong regularisation, did not exhibit this to the same degree.

**Fix:** Added a `split_role != "val_test"` filter in `train_cxg.py` before
calling the ladder, reducing the training pool from 4,123 to 2,783 shots.

**Verification:** After the fix, all models show CV–heldout AUC alignment
within ~2 percentage points (see `06_results_and_discussion.md`).
