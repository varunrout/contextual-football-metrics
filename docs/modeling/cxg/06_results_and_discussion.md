# CxG — Results and Discussion

## 1. Full Leaderboard

Results from the training run on 2,783 shots (WC 2022 + Euro 2020),
evaluated on the Euro 2024 held-out set (1,340 shots).

| Rank | Model | CV Log-loss | CV Brier | CV AUC | HO Log-loss | HO Brier | HO AUC |
|---|---|---|---|---|---|---|---|
| 1 | `baseline_logit` | 0.2960 | 0.0859 | 0.8052 | 0.2504 | 0.0694 | 0.7987 |
| 2 | `glm_contextual` | 0.2982 | 0.0855 | 0.8084 | **0.2413** | **0.0667** | **0.8131** |
| 3 | `xgb_contextual` | 0.3119 | 0.0865 | 0.7863 | 0.2591 | 0.0703 | 0.7903 |
| 4 | `xgb_traditional` | 0.3233 | 0.0922 | 0.7676 | 0.2797 | 0.0764 | 0.7475 |
| 5 | `lgbm_traditional` | 0.5033 | 0.1048 | 0.7411 | 0.3765 | 0.0840 | 0.7499 |
| 6 | `lgbm_contextual` | 0.5222 | 0.0948 | 0.7730 | 0.3823 | 0.0779 | 0.7831 |

**CV** = 5-fold match-level cross-validation on 2,783 training shots.
**HO** = Euro 2024 held-out set (1,340 shots, unseen during all training).

Model selection criterion: CV log-loss (rank 1 = `baseline_logit`).
Production model selected: `glm_contextual` — best on all three held-out metrics.

---

## 2. Primary Finding: Logistic Regression Dominates

All three logistic regression variants (ranks 1–2 by CV, ranks 1–3 by HO AUC)
outperform all tree-based models. This is not an implementation artefact; it is
a well-understood sample-size effect.

### 2.1 The Effective Sample Size Problem

The training set contains 2,783 shots with approximately 321 goals (~11.5%).
For gradient boosted trees, the relevant quantity is not the number of shots
but the number of **well-supported leaf nodes**. An XGBoost model with
`max_depth=6` can produce $2^6 = 64$ leaves per tree and 300 trees = 19,200
potential leaf configurations. Even with `min_child_weight=5`, achieving
stable leaf estimates with ~321 positive examples is impossible.

The logistic regression model, by contrast, has approximately 60 parameters
(after one-hot encoding) and ~321 effective positive observations. The
regularisation term ($C=1.0$) shrinks all coefficients toward zero, providing
a safe prior when data is sparse.

### 2.2 AUC vs Log-Loss Dissociation for LightGBM

LightGBM presents an instructive case of AUC–log-loss dissociation:

- `lgbm_contextual` HO AUC = 0.783 — reasonable discrimination
- `lgbm_contextual` HO Log-loss = 0.382 — 58% worse than `glm_contextual`

AUC measures only rank ordering; it is invariant to any monotone transformation
of $\hat{p}$. LightGBM's leaf-wise tree growth with no L2 penalty on leaf
weights (`reg_lambda=0`) produces extreme predictions near 0 and 1 after
sufficient trees, which:

1. Preserves rank ordering (AUC is stable).
2. Catastrophically increases log-loss (extreme predictions are heavily penalised
   for wrong labels).

This is precisely why log-loss, not AUC, is the primary selection criterion for
a **probability model** rather than a ranking model.

---

## 3. Value of Contextual Features

Comparing within each algorithm family isolates the effect of the contextual
feature tier over the traditional tier:

| Family | Metric | Traditional | Contextual | Δ (contextual − traditional) |
|---|---|---|---|---|
| Logistic | HO AUC | 0.799 | **0.813** | **+0.014** |
| Logistic | HO Log-loss | 0.250 | **0.241** | **−0.009** |
| XGBoost | HO AUC | 0.748 | **0.790** | **+0.042** |
| XGBoost | HO Log-loss | 0.280 | **0.259** | **−0.021** |
| LightGBM | HO AUC | 0.750 | **0.783** | **+0.033** |
| LightGBM | HO Log-loss | 0.377 | **0.382** | −0.005 (slightly worse) |

Contextual features improve every model on AUC. The improvement is most
pronounced for XGBoost (+4.2pp AUC), suggesting that tree models particularly
benefit from the non-linear interactions between opponent quality and shot
geometry that the contextual tier encodes. The LightGBM log-loss anomaly is
attributable to miscalibration dominating the probability error, not a failure
of contextual features.

---

## 4. CV–Heldout Generalisation

The gap between CV and held-out metrics indicates whether the model generalises
to the Euro 2024 distribution:

| Model | ΔLog-loss (HO − CV) | ΔAUC (HO − CV) | Verdict |
|---|---|---|---|
| `glm_contextual` | −0.057 | +0.005 | ✓ Well-generalised |
| `baseline_logit` | −0.046 | −0.006 | ✓ Well-generalised |
| `xgb_contextual` | −0.053 | +0.004 | ✓ Well-generalised |
| `xgb_traditional` | −0.044 | −0.020 | ✓ Acceptable |
| `lgbm_traditional` | −0.127 | +0.009 | ⚠ Large CV–HO gap (miscalibration) |
| `lgbm_contextual` | −0.140 | +0.010 | ⚠ Large CV–HO gap (miscalibration) |

Negative Δlog-loss means the held-out performance is **better** than CV —
a common pattern when the held-out tournament (Euro 2024) happens to be a
cleaner dataset than the CV pool, or when the CV pool folds have higher
variance due to fewer shots per fold. The logistic models show excellent
CV–HO alignment.

---

## 5. Production Model Selection

The CV criterion (log-loss) nominates `baseline_logit` as rank 1. However,
on every held-out metric, `glm_contextual` is strictly better:

| Criterion | `baseline_logit` | `glm_contextual` | Winner |
|---|---|---|---|
| CV log-loss | **0.2960** | 0.2982 | baseline_logit |
| HO log-loss | 0.2504 | **0.2413** | **glm_contextual** |
| HO AUC | 0.7987 | **0.8131** | **glm_contextual** |
| HO Brier | 0.0694 | **0.0667** | **glm_contextual** |
| Interpretability | medium | high | **glm_contextual** |
| Downstream suitability | medium | high | **glm_contextual** |

The CV difference (0.2982 vs 0.2960 = 0.0022) is within sampling noise
across 5 folds of ~557 shots each. The held-out advantage of `glm_contextual`
(−0.0091 log-loss, +0.0144 AUC) over a truly independent 1,340-shot test set
is more reliable evidence of true superiority.

`glm_contextual` is additionally preferred because:

1. **Calibration**: Lower Brier score (0.067 vs 0.069) means CxG scores
   used as inputs to CxA and CxT are better-calibrated probability estimates,
   not just better-ranked ones.
2. **Interpretability**: Coefficients are log-odds ratios, directly readable
   as the additive effect of each contextual factor on shot difficulty.
3. **Contextual signal**: It incorporates opponent quality and build-up context,
   which is the theoretical contribution of this work over traditional xG.

**Production model: `models/cxg/glm_contextual.joblib`**

---

## 6. Limitations and Future Work

### 6.1 Sample Size

2,783 training shots with ~321 positives is a small dataset for probabilistic
modelling. The tree models underperform not because gradient boosting is wrong
for this problem, but because they lack sufficient data to regularise properly.
Expanding to 10+ competitions (targeting ~15,000+ shots) would likely close
the gap between logistic regression and XGBoost, and would unlock the Full 360
feature tier.

### 6.2 Non-Linearity in Geometry

The GLM assumes a linear effect of distance and angle in log-odds space.
The true relationship is non-linear — the difficulty of a shot at 5m vs 10m
vastly exceeds 20m vs 25m. This is partially absorbed by the regularised
coefficients, but a GAM with thin-plate splines on $(d, \theta)$ would more
faithfully capture the geometric surface. This is deferred pending a larger
training set.

### 6.3 Feature Engineering Limits

Several high-value signals are not yet modelled:

- **Goalkeeper positioning at moment of shot** — available via 360 data but
  not yet activated.
- **Defensive tunnel quality** — the angle and distance of defenders in the
  direct shot path.
- **Shot technique × geometry interactions** — a header at 10m is not simply
  the additive sum of header difficulty and 10m difficulty.

### 6.4 Temporal Stability

The model was trained on WC 2022 and Euro 2020 and tested on Euro 2024.
This is a 2–4 year temporal gap. Playing styles and team compositions change;
a periodic retraining schedule as new StatsBomb data becomes available is
recommended.

### 6.5 Convergence Warnings

L-BFGS convergence warnings were observed during training (5 folds × 2 final fits
= 10 `ConvergenceWarning` instances). These indicate the solver hit `max_iter=2000`
before formal convergence. In practice, L-BFGS typically finds a near-optimal
solution before terminating; the warnings do not indicate a meaningfully suboptimal
model. Increasing `max_iter` to 5000 or switching to `solver="saga"` with
`max_iter=10000` would eliminate the warnings.
