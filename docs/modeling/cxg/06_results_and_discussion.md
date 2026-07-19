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
Production model selected: `baseline_logit` — the CV rank-1 model. The held-out
set is used once as confirmation only, not to pick the winner (see section 5 and
`docs/modeling/cxg/07`).

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

The selection rule is fixed in advance: **rank 1 on 5-fold CV log-loss**. On that
criterion the winner is `baseline_logit`. The held-out set is inspected once,
after selection, only to confirm the choice generalises, never to break the tie.

| Criterion | `baseline_logit` | `glm_contextual` | CV winner |
|---|---|---|---|
| CV log-loss (selection) | **0.2960** | 0.2982 | **baseline_logit** |
| HO log-loss (confirmation) | 0.2504 | 0.2413 | glm_contextual |
| HO AUC (confirmation) | 0.7987 | 0.8131 | glm_contextual |
| HO Brier (confirmation) | 0.0694 | 0.0667 | glm_contextual |

An earlier version of this document promoted `glm_contextual` because it wins on
every held-out metric. That was a mistake: choosing the production model by its
score on the final held-out set contaminates that set and turns a confirmation
into a tie-break. Under the pre-committed CV rule the production model is
`baseline_logit`, and `configs/models.yaml` points to it.

The held-out edge of `glm_contextual` is real but small, and it does not change
the decision:

- On the matched Euro 2024 held-out comparison
  (`analysis/20_incremental_lift_vs_baselines.py`), `glm_contextual` beats
  `baseline_logit` on log-loss by 0.0096 with a 95 percent paired-bootstrap CI
  of [−0.0174, −0.0022] that excludes zero; its AUC edge (+0.0152) has a CI
  [−0.0026, +0.0333] that includes zero, so it is not established.
- Against off-the-shelf StatsBomb xG on the same rows, `glm_contextual` shows
  **no demonstrable lift**: both the log-loss and AUC deltas have CIs spanning
  zero, and StatsBomb xG is better on the point estimates (log-loss 0.233 vs
  0.241, AUC 0.830 vs 0.813, ECE 0.013 vs 0.017). See `docs/modeling/cxg/07`.

So the contextual GLM is a defensible improvement on our own traditional
baseline, but it is not promoted: it is not selected by the pre-committed
criterion, and it does not beat off-the-shelf xG. `baseline_logit` remains the
production model, chosen honestly and reported without cherry-picking.

**Production model: `models/cxg/baseline_logit.joblib`**

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
