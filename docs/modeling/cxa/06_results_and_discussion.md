# CxA — Results and Discussion

## 1. Full Leaderboard

Results from training on 220,899 creative actions (WC 2022 + Euro 2020),
evaluated on the Euro 2024 held-out set (98,029 actions, 19.61% positive rate).

### 1.1 Stage 1 — Shot-Creation Classifier

| Rank | Pipeline | Creation AUC | Creation AP | Creation LL | Creation Brier |
|---|---|---|---|---|---|
| **1** | `logistic_contextual` | **0.7662** | **0.4430** | **0.4264** | **0.1363** |
| 2 | `xgb_contextual` | 0.7530 | 0.3955 | 0.4437 | 0.1413 |
| 3 | `lgbm_contextual` | 0.7281 | 0.3571 | 0.4885 | 0.1508 |

**Null classifier** (always predicts 19.61% base rate): AP ≈ 0.196, LL ≈ 0.517.

### 1.2 Stage 2 — Shot-Quality Regressor

| Rank | Pipeline | Quality MAE | Quality RMSE | Quality Spearman |
|---|---|---|---|---|
| **1** | `logistic_contextual` | **0.0694** | **0.0934** | 0.1855 |
| 2 | `lgbm_contextual` | 0.0714 | 0.0981 | **0.2067** |
| 3 | `xgb_contextual` | 0.0722 | 0.0987 | 0.1937 |

**Production model selected: `logistic_contextual`** — rank 1 on creation AUC
and all three Stage 1 metrics, and rank 1 on Stage 2 MAE/RMSE.

Saved to: `models/cxa/cxa_logistic_contextual.pkl`
Config pointer: `configs/models.yaml → production.cxa`

---

## 2. Primary Finding: Logistic Regression Dominates

All Stage 1 metrics favour the logistic pipeline over both tree-based models.
The margin is largest on log-loss and AP:

- **Creation AUC**: logistic (0.766) vs. lgbm (0.728) → +3.8pp
- **Creation AP**: logistic (0.443) vs. lgbm (0.357) → +8.6pp
- **Creation LL**: logistic (0.426) vs. lgbm (0.489) → 14.7% lower loss

### 2.1 Why Logistic Regression Wins on a Large Dataset

CxA trains on 220,899 actions — far larger than the CxG dataset (2,783 shots).
This might suggest tree models would thrive. The opposite occurs because the
**class imbalance** (15.93% positive rate = 35,168 positives) limits the
effective training signal for tree models, even though total $N$ is large:

- XGBoost with `max_depth=4` produces $2^4 = 16$ leaves per tree and 300 trees.
  The $\approx 35{,}168$ positive examples are spread across trees and can
  support stable leaf splits — but the interaction of 16 leaves × categorical
  one-hot dimensions produces many sparsely occupied leaves.
- LightGBM leaf-wise growth finds the highest-gain split at each step. With
  `num_leaves=15`, it can model complex non-linearities, but `is_unbalance=True`
  introduces effective upweighting that can destabilise probability estimates
  for the negative class.

The logistic model, by contrast, has approximately $60$–$80$ parameters after
one-hot encoding and is regularised with $C = 1.0$. With 35,168 effective
positive training examples, the parameter-to-data ratio is approximately
$440:1$ — far more favourable than for the tree models.

### 2.2 LightGBM: AUC–AP Dissociation

LightGBM's AUC (0.728) is lower than XGBoost (0.753) but its Stage 2 Spearman
(0.207) is highest. This reflects LightGBM's leaf-wise growth strategy:

- **Stage 1**: Leaf-wise splits may overfit noisy regions of the feature space
  at low depth, reducing rank ordering of creation probabilities.
- **Stage 2**: On the smaller shot-quality subset (35,168 rows), leaf-wise
  growth can better capture the non-linear relationship between action features
  and shot quality, producing higher rank correlation.

This dissociation is why Stage 1 and Stage 2 are evaluated separately:
a pipeline that is mediocre at shot-creation ranking but strong at shot-quality
ranking is distinct from one that excels at both.

---

## 3. Stage 2 — Shot-Quality Interpretation

### 3.1 Weak Spearman is Expected

The best Spearman $\rho = 0.186$ (logistic pipeline) is modest. This is not a
model failure — it reflects the **structural information gap** between pre-shot
creative actions and shot outcomes.

Consider: two passes into the penalty box at identical locations (`receiver_in_box=True`,
`receiver_distance_to_goal=8m`) can precede:
- A tap-in with CxG = 0.80
- A difficult first-time shot under pressure with CxG = 0.11

The creative action features observe neither the shot geometry nor the shot
technique. The quality model estimates $\mathbb{E}[G_i \mid \mathbf{x}_i]$ as
a smooth function of action-level context — a population average, not a
shot-specific estimate. The Spearman $\rho \approx 0.19$ is what can be
extracted from the available signal.

### 3.2 MAE in Context

The MAE of 0.069 (best model) is in CxG units. The mean CxG of a shot in
the training data is approximately 0.08–0.10. An MAE of 0.069 is therefore
substantial — roughly 70–86% of the mean shot value. However, Stage 2 is not
predicting individual shot outcomes; it is estimating the **expected value** of
shots that follow from a given action context. This averaging role means that
the relevant benchmark is not how precisely it predicts any individual shot,
but whether it correctly stratifies action types by their expected downstream
shot quality.

---

## 4. CxA Score Distribution

Population averages from the Euro 2024 held-out set:

| Zone | Mean CxA |
|---|---|
| All actions | ~0.018 |
| $x \leq 60$ m (own half / midfield) | ~0.012 |
| $60 < x \leq 80$ m (attacking midfield) | ~0.024 |
| $x > 80$ m (final third) | ~0.029 |

The strong spatial gradient (final third CxA ≈ 2.4× midfield CxA) reflects
that actions in advanced positions are both more likely to create shots
(Stage 1) and that those shots tend to come from better positions (Stage 2).

**Representative high-CxA actions** (Euro 2024 held-out):

| Type | x | y | p_shot | E[CxG\|shot] | CxA | Realised CxG |
|---|---|---|---|---|---|---|
| Pass | 101.7 m | 18.4 m | 0.954 | 0.151 | **0.144** | 0.159 |
| Carry | 87.3 m | 23.5 m | 0.957 | 0.132 | **0.126** | 0.159 |

Both actions have $p_{\text{shot}} \approx 0.955$ — the Stage 1 model is
near-certain a shot will follow from inside or near the penalty box. The
difference in CxA (0.144 vs. 0.126) is entirely driven by Stage 2: the pass's
destination geometry (y = 18.4 m, central area) signals marginally higher
expected shot quality than the carry's end point (y = 23.5 m, slightly wider).

---

## 5. Comparison to CxG Leaderboard

The CxA training context differs from CxG in key ways:

| Dimension | CxG | CxA |
|---|---|---|
| Task | Binary classification (goal/no-goal) | Two-stage (creation + quality) |
| Training $N$ | 2,783 shots | 220,899 creative actions |
| Positive rate | 11.5% | 15.93% |
| Primary metric | Log-loss (calibration critical) | AUC (ranking critical) |
| Winning family | Logistic | Logistic |

In both CxG and CxA, logistic regression outperforms tree-based models.
The CxG rationale was primarily data scarcity (2,783 shots). For CxA, logistic
wins for a different reason: the contextual feature set provides sufficient
additive signal and the regularised logistic model generalises better to the
Euro 2024 distribution shift than the tree models, which appear to overfit
to WC 2022 + Euro 2020-specific patterns.

---

## 6. Limitations and Future Work

### 6.1 Stage 2 Information Gap

The fundamental limitation of CxA Stage 2 is that it must estimate shot quality
from creative-action features, not shot features. The most predictive shot
features (distance, angle, body part) are not known at the time of the creative
action. Future work could explore:

- **Conditional CxG lookup**: Rather than training a regressor, use the
  distribution of destination-location CxG values empirically from the training
  data.
- **Sequence-conditioned quality**: Use the full possession sequence leading to
  the shot (not just the immediately preceding action) to better predict shot
  quality.

### 6.2 Single-Feature-Tier Ladder

The current ladder trains only on the **CONTEXTUAL** feature tier. A full
ablation (TRADITIONAL vs. CONTEXTUAL) would quantify the contribution of
opponent quality, match state, and receiver context to CxA prediction,
analogous to the CxG tier ablation.

### 6.3 Window Sensitivity

The `shot_created` label uses same-possession linkage. Four alternative
window definitions are implemented in `shot_creation_model.py`
(`shot_within_5_actions`, `shot_within_10s`, `shot_within_15s`) but not
evaluated in the current training run. A stability analysis across windows
would confirm robustness of the label definition.

### 6.4 CxA Depends on CxG Version

Any upgrade to the CxG production model changes the `resulting_shot_cxg`
quality labels and therefore changes both Stage 2 training targets and
evaluation benchmarks. CxA version numbers should be tied to CxG version
numbers in the model registry.

### 6.5 No FULL_360 Tier

Freeze-frame features (`receiver_nearest_defender_distance`,
`open_passing_lanes`, etc.) are defined in `feature_sets.py` but not trained
because the training competitions lack uniform 360 coverage. When a 360-coverage
competition is added to the training set, the FULL_360 tier should be evaluated.
