# CxG — Training Methodology

## 1. Pipeline Overview

Training is orchestrated by `scripts/train_cxg.py` and follows a strict
sequential pipeline:

```
shots.parquet (4,123 shots)
    │
    ▼
[1] Split filter — exclude val_test (Euro 2024)
    │
    ▼
[2] CxGLadder.run() — 5-fold match-level CV for each candidate
    │
    ├─ baseline_logit   (Traditional / Logistic)
    ├─ glm_contextual   (Contextual / Logistic)
    ├─ xgb_traditional  (Traditional / XGBoost)
    ├─ xgb_contextual   (Contextual / XGBoost)
    ├─ lgbm_traditional (Traditional / LightGBM)
    └─ lgbm_contextual  (Contextual / LightGBM)
    │
    ▼
[3] Rank by mean CV log-loss
    │
    ▼
[4] Refit each candidate on full CV pool (2,783 shots)
    │
    ▼
[5] Held-out evaluation on Euro 2024 (1,340 shots)
    │
    ▼
[6] Save all 6 models as joblib → models/cxg/
    │
    ▼
[7] Generate report + charts
```

---

## 2. Train/Test Split Design

### 2.1 Competition-Level Partitioning

The data are partitioned at the **competition level**, not randomly. This
preserves temporal and distributional structure:

| Role | Competition | Season | Shots | Purpose |
|---|---|---|---|---|
| `train_val` | FIFA World Cup | 2022 | ~1,440 | Contributes to CV folds |
| `train` | UEFA Euro | 2020 | ~1,343 | Contributes to CV folds |
| `val_test` | UEFA Euro | 2024 | 1,340 | Held out — never used in CV |

The CV pool contains **2,783 shots** from WC 2022 and Euro 2020.
Euro 2024 is excluded by joining `matches.parquet` on `match_internal_id`
and filtering `split_role != "val_test"` before calling the ladder.

### 2.2 Rationale for Competition-Level Holdout

Random row-level splitting would create data leakage through:

1. **Opponent quality features** — rolling statistics for a team computed from
   future matches in the same competition would contaminate training.
2. **Match context correlation** — shots from the same match share score state,
   match minute trajectory, and opponent identity. Splitting within a match
   allows the model to see the validation context during training.
3. **Distribution shift testing** — holding out an entire tournament tests
   whether the model generalises to a new cohort of teams and playing styles,
   which is the operationally relevant question.

---

## 3. Match-Level K-Fold Cross-Validation

### 3.1 Algorithm

Cross-validation is performed at match granularity. Whole matches are assigned
to folds; no match appears in both train and validation within a fold.

**Procedure** (`src/evaluation/validation_splits.py :: match_kfold`):

1. Extract the unique set of match IDs from the CV pool: $\mathcal{M} = \{m_1, \ldots, m_M\}$ where $M = 115$ matches.
2. Shuffle $\mathcal{M}$ with `numpy.random.default_rng(seed=42)`.
3. Partition $\mathcal{M}$ into $K = 5$ approximately equal folds:
   $\mathcal{M} = \mathcal{F}_1 \cup \mathcal{F}_2 \cup \cdots \cup \mathcal{F}_5$, $\mathcal{F}_i \cap \mathcal{F}_j = \emptyset$.
4. For fold $k$: validation set = all shots from matches in $\mathcal{F}_k$;
   training set = all shots from matches in $\mathcal{M} \setminus \mathcal{F}_k$.

Each fold contains approximately $115 / 5 = 23$ matches and
$2{,}783 / 5 \approx 557$ shots.

### 3.2 Fold Validity Checks

Before fitting in each fold, two conditions are enforced:

- Training fold must contain $\geq 10$ shots and $\geq 1$ positive example.
- Validation fold must contain $\geq 1$ positive example.

Folds that fail either check are skipped and their metrics are excluded from
the mean. The `n_cv_folds_used` field in the leaderboard reports how many folds
contributed. All models used all 5 folds.

### 3.3 Why Not Stratified K-Fold?

Stratified k-fold ensures each fold has the same positive class proportion.
However, it operates at row level and cannot guarantee match integrity.
At $K=5$ with 2,783 shots and ~11.5% positive rate, each fold already contains
$\approx 64$ goals on average — sufficient for stable AUC and Brier estimates.
The variance reduction from stratification is outweighed by the risk of
within-match leakage.

---

## 4. Model Fitting Procedure

### 4.1 Cross-Validation Phase (Metric Estimation)

For each candidate model and each of the 5 folds, a **fresh model instance**
is constructed via the factory function and fitted on the fold's training split.
This ensures:
- No information from the validation fold reaches the fitted model.
- The preprocessing pipeline (imputer, scaler, encoder) is fitted independently
  on each training fold — no contamination of imputation statistics or scaling
  parameters.

```python
model = model_factory()       # fresh instance
model.fit(tr_df, target_col)  # fit on fold train split only
p = model.predict_proba(va_df)
```

The fold-level metrics (log-loss, Brier, AUC) are computed on the validation
split and averaged across folds.

### 4.2 Final Fitting Phase (Production Model)

After CV metric estimation, each candidate is **refit on the full CV pool**
(all 2,783 shots). This is the model that is serialised to disk and used for
downstream scoring.

Refitting on the full training set is standard practice: the CV metrics provide
an unbiased estimate of generalisation performance, and the final model benefits
from seeing all available training data.

### 4.3 Regularisation in Context

For logistic regression, the regularisation strength is fixed at `C = 1.0`
(the sklearn default, equivalent to $\lambda = 1$) in this training run.
The `GlmContextualCxG` class supports a `tune=True` flag which performs a
grid search over `C ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0}` via stratified CV
before the final fit. This was not activated here due to the small dataset
size — with ~320 positives the risk of the grid search over-selecting a lucky C
is non-trivial and a modest ridge penalty of C=1 is well-motivated.

---

## 5. Reproducibility

All random operations use `random_state=42`:
- `numpy.random.default_rng(42)` for fold shuffling
- `random_state=42` passed to `LogisticRegression`, `XGBClassifier`,
  and `LGBMClassifier`
- `numpy` global seed is not modified — only `default_rng` is used to avoid
  polluting the global state

The full pipeline is deterministic given the input data and seed.

---

## 6. Serialisation

All six fitted models are serialised using `joblib.dump` into `models/cxg/`:

```
models/cxg/
    baseline_logit.joblib
    glm_contextual.joblib       ← production model
    xgb_contextual.joblib
    xgb_traditional.joblib
    lgbm_contextual.joblib
    lgbm_traditional.joblib
```

`joblib` is preferred over `pickle` for sklearn pipelines because it uses
memory-mapped numpy arrays, which is more efficient for the large coefficient
arrays in the contextual GLM pipeline.

The production pointer in `configs/models.yaml` is updated automatically
after each training run to point to the best model by CV log-loss:

```yaml
production:
  cxg: models/cxg/glm_contextual.joblib
```
