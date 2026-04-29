# CxT — Training Methodology

## 1. Pipeline Overview

Training is orchestrated by `scripts/train_cxt.py` and follows a strict
sequential pipeline:

```
data/features/features.parquet (615,202 all-event rows)
    │
    ▼
[1] Attach CxG scores — load production CxG model, score shot rows inline
    │                   adds 'cxg' column to shot rows
    ▼
[2] Build possession_cxg target — discounted sum of cxg per possession
    │                              assigns same value to all actions in possession
    ▼
[3] Split train/held-out — exclude val_test (Euro 2024) rows
    │                       220,899 train | 98,029 held-out
    ▼
[4] Filter to CxT action types — pass, carry, cross, cutback only
    │
    ▼
[5] Derive pos_df — subset of train where possession_cxg > 0 (~16%)
    │
    ▼
[6] StateValueLadder.run() — 5-fold match-level CV for 5 candidates
    │
    ├─ gamma_glm        (GLM / Gamma,   trained on pos_df)
    ├─ tweedie_glm      (GLM / Tweedie, trained on actions_df)
    ├─ gam_contextual   (GAM / Gamma,   trained on pos_df)
    ├─ xgb_contextual   (XGBoost,       trained on actions_df)
    └─ lgbm_contextual  (LightGBM,      trained on actions_df)
    │
    ▼
[7] Rank by mean CV MAE → refit each candidate on full train set
    │
    ▼
[8] Held-out evaluation on Euro 2024 (98,029 actions)
    │
    ▼
[9] Save all 5 models as joblib → models/cxt/
    │
    ▼
[10] Generate 6 diagnostic figures + write cxt_training_summary.json
     │
     ▼
[11] Update configs/models.yaml production.cxt pointer
```

---

## 2. Inline CxG Scoring

The `features.parquet` file does not contain a pre-computed CxG column.
CxG scores are attached at training time:

```python
# scripts/train_cxt.py :: _attach_cxg_scores()
cxg_model = joblib.load(cfg["production"]["cxg"])  # loads glm_contextual.joblib
shot_mask = features_df["event_type"] == "shot"
proba = cxg_model.predict_proba(features_df[shot_mask])
features_df.loc[shot_mask, "cxg"] = proba[:, 1]   # goal probability
```

Non-shot rows receive `cxg = NaN`, which becomes 0 when computing
`possession_cxg`. This ensures:

1. The CxG model and CxT model are always trained on consistent signal.
2. If the CxG model is updated, the CxT target is automatically updated on
   the next CxT training run — no stale targets cached in parquet files.

The production CxG pointer is read from `configs/models.yaml`:

```yaml
production:
  cxg: models/cxg/glm_contextual.joblib
```

---

## 3. Discounted Possession CxG Target

After CxG scores are attached, the `possession_cxg` target is computed per
possession via `compute_possession_cxg()`:

$$\texttt{possession\_cxg}_i = \sum_{k=0}^{K_p} \gamma^k \cdot \text{CxG}_{i,k}$$

where:
- $p$ is the possession (identified by `match_internal_id` + `possession_internal_id`)
- $k$ is the 0-indexed shot index within possession $p$
- $\gamma = 0.9$ is the discount factor
- All actions in possession $p$ share the same `possession_cxg` value

The discount factor $\gamma = 0.9$ downweights shots that are further from the
current action in the possession sequence. This rewards actions that more
immediately generate threat: a carry into the box that is followed on the next
touch by a shot is credited more than a carry followed by a 5-pass buildup
before the shot.

For possessions with no shots, `possession_cxg = 0`. In the training data,
approximately 84% of creative actions occur in such possessions.

---

## 4. Train/Held-out Split

The data are split at the **competition level** by joining `data/processed/matches.parquet`
on `match_internal_id` and filtering on `split_role`:

| Role | Competition | Season | Creative actions |
|---|---|---|---|
| `train_val` | FIFA World Cup | 2022 | ~135,000 |
| `train` | UEFA Euro | 2020 | ~85,000 |
| **CV pool total** | — | — | **220,899** |
| `val_test` | UEFA Euro | 2024 | **98,029** |

Euro 2024 rows (`split_role == "val_test"`) are excluded from the training
ladder entirely. They are only used for held-out evaluation after all models
have been fitted.

**Rationale for competition-level holdout:**

1. Opponent quality features (rolling statistics over prior matches) would be
   contaminated by data from later in the same competition if splits were made
   at match level within a tournament.
2. All actions in a match share context features (score state, opponent,
   match minute trajectory). Row-level splitting would allow the model to see
   near-identical context vectors in both train and validation.
3. Holding out a complete tournament tests generalisation to an unseen cohort
   of teams — the operationally relevant scenario for applying these metrics
   to new competitions.

---

## 5. Match-Level K-Fold Cross-Validation

### 5.1 Algorithm

Cross-validation is performed at match granularity. Whole matches are assigned
to folds; no match appears in both train and validation within a fold.

Procedure (`src/evaluation/validation_splits.py :: match_kfold`):

1. Extract unique match IDs from the CV pool: $\mathcal{M}$ ($M = 115$ matches).
2. Shuffle $\mathcal{M}$ with `numpy.random.default_rng(seed=42)`.
3. Partition $\mathcal{M}$ into $K = 5$ approximately equal folds.
4. For fold $k$: validation = all actions from $\mathcal{F}_k$;
   training = all actions from $\mathcal{M} \setminus \mathcal{F}_k$.

Each fold contains approximately $115 / 5 = 23$ matches and
$220{,}899 / 5 \approx 44{,}000$ actions.

### 5.2 GLM Fold Handling (pos_df Candidates)

For `gamma_glm` and `gam_contextual`, the training split is additionally
filtered to `possession_cxg > 0` within each fold. Validation metrics are
computed on the **full validation fold** (including zero-valued rows), not just
the positive subset. This ensures the reported CV MAE reflects the model's
performance on the full evaluation population.

### 5.3 Fold Validity Checks

Before fitting within each fold:
- Training fold must contain $\geq 10$ rows and $\geq 1$ positive example.
- Folds failing this check are skipped; `n_cv_folds_used` in the leaderboard
  reports how many folds contributed.

All 5 models used all 5 folds in the current training run.

---

## 6. Model Fitting Procedure

### 6.1 Cross-Validation Phase (Metric Estimation)

For each candidate and each fold, a **fresh model instance** is constructed and
fitted on the fold's training split:

```python
m = factory()             # fresh instance — no state carried over
m.fit(tr_df, "possession_cxg")
p = m.predict(va_df)
mae = mean_absolute_error(va_df["possession_cxg"], p)
```

Preprocessing (imputer, scaler, encoder) is fitted independently on each
training fold. This prevents imputation statistics or scaling parameters from
the validation fold contaminating the training fold fit.

### 6.2 Final Fitting Phase (Production Model)

After CV metric estimation, each candidate is **refit on the full CV pool**
(all 220,899 training actions, or `pos_df` for GLM candidates). This is the
model serialised to disk and used for downstream scoring and CxT computation.

Refitting on the full training set is standard practice: the CV metrics provide
an unbiased estimate of generalisation performance, and the final model benefits
from seeing all available training data.

---

## 7. Reproducibility

All random operations use `random_state=42`:

- `numpy.random.default_rng(42)` for fold shuffling
- `random_state=42` passed to all model constructors
- `torch.manual_seed(42)` for the FFNN (if used)

The pipeline is fully deterministic given the input data and seed.

---

## 8. Serialisation

All five fitted models are serialised using `joblib.dump` into `models/cxt/`:

```
models/cxt/
    gamma_glm.joblib
    tweedie_glm.joblib
    gam_contextual.joblib
    xgb_contextual.joblib
    lgbm_contextual.joblib     ← production model
```

`joblib` is preferred over `pickle` for sklearn pipelines because it
memory-maps numpy arrays, providing more efficient I/O for the transformer
coefficient arrays.

The production pointer in `configs/models.yaml` is updated automatically
after each training run to point to the best model by CV MAE:

```yaml
production:
  cxt: models/cxt/lgbm_contextual.joblib
```

---

## 9. MLflow Tracking

Each training run is logged to MLflow under experiment `cfm/cxt`:

- Run-level parameters: `n_folds`, `n_estimators`, `seed`
- Per-candidate metrics: `{name}_cv_mae`, `{name}_cv_spearman`, `{name}_heldout_mae`
- Artifact: `cxt_training_summary.json`

Tracking URI: `mlruns/` (local file store).
