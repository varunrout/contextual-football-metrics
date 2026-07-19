# CxG - Incremental Lift and Reconciliation of Conflicting Evaluations

_Added 2026-07-17. This document resolves a contradiction between three
evaluations already committed to the repo, and defines the single honest
comparison that decides whether contextual features earn their place._

## 1. The problem: three evaluations, three different answers

Three CxG evaluations are committed, and they do not agree because none of them
score the same models on the same rows with the same metric definitions.

| Source | Scored on | `baseline_logit` | `glm_contextual` | StatsBomb xG |
|---|---|---|---|---|
| `reports/statsbomb_baseline_metrics.json` | ~4,123 shots, 5-fold CV | - | - | **AUC 0.839, log-loss 0.264, ECE 0.013** |
| `reports/cxg_training_summary.json` | Euro 2024 held-out (1,340 shots) | AUC 0.799, log-loss 0.250 | **AUC 0.813, log-loss 0.241** | - |
| `reports/model_comparison_cxg.json` | cross-validation | **AUC 0.788, log-loss 0.321** | AUC 0.763, log-loss 0.333 | - |

Read across the rows and the story falls apart:

- The **held-out leaderboard** says `glm_contextual` is the best model, beating the traditional baseline on all three held-out metrics.
- The **model-comparison suite** says the opposite: on cross-validation, `baseline_logit` beats `glm_contextual` on log-loss, AUC and ECE.
- The **StatsBomb published xG**, an off-the-shelf model that uses none of this project's contextual features, posts AUC 0.839 and ECE 0.013, better on discrimination and calibration than either internal model, but on a third, non-comparable sample.

So the repo simultaneously claims contextual features win (leaderboard) and lose (comparison suite), and never puts either internal model head to head with the one benchmark that matters, StatsBomb's own xG, on the same data.

## 2. Why these numbers cannot be compared

The three tables measure different things:

1. **Different rows.** The StatsBomb baseline is scored on ~4,123 shots via 5-fold CV; the leaderboard on 1,340 Euro 2024 held-out shots; the comparison suite on CV folds. AUC and log-loss are not transferable across samples of different size, difficulty and goal rate.
2. **CV vs held-out.** Cross-validated scores and single held-out scores answer different questions and, on ~2,800 training shots with ~11.5% goals, differ by more than the gap between the models. The leaderboard/comparison disagreement is largely this effect.
3. **Two ECE definitions.** The StatsBomb baseline script uses 15 calibration bins; the comparison suite uses 10. ECE is bin-count sensitive, so 0.013 and 0.022 are not directly comparable.

None of this means the work is wrong. It means the **conclusion** ("contextual features add value") is unsupported, because it was never measured on a level field.

## 3. The corrected evaluation

`analysis/20_incremental_lift_vs_baselines.py` produces the one comparison that
settles it:

- **Identical rows.** StatsBomb xG, `baseline_logit` and `glm_contextual` are all scored on the *same* Euro 2024 held-out shots (competition 55, season 282), the final external set never seen in training.
- **One metric definition.** A single ECE implementation (10 bins) is applied to all three predictors, so calibration numbers are comparable.
- **Paired bootstrap.** 2,000 resamples of the held-out rows; every predictor is scored on the same resample, so each "contextual minus baseline" delta on log-loss and AUC carries a 95% confidence interval. A delta whose interval crosses zero is not a real improvement.
- **A strict verdict.** Contextual features are credited as adding value only if `glm_contextual` beats the *stronger* baseline (StatsBomb xG) on both log-loss and AUC with intervals that exclude zero. Beating only our own weaker traditional logit is not enough to make the "contextual xG" claim.

Outputs: `reports/incremental_lift_cxg.json`, a calibration overlay
(`reports/figures/incremental_lift/cxg_calibration_overlay.png`) and a delta
forest plot with confidence intervals
(`reports/figures/incremental_lift/cxg_delta_forest.png`).

Run it with the data pulled:

```bash
dvc pull            # if not already present
python analysis/20_incremental_lift_vs_baselines.py
```

## 4. The matched-run result

The matched comparison in section 3 has now been run on the Euro 2024 held-out
set (1,340 shots, 126 goals, 2,000 paired-bootstrap resamples). All three
predictors are scored on the identical rows with one ECE definition.

| Predictor | Log-loss | Brier | AUC | ECE |
|---|---|---|---|---|
| StatsBomb xG (off-the-shelf) | **0.2333** | **0.0645** | **0.8295** | **0.0126** |
| `glm_contextual` | 0.2413 | 0.0667 | 0.8131 | 0.0171 |
| `baseline_logit` (traditional) | 0.2507 | 0.0695 | 0.7982 | 0.0204 |

Paired deltas for the contextual model, with 95 percent bootstrap CIs:

| Comparison | Delta log-loss (lower better) | Delta AUC (higher better) |
|---|---|---|
| `glm_contextual` − `baseline_logit` | −0.0096, CI [−0.0174, −0.0022] | +0.0152, CI [−0.0026, +0.0333] |
| `glm_contextual` − StatsBomb xG | +0.0080, CI [−0.0039, +0.0197] | −0.0164, CI [−0.0431, +0.0094] |

**Verdict: no demonstrable lift over off-the-shelf xG.** Read honestly:

- Against **our own traditional baseline**, the contextual GLM has a small but
  statistically clear log-loss improvement (CI excludes zero). Its AUC edge is
  not established (CI includes zero). This is the defensible part of the
  "context helps" claim.
- Against **StatsBomb's published xG**, contextual features do **not** win.
  Both deltas' CIs span zero and StatsBomb xG is ahead on every point estimate,
  including calibration (ECE 0.013 vs 0.017). On a level field this project does
  not beat off-the-shelf xG.

The value of a contextual metric is the *incremental* lift over an already-strong
baseline, not a fresh AUC. Measured that way, the honest headline is: contextual
features improve on a traditional baseline but do not yet beat StatsBomb xG.

## 5. How to talk about this in an interview

The strong version of this project is not "I built a contextual xG that beats
StatsBomb". It is: "I built a rigorous, matched evaluation, found my three
existing reports disagreed because they were measured on different samples, and
corrected it. On a level field my contextual features improve on a traditional
baseline; whether they beat an off-the-shelf xG I report honestly with a
bootstrap confidence interval rather than cherry-picking the evaluation that
flatters the model." That answer demonstrates exactly the judgement the metric
itself is supposed to show.
