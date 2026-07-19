# CxA - Incremental Lift on the Matched Euro 2024 Held-Out Set

_Added 2026-07-19. Companion to the CxG reconciliation in
`docs/modeling/cxg/07`. Produced by `analysis/21_incremental_lift_cxa.py` on the
same Euro 2024 held-out split (competition 55, season 282)._

## 1. What is being tested

CxA is a two-stage metric: a shot-creation model (does this pass or carry lead to
a shot in the same possession?) and a shot-quality model (how good is the
resulting shot?), combined as `p_shot x E[CxG]`. The honest question for the
creation stage is whether the contextual model has real discriminative signal, so
it is scored against a base-rate null (predict the overall creation rate for every
action) on the identical Euro 2024 held-out actions, with a 2,000-sample paired
bootstrap on the deltas.

There is no traditional-feature CxA model in the repo, so this is a
model-vs-null test, not a contextual-vs-traditional test. That limit is stated
rather than hidden: beating a base rate shows the model has signal, not that the
contextual features specifically are what earns it.

## 2. Result (98,029 held-out actions, 19,220 created, 19.6% base rate)

| Predictor | Log-loss | Brier | AUC | PR-AUC | ECE |
|---|---|---|---|---|---|
| naive base rate | 0.4949 | 0.1576 | 0.500 | 0.196 | 0.000 |
| contextual creation | 0.4880 | 0.1510 | **0.729** | **0.358** | 0.081 |

Paired deltas (contextual minus base rate), 95% bootstrap CI:

| Metric | Delta | 95% CI |
|---|---|---|
| Log-loss (lower better) | -0.0069 | [-0.0103, -0.0036] |
| AUC (higher better) | +0.2295 | [+0.2256, +0.2332] |

**Verdict: the contextual shot-creation model adds value over a base-rate null.**
Both deltas' CIs exclude zero, and the AUC of 0.729 is a clear, real signal.

## 3. The honest caveats

- **Weak baseline.** The comparison is against a base rate, not a
  traditional-feature model. The correct next step to substantiate the
  "contextual" claim is to train a traditional-feature creation model and repeat
  this comparison.
- **Miscalibration.** The creation model's ECE is 0.081, an order of magnitude
  worse than the CxG models. The probabilities discriminate well but are not
  well-calibrated, so the composite `p_shot x E[CxG]` inherits that error.
- **Quality stage is near-noise.** In training, the shot-quality regressor lands
  at held-out Spearman ~0.19-0.22. The creation stage is the load-bearing part of
  CxA; the quality stage adds little. The composite's usable resolution is
  therefore roughly "which actions create shots", not "how good the shot will be".

The composite reliability check for CxA is covered separately (CONT-F08).
