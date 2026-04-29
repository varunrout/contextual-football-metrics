# CxG — Problem Formulation

## 1. Background

Expected Goals (xG) is the probability that a given shot results in a goal,
conditioned on observable features of that shot. The canonical formulation
(introduced independently by Caley, 2014 and Rathke, 2017) conditions primarily
on shot geometry — distance and angle to goal — and technique — body part,
open play, set piece — drawn directly from event data.

The central limitation of traditional xG is its **context-blindness**: it treats
every shot against every opponent in every match situation identically, provided
the geometry and technique are the same. A 15-metre open-play shot against
a high-defensive-block Spanish side in a Champions League knockout is scored
identically to the same shot against a low-block side conceding three times per
match. This conflation reduces the signal-to-noise ratio when xG is used as an
input to downstream models such as action value (CxA) and possession threat (CxT).

**Contextual Expected Goals (CxG)** extends the probability estimate with three
additional adjustment layers: opponent quality, match state, and possession
build-up context.

---

## 2. Formal Problem Statement

Let $i = 1, \ldots, N$ index shot events and let $\mathbf{x}_i \in \mathbb{R}^p$
denote the observed feature vector for shot $i$. Let $Y_i \in \{0, 1\}$ denote
the binary outcome (1 = goal).

The modelling objective is to estimate the conditional probability

$$\hat{p}_i = \hat{P}(Y_i = 1 \mid \mathbf{x}_i)$$

such that $\hat{p}_i$ is **calibrated** — i.e. among all shots with predicted
probability $q$, approximately a fraction $q$ result in goals — and
**discriminative** — i.e. the model ranks shot quality correctly.

### 2.1 Response Distribution

$$Y_i \mid \mathbf{x}_i \sim \text{Bernoulli}(p_i)$$

The joint likelihood over $N$ independent shots is:

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} p_i^{y_i} (1 - p_i)^{1 - y_i}$$

and the log-likelihood is the standard binary cross-entropy:

$$\ell(\boldsymbol{\theta}) = \sum_{i=1}^{N} \left[ y_i \log p_i + (1 - y_i) \log(1-p_i) \right]$$

---

## 3. Contextual Conditioning

Traditional xG conditions on the **shot geometry and technique** alone:

$$p_i^{(\text{trad})} = P(Y_i = 1 \mid d_i, \theta_i, \mathbf{t}_i)$$

where $d_i$ is distance to goal, $\theta_i$ is shot angle, and $\mathbf{t}_i$
is a technique vector (body part, pressure, etc.).

CxG additionally conditions on three context groups:

$$p_i^{(\text{cxg})} = P\!\left(Y_i = 1 \;\middle|\; d_i, \theta_i, \mathbf{t}_i,\;
\underbrace{\mathbf{o}_i}_{\text{opponent}},\;
\underbrace{\mathbf{m}_i}_{\text{match state}},\;
\underbrace{\mathbf{s}_i}_{\text{sequence}}\right)$$

The three additional groups are:

| Group | Intuition |
|---|---|
| $\mathbf{o}_i$ — Opponent quality | The same shot is objectively harder against a top-10 keeper |
| $\mathbf{m}_i$ — Match state | Score, minute, tournament stage shift player and goalkeeper behaviour |
| $\mathbf{s}_i$ — Sequence context | A shot from a 20-pass build-up differs structurally from a transition counter |

---

## 4. Key Modelling Constraints

Three constraints shaped the design:

**C1 — No target leakage.** Features must be computable strictly from
pre-shot information. `shot_outcome` and any shot-derived flag must not appear
in $\mathbf{x}_i$.

**C2 — Temporal integrity.** Opponent rolling features (e.g.
`opponent_xg_conceded_rolling_5`) must be computed over matches strictly prior
to match $k$. No future leakage.

**C3 — Match-level split integrity.** Cross-validation and train/test splits
must keep all shots from a single match together. Match context features
(score state, minute, opponent) are correlated within a match; row-level
splitting would inflate CV scores by allowing the model to see near-identical
context vectors in both train and validation.

---

## 5. Dataset

| Split | Competition | Season | Matches | Shots | Goals | Goal Rate |
|---|---|---|---|---|---|---|
| Train/Val (CV) | FIFA World Cup | 2022 | 64 | — | — | — |
| Train (CV) | UEFA Euro | 2020 | 51 | — | — | — |
| **CV pool total** | — | — | **115** | **2,783** | **~321** | **~11.5%** |
| **Held-out test** | UEFA Euro | 2024 | 51 | **1,340** | ~154 | ~11.5% |

All three competitions have StatsBomb 360 data, providing freeze-frame
spatial context. The 360 feature set was not activated in this training run
(`include_360=False`); this is reserved for a subsequent experiment.

The held-out test set (Euro 2024) was **never seen during any training fold**
and was not used for model selection.
