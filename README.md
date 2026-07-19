# Contextual Football Metrics (CFM)

[![CI](https://github.com/varunrout/contextual-football-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/varunrout/contextual-football-metrics/actions/workflows/ci.yml)

A suite of **contextual** football event metrics built on StatsBomb Open Data:

| Metric | Meaning | Unit of analysis |
|---|---|---|
| **CxG** | Contextual expected goals — goal probability of a shot | shot |
| **CxA** | Contextual expected assists — probability an action creates a shot, weighted by the resulting shot's quality | pass / carry / cutback |
| **CxT** | Contextual threat — expected value of a game state / possession | action within a possession |

Each metric is trained as a **ladder** of candidate models (baseline → GLM →
GBM → optional neural) evaluated on a held-out competition split. The
production model is chosen by a single pre-committed criterion (cross-validation
log-loss), and the held-out set is used once as confirmation, not to pick the
winner. The whole pipeline runs as a single Prefect flow logging nested runs to
MLflow, and results are explorable through a Streamlit dashboard.

## Headline result (read this first)

This project builds three context-aware football metrics (contextual xG, xA and
threat) and, more to the point, evaluates them honestly. The CxG result on a
matched Euro 2024 held-out set (1,340 shots, 126 goals, 2,000-sample paired
bootstrap with 95% confidence intervals):

- Contextual features improve on our own traditional-feature baseline: held-out
  log-loss 0.241 vs 0.251, a delta of −0.0096 whose CI [−0.017, −0.002] excludes
  zero. This is the defensible part of the "context helps" claim.
- They do **not** beat off-the-shelf StatsBomb xG. On the same rows StatsBomb xG
  is ahead on every metric (log-loss 0.233, AUC 0.830, ECE 0.013 vs the
  contextual model's 0.241 / 0.813 / 0.017), and both contextual-minus-StatsBomb
  deltas have confidence intervals spanning zero. The honest verdict is no
  demonstrable lift over off-the-shelf xG.
- The production CxG model is therefore `baseline_logit`, the model ranked first
  on cross-validation log-loss. The contextual GLM is a real but small
  improvement on our traditional baseline; it is not promoted because it is not
  selected by the pre-committed criterion and does not beat off-the-shelf xG.

The same matched, CI-bounded evaluation is run for the other two metrics on the
identical Euro 2024 held-out split (98,029 actions, 2,000-sample bootstrap):

- **CxA** ([`analysis/21`](analysis/21_incremental_lift_cxa.py),
  [writeup](docs/modeling/cxa/07_incremental_lift.md)): the contextual
  shot-creation model beats a base-rate null decisively (AUC 0.729, log-loss
  −0.0069, both CIs exclude zero), but it is miscalibrated (ECE 0.081) and has no
  traditional-feature baseline yet, so this shows signal, not that context
  specifically earns it.
- **CxT** ([`analysis/22`](analysis/22_incremental_lift_cxt.py),
  [writeup](docs/modeling/cxt/07_incremental_lift.md)): the contextual
  state-value model beats a static per-zone baseline on the same target (MAE
  0.0304 vs 0.0324, Spearman 0.303 vs 0.180, CIs exclude zero). This is the
  clearest win for context of the three.

The point of a contextual metric is the incremental lift over an already-strong
baseline, measured on the same data with an error bar, not a fresh AUC. Each
comparison is produced by its `analysis/2x` script and written up in the matching
`docs/modeling/<metric>/07` note.

```bash
python analysis/20_incremental_lift_vs_baselines.py          # real data (dvc pull first)
python analysis/20_incremental_lift_vs_baselines.py --smoke  # synthetic, no data needed
```

## Repository layout

```
.
├── app.py                  # Streamlit dashboard for scored events
├── analysis/                # Pre-modelling EDA / data-quality scripts (registered stages)
├── configs/
│   └── profiles/            # Runtime profiles: cpu.yaml / gpu.yaml / cloud.yaml
├── data/                    # raw / processed / features parquet tables (gitignored, DVC-tracked)
├── docs/
│   ├── pipeline.md           # Prefect + MLflow pipeline overview
│   ├── cloud_bootstrap.md    # Colab + DagsHub + DVC setup guide
│   ├── analysis_reports/     # EDA write-ups (data quality, CxG/CxA/CxT findings)
│   └── modeling/{cxg,cxa,cxt}/  # Per-metric problem formulation, features, model specs, results
├── models/                  # Saved model artifacts + configs/models.yaml production pointers
├── pipelines/               # Prefect flow, stage registries, quality gates, MLflow helpers
├── reports/                 # Training summaries (JSON) + generated figures
├── scripts/                 # CLI entry points (ingest, build_features, train_*, score*, monitor)
├── src/
│   ├── ingestion/            # StatsBomb download + event/possession mapping
│   ├── features/             # Feature store: traditional, sequence, freeze-frame features
│   ├── models/
│   │   ├── cxg/               # Baseline, GLM, LightGBM, XGBoost, SetTransformer, ladder
│   │   ├── cxa/               # Two-stage shot-creation + shot-quality models, GNN passing network
│   │   ├── cxt/               # Zone baseline, state-value model, GNN / SetTransformer variants
│   │   ├── neural/            # Shared torch primitives: encoders, freeze-frame loader, base mixin
│   │   ├── sequence/          # Possession sequence classifiers
│   │   ├── statistical/       # Statistical priors (e.g. zone-xT)
│   │   └── trees/             # Shared tree-model utilities
│   ├── evaluation/            # Interpretability + model comparison
│   ├── interpretation/        # SHAP / feature-importance helpers
│   ├── monitoring/            # Drift detection (PSI)
│   ├── pipeline/              # Inference pipeline (combines cxg/cxa/cxt for scoring)
│   ├── runtime/                # Profile loading, GBM device selection
│   └── dashboards/             # Shared plotting helpers for app.py
└── tests/                   # Mirrors src/ layout; one test module per phase/feature
```

## Installation

Requires Python ≥3.11, <3.14 and [Poetry](https://python-poetry.org/).

```bash
# Core + statistical/tree models (default groups)
poetry install

# Add neural models (torch, captum) — CPU wheel by default
poetry install --with neural
# For CUDA: install a matching torch wheel afterwards, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Note: the GNN models implement graph attention by hand on top of plain
# torch (src/models/neural/encoders.py) — torch-geometric/torch-scatter/
# torch-sparse are not used and do not need to be installed.

# Add visualization/explainability (matplotlib, shap, plotly, …)
poetry install --with viz

# Add dev tooling (pytest, ruff, pre-commit)
poetry install --with dev

# Cloud-only extras (Colab/Kaggle): DagsHub + DVC remote tracking
poetry install --with cloud
```

## Quickstart

```bash
# 1. Ingest StatsBomb Open Data → data/processed/{events,matches,possessions}.parquet
python scripts/ingest.py

# 2. Build the feature store → data/features/{features,shots,actions}.parquet
python scripts/build_features.py

# 3. Train each metric's model ladder
python scripts/train_cxg.py
python scripts/train_cxa.py
python scripts/train_cxt.py

# 4. Score events with the production models → outputs/scores/*.parquet
python scripts/score.py --events data/features/features.parquet

# 5. Explore results
streamlit run app.py
```

Or run the whole thing as one Prefect flow (recommended — wraps every stage
under a single MLflow parent run):

```bash
python -m pipelines.flow                   # auto-detects cpu/gpu/cloud
python -m pipelines.flow --profile gpu     # force a runtime profile
python -m pipelines.flow --only-group train
python -m pipelines.flow --only train_cxg
```

See [docs/pipeline.md](docs/pipeline.md) for stage groups, quality gates, and
MLflow run structure, and [docs/cloud_bootstrap.md](docs/cloud_bootstrap.md)
for running on Colab with DagsHub + DVC.

## Neural / graph model variants (exploratory negatives)

Each metric has an optional neural candidate that consumes 360 freeze-frame
data. They are built and runnable, but on this dataset (a few thousand shots,
~220k actions from three tournaments) they **underperform the simple
logistic/tree models and none is promoted to production.** Measured on the
Euro 2024 held-out split:

| Metric | Neural model | Held-out | Simple production model | Verdict |
|---|---|---|---|---|
| CxG | SetTransformer over freeze-frames | log-loss 0.273, AUC 0.778 (rank 5 of 7) | `baseline_logit` 0.250 / 0.799 | loses |
| CxA | GNN passing network (creation) | creation AUC 0.649, log-loss 0.718 (last) | `logistic` 0.766 / 0.427 | loses badly |
| CxT | FFNN / SetTransformer / GNN state-value | exploratory, not benchmarked | `lgbm_contextual` (production) | see note |

The honest reading: contextual **features** help (see the headline result), but
contextual **neural architectures** do not earn their complexity at this data
scale. The production model for every metric is a logistic or tree model; the
neural variants are kept as exploratory negative results, not selling points.

**CxT note:** the CxT neural state-value variants train over ~220k actions and
take hours on CPU (the CxA GNN alone took ~1.5h), so they are left buildable but
not benchmarked. The evaluated CxG and CxA neural results above are
representative of the pattern.

The scripts to (re)train and score the neural candidates:

| Script | Model | Requires |
|---|---|---|
| `scripts/train_cxg.py --include-neural` | SetTransformer over freeze-frame tokens | `--frames data/processed/frames.parquet` |
| `scripts/train_cxa.py --include-neural` | GNN passing network (creation stage) | `--frames <path>` |
| `scripts/train_cxt.py --include-neural` | FFNN / SetTransformer / GNN state-value | `--frames <path>` (SetTransformer/GNN only) |
| `scripts/score_set_transformer.py` | Score shots with a trained SetTransformer CxG model | trained `.joblib` |
| `scripts/score_gnn_cxa.py` | Score actions with the GNN passing-network creation model | trained creation + quality models |
| `scripts/score_state_value_gnn.py --model-class {gnn,set_transformer}` | Score actions with a trained CxT GNN or SetTransformer state-value model | trained `.joblib` |

Neural model classes (`SetTransformerCxGModel`, `GNNPassingNetworkCxAModel`,
`GNNStateValueModel`, `SetTransformerStateValueModel`) are **not** re-exported
from their package's `__init__.py` (`src/models/cxg/__init__.py` etc.) — this
is deliberate, so importing `src.models.cxg`/`cxa`/`cxt` never requires torch
to be installed. Import them directly from their module instead, e.g.:

```python
from src.models.cxg.set_transformer_model import SetTransformerCxGModel
from src.models.cxa.gnn_passing_network import GNNPassingNetworkCxAModel
from src.models.cxt.state_value_gnn import GNNStateValueModel
from src.models.cxt.state_value_set_transformer import SetTransformerStateValueModel
```

## Runtime profiles

`src.runtime.load_profile` resolves `configs/profiles/{cpu,gpu,cloud}.yaml`
(explicit `--profile` → `CFM_PROFILE` env → autodetect) to configure the
accelerator, GBM device, per-model-family batch sizes, paths, and MLflow
tracking URI. See [docs/pipeline.md#profiles](docs/pipeline.md).

## Testing

```bash
poetry run pytest
```

Tests mirror `src/`'s layout under `tests/` (e.g. `tests/models/`,
`tests/features/`, `tests/pipeline/`, `tests/runtime/`).

## Further reading

- [docs/pipeline.md](docs/pipeline.md) — Prefect/MLflow pipeline internals
- [docs/cloud_bootstrap.md](docs/cloud_bootstrap.md) — Colab + DagsHub + DVC
- [docs/analysis_reports/](docs/analysis_reports/) — EDA and headline findings
- [docs/modeling/cxg](docs/modeling/cxg), [cxa](docs/modeling/cxa), [cxt](docs/modeling/cxt) — per-metric problem formulation, feature architecture, model specs, and results
