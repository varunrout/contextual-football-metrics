# Contextual Football Metrics (CFM)

A suite of **contextual** football event metrics built on StatsBomb Open Data:

| Metric | Meaning | Unit of analysis |
|---|---|---|
| **CxG** | Contextual expected goals — goal probability of a shot | shot |
| **CxA** | Contextual expected assists — probability an action creates a shot, weighted by the resulting shot's quality | pass / carry / cutback |
| **CxT** | Contextual threat — expected value of a game state / possession | action within a possession |

Each metric is trained as a **ladder** of candidate models (baseline → GLM →
GBM → optional neural) evaluated on a held-out competition split, with the
best candidate promoted to production. The whole pipeline runs as a single
Prefect flow logging nested runs to MLflow, and results are explorable through
a Streamlit dashboard.

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
# torch-geometric is not a declared dependency — install separately if a
# GNN model needs it:
pip install torch-geometric torch-scatter torch-sparse

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

## Neural / graph model variants

Beyond the tree/GLM ladder, each metric has an optional neural candidate that
consumes 360 freeze-frame data:

| Script | Model | Requires |
|---|---|---|
| `scripts/train_cxg.py --include-neural` | SetTransformer over freeze-frame tokens | `--frames data/processed/frames.parquet` |
| `scripts/train_cxa.py --include-neural` | GNN passing network (creation stage) | `--frames <path>` |
| `scripts/train_cxt.py --include-neural` | FFNN / SetTransformer / GNN state-value | `--frames <path>` (SetTransformer/GNN only) |
| `scripts/score_set_transformer.py` | Score shots with a trained SetTransformer CxG model | trained `.joblib` |
| `scripts/score_gnn_cxa.py` | Score actions with the GNN passing-network creation model | trained creation + quality models |

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
