# Quickstart

One page from clone to results. Requires Python 3.11-3.13 and
[Poetry](https://python-poetry.org/). A `Makefile` wraps the common commands
(run `make <target>`, or run the underlying command directly if you do not have
`make`).

## 1. Install

```bash
git clone https://github.com/varunrout/contextual-football-metrics
cd contextual-football-metrics
make install            # poetry install (core + models + viz + dev)
```

## 2. Zero-data demo (no download needed)

Every evaluation script has a `--smoke` mode that runs on synthetic data, so you
can see the honest-evaluation harness run and build the results store without any
external data or trained models:

```bash
make demo               # runs all analysis/2x --smoke evals, then builds results.db
```

That runs the CxG/CxA/CxT incremental-lift comparisons, the CxA composite
calibration, external validity and the downstream reranking, all on synthetic
data, then builds `results.db` from the committed JSON reports. Query it:

```python
from src import results_store
results_store.leaderboard("cxg")        # held-out CxG leaderboard, promoted model flagged
results_store.incremental_lift("cxt")   # CxT lift vs the zone baseline, with CIs
```

## 3. Full pipeline on real data

The parquet data and trained models are gitignored and DVC-tracked (see
[data.md](data.md)). Pull them, or rebuild from StatsBomb Open Data:

```bash
dvc pull                                 # if a DVC remote is configured
# or rebuild from source:
python scripts/ingest.py                 # raw StatsBomb JSON -> data/processed/*.parquet
python scripts/build_features.py         # processed -> data/features/*.parquet
```

Train the three metric ladders and score events end to end:

```bash
make train                               # train_cxg / train_cxa / train_cxt
make score                               # scripts/score.py -> outputs/scores/scored.parquet
```

Reproduce the honest headline numbers (each writes a report + figures under
`reports/`):

```bash
python analysis/20_incremental_lift_vs_baselines.py   # CxG vs StatsBomb xG + baseline
python analysis/21_incremental_lift_cxa.py            # CxA vs base rate
python analysis/22_incremental_lift_cxt.py            # CxT vs zone baseline
python analysis/24_external_validity.py              # CxG vs goals, CxA vs assists
python analysis/25_downstream_reranking.py           # CxT reranking vs static xT
```

## 4. Explore

```bash
make results-db                          # (re)build results.db from reports
streamlit run app.py                     # dashboard: player leaderboards + Model Evaluation tab
```

## 5. Where results land

| Output | Location |
|---|---|
| Trained models | `models/<metric>/*.joblib\|*.pkl` (gitignored) |
| Report JSON + figures | `reports/` (committed) |
| Results store | `results.db` (gitignored; `make results-db`) |
| Scored events | `outputs/scores/scored.parquet` |

## 6. Checks

```bash
make lint                                # ruff check + format check (matches CI)
make test                                # pytest
```
