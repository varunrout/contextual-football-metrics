# Pipeline overview

The repo runs as a single Prefect flow that wraps every analysis script
and training step under one MLflow parent run. Pre-modelling analysis,
quality gates, training, and post-modelling analysis all log to the same
hierarchy, and the active runtime profile (cpu / gpu / cloud) decides
where artifacts live and which device GBM models target.

## TL;DR

```bash
# Fastest way to validate the wiring (auto-detects hardware)
python -m pipelines.flow

# Force a profile
python -m pipelines.flow --profile cpu     # local laptop
python -m pipelines.flow --profile gpu     # local CUDA box (e.g., GTX 1050)
python -m pipelines.flow --profile cloud   # Colab/Kaggle + DagsHub MLflow

# Slice the run
python -m pipelines.flow --only-group train         # skip pre-analysis
python -m pipelines.flow --only train_cxg           # one model
python -m pipelines.flow --skip-group pre --skip-group post

# Opt-in scoring + drift monitor (need parquet inputs)
python -m pipelines.flow --only score --score-events data/processed/events.parquet
python -m pipelines.flow --only drift_monitor \
    --drift-reference data/features/shots.parquet \
    --drift-current   data/features/shots_new.parquet
```

## Stage groups

| Group | Stages | Purpose |
|---|---|---|
| `pre` | data_quality, feature_stability, univariate, bivariate_{cxg,cxa,cxt}, correlations, eda_{shots,sequences,opponents}, hypothesis_{cxg,cxa,cxt}, statsbomb_baseline, zone_xt_priors, deep_eda | Validate inputs, surface drift, document hypotheses |
| `gate` | gate.data_quality, gate.feature_stability | Hard-stop the run when thresholds fail |
| `train` | train_cxg, train_cxa, train_cxt | Re-train production models |
| `post` | scoring_validation, interpretability, model_comparison | Validate trained artifacts |
| `optional` | score, drift_monitor | Opt-in (need explicit parquet inputs) |

Default execution order: `pre → gate → train → post`. Optional stages
must be requested via `--only` or `--only-group optional`.

## Profiles

Profiles live in `configs/profiles/{cpu,gpu,cloud}.yaml` and are loaded by
`src.runtime.load_profile`. They control:

- Accelerator (cpu / cuda) and precision (fp32, bf16-mixed)
- GBM device (`device` kwarg passed through `src.runtime.gbm_device`)
- DataLoader workers + pin_memory
- Default batch sizes per neural family
- Paths (data/reports/models/outputs/checkpoints) — local on cpu/gpu,
  Drive (`/content/drive/MyDrive/cfm/...`) on cloud
- MLflow tracking URI (file store locally, DagsHub remote in cloud)
- Prefect task runner (Concurrent on cpu, Sequential on gpu/cloud)

Selection precedence: explicit `--profile` arg → `CFM_PROFILE` env →
autodetect (Colab/Kaggle → cloud, CUDA available → gpu, else → cpu).

## MLflow run shape

Every flow invocation opens a parent run named `pipeline:cfm` with
standard tags (profile, accelerator, gbm_device, git_sha, git_branch,
git_dirty, user, stages). Each stage opens a nested run with its own
tags (`stage=<name>`) and logs:

- `params`: function arguments visible at task entry
- `metrics`: scalar fields lifted from the JSON summary written under
  `<reports_root>/<name>.json`
- `artifacts`: the JSON/HTML report itself plus any side-effects in
  `<reports_root>` or `<outputs_root>`

## Quality gates

`pipelines.gates` exposes two opinionated gates (extend as needed):

- `gate.data_quality` — fails when overall missing rate > 40 % or
  more than 5 dtype mismatches are detected in `data_quality_summary.json`.
- `gate.feature_stability` — fails when any feature's PSI > 0.25 in
  `feature_stability_summary.json`.

Tune via CLI: `--gate-max-missing 0.30 --gate-max-psi 0.15 --gate-max-dtype 0`.

## Adding a new stage

1. Pre-analysis script: drop `analysis/NN_<name>.py` with a `def main() -> None`,
   then add a single line to `ANALYSIS_REGISTRY` in
   `pipelines/stages/analysis.py`.
2. Post-analysis: same pattern in `pipelines/stages/post.py::POST_REGISTRY`.
3. Training: add a `@task`-decorated function in `pipelines/stages/train.py`
   and register it in `pipelines/flow.py::TRAIN_STAGES`.
4. Gate: add a `@task` in `pipelines/gates.py` and append to
   `GATE_TASKS` + `GATE_STAGES` in `pipelines/flow.py`.

No CLI changes required — the new stage automatically appears in
`--only` / `--skip` choices.

## Cloud setup

See [`cloud_bootstrap.md`](cloud_bootstrap.md) for the Colab + DagsHub +
DVC walkthrough.
