# Data layer

The dataset is StatsBomb Open Data (FIFA World Cup 2022, UEFA Euro 2020, UEFA
Euro 2024, La Liga 2020/21) flowing through three layers: **raw** (StatsBomb
JSON) to **processed** (typed, normalised parquet) to **features** (per-metric
modelling tables). All parquet is columnar and analytics-heavy, so parquet is the
primary store; the derived model metrics live in a small SQLite results store
(see `docs` on the results DB and `scripts/build_results_db.py`).

Everything under `data/` is gitignored and DVC-tracked. Rebuild it with
`scripts/ingest.py` (raw to processed) and `scripts/build_features.py`
(processed to features), or `dvc pull` if a remote is configured.

## Layout

```
data/
  raw/statsbomb/
    competitions/            competitions.json (1 file)
    matches/                 matches per competition (12 files)
    events/                  one events file per match (874 files)
    lineups/                 one lineup file per match (874 files)
    frames/                  360 freeze-frame files (166 files, 360-covered matches)
  processed/
    matches.parquet          match registry: ids, split_role, teams, score, 360 flag
    events.parquet           flattened event stream (StatsBomb schema)
    possessions.parquet      one row per possession, with sequence labels
    frames.parquet           360 freeze-frame points (one row per tracked player-event)
  features/
    shots.parquet            CxG modelling table (shots only)
    actions.parquet          CxA modelling table (passes and carries)
    features.parquet         full feature store (all event types; CxT modelling table)
    zone_xt_priors.parquet   static per-zone threat surface (16x12 grid)
```

The layout is flat parquet, not Hive-partitioned directories. Partitioning is
logical, by columns: `competition_id` / `season_id` / `split_role` live on
`matches.parquet`, and every other table joins to it by `match_internal_id`.

## Identity and join keys

Rows carry a provider-agnostic `internal_id` (a 16-char hash) plus the original
StatsBomb id. The stable join keys are:

- `match_internal_id` -> `matches.internal_id`
- `possession_internal_id` -> `possessions.internal_id`
- `event_internal_id` / `event_id` -> `events` row
- `team_internal_id`, `player_internal_id`, `opponent_id`

**Gotcha:** in the feature tables (`shots`, `actions`, `features`) the
`competition_id` column is a hashed value, not the StatsBomb integer, and
`season_id` is absent. To select by competition or season (for example the
Euro 2024 held-out split), join to `matches.parquet` on `match_internal_id` and
filter on its real `competition_id` / `season_id` / `split_role`. The analysis
scripts do this via `analysis/_utils.heldout_mask`.

## Train / validation / test split

`matches.split_role` defines the split, so it is a match-level (leakage-safe)
partition:

| split_role | competition | season | role |
|---|---|---|---|
| `train` | UEFA Euro | 2020 | training |
| `train_val` | FIFA World Cup | 2022 | training + validation |
| `val_test` | UEFA Euro | 2024 | validation + final held-out (the evaluation set) |
| `test` | La Liga | 2020/21 | out-of-sample scoring only, excluded from training |

## Processed datasets

**matches.parquet** (201 rows). Registry and the source of truth for
competition/season/split. Columns: `internal_id`, `statsbomb_match_id`,
`competition_id`, `season_id`, `has_360`, `split_role`, `home_team_name`,
`away_team_name`, `home_team_id`, `away_team_id`, `home_team_internal_id`,
`away_team_internal_id`, `match_date`, `home_score`, `away_score`.

**events.parquet** (754,232 rows, 129 cols). The flattened StatsBomb event
stream (one row per event, StatsBomb attribute names such as `shot_statsbomb_xg`,
`pass_cut_back`, `play_pattern`, `action_type`). Source for `shot_statsbomb_xg`
(the off-the-shelf xG baseline) and for the feature builders.

**possessions.parquet** (34,867 rows, 23 cols). One row per possession:
`internal_id`, `match_internal_id`, `team_internal_id`, `possession_index`,
start/end event ids and timestamps, `start_x`/`start_y`, `regain_zone`,
`n_events`/`n_passes`/`n_carries`/`n_shots`, `vertical_progression`,
`distance_progressed`, `set_piece_flag`, `counterpress_regain_flag`, and the
`sequence_type*` labels.

**frames.parquet** (8,306,252 rows, 6 cols). 360 freeze-frame points, one row per
tracked player per event: `event_internal_id`, `x`, `y`, `teammate`, `keeper`,
`match_internal_id`. Consumed by the neural / freeze-frame features.

## Feature datasets

The three modelling tables share the same wide schema (147-149 columns): a block
of identifier columns followed by feature columns grouped as traditional,
sequence, opponent-context and freeze-frame features. The full column contract
is defined in `configs/features.yaml` (parsed by
`analysis/_utils.feature_groups`); that YAML is the single source of truth for
feature names, so it is not duplicated here.

Identifier columns (common to all three): `player_id`, `team_id`, `opponent_id`,
`competition_id` (hashed), `match_id`, `possession_id`, `event_id`,
`match_internal_id`, `possession_internal_id`, `team_internal_id`,
`player_internal_id`, plus `x_location`, `y_location` and `event_type`.

| Dataset | Rows | Grain | Used by |
|---|---|---|---|
| `shots.parquet` | 4,962 | one row per shot | CxG (`shot_statsbomb_xg`, `goal` attached from events) |
| `actions.parquet` | 392,165 | one row per pass/carry | CxA creation + quality |
| `features.parquet` | 754,232 | all event types (incl shots) | CxT state-value; shot CxG scoring |
| `zone_xt_priors.parquet` | 192 | one row per pitch zone | CxT zone baseline (`zone_id`, `x_bin`, `y_bin`, `xt_value`, `shot_prob`, `shot_value`) |

Note that `actions.parquet` contains only passes and carries (no shot rows); shot
CxG for the CxA/CxT targets is sourced from `features.parquet` and linked back to
actions by possession. This is why `analysis/21`, `22` and `23` load both tables.

## Rebuilding

```bash
python scripts/ingest.py          # raw StatsBomb JSON -> data/processed/*.parquet
python scripts/build_features.py  # processed -> data/features/*.parquet
```

Competition and season coverage is configured in `configs/competitions.yaml`.

## Results store (SQLite)

The parquet layer is the analytics store; the small, relational, dashboard-facing
numbers live in a SQLite `results.db` so the app and reviewers can query
leaderboards and evaluation results without scanning parquet or re-running
training. It is built from the committed JSON reports and is gitignored (rebuild
in seconds):

```bash
python scripts/build_results_db.py       # reads reports/*.json -> results.db
```

Tables:

| Table | Grain | Contents |
|---|---|---|
| `model_run` | metric x model | family, feature_set, `is_promoted`, held-out competition |
| `model_metric` | metric x model x metric_name x split | metric values at split in {cv, train, holdout} |
| `incremental_lift` | metric x candidate x baseline x delta | delta mean, 95% CI, `excludes_zero`, verdict |
| `calibration` | metric x kind | e.g. CxA composite ECE and Spearman |

Read it with `src/results_store.py`:

```python
from src import results_store
results_store.leaderboard("cxg")        # held-out CxG leaderboard, promoted model flagged
results_store.incremental_lift("cxt")   # CxT lift vs the zone baseline, with CIs
results_store.calibration()             # composite CxA calibration summary
```

The Streamlit app's "Model Evaluation" tab reads these tables directly.
