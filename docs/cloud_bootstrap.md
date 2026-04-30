# Cloud bootstrap (Google Colab)

This guide gets the contextual-football-metrics pipeline running on Colab's
free GPU tier with persistent storage on Google Drive and remote experiment
tracking on DagsHub.

## 0. One-time setup (your laptop)

1. Create a free DagsHub account → create a new repo "contextual-football-metrics".
2. Copy the MLflow tracking URI shown in *Remote → Experiments*. It looks like
   `https://dagshub.com/<USER>/contextual-football-metrics.mlflow`.
3. Generate a token: *Settings → Tokens → New token*. Save the username + token.
4. (optional but recommended) initialise DVC for raw data:

   ```bash
   pip install "dvc[gdrive]"
   dvc init
   dvc remote add -d gdrive gdrive://<your-folder-id>
   dvc add data/raw/statsbomb
   git add data/raw/statsbomb.dvc .gitignore .dvc/
   git commit -m "Track raw StatsBomb data with DVC"
   dvc push
   ```

## 1. Open a Colab notebook

```python
# Cell 1 — clone repo
!git clone https://github.com/<USER>/contextual-football-metrics.git
%cd contextual-football-metrics

# Cell 2 — mount Drive (gives /content/drive/MyDrive/...)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3 — install
!pip install -q -e .
!pip install -q prefect mlflow omegaconf "dvc[gdrive]" dagshub

# Cell 4 — secrets (Colab → key icon → add DAGSHUB_USERNAME, DAGSHUB_TOKEN, DAGSHUB_MLFLOW_URI)
from google.colab import userdata
import os
os.environ["MLFLOW_TRACKING_USERNAME"] = userdata.get("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = userdata.get("DAGSHUB_TOKEN")
os.environ["DAGSHUB_MLFLOW_URI"]       = userdata.get("DAGSHUB_MLFLOW_URI")
os.environ["CFM_PROFILE"]              = "cloud"

# Cell 5 — pull data with DVC (skip if you uploaded data manually)
!dvc pull -q

# Cell 6 — run the full pipeline
!python -m pipelines.flow --profile cloud
```

## 2. What the cloud profile changes

| | cpu | gpu | cloud |
|---|---|---|---|
| Accelerator | cpu | cuda (fp32) | cuda (bf16-mixed) |
| Prefect runner | Concurrent | Sequential | Sequential |
| Workers | 4 | 6 | 2 |
| Data root | `./data` | `./data` | `/content/drive/MyDrive/cfm/data` |
| Tracking URI | `file:./mlruns` | `file:./mlruns` | DagsHub (env-driven) |
| Models cached to | `./models` | `./models` | Drive |

## 3. Selecting stage groups

```bash
# Only the long pre-analysis suite (skip training)
python -m pipelines.flow --profile cloud --only-group pre

# Skip pre-analysis (already cached on Drive); run gates + train + post
python -m pipelines.flow --profile cloud --skip-group pre

# Train one model
python -m pipelines.flow --profile cloud --only train_cxg
```

## 4. Recovering from session disconnects

- All artifacts (models, reports, scores) live under `/content/drive/MyDrive/cfm/...`,
  so reconnecting picks up where you left off.
- MLflow runs on DagsHub persist independently of the Colab VM.
- Prefect's per-task cache also persists (under `~/.prefect`); use `--no-cache`
  to force re-run.
