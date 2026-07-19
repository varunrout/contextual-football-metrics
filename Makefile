.PHONY: install lint format test smoke results-db demo score train

install:  ## Install core + models + viz + dev dependency groups
	poetry install

lint:  ## Ruff lint + format check (matches CI)
	poetry run ruff check .
	poetry run ruff format --check .

format:  ## Auto-fix lint + format
	poetry run ruff check --fix .
	poetry run ruff format .

test:  ## Run the test suite
	poetry run pytest -q

# Data-free honest-evaluation harness. Every analysis/2x script has a --smoke
# mode that runs on synthetic data, so this needs no dvc pull and no models.
smoke:  ## Run every incremental-lift / validity script on synthetic data
	poetry run python analysis/20_incremental_lift_vs_baselines.py --smoke
	poetry run python analysis/21_incremental_lift_cxa.py --smoke
	poetry run python analysis/22_incremental_lift_cxt.py --smoke
	poetry run python analysis/23_cxa_composite_calibration.py --smoke
	poetry run python analysis/24_external_validity.py --smoke
	poetry run python analysis/25_downstream_reranking.py --smoke

results-db:  ## Build results.db from the committed JSON reports
	poetry run python scripts/build_results_db.py

# One-command demo for a fresh clone: no external data required. Builds the
# results store from the committed real reports FIRST, then proves the evaluation
# harness runs on synthetic data, then restores the committed reports (the smoke
# runs write to the same report paths, so we git-restore them to leave the repo
# clean).
demo: results-db  ## Zero-data demo: build results.db, run smoke evals, restore reports
	$(MAKE) smoke
	@git checkout -- reports/ 2>/dev/null && echo "(restored committed reports after smoke run)" || true
	@echo "Demo complete. results.db was built from the committed reports and the evaluation harness ran on synthetic data."
	@echo "For the full pipeline on real data, run 'dvc pull' then 'make train' and 'make score'."

train:  ## Train all three metric ladders on real data (needs data/ via dvc pull)
	poetry run python scripts/train_cxg.py
	poetry run python scripts/train_cxa.py --features data/features/actions.parquet
	poetry run python scripts/train_cxt.py --features data/features/features.parquet

score:  ## Score events end to end with the production models
	poetry run python scripts/score.py --events data/features/features.parquet
