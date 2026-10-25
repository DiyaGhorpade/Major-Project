SHELL := /bin/bash
.DEFAULT_GOAL := help

ROOT   := $(CURDIR)
ML     := $(ROOT)/money_plant_ml_pipeline
BE     := $(ROOT)/backend
FE     := $(ROOT)/frontend

.PHONY: help setup dev test seed reset-db ml-explain

help:
	@echo "make setup        create venvs, install deps, check ML artifacts, generate the SHAP report (no servers)"
	@echo "make dev          set up if needed, then start ML :8001, backend :8000, frontend :5173"
	@echo "make test         backend pytest + frontend typecheck, lint, tests, build"
	@echo "make seed         create the DB and load the bundled dataset (no servers)"
	@echo "make reset-db     delete backend/data/app.db* (re-seeded on next start)"
	@echo "make ml-explain   generate the global SHAP report for the current model run (no training)"

setup:
	./scripts/dev.sh --setup-only

dev:
	./scripts/dev.sh

# Start the ML service first (make dev) so the real-ML integration test runs; otherwise it skips.
test:
	cd $(BE) && .venv/bin/python -m pytest -q
	cd $(FE) && npm run typecheck && npm run lint && npm test && npm run build

seed:
	cd $(BE) && .venv/bin/python scripts/seed.py

reset-db:
	rm -f $(BE)/data/app.db $(BE)/data/app.db-shm $(BE)/data/app.db-wal
	@echo "Database deleted. Stop the backend first if it is running; it re-seeds on next start."

ml-explain:
	cd $(ML) && .venv/bin/python scripts/generate_explanations.py --input data/raw/plant_sensor_data.csv \
		--run-id "$$(python3 -c 'import json; print(json.load(open("artifacts/latest.json"))["run_id"])')"
