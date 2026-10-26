# Money Plant ML Pipeline (Jetson Orin Nano)

A complete, local, CPU-only ML pipeline for monitoring 16 Money Plants (*Epipremnum aureum*) in four experimental groups.
Three models + SHAP explanations + rule-based recommendations, exposed through a FastAPI service.

> **DISCLAIMER - EXPERIMENTAL.** The supplied dataset (`data/raw/plant_sensor_data.csv`) carries a synthetic anomaly helper
> column and is treated as **SYNTHETIC** (`config/config.yaml: data.provenance`). Its labels are simulation labels, **not biological
> ground truth**. Every metric here is a *synthetic-data metric*, never validated plant-health performance. Model outputs and
> recommendations are heuristic and are **not** plant-health diagnoses. Change `data.provenance` only if you have verified real data.

## What it does / does not do
| Does | Does not |
|---|---|
| Validate CSVs (schema, ranges, duplicates, missingness) without silently deleting rows | Prove any biological result |
| Leakage-safe, per-plant, past-only features shared by training and inference | Need a GPU, CUDA, TensorRT, PyTorch or internet at inference |
| XGBoost 4-class stress classifier (healthy/low/moderate/high) | Train silently at API start-up (training is always explicit) |
| K-Means environmental states; Isolation Forest sensor/data anomalies | Treat cluster IDs as health states, or anomalies as plant stress |
| Optional SHAP explanations, transparent heuristic recommendations | Recommend watering volumes or rigid biological thresholds |
| Local REST API (`/health /metadata /predict /predict/batch /validate`) | Own or migrate your SQLite database |

## Architecture / data flow
```
ESP32 -> existing receiver -> SQLite --(last N rows for one plant)--> POST /predict
                                                                          |
   CSV --load/validate--> features (per plant, past-only) --> XGBoost | K-Means | Isolation Forest --> SHAP + rules --> JSON
```
See `docs/ARCHITECTURE.md`.

## Folder structure
```
config/            config.yaml (all settings), feature_config.yaml (model inputs)
data/raw/          plant_sensor_data.csv (your dataset)   data/sample/  small samples (see data/README.md)
src/plant_ml/      schema, data_loader, validation, feature_engineering, preprocessing, splitters, labels,
                   train_classifier / train_clustering / train_anomaly, training, evaluate, evaluation_run,
                   explainability, recommendations, inference, artifact_manager, reports, synthetic, api/
scripts/           validate_data, train_all, evaluate_all, generate_explanations, run_inference,
                   generate_synthetic_data, export_artifacts, package_info
tests/             103 pytest tests          docs/  8 documents          artifacts/  versioned model runs
reports/           per-run reports + plots   BUILD_REPORT.md  what was executed, versions, limitations
```

## Install
Python 3.10+ (tested on 3.13 and, for portability, a venv with pandas 2.2 / numpy 2.1 / scikit-learn 1.5 / xgboost 2.1).
```bash
python -m venv .venv
source .venv/bin/activate            # Linux / Jetson
# .venv\Scripts\Activate.ps1         # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt                   # core
pip install -r requirements-optional.txt          # SHAP + plots (optional)
pip install -r requirements-dev.txt               # + pytest, httpx
pip install -e .                                  # optional; scripts also work without it
```
Jetson specifics: `docs/JETSON_DEPLOYMENT.md`.

## Commands (run from the project root; all were executed - see BUILD_REPORT.md)
```bash
python scripts/validate_data.py --input data/raw/plant_sensor_data.csv
python scripts/train_all.py     --input data/raw/plant_sensor_data.csv     # -> artifacts/<run>/ + reports/<run>/
python scripts/evaluate_all.py  --input data/raw/plant_sensor_data.csv --shap
python scripts/run_inference.py --input data/raw/plant_sensor_data.csv     # latest row per plant (labels not needed)
python -m uvicorn plant_ml.api.main:app --host 127.0.0.1 --port 8001       # if not pip-installed: PYTHONPATH=src
python -m pytest -q
# deterministic synthetic fallback data:
python scripts/generate_synthetic_data.py --output data/sample/synthetic_sensor_readings.csv --rows 8000 --seed 42
```
Every script has `--help`. Useful flags: `--split group_holdout`, `--validation-mode fail|warn|quarantine`, `--provenance`, `--no-set-latest`.

## Expected outputs
`artifacts/<vX.Y.Z_timestamp>/` : `classifier.xgb.json`, `clustering.joblib`, `anomaly.joblib`, `*.meta.json`, `manifest.json`; `artifacts/latest.json`.
`reports/<run>/` : `REPORT.md`, `data_quality_report.json`, `class_distribution.json`, `classifier_metrics.json`, `clustering_report.json`,
`anomaly_report.json`, `artifact_manifest.json`, `classifier_test_predictions.csv`, `confusion_matrix.png`, `feature_importance_gain.png`,
`silhouette_by_k.png`, `anomaly_score_hist.png`, `shap_global_importance.{json,png}`.

## Sensor schema (units)
Training CSV columns, in this exact order: `timestamp, session_id, node_id, sampling_point, plant_id, soil, temperature, humidity, light,
stress_level, stress_type`, then optional `experimental_group, is_synthetic_anomaly`. `soil` % (0-100), `temperature` deg C (15-40),
`humidity` % RH (20-100), `light` lux (0-100000). Inference rows need only `timestamp, plant_id, soil, temperature, humidity, light`.
Full detail: `docs/DATA_DICTIONARY.md`. Timestamps are normalised to Asia/Kolkata (naive values are assumed to be in it).

## Bengaluru simulation assumptions
Only used by `scripts/generate_synthetic_data.py` (indoor/shaded 23-30 C days, 17-23 C nights, 40-75% RH, rainy sessions to 90%, light
100-15,000 lux with rare sun peaks, near-zero at night). They are simulation assumptions, not weather observations or plant-health thresholds.

## Feature engineering (all per plant, trailing/past-only)
`vpd` (approximate air VPD, kPa), `log_light`, `light_soil_interaction`, `soil_rate_per_hr`, lag1/diff1/trend3 and trailing rolling mean/std
(windows 3, 6, 12) of soil/temperature/humidity/light/VPD, past-only 48-observation baseline deviations (warm-up 6), hour, day-of-week,
sin/cos hour, 24/48-observation context, and deviations from the concurrent healthy-control plants 1-4 (as-of join, never later
readings). The classifier uses 53 of them (`config/feature_config.yaml`), and its probabilities are smoothed over each plant's last 12
readings (HMM forward filter). Definitions and leakage tests: `docs/FEATURE_ENGINEERING.md`.

## Models
* **A - XGBoost** (`multi:softprob`): inputs = sensor + engineered features only; labels, IDs, timestamps, group and helper label are excluded
  (a `LeakageError` guard + tests enforce this). Baselines: majority class, logistic regression.
* **B - K-Means**: median-impute + standardise (fit on train), k chosen by silhouette in [2, 8] (not forced to 4). Returns cluster id + centroid distance.
  IDs are arbitrary.
* **C - Isolation Forest**: contamination 0.02 (configurable). `anomaly_score = -decision_function` (**higher = more anomalous**), flag if > 0.
  `is_synthetic_anomaly` is used only for offline evaluation.

## Split and how to read metrics
Default: **chronological** - the last 25% of timestamps (the final weekly session) is test; features use only past rows. `group_holdout` tests unseen plants
(4, 8, 12, 16). No random row-level split is offered. On the supplied data XGBoost beats both baselines (see `BUILD_REPORT.md` for numbers), but the
class "low" is weak, and the numbers describe recovery of **simulator labels**. Read per-class recall and the confusion matrix, not just accuracy.
Metric scores can shift slightly between library versions (we observed ~1 point of accuracy between xgboost 3.4 and 2.1).

## API
`GET /health`, `GET /metadata`, `POST /validate`, `POST /predict`, `POST /predict/batch` - request/response examples and error codes in `docs/API_CONTRACT.md`.
The service is **stateless**: send the latest reading plus a chronological history for that plant (>= 60 rows for full feature parity)
and, as `reference`, the concurrent readings of the control plants listed in `/metadata` `cross_plant.reference_plants`.
A cold-start plant still gets a prediction, flagged with a warning. History from another plant is rejected (HTTP 422).

## Frontend/backend integration
`docs/API_CONTRACT.md` contains the step-by-step flow, a `httpx` client with timeout/retry, a proposed SQLite `ml_predictions` table, and rate-limiting guidance.

## Artifact versioning
Each training run writes a new immutable directory `v<version>_<UTC timestamp>/`; `latest.json` points at the newest and remembers the previous one (rollback =
edit `latest.json`). SHA-256 hashes and library versions are verified at API start-up; a mismatch gives an actionable error. Models are never trained at start-up.
`joblib` files are pickle-based: only load artifacts you trust.

## Troubleshooting
`docs/TROUBLESHOOTING.md`.
