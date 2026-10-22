# Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: plant_ml` | `pip install -e .` or `export PYTHONPATH=src` (scripts add `src` themselves; uvicorn does not) |
| `/health` -> `unavailable`, "No trained models" | run `python scripts/train_all.py` first; the API never trains |
| `Artifact/runtime library mismatch` | retrain on this machine, install the versions in `manifest.json`, or `PLANT_ML_ALLOW_VERSION_MISMATCH=1` (risky) |
| `Checksum mismatch` | artifact modified/corrupt: re-copy or retrain |
| `missing required column(s)` | training CSV needs the exact column order in README; inference needs timestamp, plant_id, soil, temperature, humidity, light |
| API 422 `invalid_readings` | value outside hard range (see `/validate`) |
| API 422 `mixed_plants` / `history_not_before_reading` | send one plant's history, all strictly earlier than the reading |
| Cold-start warning | fewer than 49 history rows; prediction still returned but less reliable |
| `pip` fails for xgboost/shap on Jetson | upgrade pip; use wheels for aarch64; skip SHAP (optional) |
| `LeakageError` | a forbidden column (label/identifier/timestamp) is listed in `feature_config.yaml` |
