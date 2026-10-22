# BUILD REPORT

Built 2026-10-21 in a Linux x86_64 cloud container (no Jetson hardware, no Windows, no Docker daemon).

## Input
`plant_sensor_data.csv`: 5,376 rows, 16 plants x 336 readings (2-hourly, 2026-09-21 -> 2026-10-18), sha256 `92ea22e21bf3ffa8...`.
Schema OK (the supplied file has no `experimental_group` column; group is derived from `plant_id` for reports only). 0 range violations, 0 duplicates,
missing sensor values 1.3-1.6% per sensor, 65 helper anomaly labels. **Treated as SYNTHETIC** because it carries `is_synthetic_anomaly` and nothing states it is
real measurement data. Change `data.provenance` in `config/config.yaml` if that is wrong.

## Commands executed and results
| # | Check | Result |
|---|---|---|
| 1 | `scripts/validate_data.py` on supplied CSV, generated synthetic CSV, sample CSV (`--mode fail`) | PASS (rc 0) |
| 2 | `scripts/generate_synthetic_data.py --rows 8000 --seed 42` twice, `cmp` | PASS: 8,064 rows, byte-identical; built-in assertions (rows, ranges, 4x4 groups, 4 classes, no duplicates, chronology) pass |
| 3 | `scripts/train_all.py --input data/raw/plant_sensor_data.csv` | PASS (about 10 s) |
| 4 | `scripts/evaluate_all.py --shap`, `scripts/generate_explanations.py` | PASS |
| 5 | `scripts/run_inference.py` (full CSV; sample CSV with `--plant-id 3`; missing file -> rc 1 with message) | PASS |
| 6 | `scripts/package_info.py`, `scripts/export_artifacts.py`, `--help` | PASS |
| 7 | Live `uvicorn` on 127.0.0.1:8001: `/health`, `/predict` (plant 5, 80 rows history), 422 on out-of-range, `/metadata` | PASS (about 0.17 s/predict average, dev machine, no SHAP) |
| 8 | `pytest` main environment (Python 3.13) | **103 passed** (re-run after final code change) |
| 9 | `pytest` + `train_all.py` in a second venv with older libraries (pandas 2.2.3, numpy 2.1.3, scikit-learn 1.5.2, xgboost 2.1.4) | **103 passed**; training OK |
One bug was found by tests (validation report crashed for invalid plant ids) and fixed; a pandas FutureWarning in duplicate detection was also fixed.

## Shipped model run: `v1.0.0_20261021T063729Z`
Provenance: **synthetic**, `validated_on_real_plants=false`, seed 42, chronological split (train 4,032 rows up to 2026-10-11T22:05:15+05:30, test 1,344 rows = session 4).
**All numbers are synthetic-data metrics, not validated plant-health performance.**
| Model | Test result |
|---|---|
| XGBoost (62 features) | accuracy 0.627, macro-F1 0.572, weighted-F1 0.630, balanced acc. 0.591 |
| baseline: majority class | accuracy 0.141, macro-F1 0.062 |
| baseline: logistic regression | accuracy 0.461, macro-F1 0.452 |
| per-class F1 (healthy / low / moderate / high) | 0.717 / 0.298 / 0.491 / 0.783; "low" is weak (many predicted healthy) |
| K-Means | k = 3 by silhouette (0.263 on train; 0.225 on test); IDs arbitrary |
| Isolation Forest | contamination 0.02; vs synthetic helper labels on test (14 positives): precision 0.286, recall 0.429, F1 0.343, PR-AUC 0.334 (chance 0.010), ROC-AUC 0.951 |
The older-library venv gave slightly different XGBoost numbers (accuracy 0.638, macro-F1 0.583), so scores shift a little between library versions.
Artifacts were produced with Python 3.13.16, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, xgboost 3.4.1 (exact test-environment pins in `requirements-tested.txt`). The API
**refuses to load** them under a different major.minor of scikit-learn/xgboost/numpy; retrain (about 10 s) or install those versions. Sizes: classifier 2.1 MB, anomaly 1.9 MB, clustering 19 KB.

## Not run / not verified
* **Jetson Orin Nano hardware, JetPack, aarch64 wheels, RAM/disk/latency on device** - not tested; `docs/JETSON_DEPLOYMENT.md` is guidance only.
* **Windows / PowerShell** install - commands documented, not executed.
* **Dockerfile / docker-compose.yml** - not built or run (no Docker available).
* SHAP is implemented and ran here (shap 0.54.1); on a device without `shap` the API reports explanations as unavailable (covered by a code path, not tested with `shap` uninstalled).
* No real plant observations were available, so nothing is biologically validated.

## Known limitations
* Labels are simulated; "low" vs "moderate" separation is weak; healthy-control plants never exceed "low" in this data, so the model partly learns plant-group-like sensor patterns.
* 16 plants over 4 weeks; test period is one session; confidence intervals are not computed.
* Anomaly contamination (0.02) is an assumption; helper labels are few.
* The API has no authentication and no persistence (stateless by design).
* Artifacts contain joblib (pickle) files: load only trusted ones.
