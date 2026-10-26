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
| 8 | `pytest` main environment (Python 3.13) | **124 passed** (103 original + 20 CV/tuning tests + 1 metadata-strings test; re-run after the issue #5 changes) |
| 9 | `pytest` + `train_all.py` in a second venv with older libraries (pandas 2.2.3, numpy 2.1.3, scikit-learn 1.5.2, xgboost 2.1.4) | **103 passed**; training OK |
One bug was found by tests (validation report crashed for invalid plant ids) and fixed; a pandas FutureWarning in duplicate detection was also fixed.

## Shipped model run: `v1.0.0_20261021T145244Z`
(`artifacts/latest.json` points here; the earlier `...T063729Z` run, accuracy 0.627 / macro-F1 0.572, was trained in the build container and is kept only as the previous run.)
Provenance: **synthetic**, `validated_on_real_plants=false`, seed 42, chronological split (train 4,032 rows up to 2026-10-11T22:05:15+05:30, test 1,344 rows = session 4).
| Model | Test result |
|---|---|
| XGBoost (62 features) | accuracy 0.646, macro-F1 0.586, weighted-F1 0.644, balanced acc. 0.606 |
| baseline: majority class | accuracy 0.141, macro-F1 0.062 |
| baseline: logistic regression | accuracy 0.461, macro-F1 0.452 |
| per-class F1 (healthy / low / moderate / high) | 0.730 / 0.321 / 0.495 / 0.800; "low" is weak (many predicted healthy) |
| K-Means | k = 3 by silhouette (0.263 on train); IDs arbitrary |
| Isolation Forest | contamination 0.02; vs helper labels on test (14 positives): precision 0.286, recall 0.429, F1 0.343, PR-AUC 0.334 (chance 0.010), ROC-AUC 0.951 |
Trained on Windows 11 / Python 3.13.1, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.1. Same-seed re-runs on another machine/thread count move the XGBoost macro-F1 by about 0.01 (5 seeds on this machine: 0.580 +- 0.007), so differences below about 0.02 are noise. The API **refuses to load** artifacts under a different major.minor of scikit-learn/xgboost/numpy.

## Cross-validation and tuning (issue #5, part A)
Tooling: `scripts/cv_evaluate.py` (folds + mean +- sd), `scripts/tune_classifier.py` (validation-only search), library code in `src/plant_ml/cv_evaluation.py`, `tuning.py`, `classifier_variants.py`. Reports are written to `reports/cv/` and `reports/tuning/` (git-ignored).
* Chronological hold-out = the primary metric (unchanged). `leave_one_session` trains on three sessions and tests on the fourth (3 of 4 folds train on LATER data: a robustness estimate, not a forecast). `group_kfold_plants` holds out one plant from each of the 4 experimental groups per fold (unseen plants).
* Tuning never sees session 4: inner folds validate on session 2 (train 1+3) and session 3 (train 1+2); 116 trials of depth / min_child_weight / subsample / colsample / eta / lambda x class_weight x {softprob, cumulative-ordinal, regression-ordinal with tuned cut-points}, early stopping for the number of rounds, selection by validation macro-F1, plus a feature-set ablation.
* Validation winner: softprob, `class_weight: balanced`, depth 4, mcw 5, eta 0.03, colsample 1.0, lambda 1, 209 rounds, calendar features (hour, dayofweek, hour_sin/cos) dropped (validation macro-F1 0.649 vs 0.625 for the current config). Best validation macro-F1 per variant: softprob/balanced 0.635, regression-ordinal 0.631, softprob/none 0.629, cumulative-ordinal 0.610: ordinal variants gave no gain.
* Confirmation on session 4 (5 seeds, mean +- sd): current macro-F1 0.580 +- 0.007 vs candidate 0.585 +- 0.002 (gain +0.005 < 0.02 noise margin) -> **the shipped model was not changed.**

mean +- sd across folds (5 seeds averaged inside each fold; macro-F1 averaged over the classes present in the fold's test set):
| Scheme (folds) | Model | macro-F1 | accuracy | balanced acc. |
|---|---|---|---|---|
| chronological hold-out (1) | current config | 0.580 | 0.637 | 0.599 |
| | candidate | 0.585 | 0.665 | 0.605 |
| | logistic regression | 0.452 | 0.461 | 0.465 |
| leave-one-session-out (4) | current config | 0.601 +- 0.072 | 0.616 +- 0.097 | 0.635 +- 0.042 |
| | candidate | 0.598 +- 0.059 | 0.626 +- 0.092 | 0.635 +- 0.031 |
| | logistic regression | 0.467 +- 0.075 | 0.491 +- 0.093 | 0.510 +- 0.056 |
| unseen plants, 4 group-stratified folds | current config | 0.554 +- 0.082 | 0.577 +- 0.073 | 0.601 +- 0.065 |
| | candidate | 0.554 +- 0.068 | 0.583 +- 0.067 | 0.603 +- 0.063 |
| | logistic regression | 0.467 +- 0.081 | 0.490 +- 0.099 | 0.504 +- 0.089 |
Per-class F1, current config (healthy / low / moderate / high): chronological 0.72 / 0.31 / 0.50 / 0.79; leave-one-session 0.71 / 0.36 / 0.60 / 0.81; unseen plants 0.74 / 0.50 / 0.50 / 0.49. On unseen plants "high" drops sharply (recall 0.55, mostly confused with "moderate"): part of the in-time accuracy comes from plant-specific trajectories seen during training. Both configs agree within noise everywhere, so the classifier is limited by the labels / data (control plants 1-4 have "low" episodes with no sensor signature), not by hyper-parameters.

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
