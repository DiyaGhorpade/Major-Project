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
| 8 | `pytest` main environment (Python 3.13) | **160 passed** after issue #5 part B (124 before + cross-plant causality / parity, smoothing, per-fold fit, augmentation, inference and API reference tests) |
| 8b | backend `pytest` after issue #5 part B (reference plants, feature dictionary) | **319 passed, 2 skipped** (ML-service integration tests skip without a running service) |
| 9 | `pytest` + `train_all.py` in a second venv with older libraries (pandas 2.2.3, numpy 2.1.3, scikit-learn 1.5.2, xgboost 2.1.4) | **103 passed**; training OK |
One bug was found by tests (validation report crashed for invalid plant ids) and fixed; a pandas FutureWarning in duplicate detection was also fixed.

## Shipped model run: `v1.0.0_20261026T145547Z` (issue #5, part B)
(`artifacts/latest.json` points here; `previous_run_id` is `v1.0.0_20261021T145244Z`, the model shipped before issue #5 part B.)
Provenance: **synthetic**, `validated_on_real_plants=false`, seed 42, chronological split (train 4,032 rows up to 2026-10-11T22:05:15+05:30, test 1,344 rows = session 4).
| Model | Test result (session 4) |
|---|---|
| XGBoost (53 features, balanced weights, 12-reading HMM smoothing) | accuracy 0.803, macro-F1 0.743, weighted-F1 0.792, balanced acc. 0.748 |
| same model before smoothing (`metrics_unsmoothed`) | accuracy 0.765, macro-F1 0.723, balanced acc. 0.732 |
| previous shipped model `...20261021T145244Z` (62 features) | accuracy 0.646, macro-F1 0.586, balanced acc. 0.606 |
| baseline: majority class | accuracy 0.141, macro-F1 0.062 |
| baseline: logistic regression (same 53 features) | accuracy 0.459, macro-F1 0.464 |
| per-class F1 (healthy / low / moderate / high) | 0.822 / 0.495 / 0.762 / 0.895 (previously 0.730 / 0.321 / 0.495 / 0.800); "low" is still the weak class: 93 of 189 "low" rows are predicted healthy |
| K-Means | k = 3 by silhouette; IDs arbitrary (unchanged features) |
| Isolation Forest | contamination 0.02; vs helper labels on test (14 positives): precision 0.286, recall 0.429, F1 0.343, PR-AUC 0.334, ROC-AUC 0.951 (unchanged features) |
Trained on Linux x86_64 / Python 3.13.11, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, xgboost 3.4.1. Re-evaluating the saved artifact (`make evaluate`) reproduces macro-F1 0.743. Seed-to-seed sd on the hold-out is about 0.006.

## Cross-validation and tuning (issue #5, part A; superseded by part B below)
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

## Issue #5, part B: cross-plant features, longer context, temporal smoothing
Part A showed hyper-parameters were not the bottleneck (+0.005 macro-F1). Part B changed what the model sees. Every choice below was made on validation folds inside sessions 1-3 (train 1+3 / validate 2, train 1+2 / validate 3, and unseen-plant folds restricted to sessions 1-3). Session 4 was used for the final confirmation. (One early exploratory feature ablation printed session-4 numbers before this protocol was fixed; it did not decide anything the validation folds did not also show. The missing-reference failure described below was first seen in a session-4 diagnostic, and the fix and its rate were chosen on validation.)

**Ceiling.** The supplied CSV is **not** output of `src/plant_ml/synthetic.py` (with the same seed that generator gives 24 "low" rows per control plant; the CSV has 83-149), and its own generator is not in the repository. So there is no latent severity to fit, and the ceiling was estimated from the labels. A label oracle predicts each row from the mean of its neighbours' TRUE labels (cut-points tuned on sessions 1-3); it knows the plant's latent trajectory exactly and only misses the per-row label flicker. On session 4 it reaches **macro-F1 0.82** with neighbours on both sides (±4 rows) and **0.80** with past labels only. Accuracy is 0.90 on treated plants (5-16) and **0.77 on control plants (1-4)**, where "always healthy" already scores 0.75: the controls' "low" episodes are close to unpredictable even from the true labels around them. A sensor-only model cannot beat this oracle, so about 0.80 is a realistic upper bound for this data. The shipped model is at 0.74.

**Ablation** (3 seeds; mean ± sd; sd is across seeds for the single hold-out fold, across folds otherwise; macro-F1 over the classes present in each test fold):
| Configuration | chronological hold-out | leave-one-session-out (4) | unseen plants (4) |
|---|---|---|---|
| A previous model (62 features) | 0.582 ± 0.009 | 0.601 ± 0.072 | 0.554 ± 0.082 |
| A + HMM smoothing only | 0.612 ± 0.006 | 0.597 ± 0.089 | 0.562 ± 0.092 |
| B A + cross-plant deviations (`*_ref_dev*`) | 0.696 ± 0.004 | 0.658 ± 0.119 | 0.613 ± 0.072 |
| C B + 24/48-reading context (`*_long*`) | 0.707 ± 0.002 | 0.657 ± 0.159 | 0.623 ± 0.066 |
| D pruned 53-feature set | 0.714 ± 0.002 | 0.662 ± 0.176 | 0.627 ± 0.071 |
| E D + balanced class weights | 0.734 ± 0.006 | 0.674 ± 0.164 | 0.628 ± 0.062 |
| F E + HMM smoothing | 0.763 ± 0.004 | 0.681 ± 0.191 | 0.641 ± 0.084 |
| **G shipped: F + missing-reference augmentation** | **0.743 ± 0.006** | **0.678 ± 0.155** | **0.627 ± 0.069** |
| G with NO reference readings at all | 0.719 ± 0.026 | 0.662 ± 0.099 | 0.610 ± 0.057 |
Balanced accuracy, shipped vs previous: 0.747 vs 0.600 (hold-out), 0.712 ± 0.095 vs 0.635 ± 0.042 (sessions), 0.667 ± 0.049 vs 0.601 ± 0.065 (unseen plants). Per-class F1 (healthy / low / moderate / high), shipped: hold-out 0.83 / 0.51 / 0.74 / 0.89, sessions 0.76 / 0.49 / 0.69 / 0.88, unseen plants 0.77 / 0.54 / 0.63 / 0.56. Hold-out accuracy on treated plants 0.806 (previously 0.607) and on controls 0.770 (previously 0.726).
* **Cross-plant deviations are the main gain** (+0.11 hold-out, +0.06 on both CV schemes). All plants share one room, so subtracting the concurrent median of the healthy controls cancels the weather, the day-night cycle and session drift. Reference values come from an as-of join, so a row never uses a later reading of another plant.
* **Smoothing** (HMM forward filter over the last 12 predictions; transitions and prior fitted on training labels, temperature 0.3 chosen on validation) adds +0.03 on the hold-out and +0.01 on the CV schemes. A plain rolling mean or EMA was within 0.01 of it.
* **Balanced weights** gave +0.018 validation macro-F1 in 14 of 14 paired hyper-parameter draws. A light re-tune (28 specs) found no hyper-parameters better than the defaults; ordinal (cumulative) models were within noise and are not inference-compatible, so they were not shipped.
* **Missing-reference augmentation** (25% of training rows duplicated with the cross-plant features blanked). Without it, a reading that arrives with no reference plants scores macro-F1 **0.11** on validation, worse than the majority class. With it, such readings score 0.69 (0.72 on session 4), still above the previous model. The price is about 0.01-0.02 when references are present (validation 0.740 to 0.727; session 4 0.763 to 0.743). It ships because a model that collapses when the control sensors go offline is not deployable.
* Tried and rejected on validation (within noise): an all-plant reference (it also hurt unseen plants), longer-window trends of the deviations, daytime-only light deviation, rolling min/max of deviations, and keeping the raw absolute sensor features.
* **Regression in one fold:** the leave-one-session fold that tests on session 1 (trained on sessions 2-4, later data) falls from 0.507 to 0.447. Session 1 is the experiment's cold start: the 24/48-reading windows are still filling for its first 2-4 days, a state that later-session training rows never show. All other folds improve (sessions 2/3/4: 0.663 / 0.653 / 0.582 previously, 0.750 / 0.773 / 0.743 now). The API flags such readings as cold start (fewer than 60 history rows).
* Unseen plants remain the hardest case ("high" F1 0.56): part of the in-time accuracy still comes from trajectories of plants seen during training.

**Deployment** (see `docs/TRAINING_AND_EVALUATION.md`):
* `/predict` and `/predict/batch` accept `reference`: readings of the reference plants listed in `/metadata` `cross_plant.reference_plants` (per item, or once per batch). Responses keep their shape and add `stress_prediction.unsmoothed_probabilities`. When no reference reading is found the response carries a warning.
* Smoothing is stateless: the service re-predicts the plant's trailing 12 rows from the supplied history. `required_history_rows_for_full_features` is now 60 (49 + 11), and the backend already sends 60 rows.
* Backend: `/metadata` is read for the reference plants. The pipeline run sends the control plants' readings once per batch, covering the union of the items' history spans. The on-demand prediction sends them from the database. Nothing changes when the ML service does not advertise reference plants (an older model).

## Not run / not verified
* **Jetson Orin Nano hardware, JetPack, aarch64 wheels, RAM/disk/latency on device** - not tested; `docs/JETSON_DEPLOYMENT.md` is guidance only.
* **Windows / PowerShell** install - commands documented, not executed.
* **Dockerfile / docker-compose.yml** - not built or run (no Docker available).
* SHAP is implemented and ran here (shap 0.54.1); on a device without `shap` the API reports explanations as unavailable (covered by a code path, not tested with `shap` uninstalled).
* No real plant observations were available, so nothing is biologically validated.

## Known limitations
* Labels are simulated; "low" is the weakest class (F1 about 0.5); healthy-control plants never exceed "low" in this data, and their "low" episodes have almost no sensor signature (even a label oracle is barely above "always healthy" on controls).
* The cross-plant features assume the experiment's design: plants 1-4 are healthy controls in the same room as the treated plants and are read in the same 2-hour slot. Without concurrent control readings, predictions fall back to per-plant features (macro-F1 about 0.72 instead of 0.74 on session 4) and carry a warning.
* 16 plants over 4 weeks; test period is one session; confidence intervals are not computed.
* Anomaly contamination (0.02) is an assumption; helper labels are few.
* The API has no authentication and no persistence (stateless by design).
* Artifacts contain joblib (pickle) files: load only trusted ones.
