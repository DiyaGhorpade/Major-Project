# Architecture
## Layers
1. **I/O + validation** (`data_loader`, `validation`): CSV -> normalised frame (tz-aware Asia/Kolkata timestamps, numeric sensors) -> validation report. Modes: `fail` (raise), `warn` (mask invalid sensor values to NaN, quarantine structurally unusable rows), `quarantine` (move every flagged row out). Nothing is dropped silently; quarantined rows are written to `data/processed/quarantine_*.csv`.
2. **Features** (`feature_engineering`): single deterministic function used by training, CLI inference and the API. Sorted by (plant_id, timestamp); every temporal feature is grouped by plant and uses only current/past rows.
3. **Selection / leakage guard** (`preprocessing`, `config/feature_config.yaml`): per-model feature lists; forbidden columns raise `LeakageError`.
4. **Models** (`train_classifier`, `train_clustering`, `train_anomaly`): separate training code; inference loads them from artifacts (`inference`).
5. **Explain + advise** (`explainability`, `recommendations`): optional SHAP; rule engine.
6. **Artifacts** (`artifact_manager`): immutable versioned runs, checksum + version compatibility checks.
7. **API** (`api/`): FastAPI, stateless, loads (never trains) models at start-up; if artifacts are missing `/health` reports `unavailable` and `/predict` returns 503.
## Key design decisions
* Stateless inference: caller supplies history. Avoids a hidden feature store and cross-plant contamination.
* XGBoost handles NaN natively (no imputation for the classifier); K-Means / Isolation Forest use median imputation + scaling fit on training rows only; both are recorded in metadata.
* `elapsed_hours` is excluded from the classifier because it depends on how much history a caller sends (train/serve skew).
* No identifiers in models: plant_id / session / node / timestamp would let a model memorise the experimental schedule.
