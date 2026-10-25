# Money Plant ML Pipeline Handoff

## 1. Purpose

This folder contains the machine-learning service for the Money Plant monitoring application. It is a local, CPU-first inference and evaluation pipeline for 16 plants across four experimental groups, with:

- XGBoost multiclass stress classification
- K-Means clustering for environmental states
- Isolation Forest anomaly detection
- SHAP-based explainability (optional)
- heuristic recommendations
- FastAPI endpoints for validation and prediction

This package is designed to be reused by a separate application backend, not embedded into the frontend directly.

## 2. Repository location

- `money_plant_ml_pipeline/`
- Primary app entry: `money_plant_ml_pipeline/src/plant_ml/api/main.py`
- API routes: `money_plant_ml_pipeline/src/plant_ml/api/routes.py`
- Request/response schema: `money_plant_ml_pipeline/src/plant_ml/api/schemas.py`
- Core config: `money_plant_ml_pipeline/config/config.yaml`

## 3. What the pipeline does

### In scope

- Validate CSV and incoming sensor readings for schema, ranges, missingness, and duplicates
- Create per-plant, past-only engineered features
- Train or load separate models for:
  - classifier
  - clustering
  - anomaly detection
- Produce explainability results and recommendation hints
- Serve predictions via a FastAPI API
- Expose metadata and validation endpoints for backend integration

### Out of scope

- Real biological diagnosis or validation against field-truth plant-health measurements
- Database persistence or user-facing app logic
- Automatic retraining on API startup
- Ownership of a backend persistence layer

## 4. Important warning and data provenance

The project explicitly treats the current dataset as synthetic.

From `money_plant_ml_pipeline/config/config.yaml`:

- `data.provenance: synthetic`
- `validated_on_real_plants: false`

This means:

- labels are simulation labels, not biological ground truth
- model metrics are synthetic-data metrics
- recommendations are heuristics, not expert agronomy guidance
- model outputs and recommendations should be handled as experimental signals only

This is a critical requirement for any downstream backend or user interface.

## 5. Folder structure

```text
money_plant_ml_pipeline/
├── config/
│   ├── config.yaml
│   └── feature_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── docs/
│   ├── API_CONTRACT.md
│   ├── ARCHITECTURE.md
│   ├── DATA_DICTIONARY.md
│   ├── FEATURE_ENGINEERING.md
│   ├── JETSON_DEPLOYMENT.md
│   ├── MODEL_CARD.md
│   ├── TRAINING_AND_EVALUATION.md
│   └── TROUBLESHOOTING.md
├── artifacts/
├── reports/
├── scripts/
│   ├── validate_data.py
│   ├── train_all.py
│   ├── evaluate_all.py
│   ├── run_inference.py
│   └── ...
├── src/
│   └── plant_ml/
│       ├── api/
│       ├── feature_engineering.py
│       ├── inference.py
│       ├── train_classifier.py
│       ├── train_clustering.py
│       ├── train_anomaly.py
│       ├── explainability.py
│       ├── recommendations.py
│       └── ...
├── tests/
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## 6. Core architecture

The pipeline follows a layered design:

1. I/O and validation
   - CSV or reading ingestion
   - timestamp normalization to Asia/Kolkata
   - sensor validation and quarantine handling

2. Feature engineering
   - per-plant, past-only features only
   - lag, diff, trend, rolling statistics, VPD, and light-related transforms
   - explicit leakage protections

3. Model training
   - XGBoost classifier
   - K-Means clustering
   - Isolation Forest anomaly detector

4. Explainability and recommendations
   - SHAP for feature importance/explanations
   - heuristic rules based on conditions and trends

5. Artifact management
   - immutable versioned model artifacts
   - checksum and compatibility validation

6. FastAPI service
   - stateless inference contract
   - model artifacts loaded at startup, not trained implicitly

## 7. Model details

### 7.1 Classifier

- Model type: XGBoost multi-class classifier (`multi:softprob`)
- Target classes: `healthy`, `low`, `moderate`, `high`
- Features are sensor + engineered temporal features only
- IDs, timestamps, experimental labels, and synthetic helper columns are excluded from training inputs
- There is a leakage guard to prevent accidental data leakage

### 7.2 Clustering

- Model type: K-Means
- Used to estimate an environmental state or cluster label
- Not treated as a direct health state by itself
- Cluster IDs are not equivalent to plant stress labels

### 7.3 Anomaly detection

- Model type: Isolation Forest
- Intended to detect unusual sensor/data patterns
- `anomaly_score = -decision_function`
- Higher score means more anomalous
- A positive anomaly is interpreted as "check the sensor or reading", not "plant is unhealthy"

## 8. Data contract and sensor schema

The training dataset expects columns in this order:

```text
timestamp, session_id, node_id, sampling_point, plant_id, soil, temperature, humidity, light, stress_level, stress_type
```

Optional columns may include:

- `experimental_group`
- `is_synthetic_anomaly`

Inference rows only need:

```text
timestamp, plant_id, soil, temperature, humidity, light
```

Expected value ranges:

- `soil`: 0-100%
- `temperature`: 15-40 C
- `humidity`: 20-100%
- `light`: 0-100000 lux

Timestamps are normalized to `Asia/Kolkata` and naive timestamps are assumed to be in that timezone.

## 9. Feature engineering rules

The model uses per-plant, trailing/past-only features such as:

- VPD approximation
- `log_light`
- `light_soil_interaction`
- `soil_rate_per_hr`
- lag and delta values
- rolling mean/std over windows `[3, 6, 12]`
- baseline deviation features over a past-only window
- hour-of-day and day-of-week encodings

Important constraint:

- Features are built from prior readings only
- The inference service is designed to receive a current reading plus the plant's history
- The same history contract must be used by any backend integrating with this pipeline

## 10. API contract

Service default URL:

- `http://127.0.0.1:8001`

Endpoints provided by the ML package:

- `GET /health`
- `GET /metadata`
- `POST /validate`
- `POST /predict`
- `POST /predict/batch`

### 10.1 Validation endpoint

Request:

```json
{
  "readings": [
    {"timestamp": "2026-10-21T10:30:00+05:30", "plant_id": 3, "soil": 48.2, "temperature": 26.4, "humidity": 68.0, "light": 3200}
  ]
}
```

Behavior:

- Returns a validity summary and detailed row issues
- HTTP 200 is used even when invalid values are present
- This endpoint is designed for reporting, not blocking ingestion

### 10.2 Prediction endpoint

Request:

```json
{
  "reading": {
    "timestamp": "2026-10-21T10:30:00+05:30",
    "plant_id": 3,
    "soil": 48.2,
    "temperature": 26.4,
    "humidity": 68.0,
    "light": 3200
  },
  "history": [
    {
      "timestamp": "2026-10-21T08:30:00+05:30",
      "plant_id": 3,
      "soil": 48.9,
      "temperature": 25.1,
      "humidity": 70.2,
      "light": 2100
    }
  ],
  "explain": false
}
```

Response includes:

- `stress_prediction`
- `cluster`
- `anomaly`
- `explanation`
- `recommendations`
- `history`
- `warnings`
- `model_metadata`

## 11. Backend integration contract

The ML service is stateless and expects all context to be provided in the request.

Required flow for any application backend:

1. Receive sensor reading
2. Validate locally or through `/validate`
3. Fetch historical rows for the same plant only
4. Send current reading + plant history to `POST /predict`
5. Store raw reading and returned prediction in the application backend
6. Use response metadata and recommendations for UI display

### Critical rules

- Do not mix readings from different plants in the same prediction request
- All history timestamps must be earlier than the current reading
- History should be kept strictly per plant
- A cold-start plant can still get a prediction, but it will be lower reliability
- Model service failures should degrade gracefully without blocking ingestion

## 12. API behavior and error conditions

The ML service returns HTTP 422 for invalid prediction inputs, including:

- mixed plant history
- history not earlier than the reading
- too much history
- empty or malformed requests
- schema/type issues
- invalid feature rows

The service returns HTTP 503 when model artifacts are unavailable or cannot be loaded.

## 13. Artifact management and training behavior

The repo uses immutable artifact directories under `money_plant_ml_pipeline/artifacts/`.

Structure example:

```text
artifacts/
├── latest.json
├── v1.0.0_20261021T063729Z/
├── v1.0.0_20261021T145244Z/
└── ...
```

Each artifact bundle contains:

- classifier model file
- clustering model file
- anomaly model file
- metadata JSON files
- manifest

Important behavior:

- Models are not trained at startup
- The service loads prebuilt artifacts on initialization
- Version mismatch can trigger a compatibility warning or fail depending on config
- `joblib` artifacts should only be loaded from trusted sources

## 14. Recommended usage pattern for the app backend

The backend should treat this package as an inference service and not rebuild the ML logic.

Suggested backend responsibilities:

- persist sensor readings to the DB
- fetch plant history from the DB
- call ML `/predict`
- store the prediction result
- retrieve aggregated data and expose API to frontend

Suggested integration pattern:

```python
# pseudo-flow
history = backend.get_recent_plant_history(plant_id)
response = httpx.post(
    "http://127.0.0.1:8001/predict",
    json={"reading": current_reading, "history": history, "explain": False},
    timeout=5.0,
)
```

If the ML service is unavailable, the backend should:

- keep the raw reading
- mark the prediction as unavailable
- return degraded UI state rather than hard-failing the entire ingestion flow

## 15. Safety and UX considerations

This pipeline should not be presented as a validated agronomic diagnosis.

The frontend and backend must treat outputs as:

- experimental
- synthetic-data derived
- heuristic
- advisory only

Warning labels should be included in UI copy for:

- synthetic provenance
- model uncertainty
- recommendation disclaimers
- anomaly interpretation

## 16. Operational setup

From `money_plant_ml_pipeline/README.md`, the expected setup is:

```bash
cd money_plant_ml_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt
pip install -r requirements-dev.txt
pip install -e .
```

Run the API:

```bash
cd money_plant_ml_pipeline
PYTHONPATH=src python -m uvicorn plant_ml.api.main:app --host 127.0.0.1 --port 8001
```

Validate or train:

```bash
python scripts/validate_data.py --input data/raw/plant_sensor_data.csv
python scripts/train_all.py --input data/raw/plant_sensor_data.csv
python scripts/evaluate_all.py --input data/raw/plant_sensor_data.csv --shap
```

## 17. Critical implementation notes for the app team

- Keep the training pipeline and production backend separate
- Use the existing ML API rather than duplicating logic in a backend service
- Preserve the stateless `reading + history` contract
- Store raw sensor data and prediction outputs separately
- Respect model provenance and synthetic warnings in UI and logs
- Treat anomaly classification as a sensor/operation signal, not plant health diagnosis
- Do not assume cluster IDs imply stress severity

## 18. Summary

The `money_plant_ml_pipeline` folder is the project’s ML inference and evaluation engine. It already contains the core logic needed to:

- validate readings
- engineer features
- classify plant stress
- flag anomalies
- generate explainability and recommendations
- expose a clean FastAPI contract

The main remaining application work is to build the surrounding backend and data persistence layer that calls this service correctly and maintains the required frontend contract.

## 19. Primary files to read first

- `money_plant_ml_pipeline/README.md`
- `money_plant_ml_pipeline/config/config.yaml`
- `money_plant_ml_pipeline/docs/API_CONTRACT.md`
- `money_plant_ml_pipeline/src/plant_ml/api/routes.py`
- `money_plant_ml_pipeline/src/plant_ml/api/schemas.py`
- `money_plant_ml_pipeline/docs/ARCHITECTURE.md`

This handoff should be sufficient to understand the ML service boundary, requirements, and integration constraints before building the application backend or frontend data layer.

