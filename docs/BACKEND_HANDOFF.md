# Money Plant Monitoring System Backend Handoff

## 1. Project overview

The repository currently contains:
- A frontend in `frontend/` built with React + Vite
- A machine-learning service in `money_plant_ml_pipeline/` built with Python + FastAPI
- No dedicated application backend in the application branch yet
- Frontend pages that are wired to a mock-data layer and UI state, rather than a live database-backed API

The application is intended to monitor 16 tulsi plants across four experimental clusters, ingest sensor readings, persist the data, run inference with the existing ML pipeline, and surface insights to the frontend.

The current architecture is best understood as:
- Frontend: dashboard and monitoring UI (`frontend/src/pages/*`)
- ML service: model loading, feature preparation, validation, prediction, and explainability (`money_plant_ml_pipeline/src/plant_ml/...`)
- Future backend: a thin orchestration layer that manages data persistence, sensor ingestion, prediction storage, plant/device metadata, dashboards, and API endpoints consumed by the frontend

This backend should be built as a Python FastAPI application for consistency with the existing ML package and because the ML pipeline already exposes a FastAPI contract.

## 2. Repository findings

### 2.1 Frontend structure

The application branch contains a Vite React UI under `frontend/`.

Core routing and page composition:
- `frontend/src/App.tsx`
- Pages include:
  - `OverviewPage`
  - `LiveMonitoringPage`
  - `PlantsPage`
  - `PlantDetailPage`
  - `AiInsightsPage`
  - `ShapAnalysisPage`
  - `ModelPerformancePage`
  - `RecommendationsPage`
  - `DataLogsPage`
  - `DevicesNetworkPage`
  - `SessionsPage`
  - `SettingsPage`

Relevant shared types:
- `frontend/src/types/index.ts`

API layer:
- `frontend/src/api/client.ts`
- `frontend/src/api/mockData.ts`

From the frontend structure, the intended application domain is:
- Plants
- Device/network status
- Sensor readings
- Sessions
- Recommendations
- SHAP explanations
- Model metrics
- Data logs

The UI includes many pages and statuses, but the application branch currently appears to use a local mock-data pattern rather than a database-backed service.

### 2.2 ML pipeline structure

The ML package is in:
- `money_plant_ml_pipeline/`

Key directories:
- `config/`
- `data/raw/`
- `data/processed/`
- `src/plant_ml/`
- `scripts/`
- `tests/`
- `artifacts/`
- `reports/`
- `docs/`

The ML pipeline already supports:
- Data validation
- Feature engineering
- Model training (explicit, not automatic)
- XGBoost classifier
- K-means clustering
- Isolation Forest anomaly detection
- SHAP explanation support
- FastAPI inference service

Important files:
- `money_plant_ml_pipeline/README.md`
- `money_plant_ml_pipeline/config/config.yaml`
- `money_plant_ml_pipeline/src/plant_ml/api/main.py`
- `money_plant_ml_pipeline/src/plant_ml/api/routes.py`
- `money_plant_ml_pipeline/src/plant_ml/api/schemas.py`
- `money_plant_ml_pipeline/docs/API_CONTRACT.md`

The pipeline API exposes:
- `GET /health`
- `GET /metadata`
- `POST /validate`
- `POST /predict`
- `POST /predict/batch`

The inference contract is stateless:
- request includes a current reading and prior history for the same plant
- readings are validated and rejected if invalid
- predictions are generated with the same feature engineering/training pipeline

### 2.3 Existing functionality and gaps

Existing useful code:
- Frontend UI structure is complete and polished
- ML service is already implemented and ready to be reused
- Data validation and prediction interfaces already exist
- Model artifacts are present under `money_plant_ml_pipeline/artifacts`

Main gap:
- There is no real backend in the application branch that:
  - persists readings
  - maintains plants/devices/sessions
  - wires the frontend to live API responses
  - stores predictions and recommendations
  - provides dashboard aggregation from stored data

## 3. Target backend architecture

Recommended backend framework:
- Python FastAPI
- SQLite for local single-device deployment
- SQLAlchemy or SQLModel for ORM/database model layer
- Pydantic validation models
- Scheduled or request-triggered ML inference using existing predictor service

Recommended app layers:
- `app/api/routes/...`
- `app/core/config.py`
- `app/core/logging.py`
- `app/db/database.py`
- `app/db/models.py`
- `app/services/ingestion.py`
- `app/services/prediction_service.py`
- `app/services/dashboard_service.py`
- `app/services/recommendation_service.py`
- `app/services/device_service.py`
- `app/integrations/ml_client.py`

Data flow:
1. ESP32 or simulator sends readings
2. Backend validates and normalizes payload
3. Backend stores raw sensor readings
4. Backend fetches plant history and calls ML service
5. Prediction result is stored
6. Frontend requests dashboard and plant APIs
7. Recommendations and insight endpoints aggregate data from backend and ML metadata

This architecture keeps model training separate from inference and avoids duplicating ML logic.

## 4. Proposed folder structure

This is the recommended structure for the new backend without disturbing frontend or ML pipeline code:

```text
.
├── frontend/
│   └── existing React app (leave intact, update API integration)
├── money_plant_ml_pipeline/
│   └── existing ML pipeline (do not duplicate or retrain)
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── health.py
│   │   │   │   ├── plants.py
│   │   │   │   ├── sensors.py
│   │   │   │   ├── monitoring.py
│   │   │   │   ├── predictions.py
│   │   │   │   ├── ai.py
│   │   │   │   ├── devices.py
│   │   │   │   ├── sessions.py
│   │   │   │   └── recommendations.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   └── settings.py
│   │   ├── db/
│   │   │   ├── database.py
│   │   │   ├── models.py
│   │   │   ├── repositories.py
│   │   │   └── migrations/
│   │   ├── services/
│   │   │   ├── ingestion_service.py
│   │   │   ├── plant_service.py
│   │   │   ├── device_service.py
│   │   │   ├── prediction_service.py
│   │   │   ├── monitoring_service.py
│   │   │   └── recommendation_service.py
│   │   ├── integrations/
│   │   │   ├── ml_client.py
│   │   │   └── ml_types.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── plant.py
│   │   │   ├── sensor.py
│   │   │   ├── prediction.py
│   │   │   ├── dashboard.py
│   │   │   └── device.py
│   │   └── main.py
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_ingestion.py
│   │   ├── test_ml_integration.py
│   │   └── test_database.py
│   ├── .env.example
│   ├── requirements.txt
│   ├── README.md
│   └── alembic.ini
├── docs/
│   └── backend-level docs plus handoff files
└── README.md
```

This keeps the ML pipeline and frontend intact while adding the real application backend.

## 5. Database design

Recommended database: SQLite for a self-contained/local deployment.

Core tables:
- `plants`
  - `id`
  - `name`
  - `species`
  - `variety`
  - `cluster`
  - `node_id`
  - `sampling_point`
  - `status`
  - `description`
  - `created_at`
  - `updated_at`

- `devices`
  - `id`
  - `name`
  - `type`
  - `node_id`
  - `ip_address`
  - `mac_address`
  - `status`
  - `last_heartbeat`
  - `last_transmission`

- `sensor_readings`
  - `id`
  - `record_id`
  - `timestamp`
  - `plant_id`
  - `node_id`
  - `sampling_point`
  - `session_id`
  - `soil`
  - `temperature`
  - `humidity`
  - `light`
  - `status`
  - `received_at`
  - `accepted_at`
  - `validation_error`
  - `created_at`

- `sessions`
  - `id`
  - `session_id`
  - `scheduled_at`
  - `started_at`
  - `ended_at`
  - `status`
  - `expected_records`
  - `received_records`
  - `rejected_records`
  - `trigger_type`

- `ml_predictions`
  - `id`
  - `plant_id`
  - `reading_timestamp`
  - `stress_label`
  - `p_healthy`
  - `p_low`
  - `p_moderate`
  - `p_high`
  - `cluster_id`
  - `cluster_distance`
  - `is_anomaly`
  - `anomaly_score`
  - `model_run_id`
  - `created_at`

- `anomaly_records`
  - `id`
  - `plant_id`
  - `reading_timestamp`
  - `score`
  - `threshold`
  - `is_anomaly`
  - `details`

- `recommendations`
  - `id`
  - `plant_id`
  - `title`
  - `category`
  - `priority`
  - `status`
  - `simple_action`
  - `reason`
  - `scientific_context`
  - `created_at`

Recommended indexes:
- `sensor_readings(plant_id, timestamp)`
- `sensor_readings(plant_id, node_id, timestamp)`
- `sensor_readings(session_id)`
- `ml_predictions(plant_id, reading_timestamp)`
- `recommendations(plant_id, created_at)`

## 6. API contract

The frontend pages imply these actor-facing API groups:

### Health
- `GET /api/health`

### Plants
- `GET /api/plants`
- `GET /api/plants/{plant_id}`

### Sensor ingestion
- `POST /api/sensors/readings`
- `GET /api/sensors/readings`

### Monitoring
- `GET /api/monitoring/overview`
- `GET /api/monitoring/plants/{plant_id}`

### Predictions
- `POST /api/predictions`
- `GET /api/predictions`

### AI and explainability
- `GET /api/ai/insights`
- `GET /api/ai/shap/{plant_id}`
- `GET /api/ai/anomalies`

### Recommendations
- `GET /api/recommendations`

### Devices
- `GET /api/devices`

### Sessions
- `GET /api/sessions`

### Data logs
- `GET /api/data-logs`

### Model performance
- `GET /api/models/performance`

### Example request payload
```json
{
  "timestamp": "2026-10-21T10:30:00+05:30",
  "plant_id": 3,
  "node_id": "1",
  "sampling_point": 6,
  "session_id": 4,
  "soil": 48.2,
  "temperature": 26.4,
  "humidity": 68.0,
  "light": 3200
}
```

### Example prediction response
```json
{
  "plant_id": 3,
  "timestamp": "2026-10-21T10:30:00+05:30",
  "stress_prediction": {
    "label": "healthy",
    "probabilities": {
      "healthy": 0.71,
      "low": 0.26,
      "moderate": 0.02,
      "high": 0.01
    }
  },
  "anomaly": {
    "is_anomaly": false,
    "score": -0.15,
    "threshold": 0.0
  },
  "warnings": []
}
```

## 7. ML integration

The ML pipeline already has a robust contract. The backend should not reimplement model training. It should:
- load the model from `money_plant_ml_pipeline/artifacts/...`
- call `POST /predict` on the ML service
- persist the returned metadata
- separately store raw sensor data and derived prediction outputs

Important compatibility points from the pipeline:
- Model artifacts are stored under versioned directories
- Model loading is done through config and artifact metadata, not training at startup
- Feature engineering is tied to a specific set of past-only sensor features
- History must be sent with the latest reading and same plant data
- Anomaly scores are higher is more anomalous and are not equivalent to plant stress
- The pipeline clearly warns that outputs are synthetic and should not be presented as empirical biological diagnoses

The backend should respect these boundaries and never misrepresent model outputs.

## 8. Sensor data ingestion

The existing frontend and pipeline imply a record schema with:
- `record_id`
- `timestamp`
- `session_id`
- `node_id`
- `sampling_point`
- `plant_id`
- `soil`
- `temperature`
- `humidity`
- `light`
- `status`
- `received_at`
- `accepted_at`
- `validation_error`
- `rssi`
- `crc_ok`

The backend should:
- validate required fields and ranges
- reject malformed payloads with 400 or 422
- accept a batch ingestion mode if ESP32 or simulator sends multiple readings
- deduplicate by stable identifiers when available
- log missing values without silently fabricating them

## 9. Frontend integration

The frontend already expects the following concepts:
- list of plants
- plant detail view
- stress status
- recommendation cards
- device health
- sessions
- monitoring graphs
- AI insight pages

Therefore the backend needs to return data in a shape that matches the frontend TypeScript types.

Goal:
- Replace mock data access in `frontend/src/api/mockData.ts` and/or `frontend/src/api/client.ts`
- Preserve the visual design and routing
- Use API responses for dashboards and details

## 10. Execution and setup

Recommended environment:
- Python 3.10+
- Node.js 18+
- SQLite
- FastAPI + Uvicorn
- Python model dependencies from `money_plant_ml_pipeline/requirements.txt` if available in the project

Recommended commands:
```bash
cd money_plant_ml_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn plant_ml.api.main:app --host 127.0.0.1 --port 8001
```

Backend service:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

## 11. Testing strategy

The backend should include:
- unit tests for validation and normalization
- API tests for ingestion and retrieval flows
- database tests for persistence and duplicate handling
- integration tests for the ML service call
- end-to-end tests for a full sensor-read -> prediction -> dashboard flow

Representative checks:
- invalid reading rejected
- missing plant history handled gracefully
- model unavailable returns 503
- mixed plant history is rejected
- successful prediction persisted and queryable
- recommendations generated correctly

## 12. Implementation checklist

1. Confirm repository conventions and actual frontend/ML API contracts
2. Add backend app folder and Python dependency manifest
3. Create SQLite schema and models
4. Implement sensor ingestion and validation
5. Integrate with the ML prediction service
6. Persist predictions and recommendations
7. Connect frontend to API layer
8. Add health checks and structured logging
9. Write automated tests
10. Verify end-to-end workflow

## Final assessment

The application branch already contains the primary user experience and machine learning contract. The missing piece is the actual backend service and persistence layer. The cleanest path is to build a FastAPI backend around SQLite and the already-implemented ML service, while keeping the frontend and ML package intact.

This is the most accurate and least disruptive integration strategy based on the code that exists in the repository today.
