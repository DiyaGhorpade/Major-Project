# Backend Implementation Master Prompt

Read the repository carefully before making changes. This project has a frontend in `frontend/` and an ML pipeline in `money_plant_ml_pipeline/`. The application branch currently does not include a dedicated backend service. The frontend uses a structured UI and the ML pipeline already exposes a FastAPI contract. Use that as the source of truth.

## Mandatory repository review

Before implementing, inspect:
- `frontend/src/App.tsx`
- `frontend/src/pages/*`
- `frontend/src/api/client.ts`
- `frontend/src/api/mockData.ts`
- `frontend/src/types/index.ts`
- `money_plant_ml_pipeline/README.md`
- `money_plant_ml_pipeline/config/config.yaml`
- `money_plant_ml_pipeline/src/plant_ml/api/main.py`
- `money_plant_ml_pipeline/src/plant_ml/api/routes.py`
- `money_plant_ml_pipeline/src/plant_ml/api/schemas.py`
- `money_plant_ml_pipeline/docs/API_CONTRACT.md`

Do not assume the frontend and ML service are enough by themselves. Verify every assumption against the real repository before writing code.

## Architecture target

Build a production-minded Python backend using FastAPI and SQLite, designed to sit between the frontend and the existing ML service.

Do not duplicate the ML training or feature engineering logic. Reuse the machine-learning pipeline exactly as-is.

The backend must:
- accept and validate sensor readings
- store raw readings
- query historical readings
- call the ML service for predictions when appropriate
- persist prediction results
- serve dashboard, plant, device, session, and recommendation data
- expose endpoints required by the frontend
- return structured error payloads
- provide health checking and structured logs

## Required modules

Create the backend app following a maintainable layered design:
- `app/api/routes/...`
- `app/core/config.py`
- `app/core/logging.py`
- `app/db/database.py`
- `app/db/models.py`
- `app/db/repositories.py`
- `app/services/ingestion_service.py`
- `app/services/plant_service.py`
- `app/services/device_service.py`
- `app/services/prediction_service.py`
- `app/services/monitoring_service.py`
- `app/services/recommendation_service.py`
- `app/integrations/ml_client.py`
- `app/schemas/...`

Use environment variables for configuration, not hardcoded paths or values.

## Data model requirements

Implement database tables that reflect the actual frontend and ML pipeline requirements:
- plants
- devices
- sensor_readings
- sessions
- ml_predictions
- anomaly_records
- recommendations

Preserve important fields such as:
- `record_id`
- `plant_id`
- `node_id`
- `sampling_point`
- `session_id`
- `timestamp`
- `status`
- `received_at`
- `accepted_at`

Use SQLite unless there is a strong reason not to. For this project, single-device/local deployment is the likely fit.

## API contract requirements

Implement the endpoints the frontend expects:
- `GET /api/health`
- `GET /api/plants`
- `GET /api/plants/{plant_id}`
- `POST /api/sensors/readings`
- `GET /api/sensors/readings`
- `GET /api/monitoring/overview`
- `GET /api/monitoring/plants/{plant_id}`
- `POST /api/predictions`
- `GET /api/predictions`
- `GET /api/ai/insights`
- `GET /api/ai/shap/{plant_id}`
- `GET /api/ai/anomalies`
- `GET /api/recommendations`
- `GET /api/devices`
- `GET /api/sessions`
- `GET /api/data-logs`
- `GET /api/models/performance`

Match the frontend’s expected semantics and use consistent JSON responses.

Validate:
- missing required fields
- malformed timestamps
- invalid numeric ranges
- mixed plant history in predictions
- duplicate readings when stable identifiers exist
- blocked DB/model errors with proper status codes

## ML integration requirements

Do not replace or retrain models. Use the existing ML pipeline service:
- keep the pipeline under `money_plant_ml_pipeline`
- load its metadata and artifact version information
- call the model endpoint using the correct request format
- store all prediction results
- map prediction output to frontend-friendly response shapes

Important constraints:
- do not claim biological certainty
- do not fabricate model performance metrics
- do not convert anomaly flags into diagnosis
- preserve the distinction between raw observations and ML-derived recommendations

If the ML service is unavailable, return a clean 503-style error and do not crash the application.

## Frontend integration requirements

Once the backend is running:
- update the frontend API client and remove or isolate mock-data dependencies
- ensure the dashboard pages use backend data
- preserve the existing visual style and route structure
- ensure pages handle loading, empty, and error states gracefully

The frontend likely uses strongly typed data models defined in `frontend/src/types/index.ts`; align backend responses to those types.

## Operational requirements

Implement:
- health checks
- structured logging
- error handling for DB failure, malformed payloads, and missing models
- config via environment variables
- database initialization and migration support
- test fixtures for representative reads and predictions

## Testing requirements

Write automated tests covering:
- sensor payload validation
- successful storage of readings
- read history retrieval
- prediction call to ML service
- empty and invalid responses
- database warnings and failure cases
- API route coverage

Do not claim tests passed unless they are actually executed.

## Documentation requirements

Update or add docs when the implementation detail changes. The docs should include:
- setup instructions
- environment variables
- backend API contract
- database overview
- ML integration notes

## Final reporting requirements

When finished, report:
- files created or modified
- endpoints implemented
- tests executed
- test results
- remaining limitations and blockers

## Success criteria

The application is complete when:
- a valid sensor reading can be ingested and stored
- the reading can be retrieved through a historical API
- the backend can trigger supported ML inference
- predictions are persisted and retrievable
- the resulting data appears in the frontend pages
- the system behaves sensibly when the model is unavailable or the data is malformed
