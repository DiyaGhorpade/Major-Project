# BACKEND IMPLEMENTATION MASTER PROMPT
## Money Plant Monitoring System: one-click "Run" end-to-end build

> Paste this whole file into your coding agent (Claude Code, etc.) at the repo root.
> It is self-contained. Read it fully, then execute the phases in order.

---

## 0. ROLE AND MISSION

You are a senior full-stack engineer. In this repo you will build the missing application backend and wire the existing frontend to it, so that:

1. The user opens the dashboard and presses a single **Run** button.
2. The backend uses the **bundled dataset `plant_sensor_data.csv`** (5,376 rows, 16 plants) and nothing else. **No CSV upload UI, no file pickers, no "demo vs live" toggles, no mock data.**
3. The backend loads that data into SQLite, runs every reading through the **existing ML pipeline** (`money_plant_ml_pipeline`), stores predictions, anomalies, and recommendations, and the frontend shows all of it: plant status, charts, AI insights, SHAP, model performance, and **recommendations derived from the ML output**.
4. Everything works together on a fresh clone with 3 commands (or 1 script). No manual steps.

Do not redesign the UI. Preserve routes, layout, and styling. Replace the data layer only.

---

## 1. REPOSITORY FACTS (already established, do not re-derive)

```
.
├── frontend/                      React + Vite + TS + Tailwind (complete UI, mock data layer)
│   └── src/api/{client.ts, mockData.ts, exportCsv.ts}, src/types/index.ts, src/pages/*
├── money_plant_ml_pipeline/       Python + FastAPI ML service (port 8001), artifacts prebuilt
│   ├── config/config.yaml         data.provenance: synthetic, validated_on_real_plants: false
│   ├── artifacts/latest.json      + versioned dirs (classifier, kmeans, isolation forest, manifest)
│   ├── data/raw/plant_sensor_data.csv
│   └── src/plant_ml/api/{main,routes,schemas}.py
└── backend/                       DOES NOT EXIST, you create it (FastAPI, port 8000)
```

**Dataset** `plant_sensor_data.csv` columns:
`timestamp, session_id, node_id, sampling_point, plant_id, soil, temperature, humidity, light, stress_level, stress_type, is_synthetic_anomaly`
5,376 rows = 16 plants × 336 readings. Timestamps are Asia/Kolkata. Naive timestamps are assumed IST.

**ML service endpoints:** `GET /health`, `GET /metadata`, `POST /validate`, `POST /predict`, `POST /predict/batch`.
`POST /predict` body: `{reading:{timestamp,plant_id,soil,temperature,humidity,light}, history:[...same shape, same plant, strictly earlier...], explain:bool}`.
Response contains: `stress_prediction`, `cluster`, `anomaly`, `explanation`, `recommendations`, `history`, `warnings`, `model_metadata`.
422 = invalid input (mixed plants, history not earlier, too much history). 503 = artifacts unavailable.

**Frontend** pages: Overview, LiveMonitoring, Plants, PlantDetail, AiInsights, ShapAnalysis, ModelPerformance, Recommendations, DataLogs, DevicesNetwork, Sessions, Settings.
`ApiClient` methods to replace: `getSettings, updateSettings, getDevices, getPlants, getPlantById, getRecords, getLatestRecords, getSessions, getLocalShap, getGlobalShap, getModelMetrics, getRecommendations, updateRecommendationStatus, simulateLiveReadingPush`.

---

## 2. HARD RULES (non-negotiable)

1. **Dataset-only.** The single source of truth is `money_plant_ml_pipeline/data/raw/plant_sensor_data.csv`. The backend resolves this path itself (configurable via env `DATASET_PATH`, defaulting to that file). The frontend never sends data files.
2. **Do not retrain or modify the ML pipeline.** Load existing artifacts only. Do not reimplement features, SHAP, or recommendation logic in the backend. Call the ML service.
3. **Stateless ML contract.** Every `/predict` call sends the current reading plus **only that plant's** prior readings (strictly earlier timestamps, capped to the ML service's max history. Read it from `config.yaml`/`/metadata` and do not hardcode).
4. **Never present outputs as validated diagnoses.** Dataset provenance is synthetic. Surface a persistent, visible disclaimer in the UI (see §9.6).
5. **Anomaly ≠ unhealthy.** Isolation Forest score (higher = more anomalous) means "check the sensor/reading". **Cluster ID ≠ stress severity.** Do not map them to health in the UI.
6. **Graceful degradation.** If the ML service is down or returns 503, keep raw readings, mark predictions `unavailable`, show an error banner. Never crash ingestion.
7. **No fabricated values.** Missing/invalid values are logged and flagged, not silently filled.
8. **Idempotent.** Pressing Run twice must not duplicate rows. Dedupe on `(plant_id, timestamp)`.
9. **Be honest about gaps.** If something in the repo contradicts this prompt (file path, field name, endpoint shape), inspect the real code, follow the real code, and list the discrepancy in `backend/README.md` under "Deviations".

---

## 3. PHASE 0: INSPECT BEFORE BUILDING (do this first, report briefly)

Read these and write a short findings note (do not skip, the handoff docs are summaries and may be stale):

- `money_plant_ml_pipeline/docs/API_CONTRACT.md`, `src/plant_ml/api/{routes,schemas,main}.py`, `config/config.yaml` (history cap, timezone, thresholds, artifact dir)
- `artifacts/latest.json` and the newest artifact dir's manifest/metadata. Find where **classifier metrics, confusion matrix, per-class metrics, feature names** live (also check `reports/`). These feed Model Performance and Global SHAP.
- `frontend/src/types/index.ts`, `src/api/client.ts`, `src/api/mockData.ts`, `src/App.tsx`, and every page that calls `apiClient`. Note the exact TypeScript shapes the backend must return.
- The CSV itself: row count, distinct `session_id`, `node_id` values, `sampling_point` range, `stress_level`/`stress_type` value sets, timestamp range and cadence, `is_synthetic_anomaly` count, nulls, duplicates.

Resolve these known mismatches explicitly in a single mapping module (`backend/app/core/mappings.py`) and document it:

| Concept | Frontend | ML / CSV | Action |
|---|---|---|---|
| Stress category | Healthy / Water Stress / Light Stress / Nutrient Stress | classifier labels `healthy/low/moderate/high` (severity); CSV `stress_type` (type) | Keep **both** axes: severity from ML, type from `stress_type`/heuristics. Map to the frontend `stressType` using CSV values found in Phase 0. Do not invent a type the data doesn't carry. |
| `node_id` | string in some payload examples | integer in CSV | Normalize to one type matching `types/index.ts`. |
| Node ↔ plant | Node 1 = pots 1–12, Node 2 = pots 13–16 | `node_id` column | Verify against CSV; the CSV wins. |
| Clusters | 4 clusters of plants | `experimental_group`/KMeans cluster | Verify what "cluster" means in `Plant` type; do not conflate with KMeans env-state IDs. |

---

## 4. TARGET ARCHITECTURE

```
Browser ──► Vite dev (5173, proxy /api → 8000)
              │
              ▼
        backend (FastAPI :8000)  ── SQLite (backend/data/app.db)
              │      ▲
   CSV (read-only)   │ httpx
              ▼      │
        money_plant_ml_pipeline (FastAPI :8001, artifacts loaded at startup)
```

Backend layout (follow BACKEND_HANDOFF §4; use SQLAlchemy 2.x + Pydantic v2, httpx, pytest):

### 4.1 Complete folder structure (create exactly this)

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                       App factory, CORS (localhost:5173), lifespan: create tables → ensure_seeded()
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                   DB session dependency, ML client dependency, pagination params
│   │   └── routes/
│   │       ├── __init__.py           Registers all routers under /api
│   │       ├── health.py             GET /api/health
│   │       ├── pipeline.py           POST /api/pipeline/run, GET /api/pipeline/status
│   │       ├── plants.py             GET /api/plants, /api/plants/{id}
│   │       ├── sensors.py            GET/POST /api/sensors/readings
│   │       ├── data_logs.py          GET /api/data-logs (+ ?format=csv export)
│   │       ├── monitoring.py         GET /api/monitoring/overview, /plants/{id}
│   │       ├── predictions.py        GET/POST /api/predictions
│   │       ├── ai.py                 GET /api/ai/insights, /shap/{plant_id}, /shap/global, /anomalies
│   │       ├── recommendations.py    GET /api/recommendations, PATCH /{id}
│   │       ├── devices.py            GET /api/devices
│   │       ├── sessions.py           GET /api/sessions
│   │       ├── models.py             GET /api/models/performance
│   │       └── settings.py           GET/PUT /api/settings
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                 Pydantic Settings: ML_SERVICE_URL, DATASET_PATH, DATABASE_URL, ML_TIMEOUT_S, ML_CONCURRENCY, EXPLAIN_MODE
│   │   ├── logging.py                Structured logging + request-ID middleware
│   │   ├── errors.py                 Error envelope {error:{code,message,details}} + exception handlers
│   │   └── mappings.py               Stress-type/severity mapping, node_id normalisation, node↔plant map (see §3)
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py               Engine, SessionLocal, SQLite WAL pragma, Base
│   │   ├── models.py                 plants, devices, sensor_readings, sessions, ml_predictions,
│   │   │                             anomaly_records, recommendations, pipeline_runs, model_info, settings
│   │   ├── repositories.py           Query helpers: readings by plant (ordered), latest per plant, bulk upsert, aggregates
│   │   └── migrations/               Alembic env + versions (initial schema)
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── ml_client.py              httpx async client: health, metadata, predict, predict_batch; retry/backoff; 422/503 handling
│   │   └── ml_types.py               Pydantic models mirroring ML request/response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── seed_service.py           Read CSV → validate → normalise IST → idempotent insert of plants/devices/sessions/readings
│   │   ├── history_builder.py        Per-plant, strictly-earlier, capped history for /predict
│   │   ├── pipeline_service.py       Run orchestration: preflight, per-plant walk, concurrency, progress updates, persist
│   │   ├── prediction_service.py     Single on-demand prediction + persistence of ML output
│   │   ├── recommendation_service.py ML recs + rule-based recs, dedupe, expiry, preserve user status
│   │   ├── monitoring_service.py     Overview + plant-detail aggregates (cached per run)
│   │   ├── ai_service.py             Insights, local/global SHAP shaping, anomalies list
│   │   ├── model_service.py          Model performance from artifacts/reports + /metadata
│   │   ├── device_service.py         Derive Jetson/ESP32 status from data (null for unknowns)
│   │   ├── session_service.py        Expected vs received counts, node progress
│   │   └── export_service.py         Server-side CSV generation for data logs
│   └── schemas/
│       ├── __init__.py
│       ├── common.py                 Pagination, error envelope, enums
│       ├── plant.py                  Plant, PlantDetail
│       ├── sensor.py                 SensorRecord, ReadingIn (single + batch)
│       ├── prediction.py             Prediction, Anomaly, Explanation
│       ├── recommendation.py         Recommendation, RecommendationStatusUpdate
│       ├── dashboard.py              OverviewResponse, PlantMonitoringResponse
│       ├── device.py                 Device
│       ├── session.py                Session
│       ├── model.py                  ModelMetrics, GlobalShap, LocalShap
│       ├── pipeline.py               PipelineRunResponse, PipelineStatus
│       └── settings.py               SystemSettings
├── tests/
│   ├── conftest.py                   Temp SQLite DB, fake ML client, TestClient fixtures
│   ├── test_seed.py                  Idempotent seed, bad-row flagging
│   ├── test_history_builder.py       Per-plant only, earlier-only, cap, cold-start
│   ├── test_ml_client.py             422 → rejected; 503/timeout → retry → unavailable
│   ├── test_pipeline.py              Run, concurrent-run guard, reset keeps user rec status
│   ├── test_recommendations.py       Priority rules, anomaly → "Sensor check", dedupe
│   ├── test_ingestion.py             Invalid payload, duplicate, batch
│   ├── test_api_shapes.py            Responses validate against schemas matching frontend types
│   ├── test_ml_integration.py        Real ML service (skipped if unreachable)
│   └── test_e2e.py                   seed → run → overview has 16 plants + recommendations
├── data/                             app.db created here at runtime (gitignored)
├── .env.example
├── .gitignore
├── alembic.ini
├── requirements.txt                  fastapi, uvicorn[standard], sqlalchemy>=2, alembic, pydantic>=2,
│                                     pydantic-settings, httpx, pandas, pytest, pytest-asyncio
└── README.md                         Setup, run, endpoints, Deviations section
```

### 4.2 Repo-root additions

```text
scripts/
├── dev.sh                            Venvs, deps, start ML :8001 → wait /health → backend :8000 → frontend :5173
└── dev.ps1                           Windows equivalent
Makefile                              setup | dev | test | seed | reset-db
README.md                             Quick start: ./scripts/dev.sh → open URL → press Run

frontend/src/                         (changes only)
├── api/
│   ├── client.ts                     REWRITE: thin typed fetch client against /api
│   ├── transformers.ts               Only if backend shapes can't match types directly
│   └── mockData.ts                   DELETE (and every import of it)
├── hooks/
│   ├── usePipeline.ts                Run + status polling state machine
│   ├── usePlants.ts
│   ├── useReadings.ts
│   ├── useRecommendations.ts
│   └── useSettings.ts
├── components/
│   ├── layout/TopBar.tsx             ADD Run button + progress bar
│   └── shared/
│       ├── SyntheticBanner.tsx       Persistent "synthetic dataset" disclaimer
│       ├── ErrorBanner.tsx
│       └── EmptyState.tsx            "Press Run to analyze the bundled dataset"
└── pages/SettingsPage.tsx            REMOVE demo/live toggle + API-URL field
vite.config.ts                        ADD server.proxy: /api → http://127.0.0.1:8000
```

### 4.3 File-level rules

- Routes stay thin: validate input, call a service, return a schema. No SQL or ML logic in route files.
- Services own business logic; `repositories.py` owns all DB queries; `ml_client.py` is the only code that talks to the ML service.
- Every response model in `schemas/` must match `frontend/src/types/index.ts` field-for-field (use aliases if the TS types are camelCase).
- No file in `backend/` may import from `money_plant_ml_pipeline`. Integration is HTTP only.

Config via env (`.env.example`): `ML_SERVICE_URL=http://127.0.0.1:8001`, `DATASET_PATH`, `DATABASE_URL=sqlite:///./data/app.db`, `ML_TIMEOUT_S=10`, `ML_CONCURRENCY=4`, `EXPLAIN_MODE=latest_only`.

---

## 5. DATABASE

Use the tables from BACKEND_HANDOFF §5 (`plants, devices, sensor_readings, sessions, ml_predictions, anomaly_records, recommendations`) with these additions/clarifications:

- `sensor_readings`: unique constraint `(plant_id, timestamp)`; keep `stress_level`, `stress_type`, `is_synthetic_anomaly` as **dataset labels** (nullable columns named `label_*`), clearly separate from ML output. Never mix them in API responses without labeling which is which.
- `ml_predictions`: add `reading_id` FK, `stress_type_guess` (nullable), `status` (`ok|unavailable|rejected`), `error`, `warnings` (JSON), `model_version`, `cluster_id`, `explanation_json` (nullable), `created_at`. Unique on `reading_id`.
- `recommendations`: add `source` (`ml|heuristic`), `prediction_id` FK, `status` (`new|in_progress|completed`), unique-ish key `(plant_id, category, source_reading_id)` to prevent duplicates on re-run. Status changes made by the user must **survive a re-run**.
- `pipeline_runs`: `id, started_at, finished_at, status (queued|running|completed|failed), total, processed, failed, current_plant_id, message`.
- `model_info` (or cached JSON): model version, trained_at, provenance, metrics, confusion matrix, labels, feature importance.
- Indexes per BACKEND_HANDOFF §5.

Plants (16) and devices (Jetson + ESP32 nodes) are derived from the dataset + config at seed time. Device fields that the dataset can't provide (RSSI, firmware, uptime) must be derived honestly: compute from available data (last reading time, counts) and return `null` for the rest. Do not invent fake RSSI/firmware. Update the Devices page to render `null` as "n/a".

---

## 6. THE "RUN" FLOW (core of the whole task)

### 6.1 Startup (automatic, no user action)
Lifespan hook: create tables → `seed_service.ensure_seeded()`:
- Read CSV with pandas, validate (reuse ML `/validate` for a report and also do local range checks: soil 0–100, temp 15–40, humidity 20–100, light 0–100000), normalize timestamps to IST, sort by `timestamp, plant_id`.
- Bulk-insert plants, devices, sessions, readings (**raw data only. No predictions yet**). Idempotent via the unique key.
- So Data Logs, Sessions, Plants, and Devices can already show real data before pressing Run, and `GET /api/pipeline/status` reports `idle` / `ready_to_run`.

### 6.2 `POST /api/pipeline/run`
Returns immediately with a `run_id` (202). If a run is already active, return that run's id (no double-run). Body optional: `{ "reset": false }`. `reset=true` clears predictions/anomalies/ML-generated recommendations (not user status edits) and recomputes.

Background task (`asyncio` task or FastAPI `BackgroundTasks`; no Celery/Redis):
1. Pre-flight: call ML `/health` and `/metadata`. If unavailable → run `failed` with a clear message; the frontend shows it. Cache model metadata into `model_info`.
2. Group readings by plant, ordered by time. For each plant, walk chronologically; for reading *i*, history = readings `[max(0,i-cap) .. i-1]` of the **same plant**. Prefer `POST /predict/batch` if its contract allows per-item histories; otherwise `/predict` with bounded concurrency (`ML_CONCURRENCY`). Never put two plants in one request.
3. `explain=false` for bulk; run `explain=true` only for the **latest reading per plant** (so SHAP pages have data without a 5k-row cost).
4. Persist per reading: prediction, anomaly record, warnings. On 422 mark `rejected` with the reason. On 503/timeouts, retry with backoff (3 tries), then mark `unavailable`. Continue with the rest.
5. Update `pipeline_runs.processed/failed/current_plant_id` every N rows (commit in batches, e.g. 100).
6. After predictions: `recommendation_service` builds recommendations (§7). Compute aggregates (cached per run).
7. Mark `completed`.

### 6.3 `GET /api/pipeline/status`
`{ runId, state, total, processed, failed, percent, currentPlantId, startedAt, finishedAt, message, mlServiceOk, modelVersion }`

### 6.4 Frontend Run button
- Put a prominent **Run** button in the `TopBar` (and the Overview empty state). Click → `POST /api/pipeline/run` → poll `/api/pipeline/status` every ~700 ms → show a progress bar ("Analyzing 3,120 / 5,376 readings…") → when `completed`, invalidate/refetch all page data. Auto-navigate to Overview if the user is elsewhere (optional, no hard redirect).
- Disabled while running. Show a "Re-run" label after the first completion.
- While running, pages should show partial results as they land (poll-driven), not block.
- If the run `failed`, show the error banner with the cause (e.g. "ML service not reachable at :8001").

### 6.5 Optional live-replay feel (nice-to-have, after everything works)
Add `speed` option to replay readings in timestamp order so LiveMonitoring animates. Keep it behind `POST /api/pipeline/run?mode=replay`. Default mode must be the fast bulk run above. Do not let this delay the core deliverable.

---

## 7. RECOMMENDATIONS (must be visibly driven by ML output)

The ML response already includes `recommendations` and `warnings`. Build on those:

- **Source 1: ML recommendations** from the latest prediction per plant. Store as `source='ml'`, preserving the ML text, mapped to the frontend `Recommendation` shape: `title, category, priority, simple_action, reason, scientific_context`.
- **Source 2: backend rules**, derived from the ML *outputs* plus recent trend, used only to fill gaps and set priority:
  - `high` stress or sustained `moderate` (e.g. ≥3 consecutive latest readings) → priority **high/critical**.
  - `is_anomaly` true → recommendation category **"Sensor check"** ("Verify sensor/placement/reading"), explicitly **not** a plant-health claim.
  - Stress type (from dataset/ML) → category: water (soil), light, nutrient. Reasons must cite the actual recent sensor values and the ML probability, e.g. "Soil moisture 21% (below its recent baseline); model P(high)=0.64".
  - Cold-start plants (short history) → note "lower reliability".
- Dedupe per `(plant_id, category)` for the latest state; resolved conditions should auto-expire (status `completed`/`superseded`) only if the user hasn't touched them.
- Each recommendation carries `confidence` (model probability), `basedOn` (reading timestamp + model version), and a `disclaimer` flag.
- `PATCH /api/recommendations/{id}` `{status}` persists to DB (replaces `updateRecommendationStatus` mock).

Frontend: Recommendations page, Plant Detail recommendation panel, and Overview "recent alerts / top actions" must all read these. Show priority badges, the reason line with real numbers, the ML probability, and the synthetic-data disclaimer.

---

## 8. API CONTRACT (implement; response shapes must match `frontend/src/types/index.ts`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | backend + DB + ML-service reachability + model version |
| POST | `/api/pipeline/run` | start run |
| GET | `/api/pipeline/status` | progress |
| GET | `/api/plants`, `/api/plants/{id}` | catalog + latest reading + latest prediction/status |
| GET | `/api/sensors/readings` | paginated, filters: plant, cluster, node, status, session, date range, `limit/offset`, `order` |
| POST | `/api/sensors/readings` | single/batch ingestion for future ESP32 (validate, dedupe, 400/422 on bad payload, trigger prediction best-effort) |
| GET | `/api/data-logs` (+ `?format=csv`) | log view + **server-side CSV export** (export is download, not upload) |
| GET | `/api/monitoring/overview` | totals, status counts, per-cluster summaries, recent alerts, top recommendations |
| GET | `/api/monitoring/plants/{id}` | history series (soil/temp/humidity/light), prediction history, recs |
| GET/POST | `/api/predictions` | list; POST = on-demand single prediction for one plant (reading + auto-fetched history) |
| GET | `/api/ai/insights` | aggregated class counts, anomaly counts, feature usage, model status |
| GET | `/api/ai/shap/{plant_id}` | local SHAP (from latest `explain=true`); `404`/empty-state payload if unavailable |
| GET | `/api/ai/shap/global` | global feature importance from artifacts/reports |
| GET | `/api/ai/anomalies` | anomaly records |
| GET | `/api/recommendations`, PATCH `/api/recommendations/{id}` | list + status update |
| GET | `/api/devices`, `/api/sessions` | derived from data |
| GET | `/api/models/performance` | accuracy/precision/recall/F1, per-class, confusion matrix, labels, version, eval date, **provenance=synthetic** |
| GET/PUT | `/api/settings` | polling interval, optional pH toggle, etc. (persisted in DB; **no demo-mode, no API-URL-as-data-source**) |

Rules: all list endpoints paginated and bounded; consistent error envelope `{error:{code,message,details}}`; timestamps ISO-8601 with `+05:30`; CORS for the Vite origin; OpenAPI at `/docs`.

---

## 9. FRONTEND CHANGES

1. **Rewrite `src/api/client.ts`** to a thin typed fetch client against `/api` (Vite proxy → `:8000`). Add `liveApi.ts`/`transformers.ts` only if backend shapes can't match `types/index.ts` directly. Prefer matching the backend to the types.
2. **Delete the mock layer** from production paths: remove `mockData.ts` usage, `simulateLiveReadingPush`, the localStorage-seeded datasets, and the **demo/live toggle and API-base-URL fields** in `SettingsPage`. Keep `exportCsv.ts` only if still needed. Prefer the backend's `?format=csv` export.
3. **Remove any CSV upload/import UI** anywhere. The app never asks for a dataset.
4. Add a small data hook layer (`usePlants`, `useReadings`, `useRecommendations`, `usePipeline`) with loading / empty / error / stale-data states per BRIEF §12 of the frontend handoff. Pre-run empty state: "Press **Run** to analyze the bundled plant dataset."
5. **Run button + progress UI** per §6.4.
6. **Pages**: wire every page to real endpoints; check chart components against real data shapes (sparklines, time axes in IST).
7. **Disclaimers (required):**
   - Persistent banner/footer chip: "Synthetic dataset. Model outputs are experimental, not validated plant diagnoses."
   - Anomaly tooltip: "Unusual reading, check the sensor. Not a plant-health verdict."
   - Cluster label tooltip: "Environmental state group. Not a stress level."
   - Recommendation cards: "Advisory heuristic."
   - Model Performance page: "Metrics measured on synthetic labels."
8. Show model version + "AI model connected / unreachable" in AiInsights from `/api/health`.
9. Keep all routes, navigation, and styling unchanged.

---

## 10. DEV EXPERIENCE (must "just work")

Provide at repo root:
- `scripts/dev.sh` (and `scripts/dev.ps1`): create venvs if missing, install deps, start ML service (`PYTHONPATH=src uvicorn plant_ml.api.main:app --port 8001`), wait for `/health`, start backend (`:8000`), start frontend (`npm run dev`). Ctrl-C stops all.
- `Makefile` targets: `setup`, `dev`, `test`, `seed`, `reset-db`.
- Vite `server.proxy` for `/api`.
- Root `README.md` "Quick start": clone → `./scripts/dev.sh` → open the URL → press **Run**.
- If ML artifacts are missing, `dev.sh` must tell the user to run `python scripts/train_all.py --input data/raw/plant_sensor_data.csv` (training is an explicit step, never automatic at startup).

---

## 11. PERFORMANCE AND ROBUSTNESS TARGETS

- Full run of 5,376 readings completes in a reasonable time on CPU (target: a few minutes at most; use batch endpoint/concurrency; measure and report the actual time).
- API list endpoints respond in <300 ms on the seeded DB (indexes, pagination, cached aggregates).
- SQLite: enable WAL, `check_same_thread=False` handling, short transactions, batched commits.
- Structured logging with a request ID; log pipeline progress and every rejected/unavailable reading with reason.
- Never leak stack traces to clients.

---

## 12. TESTING (write real tests, run them, report results)

Backend (`pytest`, httpx `TestClient`; mock the ML service for unit tests, run one real-ML integration test if the service is available):
- CSV seed is idempotent (run twice → 5,376 rows, no dupes); bad rows are flagged.
- Ingestion: invalid payload → 4xx; duplicate ignored; batch mode.
- History builder: per-plant only, strictly earlier, capped; cold-start case.
- ML client: 422 → `rejected`; 503/timeout → retry → `unavailable`; ingestion still succeeds.
- Pipeline run: second concurrent POST returns the same run; `reset` preserves user rec status.
- Recommendation rules: high stress → high priority; anomaly → "Sensor check" not health claim; dedupe.
- API shape tests: each response validates against Pydantic models mirroring the TS types.
- E2E: seed → run → `/monitoring/overview` has 16 plants and non-empty recommendations.

Frontend: type-check (`tsc --noEmit`), lint, build; tests for loading/empty/error states and the Run button state machine (idle → running → completed / failed).

**Acceptance gate (all must pass before you say done):**
1. Fresh clone + `./scripts/dev.sh` starts all 3 services with no manual edits.
2. Before Run: Plants/Data Logs/Sessions show real dataset rows; AI pages show the "Press Run" empty state.
3. Press **Run** → progress bar → completes → Overview, Live Monitoring, Plants, Plant Detail, AI Insights, SHAP, Model Performance, Recommendations all show real, ML-derived content.
4. Recommendations reference actual sensor values and model probabilities, and carry the disclaimer.
5. Killing the ML service mid-run yields a clear failed/degraded state, not a crash. Restarting and re-running recovers.
6. No upload control, no demo toggle, no mock data import remains (`grep` to prove it).
7. Running twice does not duplicate data.
8. Tests + type-check + build all green.

---

## 13. EXECUTION ORDER

1. **Phase 0**: inspect (§3), write findings + mapping decisions.
2. **Phase 1**: backend skeleton, config, DB models, seeding; verify the seed.
3. **Phase 2**: ML client + pipeline run + status; verify against the real ML service on a few plants, then all.
4. **Phase 3**: read APIs (plants, monitoring, readings, data-logs, sessions, devices, models, AI, recommendations).
5. **Phase 4**: frontend data layer swap, Run button + progress, remove mock/demo/upload, disclaimers.
6. **Phase 5**: dev scripts, README, tests, acceptance gate (§12).
7. **Final report** (≤ 15 lines): what was built, how to start, measured run time, deviations from this prompt, known limitations. Do not claim anything you did not verify by running it.

Work autonomously. Ask the user only if a decision is truly blocking and cannot be resolved from the repo. Otherwise pick the sensible default and record it under "Deviations" in `backend/README.md`.
