# Money Plant Monitoring System - Frontend Handoff

## 1. Objective

This document describes the current frontend architecture, routing, data model, mock-data layer, and all required backend integrations needed to make the application fully functional end-to-end.

The frontend is a React + Vite dashboard for monitoring 16 tulsi plants across four stress-cluster groups. It presents live or simulated sensor values, plant details, SHAP explainability, model metrics, sessions, recommendations, devices, and settings.

## 2. Repository context

### Application structure

```text
frontend/
├── src/
│   ├── App.tsx
│   ├── App.css
│   ├── index.css
│   ├── main.tsx
│   ├── api/
│   │   ├── client.ts
│   │   ├── mockData.ts
│   │   └── exportCsv.ts
│   ├── components/
│   │   ├── charts/
│   │   ├── layout/
│   │   └── shared/
│   ├── lib/
│   │   └── utils.ts
│   ├── pages/
│   │   ├── OverviewPage.tsx
│   │   ├── LiveMonitoringPage.tsx
│   │   ├── PlantsPage.tsx
│   │   ├── PlantDetailPage.tsx
│   │   ├── AiInsightsPage.tsx
│   │   ├── ShapAnalysisPage.tsx
│   │   ├── ModelPerformancePage.tsx
│   │   ├── RecommendationsPage.tsx
│   │   ├── DataLogsPage.tsx
│   │   ├── DevicesNetworkPage.tsx
│   │   ├── SessionsPage.tsx
│   │   ├── SettingsPage.tsx
│   │   └── ...
│   └── types/
│       └── index.ts
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── index.html
└── README.md
```

## 3. Frontend architecture

### App shell and navigation

- `frontend/src/App.tsx` is the application entry point.
- `frontend/src/components/layout/Sidebar.tsx` defines navigation routes.
- `frontend/src/components/layout/TopBar.tsx` handles app header and action area.
- `frontend/src/components/layout/AppShell.tsx` wraps pages in the main shell.

Navigation likely includes pages for:
- overview
- live monitoring
- plants
- plant detail
- AI insights
- SHAP analysis
- model performance
- recommendations
- data logs
- devices & network
- sessions
- settings

## 4. Data model

The frontend types are centralized in:
- `frontend/src/types/index.ts`

Core data contracts include:
- `Plant`
- `SensorRecord`
- `Session`
- `Device`
- `LocalShapExplanation`
- `GlobalShapData`
- `ModelMetrics`
- `Recommendation`
- `SystemSettings`

Important domain fields:
- `plant.id`
- `plant.nodeId`
- `plant.samplingPoint`
- `plant.cluster`
- `plant.stressType`
- `record_id`
- `session_id`
- `timestamp`
- `soil`
- `temperature`
- `humidity`
- `light`
- `status`
- `accepted_at`
- `validation_error`

## 5. Frontend pages and responsibilities

### 5.1 Overview page
File: `frontend/src/pages/OverviewPage.tsx`

Purpose:
- dashboard summary
- aggregate plant health
- overall telemetry state
- recent risk indicators

Required backend data:
- total plant count
- plant health summary by status
- latest readings per plant
- recent alerts
- summary metrics by cluster

### 5.2 Live Monitoring page
File: `frontend/src/pages/LiveMonitoringPage.tsx`

Purpose:
- live or near-real-time sensor stream
- sparkline charts
- toggles for poll intervals
- monitoring of plant readings

Required backend data:
- most recent readings for all plants or selected plants
- historical time series per metric
- connection status and refresh behavior
- loading/empty/error states

### 5.3 Plants page
File: `frontend/src/pages/PlantsPage.tsx`

Purpose:
- searchable/filterable list of all 16 plants
- cluster filter
- node filter
- grid/list toggle

Required backend data:
- plant catalog
- latest reading for each plant
- cluster mapping and plant status

### 5.4 Plant Detail page
File: `frontend/src/pages/PlantDetailPage.tsx`

Purpose:
- detailed per-plant diagnostics
- current status
- historical trend display
- prediction panel
- explainability and recommendation details

Required backend data:
- specific plant record
- historical sensor readings for that plant
- prediction history
- recommendation(s)
- SHAP summary if available

### 5.5 AI Insights page
File: `frontend/src/pages/AiInsightsPage.tsx`

Purpose:
- risk overview from ML classification
- feature usage summaries
- “AI model connected” status

Required backend data:
- aggregated prediction counts
- explanation summaries
- model status metadata

### 5.6 SHAP Analysis page
File: `frontend/src/pages/ShapAnalysisPage.tsx`

Purpose:
- local SHAP waterfall
- global SHAP ranking
- beeswarm interpretation view

Required backend data:
- `GET /api/ai/shap/{plant_id}`
- global feature importance data
- feature contribution arrays

### 5.7 Model Performance page
File: `frontend/src/pages/ModelPerformancePage.tsx`

Purpose:
- accuracy/precision/F1 metrics
- per-class metrics
- confusion matrix display

Required backend data:
- model metrics, confusion matrix, labels
- model version and evaluation date

### 5.8 Recommendations page
File: `frontend/src/pages/RecommendationsPage.tsx`

Purpose:
- actionable suggestions
- status transitions (new/in progress/completed)
- category and priority display

Required backend data:
- recommendations list
- recommendation details and statuses
- per-plant metadata

### 5.9 Data Logs page
File: `frontend/src/pages/DataLogsPage.tsx`

Purpose:
- searchable historical records
- filters and CSV export
- detail drawer

Required backend data:
- paginated sensor read history
- record filtering by plant, status, cluster, node
- export endpoints or generated CSV from backend

### 5.10 Devices and Network page
File: `frontend/src/pages/DevicesNetworkPage.tsx`

Purpose:
- Jetson and ESP32 device health
- last heartbeat/transmission
- connectivity and metadata

Required backend data:
- device list
- node health, status, uptime, RSSI, firmware versions

### 5.11 Sessions page
File: `frontend/src/pages/SessionsPage.tsx`

Purpose:
- scheduled data collection metadata
- expected vs received counts
- node progress

Required backend data:
- sessions list
- node progress map
- per-session statistics

### 5.12 Settings page
File: `frontend/src/pages/SettingsPage.tsx`

Purpose:
- API base URL
- demo/live mode switch
- polling interval
- optional pH sensor toggle
- device settings

Required backend data:
- app configuration values
- remote API settings
- demo/live toggle

## 6. Mock data layer

The frontend currently uses mock data extensively:
- `frontend/src/api/mockData.ts`
- `frontend/src/api/client.ts`

This layer contains:
- plant catalog and synthetic plant health states
- generated sensor records
- mock sessions
- mock SHAP explanations
- mock model metrics
- mock recommendations
- default system settings

Current pattern:
- `ApiClient` keeps in-memory datasets
- data is loaded from `localStorage` when available
- mock mode is enabled by default
- live backend fetch is attempted only when `demo_mode` is false

This is a functional demo layer, but it must be replaced or clearly isolated from production behavior once a real backend is implemented.

## 7. API client behavior

File: `frontend/src/api/client.ts`

Responsibilities:
- provide a typed interface for plant and sensor data
- maintain demo-mode fallback logic
- simulate live reading pushes
- provide recommendation updates
- persist settings in localStorage

Current methods include:
- `getSettings()`
- `updateSettings()`
- `getDevices()`
- `getPlants()`
- `getPlantById()`
- `getRecords()`
- `getLatestRecords()`
- `getSessions()`
- `getLocalShap()`
- `getGlobalShap()`
- `getModelMetrics()`
- `getRecommendations()`
- `updateRecommendationStatus()`
- `simulateLiveReadingPush()`

## 8. Current frontend assumptions

The UI is designed around this mental model:

- There are 16 plants
- They belong to four clusters
- Node 1 covers pots 1-12
- Node 2 covers pots 13-16
- Sensor values include soil, temperature, humidity, and light
- Stress categories are:
  - Healthy
  - Water Stress
  - Light Stress
  - Nutrient Stress
- SHAP and model metrics are available and described in a user-friendly way
- Recommendations are derived from heuristics and sensor thresholds

These assumptions are encoded in the mock datasets and should be preserved when wiring to a real backend.

## 9. Required backend integration points

The frontend currently expects these API contracts:

### Plant and catalog data
- `GET /api/plants`
- `GET /api/plants/{plant_id}`

### Sensor history and ingestion
- `GET /api/sensors/readings`
- `POST /api/sensors/readings`

### Dashboard and monitoring
- `GET /api/monitoring/overview`
- `GET /api/monitoring/plants/{plant_id}`

### Predictions and AI
- `POST /api/predictions`
- `GET /api/predictions`
- `GET /api/ai/insights`
- `GET /api/ai/shap/{plant_id}`
- `GET /api/ai/anomalies`

### Recommendations and settings
- `GET /api/recommendations`
- `GET /api/devices`
- `GET /api/sessions`
- `GET /api/models/performance`

### Health
- `GET /api/health`

The frontend should remain UI-first and not care whether the source is demo mode or real backend as long as the contract is consistent.

## 10. Frontend-only implementation strategy

To keep the frontend safe and reusable:

1. Add a backend API client abstraction
   - centralize fetch logic
   - preserve mock fallback when backend is unavailable
2. Split demo and real data providers
   - `demoDataProvider`
   - `apiDataProvider`
3. Normalize response shapes to `frontend/src/types/index.ts`
4. Preserve UI behavior for loading, empty, and error states
5. Treat `demo_mode` as an explicit development setting, not as the only mode
6. Keep all route names and page transitions stable

## 11. Recommended frontend folder pattern

```text
frontend/src/
├── api/
│   ├── client.ts
│   ├── mockData.ts
│   ├── liveApi.ts
│   ├── transformers.ts
│   └── types.ts
├── components/
├── hooks/
│   ├── usePlants.ts
│   ├── useSensors.ts
│   ├── useRecommendations.ts
│   └── useSettings.ts
├── pages/
├── stores/
│   └── appStore.ts
├── lib/
└── types/
```

## 12. Frontend validation and UX requirements

The frontend must handle:
- loading spinners during initial fetch
- empty states for empty plant or record lists
- error banners when API calls fail
- stale-data indicators for disconnected backend
- graceful handling when no SHAP data is available
- date/time formatting consistent with app style

## 13. Integration checklist

1. Confirm actual route names and page layout in `frontend/src/App.tsx`
2. Match frontend types to backend response payloads
3. Replace mock initialization workflow in `apiClient`
4. Keep demo data isolated to development mode only
5. Add fallback states for backend unavailability
6. Ensure all pages use the same normalized response contract
7. Verify graph rendering with real data shapes
8. Add test coverage for API failure behavior

## 14. Critical frontend note

The frontend is already polished and structurally complete. The core work is not in redesigning the UI, but in disconnecting the app from the in-memory mock layer and connecting each page to a real backend contract that matches the data already modeled in the app.

This is the correct frontend-only handoff for the current state of the repository.

## 15. Final assessment

The frontend is in good shape and ready for API-backed integration. The main required work is:
- remove or isolate mock dependencies
- map each page to a real backend endpoint
- keep the current UX intact
- ensure backend responses match the frontend’s expected types and semantics

That is the most important architecture change needed for the app to become fully operational.

---

This file is intended as a frontend-only technical handoff for the application branch. It does not replace the backend architecture document; it explains the UI and integration boundaries required to connect the current frontend to the real system.
