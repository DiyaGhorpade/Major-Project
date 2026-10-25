# Phase 0 Findings: Inspection Before Building

Status: complete (2026-10-25). Source of truth for Phases 1–5. Where this file and
`backend_implementation_master_prompt (1).md` disagree, **this file wins** (it reflects the real code/data).
Every item tagged **[DEV-n]** must be copied into `backend/README.md → Deviations`.

---

## 1. Dataset (`money_plant_ml_pipeline/data/raw/plant_sensor_data.csv`)

`docs/plant_sensor_data.csv` is byte-identical (sha256 `92ea22e2…f860`, 368,219 bytes). The backend reads only the ML copy (env `DATASET_PATH`).

| Fact | Value |
|---|---|
| Rows / columns | 5,376 × 12; no exact duplicates; no duplicate `(plant_id, timestamp)` |
| `plant_id` | int 1–16, 336 readings each |
| `session_id` | int 1–4. Each session = one calendar week (Mon 00:00 → Sun ~22:05 IST), 1,344 rows = 16 plants × 84 |
| `node_id` | int 1–2. **Node 1 = odd plants (1,3,…,15); Node 2 = even plants (2,4,…,16).** 672 rows per node per session |
| `sampling_point` | int 1–12, **varies per reading for every plant** (cycles 1–12); it is NOT a fixed pot position |
| `timestamp` | ISO-8601 **with `+05:30`** (tz-aware). 2026-09-21T00:00 → 2026-10-18T22:05:15. 2 h cadence per plant, no gaps. Each timestamp is shared by exactly 2 plants (fixed per-pair phase offsets 0 s…315 s), so cross-plant exact-time joins don't line up |
| Missing values | Empty strings, 307 cells in **233 rows** (soil 82, temperature 69, humidity 71, light 85); 9–20 rows per plant |
| Ranges (non-null) | soil 0.9–98.5; temperature 15.0–40.0 (57 rows at exactly 15.0, likely clipped); humidity 29.9–100 (58 at 100); light 0–61,467 (999 zeros). **0 rows outside the prompt's ranges** |
| `stress_level` | healthy / low / moderate / high |
| `stress_type` | none / water / light / temperature_humidity / mixed. `healthy` ⇔ `none`. **No nutrient type exists** |
| `is_synthetic_anomaly` | 0/1, 65 positives (plants 9, 6, 4 most) |
| Plant groups | Blocks of 4 with a dominant `stress_type`: 1–4 none, 5–8 water, 9–12 light, 13–16 temperature_humidity/mixed. This matches `EXPERIMENTAL_GROUPS` in `money_plant_ml_pipeline/src/plant_ml/schema.py:37-42` |

---

## 2. ML service (real contract; the code wins over `docs/API_CONTRACT.md`)

Start: `cd money_plant_ml_pipeline && PYTHONPATH=src python -m uvicorn plant_ml.api.main:app --host 127.0.0.1 --port 8001`.
Port and host come only from the uvicorn CLI; `config.api.port` is inert.

### 2.1 Environment blocker
Artifacts need **sklearn 1.9.x, numpy 2.5.x, xgboost 3.4.x** (major.minor are checked at `artifact_manager.py:171-188`), plus `shap` for explanations.
System python3 is 3.13.11 with sklearn 1.8, numpy 2.4, and no xgboost/shap, so the service would come up `unavailable` (503).
→ `scripts/dev.sh` must create `money_plant_ml_pipeline/.venv` from `requirements-tested.txt` + `requirements-optional.txt` (shap). Never set `PLANT_ML_ALLOW_VERSION_MISMATCH`.

### 2.2 Endpoints
- `GET /health` → always 200: `{status:"ok"|"unavailable", models_loaded, artifact_error, run_id, trained_on, explainability_available, warnings[]}`
- `GET /metadata` → 200/503: `run_id, model_version, trained_at, training_data{…}, trained_on, validated_on_real_plants, split{strategy,cutoff_timestamp,test_fraction,train_rows,test_rows}, models{classifier|clustering|anomaly:{feature_count,feature_names,…}}, label_mapping, anomaly_score_convention, cluster_note, explainability{available,reason}, required_history_rows_for_full_features (=49), dependency_versions_at_training, compatibility_warnings, disclaimer`. **Does not expose `max_history_rows` or `max_batch_items`.**
- `POST /validate` `{readings:[Reading]}` → `{valid, n_readings, issues:[{index,issues[flags]}], summary{rows,plants,time_range,missing_sensor_values,sensor_range_violations,duplicate_plant_timestamp}}`. Works without models.
- `POST /predict` `{reading, history:[] (default), explain: bool|null}`. Errors are all **422** (never 400) with `detail.error` ∈ `history_too_long` (>500), `mixed_plants`, `invalid_readings` (+issues), `history_not_before_reading` (strict `<`). Also 503 `models_unavailable`, Pydantic 422 (list-shaped detail), and unhandled 500.
- `POST /predict/batch` `{items:[PredictRequest]}`: **per-item history is allowed**, max **64** items. Response: `{results:[{index, ok, result|null, error:{status,error,message,issues?}|null}]}` in input order. A whole-request 503/422 (`bad_batch_size`) is possible; a schema-invalid item fails the whole batch.

Reading fields: `timestamp, plant_id, soil, temperature, humidity, light` (nulls allowed = missing). If `session_id/node_id/sampling_point/stress_*` are sent they get validated (`node_id` must be the string "1"/"2"). **The backend sends only the six core fields.**

### 2.3 PredictResponse
```
plant_id:int, timestamp:str(+05:30)
stress_prediction: {label: healthy|low|moderate|high, probabilities:{healthy,low,moderate,high}}
cluster: {id:int (KMeans, k=3), distance:float}        # environmental state, NOT severity
anomaly: {is_anomaly:bool (score>0), score:float (higher=more anomalous), threshold:0.0, interpretation:str}
explanation: {available:true, target_class, top_features:[{feature, value|null, contribution, direction:"towards_predicted_class"|"away_from_predicted_class"}] (top 5), units:"log-odds…", note}
           | {available:false, reason, top_features:[]}            # NO base/expected value anywhere
recommendations: [{id, category, text, reason, priority: low|medium|high, supporting_features:{}, type:"heuristic_suggestion"}]   # NO title/action
recommendations_disclaimer: str
history: {rows_available (=len(history)+1), rows_recommended: 49, sufficient: bool}
warnings: [str]           # cold-start (<48 history rows), missing sensor values
model_metadata: {trained_on:"synthetic", validated_on_real_plants:false, model_version:"1.0.0", run_id, disclaimer}
```
ML rec ids → category: `verify_sensor`/`retake_reading` → `data_quality`; `low_confidence` → `model_confidence`; `check_watering`/`check_soil_drying` → `soil`; `check_light_low`/`check_light_high` → `light`; `check_climate` → `temperature_humidity`; `monitor_and_inspect` → `general`.
Precedence: an anomaly yields only `verify_sensor`; missing inputs yield only `retake_reading`; max prob < 0.45 yields only `low_confidence`; otherwise condition recs apply (these need 3 consecutive rows).

### 2.4 History length
Hard cap `api.max_history_rows = 500` (`config/config.yaml:100`). Full features need 49 rows (`/metadata.required_history_rows_for_full_features`).
All feature windows are trailing and ≤ 48 rows (rolling 3/6/12, baseline `shift(1).rolling(48)`), so for the current row **history beyond 48 rows does not change the features**.
→ **[DEV-1]** The backend sends `history_rows = min(max_history_rows, max(required_rows, HISTORY_ROWS_DEFAULT=60))`. `max_history_rows` is read from `ML_CONFIG_PATH` (YAML file read only, no import) and `required_rows` from `/metadata`; the API contract itself recommends 60. Phase 2 must verify that 60 rows and 500 rows give identical predictions on sample readings.

### 2.5 Artifacts and metrics (newest run `v1.0.0_20261021T145244Z`, per `artifacts/latest.json`)
- `classifier.meta.json → metrics`: `accuracy .6458, macro_f1 .5864, weighted_f1, balanced_accuracy, n_samples 1344, per_class.{healthy,low,moderate,high}.{precision,recall,f1,support}, confusion_matrix{labels,matrix} (rows=true), class_counts`, plus `baseline_metrics`, `feature_names`(62), `created_at_utc`, `model_version`, `known_limitations`.
- `anomaly.meta.json`, `clustering.meta.json` (selected_k 3, cluster summaries), `manifest.json` (split, provenance, dependency versions).
- **`reports/` is empty**: no global SHAP and no gain importance are shipped. **[DEV-2]** See §4.6.
- Doc drift (record only, don't fix): BUILD_REPORT numbers are for the older run (acc .627); the newest manifest says it was trained on Windows; the anomaly threshold is hard-coded to 0.0.

---

## 3. Frontend (`frontend/`)

- React 19, Vite 8, TS 6, Tailwind 3, recharts 3. Hash routing in `App.tsx` with no router lib. Data fetching is `useEffect` + `apiClient` singleton, with no react-query or catch handlers.
- Scripts: `dev`, `build` (`tsc -b && vite build`), `lint` (oxlint). **No test framework.** Phase 4 adds vitest + @testing-library/react.
- `vite.config.ts` has no proxy yet.
- No CSV upload or file inputs exist (only `exportCsv.ts` for export).
- Removal targets: `mockData.ts`, everything mock in `client.ts` (localStorage, demo/model flags, SHAP templates, `simulateLiveReadingPush`), `DemoDataBanner.tsx` + its mount (`AppShell.tsx:90-95`), demo state in `AppShell.tsx:22-41,50,62`, `Sidebar.tsx:39,47,177,184-188`, the `'demo'` variant in `StatusBadge.tsx`, `AiInsightsPage` simulate toggle + hard-coded metrics (`:28,52-55,87-143`), `TopBar` hard-coded notifications/clock, `SettingsPage` api_base_url field, and hard-coded "5 classes" in `ModelPerformancePage:87,120`.
- Pages and fields consumed: see the inspection report summary. Key consumers: SHAP waterfall uses `base_value`, `model_output`, `features[].{displayName,observed_value,unit,shap_value,laymanExplanation}`; beeswarm uses `feature_value_norm ∈ [0,1]` on a hard-coded ±0.6 axis.
- Existing frontend bugs fixed incidentally: naive timestamps parsed as local time; duplicate `REC-1501` keys; plants 6/8 mislabelled; AiInsights `metrics` fetched but not rendered.

---

## 4. Mapping decisions (implement in `backend/app/core/mappings.py`)

### 4.1 Identifiers
| Concept | Decision |
|---|---|
| `plant_id` | int 1–16 everywhere (TS `Plant.id: number`, CSV int, ML int) |
| `node_id` | **string `"1"`/`"2"`** in API + DB (TS `NodeId`, ML enum). CSV int → str at seed |
| Node ↔ plant | **CSV wins: odd → node 1, even → node 2** (8 plants each). The frontend/handoff "1–12 / 13–16" claim is wrong **[DEV-3]** |
| `Plant.samplingPoint` | Not a fixed property in the data → `number \| null` = the latest reading's sampling_point, documented as "multiplexer channel of latest reading" **[DEV-4]** |
| `record_id` | `"R-<db id>"` (string, stable) |
| Recommendation id / device id | `"rec-<db id>"`; devices `jetson-01`, `esp32-node-1`, `esp32-node-2` |

### 4.2 "Cluster" (two different concepts; never conflate)
- `Plant.cluster` / `ClusterId 'Cluster 1..4'` = **experimental group** (fixed design, matches ML `EXPERIMENTAL_GROUPS`): 1 = Healthy control (1–4), 2 = Water stress (5–8), 3 = Light stress (9–12), 4 = Temperature/Humidity stress (13–16).
- KMeans output → new prediction field `envState: {id, distance}`, labelled "Environmental state group N" with tooltip "Environmental state group. Not a stress level." It is **never** used for health or status.

### 4.3 Stress: two axes
- **Severity** (ML): `healthy|low|moderate|high` + probabilities. New fields `PlantPrediction.severity`, `.probabilities`, `.confidence = max prob`.
- **Type**: `TulsiCondition` is replaced by data-backed values **[DEV-5]**: `'Healthy' | 'Water Stress' | 'Light Stress' | 'Temperature/Humidity Stress' | 'Mixed Stress' | 'Unclassified Stress'`. `'Nutrient Stress'` is removed because the dataset has no nutrient data.
  - `Plant.stressType` = the group's designed treatment (Healthy / Water / Light / Temperature/Humidity Stress).
  - `PlantPrediction.condition` = `stress_type_guess` (labelled heuristic, `conditionSource:"heuristic"`): `'Healthy'` if severity == healthy; otherwise derived from the ML condition-recommendation categories in that response (soil → Water, light → Light, temperature_humidity → Temperature/Humidity, ≥2 → Mixed); if there are none → `'Unclassified Stress'`.
  - Dataset labels (`label_stress_level`, `label_stress_type`, `label_is_synthetic_anomaly`) are exposed only under an explicit `datasetLabels` object, never as predictions.
- `Plant.status` (`PlantHealthStatus`) from the latest ML severity: healthy → `healthy`, low/moderate → `attention`, high → `critical`, no prediction → `unmeasured`. Anomalies do not change status.

### 4.4 Readings and validation
- `SensorRecord.soil/temperature/humidity/light` become `number | null` **[DEV-6]**. Missing values are kept as null and the row gets `status:'flagged'` with `validation_error:"missing: soil, light"`. Out-of-range values → `flagged` (none in the current data). `rejected` is reserved for ingestion payloads that fail validation. Values are never filled in.
- Timestamps are stored and emitted as ISO-8601 `+05:30`.

### 4.5 Recommendations
Map ML category → TS `RecommendationCategory`: `soil → irrigation`, `light → lighting`, `temperature_humidity → climate`, `data_quality → sensor_check` ("Sensor check"), `model_confidence|general → monitoring` (**new value [DEV-7]**). `nutrition` is kept in the type but never produced.
`RecommendationStatus` adds `superseded` (auto-expired, untouched by the user).
New fields: `source: 'ml'|'heuristic'`, `confidence`, `basedOn:{timestamp, modelVersion, runId}`, `disclaimer:true`, `mlRecId`.
`title` = backend label per ML rec id; `simpleAction` = ML `text`; `reason` = ML reason + cited values ("Soil moisture 21.3%; model P(high)=0.64"); `scientificContext` = static per-category text; `guidance_source` = "ML heuristic rules · model 1.0.0 (run …)".

### 4.6 SHAP **[DEV-2]**
- Local: from the latest `explain=true` prediction per plant. ML gives top-5 log-odds contributions with **no base value** → `base_value: number | null` and `model_output: number | null`, both null. The waterfall starts at 0 and is labelled "contributions relative to model baseline (not provided)". `predicted_probability` = P(predicted class). `displayName/unit/description/laymanExplanation` come from a backend feature-name dictionary (pattern-based, e.g. `soil_roll6_mean` → "Soil moisture, 6-reading average").
- Global (**as the prompt intended: read from artifacts/reports**): run the ML pipeline's own evaluation script once, as an explicit setup step (like training, never automatic in the backend), to generate `reports/<run_id>/shap_global_importance.json` (+ `classifier_metrics.json` with `feature_importance_gain`) from the existing artifacts, with no retraining. The backend's `model_service` reads these files (file read, no import). If they are missing, `/api/ai/shap/global` returns an empty-state payload and the README/dev.sh tells the user the command. `EXPLAIN_MODE=latest_only` stays the default. The SHAP axis is made data-driven (no hard-coded ±0.6).
  - **Verified (code audit):** `scripts/generate_explanations.py --input data/raw/plant_sensor_data.csv --run-id <run_id>` (from the ML dir, inside its venv with shap) loads the bundle read-only. It does not fit anything or touch `artifacts/`/`latest.json`. It explains 500 test rows (seed 42).
  - Output: **`reports/<run_id>/evaluation_<UTC>/shap_global_importance.json`** (nested timestamped folder). The backend picks the newest `evaluation_*` dir for the current `run_id`, then falls back to `reports/<run_id>/shap_global_importance.json` (written only by training).
  - JSON: `{mean_abs_shap: {feature: float} (all features, desc, mean over rows AND classes, log-odds), per_class_top15: {healthy|low|moderate|high: {feature: float}}, note}`. Same folder: `evaluation.json`, `shap_sample_explanations.csv` (`row,predicted,feature,value,contribution,direction`, top 5 for 50 rows), PNGs.
  - **[DEV-16]** There are no per-sample signed values, so a true beeswarm is impossible. `beeswarm_points` is built from `shap_sample_explanations.csv` (50 rows × top 5, signed; `feature_value_norm` = per-feature min-max of `value`) and labelled "sample of 50 test rows". `classifier_metrics.json`/`feature_importance_gain` cannot be produced without retraining, so it isn't used; Model Performance reads `classifier.meta.json`.
  - Setup step: `make ml-explain` / dev.sh prints the command when the report is missing.

### 4.7 Model performance
`overall.accuracy = metrics.accuracy`; `precision`/`recall` = macro mean of `per_class`; `f1_score = macro_f1`; `per_class` from meta (labels healthy/low/moderate/high); `confusion_matrix` verbatim; `total_samples = 1344`; `split = "chronological hold-out, last 25% (cutoff 2026-10-11T22:05:15+05:30)"`; `evaluated_at = created_at_utc`; plus `provenance:"synthetic"`, `balanced_accuracy`, `baselines`. The UI text "5 classes" → driven by `labels.length`.

### 4.8 Devices and sessions
- ESP32 nodes 1/2: `last_transmission` = latest reading timestamp, `channels` = distinct sampling points; `ip/mac/rssi/firmware/uptime` = **null** (TS fields → nullable, rendered "n/a"). Jetson: every hardware field null. `status:'unknown'` (historical dataset, no live heartbeat). `'demo'` is removed from the status union.
- Sessions 1–4: `scheduled_at/started_at` = first reading, `ended_at` = last, `expected = 16 × 84 = 1344` (84 = 7 days × 12 per day at 2 h cadence, computed from data), `received` = rows, `rejected` = 0 (flagged-count exposed separately), `status:'completed'`, `trigger_type:'scheduled'`, `node_progress{"1":{expected:672,received,status},"2":…}`.

### 4.9 Settings
Drop `api_base_url` and `demo_mode` (TS + UI). Persist the rest in the DB.

---

## 5. User decisions (2026-10-25)
- **Plant branding: keep Tulsi.** Plant name, species, variety, description, care tip, location and optimal ranges are not in the dataset. They live in one static catalog, `backend/app/core/plant_catalog.py`, which reuses the existing frontend Tulsi copy (odd = Rama, even = Krishna). Optimal ranges are aligned with the ML recommendation thresholds where they exist (soil low 25 %, light 300–20,000 lux, temperature 18–32 °C, humidity ≥ 40 %). **[DEV-8]**
- **Global SHAP: follow the prompt's intent and read it from `reports/`** (§4.6), generated once by the ML pipeline's evaluation script (no retraining; it only writes report files and changes no ML code or models).
