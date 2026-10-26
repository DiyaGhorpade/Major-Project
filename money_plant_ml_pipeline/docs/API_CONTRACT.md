# API contract (local service, default http://127.0.0.1:8001)
Interactive docs: `/docs`. No authentication; keep it on localhost or behind a firewall.

## Endpoints
| | |
|---|---|
| `GET /health` | `{status: ok\|unavailable, models_loaded, artifact_error, run_id, trained_on, explainability_available, warnings}` (always 200) |
| `GET /metadata` | run id, version, training data provenance + hash, split, per-model feature lists, label mapping, score convention, `required_history_rows_for_full_features` (60), `cross_plant` (`enabled`, `reference_plants`, `tolerance_minutes`, usage), `temporal_smoothing` (method, window, temperature), disclaimer. 503 if models unavailable |
| `POST /validate` | body `{"readings":[Reading,...]}` -> `{valid, n_readings, issues:[{index, issues:[...]}], summary}` (200 even if invalid) |
| `POST /predict` | body `{"reading": Reading, "history": [Reading,...], "explain": bool?, "reference": [Reading,...]?}` -> prediction for the latest reading |
| `POST /predict/batch` | body `{"items":[PredictRequest,...], "reference": [Reading,...]?}` (max 64 items) -> `{results:[{index, ok, result?, error?}]}`; one bad item does not fail the rest. The batch-level `reference` is shared by all items |

`Reading`: `timestamp` (ISO 8601), `plant_id` (1-16), `soil`, `temperature`, `humidity`, `light` (null = missing), optional `session_id`, `node_id`, `sampling_point`. Label fields are accepted and **ignored**.

## Example request
```json
{"reading": {"timestamp": "2026-10-21T10:30:00+05:30", "plant_id": 3, "soil": 48.2, "temperature": 26.4, "humidity": 68.0, "light": 3200},
 "history": [ {"timestamp": "2026-10-21T08:30:00+05:30", "plant_id": 3, "soil": 48.9, "temperature": 25.1, "humidity": 70.2, "light": 2100} ]}
```
## Response shape (values illustrative; real output is computed)
```json
{"plant_id": 3, "timestamp": "2026-10-21T10:30:00+05:30",
 "stress_prediction": {"label": "healthy", "probabilities": {"healthy": 0.71, "low": 0.26, "moderate": 0.02, "high": 0.01},
   "unsmoothed_probabilities": {"healthy": 0.64, "low": 0.31, "moderate": 0.04, "high": 0.01}},
 "cluster": {"id": 1, "distance": 1.14},
 "anomaly": {"is_anomaly": false, "score": -0.19, "threshold": 0.0, "interpretation": "..."},
 "explanation": {"available": true, "top_features": [{"feature": "soil_roll12_mean", "value": 48.1, "contribution": 0.9, "direction": "towards_predicted_class"}]},
 "recommendations": [], "recommendations_disclaimer": "Heuristic suggestions ... Not an expert diagnosis ...",
 "history": {"rows_available": 60, "rows_recommended": 60, "sufficient": true}, "warnings": [],
 "model_metadata": {"trained_on": "synthetic", "validated_on_real_plants": false, "model_version": "1.0.0", "run_id": "..."}}
```
Anomaly score: **higher = more anomalous**; `is_anomaly` when score > 0. A flagged anomaly means "check the sensor", not "plant stress".
## Errors
400-class: 422 with `detail:{error,message,...}` for `invalid_readings`, `mixed_plants`, `history_not_before_reading`, `history_too_long`, `reference_too_long` (more than `api.max_reference_rows`, default 5,000), `bad_batch_size`, `empty_request`, or schema/type errors; 503 `models_unavailable`.

## Reference readings (cross-plant features)
The classifier compares each plant with the healthy-control plants measured at the same time. Send the readings of the plants in `/metadata` `cross_plant.reference_plants` (1-4) as `reference`, covering the time span of the history: from the first history row (minus up to 30 min) to the reading. For each row the service uses, per reference plant, the latest reading at or before that row's timestamp (at most 30 min older). Later readings are ignored, so a wider window is harmless. Invalid reference rows are dropped, not rejected. Without reference readings the prediction still works but is less reliable, and `warnings` says so. `stress_prediction.probabilities` are smoothed over the plant's last 12 readings, re-predicted from the supplied history; `stress_prediction.unsmoothed_probabilities` is the single-reading model output.

## Backend integration flow
1. ESP32 -> existing receiver validates -> 2. persist to SQLite -> 3. fetch the last 60 rows for that plant (chronological, strictly before the new reading) and the control plants' readings over the same span -> 4. `POST /predict` (control readings as `reference`) -> 5. store result (optional) -> 6. React reads it through your existing backend.
**Cold start / missing history:** a new plant still gets a prediction with `history.sufficient=false` and a warning; treat it as low-reliability. **Rate:** readings arrive every ~2 h per plant in the supplied data; call the ML service once per new reading (or batch per cycle), not per raw packet, and skip if a prediction for that plant already exists within the last N minutes.
## Example client (httpx, with timeout + retry)
```python
import httpx, time

def predict(reading, history, base="http://127.0.0.1:8001", retries=2):
    for attempt in range(retries + 1):
        try:
            r = httpx.post(f"{base}/predict", json={"reading": reading, "history": history, "explain": False}, timeout=5.0)
            if r.status_code == 503:          # models not loaded: degrade gracefully, show "prediction unavailable"
                return None
            r.raise_for_status()
            return r.json()
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == retries:
                return None                    # never block ingestion on the ML service
            time.sleep(0.5 * (attempt + 1))
```
## Proposed SQLite table (owned by your backend, not by this package)
```sql
CREATE TABLE IF NOT EXISTS ml_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, plant_id INTEGER NOT NULL, reading_timestamp TEXT NOT NULL,
  stress_label TEXT, p_healthy REAL, p_low REAL, p_moderate REAL, p_high REAL,
  cluster_id INTEGER, cluster_distance REAL, is_anomaly INTEGER, anomaly_score REAL,
  history_rows INTEGER, recommendations_json TEXT, explanation_json TEXT,
  model_run_id TEXT, trained_on TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (plant_id, reading_timestamp, model_run_id));
```
