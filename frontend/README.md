# Plant Monitor — Tulsi Stress Monitoring & Diagnostics (NVIDIA Jetson Orin Nano)

A modern, responsive environmental analytics web application for indoor **Tulsi flora (*Ocimum sanctum* & *Ocimum tenuiflorum*)** stress monitoring, powered by an **NVIDIA Jetson Orin Nano** and sequential ESP32 microcontroller sensor nodes with explainable machine learning (TreeSHAP) and evidence-based agronomic recommendations.

---

## 🌿 Tulsi Experimental Cluster Design

The 16 monitored indoor specimens represent a controlled 4-cluster comparative stress trial:

| Cluster | Stress Condition | Specimens | Channel Mapping | Experimental Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster 1** | **Healthy (Control)** | Tulsi #1 – #4 | Node 1: `SP-01` to `SP-04` | Maintained in baseline optimal microclimate (moisture 55–65%, 3,500–5,000 lx, 22–26°C, balanced nutrition). |
| **Cluster 2** | **Water Stress** | Tulsi #5 – #8 | Node 1: `SP-05` to `SP-08` | Induced drought deficit (soil moisture &lt; 20%), modeling stomatal closure and wilting dynamics. |
| **Cluster 3** | **Light Stress** | Tulsi #9 – #12 | Node 1: `SP-09` to `SP-12` | PAR illumination deprivation (&lt; 500 lx), modeling etiolation, chlorosis, and reduced photosynthetic rate. |
| **Cluster 4** | **Nutrient Stress** | Tulsi #13 – #16 | Node 2: `SP-13` to `SP-16` | Macronutrient (N-P-K) exhaustion in inert substrate, observing stunted shoots and lower-leaf nitrogen mobilization. |

---

## Visual Direction: Modern Forest Theme

The UI adopts a calm, sophisticated, natural forest aesthetic:
- **Deep Forest**: `#102F26` (Sidebar & branding)
- **Dark Pine**: `#173F32` (Navigation active state)
- **Moss Green**: `#426B4F` (Secondary headings & accents)
- **Emerald Accent**: `#2E8B57` (Primary buttons & optimal statuses)
- **Bright Leaf Accent**: `#79B86A` (Pulse indicators & sub-badges)
- **Soft Sage**: `#DDE8D8` (Badge borders & dividers)
- **Pale Mint Surface**: `#EDF4EB` (Metric cards & pill containers)
- **Warm Ivory Canvas**: `#F6F7F0` (Application background)
- **Warning Amber**: `#C98227` (Attention thresholds)
- **Error Red**: `#C94C4C` (Critical alerts & CRC/validation rejections)

All status indicators pair colors with icons and text labels (never color alone) to maintain WCAG accessibility.

---

## Architecture & Features

### 1. Workspace
- **Overview (`/overview`)**: High-level telemetry health, Jetson connection status, ESP32 nodes online, transmission integrity rate (accepted/flagged/rejected), multi-metric environmental trend charts, and recent activity log.
- **Live Monitoring (`/live-monitoring`)**: Real-time sensor stream with configurable polling interval (2s, 3s, 5s, 10s), pause/resume controls, live sparklines for 4 channels (Soil, Temp, Humidity, Light), and an event stream feed.
- **Plants (`/plants`)**: All 16 specimens organized across 4 clusters (`Cluster A`: Ferns, `Cluster B`: Tropicals, `Cluster C`: Herbs, `Cluster D`: Succulents) with card grid / table view toggle, search, and cluster filters.
- **Plant Details (`/plant/:id`)**: Comprehensive species dossier with calibrated physical readings, target range bars, chronological oscillations, data-quality warnings, model prediction panel, and direct link to local SHAP attribution.

### 2. Intelligence & Explainability (XAI)
- **AI Insights (`/ai-insights`)**: NVIDIA Jetson TensorRT edge inference overview, model version (`v1.2.0-jetson`), sub-millisecond inference latency (1.8 ms), and phenotype condition distribution. Includes an interactive toggle to test the polished **"Model Not Connected"** offline state.
- **SHAP Analysis (`/shap-analysis`)**:
  - *Local SHAP*: Waterfall-style chart visualizing baseline expected value $E[f(x)]$, positive feature pushes, negative pushes, final model output $f(x)$, and detailed feature attribution table.
  - *Global SHAP*: Feature importance rankings (mean $|\text{SHAP}|$) and a beeswarm scatter plot showing feature value gradient vs. prediction impact.
  - *Interpretability Standard*: Adheres strictly to the guideline that SHAP explains feature influence without claiming biological causality.
- **Model Performance (`/model-performance`)**: Offline evaluation benchmark metrics on a 1,420-sample holdout test set (94.2% accuracy, 93.8% precision, 94.0% recall, 93.9% F1-score), 5x5 confusion matrix, and per-class performance metrics.
- **Recommendations (`/recommendations`)**: Evidence-based care directives derived from heuristic sensor thresholds and plant profiles. Allows updating action statuses (`New`, `In Progress`, `Completed`, `Dismissed`) with local persistence.

### 3. Operations & System
- **Data Logs (`/data-logs`)**: Searchable, filterable telemetry repository with pagination, CRC validation indicators, modal record detail drawer, and **one-click CSV export respecting active filters**.
- **Devices & Network (`/devices-network`)**: Jetson Orin Nano hardware telemetry (CPU, GPU, RAM, NVMe storage, die temperature, uptime) and ESP32 node diagnostics (Wi-Fi RSSI, MAC, channels). Does not expose secrets or credentials.
- **Sessions (`/sessions`)**: Audit registry of 3x/day scheduled collection runs with duration, expected vs. received records, and per-node progress.
- **Settings (`/settings`)**: Calibration points for Soil Capacitive ADC (Dry Air / Wet Immersion), optional pH sensor toggle (unconfigured by default), collection schedule, and Jetson receiver URL.

---

## Hardware & Data Schema

- **Target Ingestion Host**: NVIDIA Jetson Orin Nano Developer Kit (8GB) running Ubuntu 22.04 LTS / JetPack 5.1.
- **Node 1**: ESP32-WROOM-32D with CD74HC4067 multiplexer acquiring Sampling Points `SP-01` to `SP-12` (Plants 1 to 12).
- **Node 2**: ESP32-WROOM-32D acquiring Sampling Points `SP-13` to `SP-16` (Plants 13 to 16).
- **Record Schema**:
  ```json
  {
    "record_id": "REC-1048",
    "timestamp": "2026-10-20T11:45:00.000Z",
    "received_at": "2026-10-20T11:45:00.180Z",
    "accepted_at": "2026-10-20T11:45:00.180Z",
    "session_id": "SES-20261020-1145",
    "node_id": "node-1",
    "sampling_point": "SP-03",
    "plant_id": "plant-03",
    "soil": 19.4,
    "temperature": 27.8,
    "humidity": 38.2,
    "light": 1800,
    "ph": null,
    "status": "accepted",
    "rssi": -57,
    "crc_ok": true
  }
  ```

---

## Installation & Running

### Prerequisites
- Node.js `v18.0.0` or later (`v22.x` tested)
- npm `v9.x` or later

### 1. Install Dependencies
```bash
npm install
```

### 2. Run Development Server
```bash
npm run dev
```
Open your browser at `http://localhost:5173`.

### 3. Build for Production
```bash
npm run build
```
Generates optimized static assets in `dist/`.

### 4. Preview Production Build
```bash
npm run preview
```

---

## Backend Ingest Connection

By default, the application runs in **Demo Mode** with realistic, physically modeled multi-day telemetry, live random-walk simulation during polling, and precomputed SHAP explanations.

To connect to a live Jetson receiver:
1. In the top banner, click **"Switch to Live API"**.
2. Or navigate to **Settings** (`#/settings`) and enter the base URL of your Jetson Flask receiver (e.g. `http://192.168.1.120:5000` or `http://localhost:5000`).
3. Ensure the receiver exposes the REST endpoints:
   - `GET /api/v1/devices`
   - `GET /api/v1/plants`
   - `GET /api/v1/records`
   - `GET /api/v1/sessions`
   - `GET /api/v1/ml/metrics`
   - `GET /api/v1/recommendations`
