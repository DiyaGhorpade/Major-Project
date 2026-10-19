# Plant Health Monitor — Pipeline Architecture & Design Decisions

This document covers items 3–11 from your spec: the full pipeline architecture, data flow, cleaning strategy, feature engineering, model choices, fusion strategy, real-time inference mechanism, plant profiling, and deployment layout. No code yet — this is the design you asked for first. Once you confirm this is right, I'll implement it stage by stage.

---

## 1. Complete Architecture

```
RAW DATA (ESP32 logger + camera, your side)
        │
        ▼
DATA VALIDATION            [dev machine, offline]
        │
        ▼
DATA CLEANING              [dev machine, offline]
        │
        ▼
SENSOR/IMAGE SYNCHRONIZATION [dev machine, offline]
        │
        ▼
FEATURE ENGINEERING        [dev machine, offline]
        │
        ▼
DATASET CREATION           [dev machine, offline]
        │
        ▼
TRAIN / VAL / TEST SPLIT   [dev machine, offline — plant-level]
        │
        ├──────────────┬──────────────┐
        ▼              ▼              │
  VISION MODEL    SENSOR MODEL        │
  (train_vision)  (train_sensor)      │
        │              │              │
        ▼              ▼              │
  MODEL EVAL      MODEL EVAL          │
        │              │              │
        └──────┬───────┘              │
               ▼                      │
        MULTIMODAL FUSION  ◄──────────┘  (trained on held-out val predictions only)
               │
               ▼
        MODEL SERIALIZATION        [dev machine → exported artifacts]
               │
               ▼
        JETSON DEPLOYMENT          [copy artifacts to Jetson]
               │
               ▼
        REAL-TIME DATA INGESTION   [Jetson, continuous]
               │
               ▼
        REAL-TIME PROFILING        [Jetson, continuous]
               │
               ▼
        REAL-TIME CLASSIFICATION   [Jetson, continuous]
               │
               ▼
        DASHBOARD / API            [Jetson, continuous]
```

### Where each stage runs

| Stage | Runs where | Frequency |
|---|---|---|
| Data validation, cleaning, sync, feature engineering, dataset creation | Dev machine | Once per data-collection batch |
| Train/val/test split | Dev machine | Once per training run |
| Vision training | Dev machine (GPU strongly preferred) | Occasional, whenever retraining is decided |
| Sensor (XGBoost) training | Dev machine (CPU is fine) | Occasional |
| Fusion training | Dev machine | After vision + sensor models are finalized |
| Evaluation | Dev machine | After each training run |
| Model export/serialization | Dev machine | After training is finalized |
| Deployment (copy) | Manual/scripted transfer to Jetson | Whenever a new model version is promoted |
| Sensor ingestion, feature generation, sensor inference | Jetson | Continuous (every N seconds/minutes) |
| Image acquisition, vision inference | Jetson | Whenever a new image is captured |
| Fusion inference | Jetson | Whenever a new vision prediction arrives |
| Plant profile update, temporal smoothing | Jetson | Whenever a new fused prediction arrives |
| Dashboard/API | Jetson | Continuous (serves latest DB state) |

**Key principle:** training never happens on the Jetson. The Jetson only loads frozen, serialized models and runs inference. This keeps the edge device lightweight and predictable, and keeps your training data/experiments centralized on the dev machine where you can iterate freely without touching the production device.

---

## 2. Data Flow (conceptual)

Two independent streams converge at fusion:

- **Sensor stream**: ESP32 → USB serial → Jetson buffer → timestamped store → rolling-window feature generation → XGBoost → sensor probability vector (4 classes).
- **Vision stream**: Camera trigger → 3 raw images per session → preprocessing (crop/normalize/pick-or-average views) → vision model → vision probability vector (4 classes).

These two vectors are combined by the fusion model into a final probability vector, which is then temporally smoothed and written into that plant's persistent health profile.

The reason these stay as two separate models rather than one end-to-end multimodal network: sensor data arrives continuously and asynchronously from images, the two modalities have wildly different data volumes (thousands of sensor rows vs. a handful of images per day per plant), and decision-level fusion lets you evaluate, retrain, or even temporarily disable one modality (e.g., camera failure) without breaking the whole system. This matches the architecture you specified, and I agree it's the right call for this scale of dataset.

---

## 3. Sensor/Image Time-Synchronization Strategy

You asked me to choose between nearest-timestamp, interpolation, rolling average, or window-based aggregation, and justify it.

**Decision: window-based aggregation, using a short backward-looking window centered/ending at the image timestamp (e.g., the 5–10 minutes preceding capture).**

Why not the alternatives:

- **Nearest timestamp** — a single instantaneous reading is fragile. Capacitive soil-moisture sensors are noisy at the sample level, so picking one point-in-time reading risks aligning an image with a sensor glitch (a single dropout, a spike from the multiplexer switching channels, etc.).
- **Interpolation** — appropriate for filling small gaps in a continuous series, but it manufactures a value at the exact image timestamp rather than describing the plant's *recent condition*, which is what actually explains visible stress in a photo (e.g., soil moisture trending down over hours matters more than its exact instantaneous value).
- **Simple rolling average alone (no other stats)** — smooths noise but throws away trend information (a moisture level of 40% could mean "stable and fine" or "crashing down from 70%," and those are different physiological situations).
- **Window-based aggregation (chosen)** — captures both the current state *and* the recent trend by computing multiple statistics (mean, min, max, std, rate of change) over a defined window ending at the image timestamp. This is the only option that gives the sensor model enough context to be genuinely useful, and it's also exactly what we need at real-time inference time (Section 6 below), so training and inference use the same alignment logic — no train/serve skew.

Window length: default **10 minutes**, configurable in `config.yaml`. Long enough to smooth multiplexer/sensor noise, short enough that it still reflects "conditions right before this photo" rather than blending across hours. This is a tunable parameter, not a hardcoded constant.

---

## 4. Data Cleaning Strategy

**Principle: identify → flag → report → then decide whether to remove/replace. Never silently drop data, and never touch the raw files.**

Raw data is treated as immutable. All cleaning operations read from `data/raw/` and write to `data/cleaned/`, plus a cleaning report (JSON/CSV) documenting exactly what was changed and why. This gives you an audit trail and lets you re-run cleaning with different thresholds without ever losing original data.

### Sensor data

| Issue | Detection approach | Resolution |
|---|---|---|
| Missing values | Null/NaN check per column | Flag; forward-fill only within a small max gap (e.g. <2 missed samples), otherwise leave as gap (do not impute across large gaps) |
| Duplicate timestamps | Group by (plant_id, timestamp) | Keep first, flag duplicates in report |
| Invalid readings | Physically implausible values (e.g., humidity >100%, negative soil moisture) | Flag and null out — never guess a replacement value for a value that's physically impossible |
| Sensor disconnects | Long stretch of identical/null readings tagged as "sensor offline" | Flag as an outage window in the report; excluded from feature windows |
| Outliers | Rolling z-score or IQR per plant, per sensor (not a single global threshold — soil moisture baselines differ per plant/pot) | Flag; only removed if isolated single-point spikes inconsistent with physical sensor response time. Sustained "outlier" runs are *not* removed — that's likely a real stress event, which is exactly what we want to detect |
| Impossible jumps | Rate-of-change threshold (e.g., moisture can't change 40% in 10 seconds) | Flag as sensor glitch, treat that single point as missing rather than deleting the surrounding data |
| Flatlined sensors | Rolling std ≈ 0 over an implausibly long window | Flag as likely disconnected/stuck sensor |
| Missing plant IDs | Required-field check | Row dropped only if plant_id truly can't be inferred; logged |
| Timestamp inconsistencies (out of order, timezone drift) | Monotonicity check per source | Re-sorted; large drifts flagged for manual review rather than silently corrected |
| Different sampling frequencies | Per-sensor sampling rate profiling | Not resampled to a common grid at the cleaning stage — this is handled later by the window-aggregation step, which is frequency-agnostic |

### Image data

| Issue | Detection approach | Resolution |
|---|---|---|
| Missing files | Path existence check against manifest | Flagged/excluded, logged with plant/timestamp |
| Corrupted images | Attempt decode (e.g., PIL verify) | Excluded, logged |
| Incorrect dimensions | Compare to expected camera resolution | Excluded (indicates a capture-pipeline problem, not something to silently resize and hide) |
| Duplicate images | Perceptual hash comparison | Flagged; kept only once |
| Incorrect plant ID / timestamp | Cross-check against expected acquisition schedule from your logger | Flagged for manual review — this is not something to auto-correct, since a wrong plant_id silently "fixed" incorrectly would poison labels |
| Poor-quality images (blur, exposure, occlusion) | Lightweight quality heuristics (e.g., Laplacian variance for blur, mean pixel intensity for exposure clipping) | Flagged with a quality score; excluded from training only if score falls below a configurable threshold, kept but marked "low confidence" for real-time inference |

**Report output** includes: total rows in, valid rows out, rows removed by reason, values imputed, outliers detected vs. removed, sensor-outage windows, corrupted/missing/duplicate image counts, and quality-flag counts. This report is a first-class deliverable, not a side effect — you should be able to look at it and trust the cleaned dataset.

---

## 5. Sensor Feature Engineering — what to include and why

Raw instantaneous readings are not fed to XGBoost directly, because a snapshot value alone cannot distinguish "stable at 35% moisture" from "crashing through 35% moisture," and it's the *trend* that correlates with induced stress conditions in your experimental design.

**Features to use per plant, computed over the alignment window (Section 3):**

- **Soil moisture**: current value, rolling mean, rolling min, rolling max, rolling std, rate of change (slope over window), % deviation from that plant's own baseline (established during a healthy reference period), cumulative moisture deficit (running integral of below-baseline periods) — this last one specifically targets water stress, which is cumulative rather than instantaneous.
- **Temperature**: current, rolling mean, rolling min/max, short-term change. Kept simple since temperature is not one of your induced stress axes — it's mainly a confound/context variable.
- **Humidity**: current, rolling mean, change. Same reasoning as temperature.
- **Ambient light**: current, rolling mean, rolling min/max, and an exposure-trend feature (e.g., cumulative light over the day vs. that plant's baseline) — this directly targets the light-stress class.
- **Time-based features**: time-of-day (cyclical encoded, since light/temperature naturally vary with time of day and the model shouldn't confuse a normal evening dip for stress), and days-since-experiment-start (lets the model account for gradual/cumulative stress rather than only reacting to instantaneous readings).

**Deliberately excluded:** high-order interaction features, per-sensor-channel raw multiplexer index, and very long rolling windows (>1 day). With only 16 plants and a modest dataset size, an oversized feature set risks overfitting XGBoost far more than it helps — feature count should stay proportional to your sample size. I'd rather start lean and add features later if evaluation shows a specific class is hard to separate (e.g., add a dedicated deficit-integral feature if water stress recall is weak).

Feature engineering logic is shared between offline training and real-time inference (same function, different data source) specifically to prevent train/serve skew.

---

## 6. Vision Model Choice: YOLOv8n — Classification, not Detection

Your architecture named YOLOv8n as the vision model. I'm keeping YOLOv8n (matches your constraint of not redesigning without cause) but changing **how it's used**.

**Recommendation: use the YOLOv8n classification variant (`yolov8n-cls`), not object detection.**

Reasoning:
- Your camera setup is fixed-position, fixed-distance, standardized lighting, matte background, one plant per frame. There is no need to *locate* the plant — its position is already known and constant. Object detection would spend model capacity and training-label effort (bounding boxes) solving a problem you've already solved with your rig.
- The actual task is "given this whole standardized frame, what's the plant's health class?" — that is a classification problem, and `yolov8n-cls` is a lightweight image classifier that reuses the YOLOv8 backbone (small, well-optimized for edge deployment, exports cleanly to TensorRT/ONNX for the Jetson Orin Nano).
- Detection would only earn its keep if the camera position/plant position varied, if you needed to localize a specific stressed leaf/region, or if multiple plants appeared in one frame. None of those apply here.
- If down the line you want *interpretability* (e.g., highlighting which part of the leaf is driving the stress prediction), that's better solved later with a class-activation-map technique (Grad-CAM) on top of the classifier, not by switching to detection now.

So: same YOLOv8n family (keeps your architecture decision intact), but the classification head rather than the detection head — a targeted correction, not a redesign.

**Avoiding data leakage — plant-level split (Section 13 topic, flagged here too):** Because each plant is photographed repeatedly under the same fixed camera, lighting, and background, a model can trivially learn to recognize *individual plants* (pot rim shape, soil surface pattern, leaf arrangement quirks) rather than genuine stress *symptoms*. A random image-level split would let images of the same physical plant appear in both train and test, so the model would memorize plant identity instead of generalizing to stress patterns — evaluation numbers would look great and be meaningless. The split must be at the **plant level**: all images from a given plant_id go entirely into one of train/val/test. With 4 plants per group, this means every split must still contain plants from all 4 groups.

---

## 7. Sensor Model Choice: XGBoost (confirmed, as specified)

XGBoost on the engineered tabular features (Section 5), output as a full probability distribution over the 4 classes (`predict_proba`, not just `predict`). This matches your spec directly and is a good fit: tabular, moderate-sized dataset, mixed feature scales, no need for a neural net here. The feature schema used at training time is serialized alongside the model (as you specified) so that real-time inference is guaranteed to build features in the exact same order/format.

---

## 8. Multimodal Fusion Strategy

Both approaches will be implemented and compared, and the better one on **validation data** wins — no arbitrary choice.

**Approach A — Weighted probability averaging:**
`final = α · vision_probs + (1-α) · sensor_probs`
α is not fixed at 0.5. It's selected by a small grid/line search (e.g., α ∈ {0.0, 0.05, ..., 1.0}) evaluated against validation-set accuracy/F1, using vision and sensor predictions on the *validation* split (never test). This directly answers "which modality is more reliable overall" empirically rather than by assumption — for example, if vision turns out much more discriminative for light stress but sensors are more discriminative for water stress, this simple global-α approach will actually be a limitation, which motivates comparing against Approach B.

**Approach B — Logistic regression meta-classifier:**
Inputs: the 4 vision-class probabilities + 4 sensor-class probabilities (+ optionally a vision quality/confidence flag from the image-quality check in cleaning). Output: final 4-class prediction. This can learn *per-class, per-modality* trust (e.g., "trust vision more specifically for light stress, trust sensor more specifically for water stress") which a single global α cannot.

**Leakage control:** the meta-classifier is trained only on out-of-fold predictions from the vision/sensor models on the training set (or on the dedicated validation split), never on predictions the base models made on data they were trained on, and never touching the test set until final reporting. This is the standard stacking-safe procedure: base models must not have seen the rows used to train the fusion layer.

**Selection:** whichever of A or B scores higher on validation (accuracy + per-class F1, since class balance matters with only 4 groups) becomes the deployed fusion model. Both are cheap to run at inference time, so there's no latency reason to prefer one over the other — it's purely an accuracy decision made from data.

---

## 9. Real-Time Inference Mechanism

Two independent, asynchronous loops, joined at fusion:

**Sensor loop** — runs continuously on a fixed interval (default: every **60 seconds**, configurable). Each tick: read latest buffered ESP32 readings → validate → append to timestamped store → recompute the rolling feature window → run XGBoost → store the latest sensor probability vector per plant. Frequent because sensor data is cheap to process and this keeps the "current state" always fresh, which the profile page can show even between images.

**Vision loop** — event-driven, not interval-driven: triggers whenever the camera acquisition step (your logger/orchestration) produces a new image set for a plant (once/twice daily per your description, 3 viewpoints). On trigger: preprocess → run classifier on each of the 3 views → aggregate (mean of the 3 probability vectors, since they're the same plant/moment from slightly different angles — this also improves robustness to one bad-angle shot) → store as the latest vision probability vector for that plant.

**Fusion** — event-driven, triggered whenever a new vision prediction becomes available (since vision is the rarer event; there's no reason to re-fuse on every sensor tick when vision hasn't changed). At that moment: take the *latest valid sensor feature window* (from the continuous sensor loop) + the new vision probabilities → run the fusion model → temporal smoothing → update the plant's health profile.

This means: sensor probabilities update every minute (state is always current), but the *fused, official* classification updates once or twice a day, when new imagery arrives — matching how often the ground-truth-relevant signal (visual symptoms) actually changes. If the camera fails on a given day, the system should be able to fall back to sensor-only prediction for that cycle rather than stalling (flagged as a resilience requirement in Section 18, and reflected in the plant profile as "vision stale").

---

## 10. Plant Health Profiling & Temporal Smoothing

Each plant has one persistent profile record, updated (not replaced) on every fusion event. It holds: identity fields, current sensor state, rolling sensor trend summary, per-plant baseline (established once from an early healthy period per plant), most recent vision prediction + timestamp, most recent sensor prediction + timestamp, most recent fused prediction + confidence, a rolling history of past classifications (for trend display), and last-update timestamp.

**Smoothing method — Exponential Moving Average (EMA) on the fused class probabilities, not majority vote.**

Why EMA over the alternatives you listed:
- **Simple moving average** needs you to store and manage a fixed window of raw predictions; EMA achieves a similar smoothing effect with a single running value per class, which is simpler to store and update in a lightweight on-device database.
- **Majority vote** discards *how confident* each prediction was and reacts slowly/discretely — a plant sliding from Healthy into Water Stress would show no movement at all until votes literally flip, then jump abruptly. EMA on probabilities shows the emerging trend continuously (e.g., "Water Stress 82% → 87%"), matching the example you gave in Section 21 of the spec.
- EMA has one tunable parameter (decay factor), it's cheap to compute, and it directly prevents a single noisy fused prediction from flipping the displayed status — exactly the requirement you stated.

The final displayed class is the argmax of the EMA-smoothed probabilities, and the profile stores both the smoothed probabilities and the raw (pre-smoothing) fused prediction, so you can always inspect whether smoothing is masking a real fast-onset event versus noise.

---

## 11. Deployment Architecture

**Stays on the dev machine (training environment only):**
- Raw and cleaned datasets, training scripts, training logs/checkpoints, hyperparameter search artifacts, the plant-level train/val/test split definitions, evaluation reports/confusion matrices.

**Copied to the Jetson (inference-only artifacts):**
- `vision_model` — exported YOLOv8n-cls weights, converted to TensorRT/ONNX for Jetson Orin Nano
- `sensor_xgboost.json` — trained XGBoost model
- `fusion_model.pkl` (or the trained weighted-average α if Approach A wins)
- `feature_scaler.pkl` — any scaling/normalization fit on training data
- `feature_schema.json` — exact ordered feature list used at training time, so real-time feature generation can't drift out of sync
- `class_labels.json` — canonical class name ↔ index mapping shared by all three models
- A `model_manifest.json` recording model version, training date, dataset version, and evaluation metrics — this is how you track "what's actually running on the Jetson right now" versus what you've trained since.

**Loading & versioning:** at startup, `main.py` reads the manifest, loads each model by the path/version it specifies, and logs the active model versions. Promoting a new model version is a deliberate action (copy files + update manifest), never automatic — the Jetson should never silently start running a different model than the one you tested.

---

## Summary of the design decisions made here

| Decision point | Choice | Core reason |
|---|---|---|
| Sensor/image alignment | Window aggregation (10 min, configurable) ending at image timestamp | Captures trend + noise robustness; same logic usable online and offline |
| Data cleaning | Flag → report → conditional fix, never blind deletion | Preserves real stress signal, keeps raw data immutable and auditable |
| Vision task framing | YOLOv8n classification variant, not detection | Fixed camera/plant position makes localization unnecessary; classification matches the actual task |
| Train/test split | Plant-level, not row/image-level | Prevents the model memorizing individual plants instead of learning symptoms |
| Sensor model | XGBoost on engineered rolling/trend features, full probability output | Matches spec; tabular data with moderate size is XGBoost's sweet spot |
| Fusion | Both weighted-average and meta-classifier, best one wins on validation | Empirical choice beats an arbitrary α or model pick |
| Real-time cadence | Sensor: fixed interval; Vision: event-driven; Fusion: triggered by new vision result | Matches how often each modality's signal actually changes |
| Smoothing | EMA over fused probabilities | Continuous trend tracking, single-parameter, resistant to single-sample noise |
| Training location | Dev machine only; Jetson is inference-only | Keeps edge device stable, predictable, and simple to version |

---

Let me know if any of these decisions should be adjusted (e.g., window length, EMA decay, sensor-poll interval, or if you want detection kept as an option for future multi-plant frames). Once you confirm, I'll move to implementation, starting with `data_cleaning.py` and the folder scaffold, and build outward stage by stage so everything stays consistent.
