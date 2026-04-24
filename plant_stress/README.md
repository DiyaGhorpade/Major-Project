# 🌿 Plant Stress Detection & Monitoring

An end-to-end AI pipeline for real-time plant health classification, stress forecasting, and plant recommendation — designed for edge deployment on **NVIDIA Jetson Nano** and **Raspberry Pi**.

---

## Overview

This project builds a three-stage intelligent monitoring system for indoor plants using biosensor data. Rather than identifying plant species, the pipeline treats the **environment as the query** — classifying current stress levels, anticipating future stress before it occurs, and recommending which plants are best suited for a given set of conditions.

```
Sensor Readings
      │
      ▼
┌─────────────────────────┐
│  Step 1 — Classifier    │  Current stress: Healthy / Moderate / High
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Step 2 — Forecaster    │  Predicted stress 24h ahead (GRU)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Step 3 — Recommender   │  Which plants thrive in these conditions
└─────────────────────────┘
```

> **Current status:** Step 1 (Stress Classifier) is complete and documented. Steps 2 and 3 are in progress.

---

## Pipeline Stages

### Step 1 — Stress Classifier ✅
Tabular multi-class classification on a single snapshot of sensor readings. Three models are trained and compared — Logistic Regression (baseline), XGBoost (primary), and a PyTorch MLP (neural reference). XGBoost on GPU achieves **99.52% F1-macro** with only 1 misclassification across 240 test samples and zero dangerous errors.

### Step 2 — Stress Forecaster 🔄
A GRU-based sequence model that takes a sliding window of past sensor readings and predicts the stress class 24 hours ahead. Enables proactive intervention before stress becomes visible. Requires temporally correlated sensor data — planned for real hardware deployment.

### Step 3 — Plant Recommendation Engine 🔄
Given current environmental conditions (moisture, temperature, humidity, light, pH, nutrients), returns the top-N plants best suited to thrive there. Uses a range-matching / nearest-neighbor approach against a curated plant tolerance knowledge table. No ML training required — encodes domain knowledge explicitly.

---

## Results — Step 1

| Model | CV F1-macro | Test F1-macro | Test Accuracy | ROC-AUC | Dangerous Errors |
|---|---|---|---|---|---|
| Logistic Regression | 0.7525 ± 0.0238 | 0.7670 | 0.7708 | 0.9291 | 3 ⚠️ |
| **XGBoost (GPU)** | **0.9957 ± 0.0041** | **0.9952** | **0.9958** | **1.0000** | **0 ✅** |
| MLP (PyTorch) | — | 0.9349 | 0.9375 | 0.9936 | 0 ✅ |

> *Dangerous errors = True High Stress predicted as Healthy. Logistic Regression is disqualified for production on this basis.*

### SHAP — Top Features per Class

| Class | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| Healthy | Soil_Moisture (3.27) | Nitrogen_Level (1.14) | Phosphorus_Level (0.15) |
| Moderate Stress | Soil_Moisture (2.41) | Nitrogen_Level (0.77) | VPD_proxy (0.09) |
| High Stress | Soil_Moisture (3.00) | Nitrogen_Level (1.66) | Phosphorus_Level (0.06) |

---

## Dataset

**Source:** [ziya07/plant-health-data](https://www.kaggle.com/datasets/ziya07/plant-health-data) (Kaggle)

| Property | Value |
|---|---|
| Rows | 1,200 |
| Plants | 10 |
| Readings per plant | 120 (every 6 hours, 30-day window) |
| Missing values | 0 |
| Label | Healthy / Moderate Stress / High Stress |
| Nature | Synthetic — label derived from Soil_Moisture + Nitrogen_Level |

**Sensor features:** Soil Moisture, Ambient Temperature, Soil Temperature, Humidity, Light Intensity, Soil pH, Nitrogen, Phosphorus, Potassium, Chlorophyll Content, Electrochemical Signal

**Derived features (engineered):**
- `Thermal_Gradient` = Ambient_Temperature − Soil_Temperature
- `VPD_proxy` = Vapor Pressure Deficit approximation using Tetens formula

> **Note:** The Electrochemical Signal has near-zero correlation (r = 0.02) with the stress label in this dataset — it was generated independently. On real hardware, this feature is expected to be among the strongest stress indicators.

---

## Project Structure

```
plant_stress/
│
├── data/
│   └── plant_health_data.csv
│
├── src/
│   ├── config.py          # all constants, paths, hyperparameters
│   ├── utils.py           # seed utility
│   ├── preprocess.py      # data loading, temporal split, scaling
│   ├── train.py           # LR, XGBoost, MLP training
│   ├── evaluate.py        # metrics, confusion matrix, ROC, SHAP
│   └── export.py          # ONNX conversion (pending)
│
├── outputs/
│   ├── models/            # scaler.joblib, xgboost.joblib, mlp.pt
│   ├── plots/             # confusion matrices, ROC curves, SHAP plots
│   └── reports/           # per-model classification reports
│
│
├── environment.yml
└── README.md
```

---

## Setup

### Requirements
- Linux (tested on Ubuntu 24)
- NVIDIA GPU with CUDA 12.8 support (tested on RTX 5060 Laptop GPU)
- Miniconda / Anaconda

### 1. Create environment

```bash
conda env create -f environment.yml
conda activate plant-stress-step1
```

### 2. Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 3. Verify setup

```bash
python verify_env.py
```

Expected output:
```
PyTorch  : 2.7.0+cu128
CUDA OK  : True
GPU      : NVIDIA GeForce RTX 5060 Laptop GPU
XGBoost  : 2.1.3
sklearn  : 1.5.2
SHAP     : 0.46.0
ONNX     : 1.17.0
Device   : cuda
```

### 4. Add dataset

Download `plant_health_data.csv` from [Kaggle](https://www.kaggle.com/datasets/ziya07/plant-health-data) and place it in the `data/` directory.

---

## Running Step 1

Each script can be run independently. They all call `load_and_preprocess()` internally.

```bash
# Preprocessing only (sanity check)
python src/preprocess.py

# Train all three models
python src/train.py

# Full evaluation — metrics, plots, SHAP
python src/evaluate.py
```

All outputs are written to `outputs/` — models, plots, and classification reports.

---

## Key Design Decisions

**Temporal train/test split, not random**
Sensor readings from the same plant at adjacent timestamps are correlated. A random split leaks future context into training, inflating accuracy by 5–15%. All splits are chronological: first 80% of readings per plant for training, last 20% for testing.

**XGBoost over MLP for tabular data**
On structured tabular data at this scale (~1000 samples), gradient boosted trees consistently outperform neural networks. The MLP is included as a neural reference and to validate the PyTorch + CUDA pipeline for Step 2, not as a production candidate.

**class_weight='balanced' everywhere**
Healthy is the minority class at 24.9%. Without correction, all models bias toward High Stress (majority at 41.7%). Balanced class weights penalize errors on all classes equally.

**SHAP for explainability**
When a plant is flagged as stressed in production, SHAP values identify which sensor reading triggered the classification. This makes the system debuggable and trustworthy — especially important for edge deployment where sensor calibration issues can cause false alerts.

**Dropping Plant_ID entirely**
The dataset contains no plant species information. Keeping Plant_ID would cause the model to learn plant-specific patterns rather than generalizable stress physiology — useless for any new plant at deployment time.

---

## Environment

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.7.0+cu128 |
| XGBoost | 2.1.3 |
| scikit-learn | 1.5.2 |
| SHAP | 0.46.0 |
| ONNX | 1.17.0 |
| pandas | 2.2.3 |
| numpy | 1.26.4 |

---

