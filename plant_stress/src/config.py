# src/config.py

import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "plant_health_data.csv"
MODEL_DIR   = ROOT / "outputs" / "models"
PLOT_DIR    = ROOT / "outputs" / "plots"
REPORT_DIR  = ROOT / "outputs" / "reports"

for d in [MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────
SEED = 42

# ── Dataset ────────────────────────────────────────────────────────
LABEL_COL      = "Plant_Health_Status"
TIMESTAMP_COL  = "Timestamp"
DROP_COLS      = ["Plant_ID", "Timestamp"]   # never fed to model

LABEL_MAP = {
    "Healthy":         0,
    "Moderate Stress": 1,
    "High Stress":     2,
}
LABEL_NAMES = ["Healthy", "Moderate Stress", "High Stress"]

# Features used (after dropping + adding derived features)
FEATURE_COLS = [
    "Soil_Moisture",
    "Ambient_Temperature",
    "Soil_Temperature",
    "Humidity",
    "Light_Intensity",
    "Soil_pH",
    "Nitrogen_Level",
    "Phosphorus_Level",
    "Potassium_Level",
    "Chlorophyll_Content",
    "Electrochemical_Signal",
    "Thermal_Gradient",   # derived
    "VPD_proxy",          # derived
]

# ── Split ──────────────────────────────────────────────────────────
TRAIN_RATIO    = 0.80    # first 80% of readings per plant → train
N_CV_FOLDS     = 5

# ── Preprocessing ──────────────────────────────────────────────────
SCALER         = "standard"   # options: "standard" | "minmax" | "robust"

# ── Model Hyperparameters ──────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

XGB_PARAMS = {
    "n_estimators":       500,
    "max_depth":          6,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "eval_metric":        "mlogloss",
    "device":             "cuda",      # RTX 5060
    "random_state":       SEED,
    "n_jobs":             -1,
}

LR_PARAMS = {
    "max_iter":           2000,
    "class_weight":       "balanced",
    "random_state":       SEED,
    "solver":             "lbfgs",
    "C":                  1.0,
}

MLP_PARAMS = {
    "hidden_dims":        [64, 32],
    "dropout":            0.3,
    "learning_rate":      1e-3,
    "batch_size":         64,
    "epochs":             100,
    "patience":           10,       # early stopping
    "weight_decay":       1e-4,
}

# ── Evaluation ─────────────────────────────────────────────────────
SHAP_MAX_SAMPLES = 200   # keep SHAP fast on CPU fallback
