# src/preprocess.py

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from config import (
    DATA_PATH, MODEL_DIR,
    LABEL_COL, TIMESTAMP_COL, DROP_COLS,
    LABEL_MAP, LABEL_NAMES, FEATURE_COLS,
    TRAIN_RATIO, SCALER, SEED
)
from utils import set_seed


# ── Scaler factory ─────────────────────────────────────────────────────────────

def get_scaler():
    options = {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
        "robust":   RobustScaler(),
    }
    if SCALER not in options:
        raise ValueError(f"Unknown scaler '{SCALER}'. Choose from {list(options)}")
    return options[SCALER]


# ── Feature engineering ────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-informed derived features.

    Thermal_Gradient:
        Ambient_Temperature - Soil_Temperature
        Captures heat stress relationship between air and root zone.
        Hot air + hot soil = drought stress signal.

    VPD_proxy:
        Vapor Pressure Deficit approximation using Tetens formula.
        VPD = (1 - RH/100) * 0.6108 * exp(17.27*T / (T + 237.3))
        High VPD = plant transpires harder = water stress.
        Neither temperature nor humidity alone captures this.
    """
    df = df.copy()

    df["Thermal_Gradient"] = (
        df["Ambient_Temperature"] - df["Soil_Temperature"]
    )

    T  = df["Ambient_Temperature"]
    RH = df["Humidity"]
    df["VPD_proxy"] = (
        (1 - RH / 100) * 0.6108 * np.exp(17.27 * T / (T + 237.3))
    )

    return df


# ── Temporal split ─────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    """
    Split per plant by time — first train_ratio of readings → train,
    remainder → test.

    WHY NOT random split:
        A random split leaks future context into training.
        Reading at t and t+1 from the same plant could land in different
        splits — the model effectively sees the neighbourhood of every
        test point during training, inflating accuracy by 5-15%.
        Temporal split simulates real deployment: always predicting
        the future from the past.
    """
    train_frames = []
    test_frames  = []

    for plant_id, group in df.groupby("Plant_ID"):
        group = group.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        n_train = int(len(group) * train_ratio)

        train_frames.append(group.iloc[:n_train])
        test_frames.append(group.iloc[n_train:])

    df_train = pd.concat(train_frames).reset_index(drop=True)
    df_test  = pd.concat(test_frames).reset_index(drop=True)

    return df_train, df_test


# ── Class weights ──────────────────────────────────────────────────────────────

def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute balanced class weights from training labels.
    Passed to sklearn models as class_weight and to PyTorch
    loss as weight tensor.

    Returns dict: {class_int: weight_float}
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    return dict(zip(classes, weights))


# ── Main pipeline ──────────────────────────────────────────────────────────────

def load_and_preprocess(verbose: bool = True):
    """
    Full preprocessing pipeline. Returns:
        X_train, X_test  : np.ndarray  (n_samples, n_features)
        y_train, y_test  : np.ndarray  (n_samples,)  integer encoded
        scaler           : fitted scaler (saved to MODEL_DIR)
        class_weights    : dict {class_int: weight}
        feature_names    : list[str]   (column order the model expects)
    """
    set_seed()

    # ── 1. Load ────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

    if verbose:
        print(f"[data]  Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"[data]  Plants       : {sorted(df['Plant_ID'].unique())}")
        print(f"[data]  Label dist   :\n{df[LABEL_COL].value_counts().to_string()}")
        print(f"[data]  Null values  : {df.isnull().sum().sum()}")

    # ── 2. Encode label ────────────────────────────────────────────
    df["label"] = df[LABEL_COL].map(LABEL_MAP)
    assert df["label"].isnull().sum() == 0, \
        "Label mapping failed — unexpected values in Plant_Health_Status"

    # ── 3. Derived features ────────────────────────────────────────
    df = add_derived_features(df)

    if verbose:
        print(f"[feat]  Derived features added: Thermal_Gradient, VPD_proxy")

    # ── 4. Verify all expected features exist ──────────────────────
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    # ── 5. Temporal split ──────────────────────────────────────────
    df_train, df_test = temporal_split(df)

    if verbose:
        print(f"\n[split] Train : {len(df_train)} rows")
        print(f"[split] Test  : {len(df_test)} rows")
        print(f"[split] Train label dist:")
        for cls, n in df_train[LABEL_COL].value_counts().items():
            print(f"         {cls:20s}: {n}")
        print(f"[split] Test label dist:")
        for cls, n in df_test[LABEL_COL].value_counts().items():
            print(f"         {cls:20s}: {n}")

    # ── 6. Extract X, y ────────────────────────────────────────────
    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    X_test  = df_test[FEATURE_COLS].values.astype(np.float32)
    y_train = df_train["label"].values.astype(np.int64)
    y_test  = df_test["label"].values.astype(np.int64)

    # ── 7. Scale — fit on train only ───────────────────────────────
    # CRITICAL: scaler sees only training distribution.
    # Fitting on full data leaks test statistics into the scaler
    # (data leakage), making eval metrics overly optimistic.
    scaler = get_scaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Save scaler — must be used identically at inference time
    scaler_path = MODEL_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    if verbose:
        print(f"\n[scale] Scaler : {SCALER}")
        print(f"[scale] Saved  : {scaler_path}")

    # ── 8. Class weights ───────────────────────────────────────────
    class_weights = get_class_weights(y_train)

    if verbose:
        print(f"\n[weights] Class weights (balanced):")
        for cls_int, w in class_weights.items():
            print(f"          {LABEL_NAMES[cls_int]:20s} ({cls_int}): {w:.4f}")

    # ── 9. Final shape report ──────────────────────────────────────
    if verbose:
        print(f"\n[done]  X_train : {X_train.shape}")
        print(f"[done]  X_test  : {X_test.shape}")
        print(f"[done]  y_train : {y_train.shape}")
        print(f"[done]  y_test  : {y_test.shape}")
        print(f"[done]  Features: {FEATURE_COLS}")

    return X_train, X_test, y_train, y_test, scaler, class_weights, FEATURE_COLS


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, cw, feats = load_and_preprocess()

    print("\n── Sanity checks ──")
    print(f"No NaNs in X_train : {not np.isnan(X_train).any()}")
    print(f"No NaNs in X_test  : {not np.isnan(X_test).any()}")
    print(f"Train mean ≈ 0     : {X_train.mean():.6f}")
    print(f"Train std  ≈ 1     : {X_train.std():.6f}")
    print(f"Unique y_train     : {np.unique(y_train)}")
    print(f"Unique y_test      : {np.unique(y_test)}")
