"""Deterministic, leakage-safe feature engineering shared by training and inference.

Rules enforced here (and covered by tests/test_features.py):
* every time-series feature is computed WITHIN a plant after sorting by (plant_id, timestamp);
* only the current and PAST rows are used (trailing windows, shifts, past-only baselines);
* no centred windows, no whole-history statistics;
* zero / negative elapsed time never produces inf (the rate becomes NaN).
Missing sensor values stay NaN here; imputation (if any) happens in fitted model pipelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .schema import SENSOR_COLUMNS

BASE_FEATURES = list(SENSOR_COLUMNS)
DERIVED_FEATURES = ["vpd", "log_light", "light_soil_interaction", "soil_rate_per_hr"]
TIME_FEATURES = ["hour", "dayofweek", "hour_sin", "hour_cos", "elapsed_hours"]
TS_COLUMNS = ["soil", "temperature", "humidity", "light", "vpd"]       # lag / diff / trend / rolling
BASELINE_COLUMNS = ["soil", "temperature", "humidity", "log_light", "vpd"]


def compute_vpd(temperature, humidity):
    """Approximate AIR vapour-pressure deficit in kPa (Tetens). Not a leaf-level measurement.

    es = 0.6108 * exp(17.27 T / (T + 237.3));  ea = es * RH / 100;  vpd = es - ea
    """
    t = np.asarray(temperature, dtype="float64")
    h = np.asarray(humidity, dtype="float64")
    es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
    ea = es * h / 100.0
    return es - ea


def _params(cfg: Config) -> dict:
    return {
        "windows": [int(w) for w in cfg.get("features.rolling_windows", [3, 6, 12])],
        "baseline_window": int(cfg.get("features.baseline_window", 48)),
        "baseline_min_periods": int(cfg.get("features.baseline_min_periods", 6)),
        "std_min_periods": int(cfg.get("features.std_min_periods", 2)),
        "max_light": float(cfg.get("features.max_light", 100000)),
    }


def all_feature_names(cfg: Config) -> list[str]:
    """Ordered list of every engineered + raw feature the pipeline can produce."""
    p = _params(cfg)
    names = list(BASE_FEATURES) + list(DERIVED_FEATURES)
    for c in TS_COLUMNS:
        names += [f"{c}_lag1", f"{c}_diff1", f"{c}_trend3"]
    for c in TS_COLUMNS:
        for w in p["windows"]:
            names += [f"{c}_roll{w}_mean", f"{c}_roll{w}_std"]
    names += [f"{c}_base_dev" for c in BASELINE_COLUMNS]
    names += list(TIME_FEATURES)
    return names


def required_history_rows(cfg: Config) -> int:
    """Rows of per-plant history (incl. current) needed for features identical to full-history features."""
    p = _params(cfg)
    return max(p["baseline_window"] + 1, max(p["windows"]), 4)


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a frame sorted by (plant_id, timestamp) with all features appended.

    Pass-through columns (labels, ids, ...) are preserved for splitting/evaluation only.
    The original row position is kept in ``_orig_index``.
    """
    required = ["timestamp", "plant_id"] + SENSOR_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"build_features: missing columns {missing}")
    if df["timestamp"].isna().any() or df["plant_id"].isna().any():
        raise ValueError("build_features: timestamp / plant_id contain NaN; run validation (quarantine) first")
    p = _params(cfg)

    out = df.copy()
    out["_orig_index"] = np.arange(len(out))
    out = out.sort_values(["plant_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    out["vpd"] = compute_vpd(out["temperature"].to_numpy(), out["humidity"].to_numpy())
    out["log_light"] = np.log1p(out["light"].clip(lower=0))
    out["light_soil_interaction"] = (out["log_light"] / np.log1p(p["max_light"])) * out["soil"]

    g = out.groupby("plant_id", sort=False)
    dt_hours = g["timestamp"].diff().dt.total_seconds() / 3600.0
    safe_dt = dt_hours.where(dt_hours > 0)                       # zero / negative -> NaN (no division by zero)
    out["soil_rate_per_hr"] = g["soil"].diff() / safe_dt
    out["soil_rate_per_hr"] = out["soil_rate_per_hr"].replace([np.inf, -np.inf], np.nan)

    feats: dict[str, pd.Series] = {}
    for c in TS_COLUMNS:
        gc = g[c]
        lag1 = gc.shift(1)
        feats[f"{c}_lag1"] = lag1
        feats[f"{c}_diff1"] = out[c] - lag1
        feats[f"{c}_trend3"] = out[c] - gc.shift(3)
        for w in p["windows"]:
            feats[f"{c}_roll{w}_mean"] = gc.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            feats[f"{c}_roll{w}_std"] = gc.transform(
                lambda s, w=w: s.rolling(w, min_periods=p["std_min_periods"]).std())
    for c in BASELINE_COLUMNS:
        # Strictly past-only rolling baseline: shift(1) excludes the current row.
        base = g[c].transform(lambda s: s.shift(1).rolling(p["baseline_window"], min_periods=p["baseline_min_periods"]).mean())
        feats[f"{c}_base_dev"] = out[c] - base

    ts = out["timestamp"]
    hour_float = ts.dt.hour + ts.dt.minute / 60.0
    feats["hour"] = ts.dt.hour.astype("float64")
    feats["dayofweek"] = ts.dt.dayofweek.astype("float64")
    feats["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0)
    feats["elapsed_hours"] = (ts - g["timestamp"].transform("min")).dt.total_seconds() / 3600.0

    out = pd.concat([out, pd.DataFrame(feats, index=out.index)], axis=1)
    for name in all_feature_names(cfg):
        out[name] = out[name].astype("float64")
    return out
