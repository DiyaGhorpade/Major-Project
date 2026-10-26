"""Deterministic, leakage-safe feature engineering shared by training and inference.

Rules enforced here (and covered by tests/test_features.py):
* every per-plant time-series feature is computed WITHIN a plant after sorting by (plant_id, timestamp);
* only the current and PAST rows are used (trailing windows, shifts, past-only baselines);
* cross-plant features (``*_ref_dev*``) compare a plant with the configured REFERENCE plants (the healthy controls) using
  an as-of join: each row sees, per reference plant, that plant's latest reading at or before the row's own timestamp
  (within ``tolerance_minutes``). Later reference readings are never used, and a plant is never its own reference;
* no centred windows, no whole-history statistics;
* zero / negative elapsed time never produces inf (the rate becomes NaN).
Missing sensor values stay NaN here; imputation (if any) happens in fitted model pipelines.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import Config
from .schema import SENSOR_COLUMNS

BASE_FEATURES = list(SENSOR_COLUMNS)
DERIVED_FEATURES = ["vpd", "log_light", "light_soil_interaction", "soil_rate_per_hr"]
TIME_FEATURES = ["hour", "dayofweek", "hour_sin", "hour_cos", "elapsed_hours"]
TS_COLUMNS = ["soil", "temperature", "humidity", "light", "vpd"]       # lag / diff / trend / rolling
BASELINE_COLUMNS = ["soil", "temperature", "humidity", "log_light", "vpd"]
REFERENCE_COLUMNS = ["soil", "temperature", "humidity", "log_light", "vpd"]   # cross-plant deviations
REF_COUNT_COLUMN = "_ref_plants_used"     # diagnostics only (never a model input): reference plants found per row


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
        "long_windows": [int(w) for w in cfg.get("features.long_windows", []) or []],
        "cross": cross_plant_params(cfg),
    }


def feature_settings(cfg: Config) -> dict:
    """Normalised feature-engineering settings (defaults applied); what serving rebuilds features from."""
    return _params(cfg)


def cross_plant_params(cfg: Config) -> dict:
    cp = cfg.get("features.cross_plant", None) or {}
    return {
        "enabled": bool(cp.get("enabled", False)),
        "reference_plants": [int(p) for p in cp.get("reference_plants", []) or []],
        "statistic": str(cp.get("statistic", "median")),
        "tolerance_minutes": float(cp.get("tolerance_minutes", 30)),
        "windows": [int(w) for w in cp.get("windows", []) or []],
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
    for c in BASELINE_COLUMNS:
        for w in p["long_windows"]:
            names.append(f"{c}_long{w}_mean")
            if w != p["baseline_window"]:                     # the baseline_window deviation already exists as *_base_dev
                names.append(f"{c}_long{w}_dev")
    names += cross_plant_feature_names(cfg)
    return names


def cross_plant_feature_names(cfg: Config) -> list[str]:
    cp = cross_plant_params(cfg)
    if not cp["enabled"]:
        return []
    names = []
    for c in REFERENCE_COLUMNS:
        names.append(f"{c}_ref_dev")
        names += [f"{c}_ref_dev_roll{w}" for w in cp["windows"]]
    return names


def required_history_rows(cfg: Config) -> int:
    """Rows of per-plant history (incl. current) needed for features identical to full-history features."""
    p = _params(cfg)
    return max([p["baseline_window"] + 1, max(p["windows"]), 4] + [w + 1 for w in p["long_windows"]]
               + list(p["cross"]["windows"] if p["cross"]["enabled"] else []))


def build_features(df: pd.DataFrame, cfg: Config, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a frame sorted by (plant_id, timestamp) with all features appended.

    Pass-through columns (labels, ids, ...) are preserved for splitting/evaluation only.
    The original row position is kept in ``_orig_index``.

    ``reference`` (optional) holds extra readings of the reference plants (timestamp, plant_id + sensors) used ONLY for
    the cross-plant features; it adds no output rows. Reference-plant rows already in ``df`` are used as well (that is
    how training works: every plant is in one frame). When a (plant, timestamp) appears in both, ``df`` wins.
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

    for c in BASELINE_COLUMNS:
        gc = g[c]
        for w in p["long_windows"]:
            feats[f"{c}_long{w}_mean"] = gc.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            if w != p["baseline_window"]:
                past = gc.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=p["baseline_min_periods"]).mean())
                feats[f"{c}_long{w}_dev"] = out[c] - past

    out = pd.concat([out, pd.DataFrame(feats, index=out.index)], axis=1)
    if p["cross"]["enabled"]:
        out = _add_cross_plant(out, reference, p["cross"])
    names = all_feature_names(cfg)
    floats = out[names].astype("float64")
    return pd.concat([out.drop(columns=names), floats], axis=1)[list(out.columns)]


def _reference_pool(out: pd.DataFrame, reference: pd.DataFrame | None, plants: list[int]) -> pd.DataFrame:
    cols = ["timestamp", "plant_id"] + REFERENCE_COLUMNS
    parts = [out.loc[out["plant_id"].isin(plants), cols]]
    if reference is not None and len(reference):
        missing = [c for c in ["timestamp", "plant_id"] + SENSOR_COLUMNS if c not in reference.columns]
        if missing:
            raise ValueError(f"build_features: reference readings miss columns {missing}")
        r = reference.loc[reference["plant_id"].isin(plants) & reference["timestamp"].notna()].copy()
        r["vpd"] = compute_vpd(r["temperature"].to_numpy(dtype="float64"), r["humidity"].to_numpy(dtype="float64"))
        r["log_light"] = np.log1p(r["light"].astype("float64").clip(lower=0))
        parts.append(r[cols])
    pool = pd.concat(parts, ignore_index=True)
    pool["plant_id"] = pool["plant_id"].astype(int)
    return pool.drop_duplicates(["plant_id", "timestamp"], keep="first")


def _add_cross_plant(out: pd.DataFrame, reference: pd.DataFrame | None, cp: dict) -> pd.DataFrame:
    """``{c}_ref_dev`` = own value - median (or mean) of the reference plants' latest readings at or before this row's
    timestamp (as-of join, within the tolerance, own plant excluded), plus trailing per-plant means of that deviation."""
    plants = cp["reference_plants"]
    if not plants:
        raise ValueError("features.cross_plant.enabled needs reference_plants")
    pool = _reference_pool(out, reference, plants)
    order = np.argsort(out["timestamp"].to_numpy(), kind="mergesort")
    left = pd.DataFrame({"timestamp": out["timestamp"].to_numpy()[order], "_row": order})
    tol = pd.Timedelta(minutes=cp["tolerance_minutes"])
    own = out["plant_id"].astype(int).to_numpy()
    stacks = {c: np.full((len(out), len(plants)), np.nan) for c in REFERENCE_COLUMNS}
    for j, rp in enumerate(plants):
        right = pool.loc[pool["plant_id"] == rp, ["timestamp"] + REFERENCE_COLUMNS].sort_values("timestamp", kind="mergesort")
        if right.empty:
            continue
        m = pd.merge_asof(left, right, on="timestamp", direction="backward", tolerance=tol, allow_exact_matches=True)
        rows = m["_row"].to_numpy()
        for c in REFERENCE_COLUMNS:
            v = m[c].to_numpy(dtype="float64", copy=True)
            v[own[rows] == rp] = np.nan                       # a plant is never its own reference
            stacks[c][rows, j] = v
    agg = np.nanmedian if cp["statistic"] == "median" else np.nanmean
    feats: dict[str, pd.Series] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows -> NaN reference (no plants found)
        for c in REFERENCE_COLUMNS:
            feats[f"{c}_ref_dev"] = pd.Series(out[c].to_numpy(dtype="float64") - agg(stacks[c], axis=1), index=out.index)
    present = np.zeros(len(out), dtype=int)
    for j in range(len(plants)):
        present += np.any(~np.isnan(np.stack([stacks[c][:, j] for c in REFERENCE_COLUMNS], axis=1)), axis=1)
    frame = pd.DataFrame(feats, index=out.index)
    frame["plant_id"] = out["plant_id"].to_numpy()
    g = frame.groupby("plant_id", sort=False)
    for c in REFERENCE_COLUMNS:
        for w in cp["windows"]:
            feats[f"{c}_ref_dev_roll{w}"] = g[f"{c}_ref_dev"].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    extra = pd.DataFrame(feats, index=out.index)
    extra[REF_COUNT_COLUMN] = present
    return pd.concat([out, extra], axis=1)
