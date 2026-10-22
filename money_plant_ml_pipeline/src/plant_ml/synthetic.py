"""Deterministic SYNTHETIC data generator (Bengaluru-like indoor / shaded-balcony assumptions).

These are simulation assumptions - not verified weather observations and not universal plant-health thresholds.
Stress labels come from a latent severity curve; classes are balanced BY CONSTRUCTION (see docs/DATA_DICTIONARY.md)
and do not reflect real-world prevalence.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .schema import (
    EXPERIMENTAL_GROUPS, OPTIONAL_COLUMNS, REQUIRED_TRAIN_COLUMNS, SENSOR_RANGES, STRESS_LEVELS, plant_to_group,
)

OBS_PER_DAY = 12  # every 2 h -> sampling_point 1..12
N_PLANTS = 16
GROUP_STRESS_TYPE = {"healthy_control": "none", "water_stress": "water", "light_stress": "light",
                     "temperature_humidity_stress": "temperature_humidity"}


def _level_from_severity(s: float) -> str:
    # Equal-width severity bins -> balanced classes within stressed groups (documented balancing mechanism).
    if s < 0.25:
        return "healthy"
    if s < 0.50:
        return "low"
    if s < 0.75:
        return "moderate"
    return "high"


def generate(n_days: int = 45, seed: int = 42, start_date: str = "2026-09-21") -> pd.DataFrame:
    rng_master = np.random.default_rng(seed)
    start = pd.Timestamp(start_date, tz="Asia/Kolkata")
    n_obs = n_days * OBS_PER_DAY
    t_idx = np.arange(n_obs)
    hour = (t_idx % OBS_PER_DAY) * 2
    day = t_idx // OBS_PER_DAY
    stamps = [start + pd.Timedelta(hours=2 * int(i)) for i in t_idx]

    # Shared "weather": per-day offsets, rainy days, warm afternoons (same for all plants, like one room/balcony)
    day_temp_off = rng_master.normal(0, 1.0, n_days) + 1.2 * np.sin(2 * np.pi * np.arange(n_days) / 30.0)
    rainy = rng_master.random(n_days) < 0.22
    warm_pm = rng_master.random(n_days) < 0.10

    rows = []
    for plant in range(1, N_PLANTS + 1):
        rng = np.random.default_rng(seed * 1000 + plant)
        group = plant_to_group(plant)
        idx_in_group = EXPERIMENTAL_GROUPS[group].index(plant)
        # ---- latent severity (triangular wave, plant-specific period & phase) ----
        if group == "healthy_control":
            sev = np.clip(0.03 + 0.04 * rng.random(n_obs), 0, 1)
            for _ in range(max(1, n_days // 18)):  # a few mild 'low' episodes (labelled low)
                d0 = int(rng.integers(0, max(1, n_days - 2)))
                m = (day >= d0) & (day < d0 + 2)
                sev[m] = 0.3
        else:
            period = n_days / (1.6 + 0.1 * idx_in_group)
            phase = rng.random() * period
            x = ((day + hour / 24.0 + phase) % period) / period
            sev = 1.0 - np.abs(2 * x - 1.0)  # 0..1..0 triangular
            sev = np.clip(sev + rng.normal(0, 0.02, n_obs), 0, 1)
        level = np.array([_level_from_severity(s) for s in sev])

        # ---- environment ----
        base_light_peak = rng.choice([2500, 3500, 4500, 9000, 13000], p=[0.25, 0.25, 0.2, 0.15, 0.15])
        base_soil = 50 + rng.normal(0, 4)
        temp = (23.5 + day_temp_off[day] + 5.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
                + np.where(warm_pm[day] & (hour >= 13) & (hour <= 17), rng.uniform(3, 5), 0.0)
                - np.where(rainy[day], 2.0, 0.0) + rng.normal(0, 0.4, n_obs))
        hum = 62 - 2.2 * (temp - 23.5) + np.where(rainy[day], 15, 0) + rng.normal(0, 2.0, n_obs)
        daylight = np.clip(np.sin(np.pi * (hour - 6) / 12.0), 0, None)
        cloud = np.where(rainy[day], rng.uniform(0.25, 0.6, n_obs), rng.uniform(0.7, 1.15, n_obs))
        light = base_light_peak * daylight * cloud
        night = daylight == 0
        light = np.where(night, rng.integers(0, 3, n_obs), light)

        mixed_plant = group != "healthy_control" and idx_in_group >= 2
        is_mixed = mixed_plant & (level == "high") & (rng.random(n_obs) < 0.6)

        if group == "light_stress":
            if idx_in_group % 2 == 0:
                light = np.where(night, light, light * (1 - 0.9 * sev))          # deep shade
            else:
                light = np.where(night, light, light * (1 + 6.0 * sev))          # harsh sun (up to ~60k with bright base)
        if group == "temperature_humidity_stress":
            temp = temp + 7.0 * sev
            hum = hum - 28.0 * sev
        if is_mixed.any():
            temp = temp + np.where(is_mixed, 3.0, 0.0)

        vpd_proxy = np.clip(0.6108 * np.exp(17.27 * temp / (temp + 237.3)) * (1 - np.clip(hum, 20, 100) / 100), 0, None)
        if group == "water_stress":
            soil = base_soil * (1 - 0.78 * sev) + rng.normal(0, 1.0, n_obs)       # declines with severity, recovers when it falls
        elif group == "healthy_control":
            cycle = ((day + hour / 24.0 + idx_in_group * 1.3) % 4.0) / 4.0          # dries over ~4 days then is watered
            soil = (base_soil + 8) - 22 * cycle + rng.normal(0, 1.0, n_obs)
        else:
            cycle = ((day + hour / 24.0 + idx_in_group * 1.1) % 4.5) / 4.5
            soil = (base_soil + 6) - 20 * cycle - 12 * sev * (is_mixed.astype(float)) + rng.normal(0, 1.0, n_obs)
        soil = soil - 1.2 * (vpd_proxy - 1.0)

        stype = np.where(level == "healthy", "none", GROUP_STRESS_TYPE[group])
        if group == "healthy_control":
            stype = np.where(level == "healthy", "none", "none")   # mild episodes are 'low' with no stress type
        stype = np.where(is_mixed, "mixed", stype)

        # ---- measurement noise, anomalies, missing values ----
        soil_m, temp_m, hum_m, light_m = soil.copy(), temp.copy(), hum.copy(), light.copy()
        soil_m += rng.normal(0, 0.5, n_obs)
        temp_m += rng.normal(0, 0.2, n_obs)
        hum_m += rng.normal(0, 0.8, n_obs)
        light_m = np.where(night, light_m, light_m * rng.normal(1.0, 0.04, n_obs))
        anomaly = np.zeros(n_obs, dtype=int)
        n_anom = max(2, int(round(0.012 * n_obs)))
        for a in rng.choice(n_obs, size=n_anom, replace=False):
            kind = rng.integers(0, 5)
            anomaly[a] = 1
            if kind == 0:
                soil_m[a] = float(rng.choice([0.0, 100.0]))            # sensor dropout / saturation
            elif kind == 1:
                temp_m[a] += rng.uniform(8, 12)                         # temperature spike
            elif kind == 2:
                hum_m[a] = float(rng.choice([20.0, 100.0]))             # humidity stuck at rail
            elif kind == 3:
                light_m[a] = light_m[a] * rng.uniform(6, 12) + 5000     # light spike
            else:
                light_m[a] = 0.0 if hour[a] in (10, 12, 14) else light_m[a]   # subtle: dark at midday
                soil_m[a] += rng.choice([-1, 1]) * rng.uniform(12, 20)
        for arr, key in ((soil_m, "soil"), (temp_m, "temperature"), (hum_m, "humidity"), (light_m, "light")):
            lo, hi = SENSOR_RANGES[key]
            np.clip(arr, lo, hi, out=arr)
        light_m = np.round(light_m)
        for arr in (soil_m, temp_m, hum_m):
            np.round(arr, 1, out=arr)
        miss = {k: rng.random(n_obs) < 0.015 for k in ("soil", "temperature", "humidity", "light")}
        soil_m = np.where(miss["soil"], np.nan, soil_m)
        temp_m = np.where(miss["temperature"], np.nan, temp_m)
        hum_m = np.where(miss["humidity"], np.nan, hum_m)
        light_m = np.where(miss["light"], np.nan, light_m)

        sec = (rng.integers(0, 60, n_obs))  # per-plant sub-minute jitter, like asynchronous nodes
        stamp_strs = [(s + pd.Timedelta(seconds=int(j) * (plant % 2))).isoformat() for s, j in zip(stamps, sec)]
        rows.append(pd.DataFrame({
            "timestamp": stamp_strs, "session_id": day // 7 + 1, "node_id": 1 if plant % 2 else 2,
            "sampling_point": (t_idx % OBS_PER_DAY) + 1, "plant_id": plant, "soil": soil_m, "temperature": temp_m,
            "humidity": hum_m, "light": light_m, "stress_level": level, "stress_type": stype,
            "experimental_group": group, "is_synthetic_anomaly": anomaly,
        }))
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["timestamp", "plant_id"], kind="mergesort").reset_index(drop=True)
    return df[REQUIRED_TRAIN_COLUMNS + OPTIONAL_COLUMNS]


def check_generated(df: pd.DataFrame, min_rows: int = 8000) -> None:
    """Assertions required by the spec (row count, ranges, group sizes, classes, duplicates, chronology)."""
    assert len(df) >= min_rows, f"expected >= {min_rows} rows, got {len(df)}"
    assert list(df.columns) == REQUIRED_TRAIN_COLUMNS + OPTIONAL_COLUMNS, "column order mismatch"
    assert df["plant_id"].nunique() == 16
    gsize = df.groupby("experimental_group")["plant_id"].nunique()
    assert (gsize == 4).all() and len(gsize) == 4, f"group sizes: {gsize.to_dict()}"
    for col, (lo, hi) in SENSOR_RANGES.items():
        v = df[col].dropna()
        assert v.between(lo, hi).all(), f"{col} outside [{lo}, {hi}]"
    assert set(df["stress_level"].unique()) == set(STRESS_LEVELS), "not all four stress levels present"
    assert not df.duplicated(["plant_id", "timestamp"]).any(), "duplicate (plant_id, timestamp)"
    ts = pd.to_datetime(df["timestamp"], utc=True)
    for _, g in df.assign(_t=ts).groupby("plant_id"):
        assert g["_t"].is_monotonic_increasing, "timestamps not chronological within a plant"
    assert df["is_synthetic_anomaly"].isin([0, 1]).all()


def n_days_for_rows(rows: int) -> int:
    return int(math.ceil(rows / (N_PLANTS * OBS_PER_DAY)))
