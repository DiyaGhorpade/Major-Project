# Feature engineering
All features: sort by (plant_id, timestamp); group by plant; current/past rows only; NaN stays NaN.
* **vpd** (kPa) = es - ea, es = 0.6108 exp(17.27 T / (T + 237.3)), ea = es RH/100. Approximate air VPD, not leaf-level.
* **log_light** = log1p(light). **light_soil_interaction** = log_light / log1p(100000) x soil.
* **soil_rate_per_hr** = soil diff / elapsed hours; elapsed <= 0 -> NaN (never inf).
* **{soil,temperature,humidity,light,vpd}_lag1 / _diff1 / _trend3** (value - value 3 rows ago).
* **_roll{3,6,12}_mean / _std**: trailing windows (`min_periods=1` for mean, 2 for std); never centred.
* **{soil,temperature,humidity,log_light,vpd}_base_dev** = value - mean of the previous 48 observations (excludes the current row; NaN until 6 past rows: warm-up).
* **hour, dayofweek, hour_sin, hour_cos, elapsed_hours** (elapsed_hours not used by the classifier).
History needed for exact parity with full-history features: 49 rows per plant (`required_history_rows_for_full_features`). Tests (`tests/test_features.py`) cover: truncation invariance (no future leakage), perturbing the future does not change the past, perturbing another plant does not change this plant, input order independence, zero elapsed time, trailing-not-centred windows, past-only baseline, history-window parity.
