# Data dictionary
| column | type | meaning / range |
|---|---|---|
| timestamp | ISO 8601 | preferably `+05:30`; normalised to Asia/Kolkata; naive assumed Asia/Kolkata |
| session_id | int >= 1 | weekly collection session (supplied data: 1-4) |
| node_id | "1" / "2" | ESP32 node (odd plants -> node 1, even -> node 2 in the supplied data) |
| sampling_point | int 1-12 | slot within the day (2-hourly in the supplied data) |
| plant_id | int 1-16 | 1-4 healthy_control, 5-8 water_stress, 9-12 light_stress, 13-16 temperature_humidity_stress |
| soil | float % | soil moisture, sensor-calibrated scale, 0-100 |
| temperature | float deg C | ambient, hard range 15-40 |
| humidity | float % RH | hard range 20-100 |
| light | float lux | 0-100000 |
| stress_level | enum | healthy, low, moderate, high (training target) |
| stress_type | enum | none, water, light, temperature_humidity, mixed (not a model input) |
| experimental_group | enum (optional) | derived from plant_id for reports when absent (the supplied CSV has none) |
| is_synthetic_anomaly | 0/1 (optional) | helper evaluation label only; never a model input |
Observations about the supplied file: 5,376 rows, 16 plants x 336 readings (2-hourly, 4 weeks), 1.3-1.6% missing per sensor, 65 helper-anomaly rows (1.2%), no range violations, no duplicates; timestamps for plants carry a per-plant sub-minute offset (asynchronous nodes). Stress levels are roughly balanced overall, but healthy-control plants only ever contain `healthy`/`low`, and session 1 has no `high` rows - consistent with a simulator, not with real prevalence.
