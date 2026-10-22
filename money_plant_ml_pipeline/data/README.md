# data/
* `raw/plant_sensor_data.csv` - **your supplied dataset** (5,376 rows). Treated as synthetic (see config `data.provenance`).
* `sample/sample_sensor_readings.csv` - first 3 days (576 rows) of the supplied dataset, for quick smoke tests. **Not** a large dataset; not synthetic-generator output.
* `sample/synthetic_sensor_readings.csv` - 8,064 rows from `scripts/generate_synthetic_data.py --seed 42` (deterministic fallback; Bengaluru-like *simulation* assumptions; class balance is a simulator artefact).
* `processed/` - the only place processed or quarantined data is written.
