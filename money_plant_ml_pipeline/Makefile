PY ?= python
INPUT ?= data/raw/plant_sensor_data.csv

.PHONY: install install-dev validate train evaluate infer api test smoke synth clean
install:      ; $(PY) -m pip install -r requirements.txt
install-dev:  ; $(PY) -m pip install -r requirements-dev.txt
synth:        ; $(PY) scripts/generate_synthetic_data.py --output data/sample/synthetic_sensor_readings.csv --rows 8000 --seed 42
validate:     ; $(PY) scripts/validate_data.py --input $(INPUT)
train:        ; $(PY) scripts/train_all.py --input $(INPUT)
evaluate:     ; $(PY) scripts/evaluate_all.py --input $(INPUT)
infer:        ; $(PY) scripts/run_inference.py --input data/sample/sample_sensor_readings.csv
api:          ; $(PY) -m uvicorn plant_ml.api.main:app --host 127.0.0.1 --port 8001
test:         ; $(PY) -m pytest -q
smoke:        validate infer   ## CPU-only smoke test (needs trained artifacts)
clean:        ; rm -rf .pytest_cache **/__pycache__ reports/v* data/processed/*
