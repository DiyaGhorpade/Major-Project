# Jetson Orin Nano deployment (CPU only)
**Not tested on Jetson hardware** - the build environment was x86_64 Linux. Everything below is guidance; measure on the device before relying on it.
1. Check platform: `uname -m` (expect `aarch64`), `python3 --version` (JetPack 6 -> 3.10), `cat /etc/nv_tegra_release`.
2. `sudo apt install python3-venv python3-dev build-essential`; `python3 -m venv .venv && source .venv/bin/activate`.
3. `pip install -r requirements.txt` (aarch64 wheels exist for numpy, pandas, scipy, scikit-learn, xgboost >= 2.0). Do **not** install desktop CUDA PyTorch; it is not needed. `requirements-optional.txt` (SHAP/matplotlib) can be skipped: explanations then report "unavailable".
4. Either copy trained artifacts (`python scripts/export_artifacts.py` -> extract in the Jetson project root) **only if** scikit-learn/xgboost major.minor versions match (the API refuses otherwise), or simply retrain on the Jetson (about 10 s on the x86 dev machine for the supplied data; Jetson time unmeasured).
5. Smoke test: `python scripts/package_info.py`, `python scripts/run_inference.py --input data/sample/sample_sensor_readings.csv --plant-id 3`, `python -m pytest -q`.
6. Run: `uvicorn plant_ml.api.main:app --host 127.0.0.1 --port 8001`. **Keep localhost** if the backend runs on the same Jetson. Binding `0.0.0.0` exposes unauthenticated prediction endpoints to your LAN: only do so behind a firewall (`ufw allow from <backend-ip> to any port 8001`). There is no authentication.
7. Resource profile (measured on the x86 build machine, NOT Jetson): artifact sizes: classifier 2.1 MB, anomaly 1.9 MB, clustering 19 KB; single `/predict` about 0.17 s average without SHAP. RAM/disk on Jetson: measure with `/usr/bin/time -v` and `tegrastats`.
8. Run as a service with systemd (`ExecStart=<venv>/bin/uvicorn plant_ml.api.main:app --host 127.0.0.1 --port 8001`, `WorkingDirectory=<project root>`, `Restart=on-failure`).
