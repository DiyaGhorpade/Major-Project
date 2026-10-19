# Sensor data receiver

`receiver.py` is the current Flask HTTP entry point. It validates incoming
nine-field CSV rows and appends accepted rows to `sensor_data.csv`.

From the repository root, with the root environment installed:

```powershell
cd sensor_data_reciever
..\.venv\Scripts\python.exe receiver.py
```

The server listens on port 5000 and exposes `GET /health` and `POST /upload`.
Run from this directory because output paths are relative to the working directory.

## Contents

- `receiver.py`: current HTTP server with its own validation and settings.
- `config.py`, `validator.py`, `storage.py`: separate utilities retained alongside
  the current server; the server does not import them.
- `tests/`: unit, property, and integration tests; some still target earlier
  receiver interfaces and need alignment with the current HTTP implementation.
- `sample.csv`: legacy integration-test input.
- `sensor_data.csv`, `error_log.csv`: existing receiver output files.
- `docs/`: Jetson deployment guide.
- `archive/`: historical serial receiver and earlier error log.

The dashboard replay CSV has moved to
`../dashboard/data/sample_data_updated.csv`. It is demo input, not a receiver
output file. Its column order also differs from the current HTTP upload schema.
