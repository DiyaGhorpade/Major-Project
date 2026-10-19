# Plant monitoring project

The repository separates exploratory analysis from hardware and sensor ingestion.

```text
eda/
  notebooks/   Exploratory analysis and modeling notebooks
  data/        Datasets used by the analysis
  scripts/     Standalone analysis scripts
  outputs/     Generated metrics, diagnostic reports, and plots
  reports/     Markdown audits and analysis notes
  docs/        EDA and plant-analysis Word documents
hardware-code/
  data_log_producer_v1_12-remaining-plants/
  data_log_producer_v1_4-light-deprived/
  libraries/   Arduino sensor libraries
sensor_data_reciever/
  receiver.py  HTTP receiver entry point
  config.py    Configuration for the separate validation/storage modules
  validator.py
  storage.py
  tests/
  docs/        Jetson deployment guide
  archive/     Historical receiver artifacts moved from the repository root
Major-Project-Research-Papers/
```

See [EDA instructions](eda/README.md) for datasets and analysis commands.

To start the current HTTP receiver, run from its directory so its relative CSV
output stays with the receiver:

```powershell
cd sensor_data_reciever
python receiver.py
```

The receiver requires Flask. It exposes `GET /health` and `POST /upload` on port
5000. The current ESP32 sketches emit serial CSV; HTTP transmission and a shared
CSV schema still need to be aligned. The HTTP receiver currently defines its own
settings rather than importing `config.py`.

The reorganization preserves the hardware and receiver code and their existing
interfaces. It does not resolve the pre-existing receiver/test inconsistencies.
