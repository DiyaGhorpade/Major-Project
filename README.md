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

## Python 3.12 setup

Use Python 3.12 for this project. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name major-project-py312 --display-name "Python 3.12 (Major-Project)"
```

Select `.venv/Scripts/python.exe` as the IDE interpreter and
`Python 3.12 (Major-Project)` as the notebook kernel. Run tools through
`python -m pip` and `python -m pytest` using that interpreter, rather than global
executables that may still reference an old Python installation.

The root `requirements.txt` contains all analysis, receiver, and test dependencies,
pinned to the versions verified with Python 3.12.10. Use the root `.venv` for all
Python work in this repository.
Virtual environments must be recreated when their base Python is removed;
editing `pyvenv.cfg` alone does not replace incompatible compiled packages.

To start the current HTTP receiver, run from its directory so its relative CSV
output stays with the receiver:

```powershell
cd sensor_data_reciever
..\.venv\Scripts\python.exe receiver.py
```

The receiver requires Flask. It exposes `GET /health` and `POST /upload` on port
5000. The current ESP32 sketches emit serial CSV; HTTP transmission and a shared
CSV schema still need to be aligned. The HTTP receiver currently defines its own
settings rather than importing `config.py`.

The reorganization preserves the hardware and receiver code and their existing
interfaces. It does not resolve the pre-existing receiver/test inconsistencies.
