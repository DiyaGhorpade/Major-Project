# Plant Health Monitoring

A sensor-based plant health monitoring project for tracking indoor growing
conditions and detecting plant stress. ESP32 nodes collect soil moisture,
temperature, humidity, and light readings, while a Python receiver validates and
stores incoming measurements for analysis.

The project explores machine learning to classify plants as healthy, moderately
stressed, or highly stressed. It includes model training and evaluation,
exploratory analysis, supporting research, and a browser dashboard that currently
demonstrates monitoring through recorded-data replay. Future work includes live
integration, stress forecasting, plant recommendations, and edge deployment on
NVIDIA Jetson Nano or Raspberry Pi.

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

The root `requirements.txt` contains the EDA, receiver, and test dependencies,
pinned to the versions verified with Python 3.12.10. Use the root `.venv` for all
Python work in this repository.
The separate `plant_stress/` ML pipeline has additional dependencies and its own
environment instructions; see its README before training models.
Virtual environments must be recreated when their base Python is removed;
editing `pyvenv.cfg` alone does not replace incompatible compiled packages.
