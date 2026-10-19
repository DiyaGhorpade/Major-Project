# Plant health monitoring

Two ESP32 nodes collect soil, temperature, humidity and light readings. On the
Jetson, `receiver.py` saves the readings and `scheduler.py` starts each collection:
Node 1 collects 12 points, then Node 2 collects four.

## Start data collection

1. Set up Python using the commands below.
2. Configure and upload the two [ESP32 sketches](hardware-code/README.md).
3. Follow the [receiver and scheduler setup](sensor_data_reciever/README.md) to
   configure addresses, plant mappings and collection times, then run a collection.

Hardware commissioning is **PENDING HARDWARE VERIFICATION**. Use the
[commissioning checklist](docs/hardware_commissioning.md) to record the installed
wiring, sensor checks, calibration and first successful run.

## Python 3.12 setup

Run these commands from the repository root. Python 3.12 must already be installed.

On Linux / Jetson:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install Flask==3.1.3 tzdata
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install Flask==3.1.3 tzdata
```

Activate this environment in each terminal before running the receiver or
scheduler. On Windows, you can also run `.\.venv\Scripts\python.exe` directly
if environment activation is unavailable.

For notebooks, analysis and software tests, install the additional root dependencies:

```sh
python -m pip install -r requirements.txt
```

Model training uses a separate environment described in `plant_stress/README.md`.

## Other scripts

| Task | Instructions |
| --- | --- |
| Explore datasets or run clustering diagnostics | [eda/](eda/README.md) |
| Train and evaluate stress classifiers | [plant_stress/](plant_stress/README.md) |
| Open the browser replay demo | [dashboard/](dashboard/README.md) |
| Find design documents or commissioning forms | [docs/](docs/README.md) |

The dashboard and classifier are separate tools; collected readings are not
automatically sent to them.

## Run software tests

With the root dependencies installed and `g++` on PATH, run from the repository root:

```sh
python -m pytest sensor_data_reciever/tests hardware-code/tests -q
```

Tests use temporary outputs and mock ESP32 hardware. They do not verify physical
sensors or Wi-Fi. Detailed results are in the [verification report](sensor_data_reciever/VERIFICATION.md).
Developer and coding-agent instructions are in [AGENTS.md](AGENTS.md).
