# Jetson sensor data receiver

This is the Jetson side of the ESP32 collection system described in Hardware
Architecture Rev. 3. The confirmed physical setup is Node A with 12 plants and
Node B with four physically separated light-stress plants. Each ESP32 has its
own CD74HC4067, HW-103 probes, DHT22, and BH1750. Both nodes are intended to send
readings over Wi-Fi to one Jetson Orin Nano. Node IDs are `1` for A and `2`
for B; CSV represents these as strings `"1"` and `"2"`.

`sampling_point` is the physical **sensor ID**. `plant_id` is the pot identity;
a sensor keeps its ID when moved. The current code's interpretation of
`sampling_point` as a measurement number 1-6 is outdated.

## What works in the current receiver

`receiver.py` runs a Flask HTTP server bound to `0.0.0.0:5000`:

- `GET /health`: liveness check.
- `POST /upload`: one plain-text CSV row, with no header.
- Accepted rows are appended to `sensor_data.csv` under a thread lock and flushed
  to disk. Rejections are printed to the console; the current server does not
  write `error_log.csv`.

The current nine-field contract, in exact order, is:

```text
timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light
```

For example, this fits the shared nine-field layout:

```text
2026-08-28T07:00:00Z,1,1,3,7,45.2,23.1,60.5,512.0
```

In `receiver.py`, the timestamp and node ID are currently only checked for
nonblank text. The separate `validator.py` now accepts only node IDs `"1"` and
`"2"` and follows the same column order as `config.py` and `receiver.py`. It
returns the node ID as a string and preserves the existing fail-fast
`ValidationError` convention. This stricter node validation is not yet used
by the HTTP server. Both validation paths still limit sensor IDs to 1-6, and
neither checks the node/plant/sensor mapping. The firmware currently emits eight fields over
serial, so it cannot upload to this server without changes.

## Exactly what to change in config.py

**Editing `config.py` alone currently changes nothing in the HTTP server.**
`receiver.py` defines its own settings and validation and imports neither
`config.py`, `validator.py`, nor `storage.py`. The following is a change plan,
not configuration already implemented by this documentation update.

### 1. Connect the receiver to the configuration

Before relying on `config.py`, change `receiver.py` to import it and consume:

| config.py setting | Current receiver equivalent |
| --- | --- |
| `HOST` | Hardcoded `host="0.0.0.0"` in `app.run` |
| `PORT` | Local `PORT` |
| `SENSOR_DATA_FILE` | Local `CSV_FILE` |
| `CSV_HEADER`, `EXPECTED_FIELDS` | Local schema constants |
| `VALID_SESSION_ID_MIN` | Same local name |
| `VALID_SAMPLING_POINT_MIN/MAX` | Same local names |
| `VALID_PLANT_ID_MIN/MAX` | Same local names |
| `VALID_SOIL_MIN/MAX` | Same local names |
| `VALID_TEMPERATURE_MIN/MAX` | Local `VALID_TEMP_MIN/MAX` |
| `VALID_HUMIDITY_MIN/MAX`, `VALID_LIGHT_MIN/MAX` | Same local names |

Read these as `config.<name>` or otherwise remove the duplicated source of
settings. Initialize files using the same configured path and schema used by
uploads. `ERROR_LOG_FILE` only becomes active after error-file writing is added.

### 2. Correct the existing nine-field configuration

The node-ID convention is fixed: `"1"` for A and `"2"` for B. For sensor IDs,
a proposed scheme is global numeric IDs 1-16; confirm these against the labels
on the actual probes and use their real mapping.

With that convention, set:

```python
HOST = "0.0.0.0"
PORT = 5000
SENSOR_DATA_FILE = "sensor_data.csv"
ERROR_LOG_FILE = "error_log.csv"

CSV_HEADER = [
    "timestamp", "session_id", "node_id", "sampling_point", "plant_id",
    "soil", "temperature", "humidity", "light",
]
EXPECTED_FIELDS = len(CSV_HEADER)

VALID_SAMPLING_POINT_MIN = 1
VALID_SAMPLING_POINT_MAX = 16  # physical sensor IDs, not six readings per session
VALID_PLANT_ID_MIN = 1
VALID_PLANT_ID_MAX = 16
VALID_SESSION_ID_MIN = 1
VALID_NODE_IDS = {"1", "2"}  # optional centralization; see below
```

`HOST`, `PORT`, paths, plant bounds, and minimum session ID already have these
values. The changes are deriving `EXPECTED_FIELDS`, raising the sensor-ID bound
from 6 to 16, and optionally centralizing the allowed node IDs in
`VALID_NODE_IDS`. `validator.py` currently checks `{"1", "2"}` directly; to
centralize that rule, change it to read `config.VALID_NODE_IDS` as well. Adding
the setting alone does not change either validation path. Add explicit mapping
data for each node's occupied MUX channels, sensor IDs, and plant IDs using the
real wiring, then implement validation of the 12+4 allocation against it.

Leave the existing temperature (-40 to 80 C), humidity (0-100), light
(0-100000 lux), and soil (0-100) bounds unchanged for the current format. The
soil bounds assume a normalized moisture index; if calibration defines another
scale, update the bounds to that measured convention.

Keep `HOST` as `0.0.0.0`; put the Jetson's reachable LAN IP in each ESP32's upload
URL, not in this bind-address setting. Ensure the configured port is reachable
from both ESP32s. These nine-field changes support a basic upload contract,
not the complete Rev. 3 schema.

### 3. Extend the schema for full Rev. 3 records

Retain `sampling_point` as the sensor-ID field and `soil` as the calibrated
moisture index. A proposed compatible naming convention is:

```python
CSV_HEADER = [
    "timestamp", "session_id", "node_id", "sampling_point", "plant_id",
    "mux_channel", "soil_raw_adc", "soil", "sensor_status",
    "temperature", "humidity", "light",
]
EXPECTED_FIELDS = len(CSV_HEADER)  # 12
VALID_MUX_CHANNEL_MIN = 0
VALID_MUX_CHANNEL_MAX = 15
VALID_SOIL_RAW_ADC_MIN = 0
VALID_SOIL_RAW_ADC_MAX = 4095  # matches the current sketches' assumed ADC range
VALID_SENSOR_STATUSES = {"valid", "suspect", "missing"}
```

This retains the existing `session_id` for collection grouping while adding the
raw ADC, channel, and status required by Rev. 3. Do not add another `sensor_id`
column duplicating `sampling_point`. Store calibration profiles keyed by the
stable sensor ID, using measured values for each actual probe.

Change the ESP32 payload, receiver field unpacking, type/range checks, storage,
and tests to the same 12-field order together. Simply changing `EXPECTED_FIELDS`
will break the current nine-variable unpacking. Define how absent measurements
are encoded (for example, blank numeric fields with `sensor_status=missing`) and
make validation preserve those rows rather than reject every missing value.
Use a new output file or migrate existing CSVs when changing the schema; current
startup logic does not replace or verify an existing file's header.

### 4. Integrate the corrected validator

`validator.py` now reads `node_id` from position 3, `sampling_point` from
position 4, and `plant_id` from position 5, matching the configured header and
HTTP receiver. It accepts only `"1"` and `"2"` (after trimming whitespace),
rejects other identifiers, and reports blank IDs as missing. Calls to the
nonexistent node-configuration helpers have been removed.

To use it in HTTP uploads, replace or delegate the server's local validation
and adapt its returned dictionary or list of `ValidationError` objects to the
HTTP response and storage flow. Correcting this module alone does not change
`receiver.py` behavior. Sensor-ID range/messages, datetime validation,
node/sensor/plant mapping, and Rev. 3 status handling still need updating.

Focused tests in `tests/test_validator_numeric_nodes.py` cover both accepted
nodes, invalid and missing IDs, shared field order, and incorrect field counts.
Older tests may still contain the former `NODE_01` convention and column order;
those fixtures need migration before they can verify the current contract.

## Remaining work outside configuration

- Firmware: Wi-Fi uploads, stable sensor IDs, 12/4 channel maps, synchronized
  timestamps, per-probe calibration, raw ADC/channel/status fields, and buffering.
- Receiver: actual node/mapping validation, datetime parsing, missing-cycle
  detection, and retry deduplication. Agree cycle/record identity with firmware;
  session IDs alone do not identify an individual scan within a session.
- Treat `node_id` as identity/join metadata, not a predictive ML feature.
- Rev. 2 is needed for the detailed calibration and validity rules referenced by
  Rev. 3. Do not invent calibration values or fault thresholds from this README.

## Running on Jetson

Once Python and Flask are installed in the Jetson environment, run from this
folder so the current relative CSV path resolves here:

```bash
cd sensor_data_reciever
python3 receiver.py
```

On Windows, using the root environment:

```powershell
cd sensor_data_reciever
..\.venv\Scripts\python.exe receiver.py
```

Visit `http://<JETSON_LAN_IP>:5000/health` from the same network. A successful
health check verifies reachability, not complete firmware/schema integration.

## Other files and documentation

`tests/` includes tests targeting older interfaces; align them with the chosen
contract. `sample.csv` is legacy test input. Existing logs are retained, and
`archive/` contains the historical serial receiver (`serial_receiver.py`) and
an earlier error log. The leftover `old stuff/` directory has been removed.

The [hardware README](../hardware-code/README.md) covers current sketch behavior.
The guide in `docs/jetson_deployment_guide.docx` describes a broader direct-sensor
Jetson deployment and is not the wiring authority for this dual-ESP32 setup.
Older examples in `../docs/` also use other schemas and pin assignments.
The dashboard CSV is under `../dashboard/data/`; it is replay input, not this
receiver's current upload contract. Image acquisition remains open in Rev. 3.
