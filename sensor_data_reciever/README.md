# Jetson sensor data receiver

The collection system follows Hardware Architecture Rev. 3: Node A (`node_id=1`)
serves 12 plants and Node B (`node_id=2`) serves four physically separated
light-stress plants. Each ESP32 has its own CD74HC4067, HW-103 probes, DHT22,
and BH1750. The physical setup is confirmed; wireless transfer is deferred.

## Identity and shared CSV convention

Plant identity is `(node_id, plant_id)`. Plant IDs are local to the node, so
plant 1 on node 1 and plant 1 on node 2 are different pots. `sampling_point`
is the node-local sensor ID, not the scan number. Sensor identity is
`(node_id, sampling_point)`; preserve it if a probe moves to another pot.

The current fixed wiring uses channels C0-C11 on node 1 and C0-C3 on node 2.
Both local plant and sensor IDs are initially channel + 1. The firmware now
prints the same nine-field order used by `config.py`, `validator.py`, and
`receiver.py`:

```text
timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light
```

For example, node 2's fourth sensor and fourth pot:

```text
123456,1,2,4,4,45.2,23.1,60.5,512.0
```

Firmware still prints over serial, uses `millis()` timestamps, and applies a
common linear soil conversion. A common CSV layout does not provide a wireless
connection or synchronized time by itself.

## Independent receiver and validator

`receiver.py` runs a Flask server on `0.0.0.0:5000`, exposing `GET /health` and
`POST /upload`. Uploads contain one plain-text CSV row without a header. Accepted
rows are appended under a thread lock and flushed to `sensor_data.csv`.
Rejected rows are printed to the console, not written to `error_log.csv`.

`validator.py` is an independent validation utility, not part of the server's
request path. It follows the same column order, accepts node IDs `"1"` or `"2"`,
and returns either a parsed dictionary or a fail-fast list of `ValidationError`
objects. Node IDs remain strings in its result. Its checks no longer depend on
nonexistent configuration helpers. The HTTP server retains its own validation;
it currently only requires a nonblank node ID and timestamp.

Both paths now allow sensor IDs 1-12 rather than treating them as scan numbers
1-6. Plant bounds remain 1-16 for compatibility. Neither path currently enforces
node-specific limits (12 on node 1, four on node 2) or sensor-to-pot mappings.
The firmware limits actual collection to the configured 12/4 channels.

## Exactly which configuration settings take effect

`config.py` configures the separate `validator.py` and `storage.py` utilities.
It does not configure `receiver.py`, which keeps its own settings. These remain
independent; integrating them is not required by this collection convention.

| Setting | config.py utilities | Standalone receiver.py |
| --- | --- | --- |
| Bind address | `HOST` (not used by validation/storage) | `host="0.0.0.0"` in `app.run` |
| Port | `PORT` (not used by validation/storage) | `PORT` |
| Output CSV | `SENSOR_DATA_FILE` | `CSV_FILE` |
| Header / count | `CSV_HEADER`, `EXPECTED_FIELDS` | `CSV_HEADER`, `EXPECTED_FIELDS` |
| Sensor bounds | `VALID_SAMPLING_POINT_MIN/MAX` | Same names |
| Temperature bounds | `VALID_TEMPERATURE_MIN/MAX` | `VALID_TEMP_MIN/MAX` |
| Other numeric bounds | `VALID_SESSION_ID_MIN`, `VALID_PLANT_ID_MIN/MAX`, soil/humidity/light bounds | Same names |

The sensor-ID correction is already applied in both files:

```python
VALID_SAMPLING_POINT_MIN = 1
VALID_SAMPLING_POINT_MAX = 12
```

No additional `config.py` changes are needed for the present nine-field schema.
Keep its header in the exact order shown above and `EXPECTED_FIELDS = 9` (or
compute it from the header). Node validation is directly defined as `{"1", "2"}`
in `validator.py`; no `NODE_XX` helpers or extra config constants are needed.
For stricter node-specific bounds, add mapping rules and code that consumes them;
adding an unused constant to config does not enforce anything.

Change `SENSOR_DATA_FILE` to relocate output from `storage.py`; change `CSV_FILE`
in `receiver.py` to relocate HTTP output. Keep `HOST`/the server bind address as
`0.0.0.0`; the future ESP32 upload URL should use the Jetson's reachable LAN IP.
If changing the HTTP port, edit `receiver.py`'s `PORT` and match it in the sender.
`ERROR_LOG_FILE` is not used by the current HTTP server.

## Remaining Rev. 3 work

Wireless transfer, synchronized datetime timestamps, reconnection, buffering,
and missing-cycle detection remain deferred work. Session timing is unchanged.
Detailed per-probe calibration and validity rules require the referenced Rev. 2;
do not substitute invented calibration values or fault thresholds.

Full Rev. 3 records also need MUX channel, raw ADC, and sensor status. One
proposed extension, retaining existing names and session grouping, is:

```python
CSV_HEADER = [
    "timestamp", "session_id", "node_id", "sampling_point", "plant_id",
    "mux_channel", "soil_raw_adc", "soil", "sensor_status",
    "temperature", "humidity", "light",
]
EXPECTED_FIELDS = len(CSV_HEADER)  # 12
```

This extension is not implemented. Adopt it in firmware and each independent
consumer together: update unpacking, validation, output headers, and tests.
Define missing-value encoding and preserve `valid`/`suspect`/`missing` status.
Calibration profiles must use the composite sensor identity. Use a new CSV or
migrate existing data on schema changes; current startup does not verify old
headers. Treat node ID as join metadata rather than a predictive ML feature.

## Running the HTTP receiver

With Python and Flask installed in the Jetson environment:

```bash
cd sensor_data_reciever
python3 receiver.py
```

On Windows with the root environment:

```powershell
cd sensor_data_reciever
..\.venv\Scripts\python.exe receiver.py
```

Run from this directory because output paths are relative. Check
`http://<JETSON_LAN_IP>:5000/health`; this verifies reachability, not firmware
wireless integration.

## Other files

`tests/test_validator_numeric_nodes.py` covers numeric node validation and field
order. Older tests still contain earlier schemas/conventions and need migration.
`sample.csv` is legacy input; existing logs are retained. `archive/` contains the
historical serial receiver and an earlier error log. Only the leftover
`old stuff/` directory was removed.

See the [hardware README](../hardware-code/README.md) for sketch details.
`docs/jetson_deployment_guide.docx` and older examples in `../docs/` describe other
hardware arrangements, schemas, or pin assignments. The dashboard uses stored
replay data, and image acquisition remains open in Rev. 3.
