# Jetson sensor data receiver

The collection system follows Hardware Architecture Rev. 3: Node A (`node_id=1`)
serves 12 plants and Node B (`node_id=2`) serves four physically separated
light-stress plants. Each ESP32 has its own CD74HC4067, HW-103 probes, DHT22,
and BH1750. The arrangement is previously reported; commissioning verification remains pending. Both sketches include a configurable HTTP
transport; physical Wi-Fi/Jetson delivery still requires verification.

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

Firmware prints over serial and attempts HTTP uploads when configured and
explicitly invoked. It uses `millis()` timestamps and a common linear soil
conversion. The device timestamp is not synchronized UTC.

## Receiver and shared validator

`receiver.py` runs a Flask server on `0.0.0.0:5000`, exposing `GET /health` and
`POST /upload`. Uploads contain one plain-text CSV row without a header. Accepted
rows are appended under a thread lock and flushed to `sensor_data.csv`.
Rejected rows are printed to the console with a recoverable claimed `node_id`
or `unknown`; they are not written to `error_log.csv`. This identity comes from
the payload and is not authenticated. Invalid UTF-8, malformed/multiline CSV,
empty timestamps, non-finite sensor readings, and invalid field values return
HTTP 400 without appending data. Sensor values must satisfy `config.py` bounds.

`receiver.py` delegates record validation to `validator.validate_record()`. It follows the same column order, accepts node IDs `"1"` or `"2"`,
and returns either a parsed dictionary or a fail-fast list of `ValidationError`
objects. Node IDs remain strings in its result. Its checks no longer depend on
nonexistent configuration helpers. The HTTP server uses these same rules without adding separate field checks.

Both paths now allow sensor IDs 1-12 rather than treating them as scan numbers
1-6. Plant bounds remain 1-16 for compatibility. Neither path currently enforces
node-specific limits (12 on node 1, four on node 2) or sensor-to-pot mappings.
The firmware limits actual collection to the configured 12/4 channels.

## Exactly which configuration settings take effect

`config.py` supplies the shared CSV header and the validation bounds used by
`validator.py`, including HTTP uploads. The receiver retains its own port and
output-file settings; `storage.py` remains a separate utility.

| Setting | config.py utilities | Standalone receiver.py |
| --- | --- | --- |
| Bind address | `HOST` (not used by validation/storage) | `host="0.0.0.0"` in `app.run` |
| Port | `PORT` (not used by validation/storage) | `PORT` |
| Output CSV | `SENSOR_DATA_FILE` | `CSV_FILE`, overridden by `--output` |
| Header / count | `CSV_HEADER`, `EXPECTED_FIELDS` | Header from config; count checked by validator |
| Sensor bounds | `VALID_SAMPLING_POINT_MIN/MAX` | Applied through validator |
| Temperature bounds | `VALID_TEMPERATURE_MIN/MAX` | Applied through validator |
| Other numeric bounds | `VALID_SESSION_ID_MIN`, `VALID_PLANT_ID_MIN/MAX`, soil/humidity/light bounds | Applied through validator |

The shared sensor-ID bounds in `config.py` are:

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
in `receiver.py`, or pass `--output PATH`, to relocate HTTP output. Keep `HOST`/the server bind address as
`0.0.0.0`; the ESP32 upload URL should use the Jetson's reachable LAN IP.
If changing the HTTP port, edit `receiver.py`'s `PORT` and match it in the sender.
`ERROR_LOG_FILE` is not used by the current HTTP server.

## Remaining Rev. 3 work

Physical wireless verification, synchronized device datetime timestamps, buffering,
record retries, and missing-cycle detection remain open. The firmware HTTP
adapter maintains Wi-Fi reconnection; it does not queue or retry records.
The Jetson scheduler now owns session timing and detects incomplete configured point sets; see [scheduler setup](SCHEDULER.md).
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
migrate existing data on schema changes; current startup rejects incompatible existing
headers. Treat node ID as join metadata rather than a predictive ML feature.

## Running the HTTP receiver

With Python and Flask installed in the Jetson environment:

```bash
cd sensor_data_reciever
python3 receiver.py --output sensor_data_verified.csv
```

On Windows with the root environment:

```powershell
cd sensor_data_reciever
..\.venv\Scripts\python.exe receiver.py --output sensor_data_verified.csv
```

Run from this directory because output paths are relative. Check
`http://<JETSON_LAN_IP>:5000/health`; this verifies reachability, not firmware
wireless integration.

## Other files

`tests/` covers the current numeric node IDs, wire/storage schemas, malformed
HTTP input, timestamp ordering, storage failures, and generated firmware records.
Retired serial APIs and sequential-ID expectations have been replaced with the
HTTP contract and UUID4 checks. See [verification instructions](VERIFICATION.md)
for commands, software evidence, and the physical Jetson check.
`sample.csv` is legacy input; existing logs are retained. `archive/` contains the
historical serial receiver and an earlier error log. Only the leftover
`old stuff/` directory was removed.

See the [hardware README](../hardware-code/README.md) for sketch details.
`docs/jetson_deployment_guide.docx` and older examples in `../docs/` describe other
hardware arrangements, schemas, or pin assignments. The dashboard uses stored
replay data, and image acquisition remains open in Rev. 3.

## Receiver-assigned record metadata

At entry to the upload handler, before parsing or validation, HTTP ingestion
captures an ISO 8601 UTC `received_at` timestamp. Approved records retain it.
After validation succeeds, HTTP ingestion assigns a UUID4 `record_id` and an
ISO 8601 UTC `accepted_at` timestamp before writing. The device `timestamp`
is preserved. The POST body remains the same nine fields; stored CSV columns are:

```text
record_id,received_at,accepted_at,timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light
```

Each accepted upload gets a new ID, including repeated submissions; this does
not implement deduplication. Existing incompatible CSV headers cause startup
to fail (or uploads to return HTTP 500), without appending or migrating data.
Back up existing data and select a new output path or explicitly migrate it.
Approval means successful receiver validation; no manual approval step is
introduced. HTTP 200 is sent only after append, flush, and `fsync` complete. An
I/O failure or lost response can occur after bytes have reached the file; a
500/timeout does not prove non-delivery, and retries can duplicate rows.
The standalone storage utility retains its existing nine-field format.

## Scheduled ESP32 collection

See [Jetson scheduler setup](SCHEDULER.md) for sequential Node 1 / Node 2
collection, configuration, completion tracking and software verification.
The existing upload schema and validation contract are unchanged.

For hardware handover, start with the [commissioning checklist](../docs/hardware_commissioning.md)
and [configuration boundaries](../docs/hardware_configuration.md).
