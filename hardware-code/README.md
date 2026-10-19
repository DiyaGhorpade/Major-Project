# ESP32 data collection

The firmware and `../sensor_data_reciever/` form one collection system: two
ESP32 nodes measure plants, print readings over USB serial, and attempt Wi-Fi
HTTP uploads to a single Jetson Orin Nano when configured and explicitly invoked. The target architecture follows Hardware Architecture Rev. 3 and the
previously reported physical arrangement below. The source document is not stored in this repo.

## Commissioning entry point

Use the [configuration worksheet](../docs/hardware_configuration.md) and
[commissioning checklist/results](../docs/hardware_commissioning.md) before
flashing installed boards. The arrangement below is previously reported context,
not evidence that this handover has physically verified a board or sensor.
All commissioning hardware checks remain **PENDING HARDWARE VERIFICATION**.

## Reported physical arrangement

| Node | Plants | Local hardware |
| --- | --- | --- |
| A (`node_id=1`) | 12: Healthy, Water-stress, Nutrient-stress groups | ESP32-WROOM-32, CD74HC4067, 12 HW-103 resistive probes, DHT22, BH1750 |
| B (`node_id=2`) | 4: physically separate Light-stress group | ESP32-WROOM-32, CD74HC4067, 4 HW-103 resistive probes, DHT22, BH1750 |

Each node measures its own temperature, humidity, and light. Verify the installed build against this arrangement; the future Rev. 3 items
below are separate from commissioning the current implementation.
`sampling_point` means the physical **sensor ID**, not the measurement number
within a session. Keep that ID with the probe if it moves to a different pot;
`plant_id` identifies the pot and `mux_channel` identifies its current connection.

## Files and current behavior

- `data_log_producer_v1_12-remaining-plants/`: intended Node A firmware.
- `data_log_producer_v1_4-light-deprived/`: intended Node B firmware.
- `libraries/`: bundled BH1750 and DHT libraries and their examples.

The sketches use explicit `NODE_ID` and `CHANNEL_COUNT` constants:

| Sketch | Node ID | Channels | Local plant/sensor IDs | Rows per scan |
| --- | --- | --- | --- | --- |
| 12-plant | 1 | C0-C11 | 1-12 | 12 |
| 4-plant | 2 | C0-C3 | 1-4 | 4 |

Plant identity is `(node_id, plant_id)`; local IDs may overlap across nodes.
Sensor identity is `(node_id, sampling_point)`. The current fixed wiring assigns
both local IDs as channel + 1. If probes move, replace that assignment with an
explicit mapping that preserves probe identity.

Each sketch prints nine-field CSV over serial at 115200 baud:

```text
timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light
```

`sampling_point` is the local sensor ID and remains stable across scans.
Timestamps are `millis()` since boot, not UTC. Soil is a common linear
ADC-to-0-100 conversion, not per-probe calibration. Acquisition is explicit:
`loop()` idles and does not collect records automatically.

Current pin assignments in the source are:

| Signal | Node A sketch | Node B sketch |
| --- | --- | --- |
| MUX S0/S1/S2/S3 | GPIO16/17/18/19 | GPIO16/17/18/19 |
| MUX analog input | GPIO34 | GPIO34 |
| DHT22 data | GPIO4 | GPIO15 |
| BH1750 SDA/SCL | GPIO25/26 | GPIO21/22 |

Check these source assignments against the installed wiring. Some older design
examples use different pins; they are not the pin reference for these sketches.

## Shared configuration and simulation mode

Both sketches contain the same `COLLECTION CONFIGURATION` section near the top
of the `.ino` file. These settings are duplicated, not loaded from a shared
configuration file: edit each sketch separately, then rebuild and upload it to
its corresponding node. Keep `NODE_ID`, `CHANNEL_COUNT`, and the node-specific
sensor pins as listed above.

### Acquisition interface

Both sketches expose the same operation:

```cpp
void collectSample(int sessionId);
```

Call it after `setup()` has initialized the sensors. One call synchronously
reads temperature, humidity, and light once, then scans all configured MUX
channels and prints one CSV record per channel. Node A produces 12 rows; Node B
produces four. Each row uses the supplied session ID without incrementing it.
The Jetson scheduler and shared command component own session IDs, scan counts, and
invocation times. Multiple scans may use the same session ID. Acquisition has
no automatic session counter or scan-scheduling state. The function returns
`void`: completing a call does not certify sensor validity or host receipt;
existing sensor error values and CSV serial output are preserved. Each record
is also submitted through the shared HTTP adapter; network code is outside the
sensor-reading function. Uploads are synchronous and can extend scan duration
and the uptime timestamps of subsequent channels.

The command server invokes `collectSample(sessionId)` once for each accepted
Jetson request. Every generated record carries that session ID. Separate `.cpp` callers can declare
`void collectSample(int sessionId);` before calling it. Call from a single
firmware context; acquisition uses shared sensor objects and is not reentrant.

`setup()` still prints the CSV header once at 115200 baud, initializes sensors,
and starts Wi-Fi without waiting for association. `loop()` services reconnection
and services accepted commands, then waits 100 ms per iteration. No scan occurs
without a command. There is no ESP32 schedule or serial command handler.

### Shared pacing settings

All delay values are in milliseconds. `UL` marks unsigned-long constants.
These delays belong to a single acquisition, not to a repeating schedule.

| Setting | Current value | Purpose |
| --- | --- | --- |
| `SIMULATION_MODE` | `true` | Selects per-plant pacing only |
| `MUX_SETTLE_TIME` | `10UL` | Wait after selecting each MUX channel, in either mode |
| `PLANT_READ_DELAY` | 100 ms / 500 ms | Wait after each plant record in simulation / real mode |

For Wokwi pacing, keep `const bool SIMULATION_MODE = true;`. For real collection
pacing, set `const bool SIMULATION_MODE = false;` in each sketch, rebuild, and
upload to the corresponding node. This is a compile-time setting, not a runtime
command. Changing it does not cause acquisition to start automatically.

Simulation mode does not generate synthetic readings, change pins or identities,
enable Wi-Fi, or change CSV serialization. Both modes use the same sensor drivers
and ADC reads. Timestamps remain actual uptime milliseconds.

### Preserved scan behavior and removed scheduling

For each channel, acquisition captures its timestamp before channel selection,
waits for MUX settling, reads GPIO34, maps the ADC value to soil 0-100, prints
the existing nine-field CSV row, and waits the per-plant delay. Temperature,
humidity, and light are shared across the rows of that scan. `sampling_point`
and `plant_id` remain `channel + 1`. Sensor failure values are passed through
as before; the drivers and their diagnostics are unchanged.

Originally, `SIMULATION_MODE` also selected the scan interval and collection
duration. Those were scheduling responsibilities and have been removed. Its
acquisition-specific effect remains unchanged: 100 ms versus 500 ms after each
plant record, with a 10 ms MUX settling delay in both modes.

The former five-second/five-minute repeating schedule, six-scan session loop,
automatic session increment, and interval/duration constants have been removed.
There is no daily schedule or wall-clock synchronization. `collectSample()`
still blocks for sensor reads, serial writes, HTTP attempts, and per-channel delays; it does
not wait for a scan interval or invoke another scan.

## Jetson-triggered collection

Both sketches expose `POST http://<node-ip>/collect` on port 80 with
`Content-Type: application/x-www-form-urlencoded` and body `session_id=123`.
The ID must be a decimal integer in 1..2147483647. The endpoint returns `202`
for accepted work, `409` while a command is pending or collecting, and `400`
for malformed input. Acceptance does not indicate completion.

A small FreeRTOS task services HTTP commands while the Arduino loop performs
one existing blocking `collectSample()` scan. An atomic slot rejects concurrent
commands throughout pending and active work. Sensor reads, serial output and
uploads remain in the loop's context. There is no queue, retry or automatic
acquisition. A lost acceptance response can mean the scan has started; the
Jetson fails that run and does not resend the command or advance to Node 2.

Use stable node addresses (for example DHCP reservations). Configure Wi-Fi and
the receiver address as below; configure node addresses, expected points,
plant/group mappings and times in the Jetson's
[scheduler configuration](../sensor_data_reciever/SCHEDULER.md). The Jetson
waits for validated stored records for every expected sampling point before
commanding the next node. Network and target-board behavior require deployment
verification; host mocks cover command validation and busy rejection.

## Wi-Fi HTTP configuration

Both sketches use `libraries/SensorHttpTransport`, alongside the bundled sensor
libraries. Make all three libraries available to the Arduino build (copy the
library folders into your sketchbook's `libraries` folder for Arduino IDE, or
use Arduino CLI's `--libraries hardware-code/libraries` from the repository root).
`WiFi` and `HTTPClient` come from the ESP32 Arduino core.

1. Copy `libraries/SensorHttpTransport/src/TransportConfig.example.h` to
   `TransportConfig.local.h` in the same directory. If using a sketchbook copy
   of the library, configure that copy instead.
2. Set `SENSOR_WIFI_SSID`, `SENSOR_WIFI_PASSWORD`, and `SENSOR_JETSON_HOST`.
   Use the Jetson's reachable LAN IPv4 address, without a scheme, port, or path.
   Hostnames are intentionally excluded to avoid DNS lookup delays. The example
   address is a placeholder. Empty passwords are allowed for an open network.
3. Rebuild and upload each sketch. The local configuration is Git-ignored;
   credentials can alternatively be supplied as quoted compiler definitions.

Without configuration, serial output and acquisition still work; HTTP attempts
return `NotConfigured` and Wi-Fi is not started. A malformed host is handled the
same way. The adapter always targets port **5000**, path **`/upload`**, using
plain HTTP and `Content-Type: text/plain`. Each POST body is exactly one existing
CSV record without the serial header or trailing CRLF. Receiver-assigned IDs
and UTC metadata are not sent by the firmware.

### Connection and upload behavior

`SensorHttpTransport::begin()` starts association without a waiting loop.
`poll()` requests reconnection at most once every 10 seconds while offline;
this is connection maintenance, not a sensor acquisition trigger. Wi-Fi is
independent of `SIMULATION_MODE`.

`emitRecord()` first prints the unchanged serial CSV row, then calls
`sensorTransport.upload(csv)`. The shared adapter handles all Wi-Fi and HTTP
operations. It makes one attempt per row, closes the HTTP connection, and does
not follow redirects. The default connect timeout is 2,000 ms and the HTTP I/O
timeout is 3,000 ms. Override them with `SENSOR_HTTP_CONNECT_TIMEOUT_MS` and
`SENSOR_HTTP_TIMEOUT_MS` (1..60,000 ms). These are library operation/inactivity
limits, not a guaranteed total scan deadline. HTTP latency adds to acquisition
time; disconnected attempts return immediately without waiting for connection.

### Results and limitations

Inspect `sensorTransport.lastResult()`, `sensorTransport.lastFailure()`, and
`sensorTransport.failedUploads()` from future firmware callers or a debugger.
Compare the failure count before and after a scan to detect any failed attempts
even if a later row succeeds. These methods add no diagnostic lines to serial.

| Result | Meaning |
| --- | --- |
| `200` | Existing receiver reported acceptance after writing the row |
| Other positive code | HTTP rejection/error or unexpected response; not success |
| Negative HTTPClient code | Connection/write/read failure; delivery may be unknown |
| `NotConfigured` (`-1000`) | Missing SSID or invalid/missing Jetson IPv4 address |
| `WifiDisconnected` (`-1001`) | No connected Wi-Fi link; no HTTP request attempted |
| `HttpBeginFailed` (`-1002`) | HTTP request initialization failed |

`lastResult()` is initially zero before any attempt, or `NotConfigured` after
unconfigured initialization. Every failed upload attempt increments the failure
count and updates `lastFailure()`; later success does not erase those values.
There is no queue, record retry, persistent storage, or automatic replay. Failed
rows remain visible in the serial stream but are not retained by this adapter.
Capture serial externally if a fallback log is required. A lost response can
occur after receiver acceptance; retrying elsewhere can duplicate rows because
the receiver has no deduplication. No upload is triggered until acquisition is
explicitly called by the command handler.

### Software verification

From the repository root, with Python and `g++` available:

```text
python -m unittest discover -s hardware-code/tests -v
```

Tests compile and execute the real sketches and shared header against host
mocks. They cover both node configurations and pacing modes, sensor/channel
sequence, CSV and serial bytes, headerless request bodies, endpoint and timeout
settings, HTTP errors, disconnection, reconnection, configuration errors, cleanup,
and absence of automatic acquisition. Mocks do not verify the ESP32 network stack
or actual elapsed network time.

The receiver suite also compiles these actual sketches against host mocks,
passes their generated POST bodies through the production validator and CSV
writer, and verifies receipt/approval metadata. It covers both nodes in both
pacing modes, a real loopback HTTP server, concurrent uploads, and sensor-failure
rejection. From the repository root, with Flask, pytest, hypothesis, and `g++`:

```text
python -m pytest sensor_data_reciever/tests -q
```

Every test writes to temporary files. See the [verification report and Jetson
smoke check](../sensor_data_reciever/VERIFICATION.md). Software socket tests run
on this machine; they do not establish delivery to a physical Jetson.


With Arduino CLI and an installed ESP32 core, target compilation can additionally
be checked from the repository root (use the FQBN matching your actual board):

```text
arduino-cli compile --fqbn esp32:esp32:esp32 --libraries hardware-code/libraries hardware-code/data_log_producer_v1_12-remaining-plants
arduino-cli compile --fqbn esp32:esp32:esp32 --libraries hardware-code/libraries hardware-code/data_log_producer_v1_4-light-deprived
```

Target compilation and physical Wi-Fi/sensor behavior have not been verified in
this environment. See the ESP32 core's [HTTPClient interface](https://github.com/espressif/arduino-esp32/blob/master/libraries/HTTPClient/src/HTTPClient.h)
and [Wi-Fi API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/wifi.html)
for the underlying library operations.

## Firmware changes needed for Rev. 3

1. Verify that occupied MUX channels match C0-C11 on A and C0-C3 on B.
   Keep node IDs 1 and 2 and the node-local plant/sensor identity convention.
   Introduce explicit mappings if the physical assignments change.
2. Keep sensor identity separate from the scan index; this is implemented.
3. The shared HTTP adapter now sends the existing nine-field schema to
   `http://<JETSON_LAN_IP>:5000/upload` when configured and connected. Configure
   and verify it as described below; device/network behavior is not hardware-tested.
4. Use synchronized datetime timestamps, through NTP or Jetson time correction,
   rather than `millis()`. Keep both nodes on the same time convention.
5. Preserve raw ADC and MUX channel; calculate `soil` using measured calibration
   values keyed by sensor ID. Add `sensor_status` (`valid`, `suspect`, `missing`).
   Detailed calibration and validity rules are carried over from Rev. 2 and are
   not fully specified in Rev. 3; do not substitute invented thresholds.
6. Add reconnection, bounded buffering, acknowledgements/retries, and a way to
   identify a scan cycle so the receiver can detect missing records and avoid
   duplicating retried records. Preserve invalid/missing readings with status.
7. Set `SIMULATION_MODE = false` for real collection and validate each node
   independently before combined collection.

The [receiver README](../sensor_data_reciever/README.md) explains the current HTTP
contract and the exact proposed configuration changes. Receiver and firmware
must adopt the same schema together; changing a CSV header alone is insufficient.

Image acquisition is still open in Rev. 3 (ESP32-CAM, Pi camera with Pi Zero 2 W,
or direct USB camera). None of these sensor sketches implements image capture.
The browser dashboard is currently a separate replay demo.
