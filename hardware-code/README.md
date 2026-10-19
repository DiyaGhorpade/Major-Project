# ESP32 data collection

The firmware and `../sensor_data_reciever/` form one collection system: two
ESP32 nodes measure plants and currently print readings over USB serial. Wi-Fi
delivery to a single Jetson Orin Nano is planned. The target architecture follows Hardware Architecture Rev. 3 and the
confirmed physical setup below. The source document is not stored in this repo.

## Confirmed physical setup

| Node | Plants | Local hardware |
| --- | --- | --- |
| A (`node_id=1`) | 12: Healthy, Water-stress, Nutrient-stress groups | ESP32-WROOM-32, CD74HC4067, 12 HW-103 resistive probes, DHT22, BH1750 |
| B (`node_id=2`) | 4: physically separate Light-stress group | ESP32-WROOM-32, CD74HC4067, 4 HW-103 resistive probes, DHT22, BH1750 |

Each node measures its own temperature, humidity, and light. The physical build
matches this arrangement; the firmware still needs the changes listed below.
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
A future scheduling or command component owns session IDs, scan counts, and
invocation times. Multiple scans may use the same session ID. Acquisition has
no automatic session counter or scan-scheduling state. The function returns
`void`: completing a call does not certify sensor validity or host receipt;
existing sensor error values and serial behavior are preserved.

For example, another firmware component can call `collectSample(7)` to produce
one scan whose records contain `session_id=7`. This is an API usage example,
not an installed trigger. Separate `.cpp` callers can declare
`void collectSample(int sessionId);` before calling it. Call from a single
firmware context; acquisition uses shared sensor objects and is not reentrant.

`setup()` still prints the CSV header once at 115200 baud. `loop()` only waits
100 ms per iteration: **no sensor scan occurs until explicitly invoked**.
There is no serial command handler, remote command handler, or replacement schedule.

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
still blocks for sensor reads, serial writes, and per-channel delays; it does
not wait for a scan interval or invoke another scan.

## Firmware changes needed for Rev. 3

1. Verify that occupied MUX channels match C0-C11 on A and C0-C3 on B.
   Keep node IDs 1 and 2 and the node-local plant/sensor identity convention.
   Introduce explicit mappings if the physical assignments change.
2. Keep sensor identity separate from the scan index; this is implemented.
3. Wireless transfer is deferred. When implementing it, send the existing
   nine-field serial schema as a plain-text POST to
   `http://<JETSON_LAN_IP>:5000/upload`. Serial printing alone does not transmit
   to the HTTP server.
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
