# ESP32 data collection

The firmware and `../sensor_data_reciever/` form one collection system: two
ESP32 nodes measure plants and send readings over Wi-Fi to a single Jetson
Orin Nano. The target architecture follows Hardware Architecture Rev. 3 and the
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

`sampling_point` is now the local sensor ID and remains stable across scans.
`sampleNumber` independently controls the six scans in a session. Timestamps are
still `millis()` since boot. Soil is a common linear ADC-to-0-100 conversion,
not per-probe calibration. Both have `SIMULATION_MODE = true`; disabling it
selects the configured five-minute interval. Session-boundary timing is unchanged.

Current pin assignments in the source are:

| Signal | Node A sketch | Node B sketch |
| --- | --- | --- |
| MUX S0/S1/S2/S3 | GPIO16/17/18/19 | GPIO16/17/18/19 |
| MUX analog input | GPIO34 | GPIO34 |
| DHT22 data | GPIO4 | GPIO15 |
| BH1750 SDA/SCL | GPIO25/26 | GPIO21/22 |

Check these source assignments against the installed wiring. Some older design
examples use different pins; they are not the pin reference for these sketches.

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
