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

Both sketches currently scan all 16 MUX channels, label plants as channel + 1,
and send eight-field CSV over serial at 115200 baud:

```text
timestamp,session_id,sampling_point,plant_id,soil,temperature,humidity,light
```

They currently populate `sampling_point` with the session's measurement number
1-6, which must change to the stable sensor ID. Their timestamps are `millis()`
since boot. Soil is a common linear ADC-to-0-100 conversion, not per-probe
calibration. Both have `SIMULATION_MODE = true`; disabling it selects the
configured five-minute sampling interval within thirty-minute sessions.

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

1. Set node ID `1` for A and `2` for B, and define an explicit
   channel-to-plant-to-sensor mapping per ESP32.
   Read only the 12 connected probes on A and four on B. Do not assume a probe
   ID always equals a plant ID or that occupied channels are necessarily contiguous.
2. Set `sampling_point` from that mapping. Keep scan/session numbering separate
   from sensor identity. Agree numeric IDs with the receiver before collection.
3. Implement Wi-Fi connection and plain-text HTTP POSTs to
   `http://<JETSON_LAN_IP>:5000/upload`. Add `node_id` in the receiver's agreed
   column order: `timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light`.
   Send `1` or `2` in that third field. The separate Jetson `validator.py` now
   enforces these IDs and this order; `receiver.py` still uses its own looser
   node validation. The existing `Serial.println` calls do not transmit to Jetson.
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
