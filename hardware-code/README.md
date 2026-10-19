# ESP32 firmware setup

Upload one sketch to each ESP32. Once connected to Wi-Fi, the boards wait for
the Jetson to request a collection. Each scan prints CSV at 115200 baud and
uploads the same readings to the Jetson receiver on port 5000.

| Board | Sketch | Points per scan |
| --- | --- | --- |
| Node 1 | `data_log_producer_v1_12-remaining-plants/data_log_producer_v1_12-remaining-plants.ino` | 12 |
| Node 2 | `data_log_producer_v1_4-light-deprived/data_log_producer_v1_4-light-deprived.ino` | 4 |

## 1. Install the board support and libraries

Use Arduino IDE or Arduino CLI with ESP32 board support matching the actual
board. Each node uses a CD74HC4067 MUX, soil probes, a DHT22 and a BH1750.

The required project libraries are included in `hardware-code/libraries/`:

- `SensorHttpTransport`
- `DHT_sensor_library_for_ESPx`
- `BH1750`

For Arduino IDE, copy these folders into your sketchbook's `libraries` folder.
For Arduino CLI, use `--libraries hardware-code/libraries` as shown below.
Wi-Fi, HTTPClient, WebServer and FreeRTOS are supplied by the ESP32 core.

## 2. Set the Wi-Fi and Jetson address

In the library copy used by your build, copy
`SensorHttpTransport/src/TransportConfig.example.h` to `TransportConfig.local.h`
in the same directory. Fill in:

```cpp
#define SENSOR_WIFI_SSID "<actual Wi-Fi network>"
#define SENSOR_WIFI_PASSWORD "<actual Wi-Fi password>"
#define SENSOR_JETSON_HOST "<actual Jetson LAN IPv4>"
```

Replace all placeholders. The Jetson value is just an IPv4 address, without
`http://`, a port or a path. The receiver must be reachable at that address on
port **5000**. Keep the private header out of Git; the repository copy is ignored.
Rebuild both sketches whenever this configuration changes.

## 3. Check wiring and select real collection pacing

These are the pin assignments in the current sketches. Check the installed
wiring before powering or flashing the boards.

| Signal | Node 1 | Node 2 |
| --- | --- | --- |
| MUX S0 / S1 / S2 / S3 | GPIO16 / 17 / 18 / 19 | GPIO16 / 17 / 18 / 19 |
| MUX analog output | GPIO34 | GPIO34 |
| Occupied MUX channels | C0-C11 | C0-C3 |
| DHT22 data | GPIO4 | GPIO15 |
| BH1750 SDA / SCL | GPIO25 / 26 | GPIO21 / 22 |

In **both** sketches, set:

```cpp
const bool SIMULATION_MODE = false;
```

This selects a 500 ms delay after each plant reading. `true` selects 100 ms;
it still reads the real sensors. Both modes use a 10 ms MUX settle delay.
Collection times are set on the Jetson, not in these sketches.

Keep the existing node IDs and channel counts. The current firmware emits
`sampling_point` and local `plant_id` as MUX channel + 1. Record the actual
probe, pot and treatment group for every channel. If the wiring or identity
mapping needs a code change, use the [configuration boundaries](../docs/hardware_configuration.md#values-that-still-need-code-changes-if-hardware-differs).

## 4. Build and upload

In Arduino IDE, open each sketch, select its board and serial port, then verify
and upload it to the corresponding node. Label the boards to avoid swapping them.

For Arduino CLI compilation, run from the repository root. The FQBN below is an
example for a generic ESP32; use the FQBN matching the actual board:

```sh
arduino-cli compile --fqbn esp32:esp32:esp32 --libraries hardware-code/libraries hardware-code/data_log_producer_v1_12-remaining-plants
arduino-cli compile --fqbn esp32:esp32:esp32 --libraries hardware-code/libraries hardware-code/data_log_producer_v1_4-light-deprived
```

Open the serial monitor at **115200 baud** for startup and collection output.
No scan starts just because a board is powered on.

## 5. Start collection from the Jetson

Determine each node's actual LAN IPv4 address and keep it stable, for example
with a DHCP reservation. Enter those addresses in the Jetson scheduler configuration.
Then follow the [receiver and scheduler run instructions](../sensor_data_reciever/README.md).

A successful run uploads 12 unique points from Node 1, then four from Node 2,
under the same session ID. Use the scheduler's final outcome and stored CSV to
confirm success; a command being accepted does not mean collection has finished.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Board remains idle | The Jetson must send a collection command; there is no board-side schedule. |
| Node cannot be reached | Actual node address, Wi-Fi settings, board power and LAN connectivity. The command port is 80. |
| Serial rows appear but no CSV rows arrive | Receiver is running; Jetson IPv4 and port 5000 are reachable; private header was applied to the library actually built. |
| Run fails with missing points | Receiver rejection logs, sensor readings, occupied channels and scheduler mappings. Firmware does not queue or retry failed uploads. |

Device timestamps are uptime milliseconds, not UTC. Soil values use a common
linear ADC conversion and are not calibrated per probe. Target compilation,
physical sensor behavior and LAN collection remain **PENDING HARDWARE VERIFICATION**;
record results in the [commissioning checklist](../docs/hardware_commissioning.md).

For host-only checks, run from the repository root with `g++` available:

```sh
python -m unittest discover -s hardware-code/tests -v
```
