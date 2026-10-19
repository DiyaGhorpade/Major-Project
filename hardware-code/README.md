# ESP32 sensor firmware

- `data_log_producer_v1_12-remaining-plants/`: main collection sketch.
- `data_log_producer_v1_4-light-deprived/`: light-deprived collection variant.
- `libraries/`: bundled BH1750 and DHT sensor libraries, including examples.

The sketches collect soil moisture through a multiplexer and read temperature,
humidity, and ambient light. Open the corresponding `.ino` in your Arduino
toolchain. Check pin assignments and `SIMULATION_MODE` before using real hardware.

Current sketches print eight-field CSV records over serial. The HTTP receiver in
`../sensor_data_reciever/` expects nine fields including `node_id`; WiFi uploads
and a shared schema still need integration. The UI under `../dashboard/` is a
separate replay demo.
