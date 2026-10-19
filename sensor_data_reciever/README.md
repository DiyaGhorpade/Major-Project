# Jetson receiver and collection scheduler

Run these two scripts on the Jetson:

| Script | What it does |
| --- | --- |
| `receiver.py` | Listens on port 5000, validates incoming readings and appends accepted records to CSV. |
| `scheduler.py` | Requests a scan from Node 1, waits for all expected points, then requests Node 2. Records the final outcome in a journal. |

All commands below run from the **repository root**, with the Python environment
activated. Complete the [Python setup](../README.md#python-312-setup) and
[ESP32 configuration and upload](../hardware-code/README.md) first.

## 1. Create the scheduler configuration

Copy `sensor_data_reciever/scheduler.commissioning.example.json` to
`sensor_data_reciever/scheduler.local.json`. Fill in the empty values before running.
The local file is ignored by Git.

| Setting | What to enter |
| --- | --- |
| `timezone` | The site's IANA timezone, such as `Asia/Kolkata` if applicable. |
| `times` | The actual daily collection times as unique `HH:MM` strings. |
| `nodes[].url` | Each board's actual command address: `http://<node-ip>`. Use numeric IPv4 addresses; the board listens on port 80. |
| `nodes[].sampling_points` | Keep all 12 points for Node 1 and all four for Node 2. Check them against the installed probes. |
| `record_plant_id` | The local pot ID emitted by the firmware. The template matches the current channel + 1 convention. |
| `plant_id`, `group` | The actual experiment pot ID and treatment group for each point. These are host metadata; they do not rewrite raw CSV IDs. |
| `csv_file` | The receiver's output file, relative to the JSON file. For the commands below, set `sensor_data_collection.csv`. |
| `journal_file` | A persistent run-history file. For this setup, set `collection_runs.jsonl`. |
| `command_timeout`, `completion_timeout`, `poll_interval` | Limits in seconds. Initial example values are 5, 120 and 0.5; review them against measured scan/upload duration. |

Use one shared journal for all scheduler invocations and preserve it across
restarts. Both node addresses and the Jetson address must stay stable.

Check the configuration without starting a collection:

```sh
python -c "import sys; sys.path.insert(0, 'sensor_data_reciever'); from scheduler import load_settings; load_settings('sensor_data_reciever/scheduler.local.json'); print('Configuration parsed')"
```

Parsing checks format, not whether addresses or physical mappings are correct.
The [configuration reference](../docs/hardware_configuration.md) lists exact
formats and the hardware values that still require verification.

## 2. Start the receiver

In the first terminal:

```sh
python sensor_data_reciever/receiver.py --output sensor_data_reciever/sensor_data_collection.csv
```

Use a new CSV path for the first run. An existing file must have the current
stored header; incompatible older logs are rejected rather than overwritten.
The scheduler's `csv_file` must resolve to this exact file. Create any chosen
parent directories before starting the scripts.

Check the receiver locally:

```sh
curl http://127.0.0.1:5000/health
```

Expect `OK`. Also check `http://<actual-Jetson-IP>:5000/health` from the node LAN
to confirm network reachability.

## 3. Run one collection

Keep the receiver running. In a second terminal, activate the same environment
and run:

```sh
python sensor_data_reciever/scheduler.py --config sensor_data_reciever/scheduler.local.json --once
```

This sends real commands to both boards. A successful run prints a final result
with `status` equal to `complete` and both nodes marked `complete`. It exits
with code 0; a failed collection exits with code 1.

Check the output files:

- `sensor_data_reciever/sensor_data_collection.csv`: 12 unique Node 1 points and
  four unique Node 2 points for the same session ID.
- `sensor_data_reciever/collection_runs.jsonl`: a starting event and final
  outcome for that session, including node progress or the failure reason.

The scheduler advances to Node 2 only after every configured Node 1 point has
been received and validated. Duplicate rows do not count as extra points.
Record this first physical run using the [commissioning checklist](../docs/hardware_commissioning.md).

## 4. Start the daily schedule

After the one-off run succeeds, run the scheduler without `--once`:

```sh
python sensor_data_reciever/scheduler.py --config sensor_data_reciever/scheduler.local.json
```

Keep both terminals/processes running. Stop them with Ctrl+C. The scripts do
not install a boot service; arrange and verify startup/restart separately if
unattended operation is needed.

Slots missed while the scheduler is stopped or busy are skipped. Failed commands
are not retried, and a timeout does not cancel a scan already running on a board.
Do not restart a manual run until the previous board activity has finished.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Configuration exits with an error | Resolve every empty/null template field; use valid IPv4 origins, unique times and positive timeouts. |
| Receiver refuses an existing CSV | Select a fresh output path and update scheduler `csv_file` to match. Preserve the older log. |
| Node is unreachable or rejects a command | Address, board power, Wi-Fi and port 80; a collecting node returns busy. |
| Run times out with missing points | Receiver rejection logs, board serial output, sensor connections and point/local-pot mappings. |
| Another scheduler is already running | Stop the duplicate process; use one scheduler and one persistent journal/lock path. |

## Data format and script reference

Firmware sends one headerless CSV record to `POST /upload`:

```text
timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light
```

The receiver prepends `record_id`, `received_at` and `accepted_at` when storing
it. Device timestamps are uptime milliseconds; receipt and acceptance times
are UTC. Raw plant and sensor IDs are local to a node. Identify a pot by
`(node_id, plant_id)` and a probe by `(node_id, sampling_point)`.

Rejected readings are printed to the receiver console and are not appended.
Repeated valid uploads remain separate CSV rows, although scheduler completion
counts each expected point once.

| File | Use |
| --- | --- |
| `config.py` | Shared CSV/validation definitions. Use receiver `--output`, not `SENSOR_DATA_FILE`, to choose HTTP output. |
| `validator.py` | Reading validation used by the receiver. |
| `storage.py` | Separate CSV utility; not the receiver's HTTP write path. |
| `archive/` | Historical serial receiver and logs; not needed for collection. |

See [scheduler behavior](SCHEDULER.md) for restart/completion details and
[software verification](VERIFICATION.md) for tests. Physical collection and
calibration remain **PENDING HARDWARE VERIFICATION** until recorded on actual hardware.
