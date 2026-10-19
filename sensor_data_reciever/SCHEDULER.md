# Jetson-scheduled ESP32 collection

`scheduler.py` is the single collection schedule owner. There was no executable
collection scheduler in this repository to extend. It runs beside the existing
receiver; no separate command service or completion endpoint is needed.

1. Configure and flash both sketches using `hardware-code/README.md`. Reserve
   stable LAN addresses for Node 1 (12 points) and Node 2 (4 points).
2. Copy `scheduler.commissioning.example.json` to `scheduler.local.json` and resolve
   every empty/null field using the [configuration guide](../docs/hardware_configuration.md).
   The older `scheduler.example.json` contains software example values only. Set node URLs,
   collection times and mappings. Node URLs must use numeric LAN IPv4 addresses
   to avoid DNS delays outside HTTP timeouts. Times use the specified IANA timezone.
   Install `Flask` and `tzdata` (included in root requirements); Linux can also
   use its system timezone database.
3. Set `csv_file` to the **same output file** used by the receiver. Scheduler
   paths are relative to the JSON file. Keep the journal across restarts and
   use one shared journal/lock path for all scheduler invocations.
4. Start the receiver and scheduler from the repository root, in two terminals:

   ```sh
   python sensor_data_reciever/receiver.py --output sensor_data_reciever/sensor_data.csv
   python sensor_data_reciever/scheduler.py --config sensor_data_reciever/scheduler.local.json
   ```

For an immediate collection using the same ordering, locking and completion
checks, add `--once`. It exits zero on completion and one on failure.

Each scheduled run allocates one positive signed 32-bit session ID and sends
`POST /collect` with a single URL-encoded `session_id` field to Node 1. Only
HTTP 202 means accepted. It then polls the receiver's CSV until all configured
points have matching session, node and local `record_plant_id` values and pass
the existing validator. Only then does it request Node 2. A set of sampling
points prevents duplicate rows from completing missing points. The receiver
still stores duplicates under its existing contract.

`sampling_points` is keyed by the physical node-local sensor ID. Each entry's
`record_plant_id` must match the existing firmware CSV value (currently channel
+ 1). `plant_id` and `group` describe the experimental mapping controlled on
the Jetson; they do not rewrite the firmware CSV. Node 2's example maps local
IDs 1..4 to global plants 13..16 only in the older software example; the
commissioning template leaves physical pot/group assignments unresolved.

Command errors, non-202 responses and missing records fail the entire run;
remaining nodes are not commanded. `command_timeout` limits HTTP operations;
`completion_timeout` limits each accepted node's wait. Choose it above worst
case scan and upload duration (the example uses 120 seconds). There is no
command retry: a timed-out response may follow an accepted command. A failed
run does not cancel a scan already running on a node.

The JSON-lines journal stores a durable `running` event before sending commands
and a final `complete` or `failed` outcome with per-node progress and errors.
Interrupted runs are marked failed at startup. Started schedule slots are never
replayed after restart, including failures. Session IDs advance from the journal
and current epoch seconds; keep the journal and system clock valid. IDs above
2147483647 fail explicitly. Corrupt journal data fails closed.

A process lock prevents two CLI scheduler instances sharing a journal; a thread
lock prevents overlapping calls within an instance. Daily slots run once during
their configured minute. Slots missed while stopped or running another scan
are skipped; there is no catch-up queue. Keep collection times far enough apart
for the configured deadlines. Do not rotate or replace the receiver CSV during
an active run. Polling scans the CSV and is intended for this small deployment.

Software verification from the repository root:

```sh
python -m pytest sensor_data_reciever/tests -q
python -m unittest discover -s hardware-code/tests -v
```

Tests exercise mock ESP32 HTTP servers with the production receiver contract,
ordering after acceptance, duplicate/missing/invalid records, wrong sessions
and mappings, rejections, unreachable nodes, HTTP timeouts, restart behavior,
schedule slots and locks. Host C++ tests compile both real sketches in both
pacing modes and exercise busy rejection during acquisition. Arduino CLI target
compile commands are in the hardware README; the CLI is unavailable in this
environment. No physical hardware tests were performed.

Record physical results using the [commissioning checklist/results template](../docs/hardware_commissioning.md).
