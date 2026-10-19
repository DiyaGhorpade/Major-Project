# Hardware configuration boundaries and worksheet

Status: **PENDING HARDWARE VERIFICATION**. This document records source behavior
and unresolved decisions; it does not certify the installed hardware. Start with
the [commissioning checklist](hardware_commissioning.md). Developers and agents
must follow [AGENTS.md](../AGENTS.md).

## Designated configuration locations

| Location | Purpose and edit boundary |
| --- | --- |
| `hardware-code/libraries/SensorHttpTransport/src/TransportConfig.local.h` | Private build-time Wi-Fi, Jetson IPv4 and upload timeout definitions. Copy the existing `TransportConfig.example.h`; never commit credentials. Do not edit transport implementation to set credentials. |
| `sensor_data_reciever/scheduler.local.json` | Runtime node addresses, expected sampling points, plant/group metadata, schedule, paths and wait limits. Copy [scheduler.commissioning.example.json](../sensor_data_reciever/scheduler.commissioning.example.json), then resolve every placeholder. This file is already Git-ignored. |
| Both `hardware-code/data_log_producer*/*.ino` files | Existing `COLLECTION CONFIGURATION` constants only: pacing mode, MUX settle time and plant delay. These require separate builds for each sketch. |
| Receiver `--output PATH` | Select the HTTP CSV destination without editing Python. Use exactly this resolved path in scheduler `csv_file`. |
| This worksheet and the commissioning results copy | Record physical inventory, wiring, calibration evidence and changes not currently supported as configuration. These are documentation, not runtime inputs. |

The older `scheduler.example.json` is a software/example fixture. Its LAN
addresses, times, global plant IDs and groups are examples, not physical
assignments. The commissioning template retains only source-defined node IDs,
expected point keys and local record IDs. Empty strings, nulls and empty times
are deliberate unresolved values, not defaults. Do not start collection with
either example unreviewed.

## Transport configuration values

All definitions below are in the private header; strings use C++ double quotes.
Changing that header requires rebuilding and flashing both configured sketches.

| Value | Purpose / type or format | Default and safety | How to determine |
| --- | --- | --- | --- |
| `SENSOR_WIFI_SSID` | Wi-Fi network name; nonempty string | Empty fallback disables networking; safe for offline software, unusable for collection. Example's `YOUR_WIFI_SSID` is not a valid commissioning decision. | Network owner supplies the actual reachable network. |
| `SENSOR_WIFI_PASSWORD` | Network password; string | Empty only works for an intentionally open network; no verified credential default. | Network owner; record a credential reference, never the secret in results. |
| `SENSOR_JETSON_HOST` | Receiver's numeric LAN IPv4; no scheme, port or path | Empty fallback disables networking; example address is unverified. | Jetson interface address plus LAN reachability test; arrange address stability. |
| `SENSOR_HTTP_CONNECT_TIMEOUT_MS` | HTTP connection limit; integer 1..60000 milliseconds | 2000 is an existing software default, not measured LAN timing. | Retain initially; adjust only with measured connection evidence. |
| `SENSOR_HTTP_TIMEOUT_MS` | HTTP I/O timeout; integer 1..60000 milliseconds | 3000 is an existing software default, not a total scan deadline. | Measure uploads and failure behavior on the actual network. |

`TransportConfig.h` supplies fallbacks and static timeout bounds. Missing SSID
or malformed Jetson IPv4 results in `NotConfigured`; it does not prove valid
credentials, a reachable host, or successful Wi-Fi association. Existing tests
compile isolated headers and do not read your private credentials.

## Jetson JSON configuration values

Use actual JSON types (no comments or `PENDING` strings as completed values).
Scheduler paths resolve relative to the JSON file. Parent directories must
already exist with appropriate read/write permissions.

| Key | Purpose / required format | Safe default? / how to determine |
| --- | --- | --- |
| `timezone` | IANA timezone name string | No deployment default; confirm site timezone and clock. Requires system timezone data or `tzdata`. |
| `times` | Nonempty list of unique daily `HH:MM` strings | None; experiment owner chooses times after measuring run duration. No catch-up for missed slots. |
| `nodes` / `nodes[].id` | Ordered two-node list; IDs `"1"`, then `"2"` | Fixed current contract, not discoverable hardware IDs. Label actual boards against their flashed sketch; do not renumber during commissioning. |
| `nodes[].url` | HTTP origin using numeric IPv4, e.g. `http://<actual-ip>` | None. Default ESP32 command port is 80; obtain actual address and DHCP reservation/network plan. Optional port must match deployed routing. |
| `nodes[].sampling_points` | Object keyed by decimal sensor IDs: Node 1 `"1"`..`"12"`, Node 2 `"1"`..`"4"` for current firmware | Source-defined expected sets; verify every occupied channel physically. Do not shrink sets just to make a failed run pass. Parser enforces only the shared 1..12 range and a nonempty set, not the exact per-node topology. |
| `sampling_points.<id>.record_plant_id` | JSON integer matching firmware's local CSV plant ID, currently channel + 1 | Source-defined 1..12 / 1..4; physically confirm correspondence. Receiver range is 1..16. This is a completion match value, not global plant metadata. |
| `sampling_points.<id>.plant_id` | Global experiment pot label; use a positive JSON integer for this worksheet | No physical default. Owner assigns a stable label and records it against the probe/pot. Existing parser checks truthiness only; this metadata is not written into raw CSV. |
| `sampling_points.<id>.group` | Nonempty experiment group name string | None; obtain owner-approved treatment assignment. Parser checks truthiness only and does not reject arbitrary placeholder text. |
| `command_timeout` | Positive finite JSON number, seconds | Template retains software example 5; measure LAN command response. This is a socket operation limit, not a guarantee about an unresponsive node's actual scan state. |
| `completion_timeout` | Positive finite JSON number, seconds per accepted node | Template retains example 120; measure full scan/upload duration with margin. Acceptance alone is insufficient. |
| `poll_interval` | Positive finite JSON number, seconds between CSV checks | Template retains example 0.5; assess CSV size, disk load and desired detection latency. |
| `csv_file` | Nonempty filesystem path string | No deployment default; same resolved file as receiver `--output`. Do not select an incompatible legacy CSV. |
| `journal_file` | Nonempty filesystem path string, different from CSV | No deployment default; persistent writable file; associated `.lock` path must be shared by all invocations. Preserve history across restarts. |

Only the scheduler reads these mappings; they neither rewire probes nor change
the two firmware `channel + 1` expressions. Missing keys and many malformed values
raise parser exceptions; there is no complete semantic commissioning validator.
In particular blank path strings are not explicitly rejected, and nonempty
invented metadata can parse. Complete the checklist as well as the parsing check.

After editing, validate without issuing commands, from the repository root:

```sh
python -c "import sys; sys.path.insert(0, 'sensor_data_reciever'); from scheduler import load_settings; load_settings('sensor_data_reciever/scheduler.local.json'); print('Configuration parsed; hardware remains unverified')"
```

## Existing firmware settings and fixed hardware assumptions

Values in this table are **source expectations only**, all physical matches are
**PENDING HARDWARE VERIFICATION**. No new wiring or calibration has been assigned.

| Setting / source anchor in both sketches | Purpose / type or format | Current value; safety / determination |
| --- | --- | --- |
| `NODE_ID`, `CHANNEL_COUNT` | Integer logical identity and scan channel count | Node 1 / 12 and Node 2 / 4 are the current protected topology. Confirm board labels and connected channels; changes affect completion/identity contracts. |
| `SIMULATION_MODE` | Boolean pacing selector | `true` selects 100 ms; `false` selects 500 ms. Neither value proves reliable real hardware timing. Use real mode for commissioning and measure it. |
| `MUX_SETTLE_TIME` | Nonnegative unsigned milliseconds | Existing 10; must be validated with installed MUX/probes. |
| `PLANT_READ_DELAY` | Unsigned milliseconds after each row | Existing 100/500 conditional; validate sensor behavior and total scan duration. Not a collection schedule. |
| `selectChannel()`, MUX `pinMode()` calls | Four GPIO integer assignments, S0..S3 in bit order | Both sketches 16,17,18,19. No verified wiring default; trace installed connections. |
| `analogRead(34)` | MUX output ADC GPIO integer | Both 34; trace wiring and confirm ADC behavior. |
| `dht.setup(..., DHTesp::DHT22)` | DHT GPIO integer and device model enum | Node 1 GPIO4, Node 2 GPIO15; model DHT22. Read component labels and check wiring/initialization. |
| `Wire.begin(sda, scl)`, `lightMeter.begin()` | I2C GPIO pair and BH1750 default initialization | Node 1 25/26, Node 2 21/22. Device address/mode use library defaults; verify actual component and bus. |
| Two `String(channel + 1)` expressions in `collectSample()` | Physical channel to sensor ID and local pot ID | Contiguous C0..C11 / C0..C3 assumed; no independent mapping configuration. Record probe serial/label separately; preserve identity if moved. |
| `map(analogRead(34), 0, 4095, 0, 100)` | Common integer ADC-to-soil scaling | Existing source conversion only; **not calibrated soil moisture**. Endpoint counts, direction and per-probe behavior require measured references and approved methodology. |
| `Serial.begin(115200)` | Serial evidence capture baud integer | Existing software setting; match terminal to it. No need to tune for commissioning. |

## Values that still need code changes if hardware differs

Document and review the proposed change **before editing these sites**. No such
changes are made by this preparation task.

| Unavailable configuration | Exact minimal change to propose | Boundary |
| --- | --- | --- |
| GPIO assignments | Introduce named per-sketch constants for DHT, SDA/SCL, ADC and S0..S3; replace matching literals in initialization and access together. Keep existing defaults and channel-selection bit logic. | Pin-only substitutions with evidence are permitted commissioning changes under AGENTS.md; no sensor algorithm change. |
| Independent MUX/probe/local-pot mapping | Replace only the two ID serialization expressions with lookups from an explicit per-node mapping indexed by channel; document physical probe continuity and matching Jetson expected points. | Requires explicit approval first: changes emitted record identity. Do not move probes and simply relabel the scheduler. |
| Calibrated per-probe soil conversion | Propose measured parameters keyed by `(node_id, sampling_point)`, units, conversion direction, formula and evidence at the current `map(...)` site. No calibration loader or algorithm exists. | Requires approval of measurement semantics; record calibration now, implement separately. Never insert assumed dry/wet endpoints. |
| Different node IDs/counts, sparse occupied channels or sensor models | Propose the exact `NODE_ID`/`CHANNEL_COUNT`, loop/index, initialization and corresponding expectations that would change. | Requires approval of topology/identity or acquisition changes; retain current topology meanwhile. |
| Upload port/path or command port/path | Sender `http.begin(..., 5000, "/upload", ...)`, receiver `PORT=5000`/route, command `WebServer server_{80}`/`/collect`, scheduler request path. | Fixed interface, not ordinary environment configuration. Retain ports 5000/80; changing only `config.py.PORT` has no HTTP effect. |

`sensor_data_reciever/config.py` defines validation/schema constants, not hardware
calibration. `HOST`, `PORT`, `SENSOR_DATA_FILE` and `ERROR_LOG_FILE` there are not
the standalone HTTP receiver's bind/port/output controls. `storage.py` is a
separate nine-field utility; receiver storage adds three metadata columns.
Do not alter bounds to accept faulty sensors or confuse them with calibration.

## Deployment decision worksheet (copy into commissioning results)

All actual values below start **PENDING HARDWARE VERIFICATION**. Do not put passwords here.

| Item | Actual value / secure reference | Determined by / evidence |
| --- | --- | --- |
| Operator, experiment owner, date/time/timezone | PENDING HARDWARE VERIFICATION | |
| Repository revision plus local diff archive | PENDING HARDWARE VERIFICATION | |
| Board serial/MAC/label to node 1 and node 2 | PENDING HARDWARE VERIFICATION | |
| Board FQBN, ESP32 core, library/compiler versions | PENDING HARDWARE VERIFICATION | |
| SSID and private credential reference | PENDING HARDWARE VERIFICATION | |
| Jetson IPv4, both node IPv4s, address stability | PENDING HARDWARE VERIFICATION | |
| Python executable/venv, repository/install path and service user | PENDING HARDWARE VERIFICATION | |
| CSV, journal, lock, log and evidence paths; access permissions | PENDING HARDWARE VERIFICATION | |
| Schedule/timezone and measured wait/pacing limits | PENDING HARDWARE VERIFICATION | |
| Per-probe inventory, groups and calibration evidence | PENDING HARDWARE VERIFICATION | Use commissioning tables. |
| Startup/restart procedure, monitoring owner and backup/restore procedure | PENDING HARDWARE VERIFICATION | No boot supervision or backup automation is supplied by this task. |
