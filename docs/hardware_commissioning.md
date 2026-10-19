# Hardware commissioning checklist and results template

Initial status of every physical check: **PENDING HARDWARE VERIFICATION**.
Copy this document into the team's evidence location, fill the results section,
and retain links to raw evidence. A checked box must have an observation and
evidence, not just the expected result copied into it.

Use the [configuration guide/worksheet](hardware_configuration.md) to select
values and find code-change boundaries; use [AGENTS.md](../AGENTS.md) for permitted
changes. Startup and command syntax live in the [scheduler setup](../sensor_data_reciever/SCHEDULER.md)
and [hardware setup](../hardware-code/README.md); do not substitute older design
document examples. This checklist creates no new runtime service or algorithm.

## Procedures

Run offline preparation first, then physical checks in order. Stop a physical
check when an earlier dependency fails; keep its result pending or failed.
Sensor collection can be verified as a transfer path while measurement
calibration remains pending, but do not label its soil readings calibrated.

| Check | Procedure | Expected result | Evidence to retain | Without hardware? |
| --- | --- | --- | --- | --- |
| C01 Software baseline | Record revision and uncommitted diff; run the combined existing test suite from AGENTS.md with the project environment. | Tests pass; report actual counts, skips and tool versions. | Command, full output, revision/diff and environment versions. | Yes: software only. |
| C02 Configuration preparation | Copy the commissioning JSON to the private local filename. Fill reviewed schedule, addresses, paths and all mappings; run the parsing-only command. Review every worksheet item; check writable existing directories and matched receiver/CSV path. | Valid JSON parses without issuing node commands; all runtime fields have deliberate values. Any unknown physical value remains pending in the record and prevents deployment sign-off. | Redacted config copy/hash, resolved paths, parser output and owner decisions. | Partly: parsing is offline; actual network/wiring values need site evidence. |
| C03 Target build | Identify actual board/FQBN and ESP32 core; install bundled libraries; compile both sketches using hardware README commands. Record real-mode selection and binary identity. | Both target builds succeed for the selected board; no unexplained local source changes. | CLI/compiler logs, FQBN/core/library versions, source revision and binary hashes. | Compilation can be done without boards if target toolchain is installed; board identification remains physical. |
| C04 Board identity and power | Label each actual ESP32 with board serial/MAC and logical ID; trace which sketch is flashed. Inspect wiring and power against board/component documentation before energizing. | Two distinct boards mapped to source-defined IDs 1 and 2, with safe connections and stable power. | Label photographs, board model, MAC/serial, flash logs and wiring inspection. | No: **PENDING HARDWARE VERIFICATION**. |
| C05 Network and receiver | Obtain real SSID/credential reference and IPv4s. Arrange address stability; configure the private transport header and reflash. Start receiver on the configured CSV; check `/health` from the node LAN and confirm node reachability/address leases. | Jetson port 5000 reachable; nodes have the recorded addresses; commands will target their port 80. No placeholder IP or credential used. | Redacted settings, address/lease records, health output and network observations; never passwords. | Only parser/mocks offline; physical LAN path pending. |
| C06 Probe/MUX/pot inventory | With safe power state, trace every S0..S3, ADC and occupied MUX channel. Label every probe and pot; fill one mapping row for each of Node 1's 12 and Node 2's 4 points. On powered testing, use controlled per-probe changes to confirm the intended channel responds. | Complete unique physical probe assignments matching emitted local IDs and owner-approved global pots/groups. Resolve any wiring mismatch before changing IDs. | Wiring diagram/photos, complete inventory, controlled stimulus notes and corresponding serial rows. | No: **PENDING HARDWARE VERIFICATION**. |
| C07 Sensor connection/initialization | Confirm DHT22 and BH1750 model, pins and I2C connection on each board. Capture boot at 115200 baud; inspect driver diagnostics and controlled readings against independent references. Setup alone does not expose a consolidated sensor-ready status. | Each local sensor initializes/produces plausible finite readings; installed pins and device address/mode match the build. | Per-node boot/serial logs, bus observations, component labels and reference readings. | No; mocks exercise source calls only. |
| C08 Calibration characterization | Agree method/units/acceptance tolerances with experiment owner. For every labeled probe, record repeated raw reference readings over the relevant conditions and determine candidate parameters/direction with uncertainty and environmental context. Raw ADC is not in the current CSV: retain external measurement evidence or propose a separately approved diagnostic change; do not silently extend the schema. | Traceable candidate calibration for each probe, reviewed against references. Current linear scaling remains explicitly uncalibrated until a separately approved implementation and verification exists. | Raw measurements, reference method/instrument, dates, repeatability, proposed parameters/units/formula and review. | No: **PENDING HARDWARE VERIFICATION**. |
| C09 Timing | Select real pacing in each sketch; measure scan/upload durations on the installed system and network. Assess MUX settling and reading stability using repeated controlled observations. Set scheduler deadlines above measured duration with documented margin; verify chosen upload timeouts. | Stable readings, complete scans within deadlines, collection times sufficiently separated. No deadline inflated merely to hide failed sensors. | Per-node duration measurements, test conditions, configured constants/timeouts and rationale. | Software pacing checks only offline; physical limits pending. |
| C10 Complete sequential run | Use the scheduler's `--once` with a dedicated commissioning CSV and persistent commissioning journal. Capture timestamped command/receiver evidence and serial logs; inspect the final JSON outcome and CSV for its exact session ID. | Node 1 completes all 12 expected unique points before Node 2 is commanded; Node 2 completes all 4. Both use the same session ID, mappings match, and journal status is `complete`. HTTP 202 alone is not success. | Redacted config/build identity, command/request timing capture, both serial logs, receiver log, 16 expected unique `(node,point)` pairs, CSV and journal. Additional duplicate rows, if any, are investigated. | Mocked ordering offline; actual run **PENDING HARDWARE VERIFICATION**. |
| C11 Failure and recovery | On commissioning outputs, arrange an unavailable node or controlled disconnection; observe finite failure and journal. Restore connectivity and make a new scheduled/manual run after the board is idle. Do not assume a failed request cancelled a scan. | Failure is recorded; Node 2 is not commanded when Node 1 fails/incompletes; later run can succeed. No duplicate command retry or overlapping controller added. | Fault/restoration times, timeout/error, node activity, journal and recovery run evidence. | Covered in mocks offline; physical network behavior pending. |
| C12 Schedule and operations | Start the configured scheduler and receiver using approved deployment paths/user. Observe an actual configured slot. Record startup/restart method, responsible operator and backup/restore steps; test these on commissioning outputs. | One sequential run for that slot and durable outcome; one scheduler with shared lock/history. Team understands missed slots are skipped and failed scans are not cancelled. Restart and evidence retention procedures are demonstrated before unattended sign-off. | Slot/timezone/clock evidence, CSV/journal, process startup/restart logs and backup/restore evidence. | Schedule/locks tested offline; installed operations pending. |

## Results template

Copy and fill; all physical status fields begin **PENDING HARDWARE VERIFICATION**.
Allowed outcomes are `PASS`, `FAIL`, or `PENDING HARDWARE VERIFICATION` for
physical checks; offline work may say `SOFTWARE PASS` or `NOT RUN` and must not
be presented as physical verification.

### Run identity and configuration

- Date/time/timezone, operator and reviewer: **PENDING HARDWARE VERIFICATION**
- Experiment owner and approved scope (transfer path / calibrated measurements / unattended operation): **PENDING HARDWARE VERIFICATION**
- Revision and diff archive; software environment and test output: **PENDING HARDWARE VERIFICATION**
- Completed [deployment worksheet](hardware_configuration.md#deployment-decision-worksheet-copy-into-commissioning-results) attachment: **PENDING HARDWARE VERIFICATION**
- Redacted runtime config/hash and private credential reference: **PENDING HARDWARE VERIFICATION**
- Board/build identities, firmware mode/constants and pin map: **PENDING HARDWARE VERIFICATION**
- Evidence root and retention/backup owner: **PENDING HARDWARE VERIFICATION**

### Per-check observations

Duplicate this row for C01..C12; describe observations separately from expectations.

| Check ID | Verification category (source / host mock / target compile / physical) | Status | Actual observation | Raw evidence link | Operator / reviewer / date |
| --- | --- | --- | --- | --- | --- |
| C__ | | PENDING HARDWARE VERIFICATION | | | |

### Board and local sensor inventory

Fill two rows, one per actual board; logical IDs 1/2 are software expectations.

| Logical node | Actual board label/serial/MAC | Model/FQBN/build hash | Actual IPv4/reservation | DHT model/GPIO/status | BH1750 address/SDA/SCL/status | MUX select pins/ADC GPIO/status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PENDING HARDWARE VERIFICATION | | | | | | | |

### Probe, channel and pot/group mapping

Create all 16 rows; expected occupied ranges are C0..C11 on node 1 and C0..C3
on node 2. Do not assign actual pots/groups from these ranges. Keep physical
probe labels stable; source currently serializes sensor and local pot ID as
channel + 1. Record mismatches rather than hiding them in host metadata.

| Node/board | MUX channel | Physical probe label | Emitted sampling_point | Emitted local plant_id | Global pot label (`plant_id` metadata) | Actual treatment group | Trace/stimulus evidence | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | PENDING HARDWARE VERIFICATION |

### Calibration evidence (one record per probe or reference sensor)

| Node + sensor/probe ID | Reference method/instrument/units | Conditions and repeated raw observations | Candidate parameters, direction, units and formula | Uncertainty/tolerance and reviewer decision | Evidence | Applied in code? / approved change reference | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | No; existing soil conversion is uncalibrated | PENDING HARDWARE VERIFICATION |

### Timing and complete-run evidence

- Node 1 and Node 2 observed duration distributions and test conditions: **PENDING HARDWARE VERIFICATION**
- Chosen settle/plant delays, upload timeouts and host deadlines with rationale: **PENDING HARDWARE VERIFICATION**
- Run session ID, scheduled/manual trigger and observed timezone: **PENDING HARDWARE VERIFICATION**
- Node 1 command/acceptance time and all 12 unique point receipts: **PENDING HARDWARE VERIFICATION**
- Node 2 command time after Node 1's last required receipt; all 4 unique point receipts: **PENDING HARDWARE VERIFICATION**
- Matching CSV and final journal outcome; rejected/duplicate rows and disposition: **PENDING HARDWARE VERIFICATION**
- Failure/recovery observations and installed startup/backup evidence: **PENDING HARDWARE VERIFICATION**

### Necessary deviation record (complete before implementation)

- Observed physical mismatch and evidence: **PENDING HARDWARE VERIFICATION**
- Exact file/symbols, existing value and proposed value/change: **PENDING HARDWARE VERIFICATION**
- Reason existing configuration is insufficient; smallest proposed change: **PENDING HARDWARE VERIFICATION**
- Interfaces/identities/measurement semantics affected; approval required under AGENTS.md?: **PENDING HARDWARE VERIFICATION**
- Project-owner approval reference where required: **PENDING HARDWARE VERIFICATION**
- Preserved behavior, regression/physical checks, and rollback plan: **PENDING HARDWARE VERIFICATION**
- Actual files changed, tests/evidence, reviewer and unresolved items: **PENDING HARDWARE VERIFICATION**

### Handover decision

- Transfer path physically verified: **PENDING HARDWARE VERIFICATION**
- Calibration/measurement claims verified: **PENDING HARDWARE VERIFICATION**
- Unattended startup/recovery/retention verified: **PENDING HARDWARE VERIFICATION**
- Remaining blockers, owner and next action: **PENDING HARDWARE VERIFICATION**
- Reviewer sign-off, date and supported operating scope: **PENDING HARDWARE VERIFICATION**
