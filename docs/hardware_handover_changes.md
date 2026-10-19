# Hardware handover preparation change report

Prepared 19 October 2026. Scope: configuration skeleton, commissioning evidence
templates, agent contract and setup entry points only. No runtime implementation
or data contract was changed. Existing unrelated worktree changes were retained;
this report describes this preparation task, not the complete worktree diff.

## Files created

| File | Why |
| --- | --- |
| [AGENTS.md](../AGENTS.md) | Repository-wide commissioning boundaries, permitted edits, approval requirements, verification and exact-file reporting contract. |
| [hardware_configuration.md](hardware_configuration.md) | Configuration registry: purpose, types, defaults, determination method, hardcoded limitations and deployment decision worksheet. |
| [hardware_commissioning.md](hardware_commissioning.md) | Offline/physical procedures with expected outcomes and evidence; reusable inventory, calibration, timing, sequential-run, deviation and sign-off forms. |
| [scheduler.commissioning.example.json](../sensor_data_reciever/scheduler.commissioning.example.json) | Existing scheduler format with unresolved site/network/physical metadata fields. Retains source-defined IDs/local points and existing example wait limits; no new configuration mechanism. |
| [test_commissioning_configuration.py](../sensor_data_reciever/tests/test_commissioning_configuration.py) | 36 offline cases covering unresolved template rejection, completed-template parsing, missing required settings and invalid origins, mappings, schedules, deadlines and shared paths. |
| [hardware_handover_changes.md](hardware_handover_changes.md) | This report and remaining dependency record. |

## Files modified

| File | Why |
| --- | --- |
| [README.md](../README.md) | Add the top-level commissioning entry point and current physical status. |
| [docs/README.md](README.md) | Link the configuration, commissioning and change documents and agent contract. |
| [plant_health_monitor_implementation.md](plant_health_monitor_implementation.md) | Mark older code/pin/data examples as design reference and point to current commissioning documents. |
| [hardware-code/README.md](../hardware-code/README.md) | Link commissioning documents and distinguish previously reported arrangements from physical verification evidence. |
| [sensor_data_reciever/README.md](../sensor_data_reciever/README.md) | Link handover, clarify current scheduler ownership and correct the outdated claim that existing CSV headers are not checked. |
| [SCHEDULER.md](../sensor_data_reciever/SCHEDULER.md) | Direct new operators to the unresolved commissioning template and distinguish older example metadata from physical assignments. |
| [VERIFICATION.md](../sensor_data_reciever/VERIFICATION.md) | Replace outdated physical instructions to add a firmware caller with the existing scheduler commissioning procedure; append this task's software results. |

## Verification evidence

Source-level comparison: SHA-256 before/after checks found **no changes in 11
runtime files**: both production sketches, `SensorHttpTransport.h`,
`SensorCommandServer.h`, `TransportConfig.h`, `TransportConfig.example.h`, and
`scheduler.py`, `receiver.py`, `validator.py`, `storage.py`, `config.py`.
No credentials or private local configuration were inspected or modified.

Commands run from the repository root using the existing Windows `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pytest sensor_data_reciever/tests/test_commissioning_configuration.py -q
# 36 passed in 0.28s
.\.venv\Scripts\python.exe -m pytest sensor_data_reciever/tests hardware-code/tests -q
# 225 passed, 24 subtests passed in 26.45s; no skips reported
```

The combined suite includes actual-sketch host g++ compilation/execution for
both nodes and both pacing modes, mock command/network behavior, and the real
receiver's validation/storage contract. These are software and host compilation
results. Arduino CLI was not available, so **ESP32 target compilation was not
run**. No physical ESP32, Jetson, LAN, sensor or calibration test was performed.
All corresponding commissioning results remain **PENDING HARDWARE VERIFICATION**.

The existing parser rejects the unresolved template and tested malformed/missing
required values. It is not a complete semantic commissioning validator: exact
per-node point sets, physical truth, nonempty path intent and metadata types
still require the documented review. No parser changes were introduced to hide
or expand this boundary. Existing tests were preserved.

## Remaining hardware dependencies and code configuration gaps

- Obtain actual board identities, FQBN, component models, wiring, probe/pot/group
  assignments, network credentials/addresses and address reservations.
- Measure sensor initialization/validity, per-probe calibration and uncertainty,
  MUX settling, scan/upload duration and appropriate timeouts. The common soil
  conversion is not a calibrated measurement.
- Establish deployment paths/permissions, clock/timezone, schedule, evidence
  retention, startup/recovery and backup procedures; observe one actual complete
  sequential run and an actual configured schedule slot.
- Pins remain hardcoded at `selectChannel`, `pinMode`, `analogRead`, `dht.setup`
  and `Wire.begin`; named constants and consistent substitutions are the minimal
  future exposure if wiring differs.
- Probe/channel/local-pot IDs remain two `channel + 1` expressions; independent
  mapping requires an approved identity-preserving lookup change.
- Per-probe calibration has no runtime configuration/application path; record
  evidence now, propose a separately approved change at the soil `map` site.
- Node IDs/counts, driver models/default initialization, upload 5000 `/upload`
  and command 80 `/collect` remain fixed implementation/interface choices.
  Changes must follow the boundaries and exact-site proposals in the
  [configuration guide](hardware_configuration.md#values-that-still-need-code-changes-if-hardware-differs).

No new acquisition behavior, calibration algorithm, service, schema, retry
policy, scheduler functionality or configuration-boundary code was necessary
for this preparation. Review and include the required new files when packaging
the handover; this task did not commit, deploy or flash anything.

## README setup simplification ? 19 October 2026

Documentation-only follow-up: reorganized the project READMEs around setup,
working directory, configuration, run commands and output files. Removed change
history, speculative features, repeated setup layers and lengthy implementation
explanations from operator entry points. Runtime code, tests, dependencies and
configuration values were not modified; no hardware verification was performed.

| File modified | Purpose |
| --- | --- |
| `README.md` | Direct collection startup path, Linux/Windows Python setup, optional analysis/test dependencies and component links. |
| `hardware-code/README.md` | Board/library setup, private network configuration, actual source pin table, pacing, compile/upload and collection checks. |
| `sensor_data_reciever/README.md` | Complete receiver/scheduler runbook: configuration, parsing, receiver, one-off run, daily schedule, output files and troubleshooting. |
| `eda/README.md` | Notebook environment/kernel and standalone diagnostics commands with input/output locations. |
| `plant_stress/README.md` | Separate Conda setup, dataset/configuration, ordered preprocessing/training/evaluation and saved outputs. Removed references to a nonexistent export script. |
| `dashboard/README.md` | Direct browser/local-server commands and explanation of embedded replay data. |
| `docs/README.md` | Short index separating runnable setup from commissioning and design references. |
| `sensor_data_reciever/archive/README.md` | Identify archived files and link current collection setup. |
| `sensor_data_reciever/SCHEDULER.md` | Retain behavior/failure/restart reference; replace duplicate startup steps with the single receiver runbook. |
| `docs/hardware_commissioning.md` | Point startup instructions to the receiver runbook. |
| `docs/hardware_handover_changes.md` | Record this follow-up and its verification. |

Verification: `python -m pytest sensor_data_reciever/tests hardware-code/tests -q`
using the root `.venv` reported **225 passed, 24 subtests passed in 23.98s**.
`receiver.py --help` and `scheduler.py --help` confirmed the documented arguments
without issuing commands or writing collection data. Local documentation links
were checked. Training, notebook analyses and real collection were not run;
ESP32 target compilation and all physical checks remain pending.
