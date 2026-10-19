# Hardware commissioning agent contract

This repository's working collection implementation must remain stable during
hardware commissioning. Read [configuration boundaries](docs/hardware_configuration.md)
and the [commissioning checklist/results template](docs/hardware_commissioning.md)
before changes. Older design examples are not the deployed implementation.

## Designated configuration and permitted changes

- Set private Wi-Fi/Jetson/upload-timeout definitions in
  `hardware-code/libraries/SensorHttpTransport/src/TransportConfig.local.h`
  using its existing example. Never commit credentials.
- Set node origins, schedules, expected points, plant/group metadata and paths
  in Git-ignored `sensor_data_reciever/scheduler.local.json`, using
  `scheduler.commissioning.example.json`. Preserve node order 1 then 2 and
  existing 12/4-point expectations unless an approved topology change exists.
- Use receiver `--output` for the CSV path. Match scheduler `csv_file` exactly.
- Edit existing `SIMULATION_MODE`, `MUX_SETTLE_TIME`, `PLANT_READ_DELAY`
  configuration constants in each sketch when supported by commissioning
  evidence. Test both sketches and document timing effects.
- For a proven wiring mismatch, first record exact proposed literal-to-constant
  substitutions and evidence in the commissioning deviation form. Then make
  the smallest pin-only change at the sites listed in the configuration guide;
  update corresponding mock expectations while preserving acquisition tests.
- Update commissioning records and setup documentation with observed facts.
  Record unresolved calibration and mappings; do not implement speculative fixes.

## Protected working logic

Both `hardware-code/data_log_producer*/*.ino` files outside the configuration
edits above contain protected sensor sequencing, mux selection and CSV emission.
`SensorHttpTransport.h` and `SensorCommandServer.h` in the shared library contain
protected networking, command validation and pending/busy behavior.
`sensor_data_reciever/scheduler.py`, `receiver.py`, `validator.py`, `storage.py`
and the schema/validation definitions in `config.py` contain protected ordering,
completion, validation and storage contracts. Bundled sensor drivers and existing
tests must not be weakened, replaced or broadly refactored during commissioning.

Obtain explicit project-owner approval before changing CSV fields/order/types or
measurement meaning, probe/local-pot identity serialization, node IDs/counts,
sensor algorithms/models, validation bounds, timestamps, endpoints/payloads/status
codes, retry/delivery behavior, session semantics, completion criteria or schedule
ownership. A global pot/group metadata edit in Jetson JSON does not authorize
rewriting raw record IDs. No second scheduler/service/configuration framework.

## Change and verification discipline

1. Inspect current worktree and preserve unrelated/uncommitted work. Make the
   smallest necessary change. Do not relocate code for aesthetics.
2. Describe any required hardcoded-value change before editing: exact site,
   hardware evidence, intended effect, preserved contracts and tests. Use the
   deviation form; request approval only when the protected boundary above is
   affected. Do independent preparation before waiting for approval.
3. Run the existing combined suite from the repository root with Flask, pytest,
   hypothesis, tzdata and g++ available:

   ```sh
   python -m pytest sensor_data_reciever/tests hardware-code/tests -q
   ```

   Use the project's `.venv` interpreter if the shell's Python differs. Run the
   parsing-only check in the configuration guide for edited local JSON. Never
   use `--once` as an offline configuration validation command: it sends commands.
   Retain tests for valid and missing/invalid configuration, original CSV/serial
   behavior, busy rejection, sequential completion, duplicates and timeouts.
4. When Arduino CLI and the matching ESP32 core are available, compile both
   sketches using commands in `hardware-code/README.md`; record FQBN/toolchain,
   logs and versions. Host mock g++ compilation is not target compilation.
5. Software/source inspection, host simulation, target compilation and physical
   verification are separate evidence categories. A physical feature stays
   **PENDING HARDWARE VERIFICATION** until actual-device procedure, dated raw
   evidence, configuration/revision, observations and reviewer sign-off are
   attached to the commissioning record. Do not infer a passing physical test
   from mocks, source constants, HTTP 202 or an example configuration.
6. Report exactly which files you modified and why, actual tests and skips,
   remaining hardware dependencies and any requested/approved architecture
   deviations. Update the handover change report for subsequent commissioning
   changes. Do not commit or discard unrelated work unless instructed.
