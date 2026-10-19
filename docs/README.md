# Documentation

For setup and run commands, use the component READMEs:

| Task | Start here |
| --- | --- |
| Set up Python and choose a workflow | [Repository README](../README.md) |
| Configure and flash the ESP32 nodes | [Hardware setup](../hardware-code/README.md) |
| Run the Jetson receiver and scheduler | [Collection setup](../sensor_data_reciever/README.md) |
| Run notebooks or diagnostics | [Analysis setup](../eda/README.md) |
| Train the classifier | [Training setup](../plant_stress/README.md) |
| Open the dashboard demo | [Dashboard setup](../dashboard/README.md) |

## Hardware commissioning

- [Configuration reference and worksheet](hardware_configuration.md): setting
  formats, source assumptions and locations for hardware-dependent values.
- [Commissioning checklist and results template](hardware_commissioning.md):
  physical procedures and evidence to record for the handover.
- [Preparation change report](hardware_handover_changes.md): files changed,
  software verification and remaining hardware dependencies.

## Design references

- [Architecture and design decisions](plant_health_monitor_architecture.md)
- [Implementation plan and code examples](plant_health_monitor_implementation.md)

These describe the broader planned sensor/image pipeline. Older pin assignments,
CSV examples and embedded scripts may differ from the current implementation;
use the setup READMEs for the runnable system.

The older [Jetson deployment guide](../sensor_data_reciever/docs/jetson_deployment_guide.docx)
is also reference material. Analysis documents are under `eda/docs/`, model
instructions under `plant_stress/`, and research papers under `papers/`, relative
to the repository root. Developer and agent rules are in [AGENTS.md](../AGENTS.md).
