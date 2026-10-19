# Scheduler behavior

For configuration and run commands, use the [receiver and scheduler setup](README.md).
This reference explains completion, failure and restart behavior.

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

Software test commands and recorded results are in [VERIFICATION.md](VERIFICATION.md).
Physical results belong in the [commissioning checklist](../docs/hardware_commissioning.md).
