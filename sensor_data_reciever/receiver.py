"""
receiver.py — HTTP server that accepts sensor data rows from an ESP32 over WiFi
               and appends them to a local CSV file.

Run with:
    python receiver.py

Endpoints
---------
POST /upload   — body is one raw CSV row (plain text, 9 comma-separated fields)
GET  /health   — returns 200 OK; use this to confirm the server is reachable
"""

# ============================================================
# CONFIGURATION  ← edit these two values to change behaviour
# ============================================================

PORT: int = 5000                        # listening port
CSV_FILE: str = "sensor_data.csv"       # output file name

# ============================================================
# IMPORTS
# ============================================================

import csv
import io
import os
import threading
from datetime import datetime, timezone
from uuid import uuid4

from flask import Flask, request, Response

import config
import validator

CSV_HEADER = ["record_id", "received_at", "accepted_at", *config.CSV_HEADER]

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# One lock shared by every request handler — prevents interleaved writes
# when the ESP32 sends a burst of 16 rows in rapid succession.
_file_lock = threading.Lock()


# ============================================================
# STARTUP — create the CSV with a header if it doesn't exist
# ============================================================

def _init_csv() -> None:
    """Create the output or reject an incompatible existing header.

    Caller must hold _file_lock when requests may be running.
    """
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
    else:
        with open(CSV_FILE, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != CSV_HEADER:
            raise ValueError(
                f"Incompatible CSV header in {CSV_FILE}; select a new output file "
                "or explicitly migrate the existing file before collecting data."
            )


# ============================================================
# ROUTES
# ============================================================

@app.route("/health", methods=["GET"])
def health() -> Response:
    """
    Simple liveness check.
    The ESP32 can GET /health before each session to confirm the server
    is up and reachable before it starts sending data.
    """
    return Response("OK\n", status=200, mimetype="text/plain")


@app.route("/upload", methods=["POST"])
def upload() -> Response:
    """
    Accept one CSV row as plain-text POST body, validate it, and append
    it to the CSV file.

    Expected body (no header, no quotes needed unless a field contains a comma):
        2024-01-15T12:00:00,1,1,3,7,45.2,23.1,60.5,512.0

    Returns 200 on success, 400 on any validation or parsing failure.
    The server never crashes on a bad row — all exceptions are caught and
    logged to the console.
    """
    received_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")

    try:
        # --- read the raw body as text ---
        try:
            raw_body = request.get_data().decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError:
            _log_rejected("<invalid UTF-8>", "Request body must be UTF-8")
            return Response("ERROR: request body must be UTF-8\n", status=400,
                            mimetype="text/plain")

        if not raw_body:
            _log_rejected("<empty body>", "Request body is empty")
            return Response("ERROR: empty body\n", status=400, mimetype="text/plain")

        # Require exactly one CSV record, including when fields are quoted.
        try:
            rows = list(csv.reader(io.StringIO(raw_body, newline=""), strict=True))
            if len(rows) != 1 or any("\n" in f or "\r" in f for f in rows[0]):
                raise csv.Error("Expected exactly one single-line CSV record")
            fields = rows[0]
        except csv.Error as exc:
            _log_rejected(raw_body, str(exc))
            return Response("ERROR: malformed CSV; expected one record\n",
                            status=400, mimetype="text/plain")

        # Strip surrounding whitespace from every field
        fields = [f.strip() for f in fields]

        # --- validate ---
        result = validator.validate_record(fields)
        if isinstance(result, list):
            error = "; ".join(f"{item.sensor}: {item.reason}" for item in result)
            node_id = fields[2] if len(fields) > 2 else ""
            _log_rejected(raw_body, error, node_id)
            return Response(f"ERROR: {error}\n", status=400, mimetype="text/plain")

        # Approval metadata is assigned only after successful validation.
        record_id = str(uuid4())
        accepted_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        stored_fields = [record_id, received_at, accepted_at, *fields]

        # --- write to disk (lock ensures no interleaved rows) ---
        with _file_lock:
            _init_csv()
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(stored_fields)
                f.flush()           # flush Python's buffer
                os.fsync(f.fileno()) # ask the OS to commit to disk

        # --- console confirmation ---
        _log_accepted(fields)

        return Response("OK\n", status=200, mimetype="text/plain")

    except Exception as exc:
        # Catch-all: log but never crash the server
        print(f"[ERROR] Unhandled exception in /upload: {exc}")
        return Response("ERROR: internal server error\n",
                        status=500, mimetype="text/plain")


# ============================================================
# CONSOLE LOGGING HELPERS
# ============================================================

def _log_accepted(fields: list[str]) -> None:
    """Print a one-liner for every successfully written row."""
    ts = datetime.now().strftime("%H:%M:%S")
    # fields order: timestamp, session_id, node_id, sampling_point,
    #               plant_id, soil, temperature, humidity, light
    print(
        f"[{ts}] [OK]      "
        f"session={fields[1]}  node={fields[2]}  "
        f"sp={fields[3]}  plant={fields[4]}  "
        f"soil={fields[5]}%  temp={fields[6]}°C  "
        f"hum={fields[7]}%  light={fields[8]} lux"
    )


def _log_rejected(raw: str, reason: str, node_id: str = "") -> None:
    """Print a one-liner for every rejected row, with the reason."""
    ts = datetime.now().strftime("%H:%M:%S")
    # Truncate very long bodies so the console stays readable
    preview = raw if len(raw) <= 120 else raw[:117] + "..."
    # A recoverable third field is the claimed source, not authenticated identity.
    node = node_id[:40] if node_id else "unknown"
    print(f"[{ts}] [REJECT]  node={node!r} {reason} | row: {preview!r}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Receive validated sensor CSV uploads")
    parser.add_argument("--output", default=CSV_FILE,
                        help="CSV destination; existing headers must match the stored schema")
    CSV_FILE = parser.parse_args().output

    # Initialise the output file before accepting any connections
    _init_csv()

    print("=" * 56)
    print("  Sensor Data Receiver  —  HTTP mode")
    print("=" * 56)
    print(f"  Listening  : 0.0.0.0:{PORT}")
    print(f"  Upload     : POST http://<this-ip>:{PORT}/upload")
    print(f"  Health     : GET  http://<this-ip>:{PORT}/health")
    print(f"  Output     : {CSV_FILE}")
    print("  Stop       : Ctrl-C")
    print("=" * 56)
    print()

    # threaded=True lets Flask handle multiple simultaneous connections
    # (one per OS thread) — important for rapid bursts from the ESP32.
    # Use host="0.0.0.0" so the server is reachable on the local network,
    # not just from localhost.
    app.run(host="0.0.0.0", port=PORT, threaded=True)
