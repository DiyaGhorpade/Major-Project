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

# CSV column names, in the exact order the ESP32 sends them
CSV_HEADER: list[str] = [
    "timestamp",
    "session_id",
    "node_id",
    "sampling_point",
    "plant_id",
    "soil",
    "temperature",
    "humidity",
    "light",
]

EXPECTED_FIELDS: int = len(CSV_HEADER)  # 9 — derived from CSV_HEADER, not hardcoded


# ============================================================
# VALIDATION BOUNDS
# ============================================================

VALID_SESSION_ID_MIN     = 1
VALID_SAMPLING_POINT_MIN = 1
VALID_SAMPLING_POINT_MAX = 6
VALID_PLANT_ID_MIN       = 1
VALID_PLANT_ID_MAX       = 16
VALID_SOIL_MIN           = 0.0
VALID_SOIL_MAX           = 100.0
VALID_TEMP_MIN           = -40.0
VALID_TEMP_MAX           = 80.0
VALID_HUMIDITY_MIN       = 0.0
VALID_HUMIDITY_MAX       = 100.0
VALID_LIGHT_MIN          = 0.0
VALID_LIGHT_MAX          = 100_000.0   # lux; direct sunlight can exceed 1 000


# ============================================================
# IMPORTS
# ============================================================

import csv
import os
import threading
from datetime import datetime

from flask import Flask, request, Response

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
    """Create CSV_FILE with the header row if the file is absent."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"[INIT] Created '{CSV_FILE}' with header.")
    else:
        print(f"[INIT] '{CSV_FILE}' already exists — rows will be appended.")


# ============================================================
# VALIDATION
# ============================================================

def _validate(fields: list[str]) -> str | None:
    """
    Check that the 9 parsed fields are within expected types and ranges.

    Returns None on success, or a human-readable error string on failure.
    The check is fail-fast: the first problem found is returned immediately.
    """
    # --- structural ---
    if len(fields) != EXPECTED_FIELDS:
        return f"Expected {EXPECTED_FIELDS} fields, got {len(fields)}"

    timestamp, session_id_raw, node_id_raw, sampling_point_raw, \
        plant_id_raw, soil_raw, temp_raw, humidity_raw, light_raw = fields

    # --- timestamp: just check it's not blank ---
    if not timestamp.strip():
        return "timestamp is empty"

    # --- session_id ---
    try:
        session_id = int(session_id_raw)
    except ValueError:
        return f"session_id is not an integer: '{session_id_raw}'"
    if session_id < VALID_SESSION_ID_MIN:
        return f"session_id must be >= {VALID_SESSION_ID_MIN}, got {session_id}"

    # --- node_id: must be non-blank ---
    if not node_id_raw.strip():
        return "node_id is empty"

    # --- sampling_point ---
    try:
        sampling_point = int(sampling_point_raw)
    except ValueError:
        return f"sampling_point is not an integer: '{sampling_point_raw}'"
    if not (VALID_SAMPLING_POINT_MIN <= sampling_point <= VALID_SAMPLING_POINT_MAX):
        return (f"sampling_point out of range [{VALID_SAMPLING_POINT_MIN}–"
                f"{VALID_SAMPLING_POINT_MAX}]: {sampling_point}")

    # --- plant_id ---
    try:
        plant_id = int(plant_id_raw)
    except ValueError:
        return f"plant_id is not an integer: '{plant_id_raw}'"
    if not (VALID_PLANT_ID_MIN <= plant_id <= VALID_PLANT_ID_MAX):
        return (f"plant_id out of range [{VALID_PLANT_ID_MIN}–"
                f"{VALID_PLANT_ID_MAX}]: {plant_id}")

    # --- soil ---
    try:
        soil = float(soil_raw)
    except ValueError:
        return f"soil is not a number: '{soil_raw}'"
    if not (VALID_SOIL_MIN <= soil <= VALID_SOIL_MAX):
        return f"soil out of range [{VALID_SOIL_MIN}–{VALID_SOIL_MAX}]: {soil}"

    # --- temperature ---
    try:
        temp = float(temp_raw)
    except ValueError:
        return f"temperature is not a number: '{temp_raw}'"
    if not (VALID_TEMP_MIN <= temp <= VALID_TEMP_MAX):
        return f"temperature out of range [{VALID_TEMP_MIN}–{VALID_TEMP_MAX}]: {temp}"

    # --- humidity ---
    try:
        humidity = float(humidity_raw)
    except ValueError:
        return f"humidity is not a number: '{humidity_raw}'"
    if not (VALID_HUMIDITY_MIN <= humidity <= VALID_HUMIDITY_MAX):
        return (f"humidity out of range [{VALID_HUMIDITY_MIN}–"
                f"{VALID_HUMIDITY_MAX}]: {humidity}")

    # --- light ---
    try:
        light = float(light_raw)
    except ValueError:
        return f"light is not a number: '{light_raw}'"
    if not (VALID_LIGHT_MIN <= light <= VALID_LIGHT_MAX):
        return f"light out of range [{VALID_LIGHT_MIN}–{VALID_LIGHT_MAX}]: {light}"

    return None  # all checks passed


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
        2024-01-15T12:00:00,1,NODE_01,3,7,45.2,23.1,60.5,512.0

    Returns 200 on success, 400 on any validation or parsing failure.
    The server never crashes on a bad row — all exceptions are caught and
    logged to the console.
    """
    try:
        # --- read the raw body as text ---
        raw_body = request.get_data(as_text=True).strip()

        if not raw_body:
            _log_rejected("<empty body>", "Request body is empty")
            return Response("ERROR: empty body\n", status=400, mimetype="text/plain")

        # --- split on commas (csv.reader handles quoted fields correctly) ---
        try:
            fields = next(csv.reader([raw_body]))
        except StopIteration:
            _log_rejected(raw_body, "CSV parsing yielded no fields")
            return Response("ERROR: could not parse CSV row\n",
                            status=400, mimetype="text/plain")

        # Strip surrounding whitespace from every field
        fields = [f.strip() for f in fields]

        # --- validate ---
        error = _validate(fields)
        if error:
            _log_rejected(raw_body, error)
            return Response(f"ERROR: {error}\n", status=400, mimetype="text/plain")

        # --- write to disk (lock ensures no interleaved rows) ---
        with _file_lock:
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fields)
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


def _log_rejected(raw: str, reason: str) -> None:
    """Print a one-liner for every rejected row, with the reason."""
    ts = datetime.now().strftime("%H:%M:%S")
    # Truncate very long bodies so the console stays readable
    preview = raw if len(raw) <= 120 else raw[:117] + "..."
    print(f"[{ts}] [REJECT]  {reason} | row: {preview}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
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
