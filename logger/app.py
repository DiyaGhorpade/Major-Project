
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timezone
from pathlib import Path
import csv
import random
import threading
import time
from climate import Climate

app = Flask(__name__)

CSV_FILE = Path(__file__).parent / "sensor_data.csv"
UPDATE_INTERVAL = 2
LOCK = threading.Lock()

FIELDS = [
    "record_id", "received_at", "accepted_at", "timestamp", "session_id",
    "node_id", "sampling_point", "plant_id", "soil", "temperature",
    "humidity", "light",
]

plants = {
    (node, point): random.uniform(35, 65)
    for node, count in ((1, 12), (2, 4))
    for point in range(1, count + 1)
}

latest_logs = []
total_readings = 0
next_record_number = 1
last_timestamp = 0
started_at = time.monotonic()
climate = Climate()
last_scan_at = time.monotonic()


def initialize_csv():
    global latest_logs, total_readings, next_record_number, last_timestamp, started_at
    if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
        with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    with CSV_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError("Unexpected sensor_data.csv columns")
        latest_logs = list(reader)
    numbers = [int(row["record_id"].removeprefix("R-")) for row in latest_logs]
    next_record_number = max(numbers, default=0) + 1
    total_readings = len(latest_logs)
    last_timestamp = max((int(row["timestamp"]) for row in latest_logs), default=0)
    started_at = time.monotonic()
    # Restore each probe's last moisture instead of randomizing it on restart.
    for row in latest_logs:
        key = (int(row["node_id"]), int(row["plant_id"]))
        if key in plants:
            plants[key] = float(row["soil"])


def generate_scan():
    """Append one simulated collection using the receiver's stored schema."""
    global total_readings, next_record_number, last_scan_at
    with LOCK:
        scan_at = time.monotonic()
        elapsed = max(0, scan_at - last_scan_at)
        last_scan_at = scan_at
        ambient_temperature, ambient_humidity, ambient_light = climate.sample(elapsed=elapsed)
        session = max((int(row["session_id"]) for row in latest_logs), default=0) + 1
        batch = []
        for node, count in ((1, 12), (2, 4)):
            temperature = round(ambient_temperature - (0.3 if node == 2 else 0), 2)
            humidity = round(ambient_humidity + (2 if node == 2 else 0), 2)
            light = round(ambient_light * (0.06 if node == 2 else 1), 2)
            for point in range(1, count + 1):
                # Illustrative probe scale: gradual drying, slower in shade.
                drying_per_hour = (0.15 + 0.45 * light / 1000) * (0.7 if node == 2 else 1)
                plants[node, point] = max(10, plants[node, point] - drying_per_hour * elapsed / 3600)
                received = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                batch.append({
                    "record_id": f"R-{next_record_number + len(batch):05d}",
                    "received_at": received,
                    "accepted_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                    "timestamp": last_timestamp + 1 + int((time.monotonic() - started_at) * 1000),
                    "session_id": session,
                    "node_id": node,
                    "sampling_point": point,
                    "plant_id": point,
                    "soil": round(plants[node, point]),
                    "temperature": temperature,
                    "humidity": humidity,
                    "light": light,
                })
        with CSV_FILE.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerows(batch)
        latest_logs.extend(batch)
        total_readings += len(batch)
        next_record_number += len(batch)


def generate_logs():
    while True:
        generate_scan()
        time.sleep(UPDATE_INTERVAL)


@app.route("/")
def index():
    with LOCK:
        return render_template(
            "index.html", fields=FIELDS, initial_logs=latest_logs[-160:],
            initial_cursor=total_readings,
        )


@app.route("/api/logs")
def get_logs():
    with LOCK:
        return jsonify({
            "logs": latest_logs[max(0, request.args.get("after", default=0, type=int)):],
            "total_readings": total_readings,
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        })


if __name__ == "__main__":
    initialize_csv()
    threading.Thread(target=generate_logs, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
