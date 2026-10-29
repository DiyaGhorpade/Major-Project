
from flask import Flask, render_template, jsonify
from datetime import datetime
from pathlib import Path
import csv
import random
import threading
import time

app = Flask(__name__)

CSV_FILE = Path(__file__).parent / "plant_sensor_data.csv"
UPDATE_INTERVAL = 2
LOCK = threading.Lock()

FIELDS = [
    "timestamp", "plant_id", "soil_moisture",
    "temperature_c", "humidity_percent", "light_lux"
]

plants = {
    f"MP-{i:02d}": {
        "soil_moisture": round(random.uniform(35, 65), 2),
        "temperature_c": round(random.uniform(23, 29), 2),
        "humidity_percent": round(random.uniform(50, 75), 2),
        "light_lux": round(random.uniform(200, 800), 2),
    }
    for i in range(1, 17)
}

latest_logs = []
total_readings = 0


def initialize_csv():
    if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
        with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def generate_logs():
    global latest_logs, total_readings

    while True:
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        batch = []

        with LOCK:
            for plant_id, values in plants.items():
                values["soil_moisture"] = round(max(
                    10, min(90, values["soil_moisture"]
                            + random.uniform(-0.7, 0.2))
                ), 2)

                values["temperature_c"] = round(max(
                    18, min(40, values["temperature_c"]
                            + random.uniform(-0.4, 0.4))
                ), 2)

                values["humidity_percent"] = round(max(
                    25, min(95, values["humidity_percent"]
                            + random.uniform(-1.5, 1.5))
                ), 2)

                values["light_lux"] = round(max(
                    0, min(1500, values["light_lux"]
                           + random.uniform(-40, 40))
                ), 2)

                batch.append({
                    "timestamp": timestamp,
                    "plant_id": plant_id,
                    **values,
                })

            with CSV_FILE.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerows(batch)

            latest_logs = batch
            total_readings += len(batch)

        time.sleep(UPDATE_INTERVAL)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/logs")
def get_logs():
    with LOCK:
        return jsonify({
            "logs": latest_logs,
            "total_readings": total_readings,
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        })


if __name__ == "__main__":
    initialize_csv()
    threading.Thread(target=generate_logs, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
