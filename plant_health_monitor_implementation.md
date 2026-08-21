# Plant Health Monitor — Circuit, Firmware & Software Pipeline

Data format convention used throughout: `plant_id` is a string `P01`–`P16` (zero-padded, matching MUX channel + 1). All timestamps are ISO-8601 (`YYYY-MM-DD HH:MM:SS`). Every script downstream expects exactly these column names — do not rename them in one script without updating the next.

---

# PART 1 — CIRCUIT SETUP

### ESP32 → CD74HC4067

| CD74HC4067 Pin | ESP32 Pin | Notes |
|---|---|---|
| S0 | GPIO32 | channel select bit 0 |
| S1 | GPIO33 | channel select bit 1 |
| S2 | GPIO25 | channel select bit 2 |
| S3 | GPIO26 | channel select bit 3 |
| SIG (common) | GPIO36 (ADC1_CH0 / "VP") | analog input — must be an ADC1 pin because ADC2 is unusable once Wi-Fi is active |
| EN (enable) | GND | tie permanently low — chip is always enabled, since only one device uses the ADC line |
| VCC | 3.3V | do **not** use 5V — the MUX's analog output would exceed the ESP32 ADC's 3.3V max |
| GND | GND | common ground with ESP32 and all sensors |

**ADC pin used for SIG: GPIO36 (ADC1_CH0).** Reasoning: ADC2 pins share hardware with the Wi-Fi radio and give unreliable readings whenever Wi-Fi is active (irrelevant here since we're on USB serial, but ADC1 is still the safer, better-documented default and leaves ADC2 free for future expansion).

### 16 Soil Moisture Sensors

Each sensor's analog output goes to one MUX input channel (C0–C15), *not* directly to the ESP32. Every sensor shares the same VCC and GND rails (all tied to the ESP32's 3.3V and GND).

| Sensor | MUX Channel | VCC | GND | Analog Output |
|---|---|---|---|---|
| Sensor 1 → Plant P01 | C0 | 3.3V rail | GND rail | → MUX pin C0 |
| Sensor 2 → Plant P02 | C1 | 3.3V rail | GND rail | → MUX pin C1 |
| Sensor 3 → Plant P03 | C2 | 3.3V rail | GND rail | → MUX pin C2 |
| Sensor 4 → Plant P04 | C3 | 3.3V rail | GND rail | → MUX pin C3 |
| Sensor 5 → Plant P05 | C4 | 3.3V rail | GND rail | → MUX pin C4 |
| Sensor 6 → Plant P06 | C5 | 3.3V rail | GND rail | → MUX pin C5 |
| Sensor 7 → Plant P07 | C6 | 3.3V rail | GND rail | → MUX pin C6 |
| Sensor 8 → Plant P08 | C7 | 3.3V rail | GND rail | → MUX pin C7 |
| Sensor 9 → Plant P09 | C8 | 3.3V rail | GND rail | → MUX pin C8 |
| Sensor 10 → Plant P10 | C9 | 3.3V rail | GND rail | → MUX pin C9 |
| Sensor 11 → Plant P11 | C10 | 3.3V rail | GND rail | → MUX pin C10 |
| Sensor 12 → Plant P12 | C11 | 3.3V rail | GND rail | → MUX pin C11 |
| Sensor 13 → Plant P13 | C12 | 3.3V rail | GND rail | → MUX pin C12 |
| Sensor 14 → Plant P14 | C13 | 3.3V rail | GND rail | → MUX pin C13 |
| Sensor 15 → Plant P15 | C14 | 3.3V rail | GND rail | → MUX pin C14 |
| Sensor 16 → Plant P16 | C15 | 3.3V rail | GND rail | → MUX pin C15 |

The channel-select truth table the firmware uses (S3 S2 S1 S0):

| Channel | S3 | S2 | S1 | S0 |
|---|---|---|---|---|
| C0 | 0 | 0 | 0 | 0 |
| C1 | 0 | 0 | 0 | 1 |
| ... | | | | |
| C15 | 1 | 1 | 1 | 1 |

(the firmware computes this from `channel` directly with bit-shifts — no need to hardcode all 16 rows)

### DHT22

| Pin | Connection |
|---|---|
| VCC | 3.3V |
| GND | GND |
| DATA | GPIO4 |
| ESP32 GPIO | GPIO4 |
| Pull-up resistor | **Required**: 10 kΩ between DATA and VCC. Most breakout modules already include this on-board — check before adding a second one, which would over-pull the line. |

### BH1750

| Pin | Connection |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO21 |
| SCL | GPIO22 |
| ADDR | tie to GND → I2C address `0x23` |

### ESP32 → Jetson (USB Serial)

**Baud rate: 115200.**

**What gets sent:** once per full MUX sweep (all 16 channels read), the ESP32 emits 16 lines — one per plant — followed by nothing else. Ambient values (temperature, humidity, light) are sampled once per sweep and repeated on all 16 lines so every row is self-contained and independently parseable (no line depends on state from another line).

**Message format (CSV, one plant reading per line):**

```
timestamp,plant_id,soil_raw,soil_pct,temperature_c,humidity_pct,light_lux,status
```

**Example line:**

```
2026-08-21 10:30:05,P07,1820,42.30,27.4,61.2,850.0,OK
```

`status` is `OK`, `DHT_ERR`, `BH1750_ERR`, or `SOIL_ERR` (soil reading out of the 0–4095 ADC range, wiring fault). Choosing per-plant lines over one giant packed line keeps parsing trivial on the Jetson side (one `csv.reader` row = one DB row) and keeps the format identical whether you're logging 1 plant or 16.

---

# PART 2 — ESP32 SENSOR SCRIPT

**Input:** none (reads live sensors). **Output:** CSV lines over USB serial, one per plant, in the format from Part 1. **Feeds into:** Part 3 (Jetson data logger), which parses this exact format.

```cpp
// esp32_sensor_node.ino
// ESP32-WROOM-32 firmware: 16-channel soil moisture MUX + DHT22 + BH1750
// Sends one CSV line per plant per sweep over USB serial at 115200 baud.

#include <Wire.h>
#include <DHT.h>
#include <BH1750.h>

// ---- Pin definitions ----
#define MUX_S0 32
#define MUX_S1 33
#define MUX_S2 25
#define MUX_S3 26
#define MUX_SIG 36   // ADC1_CH0
#define DHT_PIN 4
#define DHT_TYPE DHT22

DHT dht(DHT_PIN, DHT_TYPE);
BH1750 lightMeter;

// ---- Calibration (see calibration section below) ----
// Per-sensor dry/wet raw ADC values. Fill these in after running the
// calibration routine for each of the 16 physical sensors.
int DRY_VALUE[16] = {3000,3000,3000,3000,3000,3000,3000,3000,
                     3000,3000,3000,3000,3000,3000,3000,3000};
int WET_VALUE[16] = {1200,1200,1200,1200,1200,1200,1200,1200,
                     1200,1200,1200,1200,1200,1200,1200,1200};

const unsigned long SWEEP_INTERVAL_MS = 5000; // one full 16-channel sweep every 5s
unsigned long lastSweep = 0;

void selectMuxChannel(uint8_t channel) {
  digitalWrite(MUX_S0, (channel >> 0) & 0x01);
  digitalWrite(MUX_S1, (channel >> 1) & 0x01);
  digitalWrite(MUX_S2, (channel >> 2) & 0x01);
  digitalWrite(MUX_S3, (channel >> 3) & 0x01);
  delayMicroseconds(50); // let the analog switch settle
}

float rawToPercent(int raw, int channel) {
  int dry = DRY_VALUE[channel];
  int wet = WET_VALUE[channel];
  if (dry == wet) return -1.0; // uncalibrated
  float pct = (float)(dry - raw) / (float)(dry - wet) * 100.0;
  if (pct < 0) pct = 0;
  if (pct > 100) pct = 100;
  return pct;
}

// Placeholder — replace with a synced RTC/NTP-derived string if available.
// The Jetson also stamps its own arrival time in Part 3, so this only
// needs to be monotonically useful, not perfectly wall-clock accurate.
String getTimestamp() {
  unsigned long ms = millis();
  char buf[32];
  snprintf(buf, sizeof(buf), "BOOT+%lums", ms);
  return String(buf);
}

void setup() {
  Serial.begin(115200);
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_S3, OUTPUT);
  analogReadResolution(12); // 0-4095

  dht.begin();
  Wire.begin(21, 22); // SDA, SCL
  lightMeter.begin();

  delay(2000); // let sensors stabilize
}

void loop() {
  if (millis() - lastSweep < SWEEP_INTERVAL_MS) return;
  lastSweep = millis();

  // Ambient sensors: read once per sweep, shared across all 16 lines
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  float lux = lightMeter.readLightLevel();

  bool dhtError = isnan(temperature) || isnan(humidity);
  bool luxError = (lux < 0);

  String ts = getTimestamp();

  for (uint8_t ch = 0; ch < 16; ch++) {
    selectMuxChannel(ch);
    int raw = analogRead(MUX_SIG);

    bool soilError = (raw < 0 || raw > 4095);
    float pct = soilError ? -1.0 : rawToPercent(raw, ch);

    String status = "OK";
    if (soilError) status = "SOIL_ERR";
    else if (dhtError) status = "DHT_ERR";
    else if (luxError) status = "BH1750_ERR";

    char plantId[5];
    snprintf(plantId, sizeof(plantId), "P%02d", ch + 1);

    Serial.print(ts); Serial.print(",");
    Serial.print(plantId); Serial.print(",");
    Serial.print(raw); Serial.print(",");
    Serial.print(pct, 2); Serial.print(",");
    Serial.print(dhtError ? -1.0 : temperature, 1); Serial.print(",");
    Serial.print(dhtError ? -1.0 : humidity, 1); Serial.print(",");
    Serial.print(luxError ? -1.0 : lux, 1); Serial.print(",");
    Serial.println(status);
  }
}
```

### Calibration section

Each of the 16 physical sensors needs its **own** `DRY_VALUE`/`WET_VALUE` pair — capacitive sensors vary sensor-to-sensor by a few hundred raw ADC counts even from the same batch, and averaging them into one shared calibration will bias every plant's moisture % by a fixed offset.

Procedure per sensor:
1. **Dry value**: leave the sensor in open air (or fully dry soil), read the raw ADC value for ~30s, average it → `DRY_VALUE[i]`.
2. **Wet value**: submerge the sensing prongs in water (not the PCB/electronics), read raw ADC for ~30s, average → `WET_VALUE[i]`.
3. **Conversion to percentage**: `pct = (dry - raw) / (dry - wet) * 100`, clamped to [0, 100]. This assumes raw value *decreases* as moisture increases (true for most capacitive probes — verify polarity on your specific board before trusting the sign).
4. Repeat for all 16 channels — a quick way is to plug one physical sensor into the multiplexer, dry/wet-calibrate it, unplug, repeat with the next physical sensor into the *same* channel (since it's the sensor's manufacturing variance that matters, not the MUX channel).

---

# PART 3 — JETSON DATA LOGGER

**Input:** raw CSV lines from ESP32 over `/dev/ttyUSB0` (or `/dev/ttyACM0`). **Output:** `sensor_log.db` (SQLite table `raw_readings`). **Feeds into:** Part 4 (cleaning script), which reads directly from this table.

```python
# jetson_data_logger.py
# Reads ESP32 serial stream, validates, and persists to SQLite.

import serial
import sqlite3
import time
import sys
from datetime import datetime

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
DB_PATH = "sensor_log.db"

EXPECTED_FIELDS = 8  # timestamp,plant_id,soil_raw,soil_pct,temp,humidity,light,status

VALID_RANGES = {
    "soil_raw": (0, 4095),
    "soil_pct": (0, 100),
    "temperature_c": (-10, 60),
    "humidity_pct": (0, 100),
    "light_lux": (0, 100000),
}


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            esp32_timestamp TEXT,
            jetson_timestamp TEXT NOT NULL,
            plant_id TEXT NOT NULL,
            soil_raw INTEGER,
            soil_pct REAL,
            temperature_c REAL,
            humidity_pct REAL,
            light_lux REAL,
            status TEXT
        )
    """)
    conn.commit()


def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) != EXPECTED_FIELDS:
        return None, f"wrong field count ({len(parts)})"

    esp32_ts, plant_id, soil_raw, soil_pct, temp, hum, light, status = parts

    try:
        soil_raw = int(soil_raw)
        soil_pct = float(soil_pct)
        temp = float(temp)
        hum = float(hum)
        light = float(light)
    except ValueError as e:
        return None, f"parse error: {e}"

    if not plant_id.startswith("P") or len(plant_id) != 3:
        return None, f"bad plant_id: {plant_id}"

    record = {
        "esp32_timestamp": esp32_ts,
        "jetson_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plant_id": plant_id,
        "soil_raw": soil_raw,
        "soil_pct": soil_pct,
        "temperature_c": temp,
        "humidity_pct": hum,
        "light_lux": light,
        "status": status,
    }
    return record, None


def insert_record(conn, record):
    conn.execute("""
        INSERT INTO raw_readings
        (esp32_timestamp, jetson_timestamp, plant_id, soil_raw, soil_pct,
         temperature_c, humidity_pct, light_lux, status)
        VALUES (:esp32_timestamp, :jetson_timestamp, :plant_id, :soil_raw, :soil_pct,
                :temperature_c, :humidity_pct, :light_lux, :status)
    """, record)
    conn.commit()


def open_serial_with_retry(port, baud, retries=10, delay=3):
    for attempt in range(retries):
        try:
            ser = serial.Serial(port, baud, timeout=2)
            print(f"Connected to {port}")
            return ser
        except serial.SerialException:
            print(f"ESP32 not found on {port}, retry {attempt+1}/{retries}...")
            time.sleep(delay)
    raise RuntimeError(f"Could not open {port} after {retries} attempts")


def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    ser = open_serial_with_retry(SERIAL_PORT, BAUD_RATE)

    while True:
        try:
            raw_bytes = ser.readline()
            if not raw_bytes:
                continue  # timeout, no data this cycle

            try:
                line = raw_bytes.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                print("Corrupted serial bytes, skipping line")
                continue

            if not line.strip():
                continue

            record, err = parse_line(line)
            if err:
                print(f"Skipping malformed line ({err}): {line.strip()}")
                continue

            # flag but still store out-of-range values instead of dropping,
            # so Part 4 can decide how to treat them with full history intact
            for field, (lo, hi) in VALID_RANGES.items():
                val = record[field]
                if val != -1.0 and not (lo <= val <= hi):
                    record["status"] = record["status"] + f"|RANGE_{field}"

            insert_record(conn, record)

        except serial.SerialException:
            print("ESP32 disconnected. Attempting to reconnect...")
            ser.close()
            ser = open_serial_with_retry(SERIAL_PORT, BAUD_RATE)
        except KeyboardInterrupt:
            print("Stopping logger.")
            break

    ser.close()
    conn.close()


if __name__ == "__main__":
    main()
```

---

# PART 4 — DATA CLEANING SCRIPT

**Input:** `sensor_log.db` (`raw_readings` table from Part 3). **Output:** `cleaned_readings.csv` + `flagged_readings.csv` (never silently deleted — flagged rows are kept, separately, with the reason). **Feeds into:** Part 6 (synchronization), which reads `cleaned_readings.csv`.

```python
# clean_sensor_data.py
# Cleans raw sensor log without blindly deleting rows: every suspicious
# reading is flagged with a reason and routed to a separate file.

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "sensor_log.db"
CLEANED_OUT = "cleaned_readings.csv"
FLAGGED_OUT = "flagged_readings.csv"

# Max plausible change between consecutive readings of the same plant
# (sweep interval is 5s, so these are deliberately generous)
SPIKE_THRESHOLD = {
    "soil_pct": 25.0,      # % points per sweep
    "temperature_c": 5.0,  # °C per sweep
    "humidity_pct": 15.0,  # % per sweep
    "light_lux": 20000.0,  # lux per sweep (light can jump a lot legitimately)
}

VALID_RANGES = {
    "soil_pct": (0, 100),
    "temperature_c": (-10, 60),
    "humidity_pct": (0, 100),
    "light_lux": (0, 100000),
}


def load_raw():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM raw_readings", conn)
    conn.close()
    return df


def flag_missing(df):
    """Rule: any core numeric field that is null, -1, or NaN is missing data."""
    core_fields = ["soil_pct", "temperature_c", "humidity_pct", "light_lux"]
    missing_mask = df[core_fields].isna().any(axis=1) | (df[core_fields] == -1.0).any(axis=1)
    df.loc[missing_mask, "flag_reason"] = df.loc[missing_mask, "flag_reason"].fillna("") + "MISSING_VALUE;"
    return df


def flag_impossible(df):
    """Rule: values outside physically possible sensor ranges."""
    for field, (lo, hi) in VALID_RANGES.items():
        out_of_range = ~df[field].between(lo, hi) & df[field].notna() & (df[field] != -1.0)
        df.loc[out_of_range, "flag_reason"] = df.loc[out_of_range, "flag_reason"].fillna("") + f"IMPOSSIBLE_{field};"
    return df


def flag_duplicates(df):
    """Rule: identical (plant_id, jetson_timestamp) pair means the same
    sweep record was logged twice — e.g. ESP32 resend after a dropped ack."""
    dup_mask = df.duplicated(subset=["plant_id", "jetson_timestamp"], keep="first")
    df.loc[dup_mask, "flag_reason"] = df.loc[dup_mask, "flag_reason"].fillna("") + "DUPLICATE;"
    return df


def flag_spikes(df):
    """Rule: a jump larger than SPIKE_THRESHOLD between two consecutive
    readings of the SAME plant, sorted by time, likely means a probe
    was bumped, unplugged, or the MUX misfired — not a real physiological
    change (soil moisture and temperature can't move that fast)."""
    df = df.sort_values(["plant_id", "jetson_timestamp"]).reset_index(drop=True)
    for field, threshold in SPIKE_THRESHOLD.items():
        diffs = df.groupby("plant_id")[field].diff().abs()
        spike_mask = diffs > threshold
        df.loc[spike_mask, "flag_reason"] = df.loc[spike_mask, "flag_reason"].fillna("") + f"SPIKE_{field};"
    return df


def flag_disconnected(df):
    """Rule: a plant_id with an identical, unchanging soil_pct across many
    consecutive rows (stuck ADC / disconnected probe reading a fixed value)."""
    df = df.sort_values(["plant_id", "jetson_timestamp"]).reset_index(drop=True)
    STUCK_RUN_LENGTH = 20  # ~100s of identical readings at 5s interval
    for plant_id, group in df.groupby("plant_id"):
        vals = group["soil_pct"].values
        run_len = 1
        for i in range(1, len(vals)):
            if vals[i] == vals[i - 1]:
                run_len += 1
                if run_len >= STUCK_RUN_LENGTH:
                    idx = group.index[i]
                    df.loc[idx, "flag_reason"] = str(df.loc[idx, "flag_reason"] or "") + "STUCK_SENSOR;"
            else:
                run_len = 1
    return df


def flag_timestamp_issues(df):
    """Rule: jetson_timestamp not parseable, or moving backwards relative
    to the previous row (clock reset / logger restart mid-run)."""
    df["jetson_timestamp_parsed"] = pd.to_datetime(df["jetson_timestamp"], errors="coerce")
    bad_ts = df["jetson_timestamp_parsed"].isna()
    df.loc[bad_ts, "flag_reason"] = df.loc[bad_ts, "flag_reason"].fillna("") + "BAD_TIMESTAMP;"

    df_sorted = df.sort_values("id")
    backwards = df_sorted["jetson_timestamp_parsed"].diff().dt.total_seconds() < 0
    backwards_idx = df_sorted.index[backwards.fillna(False)]
    df.loc[backwards_idx, "flag_reason"] = df.loc[backwards_idx, "flag_reason"].fillna("") + "TIMESTAMP_REGRESSION;"
    return df


def main():
    df = load_raw()
    df["flag_reason"] = ""

    df = flag_missing(df)
    df = flag_impossible(df)
    df = flag_duplicates(df)
    df = flag_spikes(df)
    df = flag_disconnected(df)
    df = flag_timestamp_issues(df)

    is_clean = df["flag_reason"] == ""
    cleaned = df[is_clean].drop(columns=["flag_reason", "jetson_timestamp_parsed"], errors="ignore")
    flagged = df[~is_clean]

    cleaned.to_csv(CLEANED_OUT, index=False)
    flagged.to_csv(FLAGGED_OUT, index=False)

    print(f"Clean rows: {len(cleaned)} | Flagged rows: {len(flagged)}")
    if len(flagged):
        print(flagged["flag_reason"].value_counts())


if __name__ == "__main__":
    main()
```

---

# PART 5 — IMAGE CAPTURE SCRIPT

**Input:** plant ID selection (CLI arg or loop over all 16). **Output:** `images/<plant_id>/<plant_id>_<timestamp>.jpg` + an entry appended to `image_index.csv`. **Feeds into:** Part 6 (synchronization) and Part 8 (vision training), both of which key off `image_index.csv`.

```python
# capture_plant_images.py
# Jetson + IMX219 (CSI) capture, fixed booth conditions.
# Uses GStreamer via OpenCV, which is the standard IMX219 pipeline on Jetson.

import cv2
import os
import csv
from datetime import datetime

IMAGE_ROOT = "images"
INDEX_FILE = "image_index.csv"

# nvarguscamerasrc is the Jetson CSI driver pipeline for IMX219.
# Fixed exposure/white balance are set on the sensor side in the booth,
# so this pipeline just needs to capture, not auto-adjust.
GST_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=21/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)


def ensure_index_file():
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["plant_id", "timestamp", "image_path"])


def capture_image(plant_id):
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Could not open IMX219 camera via GStreamer pipeline")

    # discard first few frames — auto-exposure/AWB settle even with
    # manual booth lighting, first frames off a cold sensor can be off
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to capture frame for {plant_id}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_for_filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    plant_dir = os.path.join(IMAGE_ROOT, plant_id)
    os.makedirs(plant_dir, exist_ok=True)

    filename = f"{plant_id}_{ts_for_filename}.jpg"
    image_path = os.path.join(plant_dir, filename)
    cv2.imwrite(image_path, frame)

    with open(INDEX_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([plant_id, timestamp, image_path])

    return image_path, timestamp


def capture_all_plants(plant_ids):
    ensure_index_file()
    results = []
    for plant_id in plant_ids:
        input(f"Place {plant_id} at the imaging station, then press Enter...")
        path, ts = capture_image(plant_id)
        print(f"Captured {plant_id} -> {path} @ {ts}")
        results.append((plant_id, path, ts))
    return results


if __name__ == "__main__":
    all_plants = [f"P{str(i).zfill(2)}" for i in range(1, 17)]
    capture_all_plants(all_plants)
```

---

# PART 6 — IMAGE + SENSOR SYNCHRONIZATION

**Input:** `image_index.csv` (Part 5) + `cleaned_readings.csv` (Part 4). **Output:** `synced_dataset.csv` — one row per image, with a matched sensor feature window. **Feeds into:** Parts 7–9 (training scripts read this file directly).

**Which window to use, and why:** a single "latest reading" is too noisy — one MUX misread or transient spike right before the photo would corrupt the label's sensor features. A 15-minute average is too slow to reflect the plant's *current* state, since soil moisture and light can shift meaningfully within that span. The best fit here is a **rolling window blend**: the **5-minute average** as the primary feature (smooths sensor noise while staying close to "now"), plus a **short-term trend** (slope of soil moisture over the preceding 15 minutes) as a secondary feature, since a plant that's been steadily drying out is a stronger water-stress signal than a single moisture percentage. This gives the model both a stable snapshot and a directionality cue, without the lag of a full 15-minute average as the primary value.

```python
# sync_image_sensor.py

import pandas as pd
from datetime import timedelta

IMAGE_INDEX = "image_index.csv"
CLEANED_SENSOR = "cleaned_readings.csv"
OUTPUT = "synced_dataset.csv"

WINDOW_AVG_MINUTES = 5
WINDOW_TREND_MINUTES = 15


def load_data():
    images = pd.read_csv(IMAGE_INDEX, parse_dates=["timestamp"])
    sensors = pd.read_csv(CLEANED_SENSOR, parse_dates=["jetson_timestamp"])
    return images, sensors


def compute_window_features(sensors, plant_id, image_time):
    plant_sensors = sensors[sensors["plant_id"] == plant_id]

    avg_start = image_time - timedelta(minutes=WINDOW_AVG_MINUTES)
    avg_window = plant_sensors[
        (plant_sensors["jetson_timestamp"] >= avg_start)
        & (plant_sensors["jetson_timestamp"] <= image_time)
    ]

    trend_start = image_time - timedelta(minutes=WINDOW_TREND_MINUTES)
    trend_window = plant_sensors[
        (plant_sensors["jetson_timestamp"] >= trend_start)
        & (plant_sensors["jetson_timestamp"] <= image_time)
    ].sort_values("jetson_timestamp")

    if avg_window.empty:
        return None  # no sensor coverage near this image — cannot label it

    features = {
        "soil_pct_avg5min": avg_window["soil_pct"].mean(),
        "temperature_c_avg5min": avg_window["temperature_c"].mean(),
        "humidity_pct_avg5min": avg_window["humidity_pct"].mean(),
        "light_lux_avg5min": avg_window["light_lux"].mean(),
        "soil_pct_std5min": avg_window["soil_pct"].std(ddof=0) or 0.0,
    }

    # trend: simple linear slope of soil moisture over the 15-min window
    if len(trend_window) >= 2:
        t0 = trend_window["jetson_timestamp"].iloc[0]
        minutes_elapsed = (trend_window["jetson_timestamp"] - t0).dt.total_seconds() / 60.0
        slope = np.polyfit(minutes_elapsed, trend_window["soil_pct"], 1)[0]
        features["soil_pct_trend_15min"] = slope
    else:
        features["soil_pct_trend_15min"] = 0.0

    return features


def main():
    import numpy as np
    global np
    images, sensors = load_data()

    rows = []
    for _, img_row in images.iterrows():
        feats = compute_window_features(sensors, img_row["plant_id"], img_row["timestamp"])
        if feats is None:
            print(f"Skipping {img_row['plant_id']} @ {img_row['timestamp']}: no sensor coverage")
            continue
        row = {
            "plant_id": img_row["plant_id"],
            "image_timestamp": img_row["timestamp"],
            "image_path": img_row["image_path"],
            **feats,
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    print(f"Synced {len(result)} image-sensor pairs -> {OUTPUT}")


if __name__ == "__main__":
    main()
```

---

# PART 7 — SENSOR ML SCRIPT (XGBoost)

**Input:** `synced_dataset.csv` (Part 6), plus a `label` column you attach based on experimental group (`A→Healthy, B→Water Stress, C→Light Stress, D→Nutrient Stress`). **Output:** `sensor_model.json` (trained model) + printed metrics. **Feeds into:** Part 9 (fusion) and Part 10 (real-time inference), which load `sensor_model.json` for inference.

```python
# train_sensor_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)
import joblib

DATA_PATH = "synced_dataset_labeled.csv"  # synced_dataset.csv + a 'label' column
MODEL_OUT = "sensor_model.json"
ENCODER_OUT = "sensor_label_encoder.pkl"

FEATURE_COLS = [
    "soil_pct_avg5min",
    "temperature_c_avg5min",
    "humidity_pct_avg5min",
    "light_lux_avg5min",
    "soil_pct_std5min",
    "soil_pct_trend_15min",
]
LABEL_COL = "label"  # Healthy / Water Stress / Light Stress / Nutrient Stress


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    return df


def main():
    df = load_data()

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    X = df[FEATURE_COLS]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ---- Validation ----
    val_preds = model.predict(X_val)
    print("=== Validation ===")
    print(classification_report(y_val, val_preds, target_names=le.classes_))

    # ---- Test ----
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)

    print("=== Test ===")
    print(f"Accuracy:  {accuracy_score(y_test, test_preds):.4f}")
    print(f"Precision: {precision_score(y_test, test_preds, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_test, test_preds, average='weighted'):.4f}")
    print(f"F1-score:  {f1_score(y_test, test_preds, average='weighted'):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    model.save_model(MODEL_OUT)
    joblib.dump(le, ENCODER_OUT)
    print(f"Saved model -> {MODEL_OUT}, label encoder -> {ENCODER_OUT}")


def predict_single(feature_dict):
    """Inference helper: feature_dict has the 6 FEATURE_COLS keys."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_OUT)
    le = joblib.load(ENCODER_OUT)

    X = pd.DataFrame([feature_dict])[FEATURE_COLS]
    probs = model.predict_proba(X)[0]
    return {cls: float(p) for cls, p in zip(le.classes_, probs)}


if __name__ == "__main__":
    main()
```

---

# PART 8 — VISION ML SCRIPT (YOLOv8n)

**Classification vs. detection:** use **YOLOv8n in classification mode** (`YOLO("yolov8n-cls.pt")`), not detection. Each image is a single, standardized, isolated photo of one plant against a matte background — there's nothing to localize (no bounding boxes, no multiple objects per frame). Detection would waste the model's capacity learning to draw boxes around something whose position is already fixed by the imaging booth, when the actual task is "assign one of 4 labels to this whole image." Classification mode outputs exactly the class probabilities the fusion stage needs.

**Input:** an image folder structured as `dataset/train/<class_name>/*.jpg`, `dataset/val/...`, `dataset/test/...` (built from `synced_dataset_labeled.csv` + `image_path`). **Output:** `vision_model.pt`. **Feeds into:** Part 9 (fusion) and Part 10 (real-time inference).

```python
# prepare_vision_dataset.py
# Input: synced_dataset_labeled.csv (image_path + label columns)
# Output: dataset/{train,val,test}/<class>/*.jpg (YOLO classification layout)

import pandas as pd
import shutil
import os
from sklearn.model_selection import train_test_split

DATA_PATH = "synced_dataset_labeled.csv"
OUTPUT_ROOT = "dataset"
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}


def main():
    df = pd.read_csv(DATA_PATH)

    train_df, temp_df = train_test_split(
        df, test_size=(SPLITS["val"] + SPLITS["test"]), stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=SPLITS["test"] / (SPLITS["val"] + SPLITS["test"]),
        stratify=temp_df["label"], random_state=42
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for _, row in split_df.iterrows():
            class_dir = os.path.join(OUTPUT_ROOT, split_name, row["label"])
            os.makedirs(class_dir, exist_ok=True)
            dest = os.path.join(class_dir, os.path.basename(row["image_path"]))
            shutil.copy(row["image_path"], dest)

    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")


if __name__ == "__main__":
    main()
```

```python
# train_vision_model.py

from ultralytics import YOLO

MODEL_OUT_DIR = "runs/classify/plant_health"


def train():
    model = YOLO("yolov8n-cls.pt")  # pretrained classification backbone
    model.train(
        data="dataset",       # expects dataset/train, dataset/val
        epochs=50,
        imgsz=224,
        batch=16,
        project="runs/classify",
        name="plant_health",
        patience=10,
    )
    return model


def validate(model):
    metrics = model.val(data="dataset", split="val")
    print(metrics)


def test(model):
    metrics = model.val(data="dataset", split="test")
    print("=== Test metrics ===")
    print(f"Top-1 accuracy: {metrics.top1}")
    print(f"Top-5 accuracy: {metrics.top5}")


if __name__ == "__main__":
    model = train()
    validate(model)
    test(model)
    # best weights auto-saved at runs/classify/plant_health/weights/best.pt
```

```python
# vision_inference.py
# Input: one image path. Output: dict of class -> probability.

from ultralytics import YOLO

VISION_MODEL_PATH = "runs/classify/plant_health/weights/best.pt"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(VISION_MODEL_PATH)
    return _model


def predict_image(image_path):
    model = get_model()
    results = model(image_path, verbose=False)
    probs = results[0].probs  # Probs object
    class_names = results[0].names

    return {class_names[i]: float(probs.data[i]) for i in range(len(class_names))}


def predict_new_camera_image(image_path):
    """Same as predict_image — kept as a named entry point since the
    real-time script (Part 10) calls this specifically on freshly
    captured images rather than dataset images."""
    return predict_image(image_path)


if __name__ == "__main__":
    import sys
    print(predict_image(sys.argv[1]))
```

---

# PART 9 — FUSION SCRIPT

**Input:** paired vision + sensor probability dicts (from Parts 7 & 8) for the training set, with ground-truth labels. **Output:** `fusion_weights.json` (for weighted averaging) and/or `fusion_meta_model.pkl` (logistic regression). **Feeds into:** Part 10 (real-time inference).

```python
# train_fusion.py
# Trains and compares both fusion strategies on held-out predictions.

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from itertools import product

CLASSES = ["Healthy", "Water Stress", "Light Stress", "Nutrient Stress"]

# Expects a CSV with columns:
# true_label, vision_Healthy, vision_Water Stress, vision_Light Stress, vision_Nutrient Stress,
#             sensor_Healthy, sensor_Water Stress, sensor_Light Stress, sensor_Nutrient Stress
FUSION_DATA_PATH = "fusion_training_data.csv"


def load_data():
    return pd.read_csv(FUSION_DATA_PATH)


def weighted_average_search(df):
    """Grid-search the vision/sensor weight that maximizes val accuracy."""
    best_w, best_acc = 0.5, 0.0
    y_true = df["true_label"]

    for w in np.arange(0.1, 1.0, 0.05):  # w = vision weight, (1-w) = sensor weight
        preds = []
        for _, row in df.iterrows():
            fused = {c: w * row[f"vision_{c}"] + (1 - w) * row[f"sensor_{c}"] for c in CLASSES}
            preds.append(max(fused, key=fused.get))
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc, best_w = acc, w

    return best_w, best_acc


def train_logistic_meta(df):
    """Meta-classifier: input is the 8 concatenated probabilities, output is final class."""
    X = df[[f"vision_{c}" for c in CLASSES] + [f"sensor_{c}" for c in CLASSES]]
    y = df["true_label"]

    meta = LogisticRegression(max_iter=1000, multi_class="multinomial")
    meta.fit(X, y)

    preds = meta.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="weighted")
    return meta, acc, f1


def main():
    df = load_data()

    w, wa_acc = weighted_average_search(df)
    print(f"Weighted averaging: vision_weight={w:.2f}, accuracy={wa_acc:.4f}")
    with open("fusion_weights.json", "w") as f:
        json.dump({"vision_weight": w, "sensor_weight": 1 - w}, f)

    meta, lr_acc, lr_f1 = train_logistic_meta(df)
    print(f"Logistic regression meta-classifier: accuracy={lr_acc:.4f}, f1={lr_f1:.4f}")
    joblib.dump(meta, "fusion_meta_model.pkl")

    print("\nUse whichever scored higher on a held-out split for production inference.")


def fuse_weighted(vision_probs, sensor_probs, vision_weight):
    return {
        c: vision_weight * vision_probs[c] + (1 - vision_weight) * sensor_probs[c]
        for c in CLASSES
    }


def fuse_logistic(vision_probs, sensor_probs, meta_model):
    X = [[vision_probs[c] for c in CLASSES] + [sensor_probs[c] for c in CLASSES]]
    probs = meta_model.predict_proba(X)[0]
    return {cls: float(p) for cls, p in zip(meta_model.classes_, probs)}


if __name__ == "__main__":
    main()
```

---

# PART 10 — REAL-TIME INFERENCE SCRIPT

**Input:** live serial stream (ESP32) + on-demand camera capture. **Output:** printed/logged final classification per plant. **Uses:** `sensor_model.json` (Part 7), vision `best.pt` (Part 8), `fusion_weights.json` or `fusion_meta_model.pkl` (Part 9).

```python
# realtime_inference.py

import serial
import time
import json
import joblib
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from collections import deque

from vision_inference import predict_new_camera_image
from capture_plant_images import capture_image

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

SENSOR_MODEL_PATH = "sensor_model.json"
LABEL_ENCODER_PATH = "sensor_label_encoder.pkl"
FUSION_WEIGHTS_PATH = "fusion_weights.json"

CLASSES = ["Healthy", "Water Stress", "Light Stress", "Nutrient Stress"]
FEATURE_COLS = [
    "soil_pct_avg5min", "temperature_c_avg5min", "humidity_pct_avg5min",
    "light_lux_avg5min", "soil_pct_std5min", "soil_pct_trend_15min",
]

# rolling buffer of recent readings per plant, for the 5min/15min windows
HISTORY_WINDOW_MINUTES = 15
plant_history = {f"P{str(i).zfill(2)}": deque() for i in range(1, 17)}

sensor_model = xgb.XGBClassifier()
sensor_model.load_model(SENSOR_MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
fusion_weights = json.load(open(FUSION_WEIGHTS_PATH))


def parse_serial_line(line):
    parts = line.strip().split(",")
    if len(parts) != 8:
        return None
    ts, plant_id, soil_raw, soil_pct, temp, hum, light, status = parts
    if status != "OK":
        return None
    try:
        return {
            "timestamp": datetime.now(),
            "plant_id": plant_id,
            "soil_pct": float(soil_pct),
            "temperature_c": float(temp),
            "humidity_pct": float(hum),
            "light_lux": float(light),
        }
    except ValueError:
        return None


def update_history(reading):
    plant_id = reading["plant_id"]
    plant_history[plant_id].append(reading)
    cutoff = datetime.now() - timedelta(minutes=HISTORY_WINDOW_MINUTES)
    while plant_history[plant_id] and plant_history[plant_id][0]["timestamp"] < cutoff:
        plant_history[plant_id].popleft()


def compute_features(plant_id):
    history = list(plant_history[plant_id])
    if not history:
        return None

    avg_cutoff = datetime.now() - timedelta(minutes=5)
    avg_window = [r for r in history if r["timestamp"] >= avg_cutoff]
    if not avg_window:
        return None

    soil_vals = [r["soil_pct"] for r in avg_window]
    features = {
        "soil_pct_avg5min": float(np.mean(soil_vals)),
        "temperature_c_avg5min": float(np.mean([r["temperature_c"] for r in avg_window])),
        "humidity_pct_avg5min": float(np.mean([r["humidity_pct"] for r in avg_window])),
        "light_lux_avg5min": float(np.mean([r["light_lux"] for r in avg_window])),
        "soil_pct_std5min": float(np.std(soil_vals)),
    }

    if len(history) >= 2:
        t0 = history[0]["timestamp"]
        minutes = [(r["timestamp"] - t0).total_seconds() / 60.0 for r in history]
        soil_series = [r["soil_pct"] for r in history]
        slope = np.polyfit(minutes, soil_series, 1)[0]
        features["soil_pct_trend_15min"] = float(slope)
    else:
        features["soil_pct_trend_15min"] = 0.0

    return features


def sensor_predict(features):
    import pandas as pd
    X = pd.DataFrame([features])[FEATURE_COLS]
    probs = sensor_model.predict_proba(X)[0]
    return {cls: float(p) for cls, p in zip(label_encoder.classes_, probs)}


def fuse(vision_probs, sensor_probs):
    w = fusion_weights["vision_weight"]
    return {c: w * vision_probs[c] + (1 - w) * sensor_probs[c] for c in CLASSES}


def run_full_prediction(plant_id):
    features = compute_features(plant_id)
    if features is None:
        print(f"{plant_id}: not enough sensor history yet")
        return None

    sensor_probs = sensor_predict(features)
    image_path, _ = capture_image(plant_id)
    vision_probs = predict_new_camera_image(image_path)
    final_probs = fuse(vision_probs, sensor_probs)

    final_class = max(final_probs, key=final_probs.get)
    confidence = final_probs[final_class]

    result = {
        "plant_id": plant_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "soil_moisture": round(features["soil_pct_avg5min"], 1),
        "temperature": round(features["temperature_c_avg5min"], 1),
        "humidity": round(features["humidity_pct_avg5min"], 1),
        "light": round(features["light_lux_avg5min"], 1),
        "sensor_prediction": max(sensor_probs, key=sensor_probs.get),
        "sensor_confidence": round(max(sensor_probs.values()) * 100, 1),
        "vision_prediction": max(vision_probs, key=vision_probs.get),
        "vision_confidence": round(max(vision_probs.values()) * 100, 1),
        "final_prediction": final_class,
        "final_confidence": round(confidence * 100, 1),
    }
    print_result(result)
    return result


def print_result(r):
    print(f"\nPlant: {r['plant_id']}")
    print(f"Soil Moisture: {r['soil_moisture']}%")
    print(f"Temperature: {r['temperature']}°C")
    print(f"Humidity: {r['humidity']}%")
    print(f"Light: {r['light']} lux")
    print(f"\nSensor Prediction:\n{r['sensor_prediction']} — {r['sensor_confidence']}%")
    print(f"\nVision Prediction:\n{r['vision_prediction']} — {r['vision_confidence']}%")
    print(f"\nFinal Prediction:\n{r['final_prediction']} — {r['final_confidence']}%")
    print(f"\nTimestamp:\n{r['timestamp']}")


def stream_sensor_loop(ser):
    """Continuously ingest serial data and keep history buffers current."""
    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore")
        reading = parse_serial_line(line)
        if reading:
            update_history(reading)
        yield reading


if __name__ == "__main__":
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    # Background-style usage: pull sensor lines continuously; call
    # run_full_prediction(plant_id) whenever you want a full classification
    # (e.g. triggered by Part 12's scheduler).
    for reading in stream_sensor_loop(ser):
        pass  # Part 12 drives this loop and calls run_full_prediction on schedule
```

---

# PART 11 — PLANT PROFILING

**Input:** the `result` dict produced by `run_full_prediction()` in Part 10, called once per plant per cycle. **Output:** `plant_profiles.json`, continuously updated. **Feeds into:** Part 12 (final script logs/reads from this file).

```python
# plant_profile.py

import json
import os
from datetime import datetime
from collections import deque

PROFILE_PATH = "plant_profiles.json"
SMOOTHING_WINDOW = 3  # number of recent final predictions considered for the "status"

_profiles = {}
_recent_predictions = {}  # plant_id -> deque of (class, confidence)


def load_profiles():
    global _profiles
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH) as f:
            _profiles = json.load(f)
    return _profiles


def save_profiles():
    with open(PROFILE_PATH, "w") as f:
        json.dump(_profiles, f, indent=2)


def _get_history_deque(plant_id):
    if plant_id not in _recent_predictions:
        _recent_predictions[plant_id] = deque(maxlen=SMOOTHING_WINDOW)
    return _recent_predictions[plant_id]


def smoothed_status(plant_id, new_class, new_confidence):
    """Majority vote over the last SMOOTHING_WINDOW final predictions.
    A single noisy prediction can't flip the plant's displayed status —
    it needs the majority of the recent window to agree."""
    history = _get_history_deque(plant_id)
    history.append((new_class, new_confidence))

    classes = [c for c, _ in history]
    majority_class = max(set(classes), key=classes.count)
    matching_confidences = [conf for c, conf in history if c == majority_class]
    avg_confidence = sum(matching_confidences) / len(matching_confidences)

    return majority_class, avg_confidence


def update_profile(prediction_result):
    plant_id = prediction_result["plant_id"]
    profile = _profiles.get(plant_id, {
        "plant_id": plant_id,
        "current": None,
        "previous": None,
        "latest_image": None,
        "history": [],
    })

    smoothed_class, smoothed_conf = smoothed_status(
        plant_id, prediction_result["final_prediction"], prediction_result["final_confidence"]
    )

    # soil moisture trend direction from the last two recorded readings
    trend = "unknown"
    if profile["current"] is not None:
        prev_moisture = profile["current"].get("soil_moisture")
        curr_moisture = prediction_result["soil_moisture"]
        if prev_moisture is not None:
            if curr_moisture < prev_moisture - 1:
                trend = "decreasing"
            elif curr_moisture > prev_moisture + 1:
                trend = "increasing"
            else:
                trend = "stable"

    profile["previous"] = profile["current"]
    profile["current"] = {
        "status": smoothed_class,
        "confidence": round(smoothed_conf, 1),
        "raw_final_prediction": prediction_result["final_prediction"],
        "raw_final_confidence": prediction_result["final_confidence"],
        "sensor_prediction": prediction_result["sensor_prediction"],
        "vision_prediction": prediction_result["vision_prediction"],
        "soil_moisture": prediction_result["soil_moisture"],
        "soil_moisture_trend": trend,
        "timestamp": prediction_result["timestamp"],
    }
    profile["latest_image"] = prediction_result.get("image_path")
    profile["history"].append(profile["current"])
    profile["history"] = profile["history"][-100:]  # cap growth

    _profiles[plant_id] = profile
    save_profiles()
    return profile


if __name__ == "__main__":
    load_profiles()
```

---

# PART 12 — FINAL REAL-TIME SCRIPT

**Input:** none (orchestrates everything above). **Output:** continuous console log + `plant_profiles.json`. This is the single entry point.

```python
# run_plant_monitor.py
# One command, full loop: sensor ingestion -> features -> XGBoost ->
# scheduled imaging -> YOLO -> fusion -> profile update -> logging.

import serial
import time
import threading
from datetime import datetime

from realtime_inference import (
    SERIAL_PORT, BAUD_RATE, parse_serial_line, update_history,
    run_full_prediction,
)
from plant_profile import load_profiles, update_profile

PLANT_IDS = [f"P{str(i).zfill(2)}" for i in range(1, 17)]
CLASSIFICATION_INTERVAL_SECONDS = 900  # run full vision+sensor classification every 15 min per plant


def sensor_ingest_thread(ser):
    """Keeps rolling sensor history current in the background at full
    ESP32 sweep rate, independent of the slower classification cycle."""
    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore")
        reading = parse_serial_line(line)
        if reading:
            update_history(reading)


def classification_cycle():
    """Runs one full sensor+vision+fusion+profile pass over all 16 plants."""
    for plant_id in PLANT_IDS:
        result = run_full_prediction(plant_id)
        if result is None:
            continue
        profile = update_profile(result)
        log_line = (
            f"[{result['timestamp']}] {plant_id} | "
            f"sensor={result['sensor_prediction']}({result['sensor_confidence']}%) | "
            f"vision={result['vision_prediction']}({result['vision_confidence']}%) | "
            f"final={result['final_prediction']}({result['final_confidence']}%) | "
            f"smoothed_status={profile['current']['status']}"
        )
        print(log_line)


def main():
    load_profiles()

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    ingest = threading.Thread(target=sensor_ingest_thread, args=(ser,), daemon=True)
    ingest.start()

    print("Sensor ingestion started. Waiting for history to build before first classification...")
    time.sleep(60)  # let at least one 5-min-ish buffer start accumulating

    while True:
        cycle_start = time.time()
        classification_cycle()
        elapsed = time.time() - cycle_start
        sleep_time = max(0, CLASSIFICATION_INTERVAL_SECONDS - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
```

---

## Script → script data contracts (quick reference)

| Script | Reads | Writes |
|---|---|---|
| `esp32_sensor_node.ino` | live sensors | serial CSV, 8 fields, `status` last |
| `jetson_data_logger.py` | serial CSV | `sensor_log.db` → `raw_readings` |
| `clean_sensor_data.py` | `raw_readings` | `cleaned_readings.csv`, `flagged_readings.csv` |
| `capture_plant_images.py` | camera | `images/<id>/*.jpg`, `image_index.csv` |
| `sync_image_sensor.py` | `image_index.csv`, `cleaned_readings.csv` | `synced_dataset.csv` |
| `train_sensor_model.py` | `synced_dataset_labeled.csv` | `sensor_model.json`, `sensor_label_encoder.pkl` |
| `prepare_vision_dataset.py` → `train_vision_model.py` | `synced_dataset_labeled.csv` | `dataset/`, `runs/classify/plant_health/weights/best.pt` |
| `train_fusion.py` | `fusion_training_data.csv` | `fusion_weights.json`, `fusion_meta_model.pkl` |
| `realtime_inference.py` | serial + camera + all trained models | prediction dict |
| `plant_profile.py` | prediction dict | `plant_profiles.json` |
| `run_plant_monitor.py` | all of the above | console log + `plant_profiles.json` |
