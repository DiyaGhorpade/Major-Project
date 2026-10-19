import csv
import os
from datetime import datetime

import serial


# ============================================================
# RECEIVER CONFIGURATION
# ============================================================

# This will be changed to the actual Wokwi serial connection
# in the next step.
SERIAL_PORT = "COM3"

BAUD_RATE = 115200

SENSOR_DATA_FILE = "sensor_data.csv"
ERROR_LOG_FILE = "error_log.csv"

# ============================================================
# EXPECTED INPUT STRUCTURE
# ============================================================

EXPECTED_FIELDS = 8

INPUT_HEADER = [
    "timestamp",
    "session_id",
    "sampling_point",
    "plant_id",
    "soil",
    "temperature",
    "humidity",
    "light"
]


# ============================================================
# OUTPUT STRUCTURES
# ============================================================

OUTPUT_HEADER = [
    "record_id",
    "timestamp",
    "session_id",
    "sampling_point",
    "plant_id",
    "soil",
    "temperature",
    "humidity",
    "light"
]

ERROR_HEADER = [
    "record_id",
    "plant_id",
    "sensor",
    "bad_value",
    "reason"
]


# ============================================================
# RECORD ID
# ============================================================

record_counter = 0


def generate_record_id():
    global record_counter

    record_counter += 1

    return f"R{record_counter:06d}"


# ============================================================
# FILE INITIALIZATION
# ============================================================

def initialize_files():

    if not os.path.exists(SENSOR_DATA_FILE):

        with open(
            SENSOR_DATA_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)
            writer.writerow(OUTPUT_HEADER)


    if not os.path.exists(ERROR_LOG_FILE):

        with open(
            ERROR_LOG_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)
            writer.writerow(ERROR_HEADER)


# ============================================================
# ERROR LOGGING
# ============================================================

def log_error(
    record_id,
    plant_id,
    sensor,
    bad_value,
    reason
):

    with open(
        ERROR_LOG_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            record_id,
            plant_id,
            sensor,
            bad_value,
            reason
        ])


# ============================================================
# VALIDATION
# ============================================================

def validate_record(fields, record_id):

    errors = []


    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    if len(fields) != EXPECTED_FIELDS:

        log_error(
            record_id,
            "",
            "record",
            ",".join(fields),
            f"Expected {EXPECTED_FIELDS} fields, received {len(fields)}"
        )

        return None


    # --------------------------------------------------------
    # Extract fields
    # --------------------------------------------------------

    timestamp = fields[0]
    session_id_raw = fields[1]
    sampling_point_raw = fields[2]
    plant_id_raw = fields[3]
    soil_raw = fields[4]
    temperature_raw = fields[5]
    humidity_raw = fields[6]
    light_raw = fields[7]


    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    try:
        session_id = int(session_id_raw)
    except ValueError:

        log_error(
            record_id,
            plant_id_raw,
            "session_id",
            session_id_raw,
            "Expected an integer"
        )

        return None


    try:
        sampling_point = int(sampling_point_raw)
    except ValueError:

        log_error(
            record_id,
            plant_id_raw,
            "sampling_point",
            sampling_point_raw,
            "Expected an integer from 1 to 6"
        )

        return None


    try:
        plant_id = int(plant_id_raw)
    except ValueError:

        log_error(
            record_id,
            plant_id_raw,
            "plant_id",
            plant_id_raw,
            "Expected an integer from 1 to 16"
        )

        return None


    try:
        soil = float(soil_raw)
    except ValueError:

        log_error(
            record_id,
            plant_id_raw,
            "soil",
            soil_raw,
            "Expected a numeric value"
        )

        return None


    try:
        temperature = float(temperature_raw)
    except ValueError:

        log_error(
            record_id,
            plant_id_raw,
            "temperature",
            temperature_raw,
            "Expected a numeric value"
        )

        return None


    try:
        humidity = float(humidity_raw)
    except ValueError:

        log_error(
            record_id,
            plant_id_raw,
            "humidity",
            humidity_raw,
            "Expected a numeric value"
        )

        return None


    try:
        light = float(light_raw)
    except ValueError:

        log_error(
            record_id,
            plant_id_raw,
            "light",
            light_raw,
            "Expected a numeric value"
        )

        return None


    # --------------------------------------------------------
    # Session ID
    # --------------------------------------------------------

    if session_id < 1:

        log_error(
            record_id,
            plant_id,
            "session_id",
            session_id,
            "Expected session_id >= 1"
        )

        return None


    # --------------------------------------------------------
    # Sampling point
    # --------------------------------------------------------

    if sampling_point < 1 or sampling_point > 6:

        log_error(
            record_id,
            plant_id,
            "sampling_point",
            sampling_point,
            "Expected sampling point 1-6"
        )

        return None


    # --------------------------------------------------------
    # Plant ID
    # --------------------------------------------------------

    if plant_id < 1 or plant_id > 16:

        log_error(
            record_id,
            plant_id,
            "plant_id",
            plant_id,
            "Expected plant_id 1-16"
        )

        return None


    # --------------------------------------------------------
    # Soil moisture
    # --------------------------------------------------------

    if soil < 0 or soil > 100:

        log_error(
            record_id,
            plant_id,
            "soil",
            soil,
            "Expected 0-100%"
        )

        return None


    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if temperature < -40 or temperature > 80:

        log_error(
            record_id,
            plant_id,
            "temperature",
            temperature,
            "Expected -40 to 80 C"
        )

        return None


    # --------------------------------------------------------
    # Humidity
    # --------------------------------------------------------

    if humidity < 0 or humidity > 100:

        log_error(
            record_id,
            plant_id,
            "humidity",
            humidity,
            "Expected 0-100%"
        )

        return None


    # --------------------------------------------------------
    # Light
    # --------------------------------------------------------

    if light < 0 or light > 1000:

        log_error(
            record_id,
            plant_id,
            "light",
            light,
            "Expected 0-1000 lux"
        )

        return None


    # --------------------------------------------------------
    # Valid record
    # --------------------------------------------------------

    return [
        record_id,
        timestamp,
        session_id,
        sampling_point,
        plant_id,
        soil,
        temperature,
        humidity,
        light
    ]


# ============================================================
# SAVE VALID RECORD
# ============================================================

def save_valid_record(record):

    with open(
        SENSOR_DATA_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)
        writer.writerow(record)


# ============================================================
# PROCESS ONE SERIAL LINE
# ============================================================

def process_line(line):

    line = line.strip()

    if not line:
        return


    # --------------------------------------------------------
    # Ignore ESP32 CSV header
    # --------------------------------------------------------

    if line == ",".join(INPUT_HEADER):
        return


    # --------------------------------------------------------
    # Generate record ID before validation
    # --------------------------------------------------------

    record_id = generate_record_id()


    # --------------------------------------------------------
    # Parse CSV
    # --------------------------------------------------------

    try:

        fields = next(
            csv.reader([line])
        )

    except Exception as error:

        log_error(
            record_id,
            "",
            "record",
            line,
            f"CSV parsing error: {error}"
        )

        print(
            f"[INVALID] {record_id} - CSV parsing error"
        )

        return


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    record = validate_record(
        fields,
        record_id
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    if record is not None:

        save_valid_record(record)

        print(
            f"[VALID] "
            f"{record_id} | "
            f"Session {record[2]} | "
            f"Sampling Point {record[3]} | "
            f"Plant {record[4]}"
        )

    else:

        print(
            f"[INVALID] {record_id}"
        )


# ============================================================
# MAIN RECEIVER
# ============================================================

def main():

    initialize_files()

    print("----------------------------------------")
    print("Sensor Data Receiver")
    print("----------------------------------------")
    print(f"Serial port : {SERIAL_PORT}")
    print(f"Baud rate   : {BAUD_RATE}")
    print(f"Output      : {SENSOR_DATA_FILE}")
    print(f"Errors      : {ERROR_LOG_FILE}")
    print("----------------------------------------")
    print("Waiting for sensor data...")
    print()


    try:

        with serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=1
        ) as serial_connection:

            while True:

                line = (
                    serial_connection
                    .readline()
                    .decode(
                        "utf-8",
                        errors="replace"
                    )
                )

                if line:

                    process_line(line)


    except KeyboardInterrupt:

        print()
        print("Receiver stopped.")


    except serial.SerialException as error:

        print()
        print("Serial connection error:")
        print(error)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()