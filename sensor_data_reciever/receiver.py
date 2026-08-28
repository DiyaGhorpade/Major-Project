"""
receiver.py — Orchestration layer for the sensor data receiver.

Owns record_counter state and drives the processing pipeline.
"""

import csv

import serial

import config
import storage
import validator
from validator import ValidationError

record_counter: int = 0


def generate_record_id() -> str:
    """
    Increment record_counter and return a formatted record ID.

    Format: R{counter:06d}  →  "R000001", "R000002", …
    Called exactly once per non-empty, non-header line.
    """
    global record_counter
    record_counter += 1
    return f"R{record_counter:06d}"


def process_line(line: str) -> None:
    """
    Full pipeline for one serial line:
      1. strip()
      2. skip if empty (no ID consumed)
      3. skip if line == INPUT_HEADER joined by comma (no ID consumed)
      4. generate_record_id()
      5. csv.reader parse → fields list
         on parse error: log_error + print [INVALID] + return
      6. validate_record(fields)
      7a. on dict: add record_id, save_valid_record, print [VALID]
      7b. on list[ValidationError]: log each error, print [INVALID]
    """
    # Step 1 — strip whitespace
    line = line.strip()

    # Step 2 — skip empty lines (no ID consumed)
    if not line:
        return

    # Step 3 — skip header line (no ID consumed)
    if line == ",".join(config.INPUT_HEADER):
        return

    # Step 4 — consume one record ID
    record_id = generate_record_id()

    # Step 5 — CSV parse
    try:
        fields = next(csv.reader([line]))
    except Exception:
        storage.log_error(record_id, "", "record", line, "CSV parsing error")
        print(f"[INVALID] {record_id} - CSV parsing error")
        return

    # Step 6 — validate
    result = validator.validate_record(fields)

    # Step 7a — valid record
    if isinstance(result, dict):
        result["record_id"] = record_id
        storage.save_valid_record(result)
        print(
            f"[VALID] {record_id} | "
            f"Session {result['session_id']} | "
            f"Sampling Point {result['sampling_point']} | "
            f"Plant {result['plant_id']}"
        )

    # Step 7b — validation errors
    else:
        for error in result:
            storage.log_error(
                record_id,
                error.plant_id,
                error.sensor,
                error.bad_value,
                error.reason,
            )
        print(f"[INVALID] {record_id}")


def main() -> None:
    """
    Entry point for the sensor data receiver.

    1. Initialize output files.
    2. Print startup banner.
    3. Open serial connection and loop, passing each decoded line to process_line.
    4. Handle KeyboardInterrupt and serial.SerialException cleanly.
    """
    # Step 1 — initialize files before touching the serial port
    storage.initialize_files()

    # Step 2 — startup banner (printed before the port is opened)
    print("----------------------------------------")
    print("Sensor Data Receiver")
    print("----------------------------------------")
    print(f"Serial port : {config.SERIAL_PORT}")
    print(f"Baud rate   : {config.BAUD_RATE}")
    print(f"Output      : {config.SENSOR_DATA_FILE}")
    print(f"Errors      : {config.ERROR_LOG_FILE}")
    print("----------------------------------------")
    print("Waiting for sensor data...")
    print()

    # Step 3 — open serial and read loop
    try:
        with serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=1) as serial_connection:
            while True:
                line = serial_connection.readline().decode("utf-8", errors="replace")
                if line:
                    process_line(line)

    # Step 4a — graceful shutdown on Ctrl-C
    except KeyboardInterrupt:
        print()
        print("Receiver stopped.")

    # Step 4b — serial port error
    except serial.SerialException as error:
        print()
        print("Serial connection error:")
        print(error)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()
