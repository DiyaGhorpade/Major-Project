import csv
import os

import config


# ============================================================
# FILE INITIALIZATION
# ============================================================

def initialize_files() -> None:
    """
    Create sensor_data.csv and error_log.csv with their headers
    if they do not already exist.

    Both absent  → create both with headers (normal startup).
    Both present → leave both unchanged (no duplicate headers).
    Exactly one missing → raise OSError without creating anything
                          (all-or-nothing invariant).
    """
    data_exists = os.path.exists(config.SENSOR_DATA_FILE)
    error_exists = os.path.exists(config.ERROR_LOG_FILE)

    if data_exists and error_exists:
        # Both present — nothing to do.
        return

    if data_exists != error_exists:
        # Exactly one file is missing — partial state is not allowed.
        missing = (
            config.ERROR_LOG_FILE if data_exists else config.SENSOR_DATA_FILE
        )
        present = (
            config.SENSOR_DATA_FILE if data_exists else config.ERROR_LOG_FILE
        )
        raise OSError(
            f"Partial output state detected: '{present}' exists but "
            f"'{missing}' is missing. Remove or restore both files and "
            f"restart the receiver."
        )

    # Both absent — create both with their respective headers.
    with open(config.SENSOR_DATA_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(config.OUTPUT_HEADER)

    with open(config.ERROR_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(config.ERROR_HEADER)


# ============================================================
# SAVE VALID RECORD
# ============================================================

def save_valid_record(record: dict) -> None:
    """
    Append one valid record to sensor_data.csv.

    Fields are written in OUTPUT_HEADER order:
        record_id, timestamp, session_id, sampling_point,
        plant_id, soil, temperature, humidity, light
    """
    row = [record[field] for field in config.OUTPUT_HEADER]

    with open(config.SENSOR_DATA_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ============================================================
# LOG ERROR
# ============================================================

def log_error(
    record_id: str,
    plant_id: str,
    sensor: str,
    bad_value: str,
    reason: str,
    node_id: str = "",
) -> None:
    """
    Append one error row to error_log.csv.

    Fields are written in ERROR_HEADER order:
        record_id, plant_id, sensor, bad_value, reason, node_id
    """
    with open(config.ERROR_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([record_id, plant_id, sensor, bad_value, reason, node_id])
