"""
storage.py — File initialisation and thread-safe CSV writing.

All writes go through a single module-level Lock so that concurrent
HTTP requests (e.g. a burst of 16 from the ESP32) never interleave
partial rows into the output file.
"""

import csv
import os
import threading

import config

# One lock shared by all write operations in this module.
_write_lock = threading.Lock()


# ============================================================
# FILE INITIALIZATION
# ============================================================

def initialize_files() -> None:
    """
    Create sensor_data.csv with its header if it does not already exist.

    Called once at server startup before any requests are accepted.
    Uses the lock so a hypothetical race at startup cannot produce a
    double-header (safe even if called from multiple threads).
    """
    with _write_lock:
        if not os.path.exists(config.SENSOR_DATA_FILE):
            with open(config.SENSOR_DATA_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(config.CSV_HEADER)
            print(f"[INIT] Created {config.SENSOR_DATA_FILE} with header.")
        else:
            print(f"[INIT] {config.SENSOR_DATA_FILE} already exists — appending.")


# ============================================================
# SAVE VALID RECORD
# ============================================================

def save_valid_record(fields: list) -> None:
    """
    Append one validated row to sensor_data.csv.

    Parameters
    ----------
    fields : list
        The 8 values in CSV_HEADER order, already validated.

    The file is flushed and synced to disk immediately after the write
    so no data is lost if the process is killed between rows.
    The lock prevents two simultaneous requests from interleaving their
    writes.
    """
    with _write_lock:
        with open(config.SENSOR_DATA_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            # Flush Python's internal buffer then ask the OS to sync
            # the data to disk — critical for crash safety.
            f.flush()
            os.fsync(f.fileno())
