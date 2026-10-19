"""
Integration test: feed every line from sample.csv through process_line and
verify the output files satisfy the gap-free record_id sequence invariant.

Validates: Requirements 8.6
"""
import csv
import os
import pathlib

import pytest

import config
import receiver
import storage

# Absolute path to the real sample.csv at the project root
SAMPLE_CSV = pathlib.Path(__file__).parent.parent / "sample.csv"

# The exact INPUT_HEADER string that process_line skips (no ID consumed)
INPUT_HEADER_LINE = ",".join(config.INPUT_HEADER)


# ---------------------------------------------------------------------------
# Helper: read all non-header rows from a CSV file
# ---------------------------------------------------------------------------

def _data_rows(path: str) -> list[list[str]]:
    """Return all rows after the first (header) row from a CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[1:]


def _all_rows(path: str) -> list[list[str]]:
    """Return every row including the header."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


# ---------------------------------------------------------------------------
# Fixture: isolated temp files + monkeypatched config paths
# ---------------------------------------------------------------------------

@pytest.fixture()
def integration_files(tmp_path, monkeypatch):
    """
    Redirect config file paths to tmp_path, initialise the output files, and
    reset receiver.record_counter to 0 before each test.
    """
    data_file = str(tmp_path / "sensor_data.csv")
    error_file = str(tmp_path / "error_log.csv")

    monkeypatch.setattr(config, "SENSOR_DATA_FILE", data_file)
    monkeypatch.setattr(config, "ERROR_LOG_FILE", error_file)

    # Initialize files (creates headers)
    storage.initialize_files()

    # Reset counter so IDs start at R000001
    receiver.record_counter = 0

    return {"data": data_file, "error": error_file}


# ---------------------------------------------------------------------------
# Pre-compute expected behaviour from sample.csv (at import time)
# ---------------------------------------------------------------------------

def _classify_lines(path: pathlib.Path) -> tuple[list[str], list[str]]:
    """
    Read sample.csv and split lines into:
      - processed_lines: non-empty lines that are NOT the INPUT_HEADER
      - valid_lines / invalid_lines: determined by running validate_record

    Returns (valid_lines, invalid_lines) where each element is the raw line.
    """
    import validator

    valid_lines: list[str] = []
    invalid_lines: list[str] = []

    with open(path, newline="", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            # Skip empty lines (process_line skips these, no ID consumed)
            if not line:
                continue
            # Skip INPUT_HEADER (process_line skips these, no ID consumed)
            if line == INPUT_HEADER_LINE:
                continue
            # This line will consume an ID; determine if valid or invalid
            fields = next(csv.reader([line]))
            result = validator.validate_record(fields)
            if isinstance(result, dict):
                valid_lines.append(line)
            else:
                invalid_lines.append(line)

    return valid_lines, invalid_lines


VALID_LINES, INVALID_LINES = _classify_lines(SAMPLE_CSV)
TOTAL_PROCESSED = len(VALID_LINES) + len(INVALID_LINES)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestSampleCsvIntegration:
    """Feed every line of sample.csv through process_line and verify outputs."""

    def _feed_sample(self):
        """Open sample.csv and pass every line through process_line."""
        with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
            for raw_line in f:
                receiver.process_line(raw_line)

    # --- helpers ----------------------------------------------------------------

    def _record_ids_in_data(self, data_path: str) -> list[str]:
        rows = _data_rows(data_path)
        # record_id is the first column (OUTPUT_HEADER order)
        return [row[0] for row in rows]

    def _record_ids_in_errors(self, error_path: str) -> list[str]:
        rows = _data_rows(error_path)
        # record_id is the first column (ERROR_HEADER order)
        return [row[0] for row in rows]

    # --- tests ------------------------------------------------------------------

    def test_valid_lines_appear_in_sensor_data(self, integration_files):
        """
        Every structurally/range-valid input line must produce exactly one row
        in sensor_data.csv (excluding the header).
        """
        self._feed_sample()
        data_rows = _data_rows(integration_files["data"])
        assert len(data_rows) == len(VALID_LINES), (
            f"Expected {len(VALID_LINES)} rows in sensor_data.csv, "
            f"got {len(data_rows)}"
        )

    def test_no_invalid_line_data_in_sensor_data(self, integration_files):
        """
        No data from an invalid input line must appear in sensor_data.csv.
        The number of data rows must equal the number of valid input lines.
        (Verified structurally: wrong-field-count rows can't produce valid records.)
        """
        self._feed_sample()
        data_rows = _data_rows(integration_files["data"])
        # All rows present must correspond to valid input lines
        assert len(data_rows) == len(VALID_LINES), (
            f"sensor_data.csv contains {len(data_rows)} rows but only "
            f"{len(VALID_LINES)} input lines were valid"
        )

    def test_each_invalid_line_produces_one_error_row(self, integration_files):
        """
        Every invalid input line must produce exactly one row in error_log.csv.
        """
        self._feed_sample()
        error_rows = _data_rows(integration_files["error"])
        assert len(error_rows) == len(INVALID_LINES), (
            f"Expected {len(INVALID_LINES)} rows in error_log.csv, "
            f"got {len(error_rows)}"
        )

    def test_record_ids_form_gap_free_sequence(self, integration_files):
        """
        The record_id values across both output files together must form the
        exact gap-free sequence R000001 … R{TOTAL_PROCESSED:06d}.
        """
        self._feed_sample()

        ids_in_data   = self._record_ids_in_data(integration_files["data"])
        ids_in_errors = self._record_ids_in_errors(integration_files["error"])

        all_ids = sorted(ids_in_data + ids_in_errors)

        expected_ids = [f"R{n:06d}" for n in range(1, TOTAL_PROCESSED + 1)]

        assert all_ids == expected_ids, (
            f"Record IDs are not a gap-free sequence.\n"
            f"  Expected: {expected_ids[:5]}…{expected_ids[-5:]}\n"
            f"  Got:      {all_ids[:5]}…{all_ids[-5:]}"
        )

    def test_record_ids_start_at_r000001(self, integration_files):
        """The first record_id assigned must be R000001."""
        self._feed_sample()

        ids_in_data   = self._record_ids_in_data(integration_files["data"])
        ids_in_errors = self._record_ids_in_errors(integration_files["error"])

        all_ids = sorted(ids_in_data + ids_in_errors)

        assert len(all_ids) > 0, "No record IDs were produced at all"
        assert all_ids[0] == "R000001", (
            f"First record_id should be R000001, got {all_ids[0]}"
        )

    def test_total_processed_count(self, integration_files):
        """
        The total number of record_ids produced must equal the number of
        non-empty, non-INPUT_HEADER lines in sample.csv.
        """
        self._feed_sample()

        ids_in_data   = self._record_ids_in_data(integration_files["data"])
        ids_in_errors = self._record_ids_in_errors(integration_files["error"])

        total_ids = len(ids_in_data) + len(ids_in_errors)

        assert total_ids == TOTAL_PROCESSED, (
            f"Expected {TOTAL_PROCESSED} total record IDs, got {total_ids}"
        )

    def test_sensor_data_header_is_preserved(self, integration_files):
        """sensor_data.csv must have exactly OUTPUT_HEADER as its first row."""
        self._feed_sample()
        all_rows = _all_rows(integration_files["data"])
        assert len(all_rows) >= 1, "sensor_data.csv is empty"
        assert all_rows[0] == config.OUTPUT_HEADER, (
            f"sensor_data.csv header mismatch: {all_rows[0]}"
        )

    def test_error_log_header_is_preserved(self, integration_files):
        """error_log.csv must have exactly ERROR_HEADER as its first row."""
        self._feed_sample()
        all_rows = _all_rows(integration_files["error"])
        assert len(all_rows) >= 1, "error_log.csv is empty"
        assert all_rows[0] == config.ERROR_HEADER, (
            f"error_log.csv header mismatch: {all_rows[0]}"
        )
