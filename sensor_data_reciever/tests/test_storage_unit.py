"""
Unit tests for storage.save_valid_record, storage.log_error, and
storage.initialize_files.
Validates: Requirements 8.2, 8.3
"""
import csv
import os
import pytest
import config
import storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_data_rows(path: str) -> list[list[str]]:
    """Return all non-header rows from a CSV file as lists of strings."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows[1:]


def _read_error_rows(path: str) -> list[list[str]]:
    """Return all non-header rows from error_log CSV as lists of strings."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows[1:]


def _read_all_rows(path: str) -> list[list[str]]:
    """Return every row from a CSV file (including the header)."""
    with open(path, newline="") as f:
        return list(csv.reader(f))


# ---------------------------------------------------------------------------
# Fixtures: redirect config file paths to tmp_path and pre-create the CSV
# files with their headers so append operations work correctly.
# ---------------------------------------------------------------------------

@pytest.fixture()
def storage_files(tmp_path, monkeypatch):
    """
    Point config.SENSOR_DATA_FILE and config.ERROR_LOG_FILE at files inside
    tmp_path. Pre-create both files with their respective headers so that
    save_valid_record and log_error can append to them.
    """
    data_file = str(tmp_path / "sensor_data.csv")
    error_file = str(tmp_path / "error_log.csv")

    monkeypatch.setattr(config, "SENSOR_DATA_FILE", data_file)
    monkeypatch.setattr(config, "ERROR_LOG_FILE", error_file)

    with open(data_file, "w", newline="") as f:
        csv.writer(f).writerow(config.OUTPUT_HEADER)
    with open(error_file, "w", newline="") as f:
        csv.writer(f).writerow(config.ERROR_HEADER)

    return {"data": data_file, "error": error_file}


# ===========================================================================
# Tests for save_valid_record
# ===========================================================================

class TestSaveValidRecord:

    def test_appends_exactly_one_row(self, storage_files):
        """save_valid_record must append exactly one row to sensor_data.csv."""
        record = {
            "record_id":      "R000001",
            "timestamp":      "2024-01-01T10:00:00",
            "session_id":     1,
            "sampling_point": 2,
            "plant_id":       3,
            "soil":           45.5,
            "temperature":    22.0,
            "humidity":       60.0,
            "light":          300.0,
        }
        storage.save_valid_record(record)
        rows = _read_data_rows(storage_files["data"])
        assert len(rows) == 1, f"Expected 1 appended row, got {len(rows)}"

    def test_row_matches_output_header_order(self, storage_files):
        """Columns must appear in OUTPUT_HEADER order and values must round-trip."""
        record = {
            "record_id":      "R000042",
            "timestamp":      "2024-06-15T08:30:00",
            "session_id":     5,
            "sampling_point": 3,
            "plant_id":       7,
            "soil":           12.3,
            "temperature":    -5.0,
            "humidity":       88.8,
            "light":          999.9,
        }
        storage.save_valid_record(record)
        rows = _read_data_rows(storage_files["data"])
        assert len(rows) == 1

        expected = [str(record[field]) for field in config.OUTPUT_HEADER]
        assert rows[0] == expected, f"Row mismatch:\n  got      {rows[0]}\n  expected {expected}"

    def test_value_with_comma(self, storage_files):
        """A timestamp or any field value containing a comma must round-trip correctly."""
        record = {
            "record_id":      "R000002",
            "timestamp":      "value,with,commas",
            "session_id":     1,
            "sampling_point": 1,
            "plant_id":       1,
            "soil":           0.0,
            "temperature":    0.0,
            "humidity":       0.0,
            "light":          0.0,
        }
        storage.save_valid_record(record)
        rows = _read_data_rows(storage_files["data"])
        assert rows[0][1] == "value,with,commas", (
            f"Comma in value not preserved: {rows[0][1]!r}"
        )

    def test_value_with_double_quote(self, storage_files):
        """A value containing double-quotes must round-trip correctly via csv.writer quoting."""
        record = {
            "record_id":      "R000003",
            "timestamp":      'say "hello" world',
            "session_id":     1,
            "sampling_point": 1,
            "plant_id":       1,
            "soil":           0.0,
            "temperature":    0.0,
            "humidity":       0.0,
            "light":          0.0,
        }
        storage.save_valid_record(record)
        rows = _read_data_rows(storage_files["data"])
        assert rows[0][1] == 'say "hello" world', (
            f"Double-quote in value not preserved: {rows[0][1]!r}"
        )

    def test_value_with_newline(self, storage_files):
        """A value containing a newline must round-trip correctly via csv.writer quoting."""
        record = {
            "record_id":      "R000004",
            "timestamp":      "line1\nline2",
            "session_id":     1,
            "sampling_point": 1,
            "plant_id":       1,
            "soil":           0.0,
            "temperature":    0.0,
            "humidity":       0.0,
            "light":          0.0,
        }
        storage.save_valid_record(record)
        rows = _read_data_rows(storage_files["data"])
        assert rows[0][1] == "line1\nline2", (
            f"Newline in value not preserved: {rows[0][1]!r}"
        )

    def test_multiple_appends_produce_multiple_rows(self, storage_files):
        """Calling save_valid_record twice must produce two rows, not overwrite."""
        def make_record(rid, plant):
            return {
                "record_id":      rid,
                "timestamp":      "2024-01-01T00:00:00",
                "session_id":     1,
                "sampling_point": 1,
                "plant_id":       plant,
                "soil":           50.0,
                "temperature":    20.0,
                "humidity":       50.0,
                "light":          500.0,
            }

        storage.save_valid_record(make_record("R000001", 1))
        storage.save_valid_record(make_record("R000002", 2))

        rows = _read_data_rows(storage_files["data"])
        assert len(rows) == 2
        assert rows[0][0] == "R000001"
        assert rows[1][0] == "R000002"


# ===========================================================================
# Tests for log_error
# ===========================================================================

class TestLogError:

    def test_appends_exactly_one_row(self, storage_files):
        """log_error must append exactly one row to error_log.csv."""
        storage.log_error("R000001", "3", "soil", "999.9", "Expected 0-100%")
        rows = _read_error_rows(storage_files["error"])
        assert len(rows) == 1, f"Expected 1 appended row, got {len(rows)}"

    def test_row_matches_error_header_order(self, storage_files):
        """Columns must appear in ERROR_HEADER order: record_id, plant_id, sensor, bad_value, reason."""
        storage.log_error("R000007", "5", "temperature", "-99.0", "Expected -40 to 80 C")
        rows = _read_error_rows(storage_files["error"])
        assert len(rows) == 1

        record_id, plant_id, sensor, bad_value, reason = rows[0]
        assert record_id == "R000007"
        assert plant_id  == "5"
        assert sensor    == "temperature"
        assert bad_value == "-99.0"
        assert reason    == "Expected -40 to 80 C"

    def test_value_with_comma(self, storage_files):
        """A bad_value or reason containing a comma must round-trip correctly."""
        storage.log_error("R000010", "1", "record", "a,b,c", "Expected 8 fields, received 3")
        rows = _read_error_rows(storage_files["error"])
        assert rows[0][3] == "a,b,c",                          f"bad_value mismatch: {rows[0][3]!r}"
        assert rows[0][4] == "Expected 8 fields, received 3",  f"reason mismatch: {rows[0][4]!r}"

    def test_value_with_double_quote(self, storage_files):
        """A value containing double-quotes must round-trip correctly via csv.writer quoting."""
        storage.log_error("R000011", "2", "sensor", '"quoted"', 'reason with "quotes"')
        rows = _read_error_rows(storage_files["error"])
        assert rows[0][3] == '"quoted"',             f"bad_value mismatch: {rows[0][3]!r}"
        assert rows[0][4] == 'reason with "quotes"', f"reason mismatch: {rows[0][4]!r}"

    def test_value_with_newline(self, storage_files):
        """A value containing a newline must round-trip correctly via csv.writer quoting."""
        storage.log_error("R000012", "4", "humidity", "line1\nline2", "reason\nwith newline")
        rows = _read_error_rows(storage_files["error"])
        assert rows[0][3] == "line1\nline2",         f"bad_value newline not preserved: {rows[0][3]!r}"
        assert rows[0][4] == "reason\nwith newline", f"reason newline not preserved: {rows[0][4]!r}"

    def test_multiple_appends_produce_multiple_rows(self, storage_files):
        """Calling log_error twice must produce two rows, not overwrite."""
        storage.log_error("R000020", "1", "soil",  "101.0", "Expected 0-100%")
        storage.log_error("R000021", "2", "light", "-1.0",  "Expected 0-1000 lux")

        rows = _read_error_rows(storage_files["error"])
        assert len(rows) == 2
        assert rows[0][0] == "R000020"
        assert rows[1][0] == "R000021"

    def test_empty_plant_id(self, storage_files):
        """log_error with an empty plant_id (record-level error) must store an empty string."""
        storage.log_error("R000030", "", "record", "bad,data", "Expected 8 fields, received 2")
        rows = _read_error_rows(storage_files["error"])
        assert rows[0][1] == "", f"plant_id should be empty string, got {rows[0][1]!r}"


# ===========================================================================
# Tests for initialize_files
# Validates: Requirements 8.2
# ===========================================================================

class TestInitializeFiles:

    def test_both_files_created_with_correct_headers(self, tmp_path, monkeypatch):
        """
        When neither file exists, initialize_files should create both with
        their respective header rows as the sole content.
        """
        data_file = str(tmp_path / "sensor_data.csv")
        error_file = str(tmp_path / "error_log.csv")

        monkeypatch.setattr(config, "SENSOR_DATA_FILE", data_file)
        monkeypatch.setattr(config, "ERROR_LOG_FILE", error_file)

        storage.initialize_files()

        assert os.path.exists(data_file),  "sensor_data.csv was not created"
        assert os.path.exists(error_file), "error_log.csv was not created"

        data_rows = _read_all_rows(data_file)
        assert data_rows == [config.OUTPUT_HEADER], (
            f"sensor_data.csv header mismatch: {data_rows}"
        )

        error_rows = _read_all_rows(error_file)
        assert error_rows == [config.ERROR_HEADER], (
            f"error_log.csv header mismatch: {error_rows}"
        )

    def test_no_modification_when_both_exist(self, tmp_path, monkeypatch):
        """
        When both files already exist, initialize_files must leave them
        completely unchanged — no duplicate headers, no data loss.
        """
        data_file = str(tmp_path / "sensor_data.csv")
        error_file = str(tmp_path / "error_log.csv")

        monkeypatch.setattr(config, "SENSOR_DATA_FILE", data_file)
        monkeypatch.setattr(config, "ERROR_LOG_FILE", error_file)

        # Pre-populate with a header and one data row each
        existing_data_row = ["R000001", "2024-01-01T10:00:00", "1", "1", "1",
                             "50.0", "20.0", "50.0", "500.0"]
        existing_error_row = ["R000002", "2", "soil", "bad", "Expected 0-100%"]

        with open(data_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(config.OUTPUT_HEADER)
            writer.writerow(existing_data_row)

        with open(error_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(config.ERROR_HEADER)
            writer.writerow(existing_error_row)

        size_data_before  = os.path.getsize(data_file)
        size_error_before = os.path.getsize(error_file)

        storage.initialize_files()

        # File sizes must not change
        assert os.path.getsize(data_file)  == size_data_before, (
            "sensor_data.csv was modified when it should not have been"
        )
        assert os.path.getsize(error_file) == size_error_before, (
            "error_log.csv was modified when it should not have been"
        )

        # Contents must be exactly as written
        assert _read_all_rows(data_file)  == [config.OUTPUT_HEADER, existing_data_row]
        assert _read_all_rows(error_file) == [config.ERROR_HEADER, existing_error_row]

    def test_oserror_when_only_data_file_missing(self, tmp_path, monkeypatch):
        """
        When error_log.csv exists but sensor_data.csv does not,
        initialize_files must raise OSError and must NOT create sensor_data.csv.
        """
        data_file = str(tmp_path / "sensor_data.csv")
        error_file = str(tmp_path / "error_log.csv")

        monkeypatch.setattr(config, "SENSOR_DATA_FILE", data_file)
        monkeypatch.setattr(config, "ERROR_LOG_FILE", error_file)

        # Create only the error log
        with open(error_file, "w", newline="") as f:
            csv.writer(f).writerow(config.ERROR_HEADER)

        with pytest.raises(OSError):
            storage.initialize_files()

        assert not os.path.exists(data_file), (
            "sensor_data.csv was created despite OSError condition"
        )

    def test_oserror_when_only_error_file_missing(self, tmp_path, monkeypatch):
        """
        When sensor_data.csv exists but error_log.csv does not,
        initialize_files must raise OSError and must NOT create error_log.csv.
        """
        data_file = str(tmp_path / "sensor_data.csv")
        error_file = str(tmp_path / "error_log.csv")

        monkeypatch.setattr(config, "SENSOR_DATA_FILE", data_file)
        monkeypatch.setattr(config, "ERROR_LOG_FILE", error_file)

        # Create only the data file
        with open(data_file, "w", newline="") as f:
            csv.writer(f).writerow(config.OUTPUT_HEADER)

        with pytest.raises(OSError):
            storage.initialize_files()

        assert not os.path.exists(error_file), (
            "error_log.csv was created despite OSError condition"
        )
