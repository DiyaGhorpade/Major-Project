"""
Unit tests for receiver.process_line — Requirement 8.4.

Validates: Requirements 8.4

Storage and validator are monkeypatched so no real files are touched.
receiver.record_counter is reset to 0 before each test to keep IDs
predictable.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, call

# Ensure project root is on path (conftest.py also does this, but be explicit)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import receiver
import config


# ---------------------------------------------------------------------------
# Fixture: monkeypatch storage and validator on the receiver module,
#          and reset the global record_counter to 0 before every test.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_counter():
    """Reset receiver.record_counter to 0 before each test."""
    receiver.record_counter = 0
    yield
    receiver.record_counter = 0


@pytest.fixture()
def mock_storage(monkeypatch):
    """Replace receiver.storage with a MagicMock."""
    mock = MagicMock()
    monkeypatch.setattr(receiver, "storage", mock)
    return mock


@pytest.fixture()
def mock_validator(monkeypatch):
    """Replace receiver.validator with a MagicMock."""
    mock = MagicMock()
    monkeypatch.setattr(receiver, "validator", mock)
    return mock


# ---------------------------------------------------------------------------
# Helper — the header line as a string
# ---------------------------------------------------------------------------

HEADER_LINE = ",".join(config.INPUT_HEADER)

# ---------------------------------------------------------------------------
# Test 1 — Empty string → no calls to storage or validator
# ---------------------------------------------------------------------------

def test_empty_string_no_calls(mock_storage, mock_validator):
    """
    Validates: Requirements 8.4

    An empty line (or whitespace-only line) must return immediately without
    touching storage or validator, and must NOT consume a record ID.
    """
    receiver.process_line("")
    mock_storage.save_valid_record.assert_not_called()
    mock_storage.log_error.assert_not_called()
    mock_validator.validate_record.assert_not_called()
    # record_counter must still be 0
    assert receiver.record_counter == 0


def test_whitespace_only_string_no_calls(mock_storage, mock_validator):
    """
    Validates: Requirements 8.4

    A whitespace-only line is equivalent to an empty line after stripping.
    """
    receiver.process_line("   \t  ")
    mock_storage.save_valid_record.assert_not_called()
    mock_storage.log_error.assert_not_called()
    mock_validator.validate_record.assert_not_called()
    assert receiver.record_counter == 0


# ---------------------------------------------------------------------------
# Test 2 — Header line → no calls to storage or validator
# ---------------------------------------------------------------------------

def test_header_line_no_calls(mock_storage, mock_validator):
    """
    Validates: Requirements 8.4

    The INPUT_HEADER line must be silently skipped without touching storage
    or validator, and must NOT consume a record ID.
    """
    receiver.process_line(HEADER_LINE)
    mock_storage.save_valid_record.assert_not_called()
    mock_storage.log_error.assert_not_called()
    mock_validator.validate_record.assert_not_called()
    assert receiver.record_counter == 0


def test_header_line_with_surrounding_whitespace_no_calls(mock_storage, mock_validator):
    """
    Validates: Requirements 8.4

    The header line with leading/trailing whitespace is still skipped after
    stripping.
    """
    receiver.process_line("  " + HEADER_LINE + "\n")
    mock_storage.save_valid_record.assert_not_called()
    mock_storage.log_error.assert_not_called()
    mock_validator.validate_record.assert_not_called()
    assert receiver.record_counter == 0


# ---------------------------------------------------------------------------
# Test 3 — Valid CSV line → exactly one call to storage.save_valid_record
#           with a dict containing the correct record_id and field values
# ---------------------------------------------------------------------------

def test_valid_csv_line_calls_save_valid_record(mock_storage, monkeypatch):
    """
    Validates: Requirements 8.4

    A valid data line must:
    - call validator.validate_record once
    - call storage.save_valid_record exactly once
    - pass a dict that includes record_id="R000001" and all parsed fields
    - NOT call storage.log_error
    """
    # The real validator returns a proper dict for this line; use it so the
    # dict passed to save_valid_record is realistic.
    import validator as real_validator

    # Only monkeypatch storage, let the real validator run
    mock_stor = MagicMock()
    monkeypatch.setattr(receiver, "storage", mock_stor)
    # Restore real validator (in case mock_validator fixture ran first)
    monkeypatch.setattr(receiver, "validator", real_validator)

    valid_line = "2024-01-01T10:00:00,1,3,5,NODE_01,45.5,22.0,60.0,300.0"
    receiver.process_line(valid_line)

    # save_valid_record called exactly once
    mock_stor.save_valid_record.assert_called_once()
    # log_error must NOT have been called
    mock_stor.log_error.assert_not_called()

    # Inspect the argument dict
    saved_dict = mock_stor.save_valid_record.call_args[0][0]

    assert saved_dict["record_id"] == "R000001"
    assert saved_dict["timestamp"] == "2024-01-01T10:00:00"
    assert saved_dict["session_id"] == 1
    assert saved_dict["sampling_point"] == 3
    assert saved_dict["plant_id"] == 5
    assert saved_dict["node_id"] == "NODE_01"
    assert saved_dict["soil"] == 45.5
    assert saved_dict["temperature"] == 22.0
    assert saved_dict["humidity"] == 60.0
    assert saved_dict["light"] == 300.0


def test_valid_csv_line_record_counter_increments(monkeypatch):
    """
    Validates: Requirements 8.4

    Processing a valid line must consume exactly one record ID so that
    record_counter advances to 1.
    """
    import validator as real_validator
    mock_stor = MagicMock()
    monkeypatch.setattr(receiver, "storage", mock_stor)
    monkeypatch.setattr(receiver, "validator", real_validator)

    receiver.process_line("2024-01-01T10:00:00,1,3,5,NODE_01,45.5,22.0,60.0,300.0")
    assert receiver.record_counter == 1


# ---------------------------------------------------------------------------
# Test 4 — Invalid CSV line → exactly one call to storage.log_error
#           with the correct five arguments
# ---------------------------------------------------------------------------

def test_invalid_csv_line_calls_log_error(mock_storage, monkeypatch):
    """
    Validates: Requirements 8.4

    A line with an invalid field (non-integer session_id) must:
    - call validator.validate_record once
    - call storage.log_error exactly once
    - NOT call storage.save_valid_record
    """
    import validator as real_validator

    mock_stor = MagicMock()
    monkeypatch.setattr(receiver, "storage", mock_stor)
    monkeypatch.setattr(receiver, "validator", real_validator)

    # session_id = "abc" is not a valid integer → validator returns a list[ValidationError]
    invalid_line = "2024-01-01T10:00:00,abc,3,5,NODE_01,45.5,22.0,60.0,300.0"
    receiver.process_line(invalid_line)

    # log_error called exactly once
    mock_stor.log_error.assert_called_once()
    # save_valid_record must NOT have been called
    mock_stor.save_valid_record.assert_not_called()

    # Verify the five arguments passed to log_error
    # Expected: log_error(record_id, plant_id, sensor, bad_value, reason, node_id)
    args = mock_stor.log_error.call_args[0]
    record_id, plant_id, sensor, bad_value, reason, node_id = args

    assert record_id == "R000001"
    assert sensor == "session_id"
    assert bad_value == "abc"
    assert reason == "Expected an integer"
    # plant_id at index 3 of the raw line is "5" (raw string, before plant_id is parsed)
    assert plant_id == "5"
    assert node_id == "NODE_01"


def test_invalid_csv_line_record_counter_increments(monkeypatch):
    """
    Validates: Requirements 8.4

    An invalid (but non-empty, non-header) line still consumes one record ID.
    """
    import validator as real_validator
    mock_stor = MagicMock()
    monkeypatch.setattr(receiver, "storage", mock_stor)
    monkeypatch.setattr(receiver, "validator", real_validator)

    receiver.process_line("2024-01-01T10:00:00,abc,3,5,NODE_01,45.5,22.0,60.0,300.0")
    assert receiver.record_counter == 1


# ---------------------------------------------------------------------------
# Test 5 — Sequence: valid then invalid uses consecutive IDs
# ---------------------------------------------------------------------------

def test_consecutive_ids_across_valid_and_invalid(monkeypatch):
    """
    Validates: Requirements 8.4

    Processing a valid line followed by an invalid line must assign R000001
    and R000002 respectively.
    """
    import validator as real_validator
    mock_stor = MagicMock()
    monkeypatch.setattr(receiver, "storage", mock_stor)
    monkeypatch.setattr(receiver, "validator", real_validator)

    receiver.process_line("2024-01-01T10:00:00,1,3,5,NODE_01,45.5,22.0,60.0,300.0")
    receiver.process_line("2024-01-01T10:00:01,abc,3,5,NODE_02,45.5,22.0,60.0,300.0")

    # First call: save_valid_record with R000001
    saved_dict = mock_stor.save_valid_record.call_args_list[0][0][0]
    assert saved_dict["record_id"] == "R000001"

    # Second call: log_error with R000002
    log_args = mock_stor.log_error.call_args_list[0][0]
    assert log_args[0] == "R000002"
