"""
Parametrized unit tests for validate_record.
Validates: Requirements 8.1
"""
import pytest
from validator import validate_record, ValidationError

# ---------------------------------------------------------------------------
# Baseline valid record
# ---------------------------------------------------------------------------
VALID = ["2024-01-01T10:00:00", "1", "1", "1", "50.0", "20.0", "50.0", "500.0"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _replace(fields: list, index: int, value: str) -> list:
    """Return a copy of fields with fields[index] replaced by value."""
    copy = fields.copy()
    copy[index] = value
    return copy


# ===========================================================================
# 1. Valid records at boundary values for every numeric field
# ===========================================================================
@pytest.mark.parametrize("fields", [
    # session_id min
    _replace(VALID, 1, "1"),
    # sampling_point min / max
    _replace(VALID, 2, "1"),
    _replace(VALID, 2, "6"),
    # plant_id min / max
    _replace(VALID, 3, "1"),
    _replace(VALID, 3, "16"),
    # soil min / max
    _replace(VALID, 4, "0.0"),
    _replace(VALID, 4, "100.0"),
    # temperature min / max
    _replace(VALID, 5, "-40.0"),
    _replace(VALID, 5, "80.0"),
    # humidity min / max
    _replace(VALID, 6, "0.0"),
    _replace(VALID, 6, "100.0"),
    # light min / max
    _replace(VALID, 7, "0.0"),
    _replace(VALID, 7, "1000.0"),
])
def test_valid_boundary_values(fields):
    """Boundary values within valid range must return a dict, not errors."""
    result = validate_record(fields)
    assert isinstance(result, dict), f"Expected dict for {fields}, got {result}"


# ===========================================================================
# 2. Each numeric field one unit below its minimum (invalid)
# ===========================================================================
@pytest.mark.parametrize("fields,expected_sensor", [
    # session_id: min=1, so 0 is below
    (_replace(VALID, 1, "0"),      "session_id"),
    # sampling_point: min=1, so 0 is below
    (_replace(VALID, 2, "0"),      "sampling_point"),
    # plant_id: min=1, so 0 is below
    (_replace(VALID, 3, "0"),      "plant_id"),
    # soil: min=0.0, so -0.1 is below
    (_replace(VALID, 4, "-0.1"),   "soil"),
    # temperature: min=-40.0, so -40.1 is below
    (_replace(VALID, 5, "-40.1"),  "temperature"),
    # humidity: min=0.0, so -0.1 is below
    (_replace(VALID, 6, "-0.1"),   "humidity"),
    # light: min=0.0, so -0.1 is below
    (_replace(VALID, 7, "-0.1"),   "light"),
])
def test_below_minimum(fields, expected_sensor):
    """Fields one unit below minimum must return a ValidationError for that sensor."""
    result = validate_record(fields)
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == expected_sensor, (
        f"Expected sensor={expected_sensor!r}, got {result[0].sensor!r}"
    )


# ===========================================================================
# 3. Each numeric field one unit above its maximum (invalid)
# ===========================================================================
@pytest.mark.parametrize("fields,expected_sensor", [
    # sampling_point: max=6, so 7 is above
    (_replace(VALID, 2, "7"),       "sampling_point"),
    # plant_id: max=16, so 17 is above
    (_replace(VALID, 3, "17"),      "plant_id"),
    # soil: max=100.0, so 100.1 is above
    (_replace(VALID, 4, "100.1"),   "soil"),
    # temperature: max=80.0, so 80.1 is above
    (_replace(VALID, 5, "80.1"),    "temperature"),
    # humidity: max=100.0, so 100.1 is above
    (_replace(VALID, 6, "100.1"),   "humidity"),
    # light: max=1000.0, so 1000.1 is above
    (_replace(VALID, 7, "1000.1"),  "light"),
])
def test_above_maximum(fields, expected_sensor):
    """Fields one unit above maximum must return a ValidationError for that sensor."""
    result = validate_record(fields)
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == expected_sensor, (
        f"Expected sensor={expected_sensor!r}, got {result[0].sensor!r}"
    )


# ===========================================================================
# 4 & 5. Wrong field count — fewer or more than 8 fields
# ===========================================================================
@pytest.mark.parametrize("fields", [
    # Fewer than 8
    [],
    ["2024-01-01T10:00:00"],
    ["2024-01-01T10:00:00", "1", "1", "1", "50.0", "20.0", "50.0"],  # 7 fields
    # More than 8
    VALID + ["extra"],
    VALID + ["extra1", "extra2"],
])
def test_wrong_field_count(fields):
    """Records with field count != 8 must return a ValidationError with sensor='record'."""
    result = validate_record(fields)
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == "record"
    assert result[0].plant_id == ""


# ===========================================================================
# 6. Non-numeric string in each numeric field position
# ===========================================================================
@pytest.mark.parametrize("index,expected_sensor", [
    (1, "session_id"),
    (2, "sampling_point"),
    (3, "plant_id"),
    (4, "soil"),
    (5, "temperature"),
    (6, "humidity"),
    (7, "light"),
])
def test_non_numeric_string(index, expected_sensor):
    """A non-numeric string in any numeric field must return a ValidationError for that sensor."""
    fields = _replace(VALID, index, "not_a_number")
    result = validate_record(fields)
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == expected_sensor, (
        f"Expected sensor={expected_sensor!r} at index {index}, got {result[0].sensor!r}"
    )


# ===========================================================================
# 7. All-valid fields list: correct keys and Python types in returned dict
# ===========================================================================
def test_valid_record_dict_keys_and_types():
    """A fully valid record must return a dict with the exact expected keys and types."""
    result = validate_record(VALID)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    expected_keys = {
        "timestamp", "session_id", "sampling_point", "plant_id",
        "soil", "temperature", "humidity", "light",
    }
    assert set(result.keys()) == expected_keys, (
        f"Key mismatch: {set(result.keys())} != {expected_keys}"
    )

    assert isinstance(result["timestamp"],      str),   "timestamp must be str"
    assert isinstance(result["session_id"],     int),   "session_id must be int"
    assert isinstance(result["sampling_point"], int),   "sampling_point must be int"
    assert isinstance(result["plant_id"],       int),   "plant_id must be int"
    assert isinstance(result["soil"],           float), "soil must be float"
    assert isinstance(result["temperature"],    float), "temperature must be float"
    assert isinstance(result["humidity"],       float), "humidity must be float"
    assert isinstance(result["light"],          float), "light must be float"
