"""
Property-based test for process_line — Property 12.

**Validates: Requirements 5.5**

Property 12: Valid serial line causes save_valid_record to be called with
correct record.

For any valid CSV line (8 fields, all numeric fields in range), process_line
results in:
  - exactly one call to storage.save_valid_record with the correct record_id
    ("R000001") and all correctly typed field values, and
  - a [VALID] console message containing record_id, session_id,
    sampling_point, and plant_id.
"""

import csv
import io
import sys
import os
from unittest.mock import patch, MagicMock

import hypothesis.strategies as st
from hypothesis import given, settings

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import receiver


# ---------------------------------------------------------------------------
# Strategies for valid field values
# ---------------------------------------------------------------------------

valid_timestamp = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), max_codepoint=127),
    min_size=1,
)
valid_session_id = st.integers(min_value=1)
valid_sampling_point = st.integers(min_value=1, max_value=6)
valid_plant_id = st.integers(min_value=1, max_value=16)
valid_soil = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
valid_temperature = st.floats(min_value=-40.0, max_value=80.0, allow_nan=False, allow_infinity=False)
valid_humidity = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
valid_light = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


def build_csv_line(timestamp, session_id, sampling_point, plant_id,
                   soil, temperature, humidity, light) -> str:
    """Serialize eight fields to a CSV line using csv.writer to correctly
    handle any special characters that may appear in the timestamp."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        timestamp,
        session_id,
        sampling_point,
        plant_id,
        soil,
        temperature,
        humidity,
        light,
    ])
    return buf.getvalue().rstrip("\r\n")


# ---------------------------------------------------------------------------
# Property 12
# ---------------------------------------------------------------------------

@given(
    timestamp=valid_timestamp,
    session_id=valid_session_id,
    sampling_point=valid_sampling_point,
    plant_id=valid_plant_id,
    soil=valid_soil,
    temperature=valid_temperature,
    humidity=valid_humidity,
    light=valid_light,
)
@settings(max_examples=200)
def test_valid_serial_line_calls_save_valid_record_with_correct_record(
    timestamp, session_id, sampling_point, plant_id,
    soil, temperature, humidity, light,
):
    """
    Property 12: For any valid CSV line, process_line calls
    storage.save_valid_record exactly once with a dict that contains
    record_id="R000001" and the correct typed field values, and prints a
    [VALID] message containing record_id, session_id, sampling_point, and
    plant_id.

    **Validates: Requirements 5.5**
    """
    # Reset counter so every hypothesis example starts from R000001
    receiver.record_counter = 0

    line = build_csv_line(
        timestamp, session_id, sampling_point, plant_id,
        soil, temperature, humidity, light,
    )

    mock_storage = MagicMock()

    with patch.object(receiver, "storage", mock_storage), \
         patch("builtins.print") as mock_print:
        receiver.process_line(line)

    # --- storage.save_valid_record called exactly once ---
    mock_storage.save_valid_record.assert_called_once()

    # --- log_error must NOT have been called ---
    mock_storage.log_error.assert_not_called()

    # --- Inspect the dict passed to save_valid_record ---
    saved_dict = mock_storage.save_valid_record.call_args[0][0]

    assert saved_dict["record_id"] == "R000001", (
        f"Expected record_id='R000001', got {saved_dict['record_id']!r}"
    )
    assert saved_dict["timestamp"] == timestamp, (
        f"timestamp mismatch: expected {timestamp!r}, got {saved_dict['timestamp']!r}"
    )
    assert saved_dict["session_id"] == session_id, (
        f"session_id mismatch: expected {session_id}, got {saved_dict['session_id']}"
    )
    assert saved_dict["sampling_point"] == sampling_point, (
        f"sampling_point mismatch: expected {sampling_point}, got {saved_dict['sampling_point']}"
    )
    assert saved_dict["plant_id"] == plant_id, (
        f"plant_id mismatch: expected {plant_id}, got {saved_dict['plant_id']}"
    )
    assert saved_dict["soil"] == soil, (
        f"soil mismatch: expected {soil}, got {saved_dict['soil']}"
    )
    assert saved_dict["temperature"] == temperature, (
        f"temperature mismatch: expected {temperature}, got {saved_dict['temperature']}"
    )
    assert saved_dict["humidity"] == humidity, (
        f"humidity mismatch: expected {humidity}, got {saved_dict['humidity']}"
    )
    assert saved_dict["light"] == light, (
        f"light mismatch: expected {light}, got {saved_dict['light']}"
    )

    # --- At least one print call must contain [VALID] and R000001 ---
    print_texts = [str(call_args) for call_args in mock_print.call_args_list]
    valid_messages = [t for t in print_texts if "[VALID]" in t and "R000001" in t]
    assert valid_messages, (
        f"Expected a [VALID] R000001 print call, got: {print_texts}"
    )

    # --- The [VALID] message must also contain session_id, sampling_point, plant_id ---
    valid_msg = valid_messages[0]
    assert str(session_id) in valid_msg, (
        f"session_id {session_id} not found in [VALID] message: {valid_msg}"
    )
    assert str(sampling_point) in valid_msg, (
        f"sampling_point {sampling_point} not found in [VALID] message: {valid_msg}"
    )
    assert str(plant_id) in valid_msg, (
        f"plant_id {plant_id} not found in [VALID] message: {valid_msg}"
    )
