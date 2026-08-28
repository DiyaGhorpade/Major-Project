"""
Property-based test for process_line — Property 13.

**Validates: Requirements 5.6**

Property 13: Invalid serial line causes log_error to be called exactly once.

For any invalid CSV line (e.g. session_id out of range), process_line results in:
  - exactly one call to storage.log_error, and
  - a single [INVALID] {record_id} console message (no [VALID] message).
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
# Strategy: build an invalid line using an out-of-range session_id (< 1)
#
# session_id <= 0 always triggers a ValidationError (VALID_SESSION_ID_MIN=1),
# so every generated line is guaranteed invalid.  All other fields are kept
# in their valid ranges so session_id is the ONLY failure — this ensures the
# validator returns exactly ONE ValidationError (fail-fast), which means
# log_error will be called exactly once.
# ---------------------------------------------------------------------------

valid_timestamp = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), max_codepoint=127),
    min_size=1,
)
# session_id <= 0 is always below VALID_SESSION_ID_MIN (1)
invalid_session_id = st.integers(max_value=0)

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
# Property 13
# ---------------------------------------------------------------------------

@given(
    timestamp=valid_timestamp,
    session_id=invalid_session_id,
    sampling_point=valid_sampling_point,
    plant_id=valid_plant_id,
    soil=valid_soil,
    temperature=valid_temperature,
    humidity=valid_humidity,
    light=valid_light,
)
@settings(max_examples=200)
def test_invalid_serial_line_calls_log_error_exactly_once(
    timestamp, session_id, sampling_point, plant_id,
    soil, temperature, humidity, light,
):
    """
    Property 13: For any invalid CSV line, process_line calls
    storage.log_error exactly once and prints a single [INVALID] {record_id}
    message (no [VALID] message, no save_valid_record call).

    The invalid line is constructed by using session_id <= 0, which is always
    below VALID_SESSION_ID_MIN (1).  All other fields are valid, so the
    validator returns exactly one ValidationError (fail-fast), guaranteeing
    log_error is called exactly once.

    **Validates: Requirements 5.6**
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

    # --- storage.log_error called exactly once ---
    mock_storage.log_error.assert_called_once()

    # --- save_valid_record must NOT have been called ---
    mock_storage.save_valid_record.assert_not_called()

    # --- Exactly one [INVALID] R000001 print call ---
    print_texts = [str(call_args) for call_args in mock_print.call_args_list]
    invalid_messages = [t for t in print_texts if "[INVALID]" in t and "R000001" in t]
    assert len(invalid_messages) == 1, (
        f"Expected exactly one [INVALID] R000001 print call, got: {print_texts}"
    )

    # --- No [VALID] messages ---
    valid_messages = [t for t in print_texts if "[VALID]" in t]
    assert not valid_messages, (
        f"Expected no [VALID] print calls for an invalid line, got: {valid_messages}"
    )
