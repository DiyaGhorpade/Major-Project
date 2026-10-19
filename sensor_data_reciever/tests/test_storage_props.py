"""
Property-based tests for storage module.

**Validates: Requirements 3.5**

Property 10: save_valid_record round-trip preserves all field values.

For any valid record dict (with keys matching OUTPUT_HEADER), calling
save_valid_record and then reading the last row of sensor_data.csv SHALL
produce values that match the original dict's fields in OUTPUT_HEADER
order with no data loss or corruption.
"""

import csv
import tempfile
import os

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

import config
import storage


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

# Use printable ASCII text for string fields.  The receiver decodes serial
# data with utf-8 errors='replace' and stamps record_id as "R000001" etc., so
# all string values in practice consist of printable ASCII characters.
# Using the full Unicode space would also expose whether storage.py and the
# read-back agree on an encoding, which is an implementation detail outside
# the scope of this property (Requirement 3.5 concerns field ordering and
# value preservation, not character-set coverage).
_ascii_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        max_codepoint=127,   # restrict to 7-bit ASCII
    ),
    min_size=1,
)

# Strategy: generate valid record dicts with keys matching OUTPUT_HEADER
valid_record_strategy = st.fixed_dictionaries({
    "record_id":      _ascii_text,
    "timestamp":      _ascii_text,
    "session_id":     st.integers(),
    "sampling_point": st.integers(),
    "plant_id":       st.integers(),
    "soil":           st.floats(allow_nan=False, allow_infinity=False),
    "temperature":    st.floats(allow_nan=False, allow_infinity=False),
    "humidity":       st.floats(allow_nan=False, allow_infinity=False),
    "light":          st.floats(allow_nan=False, allow_infinity=False),
    "node_id":      st.sampled_from(["NODE_01", "NODE_02", "NODE_03", "NODE_04"]),
})


# ---------------------------------------------------------------------------
# Helper: read the last row from a CSV file
# ---------------------------------------------------------------------------

def _read_last_row(path: str) -> list[str]:
    """Return the last row from a CSV file as a list of strings."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    # rows[0] is the header; rows[-1] is the most recently appended row
    return rows[-1]


# ===========================================================================
# Property 10: save_valid_record round-trip preserves all field values
# ===========================================================================

@given(record=valid_record_strategy)
@settings(max_examples=500, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_save_valid_record_round_trip_preserves_all_fields(record):
    """
    Property 10: save_valid_record round-trip preserves all field values.

    **Validates: Requirements 3.5**

    For any valid record dict with keys matching OUTPUT_HEADER, writing via
    save_valid_record and reading the last row of sensor_data.csv back via
    csv.reader must reproduce every field value (as str) in OUTPUT_HEADER
    order — no data loss, no corruption.
    """
    # Manage the temp file entirely within the test body so that Hypothesis
    # resets state cleanly between generated examples (no function-scoped
    # fixtures needed, satisfying HealthCheck.function_scoped_fixture).
    original_data_file = config.SENSOR_DATA_FILE

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_file = os.path.join(tmp_dir, "sensor_data.csv")

        # Pre-create the CSV with the header so save_valid_record can append
        with open(data_file, "w", newline="") as f:
            csv.writer(f).writerow(config.OUTPUT_HEADER)

        # Redirect config so storage writes to the isolated temp file
        config.SENSOR_DATA_FILE = data_file
        try:
            storage.save_valid_record(record)
            last_row = _read_last_row(data_file)
        finally:
            config.SENSOR_DATA_FILE = original_data_file

    # Build the expected row: each OUTPUT_HEADER field converted to str
    expected = [str(record[field]) for field in config.OUTPUT_HEADER]

    assert last_row == expected, (
        f"Round-trip mismatch.\n"
        f"  Input record : {record}\n"
        f"  Expected row : {expected}\n"
        f"  Actual row   : {last_row}"
    )
