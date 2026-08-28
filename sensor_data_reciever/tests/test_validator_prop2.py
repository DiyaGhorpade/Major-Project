"""
Property 2: Non-parseable integer field produces a ValidationError with the correct sensor name.

**Validates: Requirements 2.2, 2.3, 2.4**

For any length-8 fields list where one of session_id (idx 1), sampling_point (idx 2), or
plant_id (idx 3) contains a string that cannot be converted to int (all other fields structurally
valid), calling validate_record must return a list[ValidationError] whose single entry has
`sensor` matching the name of the failing field.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings
import hypothesis.strategies as st

from validator import validate_record, ValidationError


# ---------------------------------------------------------------------------
# Helper: detect whether a string can be parsed as int (mirrors validator logic)
# ---------------------------------------------------------------------------

def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Strategy for non-parseable integer strings
# ---------------------------------------------------------------------------

# Generate text made entirely of ascii letters — these can never be parsed as int.
non_int_string_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
    min_size=1,
)

# Mapping from field name to its index in the 8-element fields list
FIELD_INDEX = {
    "session_id":     1,
    "sampling_point": 2,
    "plant_id":       3,
}

# ---------------------------------------------------------------------------
# Property 2 test
# ---------------------------------------------------------------------------

@given(
    failing_field=st.sampled_from(["session_id", "sampling_point", "plant_id"]),
    bad_int_str=non_int_string_st,
)
@settings(max_examples=200)
def test_non_parseable_integer_field_produces_correct_sensor_name(
    failing_field: str,
    bad_int_str: str,
) -> None:
    """
    Property 2: For any length-8 fields list where exactly one integer field (session_id,
    sampling_point, or plant_id) contains a non-parseable string and all other fields are
    structurally valid, validate_record returns a list[ValidationError] whose single entry
    has sensor == failing_field.

    **Validates: Requirements 2.2, 2.3, 2.4**
    """
    # Build a valid base record
    # timestamp=0, session_id=1, sampling_point=1, plant_id=1,
    # soil=50.0, temperature=20.0, humidity=50.0, light=500.0
    fields = [
        "2024-01-01T00:00:00",  # 0: timestamp
        "1",                    # 1: session_id
        "1",                    # 2: sampling_point
        "1",                    # 3: plant_id
        "50.0",                 # 4: soil
        "20.0",                 # 5: temperature
        "50.0",                 # 6: humidity
        "500.0",                # 7: light
    ]

    # Replace the chosen integer field with a non-parseable string
    idx = FIELD_INDEX[failing_field]
    fields[idx] = bad_int_str

    result = validate_record(fields)

    # Must return a list (not a dict)
    assert isinstance(result, list), (
        f"Expected list[ValidationError] but got {type(result).__name__} "
        f"for failing_field={failing_field!r}, bad_int_str={bad_int_str!r}"
    )

    # Must contain exactly one ValidationError
    assert len(result) == 1, (
        f"Expected exactly 1 ValidationError but got {len(result)} "
        f"for failing_field={failing_field!r}, bad_int_str={bad_int_str!r}"
    )

    error = result[0]

    # The error must be a ValidationError instance
    assert isinstance(error, ValidationError), (
        f"Expected ValidationError instance but got {type(error).__name__}"
    )

    # The sensor field must match the name of the failing field
    assert error.sensor == failing_field, (
        f"Expected sensor={failing_field!r} but got sensor={error.sensor!r} "
        f"for bad_int_str={bad_int_str!r}"
    )
