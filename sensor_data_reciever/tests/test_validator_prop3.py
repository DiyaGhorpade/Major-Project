"""
Property-based test for validate_record — Property 3.

**Validates: Requirements 2.5**

Property 3: For any length-9 fields list where exactly one of soil (idx 5),
temperature (idx 6), humidity (idx 7), or light (idx 8) contains a string that
cannot be converted to float (all other fields valid and parseable), calling
validate_record must return a list[ValidationError] whose single entry has
sensor matching the name of the failing field.
"""

from hypothesis import given, settings
import hypothesis.strategies as st

from validator import validate_record, ValidationError

# Base valid 9-field list — all fields parseable and in-range
VALID_BASE = ["2024-01-01T10:00:00", "1", "1", "1", "NODE_01", "50.0", "20.0", "50.0", "500.0"]

# Float sensor fields: (name, index)
FLOAT_FIELDS = [("soil", 5), ("temperature", 6), ("humidity", 7), ("light", 8)]

# Strategy: a non-float string (purely alphabetic text cannot be parsed as float)
non_float_string = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu")),  # lower + upper letters only
    min_size=1,
).filter(lambda s: not _is_float(s))


def _is_float(s: str) -> bool:
    """Return True if s can be converted to float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


@given(
    field_info=st.sampled_from(FLOAT_FIELDS),
    bad_value=non_float_string,
)
@settings(max_examples=500)
def test_non_parseable_float_field_produces_correct_sensor_name(field_info, bad_value):
    """
    Property 3: A non-parseable float field produces a ValidationError with
    sensor matching the name of the failing field.

    For any length-8 fields list where exactly one of soil/temperature/humidity/light
    contains a non-float string, validate_record must return a list[ValidationError]
    with exactly one entry whose sensor equals the field name.
    """
    field_name, field_index = field_info

    # Build a fully valid 8-field list and swap the target index with the bad value
    fields = VALID_BASE[:]
    fields[field_index] = bad_value

    result = validate_record(fields)

    # Must return a list (not a dict)
    assert isinstance(result, list), (
        f"Expected list[ValidationError], got {type(result).__name__} "
        f"when {field_name}={bad_value!r}"
    )

    # Must contain exactly one ValidationError
    assert len(result) == 1, (
        f"Expected exactly 1 ValidationError, got {len(result)} "
        f"when {field_name}={bad_value!r}"
    )

    error = result[0]

    assert isinstance(error, ValidationError), (
        f"Expected ValidationError instance, got {type(error).__name__}"
    )

    # The sensor name must match the failing field
    assert error.sensor == field_name, (
        f"Expected sensor={field_name!r}, got sensor={error.sensor!r} "
        f"when index {field_index} contains {bad_value!r}"
    )
