"""
Property-based test for validate_record — Property 1.

**Validates: Requirements 2.1**

Property 1: For any list[str] with len != 9, calling validate_record must return a
list[ValidationError] with exactly one entry where sensor="record" and plant_id="".
"""

from hypothesis import given, settings
import hypothesis.strategies as st

from validator import validate_record, ValidationError


@given(st.lists(st.text()).filter(lambda x: len(x) != 9))
@settings(max_examples=500)
def test_wrong_field_count_produces_structural_validation_error(fields):
    """
    Property 1: Wrong field count always produces a structural ValidationError.

    For any list[str] whose length is not 9, validate_record must return a
    list containing exactly one ValidationError with sensor="record" and plant_id="".
    """
    result = validate_record(fields)

    # Must return a list
    assert isinstance(result, list), (
        f"Expected list, got {type(result).__name__} for input of length {len(fields)}"
    )

    # Must contain exactly one entry
    assert len(result) == 1, (
        f"Expected exactly 1 ValidationError, got {len(result)} for input of length {len(fields)}"
    )

    error = result[0]

    # The single entry must be a ValidationError
    assert isinstance(error, ValidationError), (
        f"Expected ValidationError instance, got {type(error).__name__}"
    )

    # sensor must be "record"
    assert error.sensor == "record", (
        f"Expected sensor='record', got sensor={error.sensor!r}"
    )

    # plant_id must be ""
    assert error.plant_id == "", (
        f"Expected plant_id='', got plant_id={error.plant_id!r}"
    )
