"""
Property-based tests for validate_record — Properties 4 through 8.

**Validates: Requirements 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14, 2.15**

Property 4 (Task 2.6): Out-of-range integer field produces the correct sensor name.
Property 5 (Task 2.7): Out-of-range float field produces the correct sensor name.
Property 6 (Task 2.8): All-valid fields list produces a correctly typed dict.
Property 7 (Task 2.9): Fail-fast — at most one ValidationError per call.
Property 8 (Task 2.10): validate_record never raises an exception.
"""

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from validator import validate_record, ValidationError
import config


# ============================================================
# Shared strategies
# ============================================================

# A valid timestamp is any non-empty string (no structural constraints in spec).
valid_timestamp = st.text(min_size=1)

# Valid integer fields
valid_session_id    = st.integers(min_value=config.VALID_SESSION_ID_MIN)
valid_sampling_point = st.integers(
    min_value=config.VALID_SAMPLING_POINT_MIN,
    max_value=config.VALID_SAMPLING_POINT_MAX,
)
valid_plant_id = st.integers(
    min_value=config.VALID_PLANT_ID_MIN,
    max_value=config.VALID_PLANT_ID_MAX,
)

# Valid float fields
valid_soil = st.floats(
    min_value=config.VALID_SOIL_MIN,
    max_value=config.VALID_SOIL_MAX,
    allow_nan=False,
    allow_infinity=False,
)
valid_temperature = st.floats(
    min_value=config.VALID_TEMPERATURE_MIN,
    max_value=config.VALID_TEMPERATURE_MAX,
    allow_nan=False,
    allow_infinity=False,
)
valid_humidity = st.floats(
    min_value=config.VALID_HUMIDITY_MIN,
    max_value=config.VALID_HUMIDITY_MAX,
    allow_nan=False,
    allow_infinity=False,
)
valid_light = st.floats(
    min_value=config.VALID_LIGHT_MIN,
    max_value=config.VALID_LIGHT_MAX,
    allow_nan=False,
    allow_infinity=False,
)


def make_valid_fields(ts, sid, sp, pid, soil, temp, hum, light):
    """Return an 8-element list[str] of valid field values."""
    return [
        ts,
        str(sid),
        str(sp),
        str(pid),
        str(soil),
        str(temp),
        str(hum),
        str(light),
    ]


@st.composite
def valid_8_fields(draw):
    """Draw a fully valid 8-field list[str]."""
    return make_valid_fields(
        draw(valid_timestamp),
        draw(valid_session_id),
        draw(valid_sampling_point),
        draw(valid_plant_id),
        draw(valid_soil),
        draw(valid_temperature),
        draw(valid_humidity),
        draw(valid_light),
    )


# ============================================================
# Property 4 — Out-of-range INTEGER field → correct sensor name
# **Validates: Requirements 2.6, 2.7, 2.8**
# ============================================================

# Out-of-range strategies for each integer field
out_of_range_session_id = st.integers(max_value=config.VALID_SESSION_ID_MIN - 1)

out_of_range_sampling_point = st.one_of(
    st.integers(max_value=config.VALID_SAMPLING_POINT_MIN - 1),
    st.integers(min_value=config.VALID_SAMPLING_POINT_MAX + 1),
)

out_of_range_plant_id = st.one_of(
    st.integers(max_value=config.VALID_PLANT_ID_MIN - 1),
    st.integers(min_value=config.VALID_PLANT_ID_MAX + 1),
)


@given(
    ts=valid_timestamp,
    bad_sid=out_of_range_session_id,
    sp=valid_sampling_point,
    pid=valid_plant_id,
    soil=valid_soil,
    temp=valid_temperature,
    hum=valid_humidity,
    light=valid_light,
)
@settings(max_examples=300)
def test_out_of_range_session_id_reports_correct_sensor(
    ts, bad_sid, sp, pid, soil, temp, hum, light
):
    """
    Property 4a: session_id below VALID_SESSION_ID_MIN produces sensor='session_id'.

    **Validates: Requirements 2.6**
    """
    fields = make_valid_fields(ts, bad_sid, sp, pid, soil, temp, hum, light)
    result = validate_record(fields)

    assert isinstance(result, list), (
        f"Expected list[ValidationError], got {type(result).__name__}"
    )
    assert len(result) == 1, f"Expected exactly 1 error, got {len(result)}"
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == "session_id", (
        f"Expected sensor='session_id', got {result[0].sensor!r} for session_id={bad_sid}"
    )


@given(
    ts=valid_timestamp,
    sid=valid_session_id,
    bad_sp=out_of_range_sampling_point,
    pid=valid_plant_id,
    soil=valid_soil,
    temp=valid_temperature,
    hum=valid_humidity,
    light=valid_light,
)
@settings(max_examples=300)
def test_out_of_range_sampling_point_reports_correct_sensor(
    ts, sid, bad_sp, pid, soil, temp, hum, light
):
    """
    Property 4b: sampling_point outside [1,6] produces sensor='sampling_point'.

    **Validates: Requirements 2.7**
    """
    fields = make_valid_fields(ts, sid, bad_sp, pid, soil, temp, hum, light)
    result = validate_record(fields)

    assert isinstance(result, list), (
        f"Expected list[ValidationError], got {type(result).__name__}"
    )
    assert len(result) == 1, f"Expected exactly 1 error, got {len(result)}"
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == "sampling_point", (
        f"Expected sensor='sampling_point', got {result[0].sensor!r} for sampling_point={bad_sp}"
    )


@given(
    ts=valid_timestamp,
    sid=valid_session_id,
    sp=valid_sampling_point,
    bad_pid=out_of_range_plant_id,
    soil=valid_soil,
    temp=valid_temperature,
    hum=valid_humidity,
    light=valid_light,
)
@settings(max_examples=300)
def test_out_of_range_plant_id_reports_correct_sensor(
    ts, sid, sp, bad_pid, soil, temp, hum, light
):
    """
    Property 4c: plant_id outside [1,16] produces sensor='plant_id'.

    **Validates: Requirements 2.8**
    """
    fields = make_valid_fields(ts, sid, sp, bad_pid, soil, temp, hum, light)
    result = validate_record(fields)

    assert isinstance(result, list), (
        f"Expected list[ValidationError], got {type(result).__name__}"
    )
    assert len(result) == 1, f"Expected exactly 1 error, got {len(result)}"
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == "plant_id", (
        f"Expected sensor='plant_id', got {result[0].sensor!r} for plant_id={bad_pid}"
    )


# ============================================================
# Property 5 — Out-of-range FLOAT field → correct sensor name
# **Validates: Requirements 2.9, 2.10, 2.11, 2.12**
# ============================================================

out_of_range_soil = st.one_of(
    st.floats(max_value=config.VALID_SOIL_MIN - 1e-9, allow_nan=False, allow_infinity=False),
    st.floats(min_value=config.VALID_SOIL_MAX + 1e-9, allow_nan=False, allow_infinity=False),
)
out_of_range_temperature = st.one_of(
    st.floats(max_value=config.VALID_TEMPERATURE_MIN - 1e-9, allow_nan=False, allow_infinity=False),
    st.floats(min_value=config.VALID_TEMPERATURE_MAX + 1e-9, allow_nan=False, allow_infinity=False),
)
out_of_range_humidity = st.one_of(
    st.floats(max_value=config.VALID_HUMIDITY_MIN - 1e-9, allow_nan=False, allow_infinity=False),
    st.floats(min_value=config.VALID_HUMIDITY_MAX + 1e-9, allow_nan=False, allow_infinity=False),
)
out_of_range_light = st.one_of(
    st.floats(max_value=config.VALID_LIGHT_MIN - 1e-9, allow_nan=False, allow_infinity=False),
    st.floats(min_value=config.VALID_LIGHT_MAX + 1e-9, allow_nan=False, allow_infinity=False),
)


@given(
    ts=valid_timestamp,
    sid=valid_session_id,
    sp=valid_sampling_point,
    pid=valid_plant_id,
    bad_soil=out_of_range_soil,
    temp=valid_temperature,
    hum=valid_humidity,
    light=valid_light,
)
@settings(max_examples=300)
def test_out_of_range_soil_reports_correct_sensor(
    ts, sid, sp, pid, bad_soil, temp, hum, light
):
    """
    Property 5a: soil outside [0.0, 100.0] produces sensor='soil'.

    **Validates: Requirements 2.9**
    """
    fields = make_valid_fields(ts, sid, sp, pid, bad_soil, temp, hum, light)
    result = validate_record(fields)

    assert isinstance(result, list), (
        f"Expected list[ValidationError], got {type(result).__name__}"
    )
    assert len(result) == 1, f"Expected exactly 1 error, got {len(result)}"
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == "soil", (
        f"Expected sensor='soil', got {result[0].sensor!r} for soil={bad_soil}"
    )


@given(
    ts=valid_timestamp,
    sid=valid_session_id,
    sp=valid_sampling_point,
    pid=valid_plant_id,
    soil=valid_soil,
    bad_temp=out_of_range_temperature,
    hum=valid_humidity,
    light=valid_light,
)
@settings(max_examples=300)
def test_out_of_range_temperature_reports_correct_sensor(
    ts, sid, sp, pid, soil, bad_temp, hum, light
):
    """
    Property 5b: temperature outside [-40.0, 80.0] produces sensor='temperature'.

    **Validates: Requirements 2.10**
    """
    fields = make_valid_fields(ts, sid, sp, pid, soil, bad_temp, hum, light)
    result = validate_record(fields)

    assert isinstance(result, list), (
        f"Expected list[ValidationError], got {type(result).__name__}"
    )
    assert len(result) == 1, f"Expected exactly 1 error, got {len(result)}"
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == "temperature", (
        f"Expected sensor='temperature', got {result[0].sensor!r} for temperature={bad_temp}"
    )


@given(
    ts=valid_timestamp,
    sid=valid_session_id,
    sp=valid_sampling_point,
    pid=valid_plant_id,
    soil=valid_soil,
    temp=valid_temperature,
    bad_hum=out_of_range_humidity,
    light=valid_light,
)
@settings(max_examples=300)
def test_out_of_range_humidity_reports_correct_sensor(
    ts, sid, sp, pid, soil, temp, bad_hum, light
):
    """
    Property 5c: humidity outside [0.0, 100.0] produces sensor='humidity'.

    **Validates: Requirements 2.11**
    """
    fields = make_valid_fields(ts, sid, sp, pid, soil, temp, bad_hum, light)
    result = validate_record(fields)

    assert isinstance(result, list), (
        f"Expected list[ValidationError], got {type(result).__name__}"
    )
    assert len(result) == 1, f"Expected exactly 1 error, got {len(result)}"
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == "humidity", (
        f"Expected sensor='humidity', got {result[0].sensor!r} for humidity={bad_hum}"
    )


@given(
    ts=valid_timestamp,
    sid=valid_session_id,
    sp=valid_sampling_point,
    pid=valid_plant_id,
    soil=valid_soil,
    temp=valid_temperature,
    hum=valid_humidity,
    bad_light=out_of_range_light,
)
@settings(max_examples=300)
def test_out_of_range_light_reports_correct_sensor(
    ts, sid, sp, pid, soil, temp, hum, bad_light
):
    """
    Property 5d: light outside [0.0, 1000.0] produces sensor='light'.

    **Validates: Requirements 2.12**
    """
    fields = make_valid_fields(ts, sid, sp, pid, soil, temp, hum, bad_light)
    result = validate_record(fields)

    assert isinstance(result, list), (
        f"Expected list[ValidationError], got {type(result).__name__}"
    )
    assert len(result) == 1, f"Expected exactly 1 error, got {len(result)}"
    assert isinstance(result[0], ValidationError)
    assert result[0].sensor == "light", (
        f"Expected sensor='light', got {result[0].sensor!r} for light={bad_light}"
    )


# ============================================================
# Property 6 — All-valid fields → correctly typed dict
# **Validates: Requirements 2.13**
# ============================================================

EXPECTED_KEYS = {
    "timestamp", "session_id", "sampling_point", "plant_id",
    "soil", "temperature", "humidity", "light",
}

EXPECTED_TYPES = {
    "timestamp":      str,
    "session_id":     int,
    "sampling_point": int,
    "plant_id":       int,
    "soil":           float,
    "temperature":    float,
    "humidity":       float,
    "light":          float,
}


@given(fields=valid_8_fields())
@settings(max_examples=500)
def test_all_valid_fields_returns_correctly_typed_dict(fields):
    """
    Property 6: When every field is valid, validate_record returns a dict with
    exactly the expected keys and correct Python types.

    **Validates: Requirements 2.13**
    """
    result = validate_record(fields)

    assert isinstance(result, dict), (
        f"Expected dict for valid input, got {type(result).__name__}"
    )

    # Exact key set — no extras, no missing
    assert set(result.keys()) == EXPECTED_KEYS, (
        f"Key mismatch: expected {EXPECTED_KEYS}, got {set(result.keys())}"
    )

    # Correct types for every key
    for key, expected_type in EXPECTED_TYPES.items():
        assert isinstance(result[key], expected_type), (
            f"Key '{key}': expected {expected_type.__name__}, "
            f"got {type(result[key]).__name__} (value={result[key]!r})"
        )


# ============================================================
# Property 7 — Fail-fast: at most one ValidationError per call
# **Validates: Requirements 2.14**
# ============================================================

@given(fields=st.lists(st.text()))
@settings(max_examples=500)
def test_at_most_one_validation_error_returned(fields):
    """
    Property 7: For any list[str] input, validate_record returns either a dict
    or a list[ValidationError] of length at most 1.

    **Validates: Requirements 2.14**
    """
    result = validate_record(fields)

    if isinstance(result, list):
        assert len(result) <= 1, (
            f"Expected at most 1 ValidationError, got {len(result)}"
        )
        if result:
            assert isinstance(result[0], ValidationError), (
                f"List element is not a ValidationError: {type(result[0]).__name__}"
            )
    else:
        assert isinstance(result, dict), (
            f"Expected dict or list, got {type(result).__name__}"
        )


# ============================================================
# Property 8 — validate_record never raises
# **Validates: Requirements 2.15**
# ============================================================

@given(fields=st.lists(st.text()))
@settings(max_examples=500)
def test_validate_record_never_raises(fields):
    """
    Property 8: validate_record never raises an exception for any list[str] input,
    including empty lists and lists of arbitrary length.

    **Validates: Requirements 2.15**
    """
    try:
        result = validate_record(fields)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"validate_record raised {type(exc).__name__}: {exc!r} "
            f"for input {fields!r}"
        ) from exc

    # Result must always be dict or list (smoke check)
    assert isinstance(result, (dict, list)), (
        f"Unexpected return type {type(result).__name__}"
    )
