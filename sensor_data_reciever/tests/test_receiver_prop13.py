import csv, io, sys, os
from unittest.mock import patch, MagicMock
import hypothesis.strategies as st
from hypothesis import given, settings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import receiver


valid_timestamp = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), max_codepoint=127),
    min_size=1,
)
invalid_session_id = st.integers(max_value=0)
valid_sampling_point = st.integers(min_value=1, max_value=6)
valid_plant_id = st.integers(min_value=1, max_value=16)
valid_node_id = st.sampled_from(["NODE_01", "NODE_02", "NODE_03", "NODE_04"])
valid_soil = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
valid_temperature = st.floats(min_value=-40.0, max_value=80.0, allow_nan=False, allow_infinity=False)
valid_humidity = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
valid_light = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


def build_csv_line(timestamp, session_id, sampling_point, plant_id,
                   node_id, soil, temperature, humidity, light):
    buf = io.StringIO()
    csv.writer(buf).writerow([timestamp, session_id, sampling_point, plant_id,
                              node_id, soil, temperature, humidity, light])
    return buf.getvalue().rstrip("\r\n")


@given(
    timestamp=valid_timestamp,
    session_id=invalid_session_id,
    sampling_point=valid_sampling_point,
    plant_id=valid_plant_id,
    node_id=valid_node_id,
    soil=valid_soil,
    temperature=valid_temperature,
    humidity=valid_humidity,
    light=valid_light,
)
@settings(max_examples=200)
def test_invalid_serial_line_calls_log_error_exactly_once(
    timestamp, session_id, sampling_point, plant_id,
    node_id, soil, temperature, humidity, light,
):
    """Property 13: invalid line calls log_error exactly once. Validates: Requirements 5.6"""
    receiver.record_counter = 0
    line = build_csv_line(timestamp, session_id, sampling_point, plant_id,
                          node_id, soil, temperature, humidity, light)
    mock_storage = MagicMock()
    with patch.object(receiver, "storage", mock_storage), patch("builtins.print") as mock_print:
        receiver.process_line(line)
    mock_storage.log_error.assert_called_once()
    mock_storage.save_valid_record.assert_not_called()
    texts = [str(c) for c in mock_print.call_args_list]
    invalid_msgs = [t for t in texts if "[INVALID]" in t and "R000001" in t]
    assert len(invalid_msgs) == 1, f"Expected 1 [INVALID] R000001 print, got: {texts}"
    valid_msgs = [t for t in texts if "[VALID]" in t]
    assert not valid_msgs, f"Expected no [VALID] prints, got: {valid_msgs}"
