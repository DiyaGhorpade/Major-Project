import pytest

import config
from validator import validate_record


def test_num_nodes_is_configured_and_valid():
    assert hasattr(config, "NUM_NODES")
    assert isinstance(config.NUM_NODES, int)
    assert config.NUM_NODES > 0
    assert config.validate_config() is True


@pytest.mark.parametrize(
    "node_id, expected",
    [
        ("NODE_01", True),
        ("NODE_02", True),
        ("NODE_04", True),
        ("NODE_05", False),
        ("", False),
        ("NODE_0", False),
        ("node_01", False),
        ("NODE_1", False),
    ],
)
def test_node_id_format_and_range(node_id, expected):
    assert config.is_valid_node_id(node_id) is expected


def test_valid_record_accepts_node_id_and_rejects_missing_or_invalid_node_id():
    valid_fields = [
        "2024-01-01T10:00:00",
        "1",
        "1",
        "1",
        "NODE_01",
        "50.0",
        "20.0",
        "50.0",
        "500.0",
    ]
    valid_result = validate_record(valid_fields)
    assert isinstance(valid_result, dict)
    assert valid_result["node_id"] == "NODE_01"

    missing_fields = valid_fields.copy()
    missing_fields[4] = ""
    missing_result = validate_record(missing_fields)
    assert isinstance(missing_result, list)
    assert missing_result[0].sensor == "node_id"

    invalid_fields = valid_fields.copy()
    invalid_fields[4] = "NODE_05"
    invalid_result = validate_record(invalid_fields)
    assert isinstance(invalid_result, list)
    assert invalid_result[0].sensor == "node_id"
