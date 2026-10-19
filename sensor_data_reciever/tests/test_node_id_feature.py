"""Claimed node identity is retained in accepted and rejected records."""
import pytest
from validator import validate_record
from .helpers import read_rows


@pytest.mark.parametrize("node", ["1", "2"])
def test_same_local_ids_are_distinct_between_nodes(client, output, valid_fields, node):
    valid_fields[2:5] = [node, "1", "1"]
    assert client.post("/upload", data=",".join(valid_fields)).status_code == 200
    row = dict(zip(read_rows(output)[0], read_rows(output)[1]))
    assert (row["node_id"], row["plant_id"], row["sampling_point"]) == (node, "1", "1")


@pytest.mark.parametrize("node", ["1", "2", "3", "NODE_01"])
def test_rejection_logs_claimed_node(client, valid_fields, capsys, node):
    valid_fields[2] = node
    valid_fields[5] = "bad"
    assert client.post("/upload", data=",".join(valid_fields)).status_code == 400
    assert f"node={node!r}" in capsys.readouterr().out


@pytest.mark.parametrize("body,node", [("123,7,2", "2"), ("123,7", "unknown"),
                                       ("123,7,,4,4,45,23,60,512", "unknown"),
                                       ('"unterminated', "unknown")])
def test_structural_rejection_reports_source_when_recoverable(client, capsys, body, node):
    assert client.post("/upload", data=body).status_code == 400
    assert f"node={node!r}" in capsys.readouterr().out


def test_validator_error_keeps_node_for_wrong_count(valid_fields):
    error = validate_record(valid_fields[:-1])[0]
    assert error.node_id == "2"


def test_current_bounds_are_preserved_not_replaced_with_wiring_rules(valid_fields):
    # Existing contract permits sensor IDs 1-12 and plant IDs 1-16 on either node.
    valid_fields[2:5] = ["2", "12", "16"]
    assert isinstance(validate_record(valid_fields), dict)
