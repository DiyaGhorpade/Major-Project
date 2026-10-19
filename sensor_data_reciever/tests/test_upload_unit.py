"""HTTP ingestion regressions, replacing the retired process_line API tests."""
import csv
import io
from unittest.mock import patch

import pytest
import receiver
from .helpers import read_rows


@pytest.mark.parametrize("body", [
    b"", b"  ", b"\xff", b"123,7,2", b"123,7,2,4,4,45,23,60",
    b"123,7,2,4,4,45,23,60,512,extra",
    b'"123,7,2,4,4,45,23,60,512',
    b'"123"oops,7,2,4,4,45,23,60,512',
    b'"123\n456",7,2,4,4,45,23,60,512',
    b"123,7,2,4,4,45,23,60,512\n124,7,1,1,1,45,23,60,512",
    b"123,7,2,4,4,45,23,60,512\r124,7,1,1,1,45,23,60,512",
    b"timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light",
])
def test_malformed_body_is_400_without_storage_and_next_upload_works(
        client, output, valid_fields, body):
    before = output.read_bytes()
    with patch.object(receiver, "uuid4") as identity:
        assert client.post("/upload", data=body, content_type="text/plain").status_code == 400
        identity.assert_not_called()
    assert output.read_bytes() == before
    assert client.post("/upload", data=",".join(valid_fields)).status_code == 200
    assert read_rows(output)[1][3:] == valid_fields


@pytest.mark.parametrize("index,value", [
    (0, ""), (0, "   "), (1, "abc"), (1, "0"),
    (2, ""), (2, "3"), (2, "NODE_01"), (2, "01"),
    (3, "0"), (3, "13"), (4, "17"), (4, "1.5"),
    (5, "-1"), (5, "101"), (6, "81"), (7, "101"), (8, "100001"),
] + [(index, value) for index in range(5, 9)
     for value in ("bad", "nan", "NaN", "inf", "-inf", "1e999")])
def test_invalid_values_never_reach_storage(client, output, valid_fields, index, value):
    valid_fields[index] = value
    before = output.read_bytes()
    response = client.post("/upload", data=",".join(valid_fields))
    assert response.status_code == 400
    assert output.read_bytes() == before


@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
def test_quoted_csv_and_serial_line_endings(client, output, valid_fields, ending):
    stream = io.StringIO(newline="")
    csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator=ending).writerow(valid_fields)
    assert client.post("/upload", data=stream.getvalue()).status_code == 200
    assert read_rows(output)[1][3:] == valid_fields


def test_health_is_liveness_only(client, output):
    before = output.read_bytes()
    assert client.get("/health").status_code == 200
    assert client.get("/upload").status_code == 405
    assert output.read_bytes() == before
