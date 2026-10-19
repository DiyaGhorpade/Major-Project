"""Real sketch generation -> Flask validation -> durable CSV, including sockets."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler
from uuid import UUID

import pytest
import receiver
from .helpers import read_rows

EXPECTED_HEADER = [
    "record_id", "received_at", "accepted_at", "timestamp", "session_id", "node_id",
    "sampling_point", "plant_id", "soil", "temperature", "humidity", "light",
]


def good_scan(scans):
    result = []
    assert set(scans) == {"1", "2"}
    for node, count in (("1", 12), ("2", 4)):
        assert len(scans[node]) == count * 2  # One valid and one sensor-failure scan.
        rows = scans[node][:count]
        assert [row.split(",")[2] for row in rows] == [node] * count
        assert [row.split(",")[3:5] for row in rows] == [[str(i), str(i)] for i in range(1, count + 1)]
        result.extend(rows)
    return result


def assert_stored(path, bodies):
    rows = read_rows(path)
    assert rows[0] == EXPECTED_HEADER
    assert [row[3:] for row in rows[1:]] == [body.split(",") for body in bodies]
    assert len({row[0] for row in rows[1:]}) == len(bodies)
    for row in rows[1:]:
        assert len(row) == len(EXPECTED_HEADER)
        assert UUID(row[0]).version == 4
        received, accepted = (datetime.fromisoformat(value) for value in row[1:3])
        assert received.utcoffset().total_seconds() == 0
        assert accepted.utcoffset().total_seconds() == 0
        assert received <= accepted


@pytest.mark.parametrize("simulation", [True, False])
def test_real_sketches_through_real_validator_and_storage(
        generated_records, simulation, client, output, capsys):
    scans = generated_records[simulation]
    bodies = good_scan(scans)
    for body in bodies:
        assert client.post("/upload", data=body, content_type="text/plain").status_code == 200
    assert_stored(output, bodies)
    before = output.read_bytes()
    for node, count in (("1", 12), ("2", 4)):
        for bad_body in scans[node][count:]:
            assert client.post("/upload", data=bad_body, content_type="text/plain").status_code == 400
        assert f"node={node!r}" in capsys.readouterr().out
    assert output.read_bytes() == before


def http_request(base_url, body=None):
    request = Request(base_url + ("/health" if body is None else "/upload"),
                      data=None if body is None else body.encode("utf-8"),
                      headers={"Content-Type": "text/plain"})
    # Ignore machine proxy configuration; the test must stay on loopback.
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
            return response.status, response.read()
    except HTTPError as response:
        return response.code, response.read()


def test_generated_scan_over_real_http_socket(generated_records, live_receiver, output):
    bodies = good_scan(generated_records[True])
    assert http_request(live_receiver) == (200, b"OK\n")
    for index, body in enumerate(bodies, 1):
        assert http_request(live_receiver, body) == (200, b"OK\n")
        # Data is readable from disk by the time each acknowledgement arrives.
        assert_stored(output, bodies[:index])
    before = output.read_bytes()
    assert http_request(live_receiver, "malformed")[0] == 400
    assert output.read_bytes() == before
    assert http_request(live_receiver, bodies[0])[0] == 200
    assert_stored(output, bodies + [bodies[0]])


def test_simultaneous_nodes_and_initial_file_creation(generated_records, live_receiver, output):
    bodies = good_scan(generated_records[True]) + good_scan(generated_records[False])
    output.unlink()  # Exercise simultaneous first uploads against a missing file.
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda body: http_request(live_receiver, body), bodies))
    assert responses == [(200, b"OK\n")] * 32
    rows = read_rows(output)
    assert rows[0] == EXPECTED_HEADER
    assert len(rows) == 33
    assert sorted(row[3:] for row in rows[1:]) == sorted(body.split(",") for body in bodies)
    assert len({row[0] for row in rows[1:]}) == 32
    assert all(len(row) == len(EXPECTED_HEADER) for row in rows)


def test_restart_appends_without_a_second_header(client, output, valid_fields):
    body = ",".join(valid_fields)
    assert client.post("/upload", data=body).status_code == 200
    receiver._init_csv()  # Same initialization performed at process startup.
    assert receiver.app.test_client().post("/upload", data=body).status_code == 200
    assert_stored(output, [body, body])
