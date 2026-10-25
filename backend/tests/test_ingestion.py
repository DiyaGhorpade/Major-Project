"""POST /api/sensors/readings: validation, flagging, dedupe, batches, best-effort prediction."""

from datetime import datetime, timedelta

import pytest

T0 = datetime.fromisoformat("2027-01-01T00:00:00+05:30")  # after the dataset (ends 2026-10-18)


def reading(i: int = 0, plant: int = 1, **over) -> dict:
    body = {
        "timestamp": (T0 + timedelta(hours=2 * i)).isoformat(), "plant_id": plant, "session_id": 4, "node_id": "1" if plant % 2 else "2",
        "sampling_point": 3, "soil": 41.5, "temperature": 24.0, "humidity": 70.0, "light": 800.0,
    }  # fmt: skip
    body.update(over)
    return body


def total_readings(client) -> int:
    return int(client.get("/api/sensors/readings?limit=1").headers["x-total-count"])


def assert_envelope(response, status=422):
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert set(error) == {"code", "message", "details"}
    return error


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"timestamp": "2027-01-01T00:00:00+05:30"},  # no plant
        reading(plant=99),
        reading(plant=0),
        reading(soil="wet"),
        reading(light="inf"),
        reading(node_id="7"),
        reading(timestamp="yesterday"),
        reading(timestamp=12345),
        reading(sampling_point=13),
        {"readings": []},
        {"readings": [reading(), {"plant_id": 2}]},
        [reading()],
        "text",
    ],
)
def test_invalid_payload_is_422_envelope_and_stores_nothing(fresh_client, body):
    before = total_readings(fresh_client)
    error = assert_envelope(fresh_client.post("/api/sensors/readings", json=body))
    assert error["code"] == "validation_error"
    assert total_readings(fresh_client) == before


def test_non_finite_numbers_in_raw_json_are_422(fresh_client):
    for literal in ("1e999", "NaN", "Infinity"):
        raw = ('{"timestamp": "2027-01-01T00:00:00+05:30", "plant_id": 1, "soil": %s}' % literal).encode()
        r = fresh_client.post("/api/sensors/readings", content=raw, headers={"content-type": "application/json"})
        assert_envelope(r)


def test_malformed_json_is_422(fresh_client):
    r = fresh_client.post("/api/sensors/readings", content=b"{not json", headers={"content-type": "application/json"})
    assert_envelope(r)


def test_single_reading_is_accepted(fresh_client):
    fresh_client.fake.up = False
    r = fresh_client.post("/api/sensors/readings", json=reading())
    assert r.status_code == 201
    body = r.json()
    assert body["accepted"] == 1 and body["flagged"] == 0 and body["duplicates"] == 0
    item = body["items"][0]
    assert item["status"] == "accepted" and item["record_id"].startswith("R-") and item["plant_id"] == 1
    assert item["timestamp"] == "2027-01-01T00:00:00+05:30"
    stored = fresh_client.get("/api/sensors/readings?plantId=1&limit=1").json()[0]
    assert stored["record_id"] == item["record_id"] and stored["status"] == "accepted"
    assert stored["received_at"] and stored["accepted_at"]  # ingested rows know their receipt time
    assert stored["datasetLabels"] == {"stressLevel": None, "stressType": None, "isSyntheticAnomaly": None}


def test_node_comes_from_the_plant_mapping_when_absent(fresh_client):
    fresh_client.fake.up = False
    r = fresh_client.post("/api/sensors/readings", json=reading(plant=2, node_id=None))
    assert r.status_code == 201
    assert fresh_client.get("/api/sensors/readings?plantId=2&limit=1").json()[0]["node_id"] == "2"
    r = fresh_client.post("/api/sensors/readings", json={"timestamp": "2027-01-02T00:00:00", "plant_id": 3, "soil": 10, "temperature": 20, "humidity": 50, "light": 5})
    assert r.status_code == 201
    row = fresh_client.get("/api/sensors/readings?plantId=3&limit=1").json()[0]
    assert row["node_id"] == "1" and row["session_id"] is None and row["timestamp"] == "2027-01-02T00:00:00+05:30"  # naive = IST


def test_out_of_range_and_missing_values_are_flagged_not_rejected(fresh_client):
    fresh_client.fake.up = False
    body = {"readings": [reading(0, soil=150.0), reading(1, light=None, humidity=None), reading(2, temperature=-5.0)]}
    r = fresh_client.post("/api/sensors/readings", json=body)
    assert r.status_code == 201
    out = r.json()
    assert (out["accepted"], out["flagged"], out["duplicates"]) == (0, 3, 0)
    rows = {x["timestamp"]: x for x in fresh_client.get("/api/sensors/readings?plantId=1&status=flagged&limit=1000&order=asc").json() if x["timestamp"].startswith("2027")}
    assert len(rows) == 3
    first, second, third = (rows[k] for k in sorted(rows))
    assert "out_of_range: soil=150.0" in first["validation_error"] and first["soil"] == 150.0  # value kept, not clipped
    assert "missing: humidity, light" in second["validation_error"] and second["light"] is None
    assert "temperature=-5.0" in third["validation_error"]


def test_duplicates_are_reported_and_not_inserted(fresh_client):
    fresh_client.fake.up = False
    before = total_readings(fresh_client)
    first = fresh_client.post("/api/sensors/readings", json=reading())
    assert first.status_code == 201
    again = fresh_client.post("/api/sensors/readings", json=reading(soil=99.0))
    assert again.status_code == 200  # nothing new
    assert again.json()["duplicates"] == 1 and again.json()["items"][0] == {
        "record_id": None, "plant_id": 1, "timestamp": "2027-01-01T00:00:00+05:30", "status": "duplicate"
    }  # fmt: skip
    # the same instant written with another offset is still the same reading
    utc = fresh_client.post("/api/sensors/readings", json=reading(timestamp="2026-12-31T18:30:00Z"))
    assert utc.json()["duplicates"] == 1
    # and so is a row that already came from the dataset
    seeded = fresh_client.get("/api/sensors/readings?plantId=1&order=asc&limit=1").json()[0]
    dup = fresh_client.post("/api/sensors/readings", json=reading(timestamp=seeded["timestamp"]))
    assert dup.status_code == 200 and dup.json()["duplicates"] == 1
    assert total_readings(fresh_client) == before + 1
    stored = fresh_client.get("/api/sensors/readings?plantId=1&limit=1").json()[0]
    assert stored["soil"] == 41.5  # the first value won


def test_mixed_batch(fresh_client):
    fresh_client.fake.up = False
    seeded = fresh_client.get("/api/sensors/readings?plantId=4&order=asc&limit=1").json()[0]
    body = {"readings": [
        reading(0, plant=1),                                   # accepted
        reading(1, plant=1, soil=140),                         # flagged
        reading(0, plant=1),                                   # duplicate within the batch
        reading(0, plant=2, node_id=None),                     # accepted (other plant, same time)
        reading(timestamp=seeded["timestamp"], plant=4),       # duplicate of a stored row
    ]}  # fmt: skip
    r = fresh_client.post("/api/sensors/readings", json=body)
    assert r.status_code == 201
    out = r.json()
    assert (out["accepted"], out["flagged"], out["duplicates"]) == (2, 1, 2)
    assert [i["status"] for i in out["items"]] == ["accepted", "flagged", "duplicate", "accepted", "duplicate"]
    assert [i["record_id"] is None for i in out["items"]] == [False, False, True, False, True]


def test_batch_limit_is_500(fresh_client):
    fresh_client.fake.up = False
    too_many = {"readings": [reading(i) for i in range(501)]}
    error = assert_envelope(fresh_client.post("/api/sensors/readings", json=too_many))
    assert any("500" in str(d) for d in error["details"])
    assert total_readings(fresh_client) == 5376
    ok = fresh_client.post("/api/sensors/readings", json={"readings": [reading(i) for i in range(500)]})
    assert ok.status_code == 201 and ok.json()["accepted"] == 500
    assert total_readings(fresh_client) == 5876


def test_session_counts_follow_ingestion(fresh_client):
    fresh_client.fake.up = False
    before = {s["session_id"]: s for s in fresh_client.get("/api/sessions").json()}
    fresh_client.post("/api/sensors/readings", json={"readings": [reading(0), reading(1, soil=150)]})
    after = {s["session_id"]: s for s in fresh_client.get("/api/sessions").json()}
    assert after[4]["received_records"] == before[4]["received_records"] + 2
    assert after[4]["flagged_records"] == before[4]["flagged_records"] + 1
    assert after[3] == before[3]  # other sessions untouched
    assert after[4]["node_progress"]["1"]["received"] == before[4]["node_progress"]["1"]["received"] + 2


def test_unknown_session_is_flagged_and_stored_without_a_session(fresh_client):
    fresh_client.fake.up = False
    r = fresh_client.post("/api/sensors/readings", json=reading(session_id=9))
    assert r.status_code == 201 and r.json()["flagged"] == 1
    row = fresh_client.get("/api/sensors/readings?plantId=1&limit=1").json()[0]
    assert row["session_id"] is None and "unknown_session: 9" in row["validation_error"]


def test_ml_down_still_201_and_reading_is_stored(fresh_client):
    fresh_client.fake.up = False
    r = fresh_client.post("/api/sensors/readings", json=reading())
    assert r.status_code == 201 and r.json()["accepted"] == 1
    assert fresh_client.get("/api/sensors/readings?plantId=1&limit=1").json()[0]["timestamp"] == "2027-01-01T00:00:00+05:30"
    assert fresh_client.get("/api/predictions").json() == []
    assert fresh_client.get("/api/health").json()["db"]["readings"] == 5377


def test_ml_up_stores_a_prediction_for_the_new_reading(fresh_client):
    r = fresh_client.post("/api/sensors/readings", json=reading(soil=20.0))
    assert r.status_code == 201
    record_id = r.json()["items"][0]["record_id"]
    preds = fresh_client.get("/api/predictions").json()  # background task has run by the time the call returns
    assert [p["record_id"] for p in preds] == [record_id]
    assert preds[0]["status"] == "ok" and preds[0]["severity"] and preds[0]["plant_id"] == 1
    # it is the plant's latest reading, so it got a local explanation and drives the plant's state
    plant = fresh_client.get("/api/plants/1").json()
    assert plant["latestReading"]["record_id"] == record_id
    assert plant["conditionPrediction"]["recordId"] == record_id
    assert fresh_client.get("/api/ai/shap/1").json()["record_id"] == record_id


def test_ml_going_away_midway_does_not_break_the_response(fresh_client):
    fresh_client.fake.up = False
    r = fresh_client.post("/api/sensors/readings", json={"readings": [reading(i) for i in range(5)]})
    assert r.status_code == 201 and r.json()["accepted"] == 5
    assert fresh_client.get("/api/predictions").json() == []


def test_overview_cache_is_invalidated_by_ingestion(fresh_client):
    before = fresh_client.get("/api/monitoring/overview").json()["totals"]["readings"]
    assert fresh_client.get("/api/monitoring/overview").json()["totals"]["readings"] == before  # cached
    fresh_client.fake.up = False
    fresh_client.post("/api/sensors/readings", json=reading())
    assert fresh_client.get("/api/monitoring/overview").json()["totals"]["readings"] == before + 1
