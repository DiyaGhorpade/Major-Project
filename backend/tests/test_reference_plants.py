"""Reference (healthy-control) plant readings sent for the ML cross-plant features (ML issue #5)."""

from datetime import datetime, timedelta
from types import SimpleNamespace

from app.integrations.ml_types import MetadataResponse
from app.services import reference_builder
from app.services.reference_builder import MARGIN, ReferenceIndex
from tests.fakes import FakeML
from tests.test_pipeline import TOTAL, running_app, wait_done

CROSS = {"enabled": True, "reference_plants": [1, 2, 3, 4], "tolerance_minutes": 30}


def _md(**extra):
    return MetadataResponse.model_validate({"run_id": "r", "model_version": "1.0.0", **extra})


def test_reference_plants_from_metadata():
    assert reference_builder.reference_plants(_md(cross_plant=CROSS)) == [1, 2, 3, 4]
    assert reference_builder.reference_plants(_md(cross_plant={**CROSS, "enabled": False})) == []
    assert reference_builder.reference_plants(_md()) == []  # older ML service: nothing is sent
    assert reference_builder.reference_plants(_md(cross_plant={"enabled": True, "reference_plants": ["x"]})) == []


def _row(i, plant, t0=datetime(2026, 10, 1)):
    return SimpleNamespace(id=i, plant_id=plant, timestamp=t0 + timedelta(hours=2 * i))


def test_reference_index_windows_merge_and_do_not_duplicate():
    rows = [_row(i, p) for i in range(20) for p in (1, 2)]
    idx = ReferenceIndex(rows, lambda r: (r.plant_id, r.timestamp))
    t = [r.timestamp for r in rows[::2]]
    got = idx.for_series([(t[2], t[4]), (t[3], t[6]), (t[15], t[16])])
    times = [g[1] for g in got]
    assert len(got) == len(set(got))  # overlapping spans are merged, never duplicated
    assert min(times) >= t[2] - MARGIN and max(times) <= t[16]
    assert not any(t[6] < x < t[15] - MARGIN for x in times)  # the gap between disjoint spans is not sent
    assert idx.for_series([]) == []


def test_run_sends_reference_readings_covering_each_item(real_settings):
    fake = FakeML()
    fake.cross_plant = CROSS
    with running_app(real_settings, fake) as client:
        client.post("/api/pipeline/run")
        assert wait_done(client)["state"] == "completed"
    assert len(fake.seen) == TOTAL
    for body in fake.batch_payloads:
        ref = body["reference"]
        assert ref and {r["plant_id"] for r in ref} <= {1, 2, 3, 4}
        stamps = sorted(datetime.fromisoformat(r["timestamp"]) for r in ref)
        for item in body["items"]:
            end = datetime.fromisoformat(item["reading"]["timestamp"])
            first = datetime.fromisoformat(item["history"][0]["timestamp"]) if item["history"] else end
            assert stamps[0] <= first  # the window reaches back to the first history row
            assert any(first - MARGIN <= s <= end for s in stamps)  # readings at or before the item are included
            assert any(end - timedelta(minutes=30) <= s <= end for s in stamps)  # incl. the concurrent slot


def test_run_without_cross_plant_metadata_sends_no_reference(real_settings):
    fake = FakeML()
    with running_app(real_settings, fake) as client:
        client.post("/api/pipeline/run")
        assert wait_done(client)["state"] == "completed"
    assert fake.batch_payloads and all("reference" not in b for b in fake.batch_payloads)


def test_on_demand_prediction_sends_reference_up_to_the_reading(real_settings):
    fake = FakeML()
    fake.cross_plant = CROSS
    with running_app(real_settings, fake) as client:
        r = client.post("/api/predictions", json={"plantId": 9, "explain": False})
        assert r.status_code == 200, r.text
    req = fake.seen[-1]
    assert req.reference and {x.plant_id for x in req.reference} <= {1, 2, 3, 4}
    assert all(x.timestamp <= req.reading.timestamp for x in req.reference)
    assert min(x.timestamp for x in req.reference) <= req.history[0].timestamp
