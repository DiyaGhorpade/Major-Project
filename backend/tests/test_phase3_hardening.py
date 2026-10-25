"""Audit fixes: neutral catalog copy, ingestion edge cases, shared sensor-value rule, JSON sanitising, relative paths,
pagination of the small lists, and cache invalidation."""

import json
import re
from datetime import timedelta

import pytest

from app.core.plant_catalog import PLANT_IDS, catalog_entry
from app.db import repositories as repo
from app.db.models import SensorReading
from app.integrations.ml_types import PredictRequest, PredictResponse
from app.services import monitoring_service, sensor_service
from app.services.prediction_service import Outcome, persist_prediction, to_ml_reading
from app.services.sensor_values import flagged_out_of_range, valid_value, valid_values
from tests.conftest import make_api_settings, open_app, run_pipeline
from tests.fakes import FakeML, make_response
from tests.helpers import feature_names, write_shap_report
from tests.test_endpoints import RUN_ID
from tests.test_ingestion import T0, reading, total_readings

VERDICT_WORDS = re.compile(
    r"healthy|thriv|optimal|stressed|wilt|vibrant|no action needed|perfect|dying|sick|diseas|doing well|unhealthy|robust|flourish|strong",
    re.I,
)


def strict_json(text: str):
    """Parse JSON refusing NaN / Infinity (what a browser's JSON.parse would reject)."""

    def refuse(token):
        raise AssertionError(f"non-finite number in response: {token}")

    return json.loads(text, parse_constant=refuse)


# --- catalog ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("plant_id", PLANT_IDS)
def test_catalog_copy_makes_no_condition_claims(plant_id):
    entry = catalog_entry(plant_id)
    for field in (entry.name, entry.description, entry.simple_care_tip, entry.location, entry.variety, entry.species):
        assert not VERDICT_WORDS.search(field), f"plant {plant_id}: {field!r}"
    assert entry.description and entry.simple_care_tip


def test_catalog_copy_is_condition_independent():
    """Same group -> same role/tip text; the copy must not differ by plant in a way that implies a state."""
    tips = {catalog_entry(p).group.cluster: set() for p in PLANT_IDS}
    for p in PLANT_IDS:
        tips[catalog_entry(p).group.cluster].add(catalog_entry(p).simple_care_tip)
    assert all(len(v) == 1 for v in tips.values())


def test_api_plants_serve_neutral_copy(fresh_client):
    for plant in fresh_client.get("/api/plants").json():
        for key in ("name", "description", "simpleCareTip", "location"):
            assert not VERDICT_WORDS.search(plant[key]), (plant["id"], key, plant[key])


# --- ingestion --------------------------------------------------------------------------------------
def test_explicit_node_that_disagrees_with_the_plant_is_flagged_and_kept(fresh_client):
    fresh_client.fake.up = False
    r = fresh_client.post("/api/sensors/readings", json=reading(plant=1, node_id="2"))
    assert r.status_code == 201 and r.json()["flagged"] == 1
    row = fresh_client.get("/api/sensors/readings?plantId=1&limit=1").json()[0]
    assert row["node_id"] == "2" and row["status"] == "flagged"  # kept as sent
    assert "node_mismatch: expected 1" in row["validation_error"]
    ok = fresh_client.post("/api/sensors/readings", json=reading(1, plant=2, node_id=2))
    assert ok.json()["accepted"] == 1  # numeric 2 normalises to "2" and matches plant 2


def test_node_mismatch_combines_with_other_problems(fresh_client):
    fresh_client.fake.up = False
    fresh_client.post("/api/sensors/readings", json=reading(plant=3, node_id="2", soil=130.0, session_id=77))
    row = fresh_client.get("/api/sensors/readings?plantId=3&limit=1").json()[0]
    for part in ("out_of_range: soil=130.0", "unknown_session: 77", "node_mismatch: expected 1"):
        assert part in row["validation_error"]


def test_integrity_error_that_is_not_a_duplicate_is_not_reported_as_one(fresh_client):
    fresh_client.fake.up = False
    with fresh_client.sf() as db:  # a constraint other than (plant_id, timestamp)
        db.execute(__import__("sqlalchemy").text(
            "CREATE TRIGGER no_13 BEFORE INSERT ON sensor_readings WHEN NEW.soil = 13.37 BEGIN SELECT RAISE(ABORT, 'nope'); END"
        ))
        db.commit()
    from fastapi.testclient import TestClient

    quiet = TestClient(fresh_client.app, raise_server_exceptions=False)
    before = total_readings(fresh_client)
    r = quiet.post("/api/sensors/readings", json={"readings": [reading(0), reading(1, soil=13.37)]})
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    assert "duplicate" not in r.text
    assert total_readings(fresh_client) == before  # the whole request was rolled back


def test_a_real_concurrent_duplicate_is_reported_as_duplicate(fresh_client, monkeypatch):
    fresh_client.fake.up = False
    seeded = fresh_client.get("/api/sensors/readings?plantId=1&order=asc&limit=1").json()[0]
    real = repo.existing_reading_keys
    calls = []

    def blind_first_time(db, plant_ids, timestamps):  # the pre-check misses the row, as in a race
        calls.append(1)
        return set() if len(calls) == 1 else real(db, plant_ids, timestamps)

    monkeypatch.setattr(sensor_service.repo, "existing_reading_keys", blind_first_time)
    before = total_readings(fresh_client)
    r = fresh_client.post("/api/sensors/readings", json=reading(timestamp=seeded["timestamp"]))
    assert r.status_code == 200 and r.json()["duplicates"] == 1 and r.json()["accepted"] == 0
    assert total_readings(fresh_client) == before


def test_targets_are_computed_once_and_only_the_newest_reading_per_plant_explains(fresh_client):
    with fresh_client.sf() as db:
        result = sensor_service.ingest(db, [__import__("app.schemas.sensor", fromlist=["IngestReading"]).IngestReading(**reading(i, plant=p)) for p in (1, 2) for i in (0, 1, 2)])
    by_plant = {}
    for t in result.targets:
        by_plant.setdefault(t.plant_id, []).append(t)
    assert len(result.targets) == 6
    for targets in by_plant.values():
        assert [t.explain for t in sorted(targets, key=lambda t: t.reading_id)] == [False, False, True]


def test_only_affected_sessions_are_recounted(fresh_client, monkeypatch):
    fresh_client.fake.up = False
    seen = []
    real = repo.session_stats
    monkeypatch.setattr(sensor_service.repo, "session_stats", lambda db, ids=None: (seen.append(None if ids is None else list(ids)), real(db, ids))[1])
    fresh_client.post("/api/sensors/readings", json=reading(session_id=3))
    assert seen == [[3]]
    seen.clear()
    fresh_client.post("/api/sensors/readings", json=reading(1, session_id=None))
    assert seen == []  # no session -> no recount


def test_ml_dying_midway_keeps_the_response_and_every_reading(api_settings):
    fake = FakeML()
    fake.die_after_predicts = 2
    settings = api_settings.model_copy(update={"ML_CONCURRENCY": 1})  # one at a time: the cut-off point is exact
    with open_app(settings, fake) as c:
        body = {"readings": [reading(0, plant=p) for p in range(1, 7)]}
        r = c.post("/api/sensors/readings", json=body)
        assert r.status_code == 201 and r.json()["accepted"] == 6
        assert total_readings(c) == 5376 + 6  # all stored
        preds = c.get("/api/predictions").json()
        assert len(preds) == 2 and {p["status"] for p in preds} == {"ok"}  # the rest: not predicted, nothing stored
        predicted = {p["plant_id"] for p in preds}
        assert predicted == {1, 2}
        for plant in range(3, 7):
            assert c.get(f"/api/plants/{plant}").json()["conditionPrediction"] is None  # no prediction for the new reading
            assert c.get(f"/api/sensors/readings?plantId={plant}&limit=1").json()[0]["timestamp"] == "2027-01-01T00:00:00+05:30"
        assert fake.predict_calls == 2 and fake.up is False
        # health keeps working and a later Run could pick the readings up
        assert c.get("/api/health").status_code == 200


# --- shared sensor-value rule ----------------------------------------------------------------------------
def _reading(soil, error=None):
    return SensorReading(soil=soil, temperature=20.0, humidity=50.0, light=100.0, validation_error=error)


def test_valid_value_rule():
    ok, none, bad = _reading(40.0), _reading(None, "missing: soil"), _reading(150.0, "out_of_range: soil=150.0")
    assert valid_value(ok, "soil") == 40.0 and valid_value(none, "soil") is None and valid_value(bad, "soil") is None
    assert valid_value(bad, "temperature") == 20.0  # only the flagged metric is excluded
    assert valid_values([ok, none, bad, _reading(60.0)], "soil") == [40.0, 60.0]
    assert flagged_out_of_range(bad, "soil") and not flagged_out_of_range(bad, "light")


def test_cluster_averages_use_the_same_rule_as_recommendations(fresh_client):
    with fresh_client.sf() as db:
        baseline = {s.cluster: s.averages.soil for s in monitoring_service._cluster_summaries(db)}
        latest = {r.plant_id: r for r in repo.latest_reading_per_plant(db, [1, 2, 3, 4])}
        latest[1].soil, latest[1].validation_error = 900.0, "out_of_range: soil=900.0"  # would wreck the mean
        latest[2].soil = None
        expected = [r.soil for pid, r in latest.items() if pid in (3, 4) and r.soil is not None]
        db.flush()
        got = {s.cluster: s.averages.soil for s in monitoring_service._cluster_summaries(db)}
    assert got["Cluster 1"] == pytest.approx(sum(expected) / len(expected), abs=1e-3)
    assert got["Cluster 2"] == baseline["Cluster 2"]


# --- JSON sanitising ---------------------------------------------------------------------------------------
def test_global_shap_with_nan_and_inf_in_the_report_does_not_500(tmp_path):
    settings = make_api_settings(tmp_path)
    folder = write_shap_report(settings.ML_REPORTS_DIR, RUN_ID)
    names = feature_names()
    data = json.loads((folder / "shap_global_importance.json").read_text())
    data["mean_abs_shap"][names[0]] = float("nan")
    data["mean_abs_shap"][names[3]] = float("inf")
    data["per_class_top15"]["low"][names[1]] = float("nan")
    (folder / "shap_global_importance.json").write_text(json.dumps(data))  # Python writes NaN / Infinity literals
    assert "NaN" in (folder / "shap_global_importance.json").read_text()
    csv_path = folder / "shap_sample_explanations.csv"
    csv_path.write_text(csv_path.read_text() + f"9,low,{names[0]},nan,0.1,towards_predicted_class\n9,low,{names[0]},4.0,inf,towards_predicted_class\n9,low,{names[1]},inf,0.3,towards_predicted_class\n")
    with open_app(settings, FakeML()) as c:
        r = c.get("/api/ai/shap/global")
        assert r.status_code == 200
        g = strict_json(r.text)
        listed = [f["name"] for f in g["features"]]
        assert names[0] not in listed and names[3] not in listed and listed[0] == names[1]
        assert all(f["name"] != names[1] for f in g["perClassTop15"]["low"])
        pts = g["beeswarm_points"]
        assert all(p["shap_value"] == p["shap_value"] and abs(p["shap_value"]) != float("inf") for p in pts)
        nan_value = [p for p in pts if p["id"].startswith("9:") and p["feature"] == names[0]]
        assert [(p["feature_value_norm"], p["shap_value"]) for p in nan_value] == [(None, 0.1)]  # NaN value -> null, point kept
        inf_value = [p for p in pts if p["id"].startswith("9:") and p["feature"] == names[1]]
        assert [(p["feature_value_norm"], p["shap_value"]) for p in inf_value] == [(None, 0.3)]
        assert len([p for p in pts if p["id"].startswith("9:")]) == 2  # the inf-contribution point was dropped
        assert c.get("/api/ai/insights").status_code == 200


def test_local_shap_with_non_finite_values_is_sanitised(fresh_client):
    c = fresh_client
    with c.sf() as db:
        r = repo.latest_reading_for_plant(db, 1)
        base = make_response(PredictRequest(reading=to_ml_reading(r), history=[]), anomaly=False, explain=True)
        base["explanation"]["top_features"] += [
            {"feature": "light", "value": float("nan"), "contribution": float("nan"), "direction": "towards_predicted_class"},
            {"feature": "humidity", "value": float("inf"), "contribution": 0.2, "direction": "towards_predicted_class"},
        ]
        persist_prediction(db, r, Outcome("ok", PredictResponse.model_validate(base), history_rows_sent=60))
        db.commit()
    resp = c.get("/api/ai/shap/1")
    assert resp.status_code == 200
    features = {f["name"]: f for f in strict_json(resp.text)["features"]}
    assert "light" not in features  # non-finite contribution: dropped
    assert features["humidity"]["observed_value"] is None and features["humidity"]["shap_value"] == 0.2


# --- paths --------------------------------------------------------------------------------------------------
def test_no_absolute_server_paths_are_exposed(tmp_path):
    settings = make_api_settings(tmp_path)
    write_shap_report(settings.ML_REPORTS_DIR, RUN_ID)
    with open_app(settings, FakeML()) as c:
        g = c.get("/api/ai/shap/global").json()
        assert g["source"]["path"] == f"reports/{RUN_ID}/evaluation_20261025T085725Z/shap_global_importance.json"
        insights = c.get("/api/ai/insights").json()
        assert insights["topFeatures"]["globalSource"]["path"] == g["source"]["path"]
        assert not c.get("/api/settings").json()["storage_path"].startswith("/")
        dump = c.get("/api/ai/shap/global").text + c.get("/api/ai/insights").text + c.get("/api/settings").text
        assert str(tmp_path) not in dump and "/home/" not in dump
    broken = settings.model_copy(update={"ML_ARTIFACTS_DIR": tmp_path / "nothing"})
    with open_app(broken, FakeML()) as c:
        details = c.get("/api/models/performance").json()["error"]["details"]
        assert details["artifactsDir"] == "nothing" and str(tmp_path) not in json.dumps(details)


def test_repo_relative_path_when_inside_the_repository():
    from app.core.config import ML_DIR, display_path

    p = ML_DIR / "reports" / "r" / "x.json"
    assert display_path(p, ML_DIR / "reports") == "money_plant_ml_pipeline/reports/r/x.json"


# --- pagination of the small lists --------------------------------------------------------------------------
def test_plants_sessions_devices_are_paginated(fresh_client):
    c = fresh_client
    for path, count in (("/api/plants", 16), ("/api/sessions", 4), ("/api/devices", 3)):
        full = c.get(path)
        assert int(full.headers["x-total-count"]) == count and len(full.json()) == count  # defaults return everything
        page = c.get(f"{path}?limit=2&offset=1")
        assert int(page.headers["x-total-count"]) == count and len(page.json()) == min(2, count - 1)
        key = "id" if path != "/api/sessions" else "session_id"
        assert [x[key] for x in page.json()] == [x[key] for x in full.json()][1:3]
        assert c.get(f"{path}?limit=501").status_code == 422 and c.get(f"{path}?limit=0").status_code == 422
        assert c.get(f"{path}?offset={count}").json() == []
    page = c.get("/api/plants?limit=3&offset=14").json()
    assert [p["id"] for p in page] == [15, 16] and all(p["latestReading"] for p in page)


# --- cache invalidation -----------------------------------------------------------------------------------------
def test_overview_refreshes_when_a_run_completes(fresh_client):
    before = fresh_client.get("/api/monitoring/overview").json()
    assert fresh_client.get("/api/monitoring/overview").json()["totals"] == before["totals"]  # cached
    assert before["totals"]["predictions"]["total"] == 0 and before["topRecommendations"] == []
    run_pipeline(fresh_client)
    after = fresh_client.get("/api/monitoring/overview").json()
    assert after["totals"]["predictions"]["ok"] == 5376 and after["topRecommendations"]
    assert fresh_client.get("/api/ai/insights").json()["hasPredictions"] is True


def test_overview_refreshes_after_a_recommendation_patch(fresh_client):
    run_pipeline(fresh_client)
    top = fresh_client.get("/api/monitoring/overview").json()["topRecommendations"]
    assert top
    rec_id = top[0]["id"]
    assert fresh_client.get("/api/monitoring/overview").json()["topRecommendations"][0]["id"] == rec_id  # cached
    r = fresh_client.patch(f"/api/recommendations/{rec_id}", json={"status": "dismissed"})
    assert r.status_code == 200
    ids = [x["id"] for x in fresh_client.get("/api/monitoring/overview").json()["topRecommendations"]]
    assert rec_id not in ids
    detail = fresh_client.get(f"/api/monitoring/plants/{top[0]['plant_id']}").json()
    assert rec_id not in [x["id"] for x in detail["recommendations"]]


def test_overview_refreshes_after_a_rejected_on_demand_prediction(fresh_client):
    assert fresh_client.get("/api/monitoring/overview").json()["totals"]["predictions"]["rejected"] == 0
    fresh_client.fake.reject_if = lambda req: True
    assert fresh_client.post("/api/predictions", json={"plantId": 4}).status_code == 422
    totals = fresh_client.get("/api/monitoring/overview").json()["totals"]["predictions"]
    assert totals["rejected"] == 1 and totals["total"] == 1
    assert fresh_client.get("/api/ai/insights").json()["predictions"]["rejected"] == 1


def test_state_token_notices_an_in_place_prediction_change(fresh_client):
    with fresh_client.sf() as db:
        r = repo.latest_reading_for_plant(db, 1)

        def make(label_soil):
            base = make_response(PredictRequest(reading=to_ml_reading(r), history=[]), anomaly=False, explain=False)
            base["stress_prediction"]["label"] = label_soil
            return Outcome("ok", PredictResponse.model_validate(base), history_rows_sent=60)

        persist_prediction(db, r, make("low"))
        db.commit()
        first = repo.state_token(db)
        persist_prediction(db, r, make("high"))  # same row, same ids and counts
        db.commit()
        second = repo.state_token(db)
    assert first != second


def test_settings_put_does_not_touch_cached_aggregates(fresh_client):
    """Settings are not part of any cached payload, so PUT needs no invalidation (documented in the README)."""
    before = fresh_client.get("/api/monitoring/overview").json()
    assert fresh_client.put("/api/settings", json={"polling_interval_seconds": 9}).status_code == 200
    assert fresh_client.get("/api/monitoring/overview").json() == before
