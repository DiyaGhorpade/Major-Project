"""Regression tests for the final review: reset/failure plant statuses, config-vs-data ML rejections, writers racing
a Run, server-side recommendation search, IST text in plain-language summaries, `daytimeHours` and the distinct
unknown-route error code. Each test fails on the code before the fix."""

import re
import time

import pytest
from sqlalchemy import func, select

from app.core.mappings import IST, severity_to_status
from app.db import repositories as repo
from app.db.models import MlPrediction, Plant, Recommendation
from app.services import recommendation_service as svc
from app.services.seed_service import ensure_seeded
from tests.conftest import make_api_settings, open_app
from tests.fakes import FakeML
from tests.test_ingestion import reading
from tests.test_pipeline import TOTAL, counts, running_app, wait_done
from tests.test_recommendations import craft, ml_rec, recs_of


def statuses(client) -> dict[int, str]:
    return {p["id"]: p["status"] for p in client.get("/api/plants").json()}


def derived_statuses(client) -> dict[int, str]:
    """What each plant's status must be, computed from the stored predictions only."""
    with client.sf() as db:
        latest = repo.latest_ok_prediction_per_plant(db)
        return {p.id: severity_to_status(latest[p.id].severity if p.id in latest else None) for p in repo.list_plants(db)}


def wait_until(predicate, timeout: float = 60, what: str = "condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def fake_app(real_settings):
    fake = FakeML()
    with running_app(real_settings, fake) as client:
        client.fake = fake
        yield client


# --- 1. plant statuses vs. reset / failed runs -----------------------------------------------------
def test_statuses_are_unmeasured_while_a_reset_run_has_no_predictions(fake_app):
    fake_app.post("/api/pipeline/run")
    assert wait_done(fake_app)["state"] == "completed"
    assert set(statuses(fake_app).values()) <= {"healthy", "attention", "critical"}  # measured

    fake_app.fake.gate.clear()  # hold the ML batches: the run stays in progress right after the reset
    fake_app.post("/api/pipeline/run", json={"reset": True})
    wait_until(lambda: counts(fake_app)[2] == 0, what="the reset to delete the predictions")
    assert fake_app.get("/api/pipeline/status").json()["state"] in ("queued", "running")
    assert set(statuses(fake_app).values()) == {"unmeasured"}
    overview = fake_app.get("/api/monitoring/overview").json()  # derived/cached views follow
    assert overview["plantStatusCounts"]["unmeasured"] == 16
    assert overview["totals"]["predictions"]["total"] == 0

    fake_app.fake.gate.set()
    assert wait_done(fake_app)["state"] == "completed"
    assert "unmeasured" not in statuses(fake_app).values()
    assert statuses(fake_app) == derived_statuses(fake_app)


def test_failed_reset_run_leaves_statuses_matching_the_remaining_predictions(fake_app):
    fake_app.post("/api/pipeline/run")
    assert wait_done(fake_app)["state"] == "completed"

    fake_app.fake.die_after_batches = fake_app.fake.batches + 10  # the ML service goes away after ~640 of 5,376 readings
    fake_app.post("/api/pipeline/run", json={"reset": True})
    status = wait_done(fake_app)
    assert status["state"] == "failed" and "became unavailable" in status["message"]
    got = statuses(fake_app)
    assert got == derived_statuses(fake_app)
    assert "unmeasured" in got.values()  # plants whose readings were never predicted
    assert got[1] != "unmeasured"  # plant 1's readings come first and were predicted


# --- 2. request-level ML rejections must not poison readings --------------------------------------
def rejected_rows(client) -> int:
    with client.sf() as db:
        return db.scalar(select(func.count()).select_from(MlPrediction).where(MlPrediction.status == "rejected")) or 0


def test_whole_batch_config_rejection_fails_the_run_and_stores_no_rejected_rows(fake_app):
    fake_app.fake.batch_error = {"error": "bad_batch_size", "message": "items must contain 1..32 entries"}
    fake_app.post("/api/pipeline/run")
    status = wait_done(fake_app)
    assert status["state"] == "failed"
    assert "bad_batch_size" in status["message"] and "api.max_batch_items" in status["message"] and "restart" in status["message"]
    assert rejected_rows(fake_app) == 0 and counts(fake_app)[2] == 0

    fake_app.fake.batch_error = None  # the config is fixed: a plain re-run now predicts everything
    fake_app.post("/api/pipeline/run")
    status = wait_done(fake_app)
    assert status["state"] == "completed" and status["total"] == TOTAL and status["failed"] == 0
    assert counts(fake_app)[0] == {"ok": TOTAL}


def test_per_item_history_too_long_fails_the_run_and_is_not_stored_as_rejected(fake_app):
    fake_app.fake.reject_if = lambda req: req.reading.plant_id == 3
    fake_app.fake.reject_code = "history_too_long"
    fake_app.post("/api/pipeline/run")
    status = wait_done(fake_app)
    assert status["state"] == "failed" and "history_too_long" in status["message"] and "api.max_history_rows" in status["message"]
    assert rejected_rows(fake_app) == 0

    fake_app.fake.reject_if = lambda req: False
    fake_app.post("/api/pipeline/run")
    status = wait_done(fake_app)
    assert status["state"] == "completed" and counts(fake_app)[0] == {"ok": TOTAL}  # nothing was skipped for good


@pytest.mark.parametrize("code", ["invalid_readings", "mixed_plants", "history_not_before_reading"])
def test_data_rejections_are_still_stored_as_rejected(fake_app, code):
    fake_app.fake.reject_if = lambda req: req.reading.plant_id == 5 and req.reading.soil is None
    fake_app.fake.reject_code = code
    fake_app.post("/api/pipeline/run")
    status = wait_done(fake_app)
    assert status["state"] == "completed" and rejected_rows(fake_app) > 0
    with fake_app.sf() as db:
        assert db.scalars(select(MlPrediction).where(MlPrediction.status == "rejected")).first().error.startswith(code)


def test_on_demand_config_rejection_stores_nothing_but_a_data_rejection_is_stored(fake_app):
    fake_app.fake.reject_if = lambda req: True
    fake_app.fake.reject_code = "history_too_long"
    r = fake_app.post("/api/predictions", json={"plantId": 4})
    assert r.status_code == 422 and r.json()["error"]["code"] == "ml_rejected"
    assert rejected_rows(fake_app) == 0

    fake_app.fake.reject_code = "invalid_readings"
    assert fake_app.post("/api/predictions", json={"plantId": 4}).status_code == 422
    assert rejected_rows(fake_app) == 1


# --- 3. writers racing a Run -----------------------------------------------------------------------
@pytest.mark.parametrize("via", ["ingestion", "on_demand"])
def test_predictions_stored_during_a_run_do_not_refresh_recommendations_but_the_run_sees_them(fake_app, via):
    fake_app.fake.gate.clear()
    fake_app.post("/api/pipeline/run")
    wait_until(lambda: fake_app.get("/api/pipeline/status").json()["state"] == "running", what="the run to start")

    if via == "ingestion":
        r = fake_app.post("/api/sensors/readings", json=reading(soil=20.0))
        assert r.status_code == 201
        record_id = r.json()["items"][0]["record_id"]
    else:
        record_id = fake_app.post("/api/predictions", json={"plantId": 1}).json()["record_id"]
    with fake_app.sf() as db:
        stored = db.scalars(select(MlPrediction).where(MlPrediction.reading_id == int(record_id[2:]))).one()
        assert stored.status == "ok"
        assert db.scalar(select(func.count()).select_from(Recommendation)) == 0  # no refresh while the run owns the rebuild
    assert statuses(fake_app)[1] == "unmeasured"

    fake_app.fake.gate.set()
    assert wait_done(fake_app)["state"] == "completed"
    plant = fake_app.get("/api/plants/1").json()
    assert plant["conditionPrediction"]["recordId"] == record_id  # the run's rebuild used the prediction stored meanwhile
    assert plant["status"] == severity_to_status(plant["conditionPrediction"]["severity"])


def test_ingestion_outside_a_run_still_refreshes_the_plant(fake_app):
    assert fake_app.post("/api/sensors/readings", json=reading(soil=20.0)).status_code == 201
    assert fake_app.get("/api/plants/1").json()["status"] != "unmeasured"


@pytest.fixture
def seeded(real_settings, db_for):
    db = db_for(real_settings)
    ensure_seeded(db, real_settings)
    return db


def test_build_for_run_survives_a_recommendation_committed_between_read_and_sync(seeded, real_settings, db_for, monkeypatch):
    """Ingestion's `refresh_plant` commits the plant's recommendation after the run read the plant's rows but before
    it wrote its own: the run must update that row instead of crashing on `uq_rec_plant_cat_reading`."""
    craft(seeded, 5, [{"severity": "high", "recs": [ml_rec("check_watering", "soil", "medium")]}])
    other = db_for(real_settings)
    real_build, fired = svc.build_candidates, []

    def racing(state):
        if not fired and state.reading.plant_id == 5:
            fired.append(True)
            svc.refresh_plant(other, 5)  # another writer commits the same plant's recommendations first
        return real_build(state)

    monkeypatch.setattr(svc, "build_candidates", racing)
    svc.build_for_run(seeded, 1)

    assert fired
    seeded.expire_all()
    rows = list(seeded.scalars(select(Recommendation).where(Recommendation.plant_id == 5)))
    active = [r for r in rows if r.status in svc.ACTIVE]
    assert len({(r.category, r.source_reading_id) for r in rows}) == len(rows)  # the unique key holds
    assert sorted(r.category for r in active) == sorted({r.category for r in active})  # one active row per category
    assert "irrigation" in recs_of(seeded, 5, active_only=True)
    assert seeded.get(Plant, 5).status == "critical"


def test_refresh_plant_failure_does_not_lose_the_stored_prediction(fake_app, monkeypatch):
    def boom(db, plant_id):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(svc, "refresh_plant", boom)
    r = fake_app.post("/api/predictions", json={"plantId": 2})
    assert r.status_code == 200 and r.json()["status"] == "ok"
    with fake_app.sf() as db:
        assert db.scalar(select(func.count()).select_from(MlPrediction)) == 1


# --- 4. server-side recommendation search ----------------------------------------------------------
def test_recommendation_search_covers_all_pages_and_fields(ran_client):
    everything = ran_client.get("/api/recommendations?limit=500").json()
    assert len(everything) > 3

    def expected(q: str) -> list[str]:
        needle = q.lower()
        return [
            r["id"] for r in everything
            if any(needle in str(r[k]).lower() for k in ("title", "simpleAction", "reason", "plant_name", "plant_id"))
        ]  # fmt: skip

    pick = everything[-1]  # a recommendation that is not on the first page of a 1-item page size
    queries = {pick["title"].split()[0], pick["plant_name"].split(":")[0].upper(), str(pick["plant_id"]), pick["reason"][5:25], "%", "_", "zzz-nothing"}
    for q in queries:
        want = expected(q)
        small = ran_client.get("/api/recommendations", params={"q": q, "limit": 1})
        assert small.status_code == 200
        assert int(small.headers["x-total-count"]) == len(want), q  # counted over every page, not the current one
        full = ran_client.get("/api/recommendations", params={"q": q, "limit": 500}).json()
        assert sorted(r["id"] for r in full) == sorted(want), q
    assert pick["id"] in [r["id"] for r in ran_client.get("/api/recommendations", params={"q": pick["title"].lower(), "limit": 500}).json()]
    assert int(ran_client.get("/api/recommendations", params={"q": "  "}).headers["x-total-count"]) == len(everything)  # blank = no filter
    assert ran_client.get("/api/recommendations", params={"q": "x" * 101}).status_code == 422


# --- 9. plain-language summaries use IST text, not ISO ---------------------------------------------
def test_to_ist_text_formats_a_human_ist_time():
    from datetime import UTC, datetime

    from app.core.mappings import to_ist_text

    assert to_ist_text(datetime(2026, 10, 18, 16, 30, 45, tzinfo=UTC)) == "18 Oct 2026, 22:00 IST"
    assert to_ist_text(datetime(2026, 1, 5, 3, 0, tzinfo=IST)) == "5 Jan 2026, 03:00 IST"
    assert to_ist_text(None) == "an unknown time"


ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def test_layman_text_has_no_raw_iso_timestamps(ran_client):
    shap = ran_client.get("/api/ai/shap/4").json()
    assert re.search(r"For the reading at \d{1,2} [A-Z][a-z]{2} \d{4}, \d{2}:\d{2} IST, the model rated", shap["laymanSummary"])
    assert not ISO.search(shap["laymanSummary"])
    assert shap["timestamp"].endswith("+05:30")  # machine-readable field stays ISO
    explanation = ran_client.get("/api/plants/4").json()["conditionPrediction"]["simpleExplanation"]
    assert not ISO.search(explanation) and " IST" in explanation


# --- 11. daytime hours reach the frontend ----------------------------------------------------------
def test_plants_expose_daytime_hours_from_the_ml_config(ran_client):
    for plant in (*ran_client.get("/api/plants").json(), ran_client.get("/api/plants/5").json()):
        assert plant["optimalRanges"]["light"]["daytimeHours"] == [9, 16]
        assert "daytimeHours" not in plant["optimalRanges"]["soil"]
    assert ran_client.get("/api/monitoring/plants/5").json()["plant"]["optimalRanges"]["light"]["daytimeHours"] == [9, 16]


def test_daytime_hours_follow_the_configured_value(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("recommendations:\n  daytime_hours: [6, 18]\n")
    settings = make_api_settings(tmp_path).model_copy(update={"ML_CONFIG_PATH": config})
    with open_app(settings, FakeML()) as client:
        assert client.get("/api/plants/1").json()["optimalRanges"]["light"]["daytimeHours"] == [6, 18]


# --- 5. unknown route vs. missing entity -----------------------------------------------------------
def test_unknown_url_and_missing_entity_have_distinct_error_codes(fresh_client):
    assert fresh_client.get("/api/plants/99").json()["error"]["code"] == "not_found"
    unknown = fresh_client.get("/api/definitely/not/a/route")
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "route_not_found"
