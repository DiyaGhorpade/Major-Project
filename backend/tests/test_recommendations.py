"""Recommendation rules, lifecycle and API. Predictions are crafted straight into a seeded database (the same
`PredictResponse` shape the ML service returns) so each rule is exercised on exact inputs; two tests run the
whole pipeline against the fake ML service."""

import time
from contextlib import contextmanager
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from app.db import database
from app.db import repositories as repo
from app.db.models import MlPrediction, Plant, Recommendation, SensorReading
from app.integrations.ml_types import PredictRequest, PredictResponse
from app.main import create_app
from app.services import recommendation_service as svc
from app.services.prediction_service import Outcome, persist_prediction, to_ml_reading
from app.core.mappings import IST, ensure_aware
from app.services.history_builder import read_daytime_hours
from tests.fakes import LABELS, FakeML, make_response

HEALTH_WORDS = ("stress", "unhealthy", "healthy", "health", "disease", "sick", "dying")


# --- helpers --------------------------------------------------------------------------------
def ml_rec(rec_id: str, category: str, priority: str = "medium", reason: str = "ml reason") -> dict:
    return {
        "id": rec_id, "category": category, "text": f"ML text for {rec_id}", "reason": reason,
        "priority": priority, "supporting_features": {}, "type": "heuristic_suggestion",
    }  # fmt: skip


def put_pred(db, reading, *, severity="healthy", anomaly=False, recs=(), warnings=(), history_rows=60, p=0.7):
    base = make_response(PredictRequest(reading=to_ml_reading(reading), history=[]), anomaly=anomaly, explain=False)
    probs = {k: round((1 - p) / 3, 4) for k in LABELS}
    probs[severity] = p
    base["stress_prediction"] = {"label": severity, "probabilities": probs}
    base["recommendations"] = list(recs)
    base["warnings"] = list(warnings)
    result = PredictResponse.model_validate(base)
    return persist_prediction(db, reading, Outcome("ok", result, history_rows_sent=history_rows))


def last_readings(db, plant_id: int, n: int) -> list[SensorReading]:
    return repo.readings_for_plant_ordered(db, plant_id)[-n:]


def craft(db, plant_id: int, specs: list[dict], soil: list[float | None] | None = None) -> list[SensorReading]:
    """Predictions for the plant's last len(specs) readings (oldest first); optionally set their soil values."""
    readings = last_readings(db, plant_id, len(specs))
    for i, (reading, spec) in enumerate(zip(readings, specs, strict=True)):
        if soil is not None:
            reading.soil = soil[i]
        put_pred(db, reading, **spec)
    db.commit()
    return readings


def build(db, run_id: int = 1) -> int:
    return svc.build_for_run(db, run_id)


def recs_of(db, plant_id: int, *, active_only=False) -> dict[str, Recommendation]:
    rows = db.scalars(select(Recommendation).where(Recommendation.plant_id == plant_id).order_by(Recommendation.id)).all()
    return {r.category: r for r in rows if not active_only or r.status in svc.ACTIVE}


def snapshot(db):
    return sorted(
        (r.id, r.plant_id, r.category, r.status, r.user_modified, r.priority, r.reason, r.updated_at)
        for r in db.scalars(select(Recommendation))
    )  # fmt: skip


@pytest.fixture
def seeded(real_settings, db_for):
    from app.services.seed_service import ensure_seeded

    db = db_for(real_settings)
    ensure_seeded(db, real_settings)
    return db


@contextmanager
def running_app(settings, fake):
    engine = database.build_engine(settings.DATABASE_URL)
    app = create_app(settings, engine, ml_client=fake)
    with TestClient(app) as client:
        client.sf = app.state.session_factory
        yield client
    engine.dispose()


@pytest.fixture
def api(real_settings):
    """App with a seeded DB and no predictions yet; build recommendations through `client.sf`."""
    with running_app(real_settings, FakeML()) as client:
        yield client


def wait_done(client, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/pipeline/status").json()
        if status["state"] in ("completed", "failed"):
            return status
        time.sleep(0.05)
    raise AssertionError("run did not finish")


# --- escalation -----------------------------------------------------------------------------
def test_ml_recommendation_is_mapped_with_real_numbers(seeded):
    craft(seeded, 5, [{"severity": "low", "recs": [ml_rec("check_watering", "soil", "low", "Last 3 soil readings are below 25%.")]}] * 3,
          soil=[30.0, 28.0, 21.3])  # fmt: skip
    build(seeded)
    rec = recs_of(seeded, 5)["irrigation"]
    assert rec.source == "ml" and rec.ml_rec_id == "check_watering" and rec.status == "new"
    assert rec.title == "Check watering" and rec.simple_action == "ML text for check_watering"
    assert rec.priority == "low"  # low severity: not escalated
    assert "Last 3 soil readings are below 25%." in rec.reason
    assert "Soil moisture 21.3 %" in rec.reason and "P(low)=0.70" in rec.reason
    assert rec.confidence == pytest.approx(0.7)
    assert rec.guidance_source == "ML heuristic rules · model 1.0.0 (run fake_run_1)"
    assert rec.based_on["modelVersion"] == "1.0.0" and rec.based_on["runId"] == "fake_run_1"
    assert rec.based_on["timestamp"].endswith("+05:30") and rec.scientific_context
    assert rec.prediction_id is not None and rec.user_modified is False


def test_recent_average_is_cited_when_history_allows(seeded):
    readings = last_readings(seeded, 6, 6)
    for r, v in zip(readings, [40.0, 40.0, 40.0, 36.0, 34.0, 22.0], strict=True):
        r.soil = v
    seeded.commit()
    craft(seeded, 6, [{"severity": "moderate", "recs": [ml_rec("check_watering", "soil")]}])
    build(seeded)
    assert "Soil moisture 22.0 % (recent average 35.3 %)" in recs_of(seeded, 6)["irrigation"].reason


def test_high_severity_raises_priority_to_high(seeded):
    craft(seeded, 5, [{"severity": "high", "recs": [ml_rec("check_watering", "soil", "medium")]}])
    build(seeded)
    rec = recs_of(seeded, 5)["irrigation"]
    assert rec.priority == "high" and rec.source == "ml"
    assert "Priority raised to high" in rec.reason and "'high'" in rec.reason


def test_high_severity_without_ml_recommendation_adds_heuristic_monitoring(seeded):
    craft(seeded, 7, [{"severity": "high"}])
    build(seeded)
    rec = recs_of(seeded, 7)["monitoring"]
    assert rec.source == "heuristic" and rec.priority == "high" and rec.ml_rec_id is None
    assert rec.title == "Sustained stress: inspect plant"
    assert rec.guidance_source.startswith("Backend rules on ML output")
    assert "P(high)=0.70" in rec.reason


def test_sustained_moderate_is_escalated_but_single_moderate_is_not(seeded):
    rec_in = [ml_rec("check_climate", "temperature_humidity", "medium")]
    craft(seeded, 13, [{"severity": "moderate", "recs": rec_in}] * 3)  # sustained
    craft(seeded, 14, [{"severity": "healthy"}, {"severity": "healthy"}, {"severity": "moderate", "recs": rec_in}])
    craft(seeded, 15, [{"severity": "moderate", "recs": rec_in}, {"severity": "healthy"}, {"severity": "moderate", "recs": rec_in}])  # not consecutive
    build(seeded)
    sustained = recs_of(seeded, 13)["climate"]
    assert sustained.priority == "high" and "last 3 consecutive readings" in sustained.reason
    assert recs_of(seeded, 14)["climate"].priority == "medium"
    assert recs_of(seeded, 15)["climate"].priority == "medium"


def test_sustained_moderate_without_ml_recommendation_adds_heuristic(seeded):
    craft(seeded, 9, [{"severity": "moderate"}] * 3)
    build(seeded)
    rec = recs_of(seeded, 9)["monitoring"]
    assert rec.source == "heuristic" and rec.priority == "high"
    craft(seeded, 10, [{"severity": "moderate"}])  # a single moderate reading: nothing to add
    build(seeded)
    assert recs_of(seeded, 10) == {}


def test_no_escalation_when_reading_is_suspect_or_model_unsure(seeded):
    craft(seeded, 1, [{"severity": "high", "anomaly": True, "recs": [ml_rec("verify_sensor", "data_quality")]}])
    craft(seeded, 2, [{"severity": "high", "recs": [ml_rec("low_confidence", "model_confidence", "low")]}])
    build(seeded)
    assert set(recs_of(seeded, 1)) == {"sensor_check"} and recs_of(seeded, 1)["sensor_check"].priority == "medium"
    assert set(recs_of(seeded, 2)) == {"monitoring"} and recs_of(seeded, 2)["monitoring"].priority == "low"


def test_critical_priority_maps_to_high(seeded):
    craft(seeded, 11, [{"severity": "healthy", "recs": [ml_rec("check_light_low", "light", "critical")]}])
    build(seeded)
    assert recs_of(seeded, 11)["lighting"].priority == "high"


# --- anomaly / sensor check -----------------------------------------------------------------
def test_anomaly_creates_sensor_check_without_health_claims(seeded):
    craft(seeded, 3, [{"severity": "low", "anomaly": True}], soil=[41.5])
    build(seeded)
    rec = recs_of(seeded, 3)["sensor_check"]
    assert rec.source == "heuristic" and rec.title == "Sensor check" and rec.priority == "medium"
    assert rec.simple_action == "Verify the sensor, its placement and this reading"
    for text in (rec.title, rec.simple_action, rec.reason, rec.scientific_context):
        assert not any(w in text.lower() for w in HEALTH_WORDS), text
    assert "Anomaly score 0.20" in rec.reason and "Soil moisture 41.5 %" in rec.reason
    assert "P(" not in rec.reason  # the stress estimate of a suspect reading is not quoted
    assert set(recs_of(seeded, 3)) == {"sensor_check"}


def test_anomaly_not_duplicated_when_ml_verify_sensor_present(seeded):
    craft(seeded, 4, [{"severity": "low", "anomaly": True, "recs": [ml_rec("verify_sensor", "data_quality")]}])
    build(seeded)
    rows = seeded.scalars(select(Recommendation).where(Recommendation.plant_id == 4)).all()
    assert [(r.category, r.source, r.ml_rec_id) for r in rows] == [("sensor_check", "ml", "verify_sensor")]


# --- reasons, cold start, supporting values -------------------------------------------------
def test_cold_start_note_added_only_for_short_history(seeded):
    rec_in = [ml_rec("check_watering", "soil")]
    craft(seeded, 5, [{"severity": "low", "recs": rec_in, "history_rows": 10, "warnings": ["Only 11 reading(s) of history (cold start)."]}])
    craft(seeded, 6, [{"severity": "low", "recs": rec_in, "history_rows": 10}])  # by row count alone
    craft(seeded, 7, [{"severity": "low", "recs": rec_in, "history_rows": 60}])
    build(seeded)
    note = "Lower reliability: limited history for this plant."
    assert recs_of(seeded, 5)["irrigation"].reason.endswith(note)
    assert note in recs_of(seeded, 6)["irrigation"].reason
    assert note not in recs_of(seeded, 7)["irrigation"].reason


def test_supporting_values_against_optimal_ranges_and_missing(seeded):
    (reading,) = craft(seeded, 5, [{"severity": "low", "recs": [ml_rec("check_watering", "soil")]}])
    reading.soil, reading.temperature, reading.humidity, reading.light = 21.3, None, 95.0, 25000.0
    seeded.commit()
    plant = seeded.get(Plant, 5)
    hour = ensure_aware(reading.timestamp).astimezone(IST).hour
    day = (hour, hour)  # the reading falls inside the daytime window
    values = {v["metric"]: v for v in svc.supporting_sensor_values(reading, plant.optimal_ranges, day)}
    assert values["Soil moisture"] == {"metric": "Soil moisture", "value": "21.3 %", "target": "25–75 %", "status": "low"}
    assert values["Temperature"]["value"] == "n/a" and values["Temperature"]["status"] == "attention"
    assert values["Temperature"]["target"] == "18–32 °C"
    assert values["Humidity"]["status"] == "high" and values["Light"]["status"] == "high"
    assert values["Light"]["value"] == "25,000 lux" and values["Light"]["target"] == "300–20,000 lux"
    reading.soil = 50.0
    assert {v["metric"]: v["status"] for v in svc.supporting_sensor_values(reading, plant.optimal_ranges, day)}["Soil moisture"] == "good"


def test_light_outside_daytime_hours_has_no_target(seeded):
    (reading,) = last_readings(seeded, 5, 1)
    reading.light, reading.soil = 1.0, 10.0
    hour = ensure_aware(reading.timestamp).astimezone(IST).hour
    night = ((hour + 1) % 24, (hour + 1) % 24)
    values = {v["metric"]: v for v in svc.supporting_sensor_values(reading, seeded.get(Plant, 5).optimal_ranges, night)}
    assert values["Light"] == {"metric": "Light", "value": "1 lux", "target": "night (no target)", "status": "good"}
    assert values["Soil moisture"]["status"] == "low"  # other sensors are unaffected
    reading.light = None
    assert svc.supporting_sensor_values(reading, seeded.get(Plant, 5).optimal_ranges, night)[3]["status"] == "attention"
    from pathlib import Path

    assert read_daytime_hours(Path("/nonexistent/config.yaml")) == (9, 16)
    real = Path(__file__).resolve().parents[2] / "money_plant_ml_pipeline" / "config" / "config.yaml"
    assert read_daytime_hours(real) == (9, 16)  # the ML config's recommendations.daytime_hours


def test_recent_average_skips_out_of_range_flagged_values(seeded):
    readings = last_readings(seeded, 6, 6)
    for r, v in zip(readings, [40.0, 40.0, 40.0, 999.0, 34.0, 22.0], strict=True):
        r.soil = v
    readings[3].status, readings[3].validation_error = "flagged", "out_of_range: soil=999.0"
    seeded.commit()
    craft(seeded, 6, [{"severity": "moderate", "recs": [ml_rec("check_watering", "soil")]}])
    reason = recs_of(seeded, 6)  # nothing built yet
    assert reason == {}
    build(seeded)
    assert "Soil moisture 22.0 % (recent average 35.2 %)" in recs_of(seeded, 6)["irrigation"].reason  # (22+34+40+40+40)/5
    assert svc.flagged_out_of_range(readings[3], "soil") and not svc.flagged_out_of_range(readings[3], "light")
    readings[3].validation_error = "missing: light; out_of_range: temperature=45.0, humidity=3.0"
    assert svc.flagged_out_of_range(readings[3], "temperature") and svc.flagged_out_of_range(readings[3], "humidity")
    assert not svc.flagged_out_of_range(readings[3], "light") and not svc.flagged_out_of_range(readings[3], "soil")


def test_missing_class_probability_falls_back_to_confidence(seeded):
    (reading,) = craft(seeded, 5, [{"severity": "low", "recs": [ml_rec("check_watering", "soil")]}])
    pred = repo.get_prediction_by_reading(seeded, reading.id)
    pred.probabilities = {"healthy": 0.5, "moderate": 0.2}  # no entry for the predicted class
    pred.confidence = 0.5
    seeded.commit()
    build(seeded)
    rec = recs_of(seeded, 5)["irrigation"]
    assert "confidence=0.50" in rec.reason and "P(" not in rec.reason


# --- dedupe and lifecycle -------------------------------------------------------------------
def test_dedupe_per_plant_and_category_keeps_highest_priority(seeded):
    recs = [ml_rec("check_soil_drying", "soil", "low", "dropping fast"), ml_rec("check_watering", "soil", "high", "stayed low")]
    craft(seeded, 8, [{"severity": "low", "recs": recs}])
    build(seeded)
    rows = seeded.scalars(select(Recommendation).where(Recommendation.plant_id == 8)).all()
    assert len(rows) == 1 and rows[0].priority == "high" and rows[0].ml_rec_id == "check_watering"
    assert "stayed low" in rows[0].reason and "dropping fast" in rows[0].reason


def test_rerun_without_reset_changes_nothing(seeded):
    craft(seeded, 5, [{"severity": "high", "recs": [ml_rec("check_watering", "soil")]}])
    craft(seeded, 3, [{"severity": "low", "anomaly": True}])
    craft(seeded, 9, [{"severity": "moderate"}] * 3)
    assert build(seeded) == 3
    before = snapshot(seeded)
    assert len(before) == 3
    assert build(seeded, 2) == 0
    assert snapshot(seeded) == before  # same ids, statuses, content and updated_at


def test_user_status_survives_rerun_and_closed_rec_is_not_resurrected(seeded):
    craft(seeded, 5, [{"severity": "high", "recs": [ml_rec("check_watering", "soil")]}])
    build(seeded)
    rec = recs_of(seeded, 5)["irrigation"]
    svc.set_status(seeded, rec, "completed")
    build(seeded, 2)
    rows = seeded.scalars(select(Recommendation).where(Recommendation.plant_id == 5)).all()
    assert len(rows) == 1 and rows[0].id == rec.id and rows[0].status == "completed" and rows[0].user_modified
    # in-progress is an active status: updated in place, status kept
    craft(seeded, 6, [{"severity": "high", "recs": [ml_rec("check_watering", "soil", "medium")]}])
    build(seeded)
    mine = recs_of(seeded, 6)["irrigation"]
    svc.set_status(seeded, mine, "in_progress")
    put_pred(seeded, last_readings(seeded, 6, 1)[0], severity="high", p=0.9, recs=[ml_rec("check_watering", "soil", "high", "newer")])
    seeded.commit()
    build(seeded, 3)
    again = recs_of(seeded, 6)["irrigation"]
    assert again.id == mine.id and again.status == "in_progress" and again.user_modified
    assert again.confidence == pytest.approx(0.9) and "newer" in again.reason


def test_new_reading_after_completion_creates_a_new_recommendation(seeded):
    (reading,) = craft(seeded, 5, [{"severity": "high", "recs": [ml_rec("check_watering", "soil")]}])
    build(seeded)
    first = recs_of(seeded, 5)["irrigation"]
    svc.set_status(seeded, first, "dismissed")
    newer = SensorReading(
        timestamp=reading.timestamp + timedelta(hours=2), plant_id=5, node_id=reading.node_id, sampling_point=reading.sampling_point,
        soil=20.0, temperature=25.0, humidity=60.0, light=500.0, status="accepted", source="dataset",
    )  # fmt: skip
    seeded.add(newer)
    seeded.flush()
    put_pred(seeded, newer, severity="high", recs=[ml_rec("check_watering", "soil")])
    seeded.commit()
    assert build(seeded, 2) == 1
    rows = seeded.scalars(select(Recommendation).where(Recommendation.plant_id == 5).order_by(Recommendation.id)).all()
    assert [(r.status, r.source_reading_id) for r in rows] == [("dismissed", reading.id), ("new", newer.id)]


def test_untouched_active_rec_is_superseded_when_condition_disappears(seeded):
    rec_in = [ml_rec("check_watering", "soil")]
    (r5,) = craft(seeded, 5, [{"severity": "low", "recs": rec_in}])
    (r6,) = craft(seeded, 6, [{"severity": "low", "recs": rec_in}])
    build(seeded)
    touched = recs_of(seeded, 6)["irrigation"]
    svc.set_status(seeded, touched, "in_progress")
    for reading in (r5, r6):  # condition gone: the next prediction has no recommendations
        put_pred(seeded, reading, severity="healthy")
    seeded.commit()
    assert build(seeded, 2) == 0
    assert recs_of(seeded, 5)["irrigation"].status == "superseded"
    assert recs_of(seeded, 6)["irrigation"].status == "in_progress"  # user-touched: left as it is
    # and it is revived (not duplicated) if the condition returns
    put_pred(seeded, r5, severity="low", recs=rec_in)
    seeded.commit()
    assert build(seeded, 3) == 0
    assert [r.status for r in seeded.scalars(select(Recommendation).where(Recommendation.plant_id == 5))] == ["new"]


def test_plant_status_follows_latest_ok_prediction(seeded):
    craft(seeded, 1, [{"severity": "healthy"}])
    craft(seeded, 2, [{"severity": "low"}])
    craft(seeded, 3, [{"severity": "moderate"}])
    craft(seeded, 4, [{"severity": "high"}])
    craft(seeded, 5, [{"severity": "healthy"}, {"severity": "high", "anomaly": True}])  # anomaly never changes status
    build(seeded)
    status = {p.id: p.status for p in seeded.scalars(select(Plant))}
    assert [status[i] for i in (1, 2, 3, 4)] == ["healthy", "attention", "attention", "critical"]
    assert status[5] == "critical" and status[16] == "unmeasured"
    craft(seeded, 5, [{"severity": "healthy", "anomaly": True}])
    build(seeded, 2)
    assert seeded.get(Plant, 5).status == "healthy"


def test_latest_ok_prediction_is_used_when_newest_reading_is_unavailable(seeded):
    readings = last_readings(seeded, 5, 3)
    put_pred(seeded, readings[0], severity="high", recs=[ml_rec("check_watering", "soil")])
    put_pred(seeded, readings[1], severity="high", recs=[ml_rec("check_watering", "soil")])
    persist_prediction(seeded, readings[2], Outcome("unavailable", error="down"))
    seeded.commit()
    build(seeded)
    rec = recs_of(seeded, 5)["irrigation"]
    assert rec.source_reading_id == readings[1].id and seeded.get(Plant, 5).status == "critical"


def test_latest_ok_prediction_found_even_when_many_newer_readings_have_none(seeded):
    old = last_readings(seeded, 5, 30)[0]
    put_pred(seeded, old, severity="high", recs=[ml_rec("check_watering", "soil")])
    seeded.commit()
    build(seeded)
    assert recs_of(seeded, 5)["irrigation"].source_reading_id == old.id and seeded.get(Plant, 5).status == "critical"


# --- reset ----------------------------------------------------------------------------------
def test_reset_generated_keeps_user_modified_and_detaches_predictions(seeded):
    rec_in = [ml_rec("check_watering", "soil")]
    craft(seeded, 5, [{"severity": "high", "recs": rec_in}])
    craft(seeded, 6, [{"severity": "high", "recs": rec_in}])
    build(seeded)
    kept = recs_of(seeded, 6)["irrigation"]
    svc.set_status(seeded, kept, "completed")
    svc.reset_generated(seeded)
    repo.delete_all_predictions(seeded)  # would violate the FK if prediction_id were still set
    seeded.commit()
    assert recs_of(seeded, 5) == {}
    survivor = recs_of(seeded, 6)["irrigation"]
    assert survivor.id == kept.id and survivor.status == "completed" and survivor.prediction_id is None
    # rebuilt from fresh predictions: the kept rec is matched, not duplicated; the untouched one is recreated
    craft(seeded, 5, [{"severity": "high", "recs": rec_in}])
    craft(seeded, 6, [{"severity": "high", "recs": rec_in}])
    assert build(seeded, 2) == 1
    assert recs_of(seeded, 6)["irrigation"].id == kept.id and recs_of(seeded, 6)["irrigation"].status == "completed"
    assert recs_of(seeded, 6)["irrigation"].prediction_id is not None
    assert recs_of(seeded, 5)["irrigation"].status == "new"


# --- API ------------------------------------------------------------------------------------
def populate(client):
    with client.sf() as db:
        rec_in = [ml_rec("check_watering", "soil", "medium")]
        craft(db, 5, [{"severity": "high", "recs": rec_in}])
        craft(db, 6, [{"severity": "low", "recs": rec_in}])
        craft(db, 13, [{"severity": "low", "recs": [ml_rec("check_climate", "temperature_humidity", "low")]}])
        craft(db, 3, [{"severity": "low", "anomaly": True}])
        craft(db, 9, [{"severity": "moderate"}] * 3)
        build(db)


def test_list_shape_matches_frontend_type(api):
    populate(api)
    response = api.get("/api/recommendations")
    assert response.status_code == 200 and response.headers["X-Total-Count"] == "5"
    items = response.json()
    assert isinstance(items, list) and len(items) == 5
    item = next(i for i in items if i["category"] == "irrigation" and i["plant_id"] == 5)
    assert set(item) == {
        "id", "plant_id", "plant_name", "cluster", "title", "simpleAction", "priority", "status", "category",
        "supporting_sensor_values", "reason", "scientificContext", "guidance_source", "created_at", "updated_at",
        "resolved_at", "source", "confidence", "basedOn", "disclaimer", "mlRecId",
    }  # fmt: skip
    assert item["id"].startswith("rec-") and item["cluster"] == "Cluster 2" and item["plant_name"].startswith("Pot 5")
    assert item["priority"] == "high" and item["status"] == "new" and item["source"] == "ml"
    assert item["disclaimer"] is True and item["mlRecId"] == "check_watering" and item["resolved_at"] is None
    assert set(item["basedOn"]) == {"timestamp", "modelVersion", "runId"}
    assert [v["metric"] for v in item["supporting_sensor_values"]] == ["Soil moisture", "Temperature", "Humidity", "Light"]
    assert set(item["supporting_sensor_values"][0]) == {"metric", "value", "target", "status"}
    assert item["created_at"].endswith("+05:30")


def test_list_ordering_filters_and_pagination(api):
    populate(api)
    items = api.get("/api/recommendations").json()
    ranks = [{"high": 0, "medium": 1, "low": 2}[i["priority"]] for i in items]
    assert ranks == sorted(ranks) and items[0]["priority"] == "high"

    def ids(**params):
        r = api.get("/api/recommendations", params=params)
        assert r.status_code == 200
        return [i["plant_id"] for i in r.json()], int(r.headers["X-Total-Count"])

    assert ids(plantId=5) == ([5], 1)
    assert sorted(ids(priority="high")[0]) == [5, 9]
    assert ids(category="sensor_check") == ([3], 1)
    assert ids(source="heuristic")[1] == 2 and sorted(ids(source="heuristic")[0]) == [3, 9]
    assert ids(status="completed") == ([], 0)
    page, total = ids(limit=2, offset=1)
    assert len(page) == 2 and total == 5 and page == [i["plant_id"] for i in items[1:3]]
    assert api.get("/api/recommendations", params={"limit": 501}).status_code == 422
    assert api.get("/api/recommendations", params={"priority": "critical"}).status_code == 422


def test_superseded_hidden_unless_requested(api):
    populate(api)
    with api.sf() as db:
        put_pred(db, last_readings(db, 13, 1)[0], severity="healthy")
        db.commit()
        build(db, 2)
    assert api.get("/api/recommendations").headers["X-Total-Count"] == "4"
    with_old = api.get("/api/recommendations", params={"includeSuperseded": "true"})
    assert with_old.headers["X-Total-Count"] == "5"
    assert [i["status"] for i in with_old.json() if i["plant_id"] == 13] == ["superseded"]
    assert api.get("/api/recommendations", params={"status": "superseded"}).headers["X-Total-Count"] == "1"


def test_patch_sets_status_user_modified_and_resolved_at(api):
    populate(api)
    rec_id = api.get("/api/recommendations", params={"plantId": 5}).json()[0]["id"]
    done = api.patch(f"/api/recommendations/{rec_id}", json={"status": "completed"})
    assert done.status_code == 200
    body = done.json()
    assert body["id"] == rec_id and body["status"] == "completed" and body["resolved_at"].endswith("+05:30")
    assert body["updated_at"] >= body["created_at"]
    with api.sf() as db:
        assert db.get(Recommendation, int(rec_id.removeprefix("rec-"))).user_modified is True
    again = api.patch(f"/api/recommendations/{rec_id.removeprefix('rec-')}", json={"status": "in_progress"})  # bare number
    assert again.status_code == 200 and again.json()["status"] == "in_progress" and again.json()["resolved_at"] is None
    assert api.patch(f"/api/recommendations/{rec_id}", json={"status": "dismissed"}).json()["resolved_at"] is not None
    # a re-run keeps what the user chose
    with api.sf() as db:
        build(db, 2)
    assert api.get("/api/recommendations", params={"plantId": 5}).json()[0]["status"] == "dismissed"


def test_patch_errors_use_the_error_envelope(api):
    populate(api)
    rec_id = api.get("/api/recommendations").json()[0]["id"]
    bad = api.patch(f"/api/recommendations/{rec_id}", json={"status": "superseded"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "validation_error"
    assert api.patch(f"/api/recommendations/{rec_id}", json={}).status_code == 422
    missing = api.patch("/api/recommendations/rec-99999", json={"status": "completed"})
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    assert api.patch("/api/recommendations/not-an-id", json={"status": "completed"}).status_code == 404
    with api.sf() as db:
        assert db.get(Recommendation, int(rec_id.removeprefix("rec-"))).user_modified is False  # failed PATCH changes nothing


# --- through the pipeline (fake ML) ---------------------------------------------------------
def test_pipeline_run_builds_recommendations_and_statuses(real_settings):
    with running_app(real_settings, FakeML()) as client:
        with client.sf() as db:
            for reading in last_readings(db, 5, 3):  # the fake emits check_watering after 3 consecutive soil < 25
                reading.soil = 20.0
            db.commit()
        client.post("/api/pipeline/run")
        assert wait_done(client)["state"] == "completed"
        items = client.get("/api/recommendations", params={"plantId": 5}).json()
        irrigation = [i for i in items if i["category"] == "irrigation"]
        assert len(irrigation) == 1 and irrigation[0]["source"] == "ml" and irrigation[0]["mlRecId"] == "check_watering"
        assert "Soil moisture 20.0 %" in irrigation[0]["reason"] and "P(" in irrigation[0]["reason"]
        with client.sf() as db:
            statuses = {p.status for p in db.scalars(select(Plant))}
            assert statuses <= {"healthy", "attention", "critical"} and "unmeasured" not in statuses
            active = db.execute(
                select(Recommendation.plant_id, Recommendation.category, func.count())
                .where(Recommendation.status.in_(svc.ACTIVE)).group_by(Recommendation.plant_id, Recommendation.category)
            ).all()  # fmt: skip
            assert active and all(n == 1 for *_, n in active)  # one active rec per (plant, category)
            before = snapshot(db)
        client.post("/api/pipeline/run")  # nothing to predict; recommendations rebuilt from the same data
        wait_done(client)
        with client.sf() as db:
            assert snapshot(db) == before


def test_pipeline_reset_preserves_user_status(real_settings):
    with running_app(real_settings, FakeML()) as client:
        with client.sf() as db:
            for pid in (5, 6):
                for reading in last_readings(db, pid, 3):
                    reading.soil = 20.0
            db.commit()
        client.post("/api/pipeline/run")
        wait_done(client)
        mine = client.get("/api/recommendations", params={"plantId": 5}).json()[0]
        other = client.get("/api/recommendations", params={"plantId": 6}).json()[0]
        assert client.patch(f"/api/recommendations/{mine['id']}", json={"status": "completed"}).status_code == 200
        client.post("/api/pipeline/run", json={"reset": True})
        status = wait_done(client)
        assert status["state"] == "completed" and status["total"] == 5376
        after_mine = client.get("/api/recommendations", params={"plantId": 5}).json()
        assert [(r["id"], r["status"]) for r in after_mine if r["category"] == "irrigation"] == [(mine["id"], "completed")]
        after_other = [r for r in client.get("/api/recommendations", params={"plantId": 6}).json() if r["category"] == "irrigation"]
        assert len(after_other) == 1 and after_other[0]["status"] == "new" and after_other[0]["id"] != other["id"]  # rebuilt
        with client.sf() as db:
            assert db.get(Recommendation, int(mine["id"].removeprefix("rec-"))).prediction_id is not None
            assert db.scalar(select(func.count()).select_from(MlPrediction)) == 5376


# --- reopen / duplicate-active protection ---------------------------------------------------
def completed_then_new(db):
    """A completed for reading R10, then a newer reading R11 produced B (new) for the same plant and category."""
    r10, r11 = last_readings(db, 5, 2)
    rec_in = [ml_rec("check_watering", "soil")]
    put_pred(db, r10, severity="high", recs=rec_in)
    db.commit()
    build(db)
    a = recs_of(db, 5)["irrigation"]
    svc.set_status(db, a, "completed")
    put_pred(db, r11, severity="high", recs=rec_in)
    db.commit()
    assert build(db, 2) == 1
    b = [r for r in db.scalars(select(Recommendation).where(Recommendation.plant_id == 5)) if r.id != a.id][0]
    return a, b, r10, r11


def test_reopening_a_closed_rec_while_another_is_active_is_a_conflict(api):
    with api.sf() as db:
        a, b, *_ = completed_then_new(db)
        a_id, b_id = a.id, b.id
    reopen = api.patch(f"/api/recommendations/rec-{a_id}", json={"status": "in_progress"})
    assert reopen.status_code == 409
    error = reopen.json()["error"]
    assert error["code"] == "conflict" and f"rec-{b_id}" in error["message"] and error["details"] == {"activeId": f"rec-{b_id}"}
    with api.sf() as db:
        assert db.get(Recommendation, a_id).status == "completed"  # unchanged by the failed PATCH
    # closing the active one first makes reopening allowed; changing an active rec's own status is always allowed
    assert api.patch(f"/api/recommendations/rec-{b_id}", json={"status": "in_progress"}).status_code == 200
    assert api.patch(f"/api/recommendations/rec-{b_id}", json={"status": "dismissed"}).status_code == 200
    assert api.patch(f"/api/recommendations/rec-{a_id}", json={"status": "in_progress"}).status_code == 200
    assert api.patch(f"/api/recommendations/rec-{a_id}", json={"status": "new"}).status_code == 200


def test_sync_repairs_a_user_reopened_duplicate(seeded):
    a, b, r10, r11 = completed_then_new(seeded)
    a.status, a.user_modified = "in_progress", True  # what an unguarded PATCH used to allow
    seeded.commit()
    a_id = a.id
    build(seeded, 3)
    rows = seeded.scalars(select(Recommendation).where(Recommendation.plant_id == 5)).all()
    assert [(r.id, r.status, r.source_reading_id) for r in rows] == [(a_id, "in_progress", r11.id)]  # user's kept, moved on
    assert build(seeded, 4) == 0  # and stable


def test_untouched_duplicate_actives_are_superseded_keeping_the_newest(seeded):
    a, b, r10, r11 = completed_then_new(seeded)
    a.status, a.user_modified = "new", False  # two untouched active rows for one (plant, category)
    seeded.commit()
    build(seeded, 3)
    seeded.expire_all()
    assert (seeded.get(Recommendation, a.id).status, seeded.get(Recommendation, b.id).status) == ("superseded", "new")


def test_several_user_modified_duplicates_are_left_with_a_warning(seeded, caplog):
    a, b, r10, r11 = completed_then_new(seeded)
    a.status = "in_progress"
    a.user_modified = b.user_modified = True
    seeded.commit()
    with caplog.at_level("WARNING", logger="app.recommendations"):
        build(seeded, 3)
    seeded.expire_all()
    assert {seeded.get(Recommendation, a.id).status, seeded.get(Recommendation, b.id).status} <= {"in_progress", "new"}
    assert sum(r.status in svc.ACTIVE for r in seeded.scalars(select(Recommendation))) == 2  # nothing destroyed
    assert any("user-modified active recommendations" in m for m in caplog.messages)


# --- foreign keys, CORS, bounds -------------------------------------------------------------
def test_deleting_a_prediction_nulls_recommendation_prediction_id(seeded):
    craft(seeded, 5, [{"severity": "high", "recs": [ml_rec("check_watering", "soil")]}])
    build(seeded)
    rec = recs_of(seeded, 5)["irrigation"]
    pred_id = rec.prediction_id
    assert pred_id is not None
    seeded.execute(delete(MlPrediction).where(MlPrediction.id == pred_id))  # no manual nulling
    seeded.commit()
    seeded.expire_all()
    assert seeded.get(Recommendation, rec.id).prediction_id is None


def test_total_count_header_is_exposed_through_cors(api):
    populate(api)
    response = api.get("/api/recommendations", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    exposed = response.headers["access-control-expose-headers"]
    assert "X-Total-Count" in exposed and response.headers["X-Total-Count"] == "5"


def test_query_and_id_bounds(api):
    for params in ({"plantId": 0}, {"plantId": 17}, {"offset": 100001}, {"offset": -1}, {"limit": 0}):
        r = api.get("/api/recommendations", params=params)
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error", params
    assert api.get("/api/recommendations", params={"offset": 100000}).status_code == 200
    huge = api.patch("/api/recommendations/rec-" + "9" * 40, json={"status": "completed"})
    assert huge.status_code == 404 and huge.json()["error"]["code"] == "not_found"
    assert api.patch("/api/recommendations/" + "9" * 40, json={"status": "completed"}).status_code == 404


# --- on-demand prediction refreshes the plant -------------------------------------------------
def test_on_demand_prediction_of_latest_reading_refreshes_recommendations(api):
    with api.sf() as db:
        for pid in (5, 6):
            for reading in last_readings(db, pid, 3):
                reading.soil = 20.0  # the fake ML service then emits check_watering and severity 'healthy'
        db.commit()
        older_id = last_readings(db, 6, 2)[0].id
    assert api.get("/api/recommendations").headers["X-Total-Count"] == "0"
    assert api.post("/api/predictions", json={"plantId": 6, "readingId": older_id, "explain": False}).status_code == 200
    assert api.get("/api/recommendations").headers["X-Total-Count"] == "0"  # not the latest reading: no refresh
    assert api.post("/api/predictions", json={"plantId": 5, "explain": False}).status_code == 200
    items = api.get("/api/recommendations").json()
    assert [(i["plant_id"], i["category"], i["source"]) for i in items] == [(5, "irrigation", "ml")]
    with api.sf() as db:
        assert db.get(Plant, 5).status == "healthy" and db.get(Plant, 6).status == "unmeasured"
