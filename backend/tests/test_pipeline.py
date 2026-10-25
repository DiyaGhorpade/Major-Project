"""Pipeline run orchestration against a fake ML client and the real seeded dataset (5,376 readings)."""

import asyncio
import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db import database
from app.db.models import AnomalyRecord, MlPrediction, ModelInfo, PipelineRun, SensorReading
from app.main import create_app
from app.services.pipeline_service import PipelineService
from app.services.prediction_service import Outcome, persist_prediction
from app.services.seed_service import ensure_seeded
from tests.fakes import RUN_ID, FakeML, make_response
from app.integrations.ml_types import PredictResponse

TOTAL = 5376


@contextmanager
def running_app(settings, fake: FakeML):
    engine = database.build_engine(settings.DATABASE_URL)
    app = create_app(settings, engine, ml_client=fake)
    with TestClient(app) as client:
        client.sf = app.state.session_factory
        yield client
    engine.dispose()


@pytest.fixture
def fake() -> FakeML:
    return FakeML()


@pytest.fixture
def app_client(real_settings, fake):
    with running_app(real_settings, fake) as client:
        yield client


def wait_done(client, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/pipeline/status").json()
        if status["state"] in ("completed", "failed"):
            return status
        time.sleep(0.05)
    raise AssertionError(f"run did not finish: {status}")


def counts(client):
    with client.sf() as db:
        by_status = dict(db.execute(select(MlPrediction.status, func.count()).group_by(MlPrediction.status)).all())
        distinct = db.scalar(select(func.count(func.distinct(MlPrediction.reading_id))))
        total = db.scalar(select(func.count()).select_from(MlPrediction))
    return by_status, distinct, total


def test_status_is_idle_before_any_run(app_client):
    status = app_client.get("/api/pipeline/status").json()
    assert status["state"] == "idle" and status["runId"] is None
    assert status["readyToRun"] is True and status["percent"] == 0 and status["mlServiceOk"] is True
    assert set(status) == {
        "runId", "state", "total", "processed", "failed", "percent", "currentPlantId", "startedAt",
        "finishedAt", "message", "mlServiceOk", "modelVersion", "readyToRun",
    }  # fmt: skip


def test_full_run_completes_with_all_predictions(app_client, fake):
    response = app_client.post("/api/pipeline/run")
    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"runId", "state"} and body["state"] in ("queued", "running")

    status = wait_done(app_client)
    assert status["state"] == "completed" and status["runId"] == body["runId"]
    assert status["total"] == TOTAL and status["processed"] == TOTAL and status["failed"] == 0
    assert status["percent"] == 100 and status["currentPlantId"] is None
    assert status["startedAt"].endswith("+05:30") and status["finishedAt"].endswith("+05:30")
    assert status["modelVersion"] == "1.0.0" and status["readyToRun"] is True

    by_status, distinct, total = counts(app_client)
    assert by_status == {"ok": TOTAL} and distinct == total == TOTAL
    assert fake.max_items_seen <= 64
    assert max(len(r.history) for r in fake.seen) == 60  # DEV-1: min(500, max(49, 60))
    with app_client.sf() as db:
        assert db.scalar(select(func.count()).select_from(ModelInfo)) == 1
        info = db.scalars(select(ModelInfo)).one()
        assert info.run_id == RUN_ID and info.labels == ["healthy", "low", "moderate", "high"]
        # explanations: only the latest reading of each plant
        explained = db.scalars(select(MlPrediction).where(MlPrediction.explanation_json.is_not(None))).all()
        assert len(explained) == 16
        # anomaly records mirror is_anomaly=true predictions exactly
        flagged = db.scalar(select(func.count()).select_from(MlPrediction).where(MlPrediction.is_anomaly.is_(True)))
        assert flagged > 0 and db.scalar(select(func.count()).select_from(AnomalyRecord)) == flagged
        first = db.scalars(select(MlPrediction).join(SensorReading).where(SensorReading.plant_id == 3).order_by(SensorReading.timestamp)).first()
        assert first.history_rows_sent == 0 and first.warnings == ["cold start"]
        assert first.confidence == 0.7 and first.cluster_id is not None and first.model_version == "1.0.0"
        assert first.stress_type_guess in ("Healthy", "Water Stress", "Unclassified Stress")


def test_explanation_only_for_latest_reading_per_plant_and_none_mode(real_settings, fake):
    settings = real_settings.model_copy(update={"EXPLAIN_MODE": "none"})
    with running_app(settings, fake) as client:
        client.post("/api/pipeline/run")
        assert wait_done(client)["state"] == "completed"
        with client.sf() as db:
            assert db.scalar(select(func.count()).select_from(MlPrediction).where(MlPrediction.explanation_json.is_not(None))) == 0


def test_latest_reading_per_plant_gets_the_explanation(app_client):
    app_client.post("/api/pipeline/run")
    wait_done(app_client)
    from app.db import repositories as repo

    with app_client.sf() as db:
        latest = {r.id for r in repo.latest_reading_per_plant(db)}
        explained = {p.reading_id for p in db.scalars(select(MlPrediction).where(MlPrediction.explanation_json.is_not(None)))}
    assert explained == latest and len(latest) == 16


def test_second_post_while_running_returns_same_run(app_client, fake):
    fake.gate.clear()
    first = app_client.post("/api/pipeline/run").json()
    second = app_client.post("/api/pipeline/run", json={"reset": True}).json()
    assert second["runId"] == first["runId"]
    status = app_client.get("/api/pipeline/status").json()
    assert status["state"] in ("queued", "running") and status["readyToRun"] is False
    fake.gate.set()
    assert wait_done(app_client)["state"] == "completed"
    with app_client.sf() as db:
        assert db.scalar(select(func.count()).select_from(PipelineRun)) == 1


def test_running_twice_does_not_duplicate(app_client, fake):
    app_client.post("/api/pipeline/run")
    wait_done(app_client)
    batches_after_first = fake.batches
    second = app_client.post("/api/pipeline/run").json()
    status = wait_done(app_client)
    assert status["runId"] == second["runId"] and status["state"] == "completed"
    assert status["total"] == 0 and status["percent"] == 100
    assert fake.batches == batches_after_first  # nothing was re-predicted
    by_status, distinct, total = counts(app_client)
    assert by_status == {"ok": TOTAL} and distinct == total == TOTAL


def test_reset_recomputes_everything(app_client, fake):
    app_client.post("/api/pipeline/run")
    wait_done(app_client)
    with app_client.sf() as db:
        first_run = set(db.scalars(select(MlPrediction.pipeline_run_id)))
    batches = fake.batches
    app_client.post("/api/pipeline/run", json={"reset": True})
    status = wait_done(app_client)
    assert status["state"] == "completed" and status["total"] == TOTAL
    assert fake.batches == batches + 84  # 5376 / 64
    by_status, distinct, total = counts(app_client)
    assert by_status == {"ok": TOTAL} and distinct == total == TOTAL
    with app_client.sf() as db:
        assert first_run == {1} and set(db.scalars(select(MlPrediction.pipeline_run_id))) == {2}  # all rows recomputed
        assert db.scalar(select(func.count()).select_from(AnomalyRecord)) == db.scalar(
            select(func.count()).select_from(MlPrediction).where(MlPrediction.is_anomaly.is_(True))
        )


def test_rejected_readings_are_stored_and_skipped_on_rerun(app_client, fake):
    fake.reject_if = lambda req: req.reading.plant_id == 5 and req.reading.soil is None
    app_client.post("/api/pipeline/run")
    status = wait_done(app_client)
    by_status, _, total = counts(app_client)
    assert status["state"] == "completed" and total == TOTAL
    assert by_status["rejected"] > 0 and status["failed"] == by_status["rejected"]
    with app_client.sf() as db:
        rejected = db.scalars(select(MlPrediction).where(MlPrediction.status == "rejected")).first()
        assert rejected.error.startswith("invalid_readings") and rejected.severity is None and rejected.probabilities is None
    batches = fake.batches
    app_client.post("/api/pipeline/run")
    assert wait_done(app_client)["total"] == 0 and fake.batches == batches


def test_ml_down_at_preflight_fails_with_clear_message(app_client, fake):
    fake.up = False
    app_client.post("/api/pipeline/run")
    status = wait_done(app_client)
    assert status["state"] == "failed" and "ML service not reachable at http://fake-ml:8001" in status["message"]
    assert status["mlServiceOk"] is False and status["readyToRun"] is True and status["finishedAt"]
    assert counts(app_client)[2] == 0


def test_models_unavailable_at_preflight(app_client, fake):
    fake.models_error = "sklearn 1.9 required"
    app_client.post("/api/pipeline/run")
    status = wait_done(app_client)
    assert status["state"] == "failed" and status["message"] == "ML models unavailable: sklearn 1.9 required"


def test_ml_dies_mid_run_keeps_partial_results_and_next_run_recovers(app_client, fake):
    fake.die_after_batches = 10
    app_client.post("/api/pipeline/run")
    status = wait_done(app_client)
    assert status["state"] == "failed"
    assert "became unavailable" in status["message"] and "not reachable" in status["message"]
    by_status, distinct, total = counts(app_client)
    assert by_status["ok"] == 10 * 64 and by_status["unavailable"] >= 3 * 64 and total == distinct
    assert 0 < total < TOTAL and status["failed"] == by_status["unavailable"]
    assert by_status["unavailable"] <= (3 + 4) * 64  # 3 consecutive down batches + those already in flight (ML_CONCURRENCY=4)
    assert f"processing {status['processed']} of {TOTAL}" in status["message"]
    # the app keeps serving
    assert app_client.get("/api/health").status_code == 200
    fake.invalidate_cache()  # the health check is cached for ~5 s
    assert app_client.get("/api/pipeline/status").json()["mlServiceOk"] is False

    # ML comes back: the next run retries only what is missing or unavailable, with no duplicates
    fake.up, fake.die_after_batches = True, None
    fake.invalidate_cache()
    app_client.post("/api/pipeline/run")
    status = wait_done(app_client)
    assert status["state"] == "completed" and status["total"] == TOTAL - 10 * 64 and status["failed"] == 0
    by_status, distinct, total = counts(app_client)
    assert by_status == {"ok": TOTAL} and distinct == total == TOTAL
    with app_client.sf() as db:
        assert db.scalar(select(func.count()).select_from(MlPrediction).where(MlPrediction.error.is_not(None))) == 0


def test_startup_marks_stale_runs_failed(real_settings, fake):
    engine = database.build_engine(real_settings.DATABASE_URL)
    database.Base.metadata.create_all(engine)
    sf = database.build_session_factory(engine)
    with sf() as db:
        db.add_all([PipelineRun(status="running", message="Predicting"), PipelineRun(status="queued"), PipelineRun(status="completed")])
        db.commit()
    engine.dispose()
    with running_app(real_settings, fake) as client:
        with client.sf() as db:
            statuses = [r.status for r in db.scalars(select(PipelineRun).order_by(PipelineRun.id))]
            stale = db.scalars(select(PipelineRun).where(PipelineRun.status == "failed")).all()
        assert statuses == ["failed", "failed", "completed"]
        assert all(r.message == "Run interrupted by a backend restart" and r.finished_at for r in stale)
        status = client.get("/api/pipeline/status").json()
        assert status["state"] == "completed" and status["readyToRun"] is True


def test_batch_size_comes_from_ml_yaml(real_settings, fake):
    real_settings.ML_CONFIG_PATH.write_text("api:\n  max_history_rows: 30\n  max_batch_items: 20\n")
    with running_app(real_settings, fake) as client:
        client.post("/api/pipeline/run")
        assert wait_done(client)["state"] == "completed"
    assert fake.max_items_seen == 20
    assert max(len(r.history) for r in fake.seen) == 30  # never above the ML history cap


def test_histories_are_same_plant_and_strictly_earlier(app_client, fake):
    app_client.post("/api/pipeline/run")
    wait_done(app_client)
    seen = fake.seen
    assert len(seen) == TOTAL
    for req in seen:  # every single request, not a sample
        assert all(h.plant_id == req.reading.plant_id for h in req.history)
        assert all(h.timestamp < req.reading.timestamp for h in req.history)
        assert len(req.history) <= 60
        assert req.explain is not None  # bulk calls say explain=false explicitly
    assert sum(1 for r in seen if len(r.history) == 0) == 16  # one cold start per plant
    assert sum(1 for r in seen if len(r.history) == 60) == 16 * (336 - 60)  # every later reading gets the full DEV-1 window
    assert any(h.soil is None for req in seen for h in req.history)  # null rows are sent, never filled
    assert sum(1 for r in seen if r.explain) == 16


def test_health_reports_ml_and_model_info(app_client):
    app_client.post("/api/pipeline/run")
    wait_done(app_client)
    body = app_client.get("/api/health").json()
    assert body["mlService"]["ok"] is True and body["mlService"]["modelsLoaded"] is True
    assert body["mlService"]["explainabilityAvailable"] is True
    assert body["modelVersion"] == "1.0.0" and body["runId"] == RUN_ID


def test_dataset_validation_summary_is_stored_best_effort(app_client):
    deadline = time.time() + 10
    info = None
    while time.time() < deadline and info is None:
        with app_client.sf() as db:
            info = db.scalars(select(ModelInfo)).first()
        time.sleep(0.05)
    assert info is not None and info.metadata_json["dataset_validation"]["n_readings"] == TOTAL


def test_on_demand_prediction_route(app_client, fake):
    created = app_client.post("/api/predictions", json={"plantId": 4, "explain": True})
    assert created.status_code == 200
    body = created.json()
    assert body["plantId"] == 4 and body["status"] == "ok" and body["severity"] in ("healthy", "low", "moderate", "high")
    assert body["explanation"]["available"] is True and body["historyRowsSent"] == 60
    assert body["envState"]["label"].startswith("Environmental state group")
    again = app_client.post("/api/predictions", json={"plantId": 4}).json()
    assert again["id"] == body["id"]  # upsert, no duplicate
    assert app_client.post("/api/predictions", json={"plantId": 99}).status_code == 404
    assert app_client.post("/api/predictions", json={"plantId": 4, "readingId": 1}).status_code == 404  # reading 1 is plant 1's

    fake.reject_if = lambda req: True
    rejected = app_client.post("/api/predictions", json={"plantId": 4})
    assert rejected.status_code == 422 and rejected.json()["error"]["code"] == "ml_rejected"
    fake.reject_if = lambda req: False
    fake.up = False
    down = app_client.post("/api/predictions", json={"plantId": 4})
    assert down.status_code == 503 and down.json()["error"]["code"] == "ml_unavailable"


async def test_concurrent_starts_share_one_run_and_one_task(real_settings, fake):
    """Five simultaneous starts: without the service's asyncio lock each would see 'no active run' and
    create its own run; with it they all get the first run's id."""
    engine = database.build_engine(real_settings.DATABASE_URL)
    database.Base.metadata.create_all(engine)
    sf = database.build_session_factory(engine)
    with sf() as db:
        ensure_seeded(db, real_settings)
    service = PipelineService(sf, real_settings, fake)
    fake.gate.clear()  # keep the run alive while the other starts arrive
    results = await asyncio.gather(*(service.start() for _ in range(5)))
    task = service._task
    assert len({run_id for run_id, _ in results}) == 1
    # the creator reports "queued"; the others report the run's current DB state (queued or running)
    assert all(state in ("queued", "running") for _, state in results)
    with sf() as db:
        assert db.scalar(select(func.count()).select_from(PipelineRun)) == 1
    fake.gate.set()
    await service.wait()
    assert service._task is task
    with sf() as db:
        assert db.scalars(select(PipelineRun)).one().status == "completed"
    # a start after completion begins a NEW run
    again, _ = await service.start()
    await service.wait()
    assert again != results[0][0]
    engine.dispose()


def test_batch_500_falls_back_to_per_item_predict(app_client, fake):
    fake.batch_status = 500
    fake.item_500_if = lambda req: req.reading.plant_id == 2 and req.reading.soil is None
    app_client.post("/api/pipeline/run")
    status = wait_done(app_client)
    by_status, distinct, total = counts(app_client)
    assert status["state"] == "completed" and fake.batches == 0 and fake.predict_calls == TOTAL
    assert by_status["unavailable"] == status["failed"] > 0 and by_status["ok"] == TOTAL - by_status["unavailable"]
    with app_client.sf() as db:
        bad = db.scalars(select(MlPrediction).where(MlPrediction.status == "unavailable")).first()
        assert "ML internal error" in bad.error and bad.severity is None
    assert distinct == total == TOTAL


def test_outage_never_overwrites_ok_predictions(real_settings, fake):
    with running_app(real_settings, fake) as client:
        client.post("/api/pipeline/run")
        assert wait_done(client)["state"] == "completed"
        # a "new model" makes every reading stale; the ML service then dies on the first batch
        fake.ml_run_id, fake.die_after_batches = "model_2", 0
        fake.invalidate_cache()
        client.post("/api/pipeline/run")
        status = wait_done(client)
        assert status["state"] == "failed" and status["failed"] > 0
        with client.sf() as db:
            rows = db.execute(select(MlPrediction.status, MlPrediction.run_id, func.count()).group_by(MlPrediction.status, MlPrediction.run_id)).all()
        assert rows == [("ok", RUN_ID, TOTAL)]  # the old good predictions are all still there


def _ok(reading, *, explain, run_id=RUN_ID) -> Outcome:
    from app.integrations.ml_types import PredictRequest
    from app.services.prediction_service import to_ml_reading

    req = PredictRequest(reading=to_ml_reading(reading), history=[], explain=explain)
    return Outcome("ok", PredictResponse.model_validate(make_response(req, anomaly=False, explain=explain, run_id=run_id)), history_rows_sent=0)


def test_persist_rules_unavailable_explanation_and_run_id(real_settings, db_for):
    db = db_for(real_settings)
    ensure_seeded(db, real_settings)
    reading = db.scalars(select(SensorReading).order_by(SensorReading.id)).first()

    persist_prediction(db, reading, _ok(reading, explain=True), pipeline_run_id=None)
    first = persist_prediction(db, reading, Outcome("unavailable", error="down"), pipeline_run_id=7)
    db.commit()
    assert first.status == "ok" and first.error is None and first.explanation_json  # outage did not overwrite

    # a later ok without explanation from the same model keeps the explanation and the old pipeline_run_id
    pred = persist_prediction(db, reading, _ok(reading, explain=False), pipeline_run_id=None)
    assert pred.explanation_json and pred.explanation_json["available"] is True
    pred.pipeline_run_id = None
    persist_prediction(db, reading, _ok(reading, explain=False), pipeline_run_id=None)
    # a different model run does not inherit the old explanation
    pred = persist_prediction(db, reading, _ok(reading, explain=False, run_id="model_2"), pipeline_run_id=None)
    assert pred.explanation_json is None and pred.run_id == "model_2"
    db.commit()
    assert db.scalar(select(func.count()).select_from(MlPrediction)) == 1

    # a rejection may replace an ok prediction (it is deterministic information about the input)
    rejected = persist_prediction(db, reading, Outcome("rejected", error="invalid_readings: x"), pipeline_run_id=None)
    assert rejected.status == "rejected" and rejected.severity is None


def test_pipeline_run_id_survives_a_run_less_upsert(real_settings, db_for):
    db = db_for(real_settings)
    ensure_seeded(db, real_settings)
    reading = db.scalars(select(SensorReading).order_by(SensorReading.id)).first()
    db.add(PipelineRun(status="completed"))
    db.commit()
    persist_prediction(db, reading, _ok(reading, explain=False), pipeline_run_id=1)
    pred = persist_prediction(db, reading, _ok(reading, explain=False), pipeline_run_id=None)
    assert pred.pipeline_run_id == 1


def test_on_demand_explain_false_keeps_stored_explanation(app_client, fake):
    first = app_client.post("/api/predictions", json={"plantId": 6, "explain": True}).json()
    assert first["explanation"]["available"] is True and fake.seen[-1].explain is True
    second = app_client.post("/api/predictions", json={"plantId": 6, "explain": False}).json()
    assert fake.seen[-1].explain is False
    assert second["id"] == first["id"] and second["explanation"]["available"] is True


def test_explain_mode_none_forces_explain_false_on_demand(real_settings, fake):
    with running_app(real_settings.model_copy(update={"EXPLAIN_MODE": "none"}), fake) as client:
        body = client.post("/api/predictions", json={"plantId": 6, "explain": True}).json()
        assert fake.seen[-1].explain is False and body["explanation"] is None


def test_on_demand_uses_cached_metadata_and_no_session_across_await(app_client, fake):
    calls = []
    original = fake.metadata

    async def counting():
        calls.append(1)
        return await original()

    fake.metadata = counting
    fake.invalidate_cache()
    for _ in range(3):
        assert app_client.post("/api/predictions", json={"plantId": 2}).status_code == 200
    assert len(calls) == 1  # /metadata fetched once, then served from the cache
