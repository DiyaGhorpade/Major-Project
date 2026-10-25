"""Real ML service (skipped when it is not reachable at ML_SERVICE_URL, default http://127.0.0.1:8001)."""

import os

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import database, repositories as repo
from app.db.models import MlPrediction, SensorReading
from app.integrations.ml_client import MLClient
from app.integrations.ml_types import PredictRequest
from app.main import create_app
from app.services import history_builder
from app.services.prediction_service import Outcome, persist_prediction, to_ml_reading

ML_URL = os.environ.get("ML_SERVICE_URL", "http://127.0.0.1:8001")


def _ml_ready() -> bool:
    try:
        body = httpx.get(f"{ML_URL}/health", timeout=2).json()
        return body.get("status") == "ok" and body.get("models_loaded") is True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ml_ready(), reason=f"ML service not reachable/ready at {ML_URL}")


@pytest.fixture
def seeded_db(real_settings, db_for):
    from app.services.seed_service import ensure_seeded

    db = db_for(real_settings)
    ensure_seeded(db, real_settings)
    return db


async def test_real_batch_for_two_plants_is_persisted(seeded_db, real_settings):
    db = seeded_db
    ml = MLClient(ML_URL, 30)
    items, readings = [], []
    for plant_id in (3, 10):
        series = repo.readings_for_plant_ordered(db, plant_id)
        for i in (0, 20, 100, len(series) - 1):  # cold start .. latest
            hist = history_builder.history_slice(series, i, 60)
            items.append(PredictRequest(reading=to_ml_reading(series[i]), history=[to_ml_reading(h) for h in hist], explain=i == len(series) - 1))
            readings.append((series[i], len(hist)))
    results = await ml.predict_batch(items)
    await ml.aclose()
    assert [r.status for r in results] == ["ok"] * 8

    for (reading, n), res in zip(readings, results, strict=True):
        persist_prediction(db, reading, Outcome.from_batch(res, n), pipeline_run_id=None)
    db.commit()
    preds = {p.reading_id: p for p in db.scalars(select(MlPrediction))}
    assert len(preds) == 8
    for reading, n in readings:
        p = preds[reading.id]
        assert p.status == "ok" and p.severity in ("healthy", "low", "moderate", "high")
        assert abs(sum(p.probabilities.values()) - 1) < 1e-6 and set(p.probabilities) == {"healthy", "low", "moderate", "high"}
        assert p.confidence == max(p.probabilities.values()) and p.cluster_id is not None and p.model_version
        assert p.run_id and p.history_rows_sent == n and p.stress_type_guess is not None
        assert isinstance(p.is_anomaly, bool) and p.anomaly_score is not None
    explained = [p for p in preds.values() if p.explanation_json]
    assert len(explained) == 2 and len(explained[0].explanation_json["top_features"]) == 5
    assert preds[readings[0][0].id].warnings  # cold start is reported, not hidden


def test_real_on_demand_route_ok_and_rejected(real_settings):
    engine = database.build_engine(real_settings.DATABASE_URL)
    settings = real_settings.model_copy(update={"ML_SERVICE_URL": ML_URL, "ML_TIMEOUT_S": 30})
    with TestClient(create_app(settings, engine)) as client:
        ok = client.post("/api/predictions", json={"plantId": 7, "explain": True})
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["status"] == "ok" and client.get("/api/ai/shap/7").status_code == 200
        with client.app.state.session_factory() as db:
            assert repo.predictions_by_reading(db, [body["readingId"]])[body["readingId"]].history_rows_sent == 60
        assert sum(body["probabilities"].values()) == pytest.approx(1)

        # a stored reading outside the ML's accepted range is rejected by the real service and stored as such
        sf = client.app.state.session_factory
        with sf() as db:
            latest = repo.latest_reading_for_plant(db, 8)
            bad = db.get(SensorReading, latest.id)
            bad.soil = 500.0
            db.commit()
        rejected = client.post("/api/predictions", json={"plantId": 8})
        assert rejected.status_code == 422 and rejected.json()["error"]["code"] == "ml_rejected"
        with sf() as db:
            row = repo.predictions_by_reading(db, [latest.id])[latest.id]
            assert row.status == "rejected" and "soil" in (row.error or "").lower() and row.severity is None
    engine.dispose()
