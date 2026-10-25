"""Persisting ML outcomes and on-demand single predictions. Values come from the ML response only;
nothing is ever fabricated or filled in.

Persistence rules (one prediction row per reading, upserted):
* an `unavailable` outcome never replaces an existing `ok` prediction (an outage must not erase good data);
* a new `ok` result without an explanation keeps the stored explanation of the same model run;
* a caller passing no `pipeline_run_id` keeps the existing one.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.mappings import env_state_label, stress_type_guess, to_ist_iso
from app.db import repositories as repo
from app.db.models import AnomalyRecord, MlPrediction, SensorReading
from app.integrations.ml_client import MLClient, MLRejected
from app.integrations.ml_types import BatchResult, MLReading, PredictRequest, PredictResponse, format_rejection, is_data_rejection
from app.schemas.prediction import AnomalyInfo, EnvState, PredictionListItem
from app.services import history_builder, recommendation_service

logger = logging.getLogger("app.predictions")


@dataclass
class Outcome:
    """What happened to one reading: an ML result, a rejection or an outage."""

    status: Literal["ok", "rejected", "unavailable"]
    result: PredictResponse | None = None
    error: str | None = None
    history_rows_sent: int | None = None

    @classmethod
    def from_batch(cls, item: BatchResult, history_rows_sent: int) -> "Outcome":
        return cls(item.status, item.result, item.error_text, history_rows_sent=history_rows_sent)


def effective_explain_mode(settings: Settings) -> str:
    """`latest_only` or `none` (unknown values fall back to `latest_only`)."""
    mode = settings.EXPLAIN_MODE.lower()
    return mode if mode in ("latest_only", "none") else "latest_only"


def env_state(pred: MlPrediction) -> EnvState | None:
    """The KMeans environmental state of a prediction (never a stress level)."""
    if pred.cluster_id is None:
        return None
    return EnvState(id=pred.cluster_id, distance=pred.cluster_distance, label=env_state_label(pred.cluster_id))


def to_item(pred: MlPrediction) -> PredictionListItem:
    """The one API shape of a stored prediction (GET list items and the POST response)."""
    return PredictionListItem(
        id=pred.id,
        reading_id=pred.reading_id,
        record_id=f"R-{pred.reading_id}",
        plant_id=pred.plant_id,
        timestamp=to_ist_iso(pred.reading_timestamp) or "",
        status=pred.status,
        error=pred.error,
        warnings=pred.warnings or [],
        model_version=pred.model_version,
        run_id=pred.run_id,
        severity=pred.severity,
        probabilities=pred.probabilities,
        confidence=pred.confidence,
        condition=pred.stress_type_guess,
        env_state=env_state(pred),
        anomaly=AnomalyInfo(is_anomaly=pred.is_anomaly, score=pred.anomaly_score),
    )


def list_items(db: Session, **filters) -> tuple[list[PredictionListItem], int]:
    rows, total = repo.list_predictions(db, **filters)
    return [to_item(p) for p in rows], total


def to_ml_reading(reading: SensorReading) -> MLReading:
    return MLReading(
        timestamp=to_ist_iso(reading.timestamp) or "",
        plant_id=reading.plant_id,
        soil=reading.soil,
        temperature=reading.temperature,
        humidity=reading.humidity,
        light=reading.light,
    )


def _apply(pred: MlPrediction, reading: SensorReading, outcome: Outcome, pipeline_run_id: int | None) -> None:
    result = outcome.result if outcome.status == "ok" else None
    kept_explanation = None
    if result is not None and not result.explanation.available and pred.explanation_json:
        if pred.run_id and pred.run_id == result.model_metadata.run_id:
            kept_explanation = pred.explanation_json  # same model: an explain=false call must not erase it
    pred.plant_id = reading.plant_id
    pred.reading_timestamp = reading.timestamp
    pred.status = outcome.status
    pred.error = outcome.error
    if pipeline_run_id is not None:
        pred.pipeline_run_id = pipeline_run_id
    pred.history_rows_sent = outcome.history_rows_sent
    # Result columns are reset first so a retried reading never keeps stale values.
    pred.warnings = list(result.warnings) if result else None
    pred.model_version = result.model_metadata.model_version if result else None
    pred.run_id = result.model_metadata.run_id if result else None
    pred.severity = result.stress_prediction.label if result else None
    pred.probabilities = dict(result.stress_prediction.probabilities) if result else None
    pred.confidence = max(result.stress_prediction.probabilities.values(), default=None) if result else None
    pred.cluster_id = result.cluster.id if result else None
    pred.cluster_distance = result.cluster.distance if result else None
    pred.anomaly_score = result.anomaly.score if result else None
    pred.is_anomaly = result.anomaly.is_anomaly if result else None
    pred.stress_type_guess = stress_type_guess(pred.severity, [r.category for r in result.recommendations]) if result else None
    if result is not None and result.explanation.available:
        pred.explanation_json = result.explanation.model_dump(mode="json")
    else:
        pred.explanation_json = kept_explanation
    pred.ml_recommendations_json = [r.model_dump(mode="json") for r in result.recommendations] if result else None


def persist_predictions(
    db: Session, pairs: Sequence[tuple[SensorReading, Outcome]], pipeline_run_id: int | None = None
) -> None:
    """Upsert the prediction row (unique reading_id) and anomaly record of each reading using a few
    batched queries. Flushes but does not commit, so many readings share one short transaction."""
    existing = repo.predictions_by_reading(db, [reading.id for reading, _ in pairs])
    rows: list[tuple[MlPrediction, Outcome]] = []
    for reading, outcome in pairs:
        pred = existing.get(reading.id)
        if pred is None:
            pred = MlPrediction(reading_id=reading.id)
            db.add(pred)
        elif outcome.status == "unavailable" and pred.status == "ok":
            continue  # keep the good prediction; the outage is reported by the run, not stored over it
        _apply(pred, reading, outcome, pipeline_run_id)
        rows.append((pred, outcome))
    db.flush()
    # Exactly one anomaly record per anomalous ok prediction; none otherwise (idempotent).
    repo.delete_anomalies_for_predictions(db, [pred.id for pred, _ in rows])
    for pred, outcome in rows:
        result = outcome.result
        if outcome.status == "ok" and result is not None and result.anomaly.is_anomaly:
            db.add(
                AnomalyRecord(
                    prediction_id=pred.id,
                    reading_id=pred.reading_id,
                    plant_id=pred.plant_id,
                    reading_timestamp=pred.reading_timestamp,
                    score=result.anomaly.score,
                    threshold=result.anomaly.threshold,
                    is_anomaly=True,
                    details=result.anomaly.interpretation or None,
                )
            )


def persist_prediction(
    db: Session, reading: SensorReading, outcome: Outcome, pipeline_run_id: int | None = None
) -> MlPrediction:
    """Single-reading form of `persist_predictions` (returns the row now stored for the reading)."""
    persist_predictions(db, [(reading, outcome)], pipeline_run_id)
    return repo.predictions_by_reading(db, [reading.id])[reading.id]


async def history_rows_and_run(ml: MLClient, settings: Settings) -> tuple[int, str]:
    """`(history_rows, ml_run_id)` from the ML YAML cap and the cached `/metadata` (DEV-1)."""
    max_history, _ = await asyncio.to_thread(history_builder.read_ml_limits, settings.ML_CONFIG_PATH)  # cached; stat only
    metadata = await ml.cached_metadata()
    rows = history_builder.history_rows_for(
        max_history, metadata.required_history_rows_for_full_features, settings.HISTORY_ROWS_DEFAULT
    )
    return rows, metadata.run_id


def _load_inputs(
    session_factory: sessionmaker, plant_id: int, reading_id: int | None, history_rows: int
) -> tuple[SensorReading, MLReading, list[MLReading]]:
    with session_factory() as db:
        reading = repo.get_reading(db, reading_id) if reading_id is not None else repo.latest_reading_for_plant(db, plant_id)
        if reading is None or reading.plant_id != plant_id:
            raise LookupError(f"no reading found for plant {plant_id}" + (f" with id {reading_id}" if reading_id else ""))
        history = history_builder.history_for_reading(db, reading, history_rows)
        return reading, to_ml_reading(reading), [to_ml_reading(h) for h in history]


def _store(session_factory: sessionmaker, reading: SensorReading, outcome: Outcome) -> MlPrediction:
    """Persist the outcome, then refresh the plant's recommendations/status when this is its newest reading.

    The refresh is skipped while a pipeline run is queued or running: the run's own rebuild at the end reads the
    stored predictions (this one included) and would otherwise race this write on the unique recommendation key
    (DEV-27). It runs in a separate transaction, so a failure there never loses the prediction.
    """
    with session_factory() as db:
        pred = persist_prediction(db, reading, outcome)
        latest = repo.latest_reading_for_plant(db, reading.plant_id)
        refresh = outcome.status == "ok" and latest is not None and latest.id == reading.id
        db.commit()
        if refresh and repo.active_run(db) is None:
            try:
                recommendation_service.refresh_plant(db, reading.plant_id)  # the plant's current state just changed
            except Exception:
                db.rollback()
                db.refresh(pred)  # the rollback expired it; the prediction itself is committed
                logger.exception("could not refresh recommendations of plant %d; the next Run rebuilds them", reading.plant_id)
        return pred


async def predict_one(
    session_factory: sessionmaker,
    ml: MLClient,
    settings: Settings,
    plant_id: int,
    reading_id: int | None = None,
    explain: bool = True,
) -> MlPrediction:
    """On-demand prediction for one reading (the plant's latest if none given), history auto-fetched.

    No DB session is held across the ML call (short read, await, short write). `EXPLAIN_MODE=none`
    forces `explain=false`. Returns the stored prediction. Raises `LookupError` for an unknown
    plant/reading, `MLRejected` (after storing a `rejected` row when the rejection is about the reading's data) or `MLUnavailable` (nothing stored).
    """
    history_rows, _ = await history_rows_and_run(ml, settings)
    reading, ml_reading, history = await asyncio.to_thread(_load_inputs, session_factory, plant_id, reading_id, history_rows)
    want_explain = explain and effective_explain_mode(settings) != "none"
    request = PredictRequest(reading=ml_reading, history=history, explain=want_explain)
    try:
        result = await ml.predict(request)
    except MLRejected as exc:
        if not is_data_rejection(exc.code):
            raise  # a config/request problem says nothing about this reading: store nothing (DEV-51)
        outcome = Outcome("rejected", error=format_rejection(exc.code, exc.message, exc.issues), history_rows_sent=len(history))
        await asyncio.to_thread(_store, session_factory, reading, outcome)
        raise
    return await asyncio.to_thread(_store, session_factory, reading, Outcome("ok", result, history_rows_sent=len(history)))
