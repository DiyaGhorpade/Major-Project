"""POST /api/predictions: on-demand prediction. (The GET list/detail views arrive in Phase 3.)"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import sessionmaker

from app.api.deps import get_app_settings, get_ml_client, get_session_factory
from app.core.config import Settings
from app.core.mappings import env_state_label, to_ist_iso
from app.db.models import MlPrediction
from app.integrations.ml_client import MLClient
from app.schemas.prediction import EnvState, PredictionOut, PredictionRequest
from app.services import prediction_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


def to_out(pred: MlPrediction) -> PredictionOut:
    return PredictionOut(
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
        env_state=(
            EnvState(id=pred.cluster_id, distance=pred.cluster_distance, label=env_state_label(pred.cluster_id))
            if pred.cluster_id is not None
            else None
        ),
        anomaly_score=pred.anomaly_score,
        is_anomaly=pred.is_anomaly,
        condition=pred.stress_type_guess,
        explanation=pred.explanation_json,
        ml_recommendations=pred.ml_recommendations_json or [],
        history_rows_sent=pred.history_rows_sent,
    )


@router.post("", response_model=PredictionOut)
async def predict(
    body: PredictionRequest,
    session_factory: sessionmaker = Depends(get_session_factory),
    ml: MLClient = Depends(get_ml_client),
    settings: Settings = Depends(get_app_settings),
) -> PredictionOut:
    """Predict one reading (the plant's latest if `readingId` is omitted) and store the result.

    503 envelope if the ML service is down, 422 envelope if it rejects the input. No DB session is held
    while waiting for the ML service.
    """
    try:
        pred = await prediction_service.predict_one(session_factory, ml, settings, body.plant_id, body.reading_id, body.explain)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return to_out(pred)
