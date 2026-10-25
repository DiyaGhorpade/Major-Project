"""GET /api/predictions (filtered, paginated list) and POST /api/predictions (on-demand prediction)."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import MAX_OFFSET, get_app_settings, get_cache, get_db, get_ml_client, get_session_factory, parse_time_param
from app.core.cache import ResponseCache
from app.core.config import Settings
from app.integrations.ml_client import MLClient, MLRejected
from app.schemas.prediction import PredictionListItem, PredictionRequest
from app.services import prediction_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=list[PredictionListItem])
def list_predictions(
    response: Response,
    plant_id: Annotated[int | None, Query(alias="plantId", ge=1, le=16)] = None,
    status: Literal["ok", "unavailable", "rejected"] | None = None,
    severity: Literal["healthy", "low", "moderate", "high"] | None = None,
    is_anomaly: Annotated[bool | None, Query(alias="isAnomaly")] = None,
    start: Annotated[str | None, Query(alias="from")] = None,
    end: Annotated[str | None, Query(alias="to")] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    db: Session = Depends(get_db),
) -> list[PredictionListItem]:
    """Stored predictions, newest reading first. Total matches in `X-Total-Count`."""
    items, total = prediction_service.list_items(
        db, plant_id=plant_id, status=status, severity=severity, is_anomaly=is_anomaly,
        start=parse_time_param("from", start), end=parse_time_param("to", end), limit=limit, offset=offset,
    )  # fmt: skip
    response.headers["X-Total-Count"] = str(total)
    return items


@router.post("", response_model=PredictionListItem)
async def predict(
    body: PredictionRequest,
    session_factory: sessionmaker = Depends(get_session_factory),
    ml: MLClient = Depends(get_ml_client),
    settings: Settings = Depends(get_app_settings),
    cache: ResponseCache = Depends(get_cache),
) -> PredictionListItem:
    """Predict one reading (the plant's latest if `readingId` is omitted) and store the result.

    503 envelope if the ML service is down, 422 envelope if it rejects the input. No DB session is held
    while waiting for the ML service.
    """
    try:
        pred = await prediction_service.predict_one(session_factory, ml, settings, body.plant_id, body.reading_id, body.explain)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MLRejected:
        cache.clear()  # a `rejected` prediction row was stored
        raise
    cache.clear()
    return prediction_service.to_item(pred)
