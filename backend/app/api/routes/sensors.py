"""GET /api/sensors/readings (filtered, paginated) and POST /api/sensors/readings (single or batch ingestion)."""

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Body, Depends, Query, Response
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import MAX_OFFSET, ReadingFilters, get_app_settings, get_cache, get_db, get_ml_client, get_session_factory, reading_filters
from app.core.cache import ResponseCache
from app.core.config import Settings
from app.integrations.ml_client import MLClient
from app.schemas.sensor import IngestBatch, IngestReading, IngestResponse, SensorRecordOut
from app.services import sensor_service

router = APIRouter(prefix="/sensors", tags=["sensors"])


@router.get("/readings", response_model=list[SensorRecordOut])
def list_readings(
    response: Response,
    filters: ReadingFilters = Depends(reading_filters),
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    db: Session = Depends(get_db),
) -> list[SensorRecordOut]:
    """Readings newest first (`order=asc` for oldest first). Total matches in `X-Total-Count`."""
    items, total = sensor_service.list_records(db, filters, limit, offset)
    response.headers["X-Total-Count"] = str(total)
    return items


def _parse_payload(body: Any) -> list[IngestReading]:
    """A single reading object or `{"readings": [...]}` (1-500). Shape errors become the standard 422 envelope."""
    try:
        if isinstance(body, dict) and "readings" in body:
            return IngestBatch.model_validate(body).readings
        return [IngestReading.model_validate(body)]
    except ValidationError as exc:
        raise RequestValidationError([{**e, "loc": ("body", *e["loc"])} for e in exc.errors(include_url=False)]) from exc


@router.post("/readings", status_code=201, response_model=IngestResponse)
async def ingest_readings(
    response: Response,
    background: BackgroundTasks,
    body: Annotated[Any, Body()],
    session_factory: sessionmaker = Depends(get_session_factory),
    ml: MLClient = Depends(get_ml_client),
    settings: Settings = Depends(get_app_settings),
    cache: ResponseCache = Depends(get_cache),
) -> IngestResponse:
    """Ingest one reading or a batch (max 500) from a sensor node.

    Bad shape -> 422. Out-of-range or missing values -> stored as `flagged`. A repeated (plant_id, timestamp)
    is reported as `duplicate` and not inserted. 201 (200 if every reading was a duplicate). New readings are
    then predicted best-effort in the background; an ML outage never affects this response.
    """
    readings = _parse_payload(body)
    result = await sensor_service.ingest_async(session_factory, readings)
    if result.targets:
        cache.clear()
        background.add_task(sensor_service.predict_new_readings, session_factory, ml, settings, result.targets, cache.clear)
    elif result.response.duplicates == len(result.response.items):
        response.status_code = 200
    return result.response
