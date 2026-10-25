"""GET /api/data-logs: the same filters as the readings list plus `q` search; `?format=csv` streams a CSV download."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import MAX_OFFSET, ReadingFilters, get_db, get_session_factory, reading_filters
from app.schemas.sensor import SensorRecordOut
from app.services import export_service, sensor_service

router = APIRouter(prefix="/data-logs", tags=["data-logs"])


@router.get("", response_model=list[SensorRecordOut])
def data_logs(
    response: Response,
    filters: ReadingFilters = Depends(reading_filters),
    format: Literal["json", "csv"] = "json",
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    db: Session = Depends(get_db),
    session_factory: sessionmaker = Depends(get_session_factory),
):
    """JSON page (total in `X-Total-Count`), or with `format=csv` every matching row as a streamed download
    (no pagination cap; `limit`/`offset` are ignored)."""
    if format == "csv":
        return StreamingResponse(
            export_service.stream_csv(session_factory, filters),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{export_service.filename()}"'},
        )
    items, total = sensor_service.list_records(db, filters, limit, offset)
    response.headers["X-Total-Count"] = str(total)
    return items
