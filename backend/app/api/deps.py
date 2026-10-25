"""FastAPI dependencies: DB session, settings, ML client, pipeline service."""

import re
from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Literal

from fastapi import HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.core.cache import ResponseCache
from app.core.config import Settings
from app.core.mappings import parse_timestamp
from app.db.repositories import ReadingFilters
from app.integrations.ml_client import MLClient
from app.services.pipeline_service import PipelineService


def get_db(request: Request) -> Iterator[Session]:
    """One SQLAlchemy session per request, taken from the app's session factory."""
    db = request.app.state.session_factory()
    try:
        yield db
    finally:
        db.close()


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_ml_client(request: Request) -> MLClient:
    """The app-wide ML client (tests replace `app.state.ml_client` with a fake)."""
    return request.app.state.ml_client


def get_pipeline_service(request: Request) -> PipelineService:
    return request.app.state.pipeline


def get_session_factory(request: Request):
    """The app's session factory, for code that must not hold a session across an `await`."""
    return request.app.state.session_factory


def get_cache(request: Request) -> ResponseCache:
    """The app-wide response cache (see `core/cache.py`)."""
    return request.app.state.cache


# --- shared query parameters -------------------------------------------------------------------
_SPACE_BEFORE_OFFSET = re.compile(r"^(.*\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?) (\d{2}:\d{2})$")
MAX_OFFSET = 100_000


def parse_time_param(name: str, value: str | None) -> datetime | None:
    """Parse an ISO-8601 query value (naive = IST). A `+` in a URL is often decoded to a space; accept that."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    if m := _SPACE_BEFORE_OFFSET.match(text):
        text = f"{m.group(1)}+{m.group(2)}"
    parsed = parse_timestamp(text)
    if parsed is None:
        raise HTTPException(
            status_code=422,
            detail={"code": "validation_error", "message": f"'{name}' must be an ISO-8601 date-time (naive values are IST)",
                    "details": [{"loc": ["query", name], "message": f"invalid date-time: {value!r}", "type": "datetime_parsing"}]},
        )  # fmt: skip
    return parsed


def reading_filters(
    plant_id: Annotated[int | None, Query(alias="plantId", ge=1, le=16)] = None,
    cluster: Literal["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"] | None = None,
    node_id: Annotated[Literal["1", "2"] | None, Query(alias="nodeId")] = None,
    status: Literal["accepted", "flagged", "rejected"] | None = None,
    session_id: Annotated[int | None, Query(alias="sessionId", ge=1)] = None,
    start: Annotated[str | None, Query(alias="from")] = None,
    end: Annotated[str | None, Query(alias="to")] = None,
    q: Annotated[str | None, Query(max_length=100)] = None,
    order: Literal["asc", "desc"] = "desc",
) -> ReadingFilters:
    """Filters shared by `/api/sensors/readings` and `/api/data-logs`."""
    return ReadingFilters(
        plant_id=plant_id, cluster=cluster, node_id=node_id, status=status, session_id=session_id,
        start=parse_time_param("from", start), end=parse_time_param("to", end), q=q, order=order,
    )  # fmt: skip
