"""GET /api/recommendations (filtered, paginated list) and PATCH /api/recommendations/{id} (user status)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from app.api.deps import MAX_OFFSET, get_app_settings, get_cache, get_db
from app.core.cache import ResponseCache
from app.core.config import Settings
from app.core.plant_catalog import PLANT_IDS
from app.schemas.recommendation import (
    Category,
    Priority,
    RecommendationOut,
    RecommendationStatusUpdate,
    RecStatus,
    Source,
)
from app.services import recommendation_service as service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("", response_model=list[RecommendationOut])
def list_recommendations(
    response: Response,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
    plant_id: Annotated[int | None, Query(alias="plantId", ge=min(PLANT_IDS), le=max(PLANT_IDS))] = None,
    status: RecStatus | None = None,
    priority: Priority | None = None,
    category: Category | None = None,
    source: Source | None = None,
    include_superseded: Annotated[bool, Query(alias="includeSuperseded")] = False,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
) -> list[RecommendationOut]:
    """Ordered high -> low priority, then newest first. The total number of matches is in `X-Total-Count`."""
    items, total = service.list_out(
        db, settings.ML_CONFIG_PATH, plant_id=plant_id, status=status, priority=priority, category=category,
        source=source, include_superseded=include_superseded, limit=limit, offset=offset,
    )  # fmt: skip
    response.headers["X-Total-Count"] = str(total)
    return items


@router.patch("/{rec_id}", response_model=RecommendationOut)
def update_recommendation(
    rec_id: str,
    body: RecommendationStatusUpdate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
    cache: ResponseCache = Depends(get_cache),
) -> RecommendationOut:
    """Set the status (`new|in_progress|completed|dismissed`); the change survives later pipeline runs.
    409 if reopening it would leave two active recommendations for the same plant and category."""
    try:
        out = service.update_status(db, rec_id, body.status, settings.ML_CONFIG_PATH)
        cache.clear()  # the overview lists active recommendations
        return out
    except service.RecommendationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except service.RecommendationConflict as exc:
        raise HTTPException(status_code=409, detail={"code": "conflict", "message": str(exc), "details": {"activeId": f"rec-{exc.other_id}"}}) from exc
