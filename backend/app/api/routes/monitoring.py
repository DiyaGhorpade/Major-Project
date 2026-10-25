"""GET /api/monitoring/overview (cached per data state) and /api/monitoring/plants/{id}."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_app_settings, get_cache, get_db, get_ml_client, get_session_factory
from app.core.cache import ResponseCache
from app.core.config import Settings
from app.integrations.ml_client import MLClient
from app.schemas.dashboard import OverviewOut, PlantMonitoringOut
from app.services import monitoring_service
from app.services.pipeline_service import build_status

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/overview", response_model=OverviewOut)
async def overview(
    session_factory: sessionmaker = Depends(get_session_factory),
    ml: MLClient = Depends(get_ml_client),
    settings: Settings = Depends(get_app_settings),
    cache: ResponseCache = Depends(get_cache),
) -> OverviewOut:
    """Totals, status/severity counts, per-group summaries, recent alerts and top recommendations. The heavy
    part is cached until the data changes (a finished run, an ingestion, a recommendation update); the
    pipeline status is always live."""
    health, _ = await ml.cached_health()
    ml_ok = health is not None and health.status == "ok" and health.models_loaded
    return await asyncio.to_thread(
        monitoring_service.overview_cached, session_factory, cache, settings.ML_CONFIG_PATH, lambda db: build_status(db, ml_ok=ml_ok)
    )


@router.get("/plants/{plant_id}", response_model=PlantMonitoringOut)
def plant_monitoring(
    plant_id: int,
    limit: Annotated[int, Query(ge=1, le=336)] = 84,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> PlantMonitoringOut:
    """The plant, its last `limit` readings (ascending), its prediction history and its active recommendations."""
    out = monitoring_service.plant_monitoring(db, settings.ML_CONFIG_PATH, plant_id, limit)
    if out is None:
        raise HTTPException(status_code=404, detail=f"plant {plant_id} not found")
    return out
