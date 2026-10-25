"""GET /api/health: backend + DB status and real ML-service reachability (cached ~5 s, short timeout)."""

import asyncio

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db, get_ml_client
from app.core.config import Settings
from app.integrations.ml_client import MLClient
from app.schemas.common import DbHealth, HealthResponse, MlServiceHealth
from app.services import health_service, model_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
    ml: MLClient = Depends(get_ml_client),
) -> HealthResponse:
    db_ok, readings, plants, model = await asyncio.to_thread(health_service.db_part, db)  # DB work off the event loop
    ml_health, reason = await ml.cached_health()
    if ml_health is None:
        ml_part = MlServiceHealth(ok=False, url=settings.ML_SERVICE_URL, detail=reason or "unreachable")
    else:
        ok = ml_health.status == "ok" and ml_health.models_loaded
        detail = "ok" if ok else f"ML models unavailable: {ml_health.artifact_error or ml_health.status}"
        ml_part = MlServiceHealth(
            ok=ok,
            url=settings.ML_SERVICE_URL,
            detail=detail,
            status=ml_health.status,
            models_loaded=ml_health.models_loaded,
            explainability_available=ml_health.explainability_available,
        )
    run_id, model_version = await model_service.resolve_model(model, ml_health, ml)
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db=DbHealth(ok=db_ok, readings=readings, plants=plants),
        ml_service=ml_part,
        model_version=model_version,
        run_id=run_id,
        dataset_name=settings.DATASET_PATH.name,
        seeded=readings > 0 and plants > 0,
    )
