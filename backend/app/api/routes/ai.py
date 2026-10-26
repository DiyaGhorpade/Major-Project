"""/api/ai/*: insights, local and global SHAP, anomaly list. Before a Run these return empty states / 404."""

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import MAX_OFFSET, get_app_settings, get_cache, get_db, get_ml_client, get_session_factory
from app.core.cache import ResponseCache
from app.core.config import Settings
from app.integrations.ml_client import MLClient
from app.schemas.ai import AnomalyOut, AnomalySummaryOut, InsightsOut
from app.schemas.shap import GlobalShapOut, LocalShapOut
from app.services import ai_service

router = APIRouter(prefix="/ai", tags=["ai"])


@router.get("/insights", response_model=InsightsOut)
async def insights(
    session_factory: sessionmaker = Depends(get_session_factory),
    ml: MLClient = Depends(get_ml_client),
    settings: Settings = Depends(get_app_settings),
    cache: ResponseCache = Depends(get_cache),
) -> InsightsOut:
    health, _ = await ml.cached_health()
    ml_ok = health is not None and health.status == "ok" and health.models_loaded
    explain = health.explainability_available if health is not None else None

    return await asyncio.to_thread(
        ai_service.insights_cached, session_factory, cache, settings, ml_ok=ml_ok, explain_available=explain
    )


# NOTE: `/shap/global` must be declared before `/shap/{plant_id}`.
@router.get("/shap/global", response_model=GlobalShapOut)
def global_shap(settings: Settings = Depends(get_app_settings)) -> GlobalShapOut:
    """Global feature importance from the newest SHAP report of the current model run (404 + generate command if none)."""
    try:
        return ai_service.global_shap(settings)
    except ai_service.GlobalShapUnavailable as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "global_shap_unavailable", "message": str(exc), "details": {"hint": exc.hint}},
        ) from exc


@router.get("/shap/{plant_id}", response_model=LocalShapOut)
def local_shap(plant_id: int, db: Session = Depends(get_db)) -> LocalShapOut:
    """Local explanation of the plant's latest explained prediction; 404 `explanation_unavailable` if none yet."""
    try:
        return ai_service.local_shap(db, plant_id)
    except ai_service.PlantNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ai_service.ExplanationUnavailable as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "explanation_unavailable", "message": str(exc), "details": {"hint": "Press Run to create predictions with explanations."}},
        ) from exc


@router.get("/anomalies", response_model=list[AnomalyOut])
def anomalies(
    response: Response,
    plant_id: Annotated[int | None, Query(alias="plantId", ge=1, le=16)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    sort: Literal["recent", "score"] = "recent",
    min_score: Annotated[float | None, Query(alias="minScore", ge=-10, le=10, allow_inf_nan=False)] = None,
    db: Session = Depends(get_db),
) -> list[AnomalyOut]:
    """Unusual readings flagged by the Isolation Forest, newest first (`sort=score`: highest score first, optional
    `minScore`). These are sensor checks, not health verdicts."""
    items, total = ai_service.list_anomalies(db, plant_id, limit, offset, sort, min_score)
    response.headers["X-Total-Count"] = str(total)
    return items


@router.get("/anomalies/summary", response_model=AnomalySummaryOut)
def anomalies_summary(db: Session = Depends(get_db)) -> AnomalySummaryOut:
    """Totals and a per-plant breakdown (most flags first) of the stored anomaly records."""
    return ai_service.anomaly_summary(db)
