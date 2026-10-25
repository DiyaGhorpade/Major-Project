"""POST /api/pipeline/run (202, one run at a time) and GET /api/pipeline/status."""

import asyncio

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_ml_client, get_pipeline_service
from app.integrations.ml_client import MLClient
from app.schemas.pipeline import PipelineStatus, RunRequest, RunResponse
from app.services.pipeline_service import PipelineService, build_status

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/run", status_code=202, response_model=RunResponse)
async def run_pipeline(
    body: RunRequest | None = Body(default=None),
    service: PipelineService = Depends(get_pipeline_service),
) -> RunResponse:
    """Start a run, or return the active run's id if one is already queued/running."""
    run_id, state = await service.start(reset=bool(body and body.reset))
    return RunResponse(run_id=run_id, state=state)


@router.get("/status", response_model=PipelineStatus)
async def pipeline_status(db: Session = Depends(get_db), ml: MLClient = Depends(get_ml_client)) -> PipelineStatus:
    health, _ = await ml.cached_health()
    ml_ok = health is not None and health.status == "ok" and health.models_loaded
    return await asyncio.to_thread(build_status, db, ml_ok=ml_ok)  # DB work off the event loop
