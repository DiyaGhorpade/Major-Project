"""GET /api/health: backend + DB status. ML-service reachability is wired up in Phase 2."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db
from app.core.config import Settings
from app.db import repositories as repo
from app.schemas.common import DbHealth, HealthResponse, MlServiceHealth

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db), settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    db_ok, readings, plants = repo.db_summary(db)
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db=DbHealth(ok=db_ok, readings=readings, plants=plants),
        ml_service=MlServiceHealth(ok=False, url=settings.ML_SERVICE_URL, detail="not checked (Phase 2)"),
        model_version=None,
        dataset_name=settings.DATASET_PATH.name,
        seeded=readings > 0 and plants > 0,
    )
