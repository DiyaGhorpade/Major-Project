"""GET /api/models/performance: evaluation metrics of the current model run (read from the artifact files)."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db
from app.core.config import Settings
from app.schemas.model import ModelMetricsOut
from app.services import model_service

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/performance", response_model=ModelMetricsOut)
def performance(db: Session = Depends(get_db), settings: Settings = Depends(get_app_settings)) -> ModelMetricsOut:
    try:
        return model_service.model_performance(db, settings.ML_ARTIFACTS_DIR)
    except model_service.ArtifactsUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_artifacts_unavailable", "message": f"Model evaluation files are not readable: {exc}",
                    "details": {"artifactsDir": model_service.artifacts_location(settings.ML_ARTIFACTS_DIR)}},
        ) from exc  # fmt: skip
