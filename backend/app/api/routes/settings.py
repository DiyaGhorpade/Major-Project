"""GET / PUT /api/settings: persisted system settings (no api_base_url, no demo mode)."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.settings import SystemSettingsOut, SystemSettingsUpdate
from app.services import settings_service

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SystemSettingsOut)
def get_settings(db: Session = Depends(get_db)) -> SystemSettingsOut:
    try:
        return settings_service.get_settings(db)
    except settings_service.SettingsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.put("", response_model=SystemSettingsOut)
def put_settings(body: SystemSettingsUpdate, db: Session = Depends(get_db)) -> SystemSettingsOut:
    """Full or partial update; invalid values give a 422 envelope. Returns the stored settings."""
    try:
        return settings_service.update_settings(db, body)
    except settings_service.SettingsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
