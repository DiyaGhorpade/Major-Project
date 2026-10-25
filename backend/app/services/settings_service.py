"""Persisted system settings (single row, id=1)."""

from sqlalchemy.orm import Session

from app.db import repositories as repo
from app.db.models import AppSettings
from app.schemas.settings import SystemSettingsOut, SystemSettingsUpdate

FIELDS = (
    "polling_interval_seconds", "jetson_receiver_port", "ph_sensor_installed", "collection_times",
    "log_retention_days", "storage_path", "soil_adc_dry", "soil_adc_wet",
)  # fmt: skip


class SettingsMissing(LookupError):
    pass


def _out(row: AppSettings) -> SystemSettingsOut:
    return SystemSettingsOut(**{f: getattr(row, f) for f in FIELDS})


def get_settings(db: Session) -> SystemSettingsOut:
    row = repo.get_settings_row(db)
    if row is None:
        raise SettingsMissing("settings have not been created yet (the dataset seed has not run)")
    return _out(row)


def update_settings(db: Session, update: SystemSettingsUpdate) -> SystemSettingsOut:
    row = repo.get_settings_row(db)
    if row is None:
        raise SettingsMissing("settings have not been created yet (the dataset seed has not run)")
    for field, value in update.model_dump(exclude_unset=True, exclude_none=True).items():
        setattr(row, field, value)
    db.commit()
    return _out(row)
