"""System settings (TS `SystemSettings` minus `api_base_url` / `demo_mode`, DEV-34). Validation ranges are
enforced on PUT; a PUT may carry any subset of fields."""

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
PollingSeconds = Annotated[StrictInt, Field(ge=1, le=3600)]
Port = Annotated[StrictInt, Field(ge=1, le=65535)]
Adc = Annotated[StrictInt, Field(ge=0, le=1_000_000)]
RetentionDays = Annotated[StrictInt, Field(ge=1, le=3650)]


class SystemSettingsOut(BaseModel):
    polling_interval_seconds: int
    jetson_receiver_port: int
    ph_sensor_installed: bool
    collection_times: list[str]
    log_retention_days: int
    storage_path: str
    soil_adc_dry: int
    soil_adc_wet: int


class SystemSettingsUpdate(BaseModel):
    """Legacy keys (`api_base_url`, `demo_mode`) and unknown keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    polling_interval_seconds: PollingSeconds | None = None
    jetson_receiver_port: Port | None = None
    ph_sensor_installed: StrictBool | None = None
    collection_times: list[str] | None = Field(default=None, max_length=96)
    log_retention_days: RetentionDays | None = None
    storage_path: str | None = Field(default=None, min_length=1, max_length=500)
    soil_adc_dry: Adc | None = None
    soil_adc_wet: Adc | None = None

    @field_validator("collection_times")
    @classmethod
    def _times(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        bad = [t for t in value if not isinstance(t, str) or not HHMM.match(t)]
        if bad:
            raise ValueError(f"collection_times must be HH:MM (24 h); invalid: {bad[:3]}")
        return value
