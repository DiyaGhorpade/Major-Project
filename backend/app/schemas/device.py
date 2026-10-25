"""Device model (TS `Device`). Fields the dataset cannot provide are null (DEV-33)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HardwareDetailsOut(BaseModel):
    model: str
    cpu_usage: float | None = None
    gpu_usage: float | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    disk_used_gb: float | None = None
    disk_total_gb: float | None = None
    temperature_c: float | None = None
    rssi_dbm: float | None = None
    channels: int | None = None
    uptime: str | None = None
    firmware_version: str | None = None


class DeviceOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    type: Literal["jetson", "esp32"]
    node_number: Literal["1", "2"] | None = Field(alias="nodeNumber")
    ip_address: str | None
    mac_address: str | None
    status: Literal["online", "offline", "unknown"]
    last_heartbeat: str | None
    last_transmission: str | None
    hardware_details: HardwareDetailsOut
    role: str
    simple_role: str = Field(alias="simpleRole")
    reading_count: int = Field(alias="readingCount")
    plant_ids: list[int] = Field(alias="plantIds")
