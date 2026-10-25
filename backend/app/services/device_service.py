"""Devices derived from the data: Jetson gateway + two ESP32 nodes. Only facts the data supports are filled in
(last reading time per node, distinct multiplexer channels, reading counts); everything else is null and the
status is `unknown` because the historical dataset has no heartbeat (DEV-33)."""

from sqlalchemy.orm import Session

from app.core.mappings import to_ist_iso
from app.db import repositories as repo
from app.schemas.device import DeviceOut, HardwareDetailsOut

MODEL_BY_TYPE = {"jetson": "Edge gateway", "esp32": "ESP32 sensor node"}  # generic: the dataset names no hardware model


def list_devices(db: Session, limit: int = 500, offset: int = 0) -> tuple[list[DeviceOut], int]:
    nodes = repo.node_stats(db)
    per_node = repo.readings_per_node(db)
    plants_by_node = repo.plant_ids_per_node(db)
    lasts = [n["last"] for n in nodes.values() if n["last"]]
    out: list[DeviceOut] = []
    for device in repo.list_devices(db, limit, offset):
        if device.type == "esp32" and device.node_id:
            info = nodes.get(device.node_id, {})
            last, channels = info.get("last"), info.get("channels")
            count, plant_ids = per_node.get(device.node_id, 0), plants_by_node.get(device.node_id, [])
        else:  # the gateway receives every node's data
            last, channels = (max(lasts) if lasts else None), None
            count, plant_ids = sum(per_node.values()), sorted(p for ids in plants_by_node.values() for p in ids)
        out.append(
            DeviceOut(
                id=device.id,
                name=device.name,
                type=device.type,
                nodeNumber=device.node_id,
                ip_address=device.ip_address,
                mac_address=device.mac_address,
                status="unknown" if device.status == "demo" else device.status,
                last_heartbeat=to_ist_iso(device.last_heartbeat),
                last_transmission=to_ist_iso(last),
                hardware_details=HardwareDetailsOut(model=MODEL_BY_TYPE.get(device.type, device.type), channels=channels),
                role=device.role,
                simpleRole=device.simple_role,
                readingCount=count,
                plantIds=plant_ids,
            )
        )
    return out, repo.count_devices(db)
