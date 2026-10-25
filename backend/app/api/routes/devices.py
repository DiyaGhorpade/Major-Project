"""GET /api/devices: the Jetson gateway and two ESP32 nodes, derived from the data (paginated)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from app.api.deps import MAX_OFFSET, get_db
from app.schemas.device import DeviceOut
from app.services import device_service

router = APIRouter(prefix="/devices", tags=["devices"])


@router.get("", response_model=list[DeviceOut])
def list_devices(
    response: Response,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    db: Session = Depends(get_db),
) -> list[DeviceOut]:
    items, total = device_service.list_devices(db, limit, offset)
    response.headers["X-Total-Count"] = str(total)
    return items
