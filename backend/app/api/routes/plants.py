"""GET /api/plants (paginated) and /api/plants/{id}: catalog + latest reading + latest ML prediction."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from app.api.deps import MAX_OFFSET, get_db
from app.schemas.plant import PlantOut
from app.services import plant_service

router = APIRouter(prefix="/plants", tags=["plants"])


@router.get("", response_model=list[PlantOut])
def list_plants(
    response: Response,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    db: Session = Depends(get_db),
) -> list[PlantOut]:
    items, total = plant_service.list_plants(db, limit, offset)
    response.headers["X-Total-Count"] = str(total)
    return items


@router.get("/{plant_id}", response_model=PlantOut)
def get_plant(plant_id: int, db: Session = Depends(get_db)) -> PlantOut:
    plant = plant_service.get_plant(db, plant_id)
    if plant is None:
        raise HTTPException(status_code=404, detail=f"plant {plant_id} not found")
    return plant
