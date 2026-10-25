"""GET /api/sessions: acquisition sessions derived from the readings, newest first (paginated)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from app.api.deps import MAX_OFFSET, get_db
from app.schemas.session import SessionOut
from app.services import session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionOut])
def list_sessions(
    response: Response,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    db: Session = Depends(get_db),
) -> list[SessionOut]:
    items, total = session_service.list_sessions(db, limit, offset)
    response.headers["X-Total-Count"] = str(total)
    return items
