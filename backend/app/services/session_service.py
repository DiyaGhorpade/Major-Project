"""Acquisition sessions derived from the stored readings (PHASE0 4.8, DEV-32)."""

from sqlalchemy.orm import Session

from app.core.mappings import to_ist_iso
from app.db import repositories as repo
from app.db.models import AcquisitionSession
from app.schemas.session import NodeProgress, SessionOut


def _node_status(session_status: str, expected: int, received: int) -> str:
    if expected and received >= expected:
        return "completed"
    if session_status == "in_progress":
        return "in_progress" if received else "waiting"
    if session_status == "scheduled":
        return "waiting"
    return "failed" if received < expected else "completed"


def to_out(session: AcquisitionSession, node_counts: dict[tuple[int, str], int]) -> SessionOut:
    duration = 0
    if session.started_at and session.ended_at:
        duration = max(0, int((session.ended_at - session.started_at).total_seconds()))
    nodes = sorted(set(session.node_expected or {}) | {n for (sid, n) in node_counts if sid == session.session_id})
    progress = {}
    for node in nodes:
        expected = int((session.node_expected or {}).get(node, 0))
        received = node_counts.get((session.session_id, node), 0)
        progress[node] = NodeProgress(expected=expected, received=received, status=_node_status(session.status, expected, received))
    return SessionOut(
        session_id=session.session_id,
        scheduled_at=to_ist_iso(session.scheduled_at),
        started_at=to_ist_iso(session.started_at),
        ended_at=to_ist_iso(session.ended_at),
        status=session.status,
        expected_records=session.expected_records,
        received_records=session.received_records,
        rejected_records=session.rejected_records,
        flagged_records=session.flagged_records,
        duration_seconds=duration,
        trigger_type=session.trigger_type,
        node_progress=progress,
    )


def list_sessions(db: Session, limit: int = 500, offset: int = 0) -> tuple[list[SessionOut], int]:
    sessions = repo.list_sessions(db, limit, offset)
    counts = repo.session_node_counts(db)
    return [to_out(s, counts) for s in sessions], repo.count_sessions(db)


def latest_session(db: Session) -> SessionOut | None:
    items, _ = list_sessions(db, limit=1)
    return items[0] if items else None
