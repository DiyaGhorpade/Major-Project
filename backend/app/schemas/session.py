"""Acquisition session model (TS `Session`, snake_case) plus `flagged_records` (DEV-32)."""

from typing import Literal

from pydantic import BaseModel


class NodeProgress(BaseModel):
    expected: int
    received: int
    status: Literal["completed", "in_progress", "waiting", "failed"]


class SessionOut(BaseModel):
    session_id: int
    scheduled_at: str | None
    started_at: str | None
    ended_at: str | None
    status: Literal["completed", "in_progress", "scheduled", "failed", "partial"]
    expected_records: int
    received_records: int
    rejected_records: int
    flagged_records: int
    duration_seconds: int
    trigger_type: Literal["scheduled", "manual", "retry"]
    node_progress: dict[str, NodeProgress]
