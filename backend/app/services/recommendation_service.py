"""ML-derived and rule-based recommendations: dedupe, expiry, preserving user status. Phase 3."""

import logging

from sqlalchemy.orm import Session

logger = logging.getLogger("app.recommendations")


def reset_generated(db: Session) -> None:
    """Remove generated (ML/heuristic) recommendations before a `reset=true` recompute.

    TODO(Phase 3): delete non-user-modified recommendations (keep `user_modified` ones' status)
    and clear their prediction_id references so predictions can be deleted. No-op for now.
    """


def build_for_run(db: Session, run_id: int) -> int:
    """Build recommendations from the stored ML output of a finished run. Returns how many were created.

    TODO(Phase 3): implement per master prompt section 7. Returns 0 for now.
    """
    return 0
