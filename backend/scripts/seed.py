#!/usr/bin/env python
"""Create the tables and load the bundled dataset into the database without starting any server.

    backend/.venv/bin/python backend/scripts/seed.py

Idempotent: a second run inserts nothing. Uses the same settings as the app (`backend/.env`, DATABASE_URL).
"""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings  # noqa: E402
from app.db import database, models  # noqa: E402, F401  (models import registers the tables)
from app.services.seed_service import ensure_seeded  # noqa: E402


def main() -> int:
    settings = get_settings()
    database.Base.metadata.create_all(database.engine)
    with database.SessionLocal() as db:
        result = ensure_seeded(db, settings)
    print(f"Seed finished: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
