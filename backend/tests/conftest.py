"""Shared fixtures: temp SQLite DBs, the real dataset, a crafted bad CSV, and a TestClient."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db import database, models  # noqa: F401
from app.main import create_app

BAD_CSV = """timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light,stress_level,stress_type,is_synthetic_anomaly
2026-09-21T00:00:00+05:30,1,1,1,1,50.0,25.0,60.0,500,healthy,none,0
2026-09-21T02:00:00+05:30,1,1,2,1,50.0,45.0,60.0,500,high,light,1
2026-09-21T00:00:00+05:30,1,2,1,2,,25.0,60.0,,low,water,0
2026-09-21T04:00:00,1,2,3,2,48.0,24.0,61.0,450,healthy,none,0
2026-09-21T00:00:00+05:30,1,1,1,1,99.0,25.0,60.0,500,healthy,none,0
not-a-date,1,1,1,1,50.0,25.0,60.0,500,healthy,none,0
2026-09-21T00:00:00+05:30,1,1,1,99,50.0,25.0,60.0,500,healthy,none,0
2026-09-21T00:00:00+05:30,1,,4,3,50.0,25.0,60.0,500,healthy,none,0
2026-09-21T00:00:00+05:30,1,2,4,4,abc,25.0,60.0,500,healthy,none,0
2026-09-21T02:00:00+05:30,1,1,5,3,120.0,25.0,10.0,500,healthy,none,0
2026-09-21T04:00:00+05:30,1,inf,inf,3,inf,25.0,60.0,500,healthy,none,0
"""


REAL_CSV = Path(__file__).resolve().parents[2] / "money_plant_ml_pipeline" / "data" / "raw" / "plant_sensor_data.csv"


def make_settings(tmp_path: Path, dataset: Path) -> Settings:
    """Fully explicit settings: ignores backend/.env and the process environment."""
    return Settings(
        _env_file=None,
        ML_SERVICE_URL="http://127.0.0.1:8001",
        ML_TIMEOUT_S=10,
        ML_CONCURRENCY=4,
        EXPLAIN_MODE="latest_only",
        HISTORY_ROWS_DEFAULT=60,
        DATASET_PATH=dataset,
        ML_CONFIG_PATH=tmp_path / "config.yaml",
        ML_REPORTS_DIR=tmp_path / "reports",
        ML_ARTIFACTS_DIR=tmp_path / "artifacts",
        DATABASE_URL=f"sqlite:///{tmp_path / 'test.db'}",
        SEED_TIMEOUT_S=60,
        CORS_ORIGINS=["http://localhost:5173"],
        LOG_LEVEL="INFO",
    )


@pytest.fixture
def real_csv() -> Path:
    return REAL_CSV


@pytest.fixture
def bad_csv(tmp_path: Path) -> Path:
    path = tmp_path / "bad.csv"
    path.write_text(BAD_CSV)
    return path


@pytest.fixture
def real_settings(tmp_path: Path, real_csv: Path) -> Settings:
    return make_settings(tmp_path, real_csv)


@pytest.fixture
def bad_settings(tmp_path: Path, bad_csv: Path) -> Settings:
    return make_settings(tmp_path, bad_csv)


@pytest.fixture
def db_for() -> Iterator:
    """Factory: `db_for(settings)` -> Session on a fresh schema; everything is closed afterwards."""
    opened: list = []

    def factory(settings: Settings) -> Session:
        engine = database.build_engine(settings.DATABASE_URL)
        database.Base.metadata.create_all(engine)
        session = database.build_session_factory(engine)()
        opened.append((session, engine))
        return session

    yield factory
    for session, engine in opened:
        session.close()
        engine.dispose()


@pytest.fixture
def client(real_settings: Settings) -> Iterator[TestClient]:
    engine = database.build_engine(real_settings.DATABASE_URL)
    with TestClient(create_app(real_settings, engine)) as test_client:
        yield test_client
    engine.dispose()
