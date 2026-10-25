"""Application settings (Pydantic Settings, read from the environment and `backend/.env`).

Relative filesystem paths and SQLite URLs are resolved against the backend directory,
never the process cwd, so the app behaves the same however it is launched.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
ML_DIR = REPO_ROOT / "money_plant_ml_pipeline"

_SQLITE_PREFIX = "sqlite:///"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    ML_SERVICE_URL: str = "http://127.0.0.1:8001"
    ML_TIMEOUT_S: float = 10
    ML_CONCURRENCY: int = 4
    EXPLAIN_MODE: str = "latest_only"
    HISTORY_ROWS_DEFAULT: int = 60

    DATASET_PATH: Path = ML_DIR / "data" / "raw" / "plant_sensor_data.csv"
    ML_CONFIG_PATH: Path = ML_DIR / "config" / "config.yaml"
    ML_REPORTS_DIR: Path = ML_DIR / "reports"
    ML_ARTIFACTS_DIR: Path = ML_DIR / "artifacts"

    DATABASE_URL: str = f"{_SQLITE_PREFIX}{BACKEND_DIR / 'data' / 'app.db'}"
    SEED_TIMEOUT_S: float = 120

    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    LOG_LEVEL: str = "INFO"

    @field_validator("DATASET_PATH", "ML_CONFIG_PATH", "ML_REPORTS_DIR", "ML_ARTIFACTS_DIR")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return (value if value.is_absolute() else BACKEND_DIR / value).resolve()

    @field_validator("DATABASE_URL")
    @classmethod
    def _resolve_sqlite_url(cls, value: str) -> str:
        if not value.startswith(_SQLITE_PREFIX):
            return value
        target = value.removeprefix(_SQLITE_PREFIX)
        if target in ("", ":memory:") or target.startswith("file:"):
            return value
        path = Path(target)
        return f"{_SQLITE_PREFIX}{(path if path.is_absolute() else BACKEND_DIR / path).resolve()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
