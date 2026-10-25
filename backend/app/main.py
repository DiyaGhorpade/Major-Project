"""App factory: CORS, request-ID middleware, error handlers, routers, and the startup lifespan
(create tables -> seed the bundled dataset in a worker thread)."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from app.api.routes import api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import RequestIDMiddleware, configure_logging
from app.db import database, models  # noqa: F401  (models import registers the tables)
from app.services.seed_service import ensure_seeded

logger = logging.getLogger("app.startup")


def _seed(session_factory: sessionmaker, settings: Settings) -> None:
    with session_factory() as db:
        ensure_seeded(db, settings)


def create_app(
    settings: Settings | None = None,
    engine: Engine | None = None,
    session_factory: sessionmaker | None = None,
) -> FastAPI:
    """Build the app. Tests inject their own settings/engine; production uses the module defaults."""
    settings = settings or get_settings()
    engine = engine or database.engine
    session_factory = session_factory or database.build_session_factory(engine)
    configure_logging(settings.LOG_LEVEL)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            database.Base.metadata.create_all(engine)
            await asyncio.wait_for(
                asyncio.to_thread(_seed, session_factory, settings), timeout=settings.SEED_TIMEOUT_S
            )
        except TimeoutError:
            logger.error("Seeding exceeded %ss; starting anyway (it may finish in the background)", settings.SEED_TIMEOUT_S)
        except Exception:
            logger.exception("Database setup or seeding failed; starting without dataset")
        yield

    app = FastAPI(title="Money Plant Monitoring Backend", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
