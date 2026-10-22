"""FastAPI application factory. Models are LOADED (never trained) at startup."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..artifact_manager import ArtifactError
from ..config import Config, load_config
from ..inference import PlantPredictor
from ..logging_utils import get_logger
from .routes import router

log = get_logger("plant_ml.api")


def create_app(cfg: Config | None = None, predictor: PlantPredictor | None = None, run_id: str | None = None) -> FastAPI:
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.predictor = predictor
        app.state.artifact_error = None
        if predictor is None:
            try:
                app.state.predictor = PlantPredictor.from_config(cfg, run_id)
                log.info("Loaded model run %s", app.state.predictor.bundle.run_dir.name)
            except (ArtifactError, FileNotFoundError, KeyError, ValueError) as exc:
                app.state.artifact_error = str(exc)
                log.error("Models unavailable: %s", exc)
        yield

    app = FastAPI(
        title="Money Plant ML Service",
        version=str(cfg.get("project.model_version", "1.0.0")),
        description=("Experimental local inference service for Money Plant monitoring. Outputs are not validated "
                     "plant-health diagnoses. Models are trained on the data provenance reported by /metadata."),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
