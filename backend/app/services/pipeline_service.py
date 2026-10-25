"""The "Run" flow: preflight -> per-plant chronological walk -> batched ML calls -> short DB transactions.

One run at a time (guarded by an asyncio lock and the DB). The run is a background asyncio task;
all blocking SQLAlchemy work happens in worker threads with short-lived sessions, so the API stays
responsive and no connection is held across ML calls. Idempotent: a reading has one prediction row, and
without `reset` readings that already have an `ok` (current model) or `rejected` prediction are skipped
while `unavailable` ones are retried.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import groupby

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.mappings import to_ist_iso
from app.db import repositories as repo
from app.db.models import PipelineRun, SensorReading, utcnow
from app.integrations.ml_client import MLClient, MLError, MLRejected, MLUnavailable
from app.integrations.ml_types import MLReading, PredictRequest, format_rejection
from app.schemas.pipeline import PipelineState, PipelineStatus
from app.services import history_builder, model_service, recommendation_service
from app.services.prediction_service import Outcome, effective_explain_mode, persist_predictions, to_ml_reading

logger = logging.getLogger("app.pipeline")

MAX_CONSECUTIVE_DOWN_BATCHES = 3


class PipelineFailure(Exception):
    """A run-ending condition with a user-facing message."""


@dataclass
class WorkItem:
    """One reading to predict. The request is built lazily (`request()`) from shared per-plant lists."""

    reading: SensorReading
    plant_series: list[MLReading]  # the plant's whole series, shared by all its items
    index: int
    start: int  # history = plant_series[start:end]
    end: int
    explain: bool

    @property
    def history_rows(self) -> int:
        return self.end - self.start

    def request(self) -> PredictRequest:
        return PredictRequest(
            reading=self.plant_series[self.index], history=self.plant_series[self.start : self.end], explain=self.explain
        )


@dataclass
class _Stats:
    total: int = 0
    ok: int = 0
    rejected: int = 0
    unavailable: int = 0
    consecutive_down: int = 0
    stop: bool = False
    current_plant_id: int | None = None
    internal_error: str | None = None
    abort_reason: str | None = None
    last_error: str | None = None
    started: float = field(default_factory=time.perf_counter)

    @property
    def processed(self) -> int:
        return self.ok + self.rejected + self.unavailable

    @property
    def failed(self) -> int:
        return self.rejected + self.unavailable


class PipelineService:
    def __init__(self, session_factory: sessionmaker, settings: Settings, ml: MLClient):
        self._sf = session_factory
        self._settings = settings
        self._ml = ml
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    # --- public API ----------------------------------------------------------------------
    async def start(self, reset: bool = False) -> tuple[int, PipelineState]:
        """Queue a run and return `(run_id, state)`; if one is active, return that run instead."""
        async with self._lock:
            active = await asyncio.to_thread(self._active_run)
            if active is not None:
                if self._task is not None and not self._task.done():
                    return active
                await asyncio.to_thread(self._fail_run, active[0], "run was left active without a worker")
            run_id = await asyncio.to_thread(self._create_run)
            self._task = asyncio.create_task(self._run(run_id, reset), name=f"pipeline-run-{run_id}")
            logger.info("pipeline run %d queued (reset=%s)", run_id, reset)
            return run_id, "queued"

    async def shutdown(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def wait(self) -> None:
        """Await the current run task (used by tests)."""
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    # --- sync DB helpers (run in threads) ------------------------------------------------
    def _active_run(self) -> tuple[int, PipelineState] | None:
        with self._sf() as db:
            run = repo.active_run(db)
            return (run.id, run.status) if run else None  # type: ignore[return-value]

    def _create_run(self) -> int:
        with self._sf() as db:
            run = PipelineRun(status="queued", message="Queued")
            db.add(run)
            db.commit()
            return run.id

    def _update_run(self, run_id: int, **values) -> None:
        with self._sf() as db:
            run = repo.get_run(db, run_id)
            if run is not None:
                for key, value in values.items():
                    setattr(run, key, value)
                db.commit()

    def _fail_run(self, run_id: int, message: str) -> None:
        try:
            self._update_run(run_id, status="failed", message=message, finished_at=utcnow())
        except Exception:
            logger.exception("could not record failure of run %d", run_id)

    # --- the run -------------------------------------------------------------------------
    async def _run(self, run_id: int, reset: bool) -> None:
        stats = _Stats()
        try:
            await asyncio.to_thread(
                self._update_run, run_id, status="running", started_at=utcnow(), message="Checking ML service"
            )
            history_rows, max_batch, ml_run_id, explain_available = await self._preflight()
            items = await asyncio.to_thread(self._prepare, run_id, reset, history_rows, ml_run_id, explain_available, stats)
            await self._process(run_id, items, max_batch, stats)
            if stats.abort_reason:
                raise PipelineFailure(
                    f"ML service became unavailable after processing {stats.processed} of {stats.total} readings "
                    f"({stats.ok} predicted; {stats.abort_reason}). Predictions made so far are saved; "
                    "run again once it is back."
                )
            await asyncio.to_thread(self._update_run, run_id, message="Building recommendations")
            created = await asyncio.to_thread(self._build_recommendations, run_id)
            await asyncio.to_thread(self._finish_ok, run_id, stats, created)
            logger.info(
                "pipeline run %d completed: total=%d ok=%d rejected=%d unavailable=%d in %.1fs",
                run_id, stats.total, stats.ok, stats.rejected, stats.unavailable, time.perf_counter() - stats.started,
            )  # fmt: skip
        except PipelineFailure as exc:
            logger.error("pipeline run %d failed: %s", run_id, exc)
            await asyncio.to_thread(self._fail_run, run_id, str(exc))
        except asyncio.CancelledError:
            self._fail_run(run_id, "Run interrupted by shutdown")
            raise
        except Exception as exc:
            logger.exception("pipeline run %d crashed", run_id)
            await asyncio.to_thread(self._fail_run, run_id, f"Internal error: {type(exc).__name__}")

    async def _preflight(self) -> tuple[int, int, str, bool]:
        """ML must be healthy with models loaded and `/metadata` readable; returns the run parameters."""
        url = self._ml.base_url
        try:
            health = await self._ml.health(retry=True)
        except MLUnavailable as exc:
            raise PipelineFailure(exc.reason or f"ML service not reachable at {url}") from exc
        if health.status != "ok" or not health.models_loaded:
            raise PipelineFailure(f"ML models unavailable: {health.artifact_error or health.status}")
        try:
            metadata = await self._ml.metadata()
        except MLUnavailable as exc:
            raise PipelineFailure(f"ML models unavailable: {exc.reason}") from exc
        except MLError as exc:
            raise PipelineFailure(f"ML metadata request failed: {exc}") from exc
        await asyncio.to_thread(self._cache_model, metadata)
        max_history, max_batch = history_builder.read_ml_limits(self._settings.ML_CONFIG_PATH)
        rows = history_builder.history_rows_for(
            max_history, metadata.required_history_rows_for_full_features, self._settings.HISTORY_ROWS_DEFAULT
        )
        explain_available = bool(metadata.explainability.get("available"))
        logger.info(
            "preflight ok: model %s run %s, history_rows=%d (cap %d), batch=%d, concurrency=%d, explain=%s",
            metadata.model_version, metadata.run_id, rows, max_history, max_batch,
            self._settings.ML_CONCURRENCY, effective_explain_mode(self._settings),
        )  # fmt: skip
        return rows, max_batch, metadata.run_id, explain_available

    def _cache_model(self, metadata) -> None:
        with self._sf() as db:
            model_service.cache_model_info(db, metadata)

    def _prepare(
        self, run_id: int, reset: bool, history_rows: int, ml_run_id: str, explain_available: bool, stats: _Stats
    ) -> list[WorkItem]:
        """Optional reset, then build the work list: every reading that still needs a prediction."""
        with self._sf() as db:
            if reset:
                recommendation_service.reset_generated(db)
                predictions, anomalies = repo.delete_all_predictions(db)
                db.commit()
                logger.info("reset: deleted %d predictions and %d anomaly records", predictions, anomalies)
            readings = repo.all_readings_ordered(db)
            existing = {} if reset else repo.all_predictions_light(db)

        explain_mode = effective_explain_mode(self._settings)
        items: list[WorkItem] = []
        for _, group in groupby(readings, key=lambda r: r.plant_id):
            series = list(group)  # one plant, ordered by time
            ml_series = [to_ml_reading(r) for r in series]
            for i, reading in enumerate(series):
                explain = explain_mode == "latest_only" and i == len(series) - 1
                prior = existing.get(reading.id)
                if prior is not None:
                    status, prior_run, has_explanation = prior
                    done = status == "rejected" or (status == "ok" and prior_run == ml_run_id)
                    if done and status == "ok" and explain and explain_available and not has_explanation:
                        done = False
                    if done:
                        continue
                start, end = history_builder.history_bounds(series, i, history_rows)
                items.append(WorkItem(reading, ml_series, i, start, end, explain))
        stats.total = len(items)
        self._update_run(
            run_id, total=len(items), processed=0, failed=0,
            message=f"Predicting {len(items)} readings" if items else "Nothing to predict: all readings already have predictions",
        )  # fmt: skip
        logger.info("run %d: %d of %d readings to predict (reset=%s)", run_id, len(items), len(readings), reset)
        return items

    async def _process(self, run_id: int, items: list[WorkItem], max_batch: int, stats: _Stats) -> None:
        chunks = deque(items[i : i + max_batch] for i in range(0, len(items), max(1, max_batch)))
        persist_lock = asyncio.Lock()

        async def worker() -> None:
            try:
                while chunks and not stats.stop:
                    chunk = chunks.popleft()
                    stats.current_plant_id = chunk[0].reading.plant_id
                    outcomes = await self._predict_chunk(chunk)
                    async with persist_lock:
                        self._count(chunk, outcomes, stats)
                        await asyncio.to_thread(self._persist_chunk, run_id, chunk, outcomes, stats)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a bug or DB failure: stop everything, report generically
                logger.exception("pipeline worker crashed")
                stats.stop = True
                stats.internal_error = stats.internal_error or type(exc).__name__

        workers = [asyncio.create_task(worker()) for _ in range(max(1, min(self._settings.ML_CONCURRENCY, len(chunks))))]
        await asyncio.gather(*workers)
        stats.stop = True
        if stats.internal_error:
            raise PipelineFailure(f"Internal error: {stats.internal_error}")

    async def _predict_chunk(self, chunk: list[WorkItem]) -> list[Outcome]:
        """ML outage -> every item unavailable; ML refusing the whole call -> every item rejected.
        Anything else is a bug and propagates to the worker's internal-error path."""
        try:
            results = await self._ml.predict_batch([item.request() for item in chunk])
            return [Outcome.from_batch(r, item.history_rows) for r, item in zip(results, chunk, strict=True)]
        except MLUnavailable as exc:
            return [Outcome("unavailable", error=exc.reason, history_rows_sent=item.history_rows) for item in chunk]
        except MLRejected as exc:
            error = format_rejection(exc.code, exc.message, exc.issues)
            return [Outcome("rejected", error=error, history_rows_sent=item.history_rows) for item in chunk]

    def _count(self, chunk: list[WorkItem], outcomes: list[Outcome], stats: _Stats) -> None:
        for item, outcome in zip(chunk, outcomes, strict=True):
            if outcome.status == "ok":
                stats.ok += 1
                continue
            setattr(stats, outcome.status, getattr(stats, outcome.status) + 1)
            stats.last_error = outcome.error
            logger.warning(
                "reading %s plant=%d ts=%s %s: %s",
                item.reading.record_id, item.reading.plant_id, to_ist_iso(item.reading.timestamp), outcome.status, outcome.error,
            )  # fmt: skip
        if all(o.status == "unavailable" for o in outcomes):
            stats.consecutive_down += 1
            if stats.consecutive_down >= MAX_CONSECUTIVE_DOWN_BATCHES and not stats.stop:
                stats.stop = True
                stats.abort_reason = stats.last_error or "no response"
        else:
            stats.consecutive_down = 0

    def _persist_chunk(self, run_id: int, chunk: list[WorkItem], outcomes: list[Outcome], stats: _Stats) -> None:
        with self._sf() as db:
            persist_predictions(db, [(item.reading, outcome) for item, outcome in zip(chunk, outcomes, strict=True)], run_id)
            run = repo.get_run(db, run_id)
            if run is not None:
                run.processed, run.failed = stats.processed, stats.failed
                run.current_plant_id = stats.current_plant_id
                run.message = f"Predicting readings: {stats.processed}/{stats.total}"
            db.commit()
        level = logging.INFO if stats.processed % 640 < len(chunk) or stats.processed == stats.total else logging.DEBUG
        logger.log(level, "run %d progress: %d/%d (failed %d), plant %d", run_id, stats.processed, stats.total, stats.failed, stats.current_plant_id)

    def _build_recommendations(self, run_id: int) -> int:
        with self._sf() as db:
            return recommendation_service.build_for_run(db, run_id)

    def _finish_ok(self, run_id: int, stats: _Stats, recommendations: int) -> None:
        message = f"Completed: {stats.ok} predicted"
        if stats.rejected:
            message += f", {stats.rejected} rejected by the ML service"
        if stats.unavailable:
            message += f", {stats.unavailable} unavailable (run again to retry)"
        if stats.total == 0:
            message = "Completed: all readings already had predictions"
        self._update_run(
            run_id, status="completed", finished_at=utcnow(), processed=stats.processed, failed=stats.failed,
            current_plant_id=None, message=message,
        )  # fmt: skip


# --- status ---------------------------------------------------------------------------------
def mark_stale_runs(session_factory: sessionmaker) -> int:
    """At startup: a run left queued/running by a previous process cannot still be running."""
    with session_factory() as db:
        count = repo.fail_stale_runs(db, "Run interrupted by a backend restart", utcnow())
    if count:
        logger.warning("marked %d stale pipeline run(s) as failed", count)
    return count


def build_status(db: Session, *, ml_ok: bool) -> PipelineStatus:
    run = repo.latest_run(db)
    model = model_service.current_model(db)
    _, readings, plants = repo.db_summary(db)
    active = run is not None and run.status in ("queued", "running")
    state: PipelineState = "idle" if run is None else run.status  # type: ignore[assignment]
    total = run.total if run else 0
    processed = run.processed if run else 0
    if run is None:
        percent = 0.0
    elif run.status == "completed" and total == 0:
        percent = 100.0
    else:
        percent = round(min(100.0, processed / total * 100), 1) if total else 0.0
    return PipelineStatus(
        run_id=run.id if run else None,
        state=state,
        total=total,
        processed=processed,
        failed=run.failed if run else 0,
        percent=percent,
        current_plant_id=run.current_plant_id if run else None,
        started_at=to_ist_iso(run.started_at) if run else None,
        finished_at=to_ist_iso(run.finished_at) if run else None,
        message=run.message if run else None,
        ml_service_ok=ml_ok,
        model_version=model.model_version if model else None,
        ready_to_run=readings > 0 and plants > 0 and not active,
    )
