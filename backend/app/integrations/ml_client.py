"""The ONLY code that talks to the ML service (httpx.AsyncClient).

* 503, timeouts and connection errors are retried with exponential backoff (3 attempts), then
  `MLUnavailable` is raised. 422 becomes `MLRejected` (both the `detail.error` dict form and FastAPI's
  list-shaped Pydantic detail are understood). Any other `httpx.HTTPError` (decoding errors, protocol
  errors...) is treated like a connection error, so callers never see raw httpx exceptions.
* `predict_batch` returns one normalised `BatchResult` per item. If the whole batch is refused with a
  Pydantic 422 or fails with a non-503 5xx, it falls back to per-item `/predict` so one bad item cannot
  sink the others.
* The class is plain and non-final on purpose: tests subclass it (or pass an `httpx.MockTransport`).
"""

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import ValidationError

from app.integrations.ml_types import (
    BatchPredictRequest,
    BatchResult,
    HealthResponse,
    MetadataResponse,
    MLReading,
    PredictRequest,
    PredictResponse,
    ValidateRequest,
    ValidateResponse,
)

logger = logging.getLogger("app.ml_client")

RETRY_ATTEMPTS = 3
BACKOFF_BASE_S = 0.5
HEALTH_TTL_S = 5.0
HEALTH_TIMEOUT_S = 2.0
BATCH_TIMEOUT_FACTOR = 6.0  # a batch holds up to 64 predictions, so it gets a longer read timeout


class MLError(Exception):
    """Base class for ML integration errors."""


class MLUnavailable(MLError):
    """The service could not be reached or has no models (after retries)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class MLRejected(MLError):
    """The service refused the input (HTTP 422)."""

    def __init__(self, code: str, message: str, issues: list[Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.issues = issues or []


def _parse_422(body: Any) -> MLRejected:
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):  # {"error": "...", "message": "...", "issues": [...]}
        return MLRejected(str(detail.get("error", "rejected")), str(detail.get("message", "")), detail.get("issues"))
    if isinstance(detail, list):  # FastAPI/Pydantic: [{"loc": [...], "msg": "...", "type": "..."}]
        first = detail[0] if detail and isinstance(detail[0], dict) else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = f"{loc}: {first.get('msg', 'invalid request')}" if loc else str(first.get("msg", "invalid request"))
        return MLRejected("invalid_request", message, detail)
    return MLRejected("rejected", str(detail or "ML service rejected the request"))


def _unavailable_text(body: Any) -> str:
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("error") or "models unavailable")
    return str(detail) if detail else "models unavailable"


class MLClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 10,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        attempts: int = RETRY_ATTEMPTS,
        backoff_s: float = BACKOFF_BASE_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.attempts = max(1, attempts)
        self.backoff_s = backoff_s
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s, transport=transport)
        self._cache: dict[str, tuple[float, Any]] = {}

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- transport with retry --------------------------------------------------------------
    async def _request(
        self, method: str, path: str, *, json: Any = None, timeout: float | None = None, attempts: int | None = None
    ) -> httpx.Response:
        """Send a request; retry 503/timeouts/connection errors; return any other response."""
        tries = attempts or self.attempts
        reason = "unknown error"
        for attempt in range(1, tries + 1):
            try:
                response = await self._http.request(method, path, json=json, timeout=timeout or self.timeout_s)
                if response.status_code != 503:
                    return response
                try:
                    reason = f"ML models unavailable: {_unavailable_text(response.json())}"
                except ValueError:
                    reason = "ML service returned 503"
            except httpx.TimeoutException:
                reason = f"ML service timed out ({path})"
            except httpx.HTTPError as exc:  # transport, decoding, protocol... all mean "cannot use this service"
                reason = f"ML service not reachable at {self.base_url}"
                logger.debug("ML %s %s transport error: %s: %s", method, path, type(exc).__name__, exc)
            if attempt < tries:
                delay = self.backoff_s * 2 ** (attempt - 1)
                logger.warning("ML %s %s failed (%s); retry %d/%d in %.1fs", method, path, reason, attempt, tries - 1, delay)
                await asyncio.sleep(delay)
        raise MLUnavailable(reason)

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return None

    def _check(self, response: httpx.Response) -> Any:
        """Map a non-retryable response to a body, MLRejected or MLUnavailable."""
        body = self._json(response)
        if response.status_code == 422:
            raise _parse_422(body)
        if response.status_code >= 500:
            raise MLUnavailable(f"ML internal error (HTTP {response.status_code})")
        if response.status_code >= 400:
            raise MLUnavailable(f"ML service returned HTTP {response.status_code}")
        if body is None:
            raise MLUnavailable("ML service returned a non-JSON response")
        return body

    @staticmethod
    def _parse(model: type, body: Any):
        try:
            return model.model_validate(body)
        except ValidationError as exc:
            raise MLUnavailable(f"unexpected ML response shape ({exc.error_count()} errors)") from exc

    # --- endpoints -------------------------------------------------------------------------
    async def health(self, *, retry: bool = False) -> HealthResponse:
        """`GET /health` (always 200 when the process is up). One short attempt unless `retry`."""
        response = await self._request(
            "GET", "/health", timeout=min(self.timeout_s, HEALTH_TIMEOUT_S), attempts=None if retry else 1
        )
        return self._parse(HealthResponse, self._check(response))

    async def metadata(self) -> MetadataResponse:
        """`GET /metadata`; extra fields are kept, so `.model_dump()` is the full document."""
        response = await self._request("GET", "/metadata")
        return self._parse(MetadataResponse, self._check(response))

    async def validate(self, readings: Sequence[MLReading]) -> ValidateResponse:
        payload = ValidateRequest(readings=list(readings)).model_dump(mode="json")
        response = await self._request("POST", "/validate", json=payload, timeout=max(self.timeout_s, 30))
        return self._parse(ValidateResponse, self._check(response))

    async def predict(self, req: PredictRequest) -> PredictResponse:
        response = await self._request("POST", "/predict", json=req.model_dump(mode="json"))
        return self._parse(PredictResponse, self._check(response))

    async def predict_batch(self, items: Sequence[PredictRequest]) -> list[BatchResult]:
        """One normalised result per item, in input order. Raises MLUnavailable if the whole call is down."""
        payload = BatchPredictRequest(items=list(items)).model_dump(mode="json")
        response = await self._request(
            "POST", "/predict/batch", json=payload, timeout=self.timeout_s * BATCH_TIMEOUT_FACTOR
        )
        if response.status_code == 422:
            rejected = _parse_422(self._json(response))
            if rejected.code != "invalid_request":  # e.g. bad_batch_size: a caller bug, not an item problem
                raise rejected
            logger.warning("batch refused (%s); falling back to per-item /predict for %d items", rejected.message, len(items))
            return await self._predict_each(items)
        if response.status_code >= 500:  # non-503 (503 was retried and raised above): one bad item may be the cause
            logger.warning("batch failed with HTTP %d; falling back to per-item /predict for %d items", response.status_code, len(items))
            return await self._predict_each(items)
        body = self._check(response)
        raw = body.get("results") if isinstance(body, dict) else None
        if not isinstance(raw, list) or len(raw) != len(items):
            raise MLUnavailable("unexpected /predict/batch response shape")
        return [self._normalise(item, i) for i, item in enumerate(raw)]

    def _normalise(self, item: Any, fallback_index: int) -> BatchResult:
        index = int(item.get("index", fallback_index)) if isinstance(item, dict) else fallback_index
        if not isinstance(item, dict):
            return BatchResult(index=index, status="unavailable", message="malformed batch item")
        if item.get("ok") and item.get("result") is not None:
            try:
                return BatchResult(index=index, status="ok", result=PredictResponse.model_validate(item["result"]))
            except ValidationError:
                return BatchResult(index=index, status="unavailable", code="bad_response", message="unexpected result shape")
        error = item.get("error") or {}
        status = int(error.get("status", 500) or 500)
        code = error.get("error")
        message = error.get("message")
        if status == 422:
            return BatchResult(index=index, status="rejected", code=code, message=message, issues=error.get("issues") or [])
        return BatchResult(index=index, status="unavailable", code=code, message=message or f"HTTP {status}")

    async def _predict_each(self, items: Sequence[PredictRequest]) -> list[BatchResult]:
        out: list[BatchResult] = []
        for i, item in enumerate(items):
            try:
                out.append(BatchResult(index=i, status="ok", result=await self.predict(item)))
            except MLRejected as exc:
                out.append(BatchResult(index=i, status="rejected", code=exc.code, message=exc.message, issues=exc.issues))
            except MLUnavailable as exc:
                out.append(BatchResult(index=i, status="unavailable", message=exc.reason))
        return out

    # --- short-lived caches for status/health endpoints -------------------------------------
    async def cached_health(self, ttl: float = HEALTH_TTL_S) -> tuple[HealthResponse | None, str | None]:
        """`(health, None)` or `(None, reason)`; cached for `ttl` seconds so polling stays cheap."""
        now = time.monotonic()
        hit = self._cache.get("health")
        if hit and now - hit[0] < ttl:
            return hit[1]
        try:
            value: tuple[HealthResponse | None, str | None] = (await self.health(), None)
        except MLError as exc:
            value = (None, getattr(exc, "reason", str(exc)))
        self._cache["health"] = (time.monotonic(), value)
        return value

    async def cached_metadata(self, ttl: float = 60.0) -> MetadataResponse:
        """`/metadata` cached for `ttl` seconds (only successes are cached). Raises MLError on failure."""
        now = time.monotonic()
        hit = self._cache.get("metadata")
        if hit and now - hit[0] < ttl:
            return hit[1]
        value = await self.metadata()
        self._cache["metadata"] = (time.monotonic(), value)
        return value

    def invalidate_cache(self) -> None:
        self._cache.clear()
