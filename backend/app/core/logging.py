"""Structured (key=value) logging and the request-ID middleware.

Every request gets an ID (the client's `X-Request-ID` if it is a sane token, otherwise a
fresh UUID). It is stored in a contextvar so all log lines emitted while handling the request
carry it, and it is echoed back in the `X-Request-ID` response header.
"""

import logging
import re
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
logger = logging.getLogger("app.request")


class _RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Install one stderr handler with a key=value formatter on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if any(getattr(h, "_app_handler", False) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler._app_handler = True  # type: ignore[attr-defined]
    handler.addFilter(_RequestIDFilter())
    handler.setFormatter(
        logging.Formatter(
            'ts=%(asctime)s level=%(levelname)s logger=%(name)s request_id=%(request_id)s msg="%(message)s"'
        )
    )
    root.addHandler(handler)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            logger.info(
                "%s %s -> %s (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - started) * 1000,
            )
        finally:
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
