"""Uniform error envelope `{"error": {"code", "message", "details"}}` and exception handlers.

Stack traces are logged server-side only; clients get a generic message plus the request ID.
"""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import REQUEST_ID_HEADER
from app.integrations.ml_client import MLRejected, MLUnavailable
from app.schemas.common import ErrorDetail, ErrorEnvelope

logger = logging.getLogger("app.errors")

_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_error",
    503: "service_unavailable",
}


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return ErrorEnvelope(error=ErrorDetail(code=code, message=message, details=details)).model_dump()


def _response(request: Request, status: int, body: dict[str, Any], headers: dict[str, str] | None = None):
    out = dict(headers or {})
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        out[REQUEST_ID_HEADER] = request_id
    return JSONResponse(body, status_code=status, headers=out)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _CODES.get(exc.status_code, f"http_{exc.status_code}")
    detail = exc.detail
    if isinstance(detail, dict) and "message" in detail:
        body = error_body(str(detail.get("code", code)), str(detail["message"]), detail.get("details"))
    else:
        message = detail if isinstance(detail, str) else HTTPStatus(exc.status_code).phrase
        body = error_body(code, message, None if isinstance(detail, str) else detail)
    return _response(request, exc.status_code, body, dict(exc.headers or {}))


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"loc": list(err.get("loc", ())), "message": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return _response(request, 422, error_body("validation_error", "Request validation failed", details))


async def ml_unavailable_handler(request: Request, exc: MLUnavailable) -> JSONResponse:
    return _response(request, 503, error_body("ml_unavailable", exc.reason))


async def ml_rejected_handler(request: Request, exc: MLRejected) -> JSONResponse:
    return _response(request, 422, error_body("ml_rejected", exc.message, {"code": exc.code, "issues": exc.issues}))


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    request_id = getattr(request.state, "request_id", None)
    return _response(
        request,
        500,
        error_body("internal_error", "Internal server error", {"requestId": request_id}),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(MLUnavailable, ml_unavailable_handler)
    app.add_exception_handler(MLRejected, ml_rejected_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
