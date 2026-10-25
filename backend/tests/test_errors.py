"""Error envelope: validation errors and unhandled exceptions never leak internals."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.db import database


def _app(real_settings):
    engine = database.build_engine(real_settings.DATABASE_URL)
    app = create_app(real_settings, engine)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail")

    @app.get("/needs-int/{n}")
    def needs_int(n: int) -> int:
        return n

    return app, engine


def test_unhandled_exception_returns_generic_500(real_settings):
    app, engine = _app(real_settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")
    engine.dispose()

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert "secret" not in response.text and "Traceback" not in response.text
    assert body["error"]["details"]["requestId"] == response.headers["X-Request-ID"]


def test_validation_error_envelope(real_settings):
    app, engine = _app(real_settings)
    with TestClient(app) as client:
        response = client.get("/needs-int/abc")
    engine.dispose()

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["loc"] == ["path", "n"]
