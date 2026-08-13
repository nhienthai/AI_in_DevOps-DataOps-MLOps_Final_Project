"""Typed application errors and uniform FastAPI error handlers."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class APIError(Exception):
    """Failure with an HTTP status and stable machine-readable code."""

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


_STATUS_TO_CODE = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
}


def _body(request: Request, error_code: str, message: str) -> dict[str, str]:
    """Build a response body with the request correlation identifier."""
    return {
        "error_code": error_code,
        "message": message,
        "request_id": getattr(request.state, "request_id", "unknown"),
    }


def install_error_handlers(app: FastAPI) -> None:
    """Install handlers that keep every expected error on one wire contract."""

    @app.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(request, exc.error_code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        message = errors[0]["msg"] if errors else "Request validation failed."
        return JSONResponse(
            status_code=422,
            content=_body(request, "validation_error", str(message)),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(
                request,
                _STATUS_TO_CODE.get(exc.status_code, "http_error"),
                str(exc.detail),
            ),
        )
