"""HTTP metrics and request-correlation middleware."""

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from sentiment.serving.metrics import REQUEST_COUNT, REQUEST_LATENCY


class MetricsMiddleware(BaseHTTPMiddleware):
    """Measure every request and attach a safe correlation identifier."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Time a request and label it by route template, never by user input."""
        request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started

        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(elapsed)
        response.headers["x-request-id"] = request.state.request_id
        return response


class RequestBodyLimitMiddleware(BaseHTTPMiddleware):
    """Reject declared request bodies above the configured service limit."""

    def __init__(self, app: object, max_bytes: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > self.max_bytes
            except ValueError:
                too_large = True
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error_code": "request_too_large",
                        "message": f"Request body exceeds {self.max_bytes} bytes.",
                        "request_id": getattr(request.state, "request_id", "unknown"),
                    },
                )
        return await call_next(request)
