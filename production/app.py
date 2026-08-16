"""Production Application Factory with Request ID Middleware & Probe Endpoints.

Demonstrates:
- Hardened production FastAPI application instance
- Trace and Request ID correlation middleware (X-Request-ID & X-Trace-ID)
- Integration of Phase 6 API routes + Phase 10 Production Health Probes
"""

import uuid
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.routes import (
    health_router,
    documents_router,
    chat_router,
    agent_router,
)
from .probes import probes_router


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """Injects unique X-Request-ID and X-Trace-ID headers and measures latency."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        trace_id = request.headers.get("X-Trace-ID") or f"trace_{uuid.uuid4().hex[:10]}"

        start_time = time.time()
        response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        response.headers["X-Request-ID"] = req_id
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response


def create_production_app() -> FastAPI:
    """Creates a production-configured FastAPI application."""
    app = FastAPI(
        title="DocMind Production Document Q&A Platform",
        description="Production-grade Document Analysis Engine with Multi-Tier Caching, Probes, and Advanced RAG.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware
    app.add_middleware(RequestCorrelationMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Routes
    api_v1_prefix = "/api/v1"
    app.include_router(probes_router, prefix=api_v1_prefix)
    app.include_router(health_router, prefix=api_v1_prefix)
    app.include_router(documents_router, prefix=api_v1_prefix)
    app.include_router(chat_router, prefix=api_v1_prefix)
    app.include_router(agent_router, prefix=api_v1_prefix)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": "DocMind Production Platform",
            "version": "1.0.0",
            "environment": "production",
            "docs_url": "/docs",
            "liveness_probe": "/api/v1/health/live",
            "readiness_probe": "/api/v1/health/ready",
            "metrics": "/api/v1/metrics",
        }

    return app


# Global production app instance
production_app = create_production_app()
