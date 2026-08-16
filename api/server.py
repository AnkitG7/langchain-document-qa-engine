"""FastAPI Application Server for DocMind.

Demonstrates:
- RESTful Document Q&A & Ingestion endpoints
- Modern CORS middleware configuration
- Server-Sent Events (SSE) streaming for real-time tokens & tool traces
- Lifespan management and router modularity
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import (
    health_router,
    documents_router,
    chat_router,
    agent_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    yield
    # Shutdown


def create_app() -> FastAPI:
    """Application factory creating the configured FastAPI instance."""
    app = FastAPI(
        title="DocMind Intelligent Document Q&A Engine API",
        description=(
            "Production-grade Document Analysis, Ingestion, Conversational RAG, "
            "and Tool-Calling Agent API with Server-Sent Events (SSE) Streaming."
        ),
        version="0.6.0",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API v1 Routers
    api_v1_prefix = "/api/v1"
    app.include_router(health_router, prefix=api_v1_prefix)
    app.include_router(documents_router, prefix=api_v1_prefix)
    app.include_router(chat_router, prefix=api_v1_prefix)
    app.include_router(agent_router, prefix=api_v1_prefix)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": "DocMind Intelligent Document Q&A Engine API",
            "version": "0.6.0",
            "docs_url": "/docs",
            "health_check": "/api/v1/health",
        }

    return app


# Default application instance for uvicorn
app = create_app()
