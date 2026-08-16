"""API Routes package for DocMind."""

from .health import router as health_router
from .documents import router as documents_router
from .chat import router as chat_router
from .agent import router as agent_router

__all__ = [
    "health_router",
    "documents_router",
    "chat_router",
    "agent_router",
]
