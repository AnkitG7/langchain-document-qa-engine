"""FastAPI Backend and Streaming Module for DocMind.

Exports:
- create_app: FastAPI application factory
- app: Global FastAPI application instance
"""

from .server import create_app, app

__all__ = [
    "create_app",
    "app",
]
