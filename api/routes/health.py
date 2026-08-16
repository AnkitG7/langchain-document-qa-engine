"""Health Check Route for DocMind API."""

from fastapi import APIRouter, Depends
from config import settings
from api.schemas import HealthCheckResponse
from api.dependencies import AppState, get_app_state

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(state: AppState = Depends(get_app_state)):
    """Returns the live operational health and subsystem metrics of DocMind."""
    store = state.vectorstore
    total_chunks = 0
    if store is not None:
        try:
            # Check FAISS or Chroma size
            if hasattr(store, "index") and hasattr(store.index, "ntotal"):
                total_chunks = store.index.ntotal
        except Exception:
            total_chunks = 0

    sessions = state.history_manager.list_sessions()

    return HealthCheckResponse(
        status="healthy",
        version="0.6.0",
        llm_provider=settings.default_llm_provider,
        embedding_provider=settings.default_embedding_provider,
        total_indexed_chunks=total_chunks,
        active_sessions_count=len(sessions),
    )
