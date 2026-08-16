"""Conversational RAG Chat Routes (Blocking and SSE Streaming) for DocMind API."""

import json
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatResponse, SourceCitation
from api.dependencies import AppState, get_app_state

router = APIRouter(prefix="/chat", tags=["Conversational Chat"])


def _extract_citations(docs) -> list[SourceCitation]:
    citations = []
    if docs:
        for d in docs:
            citations.append(
                SourceCitation(
                    source=d.metadata.get("filename", d.metadata.get("source", "unknown")),
                    file_type=d.metadata.get("file_type"),
                    page=d.metadata.get("page"),
                    row=d.metadata.get("row"),
                    content_snippet=d.page_content[:200],
                )
            )
    return citations


@router.post("", response_model=ChatResponse)
async def chat_blocking(
    request: ChatRequest,
    state: AppState = Depends(get_app_state),
):
    """Executes a blocking conversational RAG turn with session history."""
    try:
        rag = state.get_conversational_rag()
        result = rag.chat(user_input=request.input, session_id=request.session_id)

        answer_text = result.get("answer", "")
        retrieved_docs = result.get("retrieved_docs", [])
        citations = _extract_citations(retrieved_docs)

        return ChatResponse(
            session_id=request.session_id,
            input=request.input,
            answer=answer_text,
            citations=citations,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversational chat error: {str(e)}",
        )


@router.post("/stream")
async def chat_streaming(
    request: ChatRequest,
    state: AppState = Depends(get_app_state),
):
    """Streams conversational RAG response tokens and citations via Server-Sent Events (SSE)."""

    async def sse_generator() -> AsyncGenerator[str, None]:
        try:
            rag = state.get_conversational_rag()

            # Execute RAG to get answer & citations
            # Run in thread pool to avoid blocking async loop for synchronous LLMs
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: rag.chat(user_input=request.input, session_id=request.session_id)
            )

            answer = result.get("answer", "")
            retrieved_docs = result.get("retrieved_docs", [])
            citations = _extract_citations(retrieved_docs)

            # 1. Send citations event
            citations_payload = {
                "event": "citations",
                "citations": [c.model_dump() for c in citations],
            }
            yield f"data: {json.dumps(citations_payload)}\n\n"

            # 2. Stream tokens in small chunks
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                token_payload = {"event": "token", "token": chunk}
                yield f"data: {json.dumps(token_payload)}\n\n"
                await asyncio.sleep(0.01)

            # 3. Send completion event
            done_payload = {"event": "done", "session_id": request.session_id}
            yield f"data: {json.dumps(done_payload)}\n\n"

        except Exception as e:
            error_payload = {"event": "error", "message": str(e)}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
