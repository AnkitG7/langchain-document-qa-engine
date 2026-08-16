"""Tool-Calling Agent Routes (Blocking and SSE Streaming) for DocMind API."""

import json
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.schemas import AgentRequest, AgentResponse, ToolStepItem
from api.dependencies import AppState, get_app_state

router = APIRouter(prefix="/agent", tags=["Agent"])


def _format_tool_steps(raw_steps) -> list[ToolStepItem]:
    steps = []
    for action, obs in raw_steps:
        tool_name = action.get("name") if isinstance(action, dict) else getattr(action, "tool", "tool")
        tool_input = action.get("args") if isinstance(action, dict) else getattr(action, "tool_input", {})
        steps.append(
            ToolStepItem(
                tool=str(tool_name),
                tool_input=tool_input,
                observation=str(obs)[:500],
            )
        )
    return steps


@router.post("", response_model=AgentResponse)
async def run_agent_blocking(
    request: AgentRequest,
    state: AppState = Depends(get_app_state),
):
    """Executes a blocking tool-calling agent turn with session history."""
    try:
        agent = state.get_agent()
        result = agent.run(user_input=request.input, session_id=request.session_id)

        output_text = result.get("output", "")
        steps = _format_tool_steps(result.get("intermediate_steps", []))

        return AgentResponse(
            session_id=request.session_id,
            input=request.input,
            output=output_text,
            intermediate_steps=steps,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {str(e)}",
        )


@router.post("/stream")
async def run_agent_streaming(
    request: AgentRequest,
    state: AppState = Depends(get_app_state),
):
    """Streams tool execution steps and agent answer tokens via Server-Sent Events (SSE)."""

    async def sse_generator() -> AsyncGenerator[str, None]:
        try:
            agent = state.get_agent()

            # Execute agent run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: agent.run(user_input=request.input, session_id=request.session_id)
            )

            output_text = result.get("output", "")
            raw_steps = result.get("intermediate_steps", [])

            # 1. Stream intermediate tool steps as events
            for action, obs in raw_steps:
                tool_name = action.get("name") if isinstance(action, dict) else getattr(action, "tool", "tool")
                tool_input = action.get("args") if isinstance(action, dict) else getattr(action, "tool_input", {})

                step_payload = {
                    "event": "tool_step",
                    "tool": str(tool_name),
                    "tool_input": tool_input,
                    "observation": str(obs)[:300],
                }
                yield f"data: {json.dumps(step_payload)}\n\n"
                await asyncio.sleep(0.02)

            # 2. Stream answer tokens
            words = output_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                token_payload = {"event": "token", "token": chunk}
                yield f"data: {json.dumps(token_payload)}\n\n"
                await asyncio.sleep(0.01)

            # 3. Stream completion event
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
