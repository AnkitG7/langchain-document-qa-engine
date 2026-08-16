"""Pydantic Request and Response Schemas for DocMind API.

Defines schemas for:
- System health checks
- Document upload and catalog listing
- Conversational RAG (blocking and SSE streaming)
- Tool-calling agent execution (blocking and SSE streaming)
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health Schemas
# ---------------------------------------------------------------------------
class HealthCheckResponse(BaseModel):
    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(default="0.6.0", description="DocMind API version")
    llm_provider: str = Field(description="Configured LLM provider")
    embedding_provider: str = Field(description="Configured embedding provider")
    total_indexed_chunks: int = Field(default=0, description="Total chunks in vector store")
    active_sessions_count: int = Field(default=0, description="Active conversational sessions")


# ---------------------------------------------------------------------------
# Document Schemas
# ---------------------------------------------------------------------------
class DocumentMetadataItem(BaseModel):
    filename: str
    file_type: str
    size_bytes: int
    chunks_count: int = 0
    doc_id: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    file_type: str
    chunks_created: int
    doc_id: str
    character_count: int


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: List[DocumentMetadataItem]


# ---------------------------------------------------------------------------
# Conversational Chat Schemas
# ---------------------------------------------------------------------------
class SourceCitation(BaseModel):
    source: str
    file_type: Optional[str] = None
    page: Optional[int] = None
    row: Optional[int] = None
    content_snippet: str


class ChatRequest(BaseModel):
    input: str = Field(..., min_length=1, description="User question or follow-up turn")
    session_id: str = Field(default="default", description="Conversational session identifier")


class ChatResponse(BaseModel):
    session_id: str
    input: str
    answer: str
    citations: List[SourceCitation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent Schemas
# ---------------------------------------------------------------------------
class ToolStepItem(BaseModel):
    tool: str
    tool_input: Any
    observation: str


class AgentRequest(BaseModel):
    input: str = Field(..., min_length=1, description="User instruction or question for the agent")
    session_id: str = Field(default="default", description="Conversational session identifier")


class AgentResponse(BaseModel):
    session_id: str
    input: str
    output: str
    intermediate_steps: List[ToolStepItem] = Field(default_factory=list)
