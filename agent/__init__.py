"""Agent module for DocMind.

Demonstrates:
- Tool-calling agents with ChatOllama / OpenAI
- Dynamic tool routing (Search, Calculator, Metadata)
- AgentExecutor with error handling and intermediate step tracing
- Stateful conversational agents with SessionHistoryManager
"""

from .doc_agent import create_docmind_agent, DocMindAgent

__all__ = [
    "create_docmind_agent",
    "DocMindAgent",
]
