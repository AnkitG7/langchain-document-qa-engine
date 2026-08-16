"""Educational Comparison: Legacy LangChain Memory vs. Modern LCEL Message History.

Explains:
1. Legacy Paradigm: ConversationBufferMemory, ConversationSummaryMemory
2. Modern Paradigm: LCEL + BaseChatMessageHistory + RunnableWithMessageHistory
3. Architectural differences, serialization advantages, and multi-tenant scaling
"""

from typing import Dict, Any


def demonstrate_legacy_vs_modern_memory() -> Dict[str, Any]:
    """Returns an architectural breakdown contrasting legacy and modern LangChain memory."""
    comparison = {
        "legacy_approach": {
            "abstractions": [
                "ConversationBufferMemory",
                "ConversationBufferWindowMemory",
                "ConversationSummaryMemory",
                "VectorStoreRetrieverMemory",
            ],
            "characteristics": [
                "State lived mutably inside the chain object itself",
                "Coupled state management tightly with execution flow",
                "Hard to serialize across async web requests (FastAPI / SSE streaming)",
                "Did not compose cleanly with modern LCEL pipe syntax (prompt | llm | parser)",
            ],
        },
        "modern_approach": {
            "abstractions": [
                "BaseChatMessageHistory / InMemoryChatMessageHistory / FileSessionHistory",
                "RunnableWithMessageHistory wrapper",
                "trim_messages utility",
                "LangGraph state / checkpointing (for complex multi-agent workflows)",
            ],
            "characteristics": [
                "Decoupled state: Chains remain pure, stateless, reusable runnables",
                "Multi-session isolation: Session history loaded dynamically via session_id",
                "Clean serialization to Redis, PostgreSQL, SQLite, or local files",
                "Native LCEL integration and first-class async streaming support",
            ],
        },
        "key_takeaway": (
            "In modern LangChain, never store conversation state inside chain instances. "
            "Keep LCEL chains pure and wrap them in RunnableWithMessageHistory with external session stores."
        ),
    }
    return comparison
