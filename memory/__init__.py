"""Modern Chat Message History and Conversational Memory module for DocMind.

Demonstrates:
- BaseChatMessageHistory, InMemoryChatMessageHistory, and File-persisted storage
- RunnableWithMessageHistory for stateful LCEL chains
- History-aware question contextualization (standalone query condensation)
- Sliding message windows & token-based trimming
- Progressive conversation summarization
- Educational comparison with legacy memory classes
"""

from .history_store import (
    SessionHistoryManager,
    get_session_history,
    FileSessionHistory,
)
from .trimmer import create_message_trimmer, trim_conversation_history
from .conversational_rag import (
    create_contextualize_question_chain,
    create_conversational_rag_chain,
    ConversationalRAGChain,
)
from .summary_memory import ProgressiveConversationSummary
from .legacy_comparison import demonstrate_legacy_vs_modern_memory

__all__ = [
    # History Stores
    "SessionHistoryManager",
    "get_session_history",
    "FileSessionHistory",
    # Trimmers
    "create_message_trimmer",
    "trim_conversation_history",
    # Conversational RAG
    "create_contextualize_question_chain",
    "create_conversational_rag_chain",
    "ConversationalRAGChain",
    # Summarization
    "ProgressiveConversationSummary",
    # Comparison
    "demonstrate_legacy_vs_modern_memory",
]
