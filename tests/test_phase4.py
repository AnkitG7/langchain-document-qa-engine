"""Unit and integration tests for Phase 4: Modern Message History and Memory.

Covers:
- SessionHistoryManager (In-Memory and File-persisted multi-session isolation)
- Message Windowing and Trimming (trim_messages)
- Question Contextualization and Standalone Condensation
- History-Aware Conversational RAG with RunnableWithMessageHistory
- Progressive Summarization Memory
- Educational Legacy Comparison
- Negative and edge cases
"""

import shutil
import tempfile
import pytest
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from memory.history_store import (
    SessionHistoryManager,
    FileSessionHistory,
)
from memory.trimmer import (
    create_message_trimmer,
    trim_conversation_history,
)
from memory.conversational_rag import (
    create_contextualize_question_chain,
    create_conversational_rag_chain,
    ConversationalRAGChain,
)
from memory.summary_memory import ProgressiveConversationSummary
from memory.legacy_comparison import demonstrate_legacy_vs_modern_memory
from vectorstore.embedder import get_fake_embeddings
from vectorstore.store import get_or_create_faiss
from langchain_core.documents import Document


@pytest.fixture
def test_retriever():
    docs = [
        Document(
            page_content="DocMind is an intelligent document Q&A engine supporting PDF, CSV, and Markdown.",
            metadata={"source": "docmind_overview.txt", "doc_id": "d1"},
        ),
        Document(
            page_content="DocMind uses dedicated embeddings such as nomic-embed-text for vector search.",
            metadata={"source": "architecture.md", "doc_id": "d2"},
        ),
    ]
    embedder = get_fake_embeddings()
    store = get_or_create_faiss(documents=docs, embeddings=embedder)
    return store.as_retriever(search_kwargs={"k": 2})


class TestSessionHistoryStores:
    def test_in_memory_session_isolation(self):
        manager = SessionHistoryManager(storage_type="memory")

        # Session Alice
        alice_hist = manager.get_session_history("session_alice")
        alice_hist.add_user_message("Hello from Alice")
        alice_hist.add_ai_message("Hi Alice!")

        # Session Bob
        bob_hist = manager.get_session_history("session_bob")
        bob_hist.add_user_message("Hello from Bob")

        assert len(alice_hist.messages) == 2
        assert len(bob_hist.messages) == 1
        assert alice_hist.messages[0].content == "Hello from Alice"
        assert bob_hist.messages[0].content == "Hello from Bob"

    def test_session_clear(self):
        manager = SessionHistoryManager(storage_type="memory")
        hist = manager.get_session_history("session_to_clear")
        hist.add_user_message("Test message")
        assert len(hist.messages) == 1

        manager.clear_session("session_to_clear")
        assert len(hist.messages) == 0

    def test_file_session_persistence(self):
        temp_dir = tempfile.mkdtemp(prefix="chat_sessions_")
        try:
            # 1. Write to file session
            file_hist1 = FileSessionHistory(session_id="user_123", storage_dir=temp_dir)
            file_hist1.add_user_message("Message 1")
            file_hist1.add_ai_message("Response 1")

            assert file_hist1.file_path.exists()

            # 2. Reload in new instance
            file_hist2 = FileSessionHistory(session_id="user_123", storage_dir=temp_dir)
            assert len(file_hist2.messages) == 2
            assert file_hist2.messages[0].content == "Message 1"
            assert file_hist2.messages[1].content == "Response 1"

            # 3. Clear file session
            file_hist2.clear()
            assert not file_hist1.file_path.exists()
            assert len(file_hist2.messages) == 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestMessageTrimming:
    def test_trim_by_message_count(self):
        messages = [
            SystemMessage(content="System instruction"),
            HumanMessage(content="Turn 1"),
            AIMessage(content="Reply 1"),
            HumanMessage(content="Turn 2"),
            AIMessage(content="Reply 2"),
            HumanMessage(content="Turn 3"),
            AIMessage(content="Reply 3"),
        ]

        trimmed = trim_conversation_history(messages, max_messages=4)
        assert len(trimmed) == 5  # 1 System + 4 latest turns
        assert isinstance(trimmed[0], SystemMessage)
        assert trimmed[1].content == "Turn 2"
        assert trimmed[-1].content == "Reply 3"


class TestQuestionContextualization:
    def test_contextualize_question_with_history(self):
        # Mock LLM reformulating "What about its embeddings?" into standalone query
        mock_llm = FakeListChatModel(responses=[
            "What embedding models does DocMind use for vector search?"
        ])
        chain = create_contextualize_question_chain(llm=mock_llm)

        history = [
            HumanMessage(content="What is DocMind?"),
            AIMessage(content="DocMind is an intelligent document Q&A engine."),
        ]

        standalone = chain.invoke({
            "chat_history": history,
            "input": "What about its embeddings?",
        })

        assert standalone == "What embedding models does DocMind use for vector search?"


class TestConversationalRAG:
    def test_multi_turn_conversational_rag(self, test_retriever):
        mock_llm = FakeListChatModel(responses=[
            # Turn 1 answer
            "DocMind is an intelligent Q&A engine built for documents.",
            # Turn 2 question reformulation
            "What document formats are supported by DocMind?",
            # Turn 2 answer
            "DocMind supports PDF, CSV, and Markdown formats.",
        ])

        manager = SessionHistoryManager(storage_type="memory")
        rag = ConversationalRAGChain(
            retriever=test_retriever,
            llm=mock_llm,
            history_manager=manager,
        )

        # Turn 1
        res1 = rag.chat(user_input="What is DocMind?", session_id="user_session_1")
        assert "DocMind is an intelligent Q&A engine" in res1["answer"]

        # Check history was persisted
        session_hist = manager.get_session_history("user_session_1")
        assert len(session_hist.messages) == 2

        # Turn 2 (with follow-up reference)
        res2 = rag.chat(user_input="What file types does it support?", session_id="user_session_1")
        assert "PDF, CSV, and Markdown" in res2["answer"]
        assert len(session_hist.messages) == 4

        # Verify Bob's session remains empty
        bob_hist = manager.get_session_history("bob_session")
        assert len(bob_hist.messages) == 0


class TestSummaryMemory:
    def test_progressive_conversation_summary(self):
        mock_llm = FakeListChatModel(responses=[
            "Summary: User asked about DocMind capabilities and Assistant explained PDF/CSV support."
        ])
        summary_mem = ProgressiveConversationSummary(
            llm=mock_llm,
            max_recent_messages=2,
        )

        summary_mem.add_user_message("Hello!")
        summary_mem.add_ai_message("Hi! How can I assist you?")
        summary_mem.add_user_message("Can DocMind parse PDFs?")
        summary_mem.add_ai_message("Yes, DocMind parses PDFs with PyPDF.")

        # Should have summarized older turns
        ctx = summary_mem.get_context_for_prompt()
        assert "Summary:" in ctx["conversation_summary"]
        assert len(ctx["recent_messages"]) == 2


class TestLegacyComparison:
    def test_legacy_comparison_structure(self):
        comparison = demonstrate_legacy_vs_modern_memory()
        assert "legacy_approach" in comparison
        assert "modern_approach" in comparison
        assert "key_takeaway" in comparison
        assert "RunnableWithMessageHistory" in str(comparison["modern_approach"])
