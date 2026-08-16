"""Dependency Injection Providers for DocMind FastAPI application.

Manages shared state, vector stores, agents, and session history managers.
Allows easy override for testing via FastAPI's dependency_overrides.
"""

from pathlib import Path
from typing import Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel

from config import settings
from llm.provider import get_chat_model
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from memory.history_store import SessionHistoryManager
from memory.conversational_rag import ConversationalRAGChain
from agent.doc_agent import DocMindAgent
from tools import get_docmind_tools
from ingestion.pipeline import IngestionPipeline


class AppState:
    """Encapsulates shared backend state and models."""

    def __init__(self):
        self.history_manager = SessionHistoryManager(storage_type="memory")
        self.embedder = get_embeddings()
        self.vectorstore: Optional[VectorStore] = None
        self._llm: Optional[BaseChatModel] = None
        self._init_default_data()

    def _init_default_data(self) -> None:
        """Indexes default data files on startup if data directory exists."""
        data_dir = Path("data")
        if data_dir.exists():
            files = [str(f) for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in [".txt", ".md", ".csv", ".pdf"]]
            if files:
                pipeline = IngestionPipeline(chunk_size=settings.default_chunk_size, chunk_overlap=settings.default_chunk_overlap)
                chunks, _ = pipeline.run_batch(files)
                if chunks:
                    self.vectorstore = get_or_create_faiss(documents=chunks, embeddings=self.embedder)

    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_chat_model()
        return self._llm

    def get_vectorstore(self) -> VectorStore:
        if self.vectorstore is None:
            # Fallback empty or minimal store
            self.vectorstore = get_or_create_faiss(documents=[], embeddings=self.embedder)
        return self.vectorstore

    def get_conversational_rag(self) -> ConversationalRAGChain:
        store = self.get_vectorstore()
        retriever = store.as_retriever(search_kwargs={"k": 3})
        return ConversationalRAGChain(
            retriever=retriever,
            llm=self.get_llm(),
            history_manager=self.history_manager,
        )

    def get_agent(self) -> DocMindAgent:
        store = self.get_vectorstore()
        tools = get_docmind_tools(vectorstore=store)
        return DocMindAgent(
            llm=self.get_llm(),
            tools=tools,
            history_manager=self.history_manager,
        )


# Global singleton instance
app_state = AppState()


def get_app_state() -> AppState:
    return app_state


def get_history_manager() -> SessionHistoryManager:
    return app_state.history_manager


def get_vectorstore_dep() -> VectorStore:
    return app_state.get_vectorstore()


def get_agent_dep() -> DocMindAgent:
    return app_state.get_agent()


def get_rag_dep() -> ConversationalRAGChain:
    return app_state.get_conversational_rag()
