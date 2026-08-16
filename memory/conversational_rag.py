"""History-Aware Conversational RAG Pipeline using RunnableWithMessageHistory.

Demonstrates:
- History-aware query contextualization (standalone search query reformulation)
- Multi-turn conversational retrieval over vector stores
- RunnableWithMessageHistory with session-based isolation
- Returning both generated answer and source citations
"""

from typing import Any, Callable, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from llm.provider import get_chat_model
from .history_store import get_session_history, SessionHistoryManager


# ---------------------------------------------------------------------------
# 1. Question Contextualizer / Condensation Chain
# ---------------------------------------------------------------------------
def create_contextualize_question_chain(
    llm: Optional[BaseChatModel] = None,
) -> RunnableSerializable:
    """Creates a sub-chain that reformulates follow-up questions into standalone search queries.

    If chat history is present, turns like "What about its pricing?" are transformed into
    "What is the pricing of Product X?" so the vector retriever finds the right documents.
    """
    model = llm or get_chat_model()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    return contextualize_q_prompt | model | StrOutputParser()


# ---------------------------------------------------------------------------
# 2. Conversational RAG Factory
# ---------------------------------------------------------------------------
def format_docs(docs) -> str:
    """Formats retrieved Document objects into a clean context string."""
    if not docs:
        return "No relevant documents found in index."
    formatted = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("filename", doc.metadata.get("source", f"Doc {idx}"))
        formatted.append(f"--- [Source: {source}] ---\n{doc.page_content}")
    return "\n\n".join(formatted)


def create_conversational_rag_chain(
    retriever: VectorStoreRetriever,
    llm: Optional[BaseChatModel] = None,
    session_history_getter: Optional[Callable] = None,
) -> RunnableWithMessageHistory:
    """Builds a complete stateful Conversational RAG chain wrapped in RunnableWithMessageHistory.

    Flow:
    1. Check if chat_history exists:
       - If empty: search query = user input
       - If present: search query = contextualize_question_chain(chat_history, user input)
    2. Vector store retriever fetches relevant chunks using search query
    3. Final QA prompt answers using context + conversation history + user input
    4. Output is saved automatically to session history
    """
    model = llm or get_chat_model()
    history_fn = session_history_getter or get_session_history

    # Step 1: Question Contextualizer
    contextualize_chain = create_contextualize_question_chain(llm=model)

    # Condition: Only run contextualization if chat_history is non-empty
    def route_query(input_dict: Dict[str, Any]) -> str:
        chat_history = input_dict.get("chat_history", [])
        if not chat_history:
            return input_dict["input"]
        return contextualize_chain.invoke(input_dict)

    # Step 2: Main QA Prompt
    qa_system_prompt = (
        "You are DocMind, an expert document analysis assistant. "
        "Answer the user's question using the provided context and conversation history. "
        "Cite specific sources where possible. "
        "If you do not know or the context is insufficient, state that clearly.\n\n"
        "Document Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Step 3: LCEL Assembly
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_query=RunnableLambda(route_query)
        )
        | RunnablePassthrough.assign(
            retrieved_docs=lambda x: retriever.invoke(x["standalone_query"])
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["retrieved_docs"])
        )
        | RunnablePassthrough.assign(
            answer=(qa_prompt | model | StrOutputParser())
        )
    )

    # Step 4: Wrap with RunnableWithMessageHistory for automatic stateful persistence
    conversational_rag = RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=history_fn,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag


class ConversationalRAGChain:
    """Convenience wrapper for the Conversational RAG pipeline."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: Optional[BaseChatModel] = None,
        history_manager: Optional[SessionHistoryManager] = None,
    ):
        self.retriever = retriever
        self.llm = llm or get_chat_model()
        self.history_manager = history_manager or SessionHistoryManager(storage_type="memory")
        self.chain = create_conversational_rag_chain(
            retriever=self.retriever,
            llm=self.llm,
            session_history_getter=self.history_manager.get_session_history,
        )

    def chat(
        self,
        user_input: str,
        session_id: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes a multi-turn chat step for a given session."""
        cfg: Dict[str, Any] = {"configurable": {"session_id": session_id}}
        if config:
            cfg.update(config)
        return self.chain.invoke({"input": user_input}, config=cfg)
