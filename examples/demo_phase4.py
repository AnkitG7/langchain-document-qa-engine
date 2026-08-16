"""Interactive Phase 4 Demonstration Script for DocMind.

Run with:
    python examples/demo_phase4.py

Demonstrates:
1. Multi-Session History Isolation (Alice vs. Bob independent sessions)
2. Message Window Trimming (trim_messages)
3. History-Aware Question Contextualization (reformulating follow-ups)
4. Multi-Turn Conversational RAG with RunnableWithMessageHistory (live Gemma & Nomic embeddings)
5. Progressive Summarization Memory
6. Legacy Memory vs. Modern RunnableWithMessageHistory comparison
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from llm.provider import get_chat_model
from ingestion.pipeline import IngestionPipeline
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from memory.history_store import SessionHistoryManager
from memory.trimmer import trim_conversation_history
from memory.conversational_rag import ConversationalRAGChain
from memory.summary_memory import ProgressiveConversationSummary
from memory.legacy_comparison import demonstrate_legacy_vs_modern_memory

DATA_DIR = Path(__file__).parent.parent / "data"


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_demo():
    print_banner("DocMind Phase 4: Modern Message History & Memory Demo")

    # 1. Multi-Session History Isolation
    print_banner("1. Multi-Session Isolation (SessionHistoryManager)")
    history_mgr = SessionHistoryManager(storage_type="memory")

    alice_session = history_mgr.get_session_history("alice_session")
    alice_session.add_user_message("My name is Alice and I am a Data Engineer.")
    alice_session.add_ai_message("Nice to meet you Alice! How can I help you today?")
    alice_session.add_user_message("I am building an automated RAG pipeline.")
    alice_session.add_ai_message("That sounds great! What vector store are you planning to use?")

    bob_session = history_mgr.get_session_history("bob_session")
    bob_session.add_user_message("My name is Bob and I focus on Cloud DevOps.")
    bob_session.add_ai_message("Hello Bob! How can I assist with your infrastructure?")

    print("Alice's Full Session (4 messages):")
    for m in alice_session.messages:
        print(f"  [{m.type.upper()}]: {m.content}")

    print("\nBob's Session Messages (Strictly Isolated):")
    for m in bob_session.messages:
        print(f"  [{m.type.upper()}]: {m.content}")

    # 2. Message Window Trimming
    print_banner("2. Conversation Message Trimming (Windowing)")
    trimmed_alice = trim_conversation_history(alice_session.messages, max_messages=2)
    print(f"Trimmed Alice's messages (max_messages=2): {len(trimmed_alice)} messages retained.")
    for m in trimmed_alice:
        print(f"  [{m.type.upper()}]: {m.content}")

    # 3. Setup Ingestion & FAISS Vector Store for RAG
    print_banner("3. Initializing Ingestion & Vector Retriever for Conversational RAG")
    pipeline = IngestionPipeline(chunk_size=300, chunk_overlap=50)
    sources = [
        str(DATA_DIR / "sample_doc.txt"),
        str(DATA_DIR / "sample_guide.md"),
    ]
    chunks, _ = pipeline.run_batch(sources)
    embedder = get_embeddings()
    vectorstore = get_or_create_faiss(documents=chunks, embeddings=embedder)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print(f"Retriever indexed {len(chunks)} chunks into FAISS.")

    # 4. Multi-Turn Conversational RAG
    print_banner("4. Live Multi-Turn Conversational RAG with RunnableWithMessageHistory")
    llm = get_chat_model()
    conv_rag = ConversationalRAGChain(
        retriever=retriever,
        llm=llm,
        history_manager=history_mgr,
    )

    demo_session_id = "interactive_researcher_01"

    # Turn 1
    query_1 = "What is DocMind and what does it do?"
    print(f"[User (Turn 1)]: {query_1}")
    res1 = conv_rag.chat(user_input=query_1, session_id=demo_session_id)
    print(f"[DocMind]: {res1['answer']}\n")

    # Turn 2 (Ambiguous follow-up requiring contextualization)
    query_2 = "What chunking techniques does it use for processing documents?"
    print(f"[User (Turn 2)]: {query_2}")
    res2 = conv_rag.chat(user_input=query_2, session_id=demo_session_id)
    print(f"[DocMind]: {res2['answer']}\n")

    # Turn 3 (Referential follow-up)
    query_3 = "Which embedding model did you mention earlier?"
    print(f"[User (Turn 3)]: {query_3}")
    res3 = conv_rag.chat(user_input=query_3, session_id=demo_session_id)
    print(f"[DocMind]: {res3['answer']}\n")

    # 5. Progressive Summarization Memory
    print_banner("5. Progressive Token-Aware Conversation Summarizer")
    prog_summary = ProgressiveConversationSummary(llm=llm, max_recent_messages=2)
    prog_summary.add_user_message("I am testing document loaders.")
    prog_summary.add_ai_message("DocMind provides PDF, CSV, TXT, and Web loaders.")
    prog_summary.add_user_message("I also need vector retrieval.")
    prog_summary.add_ai_message("You can use Chroma or FAISS for local retrieval.")
    ctx = prog_summary.get_context_for_prompt()
    print(f"Running Summary: {ctx['conversation_summary']}")
    print(f"Recent Active Messages Count: {len(ctx['recent_messages'])}")

    # 6. Legacy vs Modern Comparison Summary
    print_banner("6. Architectural Paradigm Comparison")
    comp = demonstrate_legacy_vs_modern_memory()
    print("Why Modern LangChain Uses RunnableWithMessageHistory over ConversationBufferMemory:")
    print(f"  * {comp['key_takeaway']}")

    print_banner("Phase 4 Complete!")
    print("Modern Message History, RunnableWithMessageHistory, and Conversational RAG verified successfully.")


if __name__ == "__main__":
    run_demo()
