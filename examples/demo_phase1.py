"""Interactive Phase 1 Demonstration Script for DocMind.

Run with:
    python examples/demo_phase1.py

Demonstrates:
1. Multi-provider LLM loading & fallbacks
2. Basic QA Chain (LCEL pipe syntax)
3. Structured Pydantic QA Chain
4. Parallel Document Analysis with RunnableParallel
5. Map-Reduce Summarization
6. Multi-document Comparison
7. Composed Cognitive Analyst Pipeline
"""

import sys
import os

# Add parent directory to path so imports work cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from llm.provider import get_chat_model
from chains.qa_chain import (
    create_basic_qa_chain,
    create_structured_qa_chain,
    create_parallel_qa_chain,
)
from chains.summary_chain import create_stuff_summary_chain, create_map_reduce_summary_chain
from chains.compare_chain import create_text_compare_chain, create_structured_compare_chain
from chains.composition import create_analyst_pipeline


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_demo():
    print_banner("DocMind Phase 1: LLM Abstraction & LCEL Chains Demo")

    provider = settings.default_llm_provider
    print(f"Configured Provider: {provider}")
    print(f"Configured Model: {settings.ollama_model_name if provider == 'ollama' else settings.default_model_name}\n")

    # Test if provider is live
    use_live = True
    try:
        test_llm = get_chat_model(provider=provider)
        # Test basic invocation
        print(f"Testing connection to {provider} ({settings.ollama_model_name if provider == 'ollama' else settings.default_model_name})...")
        test_res = test_llm.invoke("Hi")
        print(f"Connection successful! Model response: {test_res.content[:50]}...\n")
    except Exception as e:
        print(f"[NOTE] Could not connect to live {provider} model: {e}")
        print("[INFO] Falling back to simulated 'fake' mode so you can see all LCEL chains in action.\n")
        use_live = False

    # 1. Basic QA Chain Demo
    print_banner("1. Basic LCEL QA Chain (prompt | llm | StrOutputParser)")
    sample_context = (
        "DocMind is an intelligent document Q&A engine built on LangChain. "
        "It supports PDF, CSV, URL, and Markdown ingestion with hybrid retrieval."
    )
    sample_question = "What document formats does DocMind support?"

    if use_live:
        llm = get_chat_model(provider=provider)
    else:
        llm = get_chat_model(provider="fake", fake_responses=["DocMind supports PDF, CSV, URL, and Markdown ingestion."])

    basic_qa = create_basic_qa_chain(llm=llm)
    answer = basic_qa.invoke({"context": sample_context, "question": sample_question})
    print(f"Question: {sample_question}")
    print(f"Answer:\n{answer}")

    # 2. Structured Pydantic Output Parser Demo
    print_banner("2. Structured QA Chain with Pydantic Schema Validation")
    if not use_live:
        structured_fake = (
            '{"answer": "DocMind supports PDF, CSV, URL, and Markdown formats.", '
            '"confidence_score": 1.0, '
            '"key_takeaways": ["Multi-format support", "Hybrid retrieval"], '
            '"citations": [{"source_name": "Architecture Doc", "quote_or_fact": "supports PDF, CSV, URL, and Markdown"}], '
            '"limitations": null}'
        )
        llm_structured = get_chat_model(provider="fake", fake_responses=[structured_fake])
    else:
        llm_structured = llm

    structured_qa = create_structured_qa_chain(llm=llm_structured)
    struct_res = structured_qa.invoke({"context": sample_context, "question": sample_question})
    print(f"Answer: {struct_res.answer}")
    print(f"Confidence: {struct_res.confidence_score * 100:.0f}%")
    print(f"Key Takeaways: {struct_res.key_takeaways}")
    print(f"Citations: {[c.model_dump() for c in struct_res.citations]}")

    # 3. Parallel Execution Demo
    print_banner("3. Advanced Parallel LCEL Execution (RunnableParallel)")
    if not use_live:
        parallel_llm = get_chat_model(
            provider="fake",
            fake_responses=[
                "DocMind uses LangChain LCEL pipes.",
                "DocMind is an advanced document processing framework.",
            ],
        )
    else:
        parallel_llm = llm

    parallel_qa = create_parallel_qa_chain(llm=parallel_llm)
    parallel_res = parallel_qa.invoke({
        "documents": [
            {"source": "spec.md", "content": "DocMind uses LangChain LCEL pipes for modular execution."},
            {"source": "overview.md", "content": "FastAPI is used for streaming SSE responses."},
        ],
        "question": "What does DocMind use for execution?",
    })
    print("Parallel Results:")
    print(f"  - Answer: {parallel_res['answer']}")
    print(f"  - Context Summary: {parallel_res['context_summary']}")
    print(f"  - Documents Analyzed: {parallel_res['documents_analyzed']}")

    # 4. Multi-Document Comparison Demo
    print_banner("4. Multi-Document Comparison Chain")
    if not use_live:
        compare_llm = get_chat_model(
            provider="fake",
            fake_responses=[
                "Chroma runs in-process with SQLite; PGVector runs in PostgreSQL for scalable enterprise deployment."
            ],
        )
    else:
        compare_llm = llm

    compare_chain = create_text_compare_chain(llm=compare_llm)
    comp_res = compare_chain.invoke({
        "doc_a_name": "Chroma Vector Store",
        "doc_a_content": "Chroma is an embedded open-source vector store designed for easy local development and zero-config setups.",
        "doc_b_name": "PGVector Store",
        "doc_b_content": "PGVector is a PostgreSQL extension offering enterprise relational queries, ACID guarantees, and distributed scaling.",
        "criteria": "Local Development vs Production Scalability",
    })
    print(comp_res)

    print_banner("Phase 1 Complete!")
    print("All Phase 1 LCEL chains & LLM abstractions executed successfully.")


if __name__ == "__main__":
    run_demo()
