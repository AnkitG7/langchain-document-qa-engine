"""Test suite for Phase 1: LLM Abstraction and LCEL Chains.

All tests run deterministically with FakeListChatModel without requiring external API keys.
"""

import json
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from llm.provider import get_chat_model, create_fallback_model
from chains.qa_chain import (
    create_basic_qa_chain,
    create_structured_qa_chain,
    create_parallel_qa_chain,
    StructuredQAResponse,
)
from chains.summary_chain import (
    create_stuff_summary_chain,
    create_map_reduce_summary_chain,
    ExecutiveSummary,
)
from chains.compare_chain import (
    create_text_compare_chain,
    create_structured_compare_chain,
    StructuredComparisonReport,
)
from chains.composition import create_analyst_pipeline


class TestLLMProvider:
    def test_get_fake_chat_model(self):
        fake_llm = get_chat_model(
            provider="fake",
            fake_responses=["Hello test world"],
        )
        assert isinstance(fake_llm, FakeListChatModel)
        result = fake_llm.invoke("Hi")
        assert result.content == "Hello test world"

    def test_fallback_model_creation(self):
        primary = FakeListChatModel(responses=["Primary answer"])
        backup = FakeListChatModel(responses=["Backup answer"])
        fallback_chain = create_fallback_model(primary, [backup])
        res = fallback_chain.invoke("Test query")
        assert res.content == "Primary answer"


class TestQAChains:
    def test_basic_qa_chain(self):
        mock_llm = FakeListChatModel(responses=["The speed of light is 299,792 km/s."])
        chain = create_basic_qa_chain(llm=mock_llm)

        result = chain.invoke({
            "context": "Physics text describing light speed.",
            "question": "What is the speed of light?",
        })

        assert isinstance(result, str)
        assert "299,792" in result

    def test_structured_qa_chain(self):
        valid_response = {
            "answer": "Retrieval-Augmented Generation improves LLM accuracy.",
            "confidence_score": 0.95,
            "key_takeaways": ["Reduces hallucination", "Allows citation"],
            "citations": [
                {"source_name": "RAG paper 2020", "quote_or_fact": "RAG models combine parametric memory with non-parametric."}
            ],
            "limitations": "Requires clean vector indexes.",
        }
        mock_llm = FakeListChatModel(responses=[json.dumps(valid_response)])
        chain = create_structured_qa_chain(llm=mock_llm)

        result = chain.invoke({
            "context": "Context on RAG architectures.",
            "question": "Why use RAG?",
        })

        assert isinstance(result, StructuredQAResponse)
        assert result.confidence_score == 0.95
        assert len(result.key_takeaways) == 2
        assert result.citations[0].source_name == "RAG paper 2020"

    def test_parallel_qa_chain(self):
        mock_llm = FakeListChatModel(responses=[
            "DocMind uses LangChain.",  # QA response
            "Summary: Overview of DocMind.",  # Summary response
        ])
        chain = create_parallel_qa_chain(llm=mock_llm)

        docs = [
            {"source": "README.md", "content": "DocMind is built on LangChain LCEL."},
            {"source": "ARCHITECTURE.md", "content": "It supports multi-document Q&A."},
        ]

        result = chain.invoke({
            "documents": docs,
            "question": "What does DocMind use?",
        })

        assert "answer" in result
        assert "context_summary" in result
        assert result["documents_analyzed"] == 2


class TestSummaryChains:
    def test_stuff_summary_chain(self):
        mock_llm = FakeListChatModel(responses=["This is a concise summary of the document."])
        chain = create_stuff_summary_chain(llm=mock_llm, structured=False)

        res = chain.invoke({"text": "A long text that fits in context."})
        assert "concise summary" in res

    def test_map_reduce_summary_chain(self):
        mock_llm = FakeListChatModel(responses=[
            "Summary of chunk 1",
            "Summary of chunk 2",
            "Final synthesized master summary",
        ])
        chain = create_map_reduce_summary_chain(llm=mock_llm)

        chunks = [
            "Chunk 1 text with details on topic A.",
            "Chunk 2 text with details on topic B.",
        ]
        res = chain.invoke(chunks)
        assert res == "Final synthesized master summary"


class TestCompareChains:
    def test_text_compare_chain(self):
        mock_llm = FakeListChatModel(responses=["Document A focuses on speed, while Document B focuses on cost."])
        chain = create_text_compare_chain(llm=mock_llm)

        res = chain.invoke({
            "doc_a_name": "Engine A",
            "doc_a_content": "Engine A is optimized for low-latency.",
            "doc_b_name": "Engine B",
            "doc_b_content": "Engine B is optimized for low storage cost.",
            "criteria": "Performance vs Cost",
        })
        assert "Document A focuses on speed" in res


class TestCompositionPipeline:
    def test_analyst_pipeline(self):
        mock_llm = FakeListChatModel(responses=[
            json.dumps({"topic": "AI Market", "claims": ["Market will grow 30%", "Enterprise adoption is up"]}),
            "Critique: 30% growth assumes steady chip supply.",
            "Final Briefing: AI sector remains bullish despite hardware supply chain risks.",
        ])
        pipeline = create_analyst_pipeline(llm=mock_llm)

        result = pipeline.invoke({
            "document_text": "The AI market is projected to grow 30% next year.",
        })

        assert "extracted" in result
        assert result["topic"] == "AI Market"
        assert len(result["claims"]) == 2
        assert "critique" in result
        assert "final_briefing" in result
