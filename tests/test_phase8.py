"""Unit and integration tests for Phase 8: RAG Evaluation & Benchmarking.

Covers:
- RAG Triad Metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall
- Evaluation Dataset Schema and JSON persistence
- Synthetic QA Test Case Generator
- RAGEvaluator engine and score aggregation
- Comparative Strategy Benchmark runner and Markdown formatting
"""

import json
import pytest
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from evaluation.metrics import (
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
)
from evaluation.dataset import EvalSample, EvalDataset, SyntheticDataGenerator
from evaluation.evaluator import RAGEvaluator
from evaluation.benchmark import StrategyBenchmark, format_benchmark_table
from rag_advanced.pipeline import AdvancedRAGPipeline
from vectorstore.embedder import get_fake_embeddings
from vectorstore.store import get_or_create_faiss


@pytest.fixture
def sample_eval_item():
    return EvalSample(
        id="sample_01",
        question="What document loaders does DocMind support?",
        ground_truth="DocMind supports PDF, Markdown, CSV, and Web document loaders.",
        retrieved_context="DocMind includes modular loaders for PDF, Markdown, CSV, and Web URLs.",
        answer="DocMind supports PDF, Markdown, CSV, and Web loaders.",
    )


class TestRAGMetrics:
    def test_faithfulness_high_score(self):
        mock_llm = FakeListChatModel(responses=[
            json.dumps({"score": 0.95, "reasoning": "All claims directly supported."})
        ])
        metric = FaithfulnessMetric(llm=mock_llm)
        res = metric.evaluate(
            answer="DocMind supports PDF and CSV.",
            context="DocMind supports PDF and CSV formats.",
        )
        assert res.score == 0.95
        assert res.passed is True

    def test_faithfulness_hallucination_low_score(self):
        mock_llm = FakeListChatModel(responses=[
            json.dumps({"score": 0.2, "reasoning": "Contains unsupported claims."})
        ])
        metric = FaithfulnessMetric(llm=mock_llm, threshold=0.7)
        res = metric.evaluate(
            answer="DocMind was developed in 1990 by Bell Labs.",
            context="DocMind is a modern LangChain system built in 2026.",
        )
        assert res.score == 0.2
        assert res.passed is False

    def test_answer_relevance(self):
        mock_llm = FakeListChatModel(responses=[
            json.dumps({"score": 0.9, "reasoning": "Directly answers question."})
        ])
        metric = AnswerRelevanceMetric(llm=mock_llm)
        res = metric.evaluate(question="What is the budget?", answer="The budget is $50,000.")
        assert res.score == 0.9
        assert res.passed is True

    def test_context_precision(self):
        mock_llm = FakeListChatModel(responses=[
            json.dumps({"score": 0.85, "reasoning": "High signal context."})
        ])
        metric = ContextPrecisionMetric(llm=mock_llm)
        res = metric.evaluate(question="What is the chunk overlap?", context="Default chunk overlap is 50 tokens.")
        assert res.score == 0.85

    def test_context_recall(self):
        mock_llm = FakeListChatModel(responses=[
            json.dumps({"score": 1.0, "reasoning": "All ground truth facts present."})
        ])
        metric = ContextRecallMetric(llm=mock_llm)
        res = metric.evaluate(
            ground_truth="Budget is $85,000.",
            context="Budget for FY2026 is $85,000 allocated for benchmarks.",
        )
        assert res.score == 1.0


class TestEvaluationDataset:
    def test_dataset_json_serialization(self, tmp_path, sample_eval_item):
        dataset = EvalDataset(name="Test Dataset", samples=[sample_eval_item])
        file_path = tmp_path / "test_eval_dataset.json"

        dataset.save_json(str(file_path))
        assert file_path.exists()

        loaded = EvalDataset.load_json(str(file_path))
        assert loaded.name == "Test Dataset"
        assert len(loaded.samples) == 1
        assert loaded.samples[0].question == sample_eval_item.question

    def test_synthetic_data_generator(self):
        mock_llm = FakeListChatModel(responses=[
            json.dumps({
                "question": "What is the primary feature of DocMind?",
                "ground_truth": "Intelligent document retrieval and analysis.",
            })
        ])
        gen = SyntheticDataGenerator(llm=mock_llm)
        doc = Document(page_content="DocMind provides intelligent document retrieval and analysis capabilities.")
        sample = gen.generate_sample_from_chunk(doc)
        assert sample is not None
        assert sample.question == "What is the primary feature of DocMind?"


class TestEvaluatorAndBenchmark:
    def test_evaluator_engine(self, sample_eval_item):
        mock_llm = FakeListChatModel(responses=[
            json.dumps({"score": 0.9, "reasoning": "Faithful"}),
            json.dumps({"score": 0.95, "reasoning": "Relevant"}),
            json.dumps({"score": 0.85, "reasoning": "Precise"}),
            json.dumps({"score": 0.9, "reasoning": "High Recall"}),
        ])
        evaluator = RAGEvaluator(llm=mock_llm)
        res = evaluator.evaluate_sample(sample_eval_item)
        assert res.overall_score == 0.9
        assert res.all_passed is True

        dataset = EvalDataset(name="Audit Set", samples=[sample_eval_item])
        report = evaluator.evaluate_dataset(dataset, strategy_name="test_strat")
        assert report.total_samples == 1
        assert report.mean_faithfulness == 0.9
        assert report.pass_rate == 100.0

    def test_strategy_benchmark_and_table_format(self, sample_eval_item):
        embedder = get_fake_embeddings()
        doc = Document(page_content="DocMind supports PDF, Markdown, CSV, and Web loaders.")
        store = get_or_create_faiss(documents=[doc], embeddings=embedder)
        dense_retriever = store.as_retriever()

        mock_pipeline_llm = FakeListChatModel(responses=[
            "Baseline Answer: PDF, CSV, Web.",
            "Hybrid Answer: PDF, CSV, Web, Markdown.",
        ])
        pipeline = AdvancedRAGPipeline(dense_retriever=dense_retriever, documents=[doc], llm=mock_pipeline_llm)

        mock_eval_llm = FakeListChatModel(responses=[
            # Sample 1: Baseline
            json.dumps({"score": 0.85, "reasoning": "Faithful"}),
            json.dumps({"score": 0.9, "reasoning": "Relevant"}),
            json.dumps({"score": 0.8, "reasoning": "Precise"}),
            json.dumps({"score": 0.85, "reasoning": "Recall"}),
            # Sample 2: Hybrid
            json.dumps({"score": 0.95, "reasoning": "Faithful"}),
            json.dumps({"score": 0.95, "reasoning": "Relevant"}),
            json.dumps({"score": 0.9, "reasoning": "Precise"}),
            json.dumps({"score": 0.95, "reasoning": "Recall"}),
        ])
        evaluator = RAGEvaluator(llm=mock_eval_llm)
        bench = StrategyBenchmark(pipeline=pipeline, evaluator=evaluator)

        dataset = EvalDataset(name="Mini Bench", samples=[sample_eval_item])
        reports = bench.run_benchmark(dataset=dataset, strategies=["baseline", "hybrid_rrf"])

        assert "baseline" in reports
        assert "hybrid_rrf" in reports
        assert reports["hybrid_rrf"].overall_rag_score > reports["baseline"].overall_rag_score

        table = format_benchmark_table(reports)
        assert "BASELINE" in table
        assert "HYBRID_RRF" in table
