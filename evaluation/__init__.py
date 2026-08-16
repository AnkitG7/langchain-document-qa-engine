"""RAG Evaluation & Benchmarking Module for DocMind.

Demonstrates:
- Core RAG Triad Metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall
- Evaluation Dataset Schema and Synthetic QA Generation
- LLM-as-a-Judge Evaluation Engine (RAGEvaluator)
- Comparative Strategy Benchmark Suite & Markdown Reporting
"""

from .metrics import (
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    MetricResult,
)
from .dataset import (
    EvalSample,
    EvalDataset,
    SyntheticDataGenerator,
)
from .evaluator import (
    RAGEvaluator,
    SampleEvalResult,
    EvalReport,
)
from .benchmark import (
    StrategyBenchmark,
    format_benchmark_table,
)

__all__ = [
    # Metrics
    "FaithfulnessMetric",
    "AnswerRelevanceMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "MetricResult",
    # Dataset
    "EvalSample",
    "EvalDataset",
    "SyntheticDataGenerator",
    # Evaluator
    "RAGEvaluator",
    "SampleEvalResult",
    "EvalReport",
    # Benchmark
    "StrategyBenchmark",
    "format_benchmark_table",
]
