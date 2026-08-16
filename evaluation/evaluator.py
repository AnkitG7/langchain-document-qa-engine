"""RAG Evaluator Engine & Audit Reporting.

Demonstrates:
- Multi-metric RAG Triad batch scoring
- Aggregation of evaluation statistics (means, pass rates, score distributions)
- Detailed per-sample auditing and hallucination flag telemetry
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

from .metrics import (
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    MetricResult,
)
from .dataset import EvalSample, EvalDataset


class SampleEvalResult(BaseModel):
    """Detailed evaluation scores for a single evaluation sample."""
    sample_id: str
    question: str
    faithfulness: MetricResult
    answer_relevance: MetricResult
    context_precision: MetricResult
    context_recall: MetricResult
    overall_score: float = Field(..., ge=0.0, le=1.0)
    all_passed: bool


class EvalReport(BaseModel):
    """Aggregated evaluation metrics for a dataset and strategy."""
    dataset_name: str
    strategy: str = "default"
    total_samples: int = 0
    mean_faithfulness: float = 0.0
    mean_answer_relevance: float = 0.0
    mean_context_precision: float = 0.0
    mean_context_recall: float = 0.0
    overall_rag_score: float = 0.0
    pass_rate: float = 0.0
    sample_results: List[SampleEvalResult] = Field(default_factory=list)


class RAGEvaluator:
    """Evaluates RAG execution results across the RAG Triad metrics."""

    def __init__(self, llm: Optional[BaseChatModel] = None, threshold: float = 0.7):
        self.llm = llm
        self.threshold = threshold
        self.faithfulness_metric = FaithfulnessMetric(llm=self.llm, threshold=self.threshold)
        self.relevance_metric = AnswerRelevanceMetric(llm=self.llm, threshold=self.threshold)
        self.precision_metric = ContextPrecisionMetric(llm=self.llm, threshold=self.threshold)
        self.recall_metric = ContextRecallMetric(llm=self.llm, threshold=self.threshold)

    def evaluate_sample(self, sample: EvalSample) -> SampleEvalResult:
        """Runs all evaluation metrics on a single sample."""
        f_res = self.faithfulness_metric.evaluate(sample.answer, sample.retrieved_context)
        rel_res = self.relevance_metric.evaluate(sample.question, sample.answer)
        prec_res = self.precision_metric.evaluate(sample.question, sample.retrieved_context)
        rec_res = self.recall_metric.evaluate(sample.ground_truth, sample.retrieved_context)

        overall = (f_res.score + rel_res.score + prec_res.score + rec_res.score) / 4.0
        all_passed = f_res.passed and rel_res.passed and prec_res.passed and rec_res.passed

        return SampleEvalResult(
            sample_id=sample.id,
            question=sample.question,
            faithfulness=f_res,
            answer_relevance=rel_res,
            context_precision=prec_res,
            context_recall=rec_res,
            overall_score=round(overall, 3),
            all_passed=all_passed,
        )

    def evaluate_dataset(
        self,
        dataset: EvalDataset,
        strategy_name: str = "standard_rag",
    ) -> EvalReport:
        """Evaluates all samples in an EvalDataset and computes aggregated metrics."""
        if not dataset.samples:
            return EvalReport(dataset_name=dataset.name, strategy=strategy_name)

        results: List[SampleEvalResult] = []
        for sample in dataset.samples:
            results.append(self.evaluate_sample(sample))

        n = len(results)
        mean_f = sum(r.faithfulness.score for r in results) / n
        mean_rel = sum(r.answer_relevance.score for r in results) / n
        mean_prec = sum(r.context_precision.score for r in results) / n
        mean_rec = sum(r.context_recall.score for r in results) / n
        mean_overall = sum(r.overall_score for r in results) / n
        pass_rate = (sum(1 for r in results if r.all_passed) / n) * 100.0

        return EvalReport(
            dataset_name=dataset.name,
            strategy=strategy_name,
            total_samples=n,
            mean_faithfulness=round(mean_f, 3),
            mean_answer_relevance=round(mean_rel, 3),
            mean_context_precision=round(mean_prec, 3),
            mean_context_recall=round(mean_rec, 3),
            overall_rag_score=round(mean_overall, 3),
            pass_rate=round(pass_rate, 1),
            sample_results=results,
        )
