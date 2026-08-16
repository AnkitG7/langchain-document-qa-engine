"""Comparative RAG Strategy Benchmark Suite.

Demonstrates:
- Side-by-side evaluation of multiple retrieval strategies (Baseline, Hybrid RRF, HyDE, Full Advanced)
- Empirical verification of whether advanced techniques improve precision, recall, and groundedness
- Markdown comparative reporting with score differentials
"""

from typing import Any, Dict, List, Optional
from tabulate import tabulate

from .dataset import EvalDataset, EvalSample
from .evaluator import RAGEvaluator, EvalReport
from rag_advanced.pipeline import AdvancedRAGPipeline, AdvancedRAGStrategy


class StrategyBenchmark:
    """Automated benchmark runner for comparing RAG retrieval strategies."""

    def __init__(
        self,
        pipeline: AdvancedRAGPipeline,
        evaluator: Optional[RAGEvaluator] = None,
    ):
        self.pipeline = pipeline
        self.evaluator = evaluator or RAGEvaluator()

    def run_benchmark(
        self,
        dataset: EvalDataset,
        strategies: Optional[List[AdvancedRAGStrategy]] = None,
    ) -> Dict[str, EvalReport]:
        """Runs evaluation across all specified strategies on the test dataset."""
        target_strategies = strategies or ["baseline", "hybrid_rrf", "full_advanced"]
        benchmark_reports: Dict[str, EvalReport] = {}

        for strat in target_strategies:
            eval_samples: List[EvalSample] = []
            for item in dataset.samples:
                # Execute pipeline with current strategy
                res = self.pipeline.query(item.question, strategy=strat)
                sample_copy = EvalSample(
                    id=item.id,
                    question=item.question,
                    ground_truth=item.ground_truth,
                    retrieved_context=res.get("context", ""),
                    answer=res.get("answer", ""),
                    metadata={**item.metadata, "strategy": strat},
                )
                eval_samples.append(sample_copy)

            strat_dataset = EvalDataset(name=f"{dataset.name} ({strat})", samples=eval_samples)
            report = self.evaluator.evaluate_dataset(strat_dataset, strategy_name=strat)
            benchmark_reports[strat] = report

        return benchmark_reports


def format_benchmark_table(reports: Dict[str, EvalReport]) -> str:
    """Formats benchmark results into a clean, comparative Markdown table."""
    headers = [
        "Strategy",
        "Faithfulness",
        "Relevance",
        "Precision",
        "Recall",
        "Overall Score",
        "Pass Rate",
    ]
    rows = []
    for strat, rep in reports.items():
        rows.append([
            strat.upper(),
            f"{rep.mean_faithfulness:.3f}",
            f"{rep.mean_answer_relevance:.3f}",
            f"{rep.mean_context_precision:.3f}",
            f"{rep.mean_context_recall:.3f}",
            f"**{rep.overall_rag_score:.3f}**",
            f"{rep.pass_rate:.1f}%",
        ])

    table = tabulate(rows, headers=headers, tablefmt="github")
    return table
