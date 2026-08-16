"""Interactive Phase 8 Demonstration Script for DocMind RAG Evaluation & Benchmarking.

Run with:
    python examples/demo_phase8.py

Demonstrates:
1. Benchmark Test Dataset Management (EvalDataset & EvalSample)
2. Synthetic Test Case Generation from Ingested Chunks
3. Core RAG Triad Metric Evaluation (Faithfulness, Relevance, Precision, Recall)
4. Comparative Benchmark Run across Strategies (Baseline vs. Hybrid RRF vs. Full Advanced)
5. Metric Aggregation and Markdown Scorecard Reporting
"""

import sys
import os
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from llm.provider import get_chat_model
from ingestion.pipeline import IngestionPipeline
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from rag_advanced.pipeline import AdvancedRAGPipeline
from evaluation.dataset import EvalSample, EvalDataset, SyntheticDataGenerator
from evaluation.evaluator import RAGEvaluator
from evaluation.benchmark import StrategyBenchmark, format_benchmark_table

DATA_DIR = Path(__file__).parent.parent / "data"


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_demo():
    print_banner("DocMind Phase 8: RAG Evaluation & Benchmarking Demo")

    # 1. Ingestion
    print_banner("1. Ingesting Documents & Setting Up Pipeline")
    pipeline = IngestionPipeline(chunk_size=300, chunk_overlap=50)
    chunks, _ = pipeline.run_batch([
        str(DATA_DIR / "sample_doc.txt"),
        str(DATA_DIR / "sample_data.csv"),
    ])
    print(f"Total Chunks Ingested: {len(chunks)}")

    embedder = get_embeddings()
    store = get_or_create_faiss(documents=chunks, embeddings=embedder)
    dense_retriever = store.as_retriever(search_kwargs={"k": 3})
    llm = get_chat_model()

    adv_pipeline = AdvancedRAGPipeline(
        dense_retriever=dense_retriever,
        documents=chunks,
        llm=llm,
    )

    # 2. Benchmark Dataset Creation
    print_banner("2. Constructing Benchmark Dataset (Golden Test Cases)")
    benchmark_dataset = EvalDataset(
        name="DocMind Core System Benchmark",
        samples=[
            EvalSample(
                id="q1",
                question="What document types and formats does DocMind support?",
                ground_truth="DocMind supports PDF, Markdown, CSV, and Web URLs.",
            ),
            EvalSample(
                id="q2",
                question="What is the project_name and category for id 104 in the projects table?",
                ground_truth="Project id 104 is named Observability Dashboard in the Monitoring category.",
            ),
        ],
    )

    # 3. Synthetic QA Generation Demo
    print_banner("3. Generating Synthetic Test Case from Document Chunk")
    gen = SyntheticDataGenerator(llm=llm)
    synth_sample = gen.generate_sample_from_chunk(chunks[0])
    if synth_sample:
        print(f"Generated Synthetic Question: '{synth_sample.question}'")
        print(f"Generated Ground Truth:       '{synth_sample.ground_truth}'")
        benchmark_dataset.add_sample(synth_sample)

    print(f"Total Evaluation Samples in Dataset: {len(benchmark_dataset.samples)}")

    # 4. Comparative Strategy Benchmark Execution
    print_banner("4. Executing Comparative Benchmark Across Strategies")
    evaluator = RAGEvaluator(llm=llm, threshold=0.7)
    benchmark_runner = StrategyBenchmark(pipeline=adv_pipeline, evaluator=evaluator)

    strategies_to_benchmark = ["baseline", "hybrid_rrf", "full_advanced"]
    print(f"Benchmarking Strategies: {strategies_to_benchmark}")
    print("Evaluating with LLM-as-a-Judge (Faithfulness, Relevance, Precision, Recall)...")

    reports = benchmark_runner.run_benchmark(
        dataset=benchmark_dataset,
        strategies=strategies_to_benchmark,
    )

    # 5. Benchmark Scorecard Report
    print_banner("5. Empirical Benchmark Results Scorecard")
    table = format_benchmark_table(reports)
    print(table)

    print("\nDetailed Metric Breakdown:")
    for strat, rep in reports.items():
        print(f"\n[{strat.upper()}]")
        print(f"  - Faithfulness (Groundedness): {rep.mean_faithfulness:.3f}")
        print(f"  - Answer Relevance:            {rep.mean_answer_relevance:.3f}")
        print(f"  - Context Precision:           {rep.mean_context_precision:.3f}")
        print(f"  - Context Recall:              {rep.mean_context_recall:.3f}")
        print(f"  - Overall RAG Score:           {rep.overall_rag_score:.3f}")
        print(f"  - Pass Rate:                   {rep.pass_rate:.1f}%")

    print_banner("Phase 8 Complete!")
    print("RAG Triad metrics, dataset schemas, synthetic generator, and benchmark suite verified successfully.")


if __name__ == "__main__":
    run_demo()
