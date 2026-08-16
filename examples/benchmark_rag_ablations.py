"""RAG Architecture Component Ablation Benchmark.

Systematically measures the contribution of each RAG subsystem:
1. Dense Vector Only (FAISS + nomic-embed-text)
2. Sparse Lexical Only (BM25 exact match)
3. Naive Fusion (Dense + BM25 Concatenation)
4. Reciprocal Rank Fusion (RRF Hybrid)
5. RRF + LLM Reranker
6. Full Pipeline (RRF + Reranker + Grounded Deduction)

Evaluates:
- Benchmark Score across all 5 difficulty levels (30 questions)
- Average Faithfulness & Answer Relevance (LLM-as-a-Judge)
- Latency (p50 & p95)
- Exact component delta breakdown (proving the marginal gain of each layer)
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.multimodal_pipeline import MultimodalIngestionPipeline
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from rag_advanced.sparse import create_bm25_retriever
from rag_advanced.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag_advanced.reranker import LLMReranker
from rag_advanced.pipeline import AdvancedRAGPipeline
from llm.provider import get_chat_model
from evaluation.metrics import FaithfulnessMetric, AnswerRelevanceMetric
from examples.benchmark_transformer_30q import BENCHMARK_QUESTIONS, PDF_PATH, INDEX_PATH


@dataclass
class AblationResult:
    config_name: str
    total_score: float
    max_score: int
    accuracy_pct: float
    avg_faithfulness: float
    avg_relevance: float
    p50_latency_ms: float
    p95_latency_ms: float
    level_breakdown: Dict[int, float]


def evaluate_configuration(
    config_name: str,
    rag: AdvancedRAGPipeline,
    docs: List[Any],
    dense_retriever: Any,
    sparse_retriever: Any,
    reranker: LLMReranker,
    llm: Any,
    faith_eval: FaithfulnessMetric,
    rel_eval: AnswerRelevanceMetric,
    strategy_mode: str,
) -> AblationResult:
    print(f"\n" + "=" * 80)
    print(f"  RUNNING ABLATION CONFIGURATION: {config_name}")
    print("=" * 80)

    scores = []
    latencies = []
    faith_scores = []
    rel_scores = []
    level_stats = {1: [], 2: [], 3: [], 4: [], 5: []}

    for item in BENCHMARK_QUESTIONS:
        qid = item["id"]
        lvl = item["level"]
        q = item["question"]
        expected = item["expected"]

        t0 = time.time()

        # Execute specific ablation retrieval strategy
        if strategy_mode == "dense_only":
            retrieved = dense_retriever.invoke(q)[:5]
            ctx_text = rag._format_context(retrieved)
            ans = rag.qa_chain.invoke({"question": q, "context": ctx_text})

        elif strategy_mode == "sparse_only":
            retrieved = sparse_retriever.invoke(q)[:5]
            ctx_text = rag._format_context(retrieved)
            ans = rag.qa_chain.invoke({"question": q, "context": ctx_text})

        elif strategy_mode == "naive_concat":
            d_docs = dense_retriever.invoke(q)[:4]
            s_docs = sparse_retriever.invoke(q)[:4]
            seen = set()
            retrieved = []
            for d in d_docs + s_docs:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    retrieved.append(d)
            retrieved = retrieved[:5]
            ctx_text = rag._format_context(retrieved)
            ans = rag.qa_chain.invoke({"question": q, "context": ctx_text})

        elif strategy_mode == "hybrid_rrf":
            d_docs = dense_retriever.invoke(q)[:8]
            s_docs = sparse_retriever.invoke(q)[:8]
            retrieved = reciprocal_rank_fusion([d_docs, s_docs], top_n=5)
            ctx_text = rag._format_context(retrieved)
            ans = rag.qa_chain.invoke({"question": q, "context": ctx_text})

        elif strategy_mode == "rrf_reranker":
            d_docs = dense_retriever.invoke(q)[:8]
            s_docs = sparse_retriever.invoke(q)[:8]
            fused = reciprocal_rank_fusion([d_docs, s_docs], top_n=8)
            retrieved = reranker.compress_documents(q, fused, top_n=5)
            ctx_text = rag._format_context(retrieved)
            ans = rag.qa_chain.invoke({"question": q, "context": ctx_text})

        elif strategy_mode == "full_system":
            res = rag.query(question=q, strategy="hybrid_rrf")
            ans = res["answer"]
            ctx_text = res.get("context", "")

        lat_ms = (time.time() - t0) * 1000
        latencies.append(lat_ms)

        f_res = faith_eval.evaluate(answer=ans, context=ctx_text)
        r_res = rel_eval.evaluate(question=q, answer=ans)

        faith_scores.append(f_res.score)
        rel_scores.append(r_res.score)

        passed = (f_res.score >= 0.8 and r_res.score >= 0.7)
        point = 1.0 if passed else (0.5 if f_res.score >= 0.8 and r_res.score >= 0.4 else 0.0)

        scores.append(point)
        level_stats[lvl].append(point)

        print(f"  [Q{qid:02d}/30 | L{lvl}] Pts: {point:.1f} | Faith: {f_res.score:.2f} | Rel: {r_res.score:.2f} | Lat: {lat_ms:.0f}ms")

    total_score = sum(scores)
    max_score = len(BENCHMARK_QUESTIONS)
    accuracy_pct = (total_score / max_score) * 100
    avg_f = float(np.mean(faith_scores))
    avg_r = float(np.mean(rel_scores))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))

    level_summary = {lvl: sum(level_stats[lvl]) for lvl in range(1, 6)}

    return AblationResult(
        config_name=config_name,
        total_score=total_score,
        max_score=max_score,
        accuracy_pct=accuracy_pct,
        avg_faithfulness=avg_f,
        avg_relevance=avg_r,
        p50_latency_ms=p50_lat,
        p95_latency_ms=p95_lat,
        level_breakdown=level_summary,
    )


def run_ablation_study():
    print("\n" + "=" * 80)
    print("  DOCMIND RAG ARCHITECTURAL ABLATION STUDY (30-QUESTION BENCHMARK)")
    print("=" * 80)

    # 1. Pipeline & Embeddings
    pipeline = MultimodalIngestionPipeline(chunk_size=600, chunk_overlap=100, enable_vision_processing=False)
    docs, _ = pipeline.ingest_pdf(PDF_PATH)
    embedder = get_embeddings()
    dense_store = get_or_create_faiss(documents=docs, embeddings=embedder, index_path=INDEX_PATH)
    dense_retriever = dense_store.as_retriever(search_kwargs={"k": 8})
    sparse_retriever = create_bm25_retriever(docs, k=8)
    llm = get_chat_model()

    reranker = LLMReranker(llm=llm, top_n=5)
    rag = AdvancedRAGPipeline(dense_retriever=dense_retriever, documents=docs, llm=llm)

    faith_eval = FaithfulnessMetric(llm=llm)
    rel_eval = AnswerRelevanceMetric(llm=llm)

    configurations = [
        ("1. Dense Vector Only (FAISS)", "dense_only"),
        ("2. Sparse BM25 Only", "sparse_only"),
        ("3. Naive Concat (Dense + BM25)", "naive_concat"),
        ("4. Hybrid RRF (Dense + BM25)", "hybrid_rrf"),
        ("5. RRF + LLM Reranker", "rrf_reranker"),
        ("6. Full System (+ Grounded Deduction)", "full_system"),
    ]

    results: List[AblationResult] = []

    for name, mode in configurations:
        res = evaluate_configuration(
            config_name=name,
            rag=rag,
            docs=docs,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            reranker=reranker,
            llm=llm,
            faith_eval=faith_eval,
            rel_eval=rel_eval,
            strategy_mode=mode,
        )
        results.append(res)

    # Print Final Ablation Matrix
    print("\n" + "=" * 95)
    print("  FINAL RAG ABLATION MATRIX: WHAT EACH COMPONENT CONTRIBUTES")
    print("=" * 95)
    print(f"{'Configuration':<38} | {'Score':<8} | {'Accuracy':<8} | {'Faith':<6} | {'Relevance':<9} | {'p50 (ms)':<8} | {'p95 (ms)':<8}")
    print("-" * 95)

    base_score = results[0].total_score
    for r in results:
        delta = f"(+{r.total_score - base_score:.1f})" if r.total_score >= base_score else f"({r.total_score - base_score:.1f})"
        print(f"{r.config_name:<38} | {f'{r.total_score:.1f}/30':<8} | {f'{r.accuracy_pct:.1f}%':<8} | {f'{r.avg_faithfulness:.2f}':<6} | {f'{r.avg_relevance:.2f}':<9} | {f'{r.p50_latency_ms:.0f}':<8} | {f'{r.p95_latency_ms:.0f}':<8}")

    print("=" * 95)

    # Marginal Gains Analysis
    print("\n" + "=" * 80)
    print("  EMPIRICAL MARGINAL GAINS OF RAG SUBSYSTEMS")
    print("=" * 80)
    print(f"1. Dense -> Sparse Baseline:       {results[0].total_score:.1f}/30 vs {results[1].total_score:.1f}/30")
    print(f"2. BM25 Addition (Naive Concat):    +{results[2].total_score - results[0].total_score:.1f} pts gain over Dense alone")
    print(f"3. Reciprocal Rank Fusion (RRF):    +{results[3].total_score - results[2].total_score:.1f} pts gain over Naive Concat")
    print(f"4. LLM Reranker Filtration:        +{results[4].total_score - results[3].total_score:.1f} pts gain over Pure RRF")
    print(f"5. Grounded Analytical Deduction:   +{results[5].total_score - results[4].total_score:.1f} pts gain over Reranker alone")
    print(f"Total Architecture Lift:            +{results[5].total_score - results[0].total_score:.1f} pts (+{(results[5].accuracy_pct - results[0].accuracy_pct):.1f}% Accuracy Lift)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_ablation_study()
