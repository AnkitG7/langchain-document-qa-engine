"""End-to-End Advanced RAG Pipeline Orchestrator.

Demonstrates:
- Strategy pattern for comparing baseline vs. advanced retrieval modes
- Multi-Query, HyDE, Step-Back, BM25 + Dense Hybrid RRF, Reranking, and Contextual Compression
- Detailed execution telemetry (strategies used, raw chunks vs compressed chunks, citations)
"""

from typing import Any, Dict, List, Literal, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from llm.provider import get_chat_model
from .query_transform import HyDETransformer, MultiQueryTransformer, StepBackTransformer
from .sparse import BM25Index, create_bm25_retriever
from .hybrid import HybridRetriever, reciprocal_rank_fusion
from .reranker import LLMReranker
from .compression import ContextualCompressor

AdvancedRAGStrategy = Literal[
    "baseline",
    "hyde",
    "multi_query",
    "step_back",
    "hybrid_rrf",
    "reranked",
    "compressed",
    "full_advanced",
]


class AdvancedRAGPipeline:
    """Unified Advanced RAG Pipeline supporting multiple retrieval & synthesis strategies."""

    def __init__(
        self,
        dense_retriever: VectorStoreRetriever,
        documents: Optional[List[Document]] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        self.dense_retriever = dense_retriever
        self.documents = documents or []
        self.llm = llm or get_chat_model()

        # Sparse & Hybrid
        self.sparse_retriever = create_bm25_retriever(self.documents, k=8)
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            k=8,
        )

        # Transformers
        self.hyde = HyDETransformer(llm=self.llm)
        self.multi_query = MultiQueryTransformer(llm=self.llm, num_queries=3)
        self.step_back = StepBackTransformer(llm=self.llm)

        # Post-retrieval processors
        self.reranker = LLMReranker(llm=self.llm, top_n=5)
        self.compressor = ContextualCompressor(llm=self.llm)

        # Generator QA Prompt
        qa_system_prompt = (
            "You are DocMind Advanced RAG Engine, an expert precision document analysis assistant. "
            "Answer the question thoroughly and accurately using the provided document context.\n\n"
            "Guidelines:\n"
            "1. Grounded Deduction: Identify relevant definitions, formulas, tables, and architectural constraints in the context. "
            "If a question asks for computational implications (e.g. scaling, bottlenecks, or reasons), logically derive the answer from the formulas and complexity terms in the context.\n"
            "2. Element-Aware Citations: Cite sources including page and element type (e.g., [Source: file.pdf, Page 4, Type: Table] or [Source: file.pdf, Page 7, Type: Chart]) for each claim.\n"
            "3. Strict Integrity: Do NOT fabricate facts not supported by the context.\n\n"
            "Document Context:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            ("human", "{question}"),
        ])
        self.qa_chain = qa_prompt | self.llm | StrOutputParser()

    def _format_context(self, docs: List[Document]) -> str:
        if not docs:
            return "No documents found."
        formatted = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("filename", d.metadata.get("source", f"Doc {i}"))
            page = d.metadata.get("page", d.metadata.get("page_number", "N/A"))
            elem_type = d.metadata.get("element_type", "text")
            formatted.append(f"--- [Source: {src} | Page: {page} | Element Type: {elem_type}] ---\n{d.page_content}")
        return "\n\n".join(formatted)

    def query(
        self,
        question: str,
        strategy: AdvancedRAGStrategy = "full_advanced",
    ) -> Dict[str, Any]:
        """Executes Advanced RAG with the specified retrieval and transformation strategy."""
        retrieved_docs: List[Document] = []
        transformed_query: Optional[str] = None
        extra_meta: Dict[str, Any] = {}

        # 1. Retrieval & Transformation by Strategy
        if strategy == "baseline":
            retrieved_docs = self.dense_retriever.invoke(question)

        elif strategy == "hyde":
            hypo_doc = self.hyde.generate_hypothetical_document(question)
            transformed_query = hypo_doc
            retrieved_docs = self.dense_retriever.invoke(hypo_doc)

        elif strategy == "multi_query":
            retrieved_docs = self.multi_query.retrieve(question, self.dense_retriever)

        elif strategy == "step_back":
            res = self.step_back.retrieve(question, self.dense_retriever)
            transformed_query = res["step_back_query"]
            # Combine specific and step-back docs
            retrieved_docs = res["specific_docs"] + res["step_back_docs"]

        elif strategy == "hybrid_rrf":
            retrieved_docs = self.hybrid_retriever.invoke(question)

        elif strategy == "reranked":
            # Hybrid search followed by LLM reranking
            candidates = self.hybrid_retriever.invoke(question)
            retrieved_docs = self.reranker.compress_documents(question, candidates, top_n=3)

        elif strategy == "compressed":
            # Hybrid search followed by contextual compression
            candidates = self.hybrid_retriever.invoke(question)
            retrieved_docs = self.compressor.compress(question, candidates)

        elif strategy == "full_advanced":
            # Full Pipeline: Multi-Query -> Hybrid RRF -> Reranking -> Compression
            mq_docs = self.multi_query.retrieve(question, self.dense_retriever)
            sparse_docs = self.sparse_retriever.invoke(question)
            fused = reciprocal_rank_fusion([mq_docs, sparse_docs], top_n=6)
            reranked = self.reranker.compress_documents(question, fused, top_n=3)
            retrieved_docs = self.compressor.compress(question, reranked)

        else:
            retrieved_docs = self.dense_retriever.invoke(question)

        # 2. Context Formatting & Answer Generation
        context_str = self._format_context(retrieved_docs)
        answer = self.qa_chain.invoke({
            "question": question,
            "context": context_str,
        })

        return {
            "question": question,
            "strategy": strategy,
            "transformed_query": transformed_query,
            "retrieved_documents_count": len(retrieved_docs),
            "documents": retrieved_docs,
            "context": context_str,
            "answer": answer,
            "extra_metadata": extra_meta,
        }
