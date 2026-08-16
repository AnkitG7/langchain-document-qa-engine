"""Cross-Encoder / LLM-Based Document Reranking Module.

Demonstrates:
- Scoring retrieved candidate documents for query relevance using an LLM evaluator
- Re-sorting candidate pool to prioritize high-precision passages
- Filtering out low-relevance noise before final answer generation
"""

import re
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from llm.provider import get_chat_model


class LLMReranker:
    """Reranks candidate documents using LLM relevance scoring."""

    def __init__(self, llm: Optional[BaseChatModel] = None, top_n: int = 3):
        self.llm = llm or get_chat_model()
        self.top_n = top_n

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert document relevance judge. "
                "Evaluate how relevant the following document excerpt is to answering the user question. "
                "Assign a relevance score from 0 to 10 (10 = directly answers the question, 0 = completely irrelevant).\n"
                "Respond with ONLY a single integer score from 0 to 10 without any explanation.",
            ),
            (
                "human",
                "Question: {question}\n\nDocument Excerpt:\n{document}\n\nScore (0-10):",
            ),
        ])
        self.scoring_chain = prompt | self.llm | StrOutputParser()

    def _score_document(self, query: str, doc: Document) -> float:
        """Scores a single document excerpt against the query."""
        try:
            raw_score = self.scoring_chain.invoke({
                "question": query,
                "document": doc.page_content[:1000],
            })
            # Extract first integer in response
            match = re.search(r"\b(10|[0-9])\b", raw_score.strip())
            if match:
                return float(match.group(1))
            return 5.0
        except Exception:
            return 5.0

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Scores all candidate documents, sorts by score descending, and returns top N with scores."""
        if not documents:
            return []

        limit = top_n or self.top_n
        scored_docs: List[Tuple[Document, float]] = []

        for doc in documents:
            score = self._score_document(query, doc)
            scored_docs.append((doc, score))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:limit]

    def compress_documents(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Document]:
        """Returns the top N documents after reranking."""
        scored = self.rerank(query, documents, top_n=top_n)
        return [doc for doc, _ in scored]
