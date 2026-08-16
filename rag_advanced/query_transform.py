"""Query Transformation Strategies for Advanced RAG.

Demonstrates:
- Multi-Query Expansion: Overcoming distance-based vector blind spots via multiple query perspectives.
- HyDE (Hypothetical Document Embeddings): Embedding in answer-space rather than question-space.
- Step-Back Prompting: Generating high-level conceptual questions to retrieve foundational context.
"""

from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from llm.provider import get_chat_model
from ingestion.cleaner import calculate_content_hash


class HyDETransformer:
    """Hypothetical Document Embeddings (HyDE) Generator and Retriever.

    Generates a hypothetical document answering the question, then retrieves chunks matching
    the hypothetical passage in embedding space.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm or get_chat_model()
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert domain document writer. Write a detailed, factual paragraph answering "
                "the user's question as if it were an excerpt from a technical document or manual. "
                "Do NOT include conversational filler, greetings, or meta-comments.",
            ),
            ("human", "{question}"),
        ])
        self.chain = prompt | self.llm | StrOutputParser()

    def generate_hypothetical_document(self, query: str) -> str:
        """Generates a hypothetical answer passage for the query."""
        return self.chain.invoke({"question": query}).strip()

    def retrieve(self, query: str, retriever: VectorStoreRetriever) -> List[Document]:
        """Generates hypothetical passage and retrieves matching documents from vector store."""
        hypo_doc = self.generate_hypothetical_document(query)
        # Query vector store with the hypothetical passage
        return retriever.invoke(hypo_doc)


class MultiQueryTransformer:
    """Multi-Query Expansion Generator and Deduplicating Retriever.

    Generates multiple distinct reformulations of the user query to capture diverse semantic angles.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None, num_queries: int = 3):
        self.llm = llm or get_chat_model()
        self.num_queries = num_queries
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"You are an AI search assistant. Generate {self.num_queries} different search query "
                "variations for the user question to retrieve comprehensive documents from a vector database. "
                "Provide exactly one query per line without numbering, bullets, or commentary.",
            ),
            ("human", "{question}"),
        ])
        self.chain = prompt | self.llm | StrOutputParser()

    def generate_queries(self, query: str) -> List[str]:
        """Generates multiple search queries including the original."""
        raw_output = self.chain.invoke({"question": query})
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        queries = [query]
        for line in lines:
            cleaned = line.lstrip("0123456789.-* ")
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
        return queries[: self.num_queries + 1]

    def retrieve(self, query: str, retriever: VectorStoreRetriever) -> List[Document]:
        """Executes retrieval across all query variants and deduplicates results."""
        all_queries = self.generate_queries(query)
        seen_hashes = set()
        unique_docs: List[Document] = []

        for q in all_queries:
            docs = retriever.invoke(q)
            for d in docs:
                h = calculate_content_hash(d.page_content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_docs.append(d)

        return unique_docs


class StepBackTransformer:
    """Step-Back Prompting Transformer.

    Generates a higher-level, broader conceptual question to retrieve foundational background principles.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm or get_chat_model()
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert at problem-solving and abstraction. "
                "Given a specific question, generate a broader, higher-level 'step-back' question "
                "that asks about the general concept, underlying architecture, or fundamental principle. "
                "Output ONLY the step-back question without quotes or preamble.\n\n"
                "Example 1:\nSpecific: What is the batch chunk overlap in DocMind's ingestion pipeline?\nStep-back: How does text chunking work in document retrieval?\n\n"
                "Example 2:\nSpecific: What was the Q2 revenue for Project Phoenix in 2026?\nStep-back: What is the overall financial performance and budget of projects?",
            ),
            ("human", "{question}"),
        ])
        self.chain = prompt | self.llm | StrOutputParser()

    def generate_step_back_query(self, query: str) -> str:
        """Generates a high-level conceptual question."""
        return self.chain.invoke({"question": query}).strip()

    def retrieve(self, query: str, retriever: VectorStoreRetriever) -> Dict[str, List[Document]]:
        """Retrieves both specific and step-back background documents."""
        step_back_q = self.generate_step_back_query(query)
        specific_docs = retriever.invoke(query)
        step_back_docs = retriever.invoke(step_back_q)

        return {
            "specific_query": query,
            "step_back_query": step_back_q,
            "specific_docs": specific_docs,
            "step_back_docs": step_back_docs,
        }
