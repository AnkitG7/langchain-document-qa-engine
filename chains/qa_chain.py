"""LCEL Question Answering Chains (Basic, Structured, and Parallel).

Demonstrates:
- ChatPromptTemplate, SystemMessage, HumanMessage
- LCEL pipe syntax: prompt | llm | parser
- StrOutputParser, JsonOutputParser, PydanticOutputParser
- RunnablePassthrough, RunnableLambda, RunnableParallel
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableSerializable,
)
from langchain_core.language_models.chat_models import BaseChatModel

from llm.provider import get_chat_model


# ---------------------------------------------------------------------------
# 1. Pydantic Models for Structured QA Output
# ---------------------------------------------------------------------------
class Citation(BaseModel):
    """Source citation for a claim in the answer."""
    source_name: str = Field(description="Name or identifier of the source document/section")
    quote_or_fact: str = Field(description="Specific snippet or fact supporting the answer")


class StructuredQAResponse(BaseModel):
    """Schema for structured document Q&A output."""
    answer: str = Field(description="Direct, comprehensive answer to the user query")
    confidence_score: float = Field(
        description="Confidence score between 0.0 (unsupported) and 1.0 (fully verified in context)",
        ge=0.0,
        le=1.0,
    )
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="Bulleted key takeaways or actionable points",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of exact citations and references found in context",
    )
    limitations: Optional[str] = Field(
        default=None,
        description="Information missing from context or caveats",
    )


# ---------------------------------------------------------------------------
# 2. Basic Text QA Chain (prompt | llm | StrOutputParser)
# ---------------------------------------------------------------------------
def create_basic_qa_chain(
    llm: Optional[BaseChatModel] = None,
    system_prompt: Optional[str] = None,
) -> RunnableSerializable:
    """Creates a basic question-answering LCEL chain returning string output.

    Pipeline:
        input dict {"context": ..., "question": ...}
          ↓
        ChatPromptTemplate
          ↓
        BaseChatModel
          ↓
        StrOutputParser
    """
    model = llm or get_chat_model()

    default_system = (
        "You are DocMind, an intelligent document analysis assistant. "
        "Answer the user's question accurately and concisely based ONLY on the provided context. "
        "If the answer cannot be deduced from the context, say 'I cannot find that in the provided documents.'"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt or default_system),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    parser = StrOutputParser()

    # Pure LCEL composition using pipe syntax
    chain = prompt | model | parser
    return chain


# ---------------------------------------------------------------------------
# 3. Structured QA Chain (Pydantic / JSON Output Parser)
# ---------------------------------------------------------------------------
def create_structured_qa_chain(
    llm: Optional[BaseChatModel] = None,
) -> RunnableSerializable:
    """Creates an LCEL chain enforcing a validated Pydantic schema output.

    Uses PydanticOutputParser with injected format instructions in the prompt.
    """
    model = llm or get_chat_model()
    parser = PydanticOutputParser(pydantic_object=StructuredQAResponse)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are DocMind, an expert document analyst. Extract and answer user questions with strict precision.\n"
            "{format_instructions}\n"
            "Ensure you evaluate confidence accurately based on whether the context contains explicit proof.",
        ),
        (
            "human",
            "Document Context:\n{context}\n\nUser Question: {question}",
        ),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser
    return chain


# ---------------------------------------------------------------------------
# 4. Advanced Parallel LCEL Chain (RunnableParallel + RunnableLambda)
# ---------------------------------------------------------------------------
def format_context_docs(docs: List[Dict[str, Any]]) -> str:
    """RunnableLambda helper to format list of document objects/dicts into a unified string."""
    formatted = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.get("source", f"Doc {idx}")
        content = doc.get("content", doc.get("page_content", ""))
        formatted.append(f"--- [Source: {source}] ---\n{content}")
    return "\n\n".join(formatted)


def create_parallel_qa_chain(
    llm: Optional[BaseChatModel] = None,
) -> RunnableSerializable:
    """Demonstrates advanced LCEL execution using RunnableParallel, RunnablePassthrough, and RunnableLambda.

    Features:
    - Parallel execution: Formats context while passing question through
    - Multi-angle analysis: Runs a quick summary and detailed Q&A in parallel
    """
    model = llm or get_chat_model()
    str_parser = StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using the formatted context."),
        ("human", "Context:\n{formatted_context}\n\nQuestion: {question}"),
    ])

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Provide a 1-sentence summary of the context provided."),
        ("human", "Context:\n{formatted_context}"),
    ])

    # 1. Input preparation branch using RunnableParallel and RunnableLambda
    context_prep = RunnableParallel({
        "formatted_context": RunnableLambda(lambda x: format_context_docs(x["documents"])),
        "question": lambda x: x["question"],
        "raw_docs_count": lambda x: len(x["documents"]),
    })

    # 2. Execution branches in parallel
    qa_branch = (
        RunnablePassthrough.assign(
            formatted_context=lambda x: x["formatted_context"],
            question=lambda x: x["question"],
        )
        | qa_prompt
        | model
        | str_parser
    )

    summary_branch = (
        RunnablePassthrough.assign(formatted_context=lambda x: x["formatted_context"])
        | summary_prompt
        | model
        | str_parser
    )

    # 3. Final aggregated parallel chain
    full_chain = context_prep | RunnableParallel({
        "answer": qa_branch,
        "context_summary": summary_branch,
        "documents_analyzed": lambda x: x["raw_docs_count"],
    })

    return full_chain
