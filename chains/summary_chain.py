"""Document Summarization Chains using Modern LCEL (Stuff, Map-Reduce, and Refine).

Demonstrates:
- Map-Reduce pattern in pure LCEL
- Batch mapping over document chunks with `.map()`
- Structured executive summary output
- Refine iterative summarization
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.language_models.chat_models import BaseChatModel

from llm.provider import get_chat_model


# ---------------------------------------------------------------------------
# 1. Structured Summary Schema
# ---------------------------------------------------------------------------
class ExecutiveSummary(BaseModel):
    """Structured executive summary of a document."""
    title: str = Field(description="Inferred or extracted document title")
    executive_summary: str = Field(description="High-level 2-3 paragraph overview")
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key factual takeaways and data points",
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="Actionable next steps or recommendations mentioned",
    )
    topics_covered: List[str] = Field(
        default_factory=list,
        description="List of main topics or domains covered",
    )


# ---------------------------------------------------------------------------
# 2. Stuff Summarization Chain
# ---------------------------------------------------------------------------
def create_stuff_summary_chain(
    llm: Optional[BaseChatModel] = None,
    structured: bool = False,
) -> RunnableSerializable:
    """Creates a 'stuff' summarizer for documents that fit into a single context window."""
    model = llm or get_chat_model()

    if structured:
        parser = PydanticOutputParser(pydantic_object=ExecutiveSummary)
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert document summarizer. Analyze the text and produce a structured executive summary.\n"
                "{format_instructions}",
            ),
            ("human", "Document Content:\n{text}"),
        ]).partial(format_instructions=parser.get_format_instructions())
        return prompt | model | parser

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert summarizer. Provide a clear, concise, and structured summary."),
        ("human", "Document Content:\n{text}"),
    ])
    return prompt | model | StrOutputParser()


# ---------------------------------------------------------------------------
# 3. Map-Reduce Summarization Chain (LCEL)
# ---------------------------------------------------------------------------
def create_map_reduce_summary_chain(
    llm: Optional[BaseChatModel] = None,
) -> RunnableSerializable:
    """Builds a Map-Reduce summarization pipeline using modern LCEL.

    1. Map Phase: Summarizes each document chunk individually.
    2. Reduce Phase: Combines chunk summaries into a cohesive, structured final summary.
    """
    model = llm or get_chat_model()
    str_parser = StrOutputParser()

    # Map Step Prompt
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Summarize the key facts, findings, and arguments from this specific document excerpt.",
        ),
        ("human", "Excerpt:\n{chunk}"),
    ])
    map_chain = map_prompt | model | str_parser

    # Reduce Step Prompt
    reduce_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Synthesize the following collection of section summaries into a comprehensive final summary with key takeaways.",
        ),
        ("human", "Section Summaries:\n{summaries}"),
    ])
    reduce_chain = reduce_prompt | model | str_parser

    def map_step(chunks: List[str]) -> List[str]:
        """Maps over chunk inputs using LangChain's batching capability."""
        # Using batch for concurrent execution of chunks
        inputs = [{"chunk": chunk} for chunk in chunks]
        return map_chain.batch(inputs)

    def format_summaries(summaries: List[str]) -> str:
        """Formats intermediate summaries into a single text block for the reducer."""
        return "\n\n".join(f"--- Section {i+1} ---\n{s}" for i, s in enumerate(summaries))

    # Compose the full Map-Reduce Runnable
    map_reduce_chain = (
        RunnableLambda(map_step)
        | RunnableLambda(format_summaries)
        | (lambda text: {"summaries": text})
        | reduce_chain
    )

    return map_reduce_chain


# ---------------------------------------------------------------------------
# 4. Refine Summarization Chain
# ---------------------------------------------------------------------------
def summarize_with_refine(
    chunks: List[str],
    llm: Optional[BaseChatModel] = None,
) -> str:
    """Iteratively summarizes chunks by passing the existing summary along with the next chunk."""
    if not chunks:
        return "No content provided to summarize."

    model = llm or get_chat_model()
    parser = StrOutputParser()

    initial_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following initial section of a document."),
        ("human", "{chunk}"),
    ])
    initial_chain = initial_prompt | model | parser

    refine_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are refining an existing document summary. We have provided an existing summary up to a certain point: "
            "{existing_summary}\n\n"
            "We have additional context below. Refine the existing summary incorporating any new crucial details.",
        ),
        ("human", "New Chunk:\n{chunk}"),
    ])
    refine_chain = refine_prompt | model | parser

    # First chunk gets initial summary
    current_summary = initial_chain.invoke({"chunk": chunks[0]})

    # Subsequent chunks refine the summary
    for chunk in chunks[1:]:
        current_summary = refine_chain.invoke({
            "existing_summary": current_summary,
            "chunk": chunk,
        })

    return current_summary
