"""Multi-Document Comparison Chains using LCEL.

Demonstrates:
- Multi-input prompt templates
- Structured schema comparison with Pydantic
- Comparative analysis and trade-off synthesis
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_core.language_models.chat_models import BaseChatModel

from llm.provider import get_chat_model


# ---------------------------------------------------------------------------
# 1. Structured Comparison Schema
# ---------------------------------------------------------------------------
class ComparisonPoint(BaseModel):
    """Specific dimension or topic of comparison between two documents."""
    dimension: str = Field(description="The feature, topic, or criteria being compared (e.g. 'Pricing', 'Security')")
    doc_a_position: str = Field(description="Summary of Document A's position or details on this dimension")
    doc_b_position: str = Field(description="Summary of Document B's position or details on this dimension")
    verdict_or_advantage: str = Field(description="Which document is stronger on this dimension, or key trade-off")


class StructuredComparisonReport(BaseModel):
    """Complete structured comparative analysis between two documents."""
    doc_a_title: str = Field(description="Title or identifier of Document A")
    doc_b_title: str = Field(description="Title or identifier of Document B")
    overall_summary: str = Field(description="High-level synthesis comparing both documents")
    common_ground: List[str] = Field(
        default_factory=list,
        description="Shared principles, overlapping features, or points of agreement",
    )
    points_of_contrast: List[ComparisonPoint] = Field(
        default_factory=list,
        description="Detailed point-by-point breakdown across distinct dimensions",
    )
    recommendation_or_conclusion: str = Field(
        description="Final takeaway or recommendation on when to choose A vs B",
    )


# ---------------------------------------------------------------------------
# 2. Text Comparison Chain
# ---------------------------------------------------------------------------
def create_text_compare_chain(
    llm: Optional[BaseChatModel] = None,
) -> RunnableSerializable:
    """Creates a basic comparative analysis LCEL chain."""
    model = llm or get_chat_model()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert document comparative analyst. Thoroughly analyze and compare the two documents provided. "
            "Highlight similarities, direct contradictions, structural differences, and unique advantages.",
        ),
        (
            "human",
            "=== DOCUMENT A ({doc_a_name}) ===\n{doc_a_content}\n\n"
            "=== DOCUMENT B ({doc_b_name}) ===\n{doc_b_content}\n\n"
            "Focus Criteria / Questions (if any): {criteria}\n\n"
            "Please provide a structured comparative analysis.",
        ),
    ])

    return prompt | model | StrOutputParser()


# ---------------------------------------------------------------------------
# 3. Structured Comparison Chain
# ---------------------------------------------------------------------------
def create_structured_compare_chain(
    llm: Optional[BaseChatModel] = None,
) -> RunnableSerializable:
    """Creates an LCEL chain outputting a strictly validated Pydantic comparison report."""
    model = llm or get_chat_model()
    parser = PydanticOutputParser(pydantic_object=StructuredComparisonReport)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert comparative analyst. Compare the two documents and return a structured analysis.\n"
            "{format_instructions}",
        ),
        (
            "human",
            "=== DOCUMENT A ({doc_a_name}) ===\n{doc_a_content}\n\n"
            "=== DOCUMENT B ({doc_b_name}) ===\n{doc_b_content}\n\n"
            "Criteria / Focus: {criteria}",
        ),
    ]).partial(format_instructions=parser.get_format_instructions())

    return prompt | model | parser
