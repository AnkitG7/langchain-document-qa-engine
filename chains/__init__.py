"""Chains module exporting basic QA, structured QA, summarization, comparison, and composition pipelines."""

from .qa_chain import (
    create_basic_qa_chain,
    create_structured_qa_chain,
    create_parallel_qa_chain,
    StructuredQAResponse,
    Citation,
)
from .summary_chain import (
    create_stuff_summary_chain,
    create_map_reduce_summary_chain,
    summarize_with_refine,
    ExecutiveSummary,
)
from .compare_chain import (
    create_text_compare_chain,
    create_structured_compare_chain,
    StructuredComparisonReport,
    ComparisonPoint,
)
from .composition import (
    create_analyst_pipeline,
    ExtractedClaims,
    DocumentBriefing,
)

__all__ = [
    # QA
    "create_basic_qa_chain",
    "create_structured_qa_chain",
    "create_parallel_qa_chain",
    "StructuredQAResponse",
    "Citation",
    # Summarization
    "create_stuff_summary_chain",
    "create_map_reduce_summary_chain",
    "summarize_with_refine",
    "ExecutiveSummary",
    # Comparison
    "create_text_compare_chain",
    "create_structured_compare_chain",
    "StructuredComparisonReport",
    "ComparisonPoint",
    # Composition
    "create_analyst_pipeline",
    "ExtractedClaims",
    "DocumentBriefing",
]
