"""Chain of Chains Composition using LCEL.

Demonstrates:
- Composing multiple specialized chains sequentially
- Passing intermediate outputs as structured inputs to subsequent chains
- Modular multi-step cognitive pipeline (Extract -> Critique -> Synthesize)
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.language_models.chat_models import BaseChatModel

from llm.provider import get_chat_model


# ---------------------------------------------------------------------------
# 1. Schemas for Intermediate and Final Outputs
# ---------------------------------------------------------------------------
class ExtractedClaims(BaseModel):
    """Schema for extracted assertions."""
    topic: str = Field(description="Main subject of the document")
    claims: List[str] = Field(description="List of factual claims extracted from document")


class DocumentBriefing(BaseModel):
    """Final synthesized briefing report."""
    topic: str
    verified_points: List[str]
    potential_risks_or_assumptions: List[str]
    executive_verdict: str


# ---------------------------------------------------------------------------
# 2. Composed Chain (Extract -> Critique -> Briefing)
# ---------------------------------------------------------------------------
def create_analyst_pipeline(
    llm: Optional[BaseChatModel] = None,
) -> RunnableSerializable:
    """Builds a 3-step cognitive chain composition:

    Step 1 (Extract): Document -> JSON list of claims
    Step 2 (Critique): Claims + Original Doc -> Critical Assessment
    Step 3 (Synthesize): Claims + Critique -> Final Executive Briefing
    """
    model = llm or get_chat_model()
    json_parser = JsonOutputParser(pydantic_object=ExtractedClaims)
    str_parser = StrOutputParser()

    # Step 1: Extraction Chain
    extract_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract the core factual claims and the main topic from the document.\n{format_instructions}",
        ),
        ("human", "{document_text}"),
    ]).partial(format_instructions=json_parser.get_format_instructions())
    extract_chain = extract_prompt | model | json_parser

    # Step 2: Critique Prompt
    critique_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a rigorous auditor. Review the extracted claims against common fallacies, unproven assumptions, or risks.",
        ),
        (
            "human",
            "Topic: {topic}\nClaims: {claims}\nOriginal Document:\n{document_text}",
        ),
    ])
    critique_chain = critique_prompt | model | str_parser

    # Step 3: Synthesis Prompt
    synthesis_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a Senior Advisor. Synthesize the findings and critique into an actionable briefing note.",
        ),
        (
            "human",
            "Topic: {topic}\nClaims: {claims}\nCritique / Risks:\n{critique}",
        ),
    ])
    synthesis_chain = synthesis_prompt | model | str_parser

    # Sequential composition in LCEL:
    # 1. Take document_text, extract claims
    # 2. Pass document_text + claims to critique_chain
    # 3. Pass topic, claims + critique to synthesis_chain
    full_pipeline = (
        RunnablePassthrough.assign(
            extracted=extract_chain,
        )
        | RunnablePassthrough.assign(
            topic=lambda x: x["extracted"].get("topic", "General"),
            claims=lambda x: x["extracted"].get("claims", []),
        )
        | RunnablePassthrough.assign(
            critique=critique_chain,
        )
        | RunnablePassthrough.assign(
            final_briefing=synthesis_chain,
        )
    )

    return full_pipeline
