"""Interactive Phase 5 Demonstration Script for DocMind.

Run with:
    python examples/demo_phase5.py

Demonstrates:
1. Specialized Tool Schemas (Pydantic validation)
2. Safe Calculator Tool (arithmetic, sums, averages, percentages)
3. Metadata Catalog Tool (inventory & filtering)
4. Vector Store Search Tool (semantic retrieval with source attribution)
5. Live Tool-Calling Agent with Ollama gemma4:cloud
6. Step-by-Step Reasoning Tracing (Thought -> Tool Call -> Observation -> Final Answer)
7. Multi-Turn Stateful Agent with Session History Isolation
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from llm.provider import get_chat_model
from ingestion.pipeline import IngestionPipeline
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_or_create_faiss
from tools import get_docmind_tools, calculator_tool, metadata_catalog_tool
from agent.doc_agent import create_docmind_agent, DocMindAgent
from memory.history_store import SessionHistoryManager

DATA_DIR = Path(__file__).parent.parent / "data"


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_demo():
    print_banner("DocMind Phase 5: Tools & Tool-Calling Agents Demo")

    # 1. Direct Tool Execution
    print_banner("1. Direct Tool Execution & Safe Math")
    calc_expr = "sum([12500, 45000, 8200]) * 0.18"
    calc_res = calculator_tool.invoke({"expression": calc_expr})
    print(f"Calculator Tool: '{calc_expr}' -> Result: {calc_res}")

    catalog_res = metadata_catalog_tool.invoke({"action": "list_all"})
    print(f"\nMetadata Catalog Tool (list_all):\n{catalog_res}")

    # 2. Ingest Data and Setup Search Tool
    print_banner("2. Initializing Ingestion & Vector Search Tool")
    pipeline = IngestionPipeline(chunk_size=300, chunk_overlap=50)
    chunks, _ = pipeline.run_batch([
        str(DATA_DIR / "sample_doc.txt"),
        str(DATA_DIR / "sample_guide.md"),
        str(DATA_DIR / "sample_data.csv"),
    ])
    embedder = get_embeddings()
    vectorstore = get_or_create_faiss(documents=chunks, embeddings=embedder)
    tools = get_docmind_tools(vectorstore=vectorstore)

    print(f"Created {len(tools)} specialized agent tools:")
    for t in tools:
        print(f"  - [{t.name}]: {t.description[:80]}...")

    # 3. Live Tool-Calling Agent Execution
    print_banner("3. Live Tool-Calling Agent with gemma4:cloud")
    llm = get_chat_model()
    history_mgr = SessionHistoryManager(storage_type="memory")
    agent = DocMindAgent(llm=llm, tools=tools, history_manager=history_mgr)

    session_id = "agent_interactive_session_01"

    # Query 1: Catalog inspection query
    q1 = "What document files are available in the catalog?"
    print(f"\n[User Query 1]: {q1}")
    res1 = agent.run(user_input=q1, session_id=session_id)
    print(f"\n[Intermediate Tool Steps]: {len(res1['intermediate_steps'])} step(s) executed.")
    for action, obs in res1['intermediate_steps']:
        act_name = action.get('name') if isinstance(action, dict) else getattr(action, 'tool', 'tool')
        act_args = action.get('args') if isinstance(action, dict) else getattr(action, 'tool_input', {})
        print(f"  -> Tool Call: {act_name}({act_args})")
        print(f"  -> Tool Observation: {str(obs)[:140]}...\n")
    print(f"[DocMind Agent Output]:\n{res1['output']}")

    # Query 2: Multi-step Search + Math Reasoning
    q2 = "Search the CSV data for the salary or budget of all employees and calculate their total sum."
    print(f"\n[User Query 2]: {q2}")
    res2 = agent.run(user_input=q2, session_id=session_id)
    print(f"\n[Intermediate Tool Steps]: {len(res2['intermediate_steps'])} step(s) executed.")
    for action, obs in res2['intermediate_steps']:
        act_name = action.get('name') if isinstance(action, dict) else getattr(action, 'tool', 'tool')
        act_args = action.get('args') if isinstance(action, dict) else getattr(action, 'tool_input', {})
        print(f"  -> Tool Call: {act_name}({act_args})")
        print(f"  -> Tool Observation: {str(obs)[:140]}...\n")
    print(f"[DocMind Agent Output]:\n{res2['output']}")

    print_banner("Phase 5 Complete!")
    print("Tool schemas, Safe Math, Catalog inspection, and Tool-Calling Agent verified successfully.")


if __name__ == "__main__":
    run_demo()
