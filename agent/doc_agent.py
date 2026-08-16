"""Tool-Calling Agent Engine for DocMind.

Demonstrates:
- Modern LangChain create_agent with native tool binding
- Multi-step reasoning loops (Thought -> Tool Call -> Observation -> Final Answer)
- Graph-based agent execution with tool validation and error recovery
- Multi-turn conversation support via SessionHistoryManager
"""

from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent

from llm.provider import get_chat_model
from tools import get_docmind_tools
from memory.history_store import SessionHistoryManager


DOCMIND_AGENT_SYSTEM_PROMPT = (
    "You are DocMind Agent, an intelligent document analysis and reasoning assistant. "
    "You have access to a set of specialized tools to inspect document catalogs, search document "
    "contents using vector retrieval, and execute safe mathematical calculations.\n\n"
    "Guidelines:\n"
    "1. When answering factual questions about documents, use 'search_documents'.\n"
    "2. When answering quantitative, financial, or statistical questions (sums, averages, margins, percentages), "
    "   first search the relevant documents for numbers, and then use 'calculator' to compute exact values.\n"
    "3. When asked what files are available or to find files by type, use 'query_document_catalog'.\n"
    "4. Cite sources (filename, row/page) when referencing document data.\n"
    "5. If the tools do not contain the answer, state that clearly."
)


def create_docmind_agent(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
):
    """Constructs a modern tool-calling agent graph with tool binding."""
    model = llm or get_chat_model()
    agent_tools = tools if tools is not None else get_docmind_tools()
    prompt_text = system_prompt or DOCMIND_AGENT_SYSTEM_PROMPT

    return create_agent(
        model=model,
        tools=agent_tools,
        system_prompt=prompt_text,
    )


def extract_intermediate_steps(messages: List[BaseMessage]) -> List[Tuple[Any, Any]]:
    """Extracts (action, observation) pairs from agent execution messages for tracing."""
    steps = []
    pending_tool_calls: Dict[str, Any] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                call_id = tc.get("id", "")
                pending_tool_calls[call_id] = tc
        elif isinstance(msg, ToolMessage):
            call_id = msg.tool_call_id
            tc = pending_tool_calls.get(call_id, {"name": msg.name, "args": {}})
            steps.append((tc, msg.content))

    return steps


class DocMindAgent:
    """Stateful, multi-session conversational agent wrapping create_agent."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        history_manager: Optional[SessionHistoryManager] = None,
    ):
        self.llm = llm or get_chat_model()
        self.tools = tools if tools is not None else get_docmind_tools()
        self.system_prompt = system_prompt or DOCMIND_AGENT_SYSTEM_PROMPT
        self.history_manager = history_manager or SessionHistoryManager(storage_type="memory")
        self.agent = create_docmind_agent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def run(
        self,
        user_input: str,
        session_id: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs the agent on user input with session history and returns output with execution steps."""
        session_hist = self.history_manager.get_session_history(session_id)
        history_messages: List[BaseMessage] = list(session_hist.messages)

        # Append new user message
        new_human_msg = HumanMessage(content=user_input)
        input_messages = history_messages + [new_human_msg]

        # Execute agent
        result = self.agent.invoke({"messages": input_messages}, config=config)

        all_msgs = result.get("messages", [])
        final_ai_msg = all_msgs[-1] if all_msgs else AIMessage(content="")
        output_text = final_ai_msg.content if isinstance(final_ai_msg, AIMessage) else str(final_ai_msg)

        # Extract intermediate tool steps
        steps = extract_intermediate_steps(all_msgs)

        # Persist conversation turn to history
        session_hist.add_user_message(user_input)
        session_hist.add_ai_message(output_text)

        return {
            "input": user_input,
            "output": output_text,
            "intermediate_steps": steps,
            "messages": all_msgs,
            "session_id": session_id,
        }
