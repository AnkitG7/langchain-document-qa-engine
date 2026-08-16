"""Progressive Conversation Summary Memory.

Demonstrates:
- Token-aware memory management
- Summarizing older conversation turns when context budget is exceeded
- Preserving high-fidelity recent turns alongside a compressed historical summary
"""

from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from llm.provider import get_chat_model


class ProgressiveConversationSummary:
    """Maintains a progressive, token-aware conversation summary alongside recent messages."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        max_recent_messages: int = 4,
        max_tokens_threshold: int = 1500,
    ):
        self.llm = llm or get_chat_model()
        self.max_recent_messages = max_recent_messages
        self.max_tokens_threshold = max_tokens_threshold
        self.running_summary: str = ""
        self.messages: List[BaseMessage] = []

        # Summarization sub-chain
        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert conversation summarizer. "
                "Update the existing summary of the conversation to incorporate the new messages. "
                "Keep the summary concise and focused on key facts, decisions, and context.",
            ),
            (
                "human",
                "Current Summary:\n{existing_summary}\n\nNew Messages to incorporate:\n{new_lines}",
            ),
        ])
        self.summary_chain = summary_prompt | self.llm | StrOutputParser()

    def add_user_message(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))
        self._check_and_summarize()

    def add_ai_message(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))
        self._check_and_summarize()

    def _format_messages_for_summary(self, msgs: List[BaseMessage]) -> str:
        lines = []
        for m in msgs:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    def _check_and_summarize(self) -> None:
        """Triggers summarization if message count exceeds max_recent_messages."""
        if len(self.messages) > self.max_recent_messages:
            # Take older messages to condense into summary
            to_summarize = self.messages[:-self.max_recent_messages]
            new_lines = self._format_messages_for_summary(to_summarize)

            self.running_summary = self.summary_chain.invoke({
                "existing_summary": self.running_summary or "None yet.",
                "new_lines": new_lines,
            })

            # Retain only recent messages
            self.messages = self.messages[-self.max_recent_messages:]

    def get_context_for_prompt(self) -> dict:
        """Returns the running summary and the active recent messages for prompt injection."""
        return {
            "conversation_summary": self.running_summary,
            "recent_messages": list(self.messages),
        }
