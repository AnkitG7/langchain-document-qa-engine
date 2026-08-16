"""Conversation Message Windowing and Token-Budget Trimming.

Demonstrates:
- langchain_core.messages.trim_messages
- Sliding window conversation management
- Pinning system messages while dropping stale middle turns
- Preventing LLM context overflow in long conversations
"""

from typing import Callable, List, Literal, Optional, Union
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.language_models.chat_models import BaseChatModel


def create_message_trimmer(
    max_tokens: int = 2000,
    strategy: Literal["last", "first"] = "last",
    include_system: bool = True,
    allow_partial: bool = False,
    start_on: Union[str, type] = "human",
    token_counter: Optional[Union[Callable, BaseChatModel]] = None,
):
    """Creates a configured message trimmer Runnable using LangChain's trim_messages.

    Args:
        max_tokens: Maximum allowed token budget.
        strategy: 'last' keeps the most recent turns.
        include_system: If True, preserves SystemMessage at the start.
        allow_partial: If False, drops full messages rather than splitting inside a message.
        start_on: 'human' ensures conversation begins on a user turn.
        token_counter: Custom token counting function or ChatModel instance.
    """
    def _default_token_counter(messages: List[BaseMessage]) -> int:
        # Simple character-based estimate (~4 chars per token) if no model passed
        total_chars = sum(len(str(m.content)) for m in messages)
        return max(1, total_chars // 4)

    counter = token_counter or _default_token_counter

    return trim_messages(
        max_tokens=max_tokens,
        strategy=strategy,
        token_counter=counter,
        include_system=include_system,
        allow_partial=allow_partial,
        start_on=start_on,
    )


def trim_conversation_history(
    messages: List[BaseMessage],
    max_messages: int = 10,
    max_tokens: Optional[int] = None,
) -> List[BaseMessage]:
    """Helper to trim a list of messages by message count and/or token budget."""
    if not messages:
        return []

    # Separate system message if present
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    chat_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    # Trim by message count first (keeping last N)
    if len(chat_msgs) > max_messages:
        chat_msgs = chat_msgs[-max_messages:]

    # Ensure conversation starts on Human turn
    while chat_msgs and not isinstance(chat_msgs[0], HumanMessage):
        chat_msgs.pop(0)

    # Optional token trimming
    if max_tokens:
        trimmer = create_message_trimmer(max_tokens=max_tokens, include_system=True)
        return trimmer.invoke(system_msgs + chat_msgs)

    return system_msgs + chat_msgs
