"""LLM abstraction module for multi-provider support and fallback logic."""

from .provider import get_chat_model, create_fallback_model

__all__ = ["get_chat_model", "create_fallback_model"]
