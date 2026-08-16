"""Multi-provider LLM Factory and Fallback Architecture.

Demonstrates:
- BaseChatModel unified interface
- ChatOpenAI, ChatAnthropic, ChatOllama, and FakeListChatModel instantiation
- Dynamic model selection via configuration
- Fallback chains using `with_fallbacks`
"""

from typing import List, Literal, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from config import settings


def get_chat_model(
    provider: Optional[Literal["openai", "anthropic", "ollama", "fake"]] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    fake_responses: Optional[List[str]] = None,
    **kwargs,
) -> BaseChatModel:
    """Instantiate and return a standardized LangChain ChatModel.

    Args:
        provider: 'openai', 'anthropic', 'ollama', or 'fake'. Defaults to settings.default_llm_provider.
        model_name: The specific model ID (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet-20241022', 'llama3.2').
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens: Max output tokens.
        fake_responses: Predefined responses if provider is 'fake' (useful for deterministic tests).
        **kwargs: Additional provider-specific parameters.

    Returns:
        BaseChatModel: An instantiated LangChain chat model.
    """
    selected_provider = provider or settings.default_llm_provider
    selected_temp = temperature if temperature is not None else settings.default_temperature
    selected_max_tokens = max_tokens or settings.max_tokens

    if selected_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Please install langchain-openai: `pip install langchain-openai`")

        api_key = settings.openai_api_key
        return ChatOpenAI(
            model=model_name or settings.default_model_name or "gpt-4o-mini",
            temperature=selected_temp,
            max_tokens=selected_max_tokens,
            api_key=api_key if api_key else None,
            **kwargs,
        )

    elif selected_provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Please install langchain-anthropic: `pip install langchain-anthropic`")

        api_key = settings.anthropic_api_key
        return ChatAnthropic(
            model_name=model_name or "claude-3-5-sonnet-20241022",
            temperature=selected_temp,
            max_tokens_to_sample=selected_max_tokens or 4096,
            api_key=api_key if api_key else None,
            **kwargs,
        )

    elif selected_provider == "ollama":
        ChatOllama = None
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                raise ImportError(
                    "Please install langchain-ollama: `pip install langchain-ollama`"
                )

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model_name or settings.ollama_model_name,
            temperature=selected_temp,
            **kwargs,
        )

    elif selected_provider == "fake":
        default_responses = fake_responses or [
            "This is a simulated AI response for testing and offline development."
        ]
        return FakeListChatModel(responses=default_responses)

    else:
        raise ValueError(
            f"Unsupported LLM provider '{selected_provider}'. Supported: 'openai', 'anthropic', 'ollama', 'fake'."
        )


def create_fallback_model(
    primary: BaseChatModel,
    fallbacks: List[BaseChatModel],
) -> BaseChatModel:
    """Wraps a primary chat model with automatic fallback models.

    If the primary model fails (e.g. rate limits, API outage, 500 error),
    LangChain will automatically retry execution using the fallback models in order.

    Example:
        primary = get_chat_model(provider="openai", model_name="gpt-4o")
        backup = get_chat_model(provider="anthropic", model_name="claude-3-5-sonnet-20241022")
        robust_llm = create_fallback_model(primary, [backup])
    """
    return primary.with_fallbacks(fallbacks)
