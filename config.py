"""Configuration settings for DocMind using Pydantic Settings."""

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable loading."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # LLM Generation / Reasoning Settings
    default_llm_provider: Literal["openai", "anthropic", "ollama", "fake"] = "ollama"
    default_model_name: str = "gemma4:cloud"
    default_temperature: float = 0.0
    max_tokens: Optional[int] = None

    # Dedicated Embedding Model Settings (Rule: Always use dedicated embedding model)
    default_embedding_provider: Literal["ollama", "openai", "huggingface", "fake"] = "ollama"
    ollama_embedding_model: str = "nomic-embed-text"
    openai_embedding_model: str = "text-embedding-3-small"
    huggingface_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Ollama Local Server
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "gemma4:cloud"

    # Ingestion & Chunking Defaults
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50
    default_splitter_type: Literal["recursive", "token", "markdown", "semantic"] = "recursive"

    # LangSmith Observability
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "docmind-langchain-qa"


settings = Settings()
