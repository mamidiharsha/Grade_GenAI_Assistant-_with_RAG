"""
Application configuration using pydantic-settings.
Loads environment variables from .env file.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Google Gemini API
    GEMINI_API_KEY: str = "your_gemini_api_key_here"

    # RAG Configuration
    SIMILARITY_THRESHOLD: float = 0.35
    TOP_K: int = 3
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50

    # LLM Configuration
    LLM_TEMPERATURE: float = 0.2
    LLM_MODEL: str = "gemini-flash-lite-latest"

    # Conversation Memory
    MAX_HISTORY: int = 5

    # JWT Configuration (Bonus)
    JWT_SECRET: str = "your_jwt_secret_key_here"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "str_strip_whitespace": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance to avoid re-reading .env on every call."""
    return Settings()
