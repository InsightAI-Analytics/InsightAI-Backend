"""
Configuration management for InsightAI backend.
Reads from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM (Gemini)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Legacy / OpenAI fallback support
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # CORS
    allowed_origins: list[str] = [
        o.strip()
        for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]

    # Upload limits
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    max_files_per_session: int = int(os.getenv("MAX_FILES_PER_SESSION", "10"))


settings = Settings()
