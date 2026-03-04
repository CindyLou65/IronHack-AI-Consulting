"""
config.py
─────────
Central configuration loader.
Reads all environment variables from .env and exposes them as typed constants.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = "gpt-4o"

# ── Tavily ────────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "ai-research-agent")
PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# ── LangSmith ─────────────────────────────────────────────────────────────────
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "ai-research-agent")

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ── Validation ────────────────────────────────────────────────────────────────
def validate_env() -> None:
    """Raise an error early if any required key is missing."""
    required = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "TAVILY_API_KEY": TAVILY_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "LANGCHAIN_API_KEY": LANGCHAIN_API_KEY,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example → .env and fill in your API keys."
        )