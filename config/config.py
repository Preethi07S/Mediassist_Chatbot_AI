"""
config.py - Central configuration file for all API keys and settings.
All keys are loaded from environment variables for security.
"""

import os


def _get_secret(key: str, default: str = "") -> str:
    """
    Read a secret from environment variables first,
    then fall back to Streamlit secrets (for cloud deployment).
    """
    val = os.environ.get(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ─────────────────────────────────────────────
# LLM PROVIDER KEYS
# ─────────────────────────────────────────────
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")

# ─────────────────────────────────────────────
# WEB SEARCH KEY (Tavily)
# ─────────────────────────────────────────────
TAVILY_API_KEY = _get_secret("TAVILY_API_KEY")

# ─────────────────────────────────────────────
# LLM MODEL DEFAULTS
# ─────────────────────────────────────────────
DEFAULT_LLM_PROVIDER = "groq"  # "openai" | "groq" | "gemini"

OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL = "llama3-8b-8192"
GEMINI_MODEL = "gemini-2.5-flash"

# ─────────────────────────────────────────────
# EMBEDDING SETTINGS
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Sentence-transformers model (local, free)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4

# ─────────────────────────────────────────────
# RESPONSE MODES
# ─────────────────────────────────────────────
RESPONSE_MODES = {
    "Concise": "Respond briefly and directly. Keep the answer under 3 sentences unless a list is essential.",
    "Detailed": "Respond thoroughly with context, examples, and explanations. Be comprehensive.",
}

# ─────────────────────────────────────────────
# APP METADATA
# ─────────────────────────────────────────────
APP_TITLE = "MediAssist AI"
APP_SUBTITLE = "Your intelligent medical knowledge companion"
APP_ICON = "🏥"

# ─────────────────────────────────────────────
# VECTOR STORE PATH
# ─────────────────────────────────────────────
VECTOR_STORE_PATH = "data/vector_store"
