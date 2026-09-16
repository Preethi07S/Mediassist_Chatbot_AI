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

# Selectable models per provider (curated, verified against each
# provider's docs — update this list as providers deprecate/release models).
PROVIDER_MODELS = {
    "openai": [
        "gpt-5",          # flagship
        "gpt-5-mini",     # fast & cheap, well-defined tasks
        "gpt-4.1-mini",   # solid previous-gen alternative
        "gpt-4o-mini",    # legacy, still served via the API
    ],
    "groq": [
        # NOTE: llama-3.3-70b-versatile and llama-3.1-8b-instant were
        # decommissioned by Groq on 2026-08-16 for free/developer tiers.
        # These are Groq's own recommended replacements.
        "openai/gpt-oss-120b",  # strongest reasoning on Groq (replaces llama-3.3-70b-versatile)
        "openai/gpt-oss-20b",   # fastest/compact (replaces llama-3.1-8b-instant)
        "qwen/qwen3.6-27b",     # alternative to gpt-oss-120b (preview, pricier)
    ],
    "gemini": [
        # NOTE: gemini-2.5-pro and gemini-2.5-flash are restricted or
        # unavailable to new Google AI Studio accounts (confirmed via a
        # live 404 pointing at the 3.x line as of Sept 2026) — using the
        # 3.x generation throughout to avoid the same wall.
        "gemini-3.5-flash",       # current flagship fast model (GA)
        "gemini-3.1-flash-lite",  # fastest/cheapest, stable GA
        "gemini-3.1-pro-preview", # strongest reasoning (preview)
    ],
}

# Default (first) model per provider — used when the user hasn't picked one yet.
OPENAI_MODEL = PROVIDER_MODELS["openai"][0]
GROQ_MODEL = PROVIDER_MODELS["groq"][0]
GEMINI_MODEL = PROVIDER_MODELS["gemini"][0]

# Human-readable display names for each model ID (used in the dropdown so
# end users see something friendlier than raw API model strings).
MODEL_LABELS = {
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4o-mini": "GPT-4o Mini",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "qwen/qwen3.6-27b": "Qwen 3.6 27B",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro (Preview)",
}

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