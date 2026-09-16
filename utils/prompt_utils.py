"""
utils/prompt_utils.py - Prompt construction for RAG + web search + response modes.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_BASE = """You are MediAssist AI — a knowledgeable, compassionate medical information assistant.

You help patients, caregivers, and healthcare professionals understand medical information clearly.

IMPORTANT RULES:
- You provide EDUCATIONAL information only, not personal medical advice.
- Always remind users to consult a qualified healthcare provider for personal health decisions.
- Be accurate, cite context when available, and acknowledge uncertainty honestly.
- Never diagnose conditions or prescribe treatments.
- If a question is outside medicine/health, politely redirect.
"""


def build_system_prompt(response_mode: str = "Detailed") -> str:
    """
    Build the full system prompt, incorporating response mode instructions.

    Args:
        response_mode: "Concise" or "Detailed"

    Returns:
        Full system prompt string.
    """
    try:
        from config.config import RESPONSE_MODES
        mode_instruction = RESPONSE_MODES.get(response_mode, RESPONSE_MODES["Detailed"])
        return f"{SYSTEM_BASE}\n\nRESPONSE STYLE: {mode_instruction}"
    except Exception as e:
        logger.error(f"System prompt build error: {e}")
        return SYSTEM_BASE


def build_user_message(
    query: str,
    rag_context: Optional[str] = None,
    web_results: Optional[str] = None,
) -> str:
    """
    Construct the full user message with injected RAG context and web results.

    Args:
        query: The raw user question.
        rag_context: Retrieved document chunks (optional).
        web_results: Web search results (optional).

    Returns:
        Enriched user message string.
    """
    try:
        parts = []

        if rag_context:
            parts.append(
                f"RELEVANT DOCUMENT CONTEXT:\n{rag_context}\n"
                f"Use the above context to inform your response when relevant."
            )

        if web_results:
            parts.append(
                f"REAL-TIME WEB SEARCH RESULTS:\n{web_results}\n"
                f"Use these results to supplement your knowledge with up-to-date information."
            )

        parts.append(f"USER QUESTION:\n{query}")

        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"User message build error: {e}")
        return query


def format_chat_history(history: list, max_turns: int = 10) -> list:
    """
    Convert Streamlit session history to LLM message format.
    Limits to last N turns to stay within token limits, and strips any
    extra UI-only keys (e.g. "sources", used for the sidebar display)
    that providers like Groq/OpenAI reject as unrecognized message fields.

    Args:
        history: List of {"role": ..., "content": ..., ...} dicts. Any
            keys beyond "role"/"content" are UI metadata and are dropped.
        max_turns: Maximum number of recent turns to include.

    Returns:
        Trimmed list of {"role": ..., "content": ...} dicts only.
    """
    try:
        if len(history) > max_turns * 2:
            history = history[-(max_turns * 2):]
        return [{"role": m["role"], "content": m["content"]} for m in history]
    except Exception as e:
        logger.error(f"History format error: {e}")
        return history