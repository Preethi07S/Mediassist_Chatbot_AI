"""
models/llm.py - LLM provider abstraction layer.
Supports OpenAI, Groq, and Google Gemini.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_llm_response(
    messages: list,
    provider: str,
    model: Optional[str] = None,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Unified interface to call any supported LLM provider.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        provider: "openai" | "groq" | "gemini"
        model: Specific model ID for the chosen provider. If omitted, each
            provider function falls back to its configured default.
        system_prompt: System-level instruction
        temperature: Creativity level (0–1)
        max_tokens: Max tokens to generate

    Returns:
        The model's response as a plain string.
    """
    try:
        if provider == "openai":
            return _call_openai(messages, system_prompt, temperature, max_tokens, model)
        elif provider == "groq":
            return _call_groq(messages, system_prompt, temperature, max_tokens, model)
        elif provider == "gemini":
            return _call_gemini(messages, system_prompt, temperature, max_tokens, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        logger.error(f"LLM call failed for provider '{provider}': {e}")
        raise


# ─────────────────────────────────────────────
# OPENAI
# ─────────────────────────────────────────────

def _call_openai(messages: list, system_prompt: str, temperature: float, max_tokens: int, model: Optional[str] = None) -> str:
    try:
        from openai import OpenAI
        from config.config import OPENAI_API_KEY, OPENAI_MODEL

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")

        client = OpenAI(api_key=OPENAI_API_KEY)

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = client.chat.completions.create(
            model=model or OPENAI_MODEL,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise


# ─────────────────────────────────────────────
# GROQ
# ─────────────────────────────────────────────

def _call_groq(messages: list, system_prompt: str, temperature: float, max_tokens: int, model: Optional[str] = None) -> str:
    try:
        from groq import Groq
        from config.config import GROQ_API_KEY, GROQ_MODEL

        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")

        client = Groq(api_key=GROQ_API_KEY)

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = client.chat.completions.create(
            model=model or GROQ_MODEL,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        raise


# ─────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────

def _call_gemini(messages: list, system_prompt: str, temperature: float, max_tokens: int, model: Optional[str] = None) -> str:
    try:
        import google.generativeai as genai
        from config.config import GEMINI_API_KEY, GEMINI_MODEL

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")

        genai.configure(api_key=GEMINI_API_KEY)

        gemini_model = genai.GenerativeModel(
            model_name=model or GEMINI_MODEL,
            system_instruction=system_prompt if system_prompt else None,
        )

        # Convert to Gemini format
        gemini_history = []
        for msg in messages[:-1]:  # all but last
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        chat = gemini_model.start_chat(history=gemini_history)
        response = chat.send_message(
            messages[-1]["content"],
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise


def get_available_providers() -> list:
    """Return list of providers that have API keys configured."""
    from config.config import OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY
    available = []
    if OPENAI_API_KEY:
        available.append("openai")
    if GROQ_API_KEY:
        available.append("groq")
    if GEMINI_API_KEY:
        available.append("gemini")
    return available if available else ["groq"]  # default fallback
