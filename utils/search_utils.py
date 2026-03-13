"""
utils/search_utils.py - Live web search integration using Tavily API.
Falls back to DuckDuckGo (no API key needed) if Tavily is unavailable.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> Optional[str]:
    """
    Perform a live web search and return summarized results.

    Tries Tavily first, then falls back to DuckDuckGo.

    Args:
        query: The search query string.
        max_results: Number of results to retrieve.

    Returns:
        A formatted string of search results, or None on failure.
    """
    try:
        from config.config import TAVILY_API_KEY
        if TAVILY_API_KEY:
            return _tavily_search(query, max_results, TAVILY_API_KEY)
        else:
            logger.info("Tavily key not set, trying DuckDuckGo fallback.")
            return _duckduckgo_search(query, max_results)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return None


def _tavily_search(query: str, max_results: int, api_key: str) -> str:
    """Search using Tavily API."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
        )
        results = response.get("results", [])
        return _format_results(results, source="Tavily")
    except ImportError:
        logger.warning("tavily-python not installed. Falling back to DuckDuckGo.")
        return _duckduckgo_search(query, max_results)
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return _duckduckgo_search(query, max_results)


def _duckduckgo_search(query: str, max_results: int) -> str:
    """Search using DuckDuckGo (no API key needed)."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "content": r.get("body", ""),
                    "url": r.get("href", ""),
                })
        return _format_results(results, source="DuckDuckGo")
    except ImportError:
        logger.error("duckduckgo_search not installed.")
        return "⚠️ Web search unavailable. Please install `duckduckgo-search` or set TAVILY_API_KEY."
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
        return f"⚠️ Web search encountered an error: {str(e)}"


def _format_results(results: List[Dict], source: str = "") -> str:
    """Format search results into a readable string."""
    if not results:
        return "No search results found."

    parts = [f"🌐 **Web Search Results** ({source}):\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        content = r.get("content", r.get("body", "No content"))
        url = r.get("url", r.get("href", ""))

        # Truncate long content
        if len(content) > 400:
            content = content[:400] + "..."

        parts.append(f"**{i}. {title}**\n{content}\n🔗 {url}\n")

    return "\n".join(parts)


def should_search_web(query: str, context: str = "") -> bool:
    """
    Heuristic: decide if a web search would improve the response.

    Returns True if query seems to need fresh/real-time information.
    """
    web_keywords = [
        "latest", "recent", "current", "today", "news", "update",
        "2024", "2025", "now", "new", "just", "released", "announced",
    ]
    q_lower = query.lower()
    return any(kw in q_lower for kw in web_keywords) or not context
