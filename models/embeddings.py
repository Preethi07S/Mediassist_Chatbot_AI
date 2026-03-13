"""
models/embeddings.py - Embedding model abstraction for RAG.
Uses sentence-transformers (local, free, no API key needed).
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

_embedding_model = None  # Singleton cache


def get_embedding_model():
    """
    Lazy-load and cache the sentence-transformer embedding model.
    Returns the model instance.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            from config.config import EMBEDDING_MODEL

            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (list of floats).
    """
    try:
        model = get_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


def embed_query(query: str) -> List[float]:
    """
    Generate an embedding for a single query string.

    Args:
        query: The user query string.

    Returns:
        Embedding vector as list of floats.
    """
    try:
        model = get_embedding_model()
        embedding = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        return embedding[0].tolist()
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        raise
