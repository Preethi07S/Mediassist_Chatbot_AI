"""
utils/rag_utils.py - RAG pipeline: document loading, chunking,
vector store creation, and similarity retrieval.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TEXT CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split a long text into overlapping chunks.

    Args:
        text: Raw text to split.
        chunk_size: Max characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    try:
        chunks = []
        start = 0
        text = text.strip()
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - overlap
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    except Exception as e:
        logger.error(f"Chunking error: {e}")
        raise


# ─────────────────────────────────────────────
# DOCUMENT LOADING
# ─────────────────────────────────────────────

def load_document(file_obj) -> str:
    """
    Load text from an uploaded file object (PDF, TXT, MD).

    Args:
        file_obj: A Streamlit UploadedFile object.

    Returns:
        Extracted text as string.
    """
    try:
        filename = file_obj.name.lower()

        if filename.endswith(".txt") or filename.endswith(".md"):
            return file_obj.read().decode("utf-8", errors="ignore")

        elif filename.endswith(".pdf"):
            try:
                import pdfplumber
                import io
                text_parts = []
                with pdfplumber.open(io.BytesIO(file_obj.read())) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                return "\n\n".join(text_parts)
            except ImportError:
                # Fallback: pypdf
                import pypdf
                import io
                reader = pypdf.PdfReader(io.BytesIO(file_obj.read()))
                return "\n\n".join(
                    page.extract_text() for page in reader.pages if page.extract_text()
                )

        else:
            raise ValueError(f"Unsupported file type: {file_obj.name}")

    except Exception as e:
        logger.error(f"Document loading error: {e}")
        raise


# ─────────────────────────────────────────────
# VECTOR STORE (In-memory + optional JSON persistence)
# ─────────────────────────────────────────────

class SimpleVectorStore:
    """
    Lightweight in-memory vector store using cosine similarity.
    Supports optional JSON-based persistence.
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict] = []

    def add_documents(self, chunks: List[str], embeddings: List[List[float]], source: str = ""):
        """Add document chunks and their embeddings."""
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            self.chunks.append(chunk)
            self.embeddings.append(emb)
            self.metadata.append({"source": source, "chunk_index": i})
        logger.info(f"Added {len(chunks)} chunks from source: '{source}'")

    def similarity_search(self, query_embedding: List[float], top_k: int = 4) -> List[Tuple[str, float, Dict]]:
        """
        Find top-k most similar chunks using cosine similarity.

        Returns:
            List of (chunk_text, score, metadata) tuples.
        """
        try:
            if not self.embeddings:
                return []

            q = np.array(query_embedding)
            scores = []
            for i, emb in enumerate(self.embeddings):
                e = np.array(emb)
                # Cosine similarity
                denom = np.linalg.norm(q) * np.linalg.norm(e)
                score = float(np.dot(q, e) / denom) if denom > 0 else 0.0
                scores.append((i, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            top = scores[:top_k]

            return [(self.chunks[i], score, self.metadata[i]) for i, score in top]
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            raise

    def save(self, path: str):
        """Persist vector store to disk as JSON."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                "chunks": self.chunks,
                "embeddings": self.embeddings,
                "metadata": self.metadata,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.info(f"Vector store saved to {path}")
        except Exception as e:
            logger.error(f"Save error: {e}")
            raise

    def load(self, path: str):
        """Load vector store from disk."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data["chunks"]
            self.embeddings = data["embeddings"]
            self.metadata = data["metadata"]
            logger.info(f"Vector store loaded from {path} ({len(self.chunks)} chunks)")
        except Exception as e:
            logger.error(f"Load error: {e}")
            raise

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def count(self) -> int:
        return len(self.chunks)


# ─────────────────────────────────────────────
# HIGH-LEVEL RAG PIPELINE
# ─────────────────────────────────────────────

def build_vector_store_from_file(file_obj, store: SimpleVectorStore) -> int:
    """
    Full pipeline: load file → chunk → embed → add to store.

    Returns:
        Number of chunks added.
    """
    try:
        from models.embeddings import embed_texts
        from config.config import CHUNK_SIZE, CHUNK_OVERLAP

        text = load_document(file_obj)
        if not text.strip():
            raise ValueError("Document appears to be empty or unreadable.")

        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        embeddings = embed_texts(chunks)
        store.add_documents(chunks, embeddings, source=file_obj.name)
        return len(chunks)
    except Exception as e:
        logger.error(f"Vector store build error: {e}")
        raise


def retrieve_context(query: str, store: SimpleVectorStore, top_k: int = 4) -> str:
    """
    Retrieve relevant context chunks for a query.

    Returns:
        Concatenated context string for prompt injection.
    """
    try:
        from models.embeddings import embed_query

        if store.is_empty():
            return ""

        q_emb = embed_query(query)
        results = store.similarity_search(q_emb, top_k=top_k)

        if not results:
            return ""

        context_parts = []
        for chunk, score, meta in results:
            source = meta.get("source", "document")
            context_parts.append(f"[Source: {source}]\n{chunk}")

        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        return ""
