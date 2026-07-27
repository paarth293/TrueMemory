"""TrueMemory Vector Search Engine (Dense Embeddings & Vector Similarity)."""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector Math Utilities
# ---------------------------------------------------------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate the cosine similarity between two 1D numerical vector arrays.

    Returns a float score between 0.0 and 1.0, where 1.0 means identical meaning.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    dot = np.dot(vec1, vec2)
    similarity = float(dot / (norm1 * norm2))

    # Clamp bounds to handle floating-point precision inaccuracies
    return max(0.0, min(1.0, similarity))


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length (L2 norm = 1.0)."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


# ---------------------------------------------------------------------------
# Embedding Generator Helpers
# ---------------------------------------------------------------------------

def _fallback_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Generate a deterministic pseudo-embedding using SHA-256 byte hashes.

    Ensures the vector engine never crashes when ML model weights are absent
    or still loading.
    """
    if not text or not text.strip():
        return np.zeros(dim, dtype=np.float32)

    # Generate SHA-256 hash bytes from input text string
    hash_digest = hashlib.sha256(text.encode("utf-8")).digest()

    # Repeat bytes to match requested vector dimension
    repeated = (hash_digest * ((dim // len(hash_digest)) + 1))[:dim]
    raw_array = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)

    # Convert to range [-1.0, 1.0] and normalize to unit length
    centered = (raw_array - 127.5) / 127.5
    return normalize_vector(centered)


def get_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Convert text into a 1D normalized vector embedding array.

    Attempts to use the lightweight model2vec Edge model first.
    Falls back gracefully to deterministic SHA-256 pseudo-embedding if
    model weights are not installed.

    Args:
        text: Input string to embed.
        dim:  Vector output dimension size (default 256).

    Returns:
        1D float32 numpy array normalized to unit length.
    """
    if not text or not text.strip():
        return np.zeros(dim, dtype=np.float32)

    try:
        # Attempt loading fast model2vec if installed
        from model2vec import StaticModel
        model = StaticModel.from_pretrained("minishlab/potion-base-8M")
        vec = model.encode(text)
        return normalize_vector(np.array(vec, dtype=np.float32))
    except Exception:
        # Fallback to deterministic pseudo-embedding
        log.debug("model2vec not available, using fallback SHA-256 embedding.")
        return _fallback_embedding(text, dim=dim)