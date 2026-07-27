"""TrueMemory Vector Search Engine (Dense Embeddings & Vector Similarity)."""

from __future__ import annotations

import logging
import numpy as np
from typing import Any

log = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate the cosine similarity between two 1D numerical vector arrays.

    Returns a float score between 0.0 and 1.0, where 1.0 means identical angle/meaning.
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