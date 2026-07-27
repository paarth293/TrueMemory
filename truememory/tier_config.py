"""Centralized tier configuration -- single source of truth for TrueMemory tiers.

Every tier-aware module (vector_search, reranker, model_server) imports from here
instead of maintaining its own mapping dicts.
"""

from __future__ import annotations

import copy
import logging
import os
import re

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in tier definitions
# ---------------------------------------------------------------------------

TIERS: dict[str, dict] = {
    "edge": {
        "embed_model": "model2vec",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "embed_dim": 256,
        "tier_group": "edge",
        "model_name": "potion-base-8M",
    },
    "base": {
        "embed_model": "qwen3_256",
        "reranker": "Alibaba-NLP/gte-reranker-modernbert-base",
        "embed_dim": 256,
        "tier_group": "basepro",
        "model_name": "Qwen3-Embedding-0.6B",
    },
    "pro": {
        "embed_model": "qwen3_256",
        "reranker": "Alibaba-NLP/gte-reranker-modernbert-base",
        "embed_dim": 256,
        "tier_group": "basepro",
        "model_name": "Qwen3-Embedding-0.6B",
    },
}

MODEL_DIMS: dict[str, int] = {
    "model2vec": 256,
    "minilm": 384,
    "bge-small": 384,
    "qwen3_256": 256,
}

VALID_TIER_GROUPS: frozenset[str] = frozenset({"edge", "basepro", "custom"})

MODEL_TO_GROUP: dict[str, str] = {
    "model2vec": "edge",
    "qwen3_256": "basepro",
    "minilm": "basepro",
    "bge-small": "basepro",
}

# Regex for validating HuggingFace model IDs
_HF_MODEL_ID_RE = re.compile(r"^[\w][\w.\-]*(\/[\w][\w.\-]*)?$")

# ---------------------------------------------------------------------------
# Custom tier resolution
# ---------------------------------------------------------------------------

def resolve_custom_tier() -> dict:
    """Build custom tier config from ``TRUEMEMORY_CUSTOM_*`` env vars."""
    if os.environ.get("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "").strip() != "1":
        raise ValueError(
            "Custom tier requires TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD=1 "
            "to acknowledge arbitrary model downloads."
        )

    embed = os.environ.get("TRUEMEMORY_CUSTOM_EMBED_MODEL", "").strip()
    if not embed:
        raise ValueError("TRUEMEMORY_CUSTOM_EMBED_MODEL must be set for custom tier")

    if not _HF_MODEL_ID_RE.fullmatch(embed):
        raise ValueError(f"Invalid model ID format: {embed!r}.")

    reranker = os.environ.get(
        "TRUEMEMORY_CUSTOM_RERANKER",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ).strip()
    if reranker and not _HF_MODEL_ID_RE.fullmatch(reranker):
        raise ValueError(f"Invalid reranker model ID format: {reranker!r}.")

    raw_dim = os.environ.get("TRUEMEMORY_CUSTOM_EMBED_DIM", "256").strip()
    try:
        dim = int(raw_dim)
    except (ValueError, TypeError):
        raise ValueError(f"TRUEMEMORY_CUSTOM_EMBED_DIM must be an integer, got {raw_dim!r}")

    if dim < 1 or dim > 4096:
        raise ValueError(f"TRUEMEMORY_CUSTOM_EMBED_DIM must be 1-4096, got {dim}")

    return {
        "embed_model": embed,
        "reranker": reranker or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "embed_dim": dim,
        "tier_group": "custom",
        "model_name": embed,
    }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tier_config(tier: str) -> dict:
    """Return a copy of the config dict for the given tier (edge, base, pro, custom)."""
    t = tier.lower().strip()
    if t == "custom":
        return resolve_custom_tier()
    if t not in TIERS:
        raise ValueError(f"Unknown tier: {tier!r}. Valid tiers: {sorted(TIERS)} + ['custom']")
    return copy.copy(TIERS[t])


def get_embed_model(tier: str) -> str:
    """Return the embedding model ID for a tier."""
    return get_tier_config(tier)["embed_model"]


def get_reranker(tier: str) -> str:
    """Return the reranker HF model ID for a tier."""
    return get_tier_config(tier)["reranker"]


def get_embed_dim(tier: str) -> int:
    """Return the embedding dimension for a tier."""
    return get_tier_config(tier)["embed_dim"]


def get_tier_group(tier: str) -> str:
    """Return the tier group for a tier (edge / basepro / custom)."""
    return get_tier_config(tier)["tier_group"]