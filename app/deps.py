"""Shared runtime dependencies (embedder, tokenizer, etc.).

This module centralizes initialization so other modules can import
light-weight getters without triggering heavy model loading at import time.
Provide `init_models()`, `get_embedder()` and `get_tokenizer()`.
"""
from importlib.resources import path
from threading import Lock
from typing import Optional

from .settings import settings

_lock = Lock()
_embedder: Optional[object] = None
_tokenizer: Optional[object] = None
_init_error: Optional[Exception] = None


def init_models() -> None:
    """Initialize embedding model and tokenizer once.

    Raises the underlying exception if initialization fails.
    """
    global _embedder, _tokenizer, _init_error
    with _lock:
        if _embedder is not None and _tokenizer is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(
                str(settings.vector.embedding_model), 
                local_files_only=True, 
                model_kwargs={"use_safetensors": False}
            )
            _tokenizer = _embedder.tokenizer
            _init_error = None
        except Exception as e:
            _embedder = None
            _tokenizer = None
            _init_error = e
            raise


def get_embedder() -> object:
    """Return the embedder, initializing it if necessary."""
    if _embedder is None:
        init_models()
    if _embedder is None:
        # Re-raise stored error for clearer messaging
        raise RuntimeError(f"Embedder not initialized: {_init_error}")
    return _embedder


def get_tokenizer() -> object:
    """Return the tokenizer, initializing it if necessary."""
    if _tokenizer is None:
        init_models()
    if _tokenizer is None:
        raise RuntimeError(f"Tokenizer not initialized: {_init_error}")
    return _tokenizer
