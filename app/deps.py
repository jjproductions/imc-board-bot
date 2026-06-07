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
_sparse_embedder: Optional[object] = None
_tokenizer: Optional[object] = None
_init_error: Optional[Exception] = None


class FastTokenizerWrapper:
    def __init__(self, tokenizer_obj):
        self._tokenizer = tokenizer_obj

    def encode(self, text: str, add_special_tokens: bool = True):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True):
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def init_models() -> None:
    """Initialize embedding model and tokenizer once.

    Raises the underlying exception if initialization fails.
    """
    global _embedder, _sparse_embedder, _tokenizer, _init_error
    with _lock:
        if _embedder is not None and _tokenizer is not None and _sparse_embedder is not None:
            return
        try:
            from fastembed import TextEmbedding, SparseTextEmbedding
            from tokenizers import Tokenizer

            _embedder = TextEmbedding(
                model_name=str(settings.vector.embedding_model),
                cache_dir=str(settings.models_dir)
            )
            # Attempt to find and load local cached tokenizer.json to avoid runtime network calls
            model_name_clean = str(settings.vector.embedding_model).split("/")[-1]
            glob_pattern = f"**/models--*{model_name_clean}*/snapshots/*/tokenizer.json"
            tokenizer_files = list(settings.models_dir.glob(glob_pattern))
            
            if tokenizer_files:
                raw_tokenizer = Tokenizer.from_file(str(tokenizer_files[0]))
            else:
                raw_tokenizer = Tokenizer.from_pretrained(str(settings.vector.embedding_model))
                
            _tokenizer = FastTokenizerWrapper(raw_tokenizer)
            _sparse_embedder = SparseTextEmbedding(
                model_name=str(settings.vector.sparse_embedding_model),
                cache_dir=str(settings.models_dir)
            )
            _init_error = None
        except Exception as e:
            _embedder = None
            _sparse_embedder = None
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


def get_sparse_embedder() -> object:
    """Return the sparse embedder, initializing it if necessary."""
    if _sparse_embedder is None:
        init_models()
    if _sparse_embedder is None:
        raise RuntimeError(f"Sparse embedder not initialized: {_init_error}")
    return _sparse_embedder
