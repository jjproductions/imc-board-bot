# deps.py
"""Shared runtime dependencies (tokenizer, and local model instantiation).

This module centralizes initialization so other modules can import
light-weight getters without triggering heavy model loading at import time.
Provide `init_models()`, `get_embedder()`, `get_sparse_embedder()`, and `get_tokenizer()`.
"""
from threading import Lock
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .settings import settings

_lock = Lock()
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
    """Initialize tokenizer once. Heavy embedding models are loaded dynamically/on-demand
    to keep peak memory utilization extremely low.
    """
    global _tokenizer, _init_error
    with _lock:
        if _tokenizer is not None:
            return
        try:
            from tokenizers import Tokenizer

            # Attempt to find and load local cached tokenizer.json to avoid runtime network calls
            model_name_clean = str(settings.vector.embedding_model).split("/")[-1]
            glob_pattern = f"**/models--*{model_name_clean}*/snapshots/*/tokenizer.json"
            tokenizer_files = list(settings.models_dir.glob(glob_pattern))
            
            if tokenizer_files:
                raw_tokenizer = Tokenizer.from_file(str(tokenizer_files[0]))
            else:
                raw_tokenizer = Tokenizer.from_pretrained(str(settings.vector.embedding_model))
                
            _tokenizer = FastTokenizerWrapper(raw_tokenizer)
            _init_error = None
        except Exception as e:
            _tokenizer = None
            _init_error = e
            raise


def get_tokenizer() -> object:
    """Return the tokenizer, initializing it if necessary."""
    if _tokenizer is None:
        init_models()
    if _tokenizer is None:
        raise RuntimeError(f"Tokenizer not initialized: {_init_error}")
    return _tokenizer


def get_embedder() -> object:
    """Return a fresh instance of the dense embedding model configured with 2 threads."""
    from fastembed import TextEmbedding
    return TextEmbedding(
        model_name=str(settings.vector.embedding_model),
        cache_dir=str(settings.models_dir),
        threads=2
    )


def get_sparse_embedder() -> object:
    """Return a fresh instance of the sparse embedding model configured with 2 threads."""
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(
        model_name=str(settings.vector.sparse_embedding_model),
        cache_dir=str(settings.models_dir),
        threads=2
    )


security = HTTPBearer()


def require_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    """Validate incoming API requests against settings.api_key."""
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


