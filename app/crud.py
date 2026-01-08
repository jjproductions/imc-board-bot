from typing import Dict, Optional, List
from app.schemas import Chunk, ChunkCreate

_store: Dict[int, Chunk] = {}
_next_id = 1


def list_chunks() -> List[Chunk]:
    return list(_store.values())


def create_chunk(chunk_in: ChunkCreate) -> Chunk:
    global _next_id
    # pydantic v2: use model_dump() instead of dict()
    chunk = Chunk(id=_next_id, **chunk_in.model_dump())
    _store[_next_id] = chunk
    _next_id += 1
    return chunk


def get_chunk(chunk_id: int) -> Optional[Chunk]:
    return _store.get(chunk_id)


def update_chunk(chunk_id: int, chunk_in: ChunkCreate) -> Optional[Chunk]:
    if chunk_id in _store:
        # pydantic v2: use model_dump() instead of dict()
        chunk = Chunk(id=chunk_id, **chunk_in.model_dump())
        _store[chunk_id] = chunk
        return chunk
    return None


def delete_chunk(chunk_id: int) -> bool:
    return _store.pop(chunk_id, None) is not None
