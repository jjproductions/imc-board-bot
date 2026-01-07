from fastapi import APIRouter, HTTPException
from typing import List, Optional
from app import crud, schemas
from ..settings import settings
from ..schemas import IngestResponse, DoclingDocument
from ..utilities import embed_texts, chunk_blocks, batched
from ..deps import get_embedder, get_tokenizer
import uuid
from qdrant_client.models import PointStruct as QdrantPointStruct

router = APIRouter()

# qdrant-client may not be available in all environments (editor/CI); import at runtime
# and surface a clear error message if missing.
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
except Exception as e:
    QdrantClient = None
    Distance = None
    VectorParams = None
    QdrantPointStruct = None
    Filter = None
    FieldCondition = None
    MatchValue = None
    _qdrant_import_err = e

if QdrantClient is None:
    raise RuntimeError(
        "Missing dependency 'qdrant-client'. Install with: pip install qdrant-client and ensure it's available in your environment."
    ) from _qdrant_import_err
qdrant = QdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)

# Note: models/tokenizer are initialized lazily via `app.deps.get_*`.
# Avoid initializing heavy models at import time to prevent circular imports
# and to make tests/imports lightweight. Endpoints call `embed_texts` and
# `chunk_blocks`, which will initialize the resources when required.

def ensure_collection(collection_name: str, vector_dim: int):
    """
    Ensure the named Qdrant collection exists; create it with the provided vector_dim
    and cosine distance if it's missing.
    """
    try:
        existing = qdrant.get_collections().collections
    except Exception as e:
        raise RuntimeError(f"Failed to list qdrant collections: {e}")

    if any(col.name == collection_name for col in existing):
        # existing = qdrant.get_collection(collection_name)
        # If size mismatch, recreate for safety
        # size = existing.vectors_count or vector_dim
        # if hasattr(existing.config, "params") and existing.config.params and existing.config.params.vectors and existing.config.params.vectors.size != vector_dim:
        return

    try:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create qdrant collection '{collection_name}': {e}")


# CRUD Endpoints for Chunks
@router.get("/chunks", response_model=List[schemas.Chunk])
def list_chunks():
    return crud.list_chunks()


@router.post("/chunks", response_model=schemas.Chunk, status_code=201)
def create_chunk(chunk: schemas.ChunkCreate):
    return crud.create_chunk(chunk)


@router.get("/chunks/{chunk_id}", response_model=schemas.Chunk)
def get_chunk(chunk_id: int):
    chunk = crud.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="chunk not found")
    return chunk


@router.put("/chunks/{chunk_id}", response_model=schemas.Chunk)
def update_chunk(chunk_id: int, chunk_in: schemas.ChunkCreate):
    chunk = crud.update_chunk(chunk_id, chunk_in)
    if not chunk:
        raise HTTPException(status_code=404, detail="chunk not found")
    return chunk


@router.delete("/chunks/{chunk_id}", status_code=204)
def delete_chunk(chunk_id: int):
    removed = crud.delete_chunk(chunk_id)
    if not removed:
        raise HTTPException(status_code=404, detail="chunk not found")
    return None


@router.get("/health")
def health():
    return {
        "status": "ok",
        "qdrant_url": settings.qdrant.url,
        "collection_default": settings.qdrant.default_collection,
        "embedding_model": settings.vector.embedding_model,
        "embedding_dim": settings.vector.embedding_dim,
    }


@router.post("/create-collection")
def create_collection(collection_name: str, vector_dim: Optional[int] = None):
    ensure_collection(collection_name, vector_dim or settings.vector.embedding_dim)
    return {"collection": collection_name, "vector_dim": vector_dim or settings.vector.embedding_dim}


@router.post("/ingest", response_model=IngestResponse)
def ingest_doc(
    doc: DoclingDocument,
    collection: Optional[str] = None,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
    batch_size: int = 256
):
    """
    Ingest a Docling-parsed document:
    - chunks with heading-aware context and overlap
    - embeds via BAAI/bge-m3 (1024-dim, normalized)
    - upserts into Qdrant with rich payloads
    """
    coll = collection or settings.qdrant.default_collection
    ensure_collection(coll, settings.vector.embedding_dim)

    chunks = chunk_blocks(doc.blocks, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from document.")

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    points = []
    for idx, (c, vec) in enumerate(zip(chunks, vectors)):
        chunk_id = str(uuid.uuid4())
        payload = {
            "source_id": doc.source_id,
            "title": doc.title,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "text": c["text"],
            "tokens": c["tokens"],
            "pages": c["pages"],
            "section_path": c["context_path"],
            "block_ids": c["block_ids"],
            "overlap_from_previous": c["overlap_from_previous"],
            "embedding_model": settings.vector.embedding_model_name,
        }
        points.append(QdrantPointStruct(id=chunk_id, vector=vec, payload=payload))

    upserted = 0
    for batch in batched(points, batch_size):
        qdrant.upsert(collection_name=coll, points=batch)
        upserted += len(batch)

    return IngestResponse(
        collection=coll,
        chunks_inserted=len(chunks),
        points_upserted=upserted,
        source_id=doc.source_id
    )