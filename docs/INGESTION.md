# Data Ingestion Guide

This document describes the hybrid ingestion pipeline — how PDFs and Docling JSON documents are processed, chunked, embedded, and stored in Qdrant for downstream RAG retrieval.

---

## Overview

The ingestion pipeline produces **both dense and sparse vectors** for each chunk and stores them as named vectors in Qdrant. This supports hybrid search (Reciprocal Rank Fusion) in the retrieval layer.

```
PDF / Docling JSON
      │
      ▼
  Docling Parser  ──► structured blocks
      │
      ▼
  Chunker (token-aware, heading-aware)
      │
      ▼
  Embedder
  ├── Dense  (sentence-transformers / BGE-M3)  ──► float32 vector
  └── Sparse (SPLADE via fastembed)             ──► sparse indices + values
      │
      ▼
  Qdrant upsert (named vectors: "dense" + "sparse")
```

---

## Qdrant Collection Schema

The collection must be created with **both** `vectors_config` (dense) and `sparse_vectors_config` (sparse) before ingesting:

```python
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

client.create_collection(
    collection_name="board-policies_chunks",
    vectors_config={
        "dense": VectorParams(
            size=1024,           # Match your dense model output dimension
            distance=Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams()
    }
)
```

Or via the API endpoint:

```bash
curl -X POST "http://localhost:8005/api/create-collection?collection_name=board-policies_chunks&vector_dim=1024"
```

---

## Embedding Models

| Type | Library | Model |
|---|---|---|
| Dense | `sentence-transformers` | `BAAI/bge-m3` (1024-dim) |
| Sparse | `fastembed` | `prithivida/Splade_PP_en_v1` |

Download models locally before running (avoids repeated HuggingFace pulls):

```bash
python scripts/download_models.py

# With explicit options
python scripts/download_models.py --models-dir ./models --hf-revision main --retries 3
```

---

## Ingesting Documents

### Option A — Upload a PDF directly

```bash
curl -X POST "http://localhost:8005/api/ingest-file" \
  -F "file=@/path/to/policy.pdf" \
  -F "collection_name=board-policies_chunks"
```

The API will parse the PDF via Docling, chunk it, embed it, and upsert to Qdrant.

### Option B — Ingest pre-processed Docling JSON

```bash
curl -X POST "http://localhost:8005/api/ingest" \
  -H "Content-Type: application/json" \
  -d @data/docling_converted_output_20260505T025458Z.json
```

---

## Point Structure in Qdrant

Each upserted point contains:

```python
{
    "id": "<uuid5 derived from content hash>",
    "vector": {
        "dense": [...],          # float32 list, length 1024
        "sparse": SparseVector(
            indices=[...],       # token indices
            values=[...]         # SPLADE weights
        )
    },
    "payload": {
        "text": "...",
        "section_path": "...",
        "block_ids": [...],
        "token_count": 123,
        "hash": "...",
        "source_doc": "..."
    }
}
```

---

## Chunking Behavior

Chunking is **token-aware and heading-aware** (see `app/utilities.py`):

- Respects heading boundaries — chunks do not cross section headings
- Token window size is configurable via `app/settings.py`
- Each chunk preserves `section_path`, `block_ids`, and token counts in its payload

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|---|---|---|
| `Collection not found` on ingest | Collection not created yet | Run `create-collection` endpoint first |
| Model download timeout | Slow HuggingFace connection | Run `download_models.py` with `--retries 5` |
| Qdrant connection timeout | Wrong host/port in env | Check `QDRANT_HOST` / `QDRANT_PORT` in `.env` |
| Dense dim mismatch | Model output ≠ collection `vector_dim` | Recreate collection with correct `vector_dim` |
