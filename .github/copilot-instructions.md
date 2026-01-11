
# Copilot Project Instructions & Preferences

> Owner: Jonathan Bagby  
> Context: Docling → FastAPI → Qdrant RAG stack

This document captures architectural and coding preferences for building and operating a retrieval pipeline using **FastAPI**, **Docling**-parsed documents, **Qdrant**, and **BAAI/bge-m3** embeddings. It is intended for consistent implementation across services, PRs, and automation.

---

## 1) Core Philosophy

- **Separate concerns**: Maintain **two services**—one for **ingestion** and one for **query/chat**. Scale, secure, and deploy them independently.
- **Clear contracts**: Use Pydantic models per service; keep schemas versionable and documented.
- **Reliability first**: The query/chat service must remain online even if ingestion evolves or fails.
- **Observability & security**: Add auth for ingestion; monitor latencies and errors for search/chat.

---

## 2) Tech Stack Defaults

- **Language**: Python 3.11+
- **Framework**: FastAPI + Uvicorn
- **Vector DB**: Qdrant (HTTP API)
- **Embeddings**: `BAAI/bge-m3` (1024-dim), normalized embeddings, cosine similarity
- **Tokenization (chunking)**: `transformers` tokenizer for `bge-m3`
- **Runtime packaging**: Docker per service; optional `docker-compose` to wire Qdrant + services

---

## 3) High-Level Architecture

```
Docling (JSON blocks) ──> Ingestion Service (chunk + embed bge-m3 + upsert) ──> Qdrant
                                                     ^
                                                     |
                                  Query/Chat Service (search, optional rerank)
```

**Why this split**
- Independent scaling (batch ingestion vs. low-latency query)
- Resilience (query unaffected by ingestion changes)
- Tighter security (ingestion behind VPN/API key)
- Cleaner deployments & simpler rollback

---

## 4) Repository Structure

Two services, each with routers under `Routes/` and models under `models/`:

```
docling-qdrant/
├─ ingestion/
│  ├─ main.py
│  ├─ Routes/
│  │  ├─ __init__.py
│  │  └─ ingestion.py            # /health, /create-collection, /ingest
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ docling.py              # DoclingBlock, DoclingDocument
│  │  └─ responses.py            # IngestResponse
│  ├─ core/
│  │  ├─ config.py               # env defaults (Qdrant URL, embedding model)
│  │  └─ deps.py                 # Qdrant client, embedder, tokenizer singletons
│  ├─ requirements.txt
│  └─ Dockerfile
└─ query/
   ├─ main.py
   ├─ Routes/
   │  ├─ __init__.py
   │  └─ query.py                # /health, /search, /chat
   ├─ models/
   │  ├─ __init__.py
   │  ├─ search.py               # SearchRequest, SearchHit, SearchResponse
   │  └─ chat.py                 # ChatMessage, ChatRequest
   ├─ core/
   │  ├─ config.py
   │  └─ deps.py                 # Qdrant client, embedder
   ├─ requirements.txt
   └─ Dockerfile
```

---

## 5) API Contracts

### Ingestion Service
- `GET /health` → basic health & model info
- `POST /create-collection?collection_name=...&vector_dim=...` → ensure/recreate Qdrant collection
- `POST /ingest` → body: `DoclingDocument`; query params: `collection`, `max_tokens`, `overlap_tokens`, `batch_size`  
  **Response**: `IngestResponse`

**DoclingDocument (summary)**
- `source_id: str` – unique document ID (path, SHA)
- `title: Optional[str]`
- `blocks: List[DoclingBlock]`

**DoclingBlock (summary)**
- `id: str`
- `type: Literal[heading|paragraph|list_item|table|figure|caption|footnote|page_break]`
- `text: Optional[str]`
- `page: Optional[int]`
- `level: Optional[int]` (for headings)
- `meta: Optional[Dict[str, Any]]` (e.g., table rows)

**IngestResponse**
- `collection: str`
- `chunks_inserted: int`
- `points_upserted: int`
- `source_id: str`

### Query/Chat Service
- `GET /health` → basic health & model info
- `POST /search` → body: `SearchRequest`; returns `SearchResponse`
- `POST /chat` → body: `ChatRequest`; returns `{answer, citations}` (LLM-free scaffold)

**SearchRequest / Response (summary)**
- `query: str`, `top_k: int = 5`, `collection?: str`, `filter_by_source_id?: str`
- `hits: List[{score: float, text: str, payload: Dict}]`

**ChatRequest (summary)**
- `messages: [{role: "user"|"assistant"|"system", content: str}]`
- `top_k: int = 5`, `collection?: str`, `source_filter?: str`

---

## 6) Embedding & Qdrant Configuration

**Embedding model defaults**
- Model: `BAAI/bge-m3` (multilingual)
- Dimension: `1024`
- Normalization: **on** (`normalize_embeddings=True`)
- Distance: **cosine** in Qdrant

**Environment variables**
```bash
# common
export QDRANT_URL=http://localhost:6333
export QDRANT_API_KEY=        # optional for local
export QDRANT_COLLECTION=docling_chunks

# embeddings (defaults)
export EMBEDDING_MODEL=BAAI/bge-m3
export EMBEDDING_DIM=1024
```

**Collection creation**
- If switching models/dim, **recreate** collection with matching `VectorParams(size=EMBEDDING_DIM, distance=cosine)`.

---

## 7) Chunking Strategy (Docling → Text Chunks)

- **Heading-aware context**: Maintain a stack of headings (`H1..H6`) and assign a `section_path` like `H1 > H2 > ...` to each chunk.
- **Token-aware windowing**: Use the model tokenizer; default `max_tokens=300` with `overlap_tokens=50` between windows.
- **Tables**: Convert tables (`meta.rows`) to **Markdown** if possible; otherwise, fall back to `text`.
- **Page boundaries**: Flush buffers on `page_break` to reduce cross-page context mixing.
- **Payload**: Store `source_id`, `title`, `pages`, `section_path`, `block_ids`, `tokens`, `overlap_from_previous`, `embedding_model`.

---

## 8) Deployment & Local Dev

**Dockerfile (per service)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml (example)**
```yaml
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  ingestion:
    build: ./ingestion
    environment:
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_COLLECTION=docling_chunks
      - EMBEDDING_MODEL=BAAI/bge-m3
      - EMBEDDING_DIM=1024
    depends_on:
      - qdrant
    ports:
      - "8001:8000"

  query:
    build: ./query
    environment:
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_COLLECTION=docling_chunks
      - EMBEDDING_MODEL=BAAI/bge-m3
      - EMBEDDING_DIM=1024
    depends_on:
      - qdrant
    ports:
      - "8002:8000"

volumes:
  qdrant_data:
```

**Local run**
```bash
# Terminal A
cd ingestion && uvicorn main:app --reload --port 8001

# Terminal B
cd query && uvicorn main:app --reload --port 8002
```

---

## 9) Security, Observability, and Ops

- **Auth**: Make ingestion private. Add API key / OAuth dependencies to `Routes/ingestion.py`.
- **Rate limiting**: Apply to query endpoints as needed.
- **Logging**: Log `source_id`, `chunk_index` on ingestion; log query latency, `top_k`, and document IDs on search/chat.
- **Metrics**: Export Prometheus counters/histograms for ingestion duration and query latency.
- **Schema evolution**: Include `payload_version` in Qdrant payloads; document changes in this file.

---

## 10) Optional Enhancements

- **Reranking**: Add a `/rerank` endpoint using a reranker model (e.g., `bge-reranker-v2-m3`) to reorder `top_k` hits.
- **Streaming ingestion**: Introduce a queue (Redis/Kafka) and worker for large volumes; keep API responsive.
- **Common package**: Extract shared models/utilities into `common/` and version it.
- **Citations & synthesis**: Integrate an LLM in the query service to produce answers with citations based on retrieved chunks.

---

## 11) Quick Reference Commands

```bash
# Create collection (1024-dim)
curl -X POST "http://localhost:8001/create-collection?collection_name=docling_chunks&vector_dim=1024"

# Ingest Docling JSON
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d @docling_sample.json

# Search
curl -X POST http://localhost:8002/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the dress code?","top_k":5,"collection":"docling_chunks","filter_by_source_id":"imc_handbook_v1"}'
```

---

## 12) Style & Code Organization

- Endpoints live under `Routes/` per service.
- Pydantic models live under `models/` per service.
- Configuration and dependency singletons live under `core/` (`config.py`, `deps.py`).
- Avoid cross-service imports unless moved to a shared `common/` package.
- Keep code blocks in Markdown fenced sections; avoid code in tables.

---

**This document is the source of truth for service layout and runtime defaults.**  
If preferences change (e.g., embedding model, vector dimension, or router structure), update this file and reference the change in PR descriptions.
