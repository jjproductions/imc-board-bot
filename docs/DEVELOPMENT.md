# Development Guide

Local setup, debugging, and testing for the Board Policy Bot.

---

## Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for running Qdrant locally)
- `zsh` / macOS

---

## Local Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.development .env
# Edit .env with your local values
```

### 4. Download embedding models

```bash
python scripts/download_models.py
# Models are saved to ./models/bge-m3 and ./models/docling by default
```

### 5. Start Qdrant (via Docker)

```bash
docker compose up qdrant -d
```

### 6. Start the API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --log-level debug
```

Open the interactive API docs at: `http://127.0.0.1:8000/docs`

---

## VS Code Debugging (debugpy attach)

The app starts a non-blocking `debugpy` listener on port `5678` in development mode (if `debugpy` is installed).

### Install debugpy

```bash
pip install debugpy
```

### Attach config — `.vscode/launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to FastAPI (debugpy)",
      "type": "python",
      "request": "attach",
      "connect": { "host": "localhost", "port": 5678 },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "${workspaceFolder}"
        }
      ],
      "subProcess": true
    }
  ]
}
```

> [!NOTE]
> Use `subProcess: true` to attach to the reloader worker process when running with `--reload`.

> [!CAUTION]
> Never leave `debugpy.wait_for_client()` or a debug listener enabled in production builds.

---

## Running Tests

```bash
pytest -q
```

The test suite uses lightweight imports to avoid loading heavy ML models. Full integration tests (requiring a live Qdrant instance and models) should be run in a CI environment with all dependencies available.

---

## Project Structure

```
board policy bot/
├── app/
│   ├── main.py          # App factory, lifespan startup (model init, debug listener)
│   ├── routes/api.py    # API router — /api/health, /api/ingest, /api/chunks, etc.
│   ├── deps.py          # Lazy getters for heavy runtime deps (embedder, tokenizer)
│   ├── utilities.py     # Chunking, token-aware helpers, embedding wrapper
│   ├── crud.py          # Simple in-memory chunk store
│   ├── schemas.py       # Pydantic request/response models
│   └── settings.py      # App configuration (loaded from env vars)
├── docs/                # Long-form documentation
├── prompts/             # LLM prompt templates
├── scripts/             # Utility scripts (model download, etc.)
├── tests/               # pytest test cases
├── data/                # Docling output JSON, local Qdrant storage
└── models/              # Downloaded embedding models (git-ignored)
```

---

## Design Notes

- **Heavy ML resources** (dense embedding model, sparse SPLADE model + tokenizer) are initialized via `app.deps.init_models()` during application lifespan startup — the app fails fast on misconfiguration.
- **Lazy getters** (`get_embedder()`, `get_sparse_embedder()`, `get_tokenizer()`) in `app/deps.py` are used by `utilities.py` to avoid circular imports and keep module imports light.
- **Chunking** is token-aware and heading-aware (see `app/utilities.py`), preserving `section_path`, `block_ids`, and token counts per chunk.

---

## Should I commit `.vscode/launch.json`?

If the launch config uses only `${workspaceFolder}` path mappings and no secrets, it's safe and useful to commit. If it contains local absolute paths or personal preferences, keep it out of VCS and provide a `launch.json.example` instead.
