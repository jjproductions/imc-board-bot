# Board Policy Bot

A hybrid RAG (Retrieval-Augmented Generation) API for Invest Manitoba board governance documents. Staff can ask natural-language questions about bylaws and board policies via an OpenWebUI frontend backed by a FastAPI ingest/retrieval service and Qdrant vector store.

---

## Documentation

| Guide | Description |
|---|---|
| [Deployment](docs/DEPLOYMENT.md) | Local Docker and Azure Container Apps deployment |
| [GitHub Actions](docs/GITHUB.md) | Setup instructions to enable CI/CD to Azure |
| [Architecture](docs/ARCHITECTURE.md) | System design, hybrid search, component overview |
| [Ingestion](docs/INGESTION.md) | How to ingest PDFs and manage the Qdrant collection |
| [Development](docs/DEVELOPMENT.md) | Local setup, debugging, testing, project structure |

---

## Quick Start (local)

**Prerequisites:** Python 3.11+, Docker Desktop

```bash
# 1. Set up Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.development .env  # edit as needed

# 3. Download embedding models
python scripts/download_models.py

# 4. Start the full stack (API + Qdrant)
docker compose up --build
```

- **API + docs:** `http://localhost:8005/docs`
- **Qdrant dashboard:** `http://localhost:6333/dashboard`

For detailed local dev setup and VS Code debugging, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check and config info |
| `POST` | `/api/create-collection` | Create/ensure Qdrant collection |
| `POST` | `/api/ingest` | Ingest a Docling JSON document |
| `POST` | `/api/ingest-file` | Upload and ingest a PDF |
| `GET/POST/PUT/DELETE` | `/api/chunks` | CRUD over in-memory chunks |

---

## Repository Layout

```
board policy bot/
├── app/              # FastAPI application (routes, deps, settings, utilities)
├── docs/             # Long-form documentation
│   ├── DEPLOYMENT.md
│   ├── GITHUB.md
│   ├── ARCHITECTURE.md
│   ├── INGESTION.md
│   └── DEVELOPMENT.md
├── prompts/          # LLM prompt templates
│   ├── system_prompt.md
│   └── rag_query.md
├── scripts/          # Utility scripts (model download, etc.)
├── tests/            # pytest test suite
├── data/             # Docling output JSON, local Qdrant storage
├── models/           # Downloaded embedding models (git-ignored)
├── docker-compose.yml
└── Dockerfile
```

---

## Running Tests

```bash
pytest -q
```
