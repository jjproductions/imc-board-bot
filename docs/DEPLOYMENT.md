# Deployment Guide

This guide covers deploying the Board Policy Bot in all environments: local Docker, and Azure Container Apps (production).

---

## Local Deployment (Docker Compose)

Docker Compose is the recommended way to run the full stack locally — it spins up the FastAPI ingest API alongside a Qdrant instance with persistent storage.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- `.env` file configured (see [Environment Variables](#environment-variables) below)

### Start the Stack

```bash
# Build images and start all services
docker compose up --build

# Or run in detached (background) mode
docker compose up -d --build
```

This will start:
- **Qdrant** at `http://localhost:6333` (persistent storage mapped to `./data/qdrant_storage`)
- **FastAPI Ingest API** at `http://localhost:8005`

### Manual Single-Container Build

If you only need the FastAPI service without Qdrant:

```bash
# Build the image
docker build -t board-ingest-api .

# Run the container
docker run -p 8005:8000 \
  --env-file .env \
  -v ~/.cache/huggingface:/app/models \
  board-ingest-api
```

> [!NOTE]
> The `Dockerfile` automatically filters out macOS-specific dependencies (`ocrmac`, `pyobjc`) from `requirements.txt` to prevent build failures on the Linux base image.

---

## Azure Container Apps Deployment (Production)

The production environment runs on Azure Container Apps with:
- **FastAPI Ingest API** — custom container image pushed to Azure Container Registry (ACR)
- **Qdrant** — container with Azure Files persistent storage
- **OpenWebUI** — frontend for staff access

> [!TIP]
> **Automated Deployments:** You can also deploy automatically on every push to the `main` branch using GitHub Actions. See the [GitHub Actions Azure Deployment Guide](GITHUB.md) for full configuration steps.

### Prerequisites

- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) installed
- Logged in: `az login`

## 1. Create Azure Resources
First, set up a Resource Group, a Container Registry (ACR), and an Azure Container Apps Environment.

```bash
RESOURCE_GROUP="imc-rag-rg"
LOCATION="eastus"
ACR_NAME="imcregistry"
ENV_NAME="imc-rag-env"


# Create Resource Group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Container Registry
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Create Container Apps Environment
az containerapp env create --name $ENV_NAME --resource-group $RESOURCE_GROUP --location $LOCATION
```

### 1. Push Image to ACR

```bash
# Log in to your registry
az acr login --name $ACR_NAME

# Build and Push (Ensure AMD64 architecture for Azure Container Apps)
docker build --platform linux/amd64 -t $ACR_NAME.azurecr.io/board-ingest-api:latest -f Dockerfile .
docker push $ACR_NAME.azurecr.io/board-ingest-api:latest
```

### 2. Deploy Qdrant with Persistent Storage

```bash
az containerapp create \
  --name qdrant \
  --resource-group <your-rg> \
  --environment <your-env> \
  --image qdrant/qdrant:latest \
  --target-port 6333 \
  --ingress internal \
  --cpu 1 --memory 2Gi \
  --min-replicas 1 --max-replicas 1
```

> [!IMPORTANT]
> Mount an Azure Files share to `/qdrant/storage` to persist vector data across restarts. Configure this via the Container Apps volume mount settings in the Azure Portal or via `az containerapp update`.

### 3. Deploy the FastAPI Ingest API

```bash
az containerapp create \
  --name board-ingest-api \
  --resource-group <your-rg> \
  --environment <your-env> \
  --image <your-acr-name>.azurecr.io/board-ingest-api:latest \
  --registry-server <your-acr-name>.azurecr.io \
  --target-port 8000 \
  --ingress external \
  --cpu 1 --memory 2Gi \
  --min-replicas 1 --max-replicas 3 \
  --env-vars \
    QDRANT_HOST=<qdrant-internal-fqdn> \
    QDRANT_PORT=6333 \
    AZURE_OPENAI_API_KEY=secretref:azure-openai-key \
    AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
```

### 4. Deploy OpenWebUI

```bash
az containerapp create \
  --name openwebui \
  --resource-group <your-rg> \
  --environment <your-env> \
  --image ghcr.io/open-webui/open-webui:main \
  --target-port 8080 \
  --ingress external \
  --cpu 0.5 --memory 1Gi \
  --min-replicas 1 --max-replicas 1 \
  --env-vars OPENAI_API_BASE_URL=https://<board-ingest-api-fqdn>
```

---

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `QDRANT_HOST` | Qdrant hostname | `localhost` or internal Azure FQDN |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `COLLECTION_NAME` | Qdrant collection | `board-policies_chunks` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | (use Azure Key Vault in prod) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | `https://....openai.azure.com/` |
| `DENSE_MODEL_PATH` | Path to dense embedding model | `./models/bge-m3` |

Copy `.env.development` to `.env` and fill in values for local runs. **Never commit `.env` to source control.**

---

## Verifying the Deployment

```bash
# Health check
curl -s https://<your-api-fqdn>/api/health | jq

# Create the Qdrant collection
curl -X POST "https://<your-api-fqdn>/api/create-collection?collection_name=board-policies_chunks&vector_dim=1024"
```
