#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "====================================================="
echo "🚀 Deploying Board Policy Bot to Azure Container Apps"
echo "====================================================="

# ==========================================
# Configuration & Variables
# ==========================================
# Modify these variables or pass them in via environment variables
RESOURCE_GROUP="${RESOURCE_GROUP:-imc-rag-rg}"
LOCATION="${LOCATION:-eastus}"
ACR_NAME="${ACR_NAME:-imcregistry}"
ENV_NAME="${ENV_NAME:-imc-rag-env}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
QDRANT_DEFAULT_COLLECTION="${QDRANT_DEFAULT_COLLECTION:-board-policies-hybrid}"

# IMPORTANT: Ensure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set in your environment
if [[ -z "$AZURE_OPENAI_API_KEY" ]] || [[ -z "$AZURE_OPENAI_ENDPOINT" ]]; then
    echo "❌ ERROR: AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set."
    echo "Run: export AZURE_OPENAI_API_KEY='your-key' && export AZURE_OPENAI_ENDPOINT='your-endpoint'"
    exit 1
fi

# ==========================================
# 1. Base Infrastructure (Idempotent)
# ==========================================
echo -e "\n[1/5] Setting up base Azure resources..."

# Create Resource Group
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none
echo "✅ Resource Group: $RESOURCE_GROUP"

# Create Azure Container Registry (ACR)
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled true -o none
echo "✅ Container Registry: $ACR_NAME"

# Create Container Apps Environment
az containerapp env create --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" --location "$LOCATION" -o none
echo "✅ Container Apps Environment: $ENV_NAME"

# Get ACR Login Server
ACR_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer --output tsv)

# ==========================================
# 2. Build & Push API Image
# ==========================================
echo -e "\n[2/5] Building and pushing Ingest API image using Azure Container Registry..."

IMAGE_URI="$ACR_SERVER/board-ingest-api:$IMAGE_TAG"

# Build directly on Azure Container Registry to leverage cloud speed and layer caching
az acr build \
    --registry "$ACR_NAME" \
    --image "board-ingest-api:$IMAGE_TAG" \
    --platform linux/amd64 \
    -f Dockerfile .

echo "✅ Built and pushed image: $IMAGE_URI"

# ==========================================
# Helper Function for Idempotent App Deploy
# ==========================================
deploy_container_app() {
    local APP_NAME=$1
    shift
    local CREATE_ARGS=("$@")

    echo "Checking if container app '$APP_NAME' exists..."
    if az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        echo "🔄 Updating existing app: $APP_NAME..."
        # Extract the --image and --env-vars arguments to pass to update
        local IMAGE=""
        local ENV_VARS=()
        local capture_envs=false
        
        for i in "${!CREATE_ARGS[@]}"; do
            if [[ "${CREATE_ARGS[$i]}" == "--image" ]]; then
                IMAGE="${CREATE_ARGS[$((i+1))]}"
            fi
        done
        
        for arg in "${CREATE_ARGS[@]}"; do
            if [[ "$arg" == "--env-vars" ]]; then
                capture_envs=true
                continue
            fi
            if [[ $capture_envs == true ]]; then
                if [[ "$arg" == --* ]]; then
                    capture_envs=false
                else
                    ENV_VARS+=("$arg")
                fi
            fi
        done
        
        # If env vars are provided, update both the image and the environment variables
        if [[ ${#ENV_VARS[@]} -gt 0 ]]; then
            az containerapp update \
                --name "$APP_NAME" \
                --resource-group "$RESOURCE_GROUP" \
                --image "$IMAGE" \
                --set-env-vars "${ENV_VARS[@]}" \
                -o none
        else
            az containerapp update \
                --name "$APP_NAME" \
                --resource-group "$RESOURCE_GROUP" \
                --image "$IMAGE" \
                -o none
        fi
    else
        echo "✨ Creating new app: $APP_NAME..."
        az containerapp create \
            --name "$APP_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --environment "$ENV_NAME" \
            "${CREATE_ARGS[@]}" \
            -o none
    fi
}

# ==========================================
# 3. Deploy Qdrant
# ==========================================
echo -e "\n[3/5] Deploying Qdrant..."
deploy_container_app "qdrant" \
    --image "qdrant/qdrant:latest" \
    --target-port 6333 \
    --ingress internal \
    --cpu 1.0 --memory 2.0Gi \
    --min-replicas 1 --max-replicas 1
echo "✅ Qdrant deployed."

# Get Qdrant Internal FQDN
QDRANT_FQDN=$(az containerapp show --name qdrant --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)

# ==========================================
# 4. Deploy FastAPI Ingest API
# ==========================================
echo -e "\n[4/5] Deploying FastAPI Ingest API..."
deploy_container_app "board-ingest-api" \
    --image "$IMAGE_URI" \
    --registry-server "$ACR_SERVER" \
    --target-port 8000 \
    --ingress external \
    --cpu 2.0 --memory 4.0Gi \
    --min-replicas 1 --max-replicas 3 \
    --secrets "openai-key=$AZURE_OPENAI_API_KEY" \
    --env-vars \
        "QDRANT_HOST=$QDRANT_FQDN" \
        "QDRANT_PORT=443" \
        "QDRANT__DEFAULT_COLLECTION=$QDRANT_DEFAULT_COLLECTION" \
        "AZURE_OPENAI_API_KEY=secretref:openai-key" \
        "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT"
echo "✅ FastAPI Ingest API deployed."

# Get API External FQDN
API_FQDN=$(az containerapp show --name board-ingest-api --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)

# ==========================================
# 5. Deploy OpenWebUI
# ==========================================
echo -e "\n[5/5] Deploying OpenWebUI..."
deploy_container_app "openwebui" \
    --image "ghcr.io/open-webui/open-webui:main" \
    --target-port 8080 \
    --ingress external \
    --cpu 0.5 --memory 1.0Gi \
    --min-replicas 1 --max-replicas 1 \
    --env-vars "OPENAI_API_BASE_URL=https://$API_FQDN/api"
echo "✅ OpenWebUI deployed."

WEBUI_FQDN=$(az containerapp show --name openwebui --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)

echo -e "\n====================================================="
echo "🎉 DEPLOYMENT COMPLETE!"
echo "====================================================="
echo "🔗 OpenWebUI URL: https://$WEBUI_FQDN"
echo "🔗 Ingest API URL: https://$API_FQDN/docs"
echo "====================================================="
