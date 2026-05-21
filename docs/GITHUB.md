# GitHub Actions Azure Deployment Setup Guide

This guide provides step-by-step instructions to configure **GitHub Actions** for continuous deployment of the Board Policy Bot to **Azure Container Apps**.

Whenever you push to the `main` branch, the workflow will automatically:
1. Authenticate with Azure using a secure Service Principal.
2. Build and push the latest FastAPI Ingest API Docker image to Azure Container Registry (ACR).
3. Deploy or update the Container Apps (`qdrant`, `board-ingest-api`, and `openwebui`).

---

## Prerequisites

Before starting, ensure you have:
- An active **Azure Subscription**.
- The **Azure CLI** installed locally (`az`).
- Access to the GitHub repository settings.
- The base Resource Group and container environment settings chosen (defaults are used below).

---

## Step 1: Create Azure Resources (If not already created)

If you haven't deployed the application yet, create the resource group first. This is necessary because we will scope our secure credentials to this specific resource group.

Run the following Azure CLI command locally:

```bash
az group create --name imc-rag-rg --location eastus
```

> [!NOTE]
> If you plan to customize the resource group name, change `imc-rag-rg` to your desired name in all subsequent commands.

---

## Step 2: Generate an Azure Service Principal

GitHub Actions authenticates with Azure using a **Service Principal**. For security best practices, we recommend scoping its access strictly to the resource group containing your deployment.

### 1. Retrieve your Subscription ID
Run the following command to get your subscription ID:

```bash
az account show --query id --output tsv
```

### 2. Create the Service Principal
Run the command below, replacing `{subscription-id}` with your actual subscription ID.

```bash
az ad sp create-for-rbac \
  --name "board-policy-bot-github-actions" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/imc-rag-rg \
  --sdk-auth
```

> [!TIP]
> If you want the script to be able to create new resource groups automatically, you can scope the Service Principal to the entire subscription by using `--scopes /subscriptions/{subscription-id}`. However, restricting it to the resource group is highly recommended for security.

### 3. Copy the Output JSON
The command will print a JSON block that looks like this:

```json
{
  "clientId": "00000000-0000-0000-0000-000000000000",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "00000000-0000-0000-0000-000000000000",
  "tenantId": "00000000-0000-0000-0000-000000000000",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:3307/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

Copy this **entire JSON block**. You will save it as a GitHub secret in the next step.

---

## Step 3: Configure GitHub Secrets

To allow the workflow to run, you must add the required credentials as Repository Secrets.

1. On GitHub, navigate to your repository.
2. Go to **Settings** > **Secrets and variables** > **Actions**.
3. Under the **Secrets** tab, click **New repository secret**.
4. Add the following secrets:

| Secret Name | Description | Example / Source |
| :--- | :--- | :--- |
| `AZURE_CREDENTIALS` | The entire JSON block output from `az ad sp create-for-rbac` | `{ "clientId": "...", ... }` |
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI instance API Key | `4f3c...` |
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI endpoint URL | `https://your-resource.openai.azure.com/` |

> [!IMPORTANT]
> Do not add spaces or newlines around the secret values. Paste them exactly as generated.

---

## Step 4: Configure GitHub Variables (Optional)

If your Azure environment uses custom resource names that differ from the defaults, you can define them as GitHub **Variables** rather than hardcoding them in the deployment script.

1. Go to your repository **Settings** > **Secrets and variables** > **Actions**.
2. Click the **Variables** tab.
3. Click **New repository variable** to configure any of the following:

| Variable Name | Description | Default Value |
| :--- | :--- | :--- |
| `RESOURCE_GROUP` | The name of your Azure Resource Group | `imc-rag-rg` |
| `LOCATION` | Azure deployment location / region | `eastus` |
| `ACR_NAME` | The name of your Azure Container Registry | `imcregistry` |
| `ENV_NAME` | The name of your Azure Container Apps Environment | `imc-rag-env` |

To make these active, make sure they are passed in the `.github/workflows/azure-deploy.yml` workflow env block:
```yaml
      - name: Run Deployment Script
        env:
          RESOURCE_GROUP: ${{ vars.RESOURCE_GROUP || 'imc-rag-rg' }}
          ACR_NAME: ${{ vars.ACR_NAME || 'imcregistry' }}
          ENV_NAME: ${{ vars.ENV_NAME || 'imc-rag-env' }}
          LOCATION: ${{ vars.LOCATION || 'eastus' }}
          AZURE_OPENAI_API_KEY: ${{ secrets.AZURE_OPENAI_API_KEY }}
          AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
```

---

## Step 5: Verify the Workflow

Once the secrets are configured, you can trigger the pipeline:

### Triggering via Push
Simply push a commit to your `main` branch:
```bash
git add .
git commit -m "docs: add GitHub Actions setup instructions"
git push origin main
```

### Triggering Manually
1. Go to the **Actions** tab in your GitHub repository.
2. Select the **Deploy to Azure Container Apps** workflow on the left sidebar.
3. Click the **Run workflow** dropdown button on the right.
4. Select the branch (e.g., `main`) and click **Run workflow**.

> [!WARNING]
> The initial build and deploy step may take **5-8 minutes** since the Docker image is being built from scratch and heavy embedding assets are compiled. Subsequent builds will benefit from layer caching if applicable.
