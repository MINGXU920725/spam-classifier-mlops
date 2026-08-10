# Azure deployment guide

This guide is for the first manual deployment. Complete it before adding CD so
the Azure resource names, registry URL, and credentials are known to work.

## Prerequisites

- An active Azure for Students subscription
- Azure CLI (`az`) and Docker Desktop
- The project passes `pytest` locally
- A unique lowercase Azure Container Registry name

Set a small Azure Cost Management budget before creating paid resources. Delete
the resource group when the demo is no longer needed.

## 1. Choose names

Replace the example values below. The registry name must be globally unique.

```powershell
$resourceGroup = "rg-spam-classifier-dev"
$location = "westeurope"
$registry = "youruniquespamregistry"
$image = "spam-classifier:v1"
$appPlan = "plan-spam-classifier-dev"
$webApp = "your-unique-spam-classifier"
```

## 2. Sign in and create resources

```powershell
az login
az account show
az group create --name $resourceGroup --location $location
az acr create --resource-group $resourceGroup --name $registry --sku Basic
az acr login --name $registry
```

## 3. Build and push the image

```powershell
docker build -t "$registry.azurecr.io/$image" .
docker push "$registry.azurecr.io/$image"
```

Alternatively, Azure can build it without a local Docker daemon:

```powershell
az acr build --registry $registry --image $image .
```

## 4. Create the Linux web app

Choose the smallest Linux SKU available to the student subscription that can
run the container, and verify its current price in the Azure portal first.

```powershell
az appservice plan create `
  --name $appPlan `
  --resource-group $resourceGroup `
  --is-linux `
  --sku B1

az webapp create `
  --resource-group $resourceGroup `
  --plan $appPlan `
  --name $webApp `
  --deployment-container-image-name "$registry.azurecr.io/$image"

az webapp config appsettings set `
  --resource-group $resourceGroup `
  --name $webApp `
  --settings WEBSITES_PORT=8000
```

In the Azure portal, give the Web App permission to pull from ACR using a
managed identity (recommended). Avoid committing registry passwords.

## 5. Verify

```powershell
$baseUrl = "https://$webApp.azurewebsites.net"
Invoke-RestMethod "$baseUrl/health"
Invoke-RestMethod `
  -Method Post `
  -Uri "$baseUrl/predict" `
  -ContentType "application/json" `
  -Body '{"text":"Congratulations! Claim your free prize."}'
```

Also visit `https://<app-name>.azurewebsites.net/docs`.

## 6. Inspect logs

```powershell
az webapp log config `
  --resource-group $resourceGroup `
  --name $webApp `
  --docker-container-logging filesystem

az webapp log tail --resource-group $resourceGroup --name $webApp
```

Do not log message bodies: SMS messages can contain personal information.

## 7. Add CD afterwards

After the manual deployment succeeds, add a GitHub Actions deployment job that:

1. runs the existing CI checks;
2. authenticates to Azure with OpenID Connect;
3. builds and pushes an immutable image tag such as the Git commit SHA;
4. updates the Web App image;
5. calls `/health` and fails if the deployment is unhealthy.

Record the real resource names in GitHub repository variables, not in source
code. Use federated identity instead of storing a long-lived Azure password.
