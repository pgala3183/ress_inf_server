# Apply manifests to a local kind cluster (run from repository root).
param(
    [string]$ClusterName = "resilient-inf"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root

if (-not (Get-Command kind -ErrorAction SilentlyContinue)) {
    $LocalKind = Join-Path $Root ".tools\kind.exe"
    if (Test-Path $LocalKind) {
        $env:PATH = "$(Split-Path $LocalKind -Parent);$env:PATH"
    } else {
        Write-Error "kind is not installed. See k8s/kind/README.md"
    }
}

if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Error "kubectl is not installed."
}

Write-Host "Building image..."
docker build -t resilient-inference-server:latest .

$clusterExists = kind get clusters 2>$null | Select-String -SimpleMatch $ClusterName
if (-not $clusterExists) {
    Write-Host "Creating kind cluster '$ClusterName'..."
    kind create cluster --name $ClusterName --config k8s/kind/cluster-config.yaml
}

Write-Host "Loading image into kind..."
kind load docker-image resilient-inference-server:latest --name $ClusterName

Write-Host "Applying manifests..."
kubectl apply -f k8s/service-ondemand.yaml `
               -f k8s/service-spot.yaml `
               -f k8s/deployment-ondemand.yaml `
               -f k8s/deployment-spot.yaml `
               -f k8s/deployment-router.yaml `
               -f k8s/service.yaml `
               -f k8s/hpa.yaml

Write-Host "Waiting for pods..."
kubectl wait --for=condition=ready pod `
    -l app=resilient-inference-server `
    --timeout=180s
kubectl wait --for=condition=ready pod `
    -l app=resilient-inference-router `
    --timeout=120s

kubectl get pods -o wide
kubectl get svc resilient-inference-server
Write-Host "Done. Port-forward with: kubectl port-forward svc/resilient-inference-server 8000:80"
