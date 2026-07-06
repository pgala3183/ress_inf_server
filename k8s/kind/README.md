# Local kind deployment (development / demo — no GCP billing)
#
# Prerequisites: Docker, kind, kubectl
#
# From the repository root:
#
#   docker build -t resilient-inference-server:latest .
#   kind create cluster --name resilient-inf --config k8s/kind/cluster-config.yaml
#   kind load docker-image resilient-inference-server:latest --name resilient-inf
#   kubectl apply -f k8s/service-ondemand.yaml -f k8s/service-spot.yaml \
#                  -f k8s/deployment-ondemand.yaml -f k8s/deployment-spot.yaml \
#                  -f k8s/deployment-router.yaml -f k8s/service.yaml -f k8s/hpa.yaml
#   kubectl get pods -o wide
#   kubectl port-forward svc/resilient-inference-server 8000:80
#
# Or on Windows: .\k8s\scripts\deploy-kind.ps1
#
# The Spot worker carries a mock cloud.google.com/gke-spot taint so
# deployment-spot.yaml schedules the same way it would on GKE, even though
# kind cannot actually preempt nodes.
#
# Teardown:
#   kind delete cluster --name resilient-inf
