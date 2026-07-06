# Prometheus Adapter for queue_depth HPA (GKE)

GKE's built-in HPA only understands **Resource** metrics (CPU/memory) out of the
box. To scale on application metrics such as `queue_depth`, expose them through the
**Custom Metrics API** (`custom.metrics.k8s.io`) via the
[Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter).

This repo ships:

- `config.yaml` — adapter rules mapping Prometheus `queue_depth` and
  `gpu_utilization_percent` series to pod-scoped custom metrics
- Updated `k8s/hpa.yaml` — targets average `queue_depth` > **5** per pod

## Prerequisites (GKE)

1. **Google Cloud Managed Service for Prometheus** or a self-hosted Prometheus that
   scrapes inference pods on `/metrics`.
2. Pod template annotations (already on inference Deployments):

   ```yaml
   prometheus.io/scrape: "true"
   prometheus.io/port: "8000"
   prometheus.io/path: "/metrics"
   ```

3. Prometheus must label series with `namespace` and `pod` (standard kube-prometheus
   relabeling).

## Install Prometheus Adapter

These steps are **not runnable on kind alone** without installing Prometheus + the
adapter chart. Use a real GKE cluster with Managed Prometheus or Helm:

```bash
# Namespace for adapter
kubectl create namespace custom-metrics

# Apply rule ConfigMap (edit namespace if your adapter expects a different one)
kubectl apply -f k8s/prometheus-adapter/config.yaml

# Helm example (values must point at your Prometheus URL + mount config.yaml):
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter \
  -n custom-metrics \
  -f k8s/prometheus-adapter/helm-values.example.yaml
```

See `helm-values.example.yaml` for a minimal values stub.

## Verify custom metric is visible

```bash
# Should return queue_depth values per pod (not 404)
kubectl get --raw \
  "/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pods/*/queue_depth" | jq .

# HPA should show queue_depth target (may take 1–2 minutes after metric appears)
kubectl describe hpa resilient-inference-server-ondemand-hpa
```

## Synthetic load → scale-out → queue recovery

```bash
# 1. Deploy stack (router + pools + HPA + adapter)
kubectl apply -f k8s/service-ondemand.yaml -f k8s/service-spot.yaml
kubectl apply -f k8s/deployment-ondemand.yaml -f k8s/deployment-spot.yaml
kubectl apply -f k8s/deployment-router.yaml -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# 2. Port-forward router
kubectl port-forward svc/resilient-inference-server 8080:80

# 3. Spike load (50 concurrent requests, classification mode)
python benchmarks/synthetic_load_hpa.py --url http://127.0.0.1:8080 --concurrency 50

# 4. Watch HPA and queue_depth (repeat during load)
watch -n5 'kubectl get hpa; echo; python k8s/scripts/hpa_dry_run.py --namespace default'
```

**Expected behavior:**

1. `queue_depth` rises on inference pods during the spike.
2. HPA `CURRENT` replicas increase toward `MAX` when average queue depth per pod > 5.
3. After load stops, queue depth falls; HPA scales down after stabilization windows.

## Local dry run (no GKE)

When the adapter is not installed, use the dry-run script against port-forwarded pods
or local uvicorn instances:

```powershell
# Terminal 1 — inference server
python -m server.main --port 8000

# Terminal 2 — simulate metrics + scaling math
python k8s/scripts/hpa_dry_run.py --url http://127.0.0.1:8000 --target 5 --max-replicas 5
```

The script prints current queue depth, implied replica count, and routing weights.

## Alternative: Custom Metrics Stackdriver Adapter

If metrics are exported to **Cloud Monitoring** instead of Prometheus, use Google's
[Custom Metrics Stackdriver Adapter](https://github.com/GoogleCloudPlatform/k8s-stackdriver/tree/master/custom-metrics-stackdriver-adapter)
and define a Stackdriver metric descriptor for `queue_depth`. The HPA manifest
(`type: Pods`, `name: queue_depth`) stays the same; only the metrics pipeline differs.
