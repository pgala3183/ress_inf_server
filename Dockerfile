# ---- Builder: install Python deps and bake model weights -----------------
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download both classification and generative model weights at build time.
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer; \
cls_id = 'distilbert-base-uncased-finetuned-sst-2-english'; \
gen_id = 'distilgpt2'; \
AutoTokenizer.from_pretrained(cls_id); \
AutoModelForSequenceClassification.from_pretrained(cls_id); \
AutoTokenizer.from_pretrained(gen_id); \
AutoModelForCausalLM.from_pretrained(gen_id)"

COPY server/ ./server/

# ---- Runtime: slim image with only what serving needs --------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

# Copy installed packages and cached weights from the builder stage.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface
COPY --from=builder /app/server /app/server

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]
