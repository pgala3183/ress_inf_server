FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image for fast container startup.
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
model_id = 'distilbert-base-uncased-finetuned-sst-2-english'; \
AutoTokenizer.from_pretrained(model_id); \
AutoModelForSequenceClassification.from_pretrained(model_id)"

COPY server/ ./server/

EXPOSE 8000

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]
