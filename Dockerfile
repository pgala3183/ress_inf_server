FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image for fast container startup.
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer; \
cls_id = 'distilbert-base-uncased-finetuned-sst-2-english'; \
gen_id = 'distilgpt2'; \
AutoTokenizer.from_pretrained(cls_id); \
AutoModelForSequenceClassification.from_pretrained(cls_id); \
AutoTokenizer.from_pretrained(gen_id); \
AutoModelForCausalLM.from_pretrained(gen_id)"

COPY server/ ./server/

EXPOSE 8000

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]
