"""Synchronous model inference worker (baseline — no batching scheduler yet)."""

from __future__ import annotations

import asyncio

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


class ModelWorker:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSequenceClassification | None = None

    async def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.eval()

    def run_batch(self, inputs: list[str]) -> list[dict[str, str | float]]:
        if not inputs:
            return []
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("ModelWorker.load() must be called before inference")

        encoded = self._tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        results: list[dict[str, str | float]] = []
        for row in probabilities:
            score, label_id = row.max(dim=0).values.item(), row.argmax(dim=0).item()
            label = self._model.config.id2label[label_id]
            results.append({"label": label, "score": score})
        return results
