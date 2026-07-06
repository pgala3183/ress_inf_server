"""Model inference: classification batching and generative step-based decoding."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from server.config import (
    CLASSIFICATION_MODEL_ID,
    DEFAULT_MAX_TOKENS,
    GENERATIVE_MODEL_ID,
)


@dataclass
class SequenceState:
    """Mutable decode state for one in-flight generative sequence.

    NOTE (simplification vs. vLLM / TensorRT-LLM):
    We store the full token list and re-run a padded forward pass each step.
    A production server would keep a paged KV cache and only compute the latest
    query position — we intentionally omit that here and focus on *scheduling*.
    """

    prompt: str
    token_ids: list[int]
    prompt_token_count: int
    generated_count: int = 0
    max_tokens: int = DEFAULT_MAX_TOKENS
    finished: bool = False
    finish_reason: str | None = None


class ModelWorker:
    def __init__(self, generative: bool = False, model_id: str | None = None) -> None:
        self.generative = generative
        self.model_id = model_id or (GENERATIVE_MODEL_ID if generative else CLASSIFICATION_MODEL_ID)
        self._tokenizer: AutoTokenizer | None = None
        self._classification_model: AutoModelForSequenceClassification | None = None
        self._generative_model: AutoModelForCausalLM | None = None
        self._eos_token_id: int | None = None
        self._pad_token_id: int | None = None

    async def load(self) -> None:
        if self._tokenizer is not None:
            if self.generative and self._generative_model is not None:
                return
            if not self.generative and self._classification_model is not None:
                return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        if self.generative:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._generative_model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self._generative_model.eval()
            self._eos_token_id = self._tokenizer.eos_token_id
            self._pad_token_id = self._tokenizer.pad_token_id
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._classification_model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._classification_model.eval()

    # ------------------------------------------------------------------
    # Classification path (Phase 1–3): single-shot batch inference
    # ------------------------------------------------------------------

    def run_batch(self, inputs: list[str]) -> list[dict[str, str | float]]:
        if not inputs:
            return []
        if self._tokenizer is None or self._classification_model is None:
            raise RuntimeError("Classification model not loaded")

        encoded = self._tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._classification_model(**encoded).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        results: list[dict[str, str | float]] = []
        for row in probabilities:
            score, label_id = row.max(dim=0).values.item(), row.argmax(dim=0).item()
            label = self._classification_model.config.id2label[label_id]
            results.append({"label": label, "score": score})
        return results

    # ------------------------------------------------------------------
    # Generative path (Phase 4): one decode step across active sequences
    # ------------------------------------------------------------------

    def start_sequence(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> SequenceState:
        """Initialize decode state for a new prompt."""
        if self._tokenizer is None or self._generative_model is None:
            raise RuntimeError("Generative model not loaded")

        encoded = self._tokenizer.encode(prompt, add_special_tokens=True)
        return SequenceState(
            prompt=prompt,
            token_ids=list(encoded),
            prompt_token_count=len(encoded),
            max_tokens=max_tokens,
        )

    def step(self, active_sequences: list[SequenceState]) -> list[SequenceState]:
        """Run exactly one greedy decode step for every non-finished sequence.

        All sequences in the list are advanced in a single batched forward pass.
        Sequences already marked finished are skipped (caller may filter them out).

        Simplifications vs. production inference engines:
        - No PagedAttention / block-sparse KV cache
        - No CUDA graphs or custom attention kernels
        - Full re-tokenization + pad-to-max-length each step (O(batch * seq_len) work)
        """
        if not active_sequences:
            return []

        if self._tokenizer is None or self._generative_model is None:
            raise RuntimeError("Generative model not loaded")

        # Only advance sequences that are still generating.
        live = [sequence for sequence in active_sequences if not sequence.finished]
        if not live:
            return active_sequences

        max_len = max(len(sequence.token_ids) for sequence in live)
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []

        pad_id = self._pad_token_id or 0
        for sequence in live:
            pad_count = max_len - len(sequence.token_ids)
            batch_input_ids.append([pad_id] * pad_count + sequence.token_ids)
            batch_attention_mask.append([0] * pad_count + [1] * len(sequence.token_ids))

        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)

        with torch.no_grad():
            logits = self._generative_model(input_ids=input_ids, attention_mask=attention_mask).logits
            next_token_ids = logits[:, -1, :].argmax(dim=-1)

        for sequence, next_token_id in zip(live, next_token_ids.tolist(), strict=True):
            sequence.token_ids.append(next_token_id)
            sequence.generated_count += 1

            if self._eos_token_id is not None and next_token_id == self._eos_token_id:
                sequence.finished = True
                sequence.finish_reason = "eos"
            elif sequence.generated_count >= sequence.max_tokens:
                sequence.finished = True
                sequence.finish_reason = "max_tokens"

        return active_sequences

    def decode_sequence(self, sequence: SequenceState) -> str:
        """Decode only the newly generated tokens (exclude the original prompt)."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        generated_ids = sequence.token_ids[sequence.prompt_token_count :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)
