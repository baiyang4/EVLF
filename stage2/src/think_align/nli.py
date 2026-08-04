from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import torch


class BaseContradictionScorer(ABC):
    @abstractmethod
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> torch.Tensor:
        """Return contradiction probabilities in [0, 1], shape [num_pairs]."""
        raise NotImplementedError


class NullContradictionScorer(BaseContradictionScorer):
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> torch.Tensor:
        return torch.zeros(len(pairs), dtype=torch.float32)


class FixedContradictionScorer(BaseContradictionScorer):
    """Lookup scorer for unit tests; unspecified pairs receive `default`."""

    def __init__(
        self,
        values: Mapping[tuple[str, str], float] | None = None,
        *,
        default: float = 0.0,
    ) -> None:
        self.values = dict(values or {})
        self.default = float(default)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> torch.Tensor:
        return torch.tensor(
            [self.values.get(pair, self.default) for pair in pairs],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)


class HFContradictionScorer(BaseContradictionScorer):
    """Frozen Hugging Face NLI model used only as a contradiction detector."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | torch.device = "cuda",
        batch_size: int = 32,
        max_length: int = 256,
        temperature: float = 1.0,
        contradiction_label_id: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "HFContradictionScorer requires `transformers`; install requirements.txt"
            ) from exc

        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.temperature = float(temperature)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        if contradiction_label_id is None:
            contradiction_label_id = self._infer_contradiction_label_id()
        self.contradiction_label_id = int(contradiction_label_id)

    def _infer_contradiction_label_id(self) -> int:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for key, value in id2label.items():
            if "contrad" in str(value).lower():
                return int(key)
        raise ValueError(
            "Could not infer the contradiction label. Pass contradiction_label_id explicitly."
        )

    @torch.inference_mode()
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> torch.Tensor:
        if not pairs:
            return torch.empty(0, dtype=torch.float32)

        outputs: list[torch.Tensor] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            premises = [premise for premise, _ in chunk]
            hypotheses = [hypothesis for _, hypothesis in chunk]
            batch = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            logits = self.model(**batch).logits.float() / self.temperature
            probs = logits.softmax(dim=-1)[:, self.contradiction_label_id]
            outputs.append(probs.cpu())
        return torch.cat(outputs, dim=0).clamp(0.0, 1.0)
