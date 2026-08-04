from __future__ import annotations

import json
import math
import string
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .text import pretokenize, truncate_words


@dataclass
class EncodedSentence:
    words: list[str]
    embeddings: torch.Tensor  # [num_words, hidden_dim]
    mass: torch.Tensor  # [num_words], normalized

    def to(self, device: torch.device | str) -> "EncodedSentence":
        return EncodedSentence(
            words=self.words,
            embeddings=self.embeddings.to(device),
            mass=self.mass.to(device),
        )


_DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "there",
    "this",
    "to",
    "was",
    "were",
    "with",
}

# These words must not be removed as ordinary stopwords. They often determine
# whether two clinical statements are compatible.
_DEFAULT_LOGIC_TOKENS = {
    "no",
    "not",
    "without",
    "absent",
    "absence",
    "present",
    "presence",
    "preserved",
    "disrupted",
    "intact",
    "increased",
    "decreased",
    "hyperreflective",
    "hyporeflective",
    "hyperdense",
    "hypodense",
    "left",
    "right",
    "bilateral",
    "unilateral",
    "mild",
    "moderate",
    "severe",
    "small",
    "large",
}


class IDFTokenWeighter:
    """Assign fixed token masses using corpus IDF and clinical logic safeguards."""

    def __init__(
        self,
        idf: Mapping[str, float] | None = None,
        *,
        default_idf: float = 1.0,
        stopword_weight: float = 0.2,
        punctuation_weight: float = 0.0,
        logic_token_min_weight: float = 1.0,
        idf_min: float = 0.5,
        idf_max: float = 6.0,
        stopwords: set[str] | None = None,
        logic_tokens: set[str] | None = None,
    ) -> None:
        if default_idf <= 0:
            raise ValueError("default_idf must be > 0")
        if stopword_weight < 0 or punctuation_weight < 0:
            raise ValueError("weights must be non-negative")
        if idf_max < idf_min:
            raise ValueError("idf_max must be >= idf_min")

        self.idf = {str(k).lower(): float(v) for k, v in (idf or {}).items()}
        self.default_idf = float(default_idf)
        self.stopword_weight = float(stopword_weight)
        self.punctuation_weight = float(punctuation_weight)
        self.logic_token_min_weight = float(logic_token_min_weight)
        self.idf_min = float(idf_min)
        self.idf_max = float(idf_max)
        self.stopwords = stopwords or _DEFAULT_STOPWORDS
        self.logic_tokens = logic_tokens or _DEFAULT_LOGIC_TOKENS

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: object) -> "IDFTokenWeighter":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("IDF JSON must contain a token-to-value object")
        return cls(payload, **kwargs)

    def raw_weights(self, words: Sequence[str]) -> torch.Tensor:
        values: list[float] = []
        for word in words:
            lower = word.lower()
            base = min(
                self.idf_max,
                max(self.idf_min, self.idf.get(lower, self.default_idf)),
            )
            if all(char in string.punctuation for char in word):
                value = self.punctuation_weight
            elif lower in self.logic_tokens:
                value = max(base, self.logic_token_min_weight)
            elif lower in self.stopwords:
                value = base * self.stopword_weight
            else:
                value = base
            values.append(value)
        return torch.tensor(values, dtype=torch.float32)

    def normalized_weights(self, words: Sequence[str]) -> torch.Tensor:
        weights = self.raw_weights(words)
        positive = weights > 0
        if not bool(positive.any()):
            weights = torch.ones_like(weights)
        total = weights.sum().clamp_min(1e-8)
        return weights / total


class BaseTokenEncoder(ABC):
    @abstractmethod
    def encode(self, sentence: str, *, max_words: int = 96) -> EncodedSentence:
        raise NotImplementedError


class FrozenHFTokenEncoder(BaseTokenEncoder):
    """Frozen Hugging Face encoder with word-piece-to-word aggregation.

    `transformers` is imported lazily so OT unit tests do not require it.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        weighter: IDFTokenWeighter | None = None,
        cache_size: int = 4096,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "FrozenHFTokenEncoder requires `transformers`; install requirements.txt"
            ) from exc

        self.device = torch.device(device)
        self.weighter = weighter or IDFTokenWeighter()
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[tuple[str, int], EncodedSentence] = OrderedDict()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required for word_ids() aggregation")

        model_kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _cache_get(self, key: tuple[str, int]) -> EncodedSentence | None:
        item = self._cache.get(key)
        if item is None:
            return None
        self._cache.move_to_end(key)
        return item.to(self.device)

    def _cache_put(self, key: tuple[str, int], value: EncodedSentence) -> None:
        if self.cache_size <= 0:
            return
        cpu_value = value.to("cpu")
        self._cache[key] = cpu_value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    @torch.inference_mode()
    def encode(self, sentence: str, *, max_words: int = 96) -> EncodedSentence:
        key = (sentence, max_words)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        words = truncate_words(pretokenize(sentence), max_words)
        if not words:
            raise ValueError("Cannot encode an empty sentence")

        batch = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=min(getattr(self.tokenizer, "model_max_length", 512), 512),
        )
        word_ids = batch.word_ids(batch_index=0)
        batch = {key_: value.to(self.device) for key_, value in batch.items()}
        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state[0].float()

        grouped: list[torch.Tensor] = []
        retained_words: list[str] = []
        unique_word_ids = sorted({wid for wid in word_ids if wid is not None})
        for wid in unique_word_ids:
            positions = [index for index, item in enumerate(word_ids) if item == wid]
            grouped.append(hidden[positions].mean(dim=0))
            retained_words.append(words[wid])

        embeddings = torch.stack(grouped, dim=0)
        raw_mass = self.weighter.raw_weights(retained_words).to(self.device)
        keep = raw_mass > 0
        if bool(keep.any()):
            embeddings = embeddings[keep]
            retained_words = [w for w, flag in zip(retained_words, keep.tolist()) if flag]
            raw_mass = raw_mass[keep]
        else:
            raw_mass = torch.ones(len(retained_words), device=self.device)

        mass = raw_mass / raw_mass.sum().clamp_min(1e-8)
        result = EncodedSentence(retained_words, embeddings, mass)
        self._cache_put(key, result)
        return result


class StaticTokenEncoder(BaseTokenEncoder):
    """Small deterministic encoder for tests and local debugging."""

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        *,
        unknown_vector: Sequence[float] | None = None,
        weighter: IDFTokenWeighter | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if not vectors:
            raise ValueError("vectors cannot be empty")
        self.vectors = {
            key.lower(): torch.tensor(value, dtype=torch.float32)
            for key, value in vectors.items()
        }
        dim = next(iter(self.vectors.values())).numel()
        if any(vector.numel() != dim for vector in self.vectors.values()):
            raise ValueError("All vectors must have the same dimensionality")
        self.unknown_vector = (
            torch.tensor(unknown_vector, dtype=torch.float32)
            if unknown_vector is not None
            else torch.zeros(dim, dtype=torch.float32)
        )
        self.weighter = weighter or IDFTokenWeighter(stopword_weight=1.0)
        self.device = torch.device(device)

    def encode(self, sentence: str, *, max_words: int = 96) -> EncodedSentence:
        words = truncate_words(pretokenize(sentence), max_words)
        if not words:
            raise ValueError("Cannot encode an empty sentence")
        embeddings = torch.stack(
            [self.vectors.get(word.lower(), self.unknown_vector) for word in words],
            dim=0,
        ).to(self.device)
        raw_mass = self.weighter.raw_weights(words).to(self.device)
        keep = raw_mass > 0
        if bool(keep.any()):
            embeddings = embeddings[keep]
            words = [w for w, flag in zip(words, keep.tolist()) if flag]
            raw_mass = raw_mass[keep]
        else:
            raw_mass = torch.ones(len(words), device=self.device)
        mass = raw_mass / raw_mass.sum().clamp_min(1e-8)
        return EncodedSentence(words, embeddings, mass)
