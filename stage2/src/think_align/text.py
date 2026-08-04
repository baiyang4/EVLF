from __future__ import annotations

import re
from typing import Iterable

_THINK_RE = re.compile(
    r"<(?:think|thinking)>\s*(.*?)\s*</(?:think|thinking)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_KW_RE = re.compile(r"<kw>.*?</kw>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", flags=re.IGNORECASE | re.DOTALL)
_OTHER_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_SPACE_RE = re.compile(r"\s+")

# English medical rationales are assumed. Decimal measurements are preserved.
_WORD_RE = re.compile(
    r"[A-Za-z]+(?:[-/][A-Za-z0-9]+)*|"
    r"\d+(?:\.\d+)?(?:%|mm|cm|µm|um)?|"
    r"[^\w\s]",
    flags=re.UNICODE,
)


def extract_reasoning(text: str) -> str:
    """Extract the content of <think>/<thinking>; fall back to the full string."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    match = _THINK_RE.search(text)
    return match.group(1) if match else text


def clean_reasoning(text: str) -> str:
    """Remove structural tags while retaining ordinary clinical prose."""

    text = extract_reasoning(text)
    text = _KW_RE.sub(" ", text)
    text = _ANSWER_RE.sub(" ", text)
    text = _OTHER_TAG_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def pretokenize(text: str) -> list[str]:
    """Deterministic word-level pre-tokenization for contextual encoding."""

    return _WORD_RE.findall(text)


def _regex_sentence_split(text: str, split_on_semicolon: bool) -> list[str]:
    # Split only when punctuation is followed by whitespace and a plausible sentence start.
    boundary = r"(?<=[.!?])\s+(?=[A-Z0-9])"
    if split_on_semicolon:
        boundary = r"(?<=[.!?;])\s+(?=[A-Z0-9])"
    pieces = re.split(boundary, text)
    return [piece.strip() for piece in pieces if piece.strip()]


def split_sentences(
    text: str,
    *,
    split_on_semicolon: bool = True,
    max_sentences: int | None = None,
) -> list[str]:
    """Split reasoning into sentences, using pysbd when available.

    The fallback deliberately does not split comma-separated clinical findings into
    clauses; ThinkAlign obtains fine-grained coverage inside each sentence through OT.
    """

    cleaned = clean_reasoning(text)
    if not cleaned:
        return []

    try:
        import pysbd  # type: ignore

        segmenter = pysbd.Segmenter(language="en", clean=False)
        sentences = [s.strip() for s in segmenter.segment(cleaned) if s.strip()]
        if split_on_semicolon:
            expanded: list[str] = []
            for sentence in sentences:
                expanded.extend(
                    part.strip() for part in sentence.split(";") if part.strip()
                )
            sentences = expanded
    except ImportError:
        sentences = _regex_sentence_split(cleaned, split_on_semicolon)

    if max_sentences is not None:
        sentences = sentences[:max_sentences]
    return sentences


def truncate_words(words: Iterable[str], max_words: int) -> list[str]:
    words_list = list(words)
    return words_list[:max_words]
