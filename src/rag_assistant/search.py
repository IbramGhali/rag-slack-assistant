"""Text normalisation, BM25, embedders and reciprocal rank fusion."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "my",
        "our",
        "your",
        "their",
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "if",
        "any",
        "as",
        "not",
        "no",
        "so",
        "than",
        "then",
        "there",
        "about",
        "into",
        "after",
        "before",
        "over",
        "under",
        "up",
        "down",
        "out",
        "per",
        "each",
    ]
)
TOKEN = re.compile(r"[a-z0-9$%]+")
# Words common in questions that carry no policy content.
SENTENCE_STOP = frozenset({"much", "many", "get", "need", "allow", "able", "someone"})


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def stem(token: str) -> str:
    """Light suffix stripping so 'refunds'/'refund' and 'paid'/'paying' meet."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def terms(text: str) -> list[str]:
    return [stem(t) for t in tokens(text)]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [terms(d) for d in documents]
        self.k1, self.b = k1, b
        self.avg_len = sum(map(len, self.docs)) / max(1, len(self.docs))
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(doc) for doc in self.docs]

    def scores(self, query: str) -> list[float]:
        q = terms(query)
        out = []
        for tf, doc in zip(self.tf, self.docs, strict=True):
            norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len)
            out.append(
                sum(
                    self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                    for t in q
                    if t in tf
                )
            )
        return out


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> Vector: ...


class HashEmbedder:
    """Deterministic, offline embedding: signed feature hashing of word unigrams, bigrams and
    character trigrams, L2-normalised. It is lexical, not semantic; it exists so the whole
    pipeline runs with no model download. Use `bge-small` for real semantic retrieval."""

    name = "hash"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = terms(text)
        feats = list(words) + [f"{a}_{b}" for a, b in itertools.pairwise(words)]
        for w in words:
            padded = f"#{w}#"
            feats += [padded[i : i + 3] for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> Vector:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        normalised: Vector = (out / np.where(norms == 0, 1, norms)).astype(np.float32)
        return normalised


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 through fastembed (ONNX). Needs the `embeddings` extra."""

    name = "bge-small"
    dim = 384

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed(self, texts: list[str]) -> Vector:
        vectors: Vector = np.asarray(list(self.model.embed(texts)), dtype=np.float32)
        return vectors


def make_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "bge-small":
        return FastEmbedder()
    raise ValueError(f"unknown RAG_EMBEDDER '{name}' (hash|bge-small)")


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: score = sum of 1 / (k + rank) over each ranking."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused
