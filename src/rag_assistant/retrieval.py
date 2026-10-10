"""Hybrid retrieval: BM25 and dense rankings fused with reciprocal rank fusion, superseded
documents demoted (not deleted), optional cross-encoder re-ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rag_assistant.documents import Chunk
from rag_assistant.index import VectorIndex
from rag_assistant.search import BM25, Embedder, rrf

CANDIDATES = 20


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    cosine: float  # dense similarity to the query, used as the evidence signal


class Retriever:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        method: str = "hybrid",
        superseded_penalty: float = 0.5,
        reranker: str = "none",
    ) -> None:
        if method not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"unknown retrieval method '{method}'")
        self.index = index
        self.embedder = embedder
        self.method = method
        self.penalty = superseded_penalty
        self.chunk_list = index.chunks()
        self.bm25 = BM25([c.text for c in self.chunk_list])
        self.cross: Any = None
        if reranker == "cross-encoder":
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self.cross = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")

    def search(self, query: str, k: int = 5) -> list[Hit]:
        qvec = self.embedder.embed([query])[0]
        bm = self.bm25.scores(query)
        bm_rank = [i for i in np.argsort(-np.asarray(bm))[:CANDIDATES] if bm[i] > 0]
        dense = self.index.dense(qvec, CANDIDATES)
        dense_rank = [i for i, _ in dense]

        if self.method == "bm25":
            scores = {i: bm[i] for i in bm_rank}
        elif self.method == "dense":
            scores = {i: s for i, s in dense}
        else:
            scores = rrf([bm_rank, dense_rank])

        # Demote, never drop: an old policy can still answer "what used to apply".
        for i in scores:
            if self.chunk_list[i].superseded:
                scores[i] *= self.penalty if scores[i] > 0 else 1 / self.penalty

        ranked = sorted(scores, key=lambda i: -scores[i])[:CANDIDATES]
        if self.cross is not None and ranked:
            texts = [self.chunk_list[i].text for i in ranked]
            rerank = list(self.cross.rerank(query, texts))
            adjusted = [
                s - (5.0 if self.chunk_list[i].superseded else 0.0)
                for i, s in zip(ranked, rerank, strict=True)
            ]
            ranked = [i for _, i in sorted(zip(adjusted, ranked, strict=True), key=lambda p: -p[0])]
        top = ranked[:k]
        if not top:
            return []
        cosines = self.index.vectors(top) @ qvec
        return [
            Hit(self.chunk_list[i], float(scores[i]), float(c))
            for i, c in zip(top, cosines, strict=True)
        ]
