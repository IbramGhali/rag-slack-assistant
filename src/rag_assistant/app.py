"""Wiring: corpus, index, retriever and assistant from settings."""

from __future__ import annotations

from dataclasses import dataclass

from rag_assistant.answer import Assistant
from rag_assistant.config import Settings
from rag_assistant.documents import load_documents
from rag_assistant.index import SyncReport, VectorIndex, open_index
from rag_assistant.llm import make_generator
from rag_assistant.retrieval import Retriever
from rag_assistant.search import Embedder, make_embedder


@dataclass
class Components:
    index: VectorIndex
    embedder: Embedder
    report: SyncReport


def build_index(
    settings: Settings, embedder: Embedder | None = None, chunking: str | None = None
) -> Components:
    embedder = embedder or make_embedder(settings.embedder)
    strategy = chunking or settings.chunking
    index = open_index(settings.database_url, settings.state_dir, embedder, strategy)
    report = index.sync(load_documents(settings.corpus_dir), strategy, embedder)
    return Components(index, embedder, report)


def build_assistant(
    settings: Settings,
    components: Components | None = None,
    method: str | None = None,
    reranker: str | None = None,
) -> Assistant:
    c = components or build_index(settings)
    retriever = Retriever(
        c.index,
        c.embedder,
        method or settings.retrieval,
        settings.superseded_penalty,
        reranker or settings.reranker,
    )
    return Assistant(
        retriever,
        make_generator(settings.llm_provider, settings.answer_model),
        settings.threshold,
        settings.top_k,
    )
