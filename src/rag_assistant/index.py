"""Vector index with idempotent re-indexing.

`sync` compares each document's content hash with what is stored: unchanged documents are
skipped, changed ones have their chunks replaced, removed ones are deleted. The local index
is SQLite plus brute-force cosine (a few hundred chunks need nothing more); the Postgres
index stores vectors in pgvector and searches with an HNSW index.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rag_assistant.documents import Chunk, Document, chunk_documents
from rag_assistant.search import Embedder, Vector


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks: int = 0


class VectorIndex(Protocol):
    def sync(self, docs: list[Document], strategy: str, embedder: Embedder) -> SyncReport: ...

    def chunks(self) -> list[Chunk]: ...

    def dense(self, query: Vector, k: int) -> list[tuple[int, float]]: ...

    def vectors(self, positions: list[int]) -> Vector: ...


def _plan(
    stored: dict[str, str], docs: list[Document]
) -> tuple[list[Document], list[str], SyncReport]:
    report = SyncReport()
    current = {d.doc_id: d for d in docs}
    changed = []
    for doc in docs:
        if doc.doc_id not in stored:
            report.added += 1
            changed.append(doc)
        elif stored[doc.doc_id] != doc.content_hash:
            report.updated += 1
            changed.append(doc)
        else:
            report.unchanged += 1
    removed = [d for d in stored if d not in current]
    report.removed = len(removed)
    return changed, removed, report


class LocalIndex:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(
            "CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, content_hash TEXT);"
            "CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT, "
            "meta TEXT, embedding BLOB);"
        )
        self._cache: tuple[list[Chunk], Vector] | None = None

    def sync(self, docs: list[Document], strategy: str, embedder: Embedder) -> SyncReport:
        stored = dict(self.conn.execute("SELECT doc_id, content_hash FROM docs").fetchall())
        changed, removed, report = _plan(stored, docs)
        new_chunks = chunk_documents(changed, strategy)
        vectors = embedder.embed([c.text for c in new_chunks]) if new_chunks else None
        with self.conn:
            for doc_id in [d.doc_id for d in changed] + removed:
                self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                self.conn.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
            for i, chunk in enumerate(new_chunks):
                assert vectors is not None
                self.conn.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        json.dumps(asdict(chunk)),
                        vectors[i].astype(np.float32).tobytes(),
                    ),
                )
            for doc in changed:
                self.conn.execute("INSERT INTO docs VALUES (?, ?)", (doc.doc_id, doc.content_hash))
        self._cache = None
        report.chunks = len(self.chunks())
        return report

    def _load(self) -> tuple[list[Chunk], Vector]:
        if self._cache is None:
            rows = self.conn.execute(
                "SELECT meta, embedding FROM chunks ORDER BY chunk_id"
            ).fetchall()
            chunks = [Chunk(**json.loads(meta)) for meta, _ in rows]
            matrix = (
                np.vstack([np.frombuffer(e, dtype=np.float32) for _, e in rows])
                if rows
                else np.zeros((0, 1), dtype=np.float32)
            )
            self._cache = (chunks, matrix)
        return self._cache

    def chunks(self) -> list[Chunk]:
        return self._load()[0]

    def dense(self, query: Vector, k: int) -> list[tuple[int, float]]:
        _, matrix = self._load()
        if not len(matrix):
            return []
        sims = matrix @ query
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def vectors(self, positions: list[int]) -> Vector:
        return self._load()[1][positions]


class PgVectorIndex:
    """Postgres + pgvector. Needs the `postgres` extra and a pgvector-enabled database."""

    def __init__(self, url: str, dim: int, namespace: str) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn: Any = psycopg.connect(
            url.replace("postgresql+psycopg://", "postgresql://"), autocommit=True
        )
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.ns = namespace
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_docs (ns TEXT, doc_id TEXT, content_hash TEXT, "
            "PRIMARY KEY (ns, doc_id))"
        )
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS rag_chunks (ns TEXT, chunk_id TEXT, doc_id TEXT, "
            f"meta JSONB, embedding vector({dim}), PRIMARY KEY (ns, chunk_id))"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS rag_chunks_hnsw ON rag_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        self._chunks: list[Chunk] | None = None
        self._ids: list[str] = []

    def sync(self, docs: list[Document], strategy: str, embedder: Embedder) -> SyncReport:
        rows = self.conn.execute(
            "SELECT doc_id, content_hash FROM rag_docs WHERE ns = %s", (self.ns,)
        ).fetchall()
        changed, removed, report = _plan(dict(rows), docs)
        new_chunks = chunk_documents(changed, strategy)
        vectors = embedder.embed([c.text for c in new_chunks]) if new_chunks else None
        with self.conn.transaction():
            for doc_id in [d.doc_id for d in changed] + removed:
                self.conn.execute(
                    "DELETE FROM rag_chunks WHERE ns = %s AND doc_id = %s", (self.ns, doc_id)
                )
                self.conn.execute(
                    "DELETE FROM rag_docs WHERE ns = %s AND doc_id = %s", (self.ns, doc_id)
                )
            for i, chunk in enumerate(new_chunks):
                assert vectors is not None
                self.conn.execute(
                    "INSERT INTO rag_chunks VALUES (%s, %s, %s, %s, %s)",
                    (self.ns, chunk.chunk_id, chunk.doc_id, json.dumps(asdict(chunk)), vectors[i]),
                )
            for doc in changed:
                self.conn.execute(
                    "INSERT INTO rag_docs VALUES (%s, %s, %s)",
                    (self.ns, doc.doc_id, doc.content_hash),
                )
        self._chunks = None
        report.chunks = len(self.chunks())
        return report

    def chunks(self) -> list[Chunk]:
        if self._chunks is None:
            rows = self.conn.execute(
                "SELECT chunk_id, meta FROM rag_chunks WHERE ns = %s ORDER BY chunk_id", (self.ns,)
            ).fetchall()
            self._ids = [r[0] for r in rows]
            self._chunks = [Chunk(**r[1]) for r in rows]
        return self._chunks

    def dense(self, query: Vector, k: int) -> list[tuple[int, float]]:
        self.chunks()
        position = {cid: i for i, cid in enumerate(self._ids)}
        rows = self.conn.execute(
            "SELECT chunk_id, 1 - (embedding <=> %s) FROM rag_chunks WHERE ns = %s "
            "ORDER BY embedding <=> %s LIMIT %s",
            (query, self.ns, query, k),
        ).fetchall()
        return [(position[cid], float(sim)) for cid, sim in rows]

    def vectors(self, positions: list[int]) -> Vector:
        self.chunks()
        ids = [self._ids[p] for p in positions]
        rows = dict(
            self.conn.execute(
                "SELECT chunk_id, embedding FROM rag_chunks WHERE ns = %s AND chunk_id = ANY(%s)",
                (self.ns, ids),
            ).fetchall()
        )
        def as_array(value: Any) -> Vector:
            # pgvector returns numpy arrays or its own Vector type depending on version.
            raw = value.to_numpy() if hasattr(value, "to_numpy") else value
            return np.asarray(raw, dtype=np.float32)

        return np.vstack([as_array(rows[i]) for i in ids])


def open_index(
    database_url: str, state_dir: Path, embedder: Embedder, strategy: str
) -> VectorIndex:
    namespace = f"{embedder.name}-{strategy}"
    if database_url.startswith("postgresql"):
        return PgVectorIndex(database_url, embedder.dim, namespace)
    return LocalIndex(state_dir / f"index-{namespace}.sqlite")
