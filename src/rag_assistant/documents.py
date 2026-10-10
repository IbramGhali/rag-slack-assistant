"""Parsing and chunking.

Section-aware chunking keeps each heading's text together and prefixes it with the document
title and heading, so a chunk carries its own context. Long sections are split on sentence
boundaries with one sentence of overlap. Fixed windows are kept as a baseline for comparison.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    area: str
    effective: str
    superseded_by: str | None
    sections: tuple[tuple[str, str], ...]
    content_hash: str

    @property
    def superseded(self) -> bool:
        return bool(self.superseded_by)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str  # what is embedded and shown as a source
    effective: str
    superseded: bool


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    _, front, body = raw.split("---", 2)
    meta = yaml.safe_load(front)
    sections: list[tuple[str, str]] = []
    heading: str | None = None
    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if heading:
                sections.append((heading, " ".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        elif heading and line.strip() and not line.startswith("# "):
            lines.append(line.strip())
    if heading:
        sections.append((heading, " ".join(lines).strip()))
    return Document(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        area=str(meta["area"]),
        effective=str(meta["effective"]),
        superseded_by=meta.get("superseded_by") or None,
        sections=tuple(sections),
        content_hash=hashlib.sha256(raw.encode()).hexdigest(),
    )


def load_documents(corpus_dir: Path) -> list[Document]:
    return [parse_document(p) for p in sorted(corpus_dir.glob("*.md"))]


def _chunk(doc: Document, section: str, text: str, n: int) -> Chunk:
    body = f"{doc.title} > {section}: {text}"
    digest = hashlib.sha256(f"{doc.doc_id}|{section}|{n}|{body}".encode()).hexdigest()[:16]
    return Chunk(
        f"{doc.doc_id}#{digest}",
        doc.doc_id,
        doc.title,
        section,
        body,
        doc.effective,
        doc.superseded,
    )


def section_chunks(doc: Document, max_words: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, text in doc.sections:
        sentences = SENTENCE.split(text)
        window: list[str] = []
        for sentence in sentences:
            if window and len(" ".join([*window, sentence]).split()) > max_words:
                chunks.append(_chunk(doc, section, " ".join(window), len(chunks)))
                window = window[-1:]  # one sentence of overlap
            window.append(sentence)
        if window:
            chunks.append(_chunk(doc, section, " ".join(window), len(chunks)))
    return chunks


def fixed_chunks(doc: Document, words: int = 40, overlap: int = 10) -> list[Chunk]:
    """Baseline: fixed word windows over the whole document, ignoring headings."""
    flat = [(section, word) for section, text in doc.sections for word in text.split()]
    chunks: list[Chunk] = []
    step = words - overlap
    for start in range(0, max(1, len(flat) - overlap), step):
        window = flat[start : start + words]
        if not window:
            break
        text = " ".join(w for _, w in window)
        digest = hashlib.sha256(f"{doc.doc_id}|{start}|{text}".encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                f"{doc.doc_id}#{digest}",
                doc.doc_id,
                doc.title,
                window[0][0],
                f"{doc.title}: {text}",
                doc.effective,
                doc.superseded,
            )
        )
    return chunks


def chunk_documents(docs: list[Document], strategy: str) -> list[Chunk]:
    fn = section_chunks if strategy == "section" else fixed_chunks
    return [c for doc in docs for c in fn(doc)]
