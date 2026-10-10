from __future__ import annotations

from pathlib import Path

import numpy as np

from rag_assistant import corpus
from rag_assistant.config import PROJECT_ROOT
from rag_assistant.documents import fixed_chunks, load_documents, parse_document, section_chunks
from rag_assistant.search import BM25, HashEmbedder, rrf, terms


def test_committed_corpus_matches_generator(tmp_path: Path) -> None:
    corpus.write_corpus(tmp_path)
    for path in sorted(tmp_path.glob("*.md")):
        committed = PROJECT_ROOT / "corpus" / path.name
        assert committed.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")
    assert len(list((PROJECT_ROOT / "corpus").glob("*.md"))) == len(corpus.DOCS)


def test_parse_round_trip(tmp_path: Path) -> None:
    corpus.write_corpus(tmp_path)
    doc = parse_document(tmp_path / "refunds.md")
    assert doc.title == "Refund Policy" and not doc.superseded
    assert [s for s, _ in doc.sections] == [
        "Refund window",
        "Processing time",
        "Partial refunds",
        "Exceptions",
    ]
    old = parse_document(tmp_path / "refunds-2024.md")
    assert old.superseded and old.superseded_by == "refunds"


def test_section_chunks_keep_headings_and_context(tmp_path: Path) -> None:
    corpus.write_corpus(tmp_path)
    doc = parse_document(tmp_path / "late-fees.md")
    chunks = section_chunks(doc)
    assert [c.section for c in chunks] == ["Late fee amount", "Limits", "Hardship waiver"]
    assert chunks[0].text.startswith("Late Fees > Late fee amount: ")


def test_long_sections_split_with_one_sentence_overlap(tmp_path: Path) -> None:
    corpus.write_corpus(tmp_path)
    doc = parse_document(tmp_path / "refunds.md")
    chunks = section_chunks(doc, max_words=12)
    window = [c for c in chunks if c.section == "Exceptions"]
    assert len(window) >= 2
    last_sentence = window[0].text.split(": ", 1)[1].split(". ")[-1]
    assert last_sentence.rstrip(".") in window[1].text


def test_fixed_chunks_ignore_section_boundaries(tmp_path: Path) -> None:
    corpus.write_corpus(tmp_path)
    doc = parse_document(tmp_path / "payment-plans.md")
    assert len(fixed_chunks(doc, words=40, overlap=10)) < len(section_chunks(doc)) + 2
    assert all(len(c.text.split(": ", 1)[1].split()) <= 40 for c in fixed_chunks(doc))


def test_chunk_ids_are_stable(tmp_path: Path) -> None:
    corpus.write_corpus(tmp_path)
    a = [c.chunk_id for d in load_documents(tmp_path) for c in section_chunks(d)]
    b = [c.chunk_id for d in load_documents(tmp_path) for c in section_chunks(d)]
    assert a == b and len(set(a)) == len(a)


def test_terms_stem_and_drop_stopwords() -> None:
    assert terms("The refunds are paid") == ["refund", "paid"]


def test_bm25_prefers_matching_documents() -> None:
    bm = BM25(["refund within 14 days", "password length rules", "refund processing time"])
    scores = bm.scores("refund window days")
    assert scores[0] > scores[2] > scores[1] == 0


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    e = HashEmbedder(dim=256)
    a, b = e.embed(["late fee amount", "late fee amount"])
    assert np.allclose(a, b) and abs(float(np.linalg.norm(a)) - 1) < 1e-5
    near, far = e.embed(["late fees charged", "screen lock timeout"])
    assert float(a @ near) > float(a @ far)


def test_rrf() -> None:
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert fused[1] == 1 / 61 + 1 / 62
    assert max(fused, key=fused.__getitem__) == 1
