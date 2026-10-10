"""Evaluation on the golden set. Retrieval and generation are measured separately, so a bad
answer can be traced to "never retrieved" or "retrieved but misused"."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rag_assistant.answer import Answer, Assistant
from rag_assistant.llm import Generator
from rag_assistant.retrieval import Retriever

ANSWERABLE = ("direct", "paraphrase", "superseded")


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    doc: str | None = None
    section: str | None = None
    answer_contains: tuple[str, ...] = ()
    wrong: tuple[str, ...] = ()


def load_golden(path: Path) -> list[Question]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Question(
            id=r["id"],
            type=r["type"],
            question=r["question"],
            doc=r.get("doc"),
            section=r.get("section"),
            answer_contains=tuple(r.get("answer_contains", [])),
            wrong=tuple(r.get("wrong", [])),
        )
        for r in rows
    ]


@dataclass
class RetrievalScore:
    n: int = 0
    hit1: int = 0
    hit3: int = 0
    rr: float = 0.0

    def add(self, rank: int | None) -> None:
        self.n += 1
        if rank is not None:
            self.hit1 += rank == 1
            self.hit3 += rank <= 3
            self.rr += 1 / rank

    def as_dict(self) -> dict[str, float]:
        n = max(1, self.n)
        return {
            "n": self.n,
            "hit@1": round(self.hit1 / n, 3),
            "hit@3": round(self.hit3 / n, 3),
            "mrr": round(self.rr / n, 3),
        }


def relevant_ranks(retriever: Retriever, q: Question, k: int = 10) -> tuple[int | None, int | None]:
    """Rank of the first chunk from the expected document, and from the expected section."""
    doc_rank = section_rank = None
    for rank, hit in enumerate(retriever.search(q.question, k), start=1):
        if hit.chunk.doc_id == q.doc:
            doc_rank = doc_rank or rank
            if hit.chunk.section == q.section:
                section_rank = section_rank or rank
    return doc_rank, section_rank


def evaluate_retrieval(retriever: Retriever, questions: list[Question]) -> dict[str, Any]:
    overall, section = RetrievalScore(), RetrievalScore()
    by_type: dict[str, RetrievalScore] = defaultdict(RetrievalScore)
    for q in questions:
        if q.type not in ANSWERABLE:
            continue
        doc_rank, section_rank = relevant_ranks(retriever, q)
        overall.add(doc_rank)
        section.add(section_rank)
        by_type[q.type].add(doc_rank)
    return {
        "overall": overall.as_dict(),
        "section": section.as_dict(),
        "by_type": {t: s.as_dict() for t, s in sorted(by_type.items())},
    }


def correct(answer: Answer, q: Question) -> bool:
    text = answer.text.lower()
    if answer.refused or not all(s.lower() in text for s in q.answer_contains):
        return False
    return not any(w.lower() in text for w in q.wrong)


@dataclass
class GenerationReport:
    by_type: dict[str, dict[str, int]] = field(default_factory=dict)
    citation_valid: int = 0
    citation_total: int = 0
    grounded: int = 0
    grounded_total: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def rate(self, kind: str, key: str) -> float:
        t = self.by_type.get(kind, {})
        return t.get(key, 0) / max(1, t.get("n", 0))

    def as_dict(self) -> dict[str, Any]:
        answerable = [self.by_type.get(t, {}) for t in ANSWERABLE]
        n = sum(t.get("n", 0) for t in answerable)
        return {
            "answer_accuracy": round(sum(t.get("correct", 0) for t in answerable) / max(1, n), 3),
            "by_type": self.by_type,
            "refusal_accuracy_unanswerable": round(self.rate("unanswerable", "refused"), 3),
            "false_refusal_rate_answerable": round(
                sum(t.get("refused", 0) for t in answerable) / max(1, n), 3
            ),
            "citation_validity": round(self.citation_valid / max(1, self.citation_total), 3),
            "groundedness": round(self.grounded / max(1, self.grounded_total), 3),
            "superseded_used_current": round(self.rate("superseded", "used_current"), 3),
        }


def split_claims(text: str) -> list[tuple[str, list[int]]]:
    """Sentences with their citations. A sentence without its own citation is covered by the
    next citation in the answer (a citation at the end of a paragraph covers it)."""
    claims: list[tuple[str, list[int]]] = []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
        cites = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        body = re.sub(r"\s*\[\d+\]", "", sentence).strip()
        if body:
            claims.append((body, cites))
    pending: list[int] = []
    for i in range(len(claims) - 1, -1, -1):
        body, cites = claims[i]
        if cites:
            pending = cites
        else:
            claims[i] = (body, pending)
    return claims


def evaluate_generation(
    assistant: Assistant, judge: Generator, questions: list[Question]
) -> GenerationReport:
    report = GenerationReport()
    for q in questions:
        answer = assistant.ask(q.question)
        stats = report.by_type.setdefault(
            q.type, {"n": 0, "correct": 0, "refused": 0, "used_current": 0}
        )
        stats["n"] += 1
        stats["refused"] += answer.refused
        ok = q.type != "unanswerable" and correct(answer, q)
        stats["correct"] += ok
        cited_docs = {answer.sources[n - 1].chunk.doc_id for n in answer.cited}
        if q.type == "superseded" and q.doc in cited_docs and ok:
            stats["used_current"] += 1
        if not answer.refused:
            report.citation_total += 1
            report.citation_valid += bool(answer.cited) and all(
                1 <= n <= len(answer.sources) for n in answer.cited
            )
            for claim, cites in split_claims(answer.text):
                report.grounded_total += 1
                sources = [
                    answer.sources[n - 1].chunk.text for n in cites if 1 <= n <= len(answer.sources)
                ]
                report.grounded += any(judge.supported(claim, s) for s in sources)
        report.rows.append(
            {
                "id": q.id,
                "type": q.type,
                "refused": answer.refused,
                "reason": answer.reason,
                "correct": ok,
                "evidence": round(answer.evidence, 3),
                "answer": answer.text,
            }
        )
    return report


def sweep(
    assistant: Assistant, questions: list[Question], thresholds: list[float]
) -> list[dict[str, float]]:
    """For each threshold: share of unanswerable questions refused, and of answerable ones
    still answered. Retrieval runs once; only the cut-off moves."""
    evidence = {
        q.id: max(
            (h.cosine for h in assistant.retriever.search(q.question, assistant.top_k)), default=0.0
        )
        for q in questions
    }
    rows = []
    for t in thresholds:
        unans = [q for q in questions if q.type == "unanswerable"]
        ans = [q for q in questions if q.type != "unanswerable"]
        rows.append(
            {
                "threshold": t,
                "unanswerable_refused": round(
                    sum(evidence[q.id] < t for q in unans) / len(unans), 3
                ),
                "answerable_kept": round(sum(evidence[q.id] >= t for q in ans) / len(ans), 3),
            }
        )
    return rows
