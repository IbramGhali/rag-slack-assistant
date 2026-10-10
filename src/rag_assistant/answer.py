"""Answering: retrieve, refuse below the evidence threshold, generate with numbered sources,
then check every citation in code before anything is shown."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_assistant.llm import REFUSAL, Generator, LLMError
from rag_assistant.retrieval import Hit, Retriever

CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    refused: bool
    reason: str  # answered | weak_evidence | model_refused | no_valid_citation | llm_error
    evidence: float
    sources: list[Hit] = field(default_factory=list)  # every retrieved source, numbered from 1
    cited: list[int] = field(default_factory=list)  # 1-based numbers that survived checking


class Assistant:
    def __init__(
        self, retriever: Retriever, generator: Generator, threshold: float, top_k: int = 5
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.threshold = threshold
        self.top_k = top_k

    def ask(self, question: str) -> Answer:
        hits = self.retriever.search(question, self.top_k)
        evidence = max((h.cosine for h in hits), default=0.0)
        if evidence < self.threshold:
            # Refuse before calling the model: weak retrieval is where hallucination starts.
            return Answer(question, REFUSAL, True, "weak_evidence", evidence, hits)
        try:
            draft = self.generator.answer(question, [h.chunk.text for h in hits])
        except LLMError:
            return Answer(question, REFUSAL, True, "llm_error", evidence, hits)
        if draft.refused:
            return Answer(question, REFUSAL, True, "model_refused", evidence, hits)
        # Post-check: a citation must point at a source that was actually retrieved.
        mentioned = {int(n) for n in CITATION.findall(draft.answer)} | set(draft.citations)
        valid = sorted(n for n in mentioned if 1 <= n <= len(hits))
        if not valid:
            return Answer(question, REFUSAL, True, "no_valid_citation", evidence, hits)
        text = CITATION.sub(
            lambda m: m.group(0) if int(m.group(1)) in valid else "", draft.answer
        ).strip()
        return Answer(question, text, False, "answered", evidence, hits, valid)
