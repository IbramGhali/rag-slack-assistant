"""Answer generation and groundedness judging, behind one interface.

`fake` (default) is deterministic and offline: it answers extractively with the best-matching
sentence from the numbered sources, and judges groundedness by term overlap. `anthropic`
uses Claude with structured outputs for both.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from rag_assistant.search import SENTENCE_STOP, terms

REFUSAL = "I can't find that in the policies."


@dataclass(frozen=True)
class Draft:
    answer: str
    citations: list[int]
    refused: bool


class LLMError(RuntimeError):
    pass


class Generator(Protocol):
    name: str

    def answer(self, question: str, sources: list[str]) -> Draft: ...

    def supported(self, claim: str, source: str) -> bool: ...


def _sentences(text: str) -> list[str]:
    body = text.split(": ", 1)[1] if ": " in text else text
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]


class FakeGenerator:
    name = "fake"

    def answer(self, question: str, sources: list[str]) -> Draft:
        """Extractive stand-in for a model given ranked context: answers by quoting source [1]
        (a policy section is one to three sentences) unless a later source matches the
        question clearly better, by two or more terms."""
        q = set(terms(question)) - SENTENCE_STOP
        scored: list[tuple[int, int, str]] = []  # (source_no, overlap, sentence)
        for n, source in enumerate(sources, start=1):
            heading = set(terms(source.split(": ", 1)[0]))
            for sentence in _sentences(source):
                scored.append((n, len(q & (set(terms(sentence)) | heading)), sentence))
        if not scored or max(s[1] for s in scored) == 0:
            return Draft(REFUSAL, [], True)
        best_by_source = {n: max(s[1] for s in scored if s[0] == n) for n, _, _ in scored}
        chosen = 1
        for n, overlap in best_by_source.items():
            if overlap >= best_by_source[chosen] + 2:
                chosen = n
        section = " ".join(s[2] for s in scored if s[0] == chosen)
        return Draft(f"{section} [{chosen}]", [chosen], False)

    def supported(self, claim: str, source: str) -> bool:
        claim_terms = set(terms(claim))
        if not claim_terms:
            return True
        return len(claim_terms & set(terms(source))) / len(claim_terms) >= 0.6


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "refused": {"type": "boolean"},
    },
    "required": ["answer", "citations", "refused"],
    "additionalProperties": False,
}
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}},
    "required": ["supported"],
    "additionalProperties": False,
}
SYSTEM = (
    "You answer staff questions using only the numbered policy sources provided. "
    "Answer in one or two sentences and put the source number in square brackets after each "
    "claim, like [2]. If the sources do not answer the question, set refused to true and "
    f"answer exactly: {REFUSAL} Prefer the current policy when an older version conflicts. "
    "Never use outside knowledge."
)


class ClaudeGenerator:
    name = "anthropic"

    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise LLMError("install the anthropic extra: uv sync --extra anthropic") from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=60.0, max_retries=2)
        self.model = model

    def _call(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        anthropic = self._anthropic
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("connection error") from exc
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"unusable response: {response.stop_reason}")
        text = next(b.text for b in response.content if b.type == "text")
        data: dict[str, Any] = json.loads(text)
        return data

    def answer(self, question: str, sources: list[str]) -> Draft:
        listing = "\n".join(f"[{n}] {s}" for n, s in enumerate(sources, start=1))
        data = self._call(SYSTEM, f"Sources:\n{listing}\n\nQuestion: {question}", ANSWER_SCHEMA)
        return Draft(
            str(data["answer"]), [int(c) for c in data["citations"]], bool(data["refused"])
        )

    def supported(self, claim: str, source: str) -> bool:
        data = self._call(
            "Decide whether the source fully supports the claim. Judge only "
            "what the source states.",
            f"Source: {source}\n\nClaim: {claim}",
            JUDGE_SCHEMA,
        )
        return bool(data["supported"])


def make_generator(provider: str, model: str) -> Generator:
    if provider == "fake":
        return FakeGenerator()
    if provider == "anthropic":
        return ClaudeGenerator(model)
    raise ValueError(f"unknown RAG_LLM_PROVIDER '{provider}' (fake|anthropic)")
