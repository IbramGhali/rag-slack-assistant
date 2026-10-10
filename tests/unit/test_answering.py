from __future__ import annotations

from rag_assistant.answer import Assistant
from rag_assistant.documents import Chunk
from rag_assistant.evaluation import Question, correct, split_claims
from rag_assistant.llm import REFUSAL, Draft, FakeGenerator, LLMError
from rag_assistant.retrieval import Hit


def _hit(text: str, cosine: float = 0.9, superseded: bool = False) -> Hit:
    chunk = Chunk(
        "c#1", "doc", "Doc", "Section", f"Doc > Section: {text}", "2025-01-01", superseded
    )
    return Hit(chunk, 1.0, cosine)


class _Retriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.hits[:k]


class _Gen:
    name = "stub"

    def __init__(self, draft: Draft | Exception) -> None:
        self.draft = draft

    def answer(self, question: str, sources: list[str]) -> Draft:
        if isinstance(self.draft, Exception):
            raise self.draft
        return self.draft

    def supported(self, claim: str, source: str) -> bool:
        return True


def _assistant(hits: list[Hit], gen: object, threshold: float = 0.5) -> Assistant:
    return Assistant(_Retriever(hits), gen, threshold)  # type: ignore[arg-type]


def test_weak_evidence_refuses_without_calling_the_model() -> None:
    gen = _Gen(LLMError("must not be called"))
    answer = _assistant([_hit("x", cosine=0.2)], gen).ask("q")
    assert answer.refused and answer.reason == "weak_evidence" and answer.text == REFUSAL


def test_invalid_citations_are_removed_and_valid_ones_kept() -> None:
    gen = _Gen(Draft("Fee is $25 [1] and also [7].", [1, 7], False))
    answer = _assistant([_hit("A fee of $25.")], gen).ask("fee?")
    assert not answer.refused and answer.cited == [1]
    assert "[7]" not in answer.text and "[1]" in answer.text


def test_answer_with_no_valid_citation_is_refused() -> None:
    answer = _assistant([_hit("x")], _Gen(Draft("Made up [9].", [9], False))).ask("q")
    assert answer.refused and answer.reason == "no_valid_citation"


def test_model_failure_fails_safe() -> None:
    answer = _assistant([_hit("x")], _Gen(LLMError("timeout"))).ask("q")
    assert answer.refused and answer.reason == "llm_error"


def test_fake_generator_quotes_top_source_with_citation() -> None:
    draft = FakeGenerator().answer(
        "late fee amount",
        [
            "Late Fees > Late fee amount: A late fee of $25 is charged.",
            "Late Fees > Limits: No more than 3 late fees.",
        ],
    )
    assert draft.citations == [1] and draft.answer.endswith("[1]") and "$25" in draft.answer


def test_fake_generator_switches_source_only_when_clearly_better() -> None:
    draft = FakeGenerator().answer(
        "parental leave primary carer weeks",
        [
            "Sick Leave > Entitlement: Staff receive 10 days of paid sick leave.",
            "Parental Leave > Primary carer: Primary carers get 16 weeks of paid parental leave.",
        ],
    )
    assert draft.citations == [2]


def test_fake_generator_refuses_with_no_overlap() -> None:
    assert FakeGenerator().answer("dog office", ["Doc > S: Passwords are long."]).refused


def test_uncited_sentences_inherit_the_next_citation() -> None:
    assert split_claims("First fact. Second fact. [2]") == [
        ("First fact.", [2]),
        ("Second fact.", [2]),
    ]


def test_superseded_answers_with_the_old_value_are_wrong() -> None:
    q = Question("s", "superseded", "?", "refunds", "Refund window", ("14 days",), ("7 days",))
    good = Assistant.__new__(Assistant)  # only the answer text matters here
    del good
    from rag_assistant.answer import Answer

    assert correct(Answer("?", "Within 14 days. [1]", False, "answered", 1.0), q)
    assert not correct(
        Answer("?", "Within 14 days, previously 7 days. [1]", False, "answered", 1.0), q
    )
    assert not correct(Answer("?", REFUSAL, True, "weak_evidence", 0.1), q)
