"""Golden-set evaluation with the same regression floors as CI."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rag_assistant import evaluation
from rag_assistant.app import build_assistant, build_index
from rag_assistant.config import Settings
from rag_assistant.llm import FakeGenerator


def _run(settings: Settings) -> tuple[dict[str, float], dict[str, object]]:
    questions = evaluation.load_golden(settings.golden_path)
    components = build_index(settings)
    assistant = build_assistant(settings, components)
    retrieval = evaluation.evaluate_retrieval(assistant.retriever, questions)
    gen = evaluation.evaluate_generation(assistant, FakeGenerator(), questions).as_dict()
    return {
        "hybrid_section_mrr": retrieval["overall"]["mrr"],
        **{
            k: gen[k]
            for k in ("answer_accuracy", "refusal_accuracy_unanswerable", "citation_validity")
        },
    }, gen


def _check(settings: Settings, key: str) -> dict[str, object]:
    current, gen = _run(settings)
    baseline = json.loads(Settings().baseline_path.read_text("utf-8"))[key]
    for name, floor in baseline.items():
        assert current[name] >= floor - 0.02, f"{name}: {current[name]} < {floor}"
    return gen


def test_golden_set_has_sixty_questions() -> None:
    qs = evaluation.load_golden(Settings().golden_path)
    counts = {
        t: sum(q.type == t for q in qs)
        for t in ("direct", "paraphrase", "unanswerable", "superseded")
    }
    assert counts == {"direct": 20, "paraphrase": 20, "unanswerable": 10, "superseded": 10}


def test_offline_config_meets_baseline(settings: Settings) -> None:
    gen = _check(settings, "hash")
    assert gen["citation_validity"] == 1.0


@pytest.mark.semantic
def test_semantic_config_meets_baseline(settings: Settings) -> None:
    gen = _check(
        replace(settings, embedder="bge-small", reranker="cross-encoder"), "bge-small+rerank"
    )
    assert gen["refusal_accuracy_unanswerable"] >= 0.9
