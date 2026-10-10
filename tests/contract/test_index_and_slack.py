from __future__ import annotations

import os
from typing import Any

import pytest

from rag_assistant.app import build_assistant, build_index
from rag_assistant.config import Settings
from rag_assistant.feedback import FeedbackStore
from rag_assistant.slack_app import (
    HELPFUL,
    NOT_HELPFUL,
    answer_blocks,
    clean_question,
    create_app,
    handle_feedback,
    handle_question,
)


def test_reindex_is_idempotent_and_tracks_changes(settings: Settings) -> None:
    first = build_index(settings).report
    assert (first.added, first.unchanged) == (31, 0)
    second = build_index(settings).report
    assert (second.added, second.updated, second.unchanged) == (0, 0, 31)
    assert second.chunks == first.chunks

    path = settings.corpus_dir / "late-fees.md"
    path.write_text(path.read_text("utf-8").replace("$25", "$30"), encoding="utf-8")
    (settings.corpus_dir / "travel.md").unlink()
    third = build_index(settings).report
    assert (third.updated, third.removed, third.unchanged) == (1, 1, 29)
    answer = build_assistant(settings).ask("How much is the late fee?")
    assert "$30" in answer.text


def test_superseded_policy_is_demoted_not_deleted(settings: Settings) -> None:
    retriever = build_assistant(settings).retriever
    hits = retriever.search("How many days of annual leave do full-time staff get?", 10)
    docs = [h.chunk.doc_id for h in hits]
    assert docs[0] == "annual-leave"
    assert "annual-leave-2024" not in docs[:1]
    # Still indexed and still findable when someone asks about the old version directly.
    assert "annual-leave-2024" in {c.doc_id for c in retriever.chunk_list}
    old = retriever.search("Annual Leave (2024) entitlement", 10)
    assert "annual-leave-2024" in [h.chunk.doc_id for h in old]


def _feedback_body(answer_id: str, action: str, user: str = "U1") -> dict[str, Any]:
    return {
        "actions": [{"action_id": action, "value": answer_id}],
        "user": {"id": user},
        "channel": {"id": "C1"},
    }


def test_slack_payload_shape_and_feedback(settings: Settings) -> None:
    store = FeedbackStore(settings.state_dir / "feedback.sqlite")
    payload = handle_question(
        build_assistant(settings), store, "<@U0BOT> How much is the late fee?", "U1", "C1"
    )
    blocks = payload["blocks"]
    assert [b["type"] for b in blocks] == ["section", "context", "actions"]
    assert "$25" in blocks[0]["text"]["text"] and "[1]" in blocks[1]["elements"][0]["text"]
    buttons = blocks[2]["elements"]
    assert {b["action_id"] for b in buttons} == {HELPFUL, NOT_HELPFUL}
    answer_id = buttons[0]["value"]

    assert handle_feedback(store, _feedback_body(answer_id, HELPFUL)) == "Thanks, noted."
    handle_feedback(store, _feedback_body(answer_id, NOT_HELPFUL))  # same user changes vote
    summary = store.summary()
    assert summary["votes"] == 1 and summary["helpful"] == 0
    assert summary["not_helpful"][0]["question"] == "How much is the late fee?"
    with pytest.raises(KeyError):
        store.record("missing", "U1", True)


def test_refusal_payload_has_guidance_and_no_sources(settings: Settings) -> None:
    store = FeedbackStore(settings.state_dir / "feedback.sqlite")
    payload = handle_question(build_assistant(settings), store, "Can I bring my dog?", "U1", "C1")
    assert "can't find" in payload["text"]
    assert "No policy covers this" in payload["blocks"][1]["elements"][0]["text"]


def test_superseded_sources_are_labelled() -> None:
    from rag_assistant.answer import Answer
    from rag_assistant.documents import Chunk
    from rag_assistant.retrieval import Hit

    old = Hit(
        Chunk("x", "refunds-2024", "Refund Policy (2024)", "Refund window", "t", "2024", True),
        1.0,
        0.9,
    )
    blocks = answer_blocks(Answer("q", "Old rule [1]", False, "answered", 0.9, [old], [1]), "a1")
    assert "superseded version" in blocks[1]["elements"][0]["text"]


def test_mentions_are_stripped() -> None:
    assert clean_question("<@U123ABC>  What is the refund window?") == "What is the refund window?"


def test_bolt_app_registers_handlers(settings: Settings) -> None:
    app = create_app(
        build_assistant(settings),
        FeedbackStore(settings.state_dir / "feedback.sqlite"),
        "xoxb-test",
        token_verification_enabled=False,
    )
    assert len(app._listeners) >= 4


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("RAG_TEST_DATABASE_URL"), reason="no Postgres")
def test_pgvector_index_sync_and_search(settings: Settings) -> None:
    from dataclasses import replace

    pg = replace(settings, database_url=os.environ["RAG_TEST_DATABASE_URL"])
    first = build_index(pg).report
    assert first.chunks > 0
    assert build_index(pg).report.unchanged == 31
    answer = build_assistant(pg).ask("How much is the late fee?")
    assert "$25" in answer.text and answer.cited
