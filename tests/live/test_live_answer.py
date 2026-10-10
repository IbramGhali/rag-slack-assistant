"""Real Claude answers on a few golden questions. Excluded by default; costs money.

Run with: RAG_LIVE_TESTS=1 uv run --extra anthropic pytest -m live
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from rag_assistant.app import build_assistant
from rag_assistant.config import Settings


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("RAG_LIVE_TESTS") != "1", reason="set RAG_LIVE_TESTS=1")
def test_claude_answers_with_valid_citations(settings: Settings) -> None:
    assistant = build_assistant(replace(settings, llm_provider="anthropic"))
    answer = assistant.ask("How much is the late fee?")
    assert not answer.refused and "$25" in answer.text and answer.cited
    assert assistant.ask("Can I bring my dog to the office?").refused
