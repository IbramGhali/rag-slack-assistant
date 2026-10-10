from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rag_assistant import corpus
from rag_assistant.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline settings with corpus and state inside tmp_path."""
    s = replace(
        Settings(),
        corpus_dir=tmp_path / "corpus",
        state_dir=tmp_path / "state",
        baseline_path=tmp_path / "eval" / "baseline.json",
    )
    corpus.write_corpus(s.corpus_dir)
    return s
