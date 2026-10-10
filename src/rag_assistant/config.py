"""Settings from environment variables (and an optional .env). Defaults run offline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


@dataclass(frozen=True)
class Settings:
    embedder: str = "hash"  # hash (offline, deterministic) | bge-small (embeddings extra)
    retrieval: str = "hybrid"  # bm25 | dense | hybrid
    reranker: str = "none"  # none | cross-encoder (embeddings extra)
    chunking: str = "section"  # section | fixed
    top_k: int = 5
    evidence_threshold: float = 0.0  # 0 means "use the default for the embedder"
    superseded_penalty: float = 0.5
    llm_provider: str = "fake"
    answer_model: str = "claude-opus-5-5"
    database_url: str = ""  # empty: local SQLite index; postgres URL: pgvector
    corpus_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "corpus")
    state_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ".rag")
    golden_path: Path = field(default_factory=lambda: PROJECT_ROOT / "golden" / "questions.yaml")
    baseline_path: Path = field(default_factory=lambda: PROJECT_ROOT / "eval" / "baseline.json")
    slack_bot_token: str = ""
    slack_app_token: str = ""

    @property
    def threshold(self) -> float:
        if self.evidence_threshold:
            return self.evidence_threshold
        # Chosen from the sweep in the README (`rag eval --sweep`): cosine of the best chunk.
        return {"hash": 0.25, "bge-small": 0.65}.get(self.embedder, 0.3)

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv(PROJECT_ROOT / ".env")
        d = cls()
        return cls(
            embedder=_env("RAG_EMBEDDER", d.embedder),
            retrieval=_env("RAG_RETRIEVAL", d.retrieval),
            reranker=_env("RAG_RERANKER", d.reranker),
            chunking=_env("RAG_CHUNKING", d.chunking),
            top_k=int(_env("RAG_TOP_K", str(d.top_k))),
            evidence_threshold=float(_env("RAG_EVIDENCE_THRESHOLD", "0")),
            superseded_penalty=float(_env("RAG_SUPERSEDED_PENALTY", str(d.superseded_penalty))),
            llm_provider=_env("RAG_LLM_PROVIDER", d.llm_provider).lower(),
            answer_model=_env("RAG_ANSWER_MODEL", d.answer_model),
            database_url=_env("RAG_DATABASE_URL", d.database_url),
            corpus_dir=Path(_env("RAG_CORPUS_DIR", str(d.corpus_dir))),
            state_dir=Path(_env("RAG_STATE_DIR", str(d.state_dir))),
            slack_bot_token=_env("SLACK_BOT_TOKEN", ""),
            slack_app_token=_env("SLACK_APP_TOKEN", ""),
        )
