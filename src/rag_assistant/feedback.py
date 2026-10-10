"""Answer log and helpful / not helpful feedback, kept for review and future evaluation."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_assistant.answer import Answer


class FeedbackStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            "CREATE TABLE IF NOT EXISTS answers (id TEXT PRIMARY KEY, question TEXT, answer TEXT,"
            " refused INTEGER, reason TEXT, sources TEXT, channel TEXT, user_id TEXT,"
            " created_at TEXT);"
            "CREATE TABLE IF NOT EXISTS feedback (answer_id TEXT REFERENCES answers(id),"
            " user_id TEXT, helpful INTEGER, created_at TEXT, PRIMARY KEY (answer_id, user_id));"
        )

    def log_answer(self, answer: Answer, channel: str, user_id: str) -> str:
        answer_id = uuid.uuid4().hex[:12]
        sources = [
            {
                "n": n,
                "doc_id": answer.sources[n - 1].chunk.doc_id,
                "section": answer.sources[n - 1].chunk.section,
            }
            for n in answer.cited
        ]
        with self.conn:
            self.conn.execute(
                "INSERT INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    answer_id,
                    answer.question,
                    answer.text,
                    int(answer.refused),
                    answer.reason,
                    json.dumps(sources),
                    channel,
                    user_id,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return answer_id

    def record(self, answer_id: str, user_id: str, helpful: bool) -> None:
        """One vote per user per answer; a second click replaces the first."""
        if not self.conn.execute("SELECT 1 FROM answers WHERE id = ?", (answer_id,)).fetchone():
            raise KeyError(answer_id)
        with self.conn:
            self.conn.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?) ON CONFLICT(answer_id, user_id) "
                "DO UPDATE SET helpful = excluded.helpful, created_at = excluded.created_at",
                (answer_id, user_id, int(helpful), datetime.now(UTC).isoformat()),
            )

    def summary(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*) AS votes, COALESCE(SUM(helpful), 0) AS helpful FROM feedback"
        ).fetchone()
        unhelpful = self.conn.execute(
            "SELECT a.question, a.answer FROM feedback f JOIN answers a ON a.id = f.answer_id "
            "WHERE f.helpful = 0 ORDER BY f.created_at DESC LIMIT 20"
        ).fetchall()
        return {
            "votes": row["votes"],
            "helpful": row["helpful"],
            "not_helpful": [dict(r) for r in unhelpful],
        }
