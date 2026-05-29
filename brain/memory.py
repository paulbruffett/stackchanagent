"""SQLite-backed conversation memory.

One DB per device at ~/.stackchan/memory.db. Three tables:

  turns(id, ts, role, content_json, summarized)
      One row per Claude message (user or assistant). content_json holds
      the original Anthropic content (either a string or a list of
      content blocks — tool_use, tool_result, text). summarized flips
      to 1 once a row has been absorbed into a summary; the agent only
      replays unsummarized rows verbatim.

  summaries(id, ts, summary, span_from, span_to)
      A natural-language summary of turns whose id falls in
      [span_from, span_to]. Replayed to Claude as a leading
      "earlier in our conversation:" message.

  known_facts(id, ts, fact)
      Things the agent has chosen to remember via the `remember_fact`
      tool. Injected as a system block on every turn.

Persists across WS reconnects so the robot remembers prior chats
within and across sessions.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("brain.memory")

DEFAULT_PATH = Path.home() / ".stackchan" / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    content_json TEXT NOT NULL,
    summarized INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_turns_unsummarized
    ON turns(id) WHERE summarized = 0;

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    summary TEXT NOT NULL,
    span_from INTEGER NOT NULL,
    span_to INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS known_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    fact TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Turn:
    id: int
    role: str
    content: Any   # str or list[dict] (Anthropic content blocks)


@dataclass(frozen=True)
class Summary:
    id: int
    summary: str
    span_from: int
    span_to: int


class Memory:
    """Thin SQLite wrapper. Single-process use; not thread-safe — call
    from one event loop. SQLite handles file locking, so multiple
    processes still won't corrupt the DB."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def append_turn(self, role: str, content: Any) -> int:
        """Append a Claude message. Returns the new turn id."""
        content_json = json.dumps(content, default=_anthropic_default)
        cur = self._conn.execute(
            "INSERT INTO turns(ts, role, content_json) VALUES (?, ?, ?)",
            (time.time(), role, content_json),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_unsummarized_turns(self) -> list[Turn]:
        rows = self._conn.execute(
            "SELECT id, role, content_json FROM turns "
            "WHERE summarized = 0 ORDER BY id"
        ).fetchall()
        return [
            Turn(id=r["id"], role=r["role"], content=json.loads(r["content_json"]))
            for r in rows
        ]

    def unsummarized_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM turns WHERE summarized = 0"
        ).fetchone()
        return int(row["n"])

    def list_summaries(self) -> list[Summary]:
        rows = self._conn.execute(
            "SELECT id, summary, span_from, span_to FROM summaries ORDER BY id"
        ).fetchall()
        return [
            Summary(
                id=r["id"],
                summary=r["summary"],
                span_from=r["span_from"],
                span_to=r["span_to"],
            )
            for r in rows
        ]

    def save_summary(self, span_from: int, span_to: int, summary: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO summaries(ts, summary, span_from, span_to) "
            "VALUES (?, ?, ?, ?)",
            (time.time(), summary, span_from, span_to),
        )
        self._conn.execute(
            "UPDATE turns SET summarized = 1 WHERE id BETWEEN ? AND ?",
            (span_from, span_to),
        )
        self._conn.commit()
        return cur.lastrowid

    def add_fact(self, fact: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO known_facts(ts, fact) VALUES (?, ?)",
            (time.time(), fact),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_facts(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT fact FROM known_facts ORDER BY id"
        ).fetchall()
        return [r["fact"] for r in rows]

    def close(self) -> None:
        self._conn.close()


def _anthropic_default(obj: Any) -> Any:
    """JSON fallback for Anthropic SDK content blocks. The SDK returns
    pydantic-like models; pull their dict form."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    raise TypeError(f"cannot serialize {type(obj).__name__} for memory")
