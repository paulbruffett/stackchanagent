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

  config(key, value, updated_ts)
      Runtime-tunable settings edited from the web UI (Phase 9a).
      value is a JSON-encoded scalar; absent keys fall back to the
      code defaults in config.py.

  mcp_servers(id, name, transport, command, args_json, url, env_ref, enabled)
      MCP servers the agent can pull tools from (Phase 9b). NO secret
      values are stored here — env_ref names an environment variable
      (loaded from .env) that the client injects when launching a
      stdio server. transport is "stdio" or "http".

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

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS mcp_servers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    transport TEXT NOT NULL DEFAULT 'stdio',
    command TEXT,
    args_json TEXT NOT NULL DEFAULT '[]',
    url TEXT,
    env_ref TEXT,
    enabled INTEGER NOT NULL DEFAULT 1
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


@dataclass(frozen=True)
class Fact:
    id: int
    ts: float
    fact: str


@dataclass(frozen=True)
class McpServer:
    id: int
    name: str
    transport: str          # "stdio" | "http"
    command: str | None     # stdio: executable
    args: list[str]         # stdio: argv after command
    url: str | None         # http: endpoint
    env_ref: str | None     # name of a .env var to inject (no value stored)
    enabled: bool


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

    def update_summary(self, summary_id: int, summary: str) -> bool:
        """Edit a summary's text in place (web UI). Span is unchanged."""
        cur = self._conn.execute(
            "UPDATE summaries SET summary = ?, ts = ? WHERE id = ?",
            (summary, time.time(), summary_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_summary(self, summary_id: int, unmark_turns: bool = True) -> bool:
        """Delete a summary. When `unmark_turns` (the default), clear the
        `summarized` flag on the turns it covered so they replay verbatim
        again — i.e. "un-summarize" that span. Spans don't overlap (each
        fold takes a contiguous oldest chunk), so this is safe."""
        row = self._conn.execute(
            "SELECT span_from, span_to FROM summaries WHERE id = ?",
            (summary_id,),
        ).fetchone()
        if row is None:
            return False
        self._conn.execute("DELETE FROM summaries WHERE id = ?", (summary_id,))
        if unmark_turns:
            self._conn.execute(
                "UPDATE turns SET summarized = 0 WHERE id BETWEEN ? AND ?",
                (row["span_from"], row["span_to"]),
            )
        self._conn.commit()
        return True

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

    def list_fact_rows(self) -> list[Fact]:
        """Facts with ids/timestamps, for the editable web-UI view."""
        rows = self._conn.execute(
            "SELECT id, ts, fact FROM known_facts ORDER BY id"
        ).fetchall()
        return [Fact(id=r["id"], ts=r["ts"], fact=r["fact"]) for r in rows]

    def update_fact(self, fact_id: int, fact: str) -> bool:
        cur = self._conn.execute(
            "UPDATE known_facts SET fact = ?, ts = ? WHERE id = ?",
            (fact, time.time(), fact_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_fact(self, fact_id: int) -> bool:
        cur = self._conn.execute(
            "DELETE FROM known_facts WHERE id = ?", (fact_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def replace_facts(self, facts: list[str]) -> None:
        """Atomically replace the entire fact set. Used by LLM fact
        compaction to apply an approved, consolidated list. Order is
        preserved (facts are injected oldest-first into the prompt)."""
        now = time.time()
        try:
            self._conn.execute("DELETE FROM known_facts")
            self._conn.executemany(
                "INSERT INTO known_facts(ts, fact) VALUES (?, ?)",
                [(now, f) for f in facts],
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def recent_turns(self, limit: int = 50) -> list[Turn]:
        """Most-recent turns (any summarized state), oldest-first within
        the returned window. For the web-UI memories view."""
        rows = self._conn.execute(
            "SELECT id, role, content_json FROM turns "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        rows.reverse()
        return [
            Turn(id=r["id"], role=r["role"], content=json.loads(r["content_json"]))
            for r in rows
        ]

    # --- config (Phase 9a web UI) -------------------------------------

    def get_all_config(self) -> dict[str, Any]:
        """Every stored config override, key → decoded JSON value."""
        rows = self._conn.execute(
            "SELECT key, value FROM config"
        ).fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def set_config(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO config(key, value, updated_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_ts = excluded.updated_ts",
            (key, json.dumps(value), time.time()),
        )
        self._conn.commit()

    # --- MCP server registry (Phase 9b) -------------------------------

    def list_mcp_servers(self) -> list[McpServer]:
        rows = self._conn.execute(
            "SELECT id, name, transport, command, args_json, url, env_ref, "
            "enabled FROM mcp_servers ORDER BY id"
        ).fetchall()
        return [self._mcp_row(r) for r in rows]

    @staticmethod
    def _mcp_row(r: sqlite3.Row) -> McpServer:
        return McpServer(
            id=r["id"],
            name=r["name"],
            transport=r["transport"],
            command=r["command"],
            args=json.loads(r["args_json"] or "[]"),
            url=r["url"],
            env_ref=r["env_ref"],
            enabled=bool(r["enabled"]),
        )

    def add_mcp_server(
        self,
        name: str,
        transport: str = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        env_ref: str | None = None,
        enabled: bool = True,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO mcp_servers(name, transport, command, args_json, "
            "url, env_ref, enabled) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, transport, command, json.dumps(args or []), url, env_ref,
             int(enabled)),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_mcp_server(self, server_id: int, **fields: Any) -> bool:
        """Update any subset of {name, transport, command, args, url,
        env_ref, enabled}. `args` is stored JSON-encoded; `enabled` as int."""
        cols: list[str] = []
        vals: list[Any] = []
        for key, value in fields.items():
            if key == "args":
                cols.append("args_json = ?")
                vals.append(json.dumps(value or []))
            elif key == "enabled":
                cols.append("enabled = ?")
                vals.append(int(bool(value)))
            elif key in ("name", "transport", "command", "url", "env_ref"):
                cols.append(f"{key} = ?")
                vals.append(value)
            else:
                raise KeyError(f"unknown mcp_server field: {key}")
        if not cols:
            return False
        vals.append(server_id)
        cur = self._conn.execute(
            f"UPDATE mcp_servers SET {', '.join(cols)} WHERE id = ?", vals
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_mcp_server(self, server_id: int) -> bool:
        cur = self._conn.execute(
            "DELETE FROM mcp_servers WHERE id = ?", (server_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

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
