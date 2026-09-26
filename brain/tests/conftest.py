"""Shared fixtures + a scripted fake OpenAI-style streaming client.

The brain is a flat collection of modules run with `.venv/bin/python`, so we
prepend brain/ to sys.path here and import the modules directly (no package).
The fake client lets us drive `AgentSession._run_loop` deterministically —
scripting each model turn as text (whole or chunked) / a tool call / an API
error before or mid-stream — without a network round-trip, which is what makes
the M6.1/M6.2/M6.4 regressions testable. `FakeWs` captures the commands the
brain sends the firmware so the turn-state handshake is assertable too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

# brain/ is the import root (flat modules: memory.py, claude_agent.py, …).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, init_config  # noqa: E402
from memory import Memory  # noqa: E402


@pytest.fixture
def mem(tmp_path):
    """A fresh Memory on a temp DB, wired to the global Config singleton with
    the noisy on-device knobs disabled so the common path stays quiet. The
    busy_indicator / ack_filler fixtures below turn one
    back on for the tests that assert that feature."""
    m = Memory(tmp_path / "memory.db")
    cfg = init_config(m)
    cfg.set("BUSY_INDICATOR", 0)
    cfg.set("ACK_FILLER", 0)
    yield m
    m.close()


@pytest.fixture
def busy_indicator(mem):
    """Re-enable the on-screen 'thinking' indicator for the tests that assert
    the brain's half of the turn-state handshake (they read `sess.ws.sent`)."""
    get_config().set("BUSY_INDICATOR", 1)


@pytest.fixture
def ack_filler(mem):
    """Re-enable the spoken pre-tool acknowledgement, so the phrases actually
    reaching TTS on a slow-tool turn are observable."""
    get_config().set("ACK_FILLER", 1)


# --- scripted fake streaming client ----------------------------------------
#
# Models the OpenAI SDK's streaming Chat Completions: `create(stream=True)`
# returns an async-iterable, async-context-managed stream of chunks, each with
# `choices[0].delta.content` / `.delta.tool_calls[i]` (index, id,
# function.name, function.arguments) and `finish_reason`, then a final
# usage-only chunk with no choices.

def _ns(**kw: Any) -> SimpleNamespace:
    return SimpleNamespace(**kw)


def _content_chunk(text: str) -> SimpleNamespace:
    return _ns(choices=[_ns(delta=_ns(content=text, tool_calls=None),
                            finish_reason=None, index=0)], usage=None)


def _tool_chunks(index: int, call_id: str, name: str, arguments: str) -> list[SimpleNamespace]:
    """A tool call as providers stream it: id and name first, then the
    arguments in fragments."""
    head = _ns(index=index, id=call_id, type="function",
               function=_ns(name=name, arguments=""))
    cut = len(arguments) // 2
    frags = [a for a in (arguments[:cut], arguments[cut:]) if a]
    out = [_ns(choices=[_ns(delta=_ns(content=None, tool_calls=[head]),
                            finish_reason=None, index=0)], usage=None)]
    for frag in frags:
        piece = _ns(index=index, id=None, type=None, function=_ns(name=None, arguments=frag))
        out.append(_ns(choices=[_ns(delta=_ns(content=None, tool_calls=[piece]),
                                    finish_reason=None, index=0)], usage=None))
    return out


def _finish_chunk(reason: str) -> SimpleNamespace:
    return _ns(choices=[_ns(delta=_ns(content=None, tool_calls=None),
                            finish_reason=reason, index=0)], usage=None)


def _usage_chunk() -> SimpleNamespace:
    return _ns(choices=[], usage=_ns(prompt_tokens=100, completion_tokens=10, cost=0.0001))


class _FakeStream:
    """Yields scripted chunks, then raises `exc` if one is given — a
    connection flap after the user has already heard part of the reply."""

    def __init__(self, chunks: list[Any], exc: BaseException | None = None) -> None:
        self._chunks = chunks
        self._exc = exc
        self.closed = False

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *a: Any) -> bool:
        self.closed = True
        return False

    def __aiter__(self):
        chunks, exc = self._chunks, self._exc

        async def gen():
            for chunk in chunks:
                yield chunk
            if exc is not None:
                raise exc
        return gen()


def _texts(text: str | list[str]) -> list[SimpleNamespace]:
    # A list models the real thing: deltas arrive in arbitrary pieces, so
    # brackets and sentence ends land across chunk boundaries.
    pieces = [text] if isinstance(text, str) else list(text)
    return [_content_chunk(p) for p in pieces if p]


def _build_stream(step: tuple):
    """Turn a script step into one fake stream (or the exception create raises).

    ("text", "spoken reply")                   → finish "stop" with that text
    ("text_chunks", ["spo", "ken reply"])      → same, streamed in pieces
    ("tool", name, call_id[, lead_text[, args]])
                                               → a tool call (optional text in
                                                 the same message), finish
                                                 "tool_calls"; args default "{}"
    ("tools", [(name, id, args), …][, lead])   → several calls in one message
    ("cutoff", name, call_id[, lead_text])     → a tool call whose arguments
                                                 are cut off, finish "length"
    ("error", exception)                       → create() raises (no tokens)
    ("stream_error", ["chu", "nks"], exc)      → deltas stream, then it raises
    """
    kind = step[0]
    if kind in ("text", "text_chunks"):
        return _FakeStream(_texts(step[1]) + [_finish_chunk("stop"), _usage_chunk()])
    if kind == "tool":
        name, cid = step[1], step[2]
        lead = step[3] if len(step) > 3 else ""
        args = step[4] if len(step) > 4 else "{}"
        return _FakeStream(_texts(lead) + _tool_chunks(0, cid, name, args)
                           + [_finish_chunk("tool_calls"), _usage_chunk()])
    if kind == "tools":
        lead = step[2] if len(step) > 2 else ""
        chunks = _texts(lead)
        for i, (name, cid, args) in enumerate(step[1]):
            chunks += _tool_chunks(i, cid, name, args)
        return _FakeStream(chunks + [_finish_chunk("tool_calls"), _usage_chunk()])
    if kind == "cutoff":
        # What a provider sends when max_tokens lands inside a tool call: the
        # call's arguments stop mid-JSON and finish_reason is "length", not
        # "tool_calls". Nothing raises.
        name, cid = step[1], step[2]
        lead = step[3] if len(step) > 3 else ""
        return _FakeStream(_texts(lead) + _tool_chunks(0, cid, name, '{"location": "Sea')
                           + [_finish_chunk("length"), _usage_chunk()])
    if kind == "error":
        return step[1]
    if kind == "stream_error":
        return _FakeStream(_texts(list(step[1])), step[2])
    raise ValueError(f"bad script step: {step!r}")


class FakeWs:
    """Records every command the brain sends the firmware. `object()` swallowed
    them (no `.send`, and `_set_busy` catches the AttributeError), which left
    the brain's half of the turn-state handshake unobservable."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def cmds(self, cmd: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if m.get("cmd") == cmd]


@pytest.fixture
def make_agent(mem, monkeypatch) -> Callable:
    """Factory: build an AgentSession whose model turns are scripted by
    `steps` and whose tool dispatch is optionally replaced by `dispatch`."""
    import claude_agent
    import tools

    def factory(steps: list[tuple], dispatch=None):
        streams = [_build_stream(s) for s in steps]

        class FakeCompletions:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            async def create(self, **kw: Any):
                self.calls.append(kw)
                nxt = streams.pop(0)
                if isinstance(nxt, BaseException):
                    raise nxt
                return nxt

        class FakeClient:
            def __init__(self, *a: Any, **k: Any) -> None:
                self.chat = SimpleNamespace(completions=FakeCompletions())

        monkeypatch.setattr(claude_agent, "AsyncOpenAI", FakeClient)
        if dispatch is not None:
            monkeypatch.setattr(tools, "dispatch", dispatch)
        return claude_agent.AgentSession(ws=FakeWs(), memory=mem)

    return factory


@pytest.fixture
def speaker() -> Callable:
    """Returns (spoken_list, speak_coro). `speak_coro` collects each spoken
    sentence so a test can assert on what reached TTS."""
    spoken: list[str] = []

    async def speak(s: str) -> None:
        spoken.append(s)

    return spoken, speak


def persisted_thread(mem) -> list[dict]:
    """The committed unsummarized thread as chat messages (for validate_thread)."""
    return [t.message for t in mem.list_unsummarized_turns()]


def model_calls(sess) -> list[dict[str, Any]]:
    """Every chat.completions.create kwargs the session sent, in order."""
    return sess.client.chat.completions.calls
