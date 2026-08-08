"""Shared fixtures + a scripted fake Anthropic streaming client.

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
    busy_indicator / ack_filler / follow_up_thinking fixtures below turn one
    back on for the tests that assert that feature."""
    m = Memory(tmp_path / "memory.db")
    cfg = init_config(m)
    cfg.set("BUSY_INDICATOR", 0)
    cfg.set("ACK_FILLER", 0)
    cfg.set("ROCKY_MODE", 0)
    cfg.set("FOLLOW_UP_THINKING", 0)
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


@pytest.fixture
def follow_up_thinking(mem):
    """Turn extended thinking on for the follow-up / stage-direction turns that
    opt into it — off in `mem` so the wake-word path stays the default."""
    get_config().set("FOLLOW_UP_THINKING", 1)


# --- scripted fake streaming client ----------------------------------------

class _FakeBlock:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        # Honouring exclude_none is not pedantry: it is what actually keeps the
        # SDK's unset optionals (`citations`) and stream-only decorations
        # (`parsed_output`) out of the message we replay next turn. A fake that
        # ignored the flag would let `_clean_block` lose it and stay green.
        d = dict(self.__dict__)
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


def _text_block(text: str) -> _FakeBlock:
    """A streamed text block as the SDK hands it over — carrying the optional
    fields it always attaches, which must not survive into persisted history."""
    return _FakeBlock(type="text", text=text, citations=None, parsed_output=None)


class _FakeUsage:
    input_tokens = output_tokens = 0
    cache_read_input_tokens = cache_creation_input_tokens = 0


class _FakeResp:
    def __init__(self, content: list[Any], stop: str) -> None:
        self.content = content
        self.stop_reason = stop
        self.usage = _FakeUsage()


class _OkStream:
    def __init__(self, text: str | list[str], resp: _FakeResp) -> None:
        # A list models the real thing: deltas arrive in arbitrary pieces, so
        # brackets and sentence ends land across chunk boundaries.
        self._chunks = [text] if isinstance(text, str) else list(text)
        self._resp = resp

    async def __aenter__(self) -> "_OkStream":
        return self

    async def __aexit__(self, *a: Any) -> bool:
        return False

    @property
    def text_stream(self):
        async def gen():
            for chunk in self._chunks:
                if chunk:
                    yield chunk
        return gen()

    async def get_final_message(self) -> _FakeResp:
        return self._resp


class _ErrStream:
    """Raises on context entry — models the request being rejected before any
    token streams (the real 400 / connection-error timing)."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *a: Any) -> bool:
        return False


class _MidStreamErrStream:
    """Streams some deltas and then raises — a connection flap after the user
    has already heard part of the reply. `_ErrStream` can't model this (it dies
    before any token), so it's the only way to exercise the `spoke_partial`
    guard that stops a retry from double-speaking."""

    def __init__(self, chunks: list[str], exc: BaseException) -> None:
        self._chunks = chunks
        self._exc = exc

    async def __aenter__(self) -> "_MidStreamErrStream":
        return self

    async def __aexit__(self, *a: Any) -> bool:
        return False

    @property
    def text_stream(self):
        chunks, exc = self._chunks, self._exc

        async def gen():
            for chunk in chunks:
                yield chunk
            raise exc
        return gen()


def _build_stream(step: tuple):
    """Turn a script step into one fake stream.

    ("text", "spoken reply")                 → end_turn with that text
    ("text_chunks", ["spo", "ken reply"])    → same, streamed in pieces
    ("tool", name, tool_id[, lead_text])     → tool_use turn (optional pre-text)
    ("cutoff", name, tool_id[, lead_text])   → tool_use cut off by max_tokens
    ("error", exception)                     → request raises (no tokens)
    ("stream_error", ["chu", "nks"], exc)    → deltas stream, then it raises
    """
    kind = step[0]
    if kind == "text":
        txt = step[1]
        return _OkStream(txt, _FakeResp([_text_block(txt)], "end_turn"))
    if kind == "text_chunks":
        chunks = list(step[1])
        return _OkStream(chunks, _FakeResp([_text_block("".join(chunks))], "end_turn"))
    if kind == "tool":
        name, tid = step[1], step[2]
        lead = step[3] if len(step) > 3 else ""
        block = _FakeBlock(type="tool_use", id=tid, name=name, input={})
        return _OkStream(lead, _FakeResp([block], "tool_use"))
    if kind == "cutoff":
        # What the SDK actually returns when the response is truncated inside
        # a tool_use: the block is present with partial-JSON input, no
        # exception is raised, and stop_reason is "max_tokens" — NOT
        # "tool_use", so the loop treats the turn as finished.
        name, tid = step[1], step[2]
        lead = step[3] if len(step) > 3 else ""
        blocks = [_FakeBlock(type="text", text=lead)] if lead else []
        blocks.append(_FakeBlock(type="tool_use", id=tid, name=name, input={}))
        return _OkStream(lead, _FakeResp(blocks, "max_tokens"))
    if kind == "error":
        return _ErrStream(step[1])
    if kind == "stream_error":
        return _MidStreamErrStream(list(step[1]), step[2])
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

        class FakeMessages:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def stream(self, **kw: Any):
                self.calls.append(kw)
                return streams.pop(0)

        class FakeClient:
            def __init__(self, *a: Any, **k: Any) -> None:
                self.messages = FakeMessages()

        monkeypatch.setattr(claude_agent, "AsyncAnthropic", FakeClient)
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
    """The committed unsummarized thread as plain dicts (for validate_thread)."""
    return [{"role": t.role, "content": t.content} for t in mem.list_unsummarized_turns()]
