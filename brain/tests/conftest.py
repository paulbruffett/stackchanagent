"""Shared fixtures + a scripted fake Anthropic streaming client.

The brain is a flat collection of modules run with `.venv/bin/python`, so we
prepend brain/ to sys.path here and import the modules directly (no package).
The fake client lets us drive `AgentSession._run_loop` deterministically —
scripting each model turn as text / a tool call / an API error — without a
network round-trip, which is what makes the M6.1/M6.2/M6.4 regressions testable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pytest

# brain/ is the import root (flat modules: memory.py, claude_agent.py, …).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import init_config  # noqa: E402
from memory import Memory  # noqa: E402


@pytest.fixture
def mem(tmp_path):
    """A fresh Memory on a temp DB, wired to the global Config singleton with
    the noisy on-device knobs disabled so a turn needs no websocket."""
    m = Memory(tmp_path / "memory.db")
    cfg = init_config(m)
    cfg.set("BUSY_INDICATOR", 0)
    cfg.set("ACK_FILLER", 0)
    cfg.set("ROCKY_MODE", 0)
    cfg.set("FOLLOW_UP_THINKING", 0)
    yield m
    m.close()


# --- scripted fake streaming client ----------------------------------------

class _FakeBlock:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        return dict(self.__dict__)


class _FakeUsage:
    input_tokens = output_tokens = 0
    cache_read_input_tokens = cache_creation_input_tokens = 0


class _FakeResp:
    def __init__(self, content: list[Any], stop: str) -> None:
        self.content = content
        self.stop_reason = stop
        self.usage = _FakeUsage()


class _OkStream:
    def __init__(self, text: str, resp: _FakeResp) -> None:
        self._text = text
        self._resp = resp

    async def __aenter__(self) -> "_OkStream":
        return self

    async def __aexit__(self, *a: Any) -> bool:
        return False

    @property
    def text_stream(self):
        async def gen():
            if self._text:
                yield self._text
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


def _build_stream(step: tuple):
    """Turn a script step into one fake stream.

    ("text", "spoken reply")                 → end_turn with that text
    ("tool", name, tool_id[, lead_text])     → tool_use turn (optional pre-text)
    ("error", exception)                     → request raises (no tokens)
    """
    kind = step[0]
    if kind == "text":
        txt = step[1]
        return _OkStream(txt, _FakeResp([_FakeBlock(type="text", text=txt)], "end_turn"))
    if kind == "tool":
        name, tid = step[1], step[2]
        lead = step[3] if len(step) > 3 else ""
        block = _FakeBlock(type="tool_use", id=tid, name=name, input={})
        return _OkStream(lead, _FakeResp([block], "tool_use"))
    if kind == "error":
        return _ErrStream(step[1])
    raise ValueError(f"bad script step: {step!r}")


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
        return claude_agent.AgentSession(ws=object(), memory=mem)

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
