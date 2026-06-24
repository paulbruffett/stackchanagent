"""End-to-end _run_loop behaviour: M6.1 atomic persistence, M6.2 tool-raise
recovery, and M6.4 graceful API-error degradation — driven by the scripted
fake client so no network is involved."""

from __future__ import annotations

import httpx
import pytest
from anthropic import BadRequestError, InternalServerError

import claude_agent
from claude_agent import validate_thread
from conftest import persisted_thread


def _api_error(cls, code):
    resp = httpx.Response(code, request=httpx.Request("POST", "http://x"))
    return cls("boom", response=resp, body=None)


async def _ok_dispatch(name, input_, ctx):
    return f"{name} done"


async def _raising_dispatch(name, input_, ctx):
    raise RuntimeError("network down")


def _tool_results(call):
    return [b for m in call["messages"] if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


# --- M6.1 atomic persistence ------------------------------------------------

async def test_simple_turn_commits_atomically(mem, make_agent, speaker):
    spoken, speak = speaker
    sess = make_agent([("text", "Hi there.")])
    full = await sess.respond("hello", speak)
    assert full == "Hi there." and spoken == ["Hi there."]
    assert validate_thread(persisted_thread(mem)) == []
    assert mem.unsummarized_count() == 2  # user + assistant


async def test_tool_turn_commits_whole_exchange(mem, make_agent, speaker):
    spoken, speak = speaker
    sess = make_agent([("tool", "describe_view", "t1"), ("text", "I see a cat.")],
                      dispatch=_ok_dispatch)
    full = await sess.respond("what do you see?", speak)
    assert full == "I see a cat."
    thread = persisted_thread(mem)
    assert validate_thread(thread) == []
    # user, assistant(tool_use), user(tool_result), assistant(text)
    assert [t["role"] for t in thread] == ["user", "assistant", "user", "assistant"]


async def test_crash_mid_turn_persists_nothing(mem, make_agent, speaker):
    # A non-API exception (e.g. a process kill surrogate) raised after the
    # assistant tool_use is staged must leave NOTHING in the DB — the M6.1
    # invariant: a half-written exchange is never durable.
    spoken, speak = speaker
    sess = make_agent([("tool", "describe_view", "t1"), ("error", RuntimeError("killed"))],
                      dispatch=_ok_dispatch)
    with pytest.raises(RuntimeError):
        await sess.respond("what do you see?", speak)
    assert mem.unsummarized_count() == 0


# --- M6.2 tool dispatch never aborts the turn -------------------------------

async def test_tool_raise_becomes_is_error_and_recovers(mem, make_agent, speaker):
    spoken, speak = speaker
    sess = make_agent([("tool", "describe_view", "t1"), ("text", "I could not.")],
                      dispatch=_raising_dispatch)
    full = await sess.respond("what do you see?", speak)
    assert full == "I could not."
    # The follow-up request answered the tool_use with an is_error result.
    tr = _tool_results(sess.client.messages.calls[1])[0]
    assert tr["tool_use_id"] == "t1" and tr.get("is_error") is True
    # Nothing dangling persisted: every tool_use is answered.
    assert validate_thread(persisted_thread(mem)) == []


# --- M6.4 graceful API-error degradation ------------------------------------

async def test_transient_error_retries_then_succeeds(mem, make_agent, speaker):
    spoken, speak = speaker
    sess = make_agent([("error", _api_error(InternalServerError, 500)), ("text", "All good now.")])
    full = await sess.respond("hello", speak)
    assert full == "All good now."
    assert len(sess.client.messages.calls) == 2  # retried once
    assert mem.unsummarized_count() == 2  # committed


async def test_persistent_error_speaks_fallback_no_persist(mem, make_agent, speaker):
    spoken, speak = speaker
    sess = make_agent([("error", _api_error(InternalServerError, 500)),
                       ("error", _api_error(InternalServerError, 500))])
    full = await sess.respond("hello", speak)
    assert full == claude_agent.API_ERROR_FALLBACK
    assert spoken == [claude_agent.API_ERROR_FALLBACK]
    assert len(sess.client.messages.calls) == 2  # exactly one retry
    assert mem.unsummarized_count() == 0  # failed turn dropped


async def test_validation_400_truncates_history_then_retries(mem, make_agent, speaker):
    mem.append_turns([
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": [{"type": "text", "text": "old a"}]},
    ])
    spoken, speak = speaker
    sess = make_agent([("error", _api_error(BadRequestError, 400)), ("text", "Recovered.")])
    assert len(sess.messages) == 2  # hydrated old history
    full = await sess.respond("new question", speak)
    assert full == "Recovered."
    retry_msgs = sess.client.messages.calls[1]["messages"]
    assert all(m.get("content") != "old q" for m in retry_msgs)
