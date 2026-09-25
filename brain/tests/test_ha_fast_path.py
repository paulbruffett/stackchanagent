"""Home Assistant intent fast path: only a definite local match is trusted and
spoken; every miss or failure falls through to the LLM (returns None)."""

from __future__ import annotations

import httpx
import pytest

import ha_fast_path
from config import get_config


def _ha_reply(response_type: str, speech: str, code: str | None = None) -> dict:
    resp = {"response_type": response_type,
            "speech": {"plain": {"speech": speech}}, "data": {}}
    if code:
        resp["data"]["code"] = code
    return {"response": resp}


@pytest.fixture
def ha(mem, monkeypatch):
    """Route the module's httpx client to a fake HA; returns the list of
    request bodies it saw and a setter for the next reply."""
    monkeypatch.setenv("HA_TOKEN", "test-token")
    get_config().set("HA_FAST_PATH", 1)
    seen: list[httpx.Request] = []
    state = {"reply": _ha_reply("action_done", "Turned off the light"), "status": 200}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(state["status"], json=state["reply"])

    real = httpx.AsyncClient
    monkeypatch.setattr(ha_fast_path.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return seen, state


async def test_action_done_is_spoken(ha):
    seen, _ = ha
    got = await ha_fast_path.try_handle("turn off the office light")
    assert got is not None and got.speech == "Turned off the light"
    body = seen[0].read()
    assert b'"agent_id":"conversation.home_assistant"' in body.replace(b" ", b"")
    assert seen[0].headers["authorization"] == "Bearer test-token"


async def test_query_answer_is_spoken(ha):
    _, state = ha
    state["reply"] = _ha_reply("query_answer", "Yes")
    got = await ha_fast_path.try_handle("is the floor lamp on")
    assert got is not None and got.response_type == "query_answer"


@pytest.mark.parametrize("reply", [
    _ha_reply("error", "Sorry, I am not aware of any area called is", "no_valid_targets"),
    _ha_reply("error", "Sorry, I couldn't understand that", "no_intent_match"),
    _ha_reply("action_done", ""),
])
async def test_anything_but_a_definite_hit_falls_through(ha, reply):
    _, state = ha
    state["reply"] = reply
    assert await ha_fast_path.try_handle("tell me a joke") is None


async def test_ha_down_falls_through(ha):
    _, state = ha
    state["status"] = 500
    assert await ha_fast_path.try_handle("turn off the office light") is None


async def test_off_without_token_or_when_disabled(ha, monkeypatch):
    seen, _ = ha
    get_config().set("HA_FAST_PATH", 0)
    assert await ha_fast_path.try_handle("turn off the office light") is None
    get_config().set("HA_FAST_PATH", 1)
    monkeypatch.delenv("HA_TOKEN")
    assert await ha_fast_path.try_handle("turn off the office light") is None
    assert seen == []


async def test_record_exchange_persists_the_fast_path_turn(mem, make_agent):
    sess = make_agent([])
    await sess.record_exchange("turn off the light", "Turned off the light", follow_up=True)
    turns = mem.list_unsummarized_turns()
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[0].content == "[follow-up] turn off the light"
    assert sess.messages[-1]["content"] == [{"type": "text", "text": "Turned off the light"}]
