"""Home Assistant intent fast path: only a definite local match is spoken;
every miss, failure or malformed reply falls through to the LLM (speech None)
and still reports what the attempt cost."""

from __future__ import annotations

import asyncio

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
    """Point the fast path's client at a fake HA. Returns the requests it saw
    and a dict to set the next reply/status on."""
    monkeypatch.setenv("HA_TOKEN", "test-token")
    monkeypatch.delenv("HA_URL", raising=False)
    get_config().set("HA_FAST_PATH", 1)
    seen: list[httpx.Request] = []
    state: dict = {"reply": _ha_reply("action_done", "Turned off the light"), "status": 200}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if state.get("delay"):
            await asyncio.sleep(state["delay"])
        return httpx.Response(state["status"], json=state["reply"])

    monkeypatch.setattr(ha_fast_path, "_client", None)
    monkeypatch.setattr(ha_fast_path, "_make_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return seen, state


async def test_action_done_is_spoken(ha):
    seen, _ = ha
    got = await ha_fast_path.try_handle("turn off the office light")
    assert got.speech == "Turned off the light"
    assert str(seen[0].url) == "http://localhost:8123/api/conversation/process"
    assert b'"agent_id":"conversation.home_assistant"' in seen[0].read().replace(b" ", b"")
    assert seen[0].headers["authorization"] == "Bearer test-token"


async def test_query_answer_is_spoken(ha):
    _, state = ha
    state["reply"] = _ha_reply("query_answer", "Yes")
    got = await ha_fast_path.try_handle("is the floor lamp on")
    assert got.speech == "Yes" and got.response_type == "query_answer"


@pytest.mark.parametrize("reply", [
    _ha_reply("error", "Sorry, I am not aware of any area called is", "no_valid_targets"),
    _ha_reply("error", "Sorry, I couldn't understand that", "no_intent_match"),
    _ha_reply("action_done", ""),
    {"response": None},
    {"response": {"response_type": "action_done", "speech": None}},
    {"response": {"response_type": "action_done", "speech": {"plain": "x"}}},
    ["not", "a", "dict"],
])
async def test_anything_but_a_definite_hit_falls_through(ha, reply):
    _, state = ha
    state["reply"] = reply
    got = await ha_fast_path.try_handle("tell me a joke")
    assert got is not None and got.speech is None
    assert got.latency_ms >= 0


async def test_ha_down_falls_through(ha):
    _, state = ha
    state["status"] = 500
    got = await ha_fast_path.try_handle("turn off the office light")
    assert got.speech is None and got.response_type == "unavailable"


async def test_total_deadline(ha, monkeypatch):
    _, state = ha
    state["delay"] = 1.0
    monkeypatch.setattr(ha_fast_path, "TIMEOUT_S", 0.05)
    got = await ha_fast_path.try_handle("turn off the office light")
    assert got.speech is None and got.latency_ms < 500


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
    await sess.record_exchange("turn off the light", "Turned off the light", follow_up=False)
    turns = mem.list_unsummarized_turns()
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[0].content == "turn off the light"
    assert sess.messages[-1] == {"role": "assistant", "content": "Turned off the light"}
    assert turns[1].message == sess.messages[-1]


def test_vocabulary_takes_exposed_names_aliases_and_areas():
    entities = [
        {"entity_id": "light.office_office", "name": None, "original_name": None,
         "device_id": "d1", "aliases": [None, "office light", "office lights"]},
        {"entity_id": "light.hidden", "name": "Secret lamp", "aliases": ["nope"]},
        {"entity_id": "switch.hue_automation", "name": "Automation: Nightlight", "aliases": []},
        {"entity_id": "sensor.sun_next_dawn", "name": "Sun next dawn", "aliases": []},
    ]
    exposed = {"light.office_office": {"conversation": True},
               "light.hidden": {"conversation": False},
               "sensor.sun_next_dawn": {"conversation": True}}
    areas = [{"name": "Mary’s room", "aliases": ["Mary's room"]}, {"name": "Office", "aliases": []}]
    devices = [{"id": "d1", "name": "Office", "name_by_user": None}]

    words = ha_fast_path._vocabulary(entities, exposed, areas, devices)

    # Curly apostrophes normalised and deduplicated case-insensitively.
    assert words == ["office light", "office lights", "Office", "Mary's room"]
