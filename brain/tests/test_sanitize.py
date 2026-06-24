"""The API-contract checker (validate_thread), the read-time repair
(_sanitize_for_api), and the summary fold boundary (_last_complete_assistant_id)."""

from __future__ import annotations

from claude_agent import (
    _last_complete_assistant_id,
    _sanitize_for_api,
    validate_thread,
)
from memory import Turn

GOOD = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "x", "input": {}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
]


def test_validate_thread_accepts_a_clean_tool_exchange():
    assert validate_thread(GOOD) == []


def test_validate_thread_flags_dangling_tool_use():
    problems = validate_thread(GOOD[:2])  # tool_use with no following result
    assert any("tool_use a" in p for p in problems)


def test_validate_thread_flags_orphan_tool_result():
    thread = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "z", "content": "x"}]}]
    assert any("orphan" in p for p in validate_thread(thread))


def test_validate_thread_flags_non_user_first_message():
    assert any("not role=user" in p for p in validate_thread(
        [{"role": "assistant", "content": "hi"}]))


def test_validate_thread_flags_consecutive_same_role():
    thread = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    assert any("consecutive" in p for p in validate_thread(thread))


def test_sanitize_drops_dangling_tool_use():
    out = _sanitize_for_api(GOOD[:2])
    assert validate_thread(out) == []
    # The text-only first message survives; the dangling assistant turn is gone.
    assert out == [{"role": "user", "content": "hi"}]


def test_sanitize_drops_orphan_tool_result_keeps_valid_one():
    thread = GOOD[:2] + [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "ok"},
        {"type": "tool_result", "tool_use_id": "ORPHAN", "content": "?"},
    ]}]
    out = _sanitize_for_api(thread)
    results = [b["tool_use_id"] for m in out if isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_result"]
    assert results == ["a"]
    assert validate_thread(out) == []


def test_sanitize_merges_consecutive_same_role():
    thread = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    out = _sanitize_for_api(thread)
    assert len(out) == 1 and out[0]["role"] == "user"


def test_sanitize_is_idempotent_on_clean_thread():
    once = _sanitize_for_api(GOOD)
    assert _sanitize_for_api(once) == once == GOOD


def _turns(rows):
    return [Turn(id=i + 1, role=r, content=c) for i, (r, c) in enumerate(rows)]


def test_last_complete_assistant_id_skips_pending_tool_use():
    # Boundary must not split inside a tool_use/tool_result pair: the only
    # complete assistant turn is the final text one (id 4).
    turns = _turns([
        ("user", "hi"),
        ("assistant", [{"type": "tool_use", "id": "a", "name": "x", "input": {}}]),
        ("user", [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]),
        ("assistant", [{"type": "text", "text": "done"}]),
    ])
    assert _last_complete_assistant_id(turns, len(turns)) == 4


def test_last_complete_assistant_id_none_when_only_pending():
    turns = _turns([
        ("user", "hi"),
        ("assistant", [{"type": "tool_use", "id": "a", "name": "x", "input": {}}]),
    ])
    assert _last_complete_assistant_id(turns, len(turns)) is None


def test_last_complete_assistant_id_plain_string_turn():
    turns = _turns([("user", "hi"), ("assistant", "hello"), ("user", "more")])
    assert _last_complete_assistant_id(turns, len(turns)) == 2
