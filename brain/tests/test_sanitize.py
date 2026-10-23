"""The chat-message contract checker (validate_thread), the read-time repair
(_sanitize_for_api), and the summary fold boundary (_last_complete_assistant_id)."""

from __future__ import annotations

from claude_agent import (
    _last_complete_assistant_id,
    _sanitize_for_api,
    validate_thread,
)
from memory import Turn


def _call(cid):
    return {"id": cid, "type": "function", "function": {"name": "x", "arguments": "{}"}}


def _asks(*ids, content=None):
    return {"role": "assistant", "content": content, "tool_calls": [_call(i) for i in ids]}


def _result(cid, content="ok"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


GOOD = [
    {"role": "user", "content": "hi"},
    _asks("a"),
    _result("a"),
    {"role": "assistant", "content": "done"},
]


def test_validate_thread_accepts_a_clean_tool_exchange():
    assert validate_thread(GOOD) == []


def test_validate_thread_accepts_a_single_round_exchange_then_a_user_turn():
    assert validate_thread(GOOD[:3] + [{"role": "user", "content": "thanks"}]) == []


def test_validate_thread_flags_dangling_tool_call():
    problems = validate_thread(GOOD[:2])  # tool call with no following result
    assert any("tool_call a" in p for p in problems)


def test_validate_thread_flags_a_call_answered_only_after_a_user_turn():
    thread = GOOD[:2] + [{"role": "user", "content": "hm"}, _result("a")]
    problems = validate_thread(thread)
    assert any("tool_call a" in p for p in problems)
    assert any("orphan" in p for p in problems)


def test_validate_thread_flags_orphan_tool_result():
    thread = [{"role": "user", "content": "q"}, _result("z")]
    assert any("orphan" in p for p in validate_thread(thread))


def test_validate_thread_flags_duplicate_result_as_orphan():
    assert any("orphan" in p for p in validate_thread(GOOD[:3] + [_result("a")]))


def test_validate_thread_flags_legacy_rows():
    thread = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
    assert any("legacy" in p for p in validate_thread(thread))


def test_sanitize_drops_dangling_tool_call():
    out = _sanitize_for_api(GOOD[:2])
    assert validate_thread(out) == []
    # The first message survives; the call-only assistant turn is gone.
    assert out == [{"role": "user", "content": "hi"}]


def test_sanitize_keeps_text_of_a_dangling_call():
    out = _sanitize_for_api([GOOD[0], _asks("a", content="Let me see.")])
    assert out[-1] == {"role": "assistant", "content": "Let me see."}


def test_sanitize_drops_orphan_tool_result_keeps_valid_one():
    thread = GOOD[:3] + [_result("ORPHAN", "?")]
    out = _sanitize_for_api(thread)
    assert [m["tool_call_id"] for m in out if m["role"] == "tool"] == ["a"]
    assert validate_thread(out) == []


def test_sanitize_keeps_the_new_utterance_after_an_aborted_turn():
    # The realistic shape: a turn aborted after its call was staged but before
    # its result, then the user spoke again. Losing the new utterance here
    # means the robot answers the previous question.
    thread = GOOD[:2] + [{"role": "user", "content": "hello again"}]
    out = _sanitize_for_api(thread)
    assert validate_thread(out) == []
    assert out == [GOOD[0], {"role": "user", "content": "hello again"}]


def test_sanitize_drops_legacy_rows():
    thread = [GOOD[0], {"role": "assistant", "content": [{"type": "text", "text": "x"}]}]
    assert _sanitize_for_api(thread) == [GOOD[0]]


def test_sanitize_is_idempotent_on_clean_thread():
    once = _sanitize_for_api(GOOD)
    assert _sanitize_for_api(once) == once == GOOD


def _turns(messages):
    return [Turn(id=i + 1, role=m["role"], content=m["content"],
                 extra={k: v for k, v in m.items() if k not in ("role", "content")})
            for i, m in enumerate(messages)]


def test_last_complete_assistant_id_skips_pending_tool_call():
    # Boundary must not split inside a call/result pair: the only complete
    # point is the final text reply (id 4).
    turns = _turns(GOOD)
    assert _last_complete_assistant_id(turns, len(turns)) == 4


def test_last_complete_assistant_id_none_when_only_pending():
    turns = _turns(GOOD[:2])
    assert _last_complete_assistant_id(turns, len(turns)) is None


def test_last_complete_assistant_id_plain_string_turn():
    turns = _turns([{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "more"}])
    assert _last_complete_assistant_id(turns, len(turns)) == 2


def test_last_complete_boundary_after_a_single_round_exchange():
    # A device command that ended on its tool result, then a new user turn:
    # the tool result (id 3) closes a complete exchange.
    turns = _turns([
        {"role": "user", "content": "light on"},
        _asks("c1", content="Turning on the light."),
        _result("c1"),
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "Any time."},
    ])
    assert _last_complete_assistant_id(turns, 4) == 3
    # …and at the very end of the backlog too.
    assert _last_complete_assistant_id(turns[:3], 3) == 3


def test_last_complete_boundary_not_inside_a_two_round_exchange():
    # A tool result followed by the model's second round is mid-exchange.
    turns = _turns(GOOD)
    assert _last_complete_assistant_id(turns, 3) is None


def test_last_complete_boundary_not_on_a_partly_answered_round():
    turns = _turns([
        {"role": "user", "content": "q"},
        _asks("a", "b"),
        _result("a"),
        {"role": "user", "content": "next"},
    ])
    assert _last_complete_assistant_id(turns, 3) is None
