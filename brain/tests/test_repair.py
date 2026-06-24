"""The M6.5 startup integrity pass (repair_memory) and hydration over
injected DB corruption."""

from __future__ import annotations

from claude_agent import repair_memory, validate_thread
from conftest import persisted_thread

GOOD = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "x", "input": {}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
]


def test_repair_heals_trailing_dangling_tool_use(mem):
    mem.append_turns(GOOD)
    # The classic pre-M6.1 corruption: a tool_use assistant turn with no result.
    mem.append_turns([{"role": "assistant", "content": [
        {"type": "tool_use", "id": "bad", "name": "x", "input": {}}]}])
    assert validate_thread(persisted_thread(mem))  # corrupt before

    counts = repair_memory(mem)
    assert counts["dangling_tool_use"] == 1
    assert counts["turns_deleted"] == 1
    assert validate_thread(persisted_thread(mem)) == []  # clean after


def test_repair_is_idempotent(mem):
    mem.append_turns(GOOD)
    mem.append_turns([{"role": "assistant", "content": [
        {"type": "tool_use", "id": "bad", "name": "x", "input": {}}]}])
    repair_memory(mem)
    second = repair_memory(mem)
    assert all(v == 0 for v in second.values())


def test_repair_clean_db_writes_nothing(mem):
    mem.append_turns(GOOD)
    before = persisted_thread(mem)
    counts = repair_memory(mem)
    assert all(v == 0 for v in counts.values())
    assert persisted_thread(mem) == before


def test_repair_resolves_cascade(mem):
    # Deleting a dangling assistant tool_use can orphan a following result;
    # the fixpoint loop must catch the cascade in a later pass.
    mem.append_turns([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "r"}]},
        # corrupt tail: a tool_use whose result row never got written
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c2", "name": "x", "input": {}}]},
    ])
    repair_memory(mem)
    assert validate_thread(persisted_thread(mem)) == []


def test_repair_rewrites_partial_turn_keeping_text(mem):
    # An assistant turn with BOTH a dangling tool_use and real text: keep the
    # text, strip the tool_use (rewrite, not delete).
    mem.append_turns([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "thinking"},
            {"type": "tool_use", "id": "z", "name": "x", "input": {}},
        ]},
    ])
    counts = repair_memory(mem)
    assert counts["turns_rewritten"] == 1
    assert counts["turns_deleted"] == 0
    thread = persisted_thread(mem)
    assert validate_thread(thread) == []
    assert thread[-1]["content"] == [{"type": "text", "text": "thinking"}]


def test_hydration_over_corruption_yields_valid_api_thread(mem, make_agent):
    # Even without the startup pass, a session that hydrates poisoned history
    # must still build an API-valid request (the read-time sanitizer net).
    mem.append_turns(GOOD)
    mem.append_turns([{"role": "assistant", "content": [
        {"type": "tool_use", "id": "bad", "name": "x", "input": {}}]}])
    sess = make_agent([("text", "ok")])
    from claude_agent import _sanitize_for_api
    assert validate_thread(_sanitize_for_api(sess.messages)) == []
