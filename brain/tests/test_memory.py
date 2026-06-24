"""Memory persistence primitives, including the M6.1 batch insert and the
M6.5 delete/update helpers."""

from __future__ import annotations

import pytest


def test_append_turns_atomic_and_ordered(mem):
    ids = mem.append_turns([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ])
    assert ids == [1, 2]
    assert mem.unsummarized_count() == 2


def test_append_turns_rollback_on_bad_batch(mem):
    class Unserializable:
        pass

    mem.append_turns([{"role": "user", "content": "ok"}])
    before = mem.unsummarized_count()
    with pytest.raises(TypeError):
        mem.append_turns([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": Unserializable()},  # fails json.dumps
        ])
    # Neither row of the failed batch may have landed.
    assert mem.unsummarized_count() == before


def test_append_turns_empty_is_noop(mem):
    assert mem.append_turns([]) == []
    assert mem.unsummarized_count() == 0


def test_update_turn(mem):
    (tid,) = mem.append_turns([{"role": "assistant", "content": [
        {"type": "tool_use", "id": "a", "name": "x", "input": {}},
        {"type": "text", "text": "hi"},
    ]}])
    assert mem.update_turn(tid, [{"type": "text", "text": "hi"}]) is True
    rows = mem.list_unsummarized_turns()
    assert rows[0].content == [{"type": "text", "text": "hi"}]
    assert mem.update_turn(9999, []) is False


def test_delete_turn(mem):
    a, b = mem.append_turns([
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ])
    assert mem.delete_turn(a) is True
    assert mem.delete_turn(a) is False  # already gone
    assert [t.id for t in mem.list_unsummarized_turns()] == [b]


def test_delete_turns_from(mem):
    ids = mem.append_turns([
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "4"},
    ])
    deleted = mem.delete_turns_from(ids[2])
    assert deleted == 2
    assert [t.id for t in mem.list_unsummarized_turns()] == ids[:2]
    # Resetting an empty tail is a clean no-op.
    assert mem.delete_turns_from(ids[2]) == 0
