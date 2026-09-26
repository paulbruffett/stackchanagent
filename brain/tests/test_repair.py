"""The M6.5 startup integrity pass (repair_memory), the one-time switch to
OpenAI-format history (migrate_turn_format), and hydration over injected DB
corruption."""

from __future__ import annotations

from claude_agent import (
    TURN_FORMAT_KEY,
    _sanitize_for_api,
    migrate_turn_format,
    repair_memory,
    validate_thread,
)
from conftest import persisted_thread


def _call(cid, name="x"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}


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


def test_repair_heals_trailing_dangling_tool_call(mem):
    mem.append_turns(GOOD)
    # The classic crash corruption: a tool-call assistant turn with no result.
    mem.append_turns([_asks("bad")])
    assert validate_thread(persisted_thread(mem))  # corrupt before

    counts = repair_memory(mem)
    assert counts["dangling_tool_call"] == 1
    assert counts["turns_deleted"] == 1
    assert validate_thread(persisted_thread(mem)) == []  # clean after
    assert persisted_thread(mem) == GOOD


def test_repair_is_idempotent(mem):
    mem.append_turns(GOOD)
    mem.append_turns([_asks("bad")])
    repair_memory(mem)
    second = repair_memory(mem)
    assert all(v == 0 for v in second.values())


def test_repair_clean_db_writes_nothing(mem):
    mem.append_turns(GOOD)
    before = persisted_thread(mem)
    counts = repair_memory(mem)
    assert all(v == 0 for v in counts.values())
    assert persisted_thread(mem) == before


def test_repair_clean_single_round_exchange_is_left_alone(mem):
    # A device command that ended without a second model round: the thread
    # ends on (then continues after) a tool result. That is valid.
    mem.append_turns([
        {"role": "user", "content": "light on"},
        _asks("c1", content="Turning on the light."),
        _result("c1"),
        {"role": "user", "content": "thanks"},
    ])
    assert all(v == 0 for v in repair_memory(mem).values())


def test_repair_one_pass_suffices(mem):
    # _repair_one_pass decides from the pre-sweep snapshot, and dropping an
    # unanswered call can't orphan a result (it has none) — so one sweep is
    # always enough, pinned here by max_passes=1.
    mem.append_turns([
        {"role": "user", "content": "q"},
        _asks("c1"),
        _result("c1"),
        # corrupt tail: a call whose result row never got written
        _asks("c2"),
    ])
    repair_memory(mem, max_passes=1)
    assert validate_thread(persisted_thread(mem)) == []


def test_repair_of_a_mid_thread_dangling_tool_call(mem):
    # The realistic post-SIGKILL shape: the process died after writing the
    # assistant tool call, rebooted, and the user spoke again — so the dangling
    # call sits in the MIDDLE, not at the tail.
    mem.append_turns([
        {"role": "user", "content": "what's the weather?"},
        _asks("k"),
        {"role": "user", "content": "hello again"},
    ])
    counts = repair_memory(mem)
    assert counts["dangling_tool_call"] == 1 and counts["turns_deleted"] == 1
    # Consecutive user messages are valid Chat Completions input, so the
    # thread left behind needs no further read-time repair.
    thread = persisted_thread(mem)
    assert validate_thread(thread) == []
    assert _sanitize_for_api(thread) == thread


def test_repair_rewrites_partial_turn_keeping_text(mem):
    # An assistant turn with BOTH a dangling call and real text: keep the
    # text, strip the call (rewrite, not delete).
    mem.append_turns([
        {"role": "user", "content": "hi"},
        _asks("z", content="thinking"),
    ])
    counts = repair_memory(mem)
    assert counts["turns_rewritten"] == 1
    assert counts["turns_deleted"] == 0
    thread = persisted_thread(mem)
    assert validate_thread(thread) == []
    assert thread[-1] == {"role": "assistant", "content": "thinking"}


def test_repair_keeps_the_answered_call_of_a_partly_answered_turn(mem):
    mem.append_turns([
        {"role": "user", "content": "hi"},
        _asks("a", "b"),
        _result("a"),
    ])
    counts = repair_memory(mem)
    assert counts["dangling_tool_call"] == 1 and counts["turns_rewritten"] == 1
    thread = persisted_thread(mem)
    assert validate_thread(thread) == []
    assert [c["id"] for c in thread[1]["tool_calls"]] == ["a"]


def test_repair_deletes_orphan_tool_results_and_legacy_rows(mem):
    mem.append_turns([
        {"role": "user", "content": "hi"},
        _result("nobody-asked"),
        {"role": "assistant", "content": [{"type": "text", "text": "old format"}]},
        {"role": "assistant", "content": "fine"},
    ])
    counts = repair_memory(mem)
    assert counts["orphan_tool_result"] == 1 and counts["legacy_format"] == 1
    assert persisted_thread(mem) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "fine"},
    ]


def test_hydration_over_corruption_yields_valid_api_thread(mem, make_agent):
    # Even without the startup pass, a session that hydrates poisoned history
    # must still build an API-valid request (the read-time sanitizer net).
    mem.append_turns(GOOD)
    mem.append_turns([_asks("bad")])
    sess = make_agent([("text", "ok")])
    assert validate_thread(_sanitize_for_api(sess.messages)) == []


# --- one-time switch to OpenAI-format history -------------------------------

def _seed_legacy(mem):
    """What memory.db holds on the Jetson before the switch: Anthropic blocks,
    a summarized span plus an unsummarized tail, and facts."""
    ids = mem.append_turns([
        {"role": "user", "content": "old folded q"},
        {"role": "assistant", "content": [{"type": "text", "text": "old folded a"}]},
        {"role": "user", "content": "recent q"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": "ok"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "recent a"}]},
    ])
    mem.save_summary(ids[0], ids[1], "They chatted.")
    mem.add_fact("The user's name is Paul.")
    return ids


def test_migration_drops_the_legacy_tail_exactly_once(mem):
    ids = _seed_legacy(mem)
    # The unsummarized tail (4) plus the summarized block-format row (1).
    assert migrate_turn_format(mem) == 5
    assert mem.unsummarized_count() == 0
    assert mem.get_runtime_state(TURN_FORMAT_KEY) == "openai"
    # Facts and summaries are plain text and survive, as does the summarized
    # plain-string user row (valid in either format).
    assert mem.list_facts() == ["The user's name is Paul."]
    assert [s.summary for s in mem.list_summaries()] == ["They chatted."]
    assert [t.id for t in mem.recent_turns()] == ids[:1]

    # New-format turns written afterwards are never touched again.
    mem.append_turns([{"role": "user", "content": "new q"},
                      {"role": "assistant", "content": "new a"}])
    assert migrate_turn_format(mem) == 0
    assert mem.unsummarized_count() == 2


def test_migration_on_a_fresh_db_just_stamps_the_format(mem):
    assert migrate_turn_format(mem) == 0
    assert mem.get_runtime_state(TURN_FORMAT_KEY) == "openai"


def test_deleting_a_summary_after_migration_resurrects_nothing_invalid(mem):
    _seed_legacy(mem)
    migrate_turn_format(mem)
    (summary,) = mem.list_summaries()
    mem.delete_summary(summary.id, unmark_turns=True)   # the console path
    thread = persisted_thread(mem)
    assert thread == [{"role": "user", "content": "old folded q"}]
    assert validate_thread(thread) == []
