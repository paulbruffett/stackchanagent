"""Memory persistence primitives: the M6.1 batch insert, the M6.5
delete/update helpers, episodic retention, and the schema-version step."""

from __future__ import annotations

import pytest


def test_append_turns_atomic_and_ordered(mem):
    ids = mem.append_turns([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
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
    call = {"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}}
    (tid,) = mem.append_turns([
        {"role": "assistant", "content": "hi", "tool_calls": [call]},
    ])
    assert mem.list_unsummarized_turns()[0].extra == {"tool_calls": [call]}
    # Stripping the calls clears extra_json entirely.
    assert mem.update_turn(tid, "hi") is True
    rows = mem.list_unsummarized_turns()
    assert rows[0].message == {"role": "assistant", "content": "hi"}
    assert mem.update_turn(9999, "x") is False


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


def test_delete_unsummarized_turns_spares_summarized_rows(mem):
    ids = mem.append_turns([
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "older reply"},
        {"role": "user", "content": "live"},
    ])
    mem.save_summary(ids[0], ids[1], "they said hello")
    assert mem.delete_unsummarized_turns() == 1
    assert mem.unsummarized_count() == 0
    assert [t.id for t in mem.recent_turns()] == ids[:2]


# --- retention -------------------------------------------------------------

def test_prune_never_deletes_turns_a_retained_summary_covers(mem):
    # Deleting a middle summary in the console un-marks its turns, and the next
    # fold re-summarizes them into a row with the HIGHEST id but the LOWEST
    # span — after which summary-id order no longer tracks span order.
    mem.append_turns([{"role": "user", "content": str(i)} for i in range(1, 41)])
    mem.save_summary(1, 10, "s1")
    s2 = mem.save_summary(11, 20, "s2")
    mem.save_summary(21, 30, "s3")
    mem.delete_summary(s2, unmark_turns=True)
    refold = mem.save_summary(11, 20, "s2, again")

    s_del, t_del = mem.prune_summaries(1)
    assert s_del == 2 and [s.id for s in mem.list_summaries()] == [refold]
    remaining = {t.id for t in mem.recent_turns(100)}
    assert set(range(11, 21)) <= remaining        # the retained summary's span
    assert not (set(range(1, 11)) & remaining)    # the doomed span is purged
    assert t_del == 10


def test_prune_keeps_recent_summaries_and_purges_older_turns(mem):
    mem.append_turns([{"role": "user", "content": str(i)} for i in range(1, 31)])
    mem.save_summary(1, 10, "s1")
    mem.save_summary(11, 20, "s2")
    s_del, t_del = mem.prune_summaries(1)
    assert (s_del, t_del) == (1, 10)
    assert [s.span_from for s in mem.list_summaries()] == [11]
    assert min(t.id for t in mem.recent_turns(100)) == 11


# --- facts -----------------------------------------------------------------

def test_list_facts_limit_keeps_the_newest_oldest_first(mem):
    for f in ["one", "two", "three"]:
        mem.add_fact(f)
    assert mem.list_facts() == ["one", "two", "three"]
    assert mem.list_facts(limit=2) == ["two", "three"]
    assert mem.list_facts(limit=0) == ["one", "two", "three"]  # 0 = no cap


# --- schema versioning -----------------------------------------------------

def test_pending_migrations_apply_once_to_an_existing_db(tmp_path, monkeypatch):
    import memory as memory_mod

    path = tmp_path / "m.db"
    memory_mod.Memory(path).close()          # a DB from before the migration
    head = len(memory_mod.MIGRATIONS)
    monkeypatch.setattr(
        memory_mod, "MIGRATIONS",
        memory_mod.MIGRATIONS + ["ALTER TABLE turns ADD COLUMN exchange_id INTEGER;"],
    )
    m = memory_mod.Memory(path)
    cols = {r[1] for r in m._conn.execute("PRAGMA table_info(turns)")}
    assert "exchange_id" in cols
    assert m._conn.execute("PRAGMA user_version").fetchone()[0] == head + 1
    m.close()
    # Re-opening must not re-run it — a second ALTER would raise.
    m2 = memory_mod.Memory(path)
    assert m2._conn.execute("PRAGMA user_version").fetchone()[0] == head + 1
    m2.close()


def test_pre_openai_db_gains_the_extra_column_and_still_reads(tmp_path):
    # The deployed memory.db predates extra_json (user_version 0). Opening it
    # must add the column rather than fail the first SELECT naming it.
    import sqlite3

    import memory as memory_mod

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE turns (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
        "role TEXT NOT NULL, content_json TEXT NOT NULL, "
        "summarized INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO turns(ts, role, content_json) VALUES (0, 'user', '\"hi\"');"
    )
    conn.close()
    m = memory_mod.Memory(path)
    (turn,) = m.list_unsummarized_turns()
    assert turn.message == {"role": "user", "content": "hi"}
    m.append_turns([{"role": "assistant", "content": None,
                     "tool_calls": [{"id": "c1", "type": "function",
                                     "function": {"name": "x", "arguments": "{}"}}]}])
    assert m.list_unsummarized_turns()[-1].extra["tool_calls"][0]["id"] == "c1"
    m.close()


def test_fresh_db_is_stamped_at_head_without_running_migrations(tmp_path, monkeypatch):
    import memory as memory_mod

    monkeypatch.setattr(
        memory_mod, "MIGRATIONS", ["ALTER TABLE turns ADD COLUMN nope INTEGER;"]
    )
    m = memory_mod.Memory(tmp_path / "fresh.db")   # SCHEMA is already current
    assert m._conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "nope" not in {r[1] for r in m._conn.execute("PRAGMA table_info(turns)")}
    m.close()


# --- episodic retention: the only path that hard-deletes conversation rows ---
# prune_summaries runs automatically after every background fold, so a bad
# cutoff silently destroys history the user still expects to be remembered.

def _turns(mem, n):
    return mem.append_turns([
        {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
        for i in range(n)
    ])


def test_prune_summaries_drops_only_the_oldest_spans(mem):
    ids = _turns(mem, 5)
    mem.save_summary(ids[0], ids[1], "oldest")
    mem.save_summary(ids[2], ids[3], "newer")
    assert mem.unsummarized_count() == 1  # save_summary marked both spans

    assert mem.prune_summaries(1) == (1, 2)  # (summaries, turns) deleted
    assert [s.summary for s in mem.list_summaries()] == ["newer"]
    # The kept summary's rows and the live verbatim tail both survive.
    assert [t.id for t in mem.recent_turns()] == ids[2:]
    assert [t.id for t in mem.list_unsummarized_turns()] == [ids[4]]


def test_prune_summaries_keep_recent_zero_keeps_everything(mem):
    ids = _turns(mem, 2)
    mem.save_summary(ids[0], ids[1], "only")
    assert mem.prune_summaries(0) == (0, 0)
    assert len(mem.list_summaries()) == 1
    assert [t.id for t in mem.recent_turns()] == ids


def test_prune_summaries_spares_un_summarized_turns_below_the_cutoff(mem):
    # A web-UI delete_summary puts its span back into verbatim replay, which
    # leaves live turns BELOW a later prune's cutoff. Only the `summarized = 1`
    # clause stops the next automatic fold from hard-deleting them.
    ids = _turns(mem, 8)
    mem.save_summary(ids[0], ids[1], "s1")
    middle = mem.save_summary(ids[2], ids[3], "s2")
    mem.save_summary(ids[4], ids[5], "s3")
    mem.save_summary(ids[6], ids[7], "s4")
    assert mem.delete_summary(middle) is True
    assert [t.id for t in mem.list_unsummarized_turns()] == ids[2:4]

    assert mem.prune_summaries(1) == (2, 4)  # s1 + s3 and their rows
    assert [t.id for t in mem.recent_turns()] == ids[2:4] + ids[6:]


def test_merge_facts_dedupes_case_insensitively(mem):
    mem.add_fact("Paul likes coffee")
    added = mem.merge_facts([
        "paul LIKES coffee",      # already known, different case
        "   ",                    # blank
        "Paul lives in Seattle",
        "paul lives in seattle",  # duplicate within the same batch
    ])
    assert added == 1
    assert mem.list_facts() == ["Paul likes coffee", "Paul lives in Seattle"]
