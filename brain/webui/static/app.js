"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, props = {}, ...kids) => {
  const e = Object.assign(document.createElement(tag), props);
  for (const k of kids) e.append(k);
  return e;
};
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const fmtTs = (t) => new Date(t * 1000).toLocaleTimeString();

// ---- tabs ----
$$("nav button").forEach((b) =>
  b.addEventListener("click", () => {
    $$("nav button").forEach((x) => x.classList.toggle("active", x === b));
    $$(".tab").forEach((t) => t.classList.toggle("active", t.id === b.dataset.tab));
    if (b.dataset.tab === "memories") loadMemories();
    if (b.dataset.tab === "config") loadConfig();
    if (b.dataset.tab === "persona") loadPersona();
    if (b.dataset.tab === "mcp") loadMcp();
    if (b.dataset.tab === "a2a") loadA2a();
  })
);

function setStatus(msg, cls = "") {
  const s = $("#status");
  s.textContent = msg;
  s.className = cls;
}

// ---- config ----
async function loadConfig() {
  const { items } = await (await fetch("/api/config")).json();
  const root = $("#config");
  root.innerHTML = "";
  const groups = {};
  for (const it of items) (groups[it.group] ||= []).push(it);
  for (const [g, rows] of Object.entries(groups)) {
    const sec = el("div", { className: "group" }, el("h2", { textContent: g }));
    for (const it of rows) {
      const input = el("input", { value: it.value });
      const label = el("div", { className: "k" });
      label.append(it.key, el("small", { textContent: it.help }));
      const save = el("button", { className: "act", textContent: "Save" });
      save.addEventListener("click", () => saveConfig(it.key, input.value, it.restart));
      const row = el("div", { className: "row" }, label, input);
      if (it.restart) row.append(el("span", { className: "badge restart", textContent: "restart" }));
      if (it.overridden) row.append(el("span", { className: "badge over", textContent: "overridden" }));
      row.append(el("span", { style: "flex:1" }), save);
      sec.append(row);
    }
    root.append(sec);
  }
}

async function saveConfig(key, value, restart) {
  const res = await fetch("/api/config", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ key, value }),
  });
  if (!res.ok) {
    setStatus(`save failed: ${(await res.json()).detail}`, "err");
    return;
  }
  setStatus(restart ? `${key} saved — applies on restart` : `${key} saved`, "ok");
  loadConfig();
}

// ---- persona / system prompt ----
async function loadPersona() {
  const root = $("#persona");
  root.innerHTML = "";
  const data = await (await fetch("/api/system_prompt")).json();

  const sec = el("div", { className: "group" },
    el("h2", { textContent: "Core persona / system prompt" }));
  sec.append(el("div", { className: "muted", style: "margin-bottom:8px",
    textContent: "The robot's core identity, prepended to every turn. Changes apply on the "
      + "next conversation turn — no restart. Leave it matching the default to track future "
      + "default changes; edit to override." }));

  const ta = el("textarea", { className: "prompt", spellcheck: false });
  ta.value = data.prompt;
  const counter = el("span", { className: "muted" });
  const count = (suffix) => counter.textContent = `${ta.value.length} chars${suffix || ""}`;
  count(data.overridden ? " · overridden" : " · default");
  ta.addEventListener("input", () => count());

  const save = el("button", { className: "act", textContent: "Save" });
  save.addEventListener("click", () => savePersona(ta.value));
  const reset = el("button", { className: "ghost", textContent: "Reset to default" });
  reset.addEventListener("click", () => {
    if (confirm("Reset the persona to the built-in default?")) savePersona("");
  });

  const bar = el("div", { className: "toolbar" },
    counter, el("span", { style: "flex:1" }), reset, save);
  sec.append(ta, bar);
  root.append(sec);
}

async function savePersona(prompt) {
  const res = await fetch("/api/system_prompt", {
    method: "PUT", headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) {
    setStatus(`save failed: ${(await res.json()).detail}`, "err");
    return;
  }
  setStatus("persona saved — applies next turn", "ok");
  loadPersona();
}

// ---- memories ----
async function loadMemories() {
  const root = $("#memories");
  root.innerHTML = "";
  const [facts, summaries, turns] = await Promise.all([
    (await fetch("/api/memories/facts")).json(),
    (await fetch("/api/memories/summaries")).json(),
    (await fetch("/api/memories/turns?limit=40")).json(),
  ]);

  // --- facts ---
  const fsec = el("div", { className: "group" }, el("h2", { textContent: `Permanent knowledge (${facts.facts.length})` }));
  for (const f of facts.facts) {
    const input = el("input", { value: f.fact });
    const save = el("button", { className: "ghost", textContent: "save" });
    save.addEventListener("click", async () => {
      await fetch(`/api/memories/facts/${f.id}`, {
        method: "PUT", headers: { "content-type": "application/json" },
        body: JSON.stringify({ fact: input.value }),
      });
      setStatus("fact updated", "ok");
    });
    const del = el("button", { className: "ghost", textContent: "✕" });
    del.addEventListener("click", async () => {
      await fetch(`/api/memories/facts/${f.id}`, { method: "DELETE" });
      loadMemories();
    });
    fsec.append(el("div", { className: "card fact" }, input, save, del));
  }
  // add-fact row
  const addInput = el("input", { placeholder: "add a fact (third person, e.g. 'The user likes tea')" });
  const addBtn = el("button", { className: "act", textContent: "Add" });
  addBtn.addEventListener("click", async () => {
    const fact = addInput.value.trim();
    if (!fact) return;
    const r = await fetch("/api/memories/facts", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ fact }),
    });
    if (r.ok) { setStatus("fact added", "ok"); loadMemories(); }
    else setStatus("add failed", "err");
  });
  fsec.append(el("div", { className: "card fact" }, addInput, addBtn));
  // compact-facts (LLM)
  const compact = el("button", { className: "ghost", textContent: "Compact facts (LLM)" });
  compact.addEventListener("click", compactFacts);
  const compactBar = el("div", { className: "toolbar" },
    el("div", { className: "muted", textContent:
      "Ask the model to merge duplicates and drop stale facts — you approve before anything changes." }),
    el("span", { style: "flex:1" }), compact);
  fsec.append(compactBar);
  fsec.append(el("div", { id: "compact-out" }));
  root.append(fsec);

  // --- summaries ---
  const ssec = el("div", { className: "group" }, el("h2", { textContent: `Recent conversation summaries (${summaries.summaries.length}) — older ones auto-expire` }));
  const sumNow = el("button", { className: "act", textContent: "Summarize now" });
  sumNow.addEventListener("click", summarizeNow);
  ssec.append(el("div", { className: "toolbar" },
    el("div", { className: "muted", textContent:
      "Episodic memory. Folds the oldest backlog into a summary now (keeps the most recent turns verbatim). Beyond the SUMMARY_RETENTION window, old summaries and their turns are purged automatically — enduring facts are lifted into Permanent knowledge first." }),
    el("span", { style: "flex:1" }), sumNow));
  if (!summaries.summaries.length) ssec.append(el("div", { className: "muted", textContent: "none yet" }));
  for (const s of summaries.summaries) {
    const c = el("div", { className: "card" });
    const ta = el("textarea", { className: "sm" });
    ta.value = s.summary;
    const save = el("button", { className: "ghost", textContent: "save" });
    save.addEventListener("click", async () => {
      const r = await fetch(`/api/memories/summaries/${s.id}`, {
        method: "PUT", headers: { "content-type": "application/json" },
        body: JSON.stringify({ summary: ta.value }),
      });
      setStatus(r.ok ? "summary updated" : "update failed", r.ok ? "ok" : "err");
    });
    const del = el("button", { className: "ghost", textContent: "delete" });
    del.addEventListener("click", async () => {
      if (!confirm(`Delete this summary? Turns ${s.span_from}–${s.span_to} will replay verbatim again.`)) return;
      const r = await fetch(`/api/memories/summaries/${s.id}?unmark=true`, { method: "DELETE" });
      setStatus(r.ok ? "summary deleted — turns restored" : "delete failed", r.ok ? "ok" : "err");
      loadMemories();
    });
    c.append(
      el("div", { className: "muted", textContent: `#${s.id} · turns ${s.span_from}–${s.span_to}` }),
      ta,
      el("div", { className: "toolbar" }, el("span", { style: "flex:1" }), del, save),
    );
    ssec.append(c);
  }
  root.append(ssec);

  // --- conversation health (M6.5) ---
  const hsec = el("div", { className: "group" }, el("h2", { textContent: "Conversation health" }));
  const repair = el("button", { className: "ghost", textContent: "Repair conversation" });
  repair.addEventListener("click", repairConversation);
  const reset = el("button", { className: "ghost", textContent: "Reset conversation" });
  reset.addEventListener("click", resetConversation);
  hsec.append(el("div", { className: "toolbar" },
    el("div", { className: "muted", textContent:
      "Repair heals dangling tool-call corruption in the live turn tail (logs what it fixed). Reset deletes the unsummarized tail when a thread is wedged — summaries and permanent facts are kept." }),
    el("span", { style: "flex:1" }), reset, repair));
  root.append(hsec);

  // --- recent turns (read-only) ---
  const tsec = el("div", { className: "group" }, el("h2", { textContent: "Recent turns" }));
  for (const t of turns.turns) {
    tsec.append(el("div", { className: "card" },
      el("div", { className: "muted", textContent: `#${t.id} ${t.role}` }),
      el("div", { innerHTML: renderContent(t.content) })));
  }
  root.append(tsec);
}

async function repairConversation() {
  setStatus("repairing…");
  const r = await fetch("/api/memories/repair", { method: "POST" });
  const j = await r.json();
  const c = j.counts || {};
  const fixed = (c.turns_rewritten || 0) + (c.turns_deleted || 0);
  setStatus(fixed ? `repaired ${fixed} turn(s): ${JSON.stringify(c)}` : "conversation clean — nothing to repair", "ok");
  loadMemories();
}

async function resetConversation() {
  if (!confirm("Delete the entire unsummarized conversation tail? Summaries and permanent facts are kept. This cannot be undone.")) return;
  setStatus("resetting…");
  const r = await fetch("/api/memories/reset", { method: "POST" });
  const j = await r.json();
  setStatus(r.ok ? `conversation reset — deleted ${j.deleted} turn(s)` : "reset failed", r.ok ? "ok" : "err");
  loadMemories();
}

async function summarizeNow() {
  setStatus("summarizing…");
  const r = await fetch("/api/memories/summarize", { method: "POST" });
  const j = await r.json();
  if (j.ok) { setStatus(`summary created (turns ${j.summary.span_from}–${j.summary.span_to})`, "ok"); loadMemories(); }
  else setStatus(`nothing summarized: ${j.reason}`, "err");
}

async function compactFacts() {
  const out = $("#compact-out");
  out.innerHTML = "";
  setStatus("asking the model to consolidate…");
  const r = await fetch("/api/memories/facts/compact", { method: "POST" });
  if (!r.ok) { setStatus("compaction failed", "err"); return; }
  const { original, proposed } = await r.json();
  setStatus(`proposed ${original.length} → ${proposed.length} facts — review and apply`, "ok");

  // Editable proposed list so the operator can tweak before applying.
  const ta = el("textarea", { className: "sm" });
  ta.value = proposed.join("\n");
  ta.style.minHeight = "160px";
  const apply = el("button", { className: "act", textContent: `Apply (${proposed.length})` });
  apply.addEventListener("click", async () => {
    const facts = ta.value.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!confirm(`Replace all known facts with these ${facts.length}? This cannot be undone.`)) return;
    const a = await fetch("/api/memories/facts/apply", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ facts }),
    });
    setStatus(a.ok ? "facts replaced" : "apply failed", a.ok ? "ok" : "err");
    if (a.ok) loadMemories();
  });
  const cancel = el("button", { className: "ghost", textContent: "Discard" });
  cancel.addEventListener("click", () => { out.innerHTML = ""; setStatus(""); });
  const card = el("div", { className: "card" });
  card.append(
    el("div", { className: "muted", textContent: `Proposed consolidated facts (was ${original.length}, now ${proposed.length}) — one per line, editable:` }),
    ta,
    el("div", { className: "toolbar" }, el("span", { style: "flex:1" }), cancel, apply),
  );
  out.append(card);
}

function renderContent(content) {
  if (typeof content === "string") return esc(content);
  if (!Array.isArray(content)) return esc(JSON.stringify(content));
  return content.map((b) => {
    if (b.type === "text") return esc(b.text);
    if (b.type === "tool_use") return `<span class="pill">tool ${esc(b.name)}</span>${esc(JSON.stringify(b.input))}`;
    if (b.type === "tool_result") return `<span class="pill">result</span>${esc(JSON.stringify(b.content))}`;
    return `<span class="pill">${esc(b.type)}</span>`;
  }).join("<br>");
}

// ---- MCP ----
async function loadMcp() {
  const root = $("#mcp");
  root.innerHTML = "";
  const data = await (await fetch("/api/mcp/servers")).json();

  const bar = el("div", { className: "toolbar" });
  const reload = el("button", { className: "act", textContent: "Reload MCP" });
  reload.addEventListener("click", async () => {
    setStatus("reloading MCP…");
    const r = await fetch("/api/mcp/reload", { method: "POST" });
    setStatus(r.ok ? "MCP reloaded" : "reload failed", r.ok ? "ok" : "err");
    loadMcp();
  });
  bar.append(el("div", { className: "muted",
    textContent: "Edits persist immediately; click Reload to (re)connect servers and apply." }),
    el("span", { style: "flex:1" }), reload);
  root.append(bar);

  for (const s of data.servers) {
    const card = el("div", { className: "card" });
    const dot = s.connected ? `<span class="ok">●</span>` : (s.enabled ? `<span class="err">●</span>` : `<span class="muted">○</span>`);
    const head = el("div", { className: "fact" });
    head.innerHTML = `${dot} <b>${esc(s.name)}</b> <span class="muted">${esc(s.transport)}</span>`;
    const toggle = el("button", { className: "ghost",
      textContent: s.enabled ? "disable" : "enable" });
    toggle.addEventListener("click", async () => {
      await fetch(`/api/mcp/servers/${s.id}`, { method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ enabled: !s.enabled }) });
      loadMcp();
    });
    const del = el("button", { className: "ghost", textContent: "✕" });
    del.addEventListener("click", async () => {
      if (!confirm(`Delete MCP server "${s.name}"?`)) return;
      await fetch(`/api/mcp/servers/${s.id}`, { method: "DELETE" });
      loadMcp();
    });
    head.append(el("span", { style: "flex:1" }), toggle, del);
    card.append(head);

    const launch = s.transport === "http" ? esc(s.url || "")
      : esc([s.command, ...(s.args || [])].join(" "));
    card.append(el("div", { className: "muted", style: "font-family:ui-monospace,monospace;font-size:12px;margin-top:6px",
      textContent: launch }));
    if (s.env_ref) card.append(el("div", { className: "muted", innerHTML: `secret env: <code>${esc(s.env_ref)}</code> (value from .env)` }));
    if (s.error) card.append(el("div", { className: "err", textContent: s.error }));
    if (s.tools && s.tools.length)
      card.append(el("div", { innerHTML: s.tools.map((t) => `<span class="pill">${esc(t)}</span>`).join("") }));
    root.append(card);
  }

  // add-server form
  const form = el("div", { className: "group" }, el("h2", { textContent: "Add server" }));
  const f = {};
  const field = (key, ph) => { const i = el("input", { placeholder: ph, style: "width:100%;margin-bottom:6px" }); f[key] = i; return i; };
  form.append(
    field("name", "name (e.g. weather)"),
    field("command", "command (stdio, e.g. python)"),
    field("args", "args, space-separated (e.g. mcp_servers/weather.py)"),
    field("env_ref", "secret env var name (optional, e.g. HUE_TOKEN)"),
  );
  const add = el("button", { className: "act", textContent: "Add (stdio)" });
  add.addEventListener("click", async () => {
    const body = {
      name: f.name.value.trim(),
      transport: "stdio",
      command: f.command.value.trim() || null,
      args: f.args.value.trim() ? f.args.value.trim().split(/\s+/) : [],
      env_ref: f.env_ref.value.trim() || null,
      enabled: true,
    };
    if (!body.name) { setStatus("name required", "err"); return; }
    const r = await fetch("/api/mcp/servers", { method: "POST",
      headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    setStatus(r.ok ? "server added — Reload to connect" : "add failed", r.ok ? "ok" : "err");
    loadMcp();
  });
  form.append(add);
  root.append(form);
}

// ---- A2A ----
async function loadA2a() {
  const root = $("#a2a");
  root.innerHTML = "";
  const data = await (await fetch("/api/a2a/servers")).json();

  const bar = el("div", { className: "toolbar" });
  const reload = el("button", { className: "act", textContent: "Reload A2A" });
  reload.addEventListener("click", async () => {
    setStatus("reloading A2A…");
    const r = await fetch("/api/a2a/reload", { method: "POST" });
    setStatus(r.ok ? "A2A reloaded" : "reload failed", r.ok ? "ok" : "err");
    loadA2a();
  });
  bar.append(el("div", { className: "muted",
    textContent: "Agent2Agent endpoints. Edits persist immediately; click Reload to (re)fetch agent cards and apply." }),
    el("span", { style: "flex:1" }), reload);
  root.append(bar);

  for (const s of data.servers) {
    const card = el("div", { className: "card" });
    const dot = s.connected ? `<span class="ok">●</span>` : (s.enabled ? `<span class="err">●</span>` : `<span class="muted">○</span>`);
    const head = el("div", { className: "fact" });
    head.innerHTML = `${dot} <b>${esc(s.name)}</b>` + (s.agent ? ` <span class="muted">${esc(s.agent)}</span>` : "");
    const toggle = el("button", { className: "ghost",
      textContent: s.enabled ? "disable" : "enable" });
    toggle.addEventListener("click", async () => {
      await fetch(`/api/a2a/servers/${s.id}`, { method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ enabled: !s.enabled }) });
      loadA2a();
    });
    const del = el("button", { className: "ghost", textContent: "✕" });
    del.addEventListener("click", async () => {
      if (!confirm(`Delete A2A server "${s.name}"?`)) return;
      await fetch(`/api/a2a/servers/${s.id}`, { method: "DELETE" });
      loadA2a();
    });
    head.append(el("span", { style: "flex:1" }), toggle, del);
    card.append(head);

    card.append(el("div", { className: "muted", style: "font-family:ui-monospace,monospace;font-size:12px;margin-top:6px",
      textContent: s.url }));
    if (s.env_ref) card.append(el("div", { className: "muted", innerHTML: `bearer token env: <code>${esc(s.env_ref)}</code> (value from .env)` }));
    if (s.error) card.append(el("div", { className: "err", textContent: s.error }));
    if (s.delegates && s.delegates.length)
      card.append(el("div", { innerHTML: s.delegates.map((d) => `<span class="pill">${esc(d)}</span>`).join("") }));
    root.append(card);
  }

  // add-server form
  const form = el("div", { className: "group" }, el("h2", { textContent: "Add endpoint" }));
  const f = {};
  const field = (key, ph) => { const i = el("input", { placeholder: ph, style: "width:100%;margin-bottom:6px" }); f[key] = i; return i; };
  form.append(
    field("name", "name (e.g. hermes)"),
    field("url", "url (agent base or /.well-known/agent.json, e.g. http://192.168.4.30:8080)"),
    field("env_ref", "bearer token env var name (optional)"),
  );
  const add = el("button", { className: "act", textContent: "Add" });
  add.addEventListener("click", async () => {
    const body = {
      name: f.name.value.trim(),
      url: f.url.value.trim(),
      env_ref: f.env_ref.value.trim() || null,
      enabled: true,
    };
    if (!body.name) { setStatus("name required", "err"); return; }
    if (!body.url) { setStatus("url required", "err"); return; }
    const r = await fetch("/api/a2a/servers", { method: "POST",
      headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    setStatus(r.ok ? "endpoint added — Reload to connect" : "add failed", r.ok ? "ok" : "err");
    loadA2a();
  });
  form.append(add);
  root.append(form);
}

// ---- live log feed ----
let logFilter = "";
$("#log-filter").addEventListener("input", (e) => { logFilter = e.target.value.toLowerCase(); });
$("#log-clear").addEventListener("click", () => { $("#log-out").innerHTML = ""; });

function appendLog(item) {
  const out = $("#log-out");
  const hay = (item.name + " " + item.msg).toLowerCase();
  if (logFilter && !hay.includes(logFilter)) return;
  const near = out.scrollHeight - out.scrollTop - out.clientHeight < 40;
  const line = el("div", { className: `logline lvl-${item.level}` });
  line.innerHTML = `<span class="muted">${fmtTs(item.ts)}</span> ` +
    `<span class="logname">${esc(item.name)}</span> ${esc(item.msg)}`;
  out.append(line);
  while (out.childElementCount > 1500) out.firstChild.remove();
  if (near) out.scrollTop = out.scrollHeight;
}

// ---- live transaction feed ----
function appendTurn(t) {
  const list = $("#tx-list");
  const card = el("div", { className: "card tx" });
  card.append(el("div", { className: "muted", textContent: fmtTs(t.ts) + (t.follow_up ? " · follow-up" : "") }));
  card.append(el("div", {}, el("span", { className: "t", textContent: "“" + (t.transcript || "") + "”" })));
  if (t.tools && t.tools.length)
    card.append(el("div", { innerHTML: t.tools.map((x) => `<span class="pill">${esc(x.name)} ${esc(JSON.stringify(x.input))}</span>`).join("") }));
  if (t.reply) card.append(el("div", { className: "r", textContent: t.reply }));
  else card.append(el("div", { className: "muted", textContent: "(silent)" }));
  const meta = [];
  if (t.stt_ms != null) meta.push(`stt ${t.stt_ms}ms`);
  if (t.total_ms != null) meta.push(`turn ${t.total_ms}ms`);
  card.append(el("div", { className: "meta", textContent: meta.join(" · ") }));
  list.prepend(card);
  while (list.childElementCount > 100) list.lastChild.remove();
}

// ---- websockets ----
function connectFeed(path, onItem) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}${path}`);
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "scrollback") m.items.forEach(onItem);
    else if (m.type === "item") onItem(m.item);
  };
  ws.onclose = () => setTimeout(() => connectFeed(path, onItem), 2000);
}

loadConfig();
connectFeed("/ws/logs", appendLog);
connectFeed("/ws/turns", appendTurn);
