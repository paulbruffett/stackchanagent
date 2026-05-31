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

// ---- memories ----
async function loadMemories() {
  const root = $("#memories");
  root.innerHTML = "";
  const [facts, summaries, turns] = await Promise.all([
    (await fetch("/api/memories/facts")).json(),
    (await fetch("/api/memories/summaries")).json(),
    (await fetch("/api/memories/turns?limit=40")).json(),
  ]);

  const fsec = el("div", { className: "group" }, el("h2", { textContent: `Known facts (${facts.facts.length})` }));
  if (!facts.facts.length) fsec.append(el("div", { className: "muted", textContent: "none yet" }));
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
  root.append(fsec);

  const ssec = el("div", { className: "group" }, el("h2", { textContent: `Summaries (${summaries.summaries.length})` }));
  if (!summaries.summaries.length) ssec.append(el("div", { className: "muted", textContent: "none yet" }));
  for (const s of summaries.summaries) {
    const c = el("div", { className: "card" });
    c.append(el("div", { className: "muted", textContent: `turns ${s.span_from}–${s.span_to}` }),
             el("div", { textContent: s.summary }));
    ssec.append(c);
  }
  root.append(ssec);

  const tsec = el("div", { className: "group" }, el("h2", { textContent: "Recent turns" }));
  for (const t of turns.turns) {
    tsec.append(el("div", { className: "card" },
      el("div", { className: "muted", textContent: `#${t.id} ${t.role}` }),
      el("div", { innerHTML: renderContent(t.content) })));
  }
  root.append(tsec);
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
