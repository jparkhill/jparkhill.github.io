---
name: durian-memory
description: Use the Durian agent as durable, cross-session memory over HTTP. RECALL project state and prior decisions at the start of a task, CHECKPOINT progress as you work, and HANDOFF at the end so the next agent (Claude, Codex, pi, human) can resume without re-deriving context. Use when the user says "where was I", "what's next", "continue", "remember this", or when starting/ending a long task.
---

# Durian as durable coordinating memory (REST)

Durian is a long-running agent that owns persistent state: **projects**, **todos**,
**memory** (facts/profile/preferences), and a **wiki** (searchable knowledge pages).
You (Claude / Codex / any coding agent) are stateless between sessions. Use Durian
to make your work *survive* the session boundary.

This skill is **transport-level HTTP**. No SDK, no install. `curl` is enough.

## Connection

- **Base URL:** `http://100.64.0.10:8787`  (Durian tailnet address; `127.0.0.1` will NOT work)
- **Auth:** send the token one of three ways
  - header `Authorization: Bearer $DURIAN_TOKEN`  ← preferred
  - header `x-durian-token: $DURIAN_TOKEN`
  - query `?token=$DURIAN_TOKEN`  (avoid; leaks into logs)
- **Health check first:** `GET /health` → `{"ok":true}`. If this fails, Durian is down; stop and tell the user.

```bash
export DURIAN_URL="http://100.64.0.10:8787"
export DURIAN_TOKEN="<ask the user for their token; never hardcode it in a repo>"
curl -sf "$DURIAN_URL/health"
```

> Never print, log, or commit the token. If a command needs it, use the env var.

## The loop: RECALL → RESUME → CHECKPOINT → HANDOFF

### 1. RECALL — what do we already know?
Before starting, search memory. This is the single highest-value call.

```bash
# Free-text semantic search across memory + wiki. Returns ranked hits.
curl -sf -H "Authorization: Bearer $DURIAN_TOKEN" \
  "$DURIAN_URL/memory?q=<url-encoded+topic>&limit=6"
```
Response: `{ "query": "...", "hits": [ { "page", "title", "heading", "breadcrumb", "snippet", "score", "path" } ] }`
The `snippet` usually answers "what do we know" on its own; only fetch the
full `page` (below) when the snippet is insufficient. `path` points at an exact
anchor (`memory/preferences.md#...`).

### 2. RESUME — reconstruct the thread
List projects and open todos, then read the one you're working on.

```bash
curl -sf -H "Authorization: Bearer $DURIAN_TOKEN" "$DURIAN_URL/projects"
curl -sf -H "Authorization: Bearer $DURIAN_TOKEN" "$DURIAN_URL/todos?status=open"
```
Each project carries a `log` (append-only history of notes). That log IS the
continuity: read the last few entries to see what the previous agent did and why.

### 3. CHECKPOINT — persist progress as you work
Writes go through the **message** endpoint (the agent decides which tools to call).

```bash
# Record a decision / nudge a project. Natural language; Durian maps it to tools.
curl -sf -X POST -H "Authorization: Bearer $DURIAN_TOKEN" \
  -H "content-type: application/json" \
  -d '{"text":"project_update coin_vol note=\"fixed vol surface interpolation; next: backtest 2024\""}' \
  "$DURIAN_URL/message"
```
Response (non-stream): the agent's `{ text, sessionId, events, ... }`. Use
`events` to confirm the write actually happened (`tool_result` with `isError:false`).

Checkpoint **at decision points**, not every step: when you pick an approach,
hit a blocker, or finish a sub-task. One good note > ten noisy ones.

### 4. HANDOFF — leave a trail for the next session
At the end of meaningful work, write a short handoff: what changed, what's next, what to watch.

```bash
curl -sf -X POST -H "Authorization: Bearer $DURIAN_TOKEN" \
  -H "content-type: application/json" \
  -d '{"text":"Append to journal: finished X, blocked on Y, next is Z (see wiki/projects/coin_vol.md)"}' \
  "$DURIAN_URL/message"

# Or add a durable fact / preference:
curl -sf -X POST -H "Authorization: Bearer $DURIAN_TOKEN" \
  -H "content-type: application/json" \
  -d '{"text":"Remember: our deploy target is the tailnet box at 100.64.0.10"}' \
  "$DURIAN_URL/message"
```
When reviewing your own output, a rule of thumb: **if losing it would cost the
next session more than a minute to re-derive, persist it.**

## Multi-agent coordination (A2A)

If another agent or a human operator owns a piece of the work, message them and
send a durable copy to Durian.

```bash
# Send to a configured peer (by name) or explicit URL
curl -sf -X POST -H "Authorization: Bearer $DURIAN_TOKEN" \
  -H "content-type: application/json" \
  -d '{"to":"<peer>","subject":"handoff: coin_vol","body":"done with X; you own Y"}' \
  "$DURIAN_URL/message"     # if it goes through the agent
# Or the raw A2A endpoint:
curl -sf -X POST -H "Authorization: Bearer $DURIAN_TOKEN" \
  -H "content-type: application/json" \
  -d '{"from":"codex","to":"durian","subject":"handoff","body":"..."}' \
  "$DURIAN_URL/a2a"
```

## Endpoint reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness; no auth |
| GET | `/status` | runtime name/model/channels |
| GET | `/projects?status=active` | all projects (+ `log`) |
| GET | `/todos?status=open` | todo list |
| GET | `/memory?q=<term>&limit=6` | semantic search over memory + wiki |
| GET | `/sessions` · `/sessions/<id>` | session list / transcript |
| GET | `/inbox?limit=50` | A2A messages received |
| POST | `/message` | **the workhorse** — `{text, sessionId?, identity?, newSession?}` |
| POST | `/a2a` | `{from,to,subject,body}` agent-to-agent |
| POST | `/projects/sync` | regenerate `wiki/projects/*.md` from projects |
| POST | `/reload` | hot-reload skills/identities |

Streaming: set `Accept: text/event-stream` on `POST /message` to receive
`event: tool_call|tool_result|done` SSE frames instead of a single JSON blob.

All responses are JSON. Non-2xx carries `{ "error": "..." }`.

## Reading and writing wiki pages

Search returns `page` paths like `wiki/projects/coin_vol.md`. To read the full
page, either `curl` it from the running filesystem if you share one, or ask the
agent to surface it:

```bash
curl -sf -X POST -H "Authorization: Bearer $DURIAN_TOKEN" \
  -H "content-type: application/json" \
  -d '{"text":"read wiki/projects/coin_vol.md and summarize the last 3 log entries"}' \
  "$DURIAN_URL/message"
```

## Guardrails

- **Never exfiltrate secrets.** Don't paste tokens, `.env`, or credentials into
  `/message` text; they get stored permanently.
- **Writes require care.** Durian runs as a real agent with file/shell tools. The
  `/message` body is a *prompt*, so keep instructions explicit and scoped.
- **Confirm with the user** before destructive prompts (`delete project`, bulk
  edits) — same as any write.
- **Curated pages are protected.** Pages with `type: curated` frontmatter are
  never overwritten by `/projects/sync`; edit them via the wiki, not by
  regenerating. The reading list is not one of them any more: it is a hierarchy
  under `wiki/knowledge/reading/` (one page per entry), added to with
  `reading_list_add` / `POST /reading-list`; `wiki/projects/reading-list.md` is a
  curated pointer to it.

## Minimal session template

```bash
export DURIAN_URL="http://100.64.0.10:8787"
H=(-H "Authorization: Bearer $DURIAN_TOKEN")

# 1. RECALL
curl -sf "${H[@]}" "$DURIAN_URL/memory?q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$TASK")&limit=6"

# 2. RESUME
curl -sf "${H[@]}" "$DURIAN_URL/projects"
curl -sf "${H[@]}" "$DURIAN_URL/todos?status=open"

# 3. ... do the work ...

# 4. CHECKPOINT + HANDOFF
curl -sf "${H[@]}" -H "content-type: application/json" \
  -d '{"text":"project_update <slug> note=\"<what changed / what is next>\""}' \
  "$DURIAN_URL/message"
```
